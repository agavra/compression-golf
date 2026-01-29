//! # XiangpengHao Codec
//!
//! Row-group column store with global dictionaries.
//! - Fixed row groups (128k rows)
//! - Sort rows by timestamp within each group
//! - Delta-encode `ts` and `id` (zigzag for id), then bit-pack
//! - Compress each column with zstd

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::{HashMap, HashSet};
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const MAGIC: &[u8; 4] = b"XPH1";
const ZSTD_LEVEL: i32 = 22;
const ROW_GROUP_ROWS: usize = 128_000;
const COLUMN_NAMES: [&str; 5] = [
    "type_indices",
    "repo_id_idx_offsets",
    "repo_name_variant_idx",
    "id_deltas",
    "ts_deltas",
];

fn debug_enabled() -> bool {
    std::env::var("XPH_DEBUG").is_ok()
}

// ============================================================================
// Basic utilities
// ============================================================================

fn parse_timestamp(ts: &str) -> u64 {
    DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.timestamp() as u64)
        .unwrap_or(0)
}

fn format_timestamp(ts: u64) -> String {
    Utc.timestamp_opt(ts as i64, 0)
        .single()
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_default()
}

fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> u32 {
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[*pos..*pos + 4]);
    *pos += 4;
    u32::from_le_bytes(arr)
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> u64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*pos..*pos + 8]);
    *pos += 8;
    u64::from_le_bytes(arr)
}

fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(encoded: u64) -> i64 {
    ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64))
}

fn row_groups(total_rows: usize, group_size: usize) -> Vec<(usize, usize)> {
    if total_rows == 0 {
        return Vec::new();
    }
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < total_rows {
        let end = (start + group_size).min(total_rows);
        groups.push((start, end));
        start = end;
    }
    groups
}

fn reorder_by_perm<T: Copy>(values: &mut [T], perm: &[u32]) {
    let mut tmp = Vec::with_capacity(values.len());
    for &idx in perm {
        tmp.push(values[idx as usize]);
    }
    values.copy_from_slice(&tmp);
}

// ============================================================================
// Bit-packing
// ============================================================================

fn pack_bits_u64(values: &[u64]) -> Vec<u8> {
    if values.is_empty() {
        return vec![0];
    }

    let max_val = values.iter().copied().max().unwrap_or(0);
    let bit_width = if max_val == 0 {
        1
    } else {
        64 - max_val.leading_zeros() as u8
    };

    let mut buf = Vec::with_capacity((values.len() * bit_width as usize + 7) / 8 + 1);
    buf.push(bit_width);

    let mut bit_pos: usize = 0;
    let mut current_byte: u8 = 0;

    for &value in values {
        for bit_idx in 0..bit_width {
            let bit = ((value >> bit_idx) & 1) as u8;
            current_byte |= bit << (bit_pos % 8);
            bit_pos += 1;
            if bit_pos % 8 == 0 {
                buf.push(current_byte);
                current_byte = 0;
            }
        }
    }

    if bit_pos % 8 != 0 {
        buf.push(current_byte);
    }

    buf
}

fn unpack_bits_u64(bytes: &[u8], count: usize) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }
    let bit_width = bytes[0] as usize;
    if bit_width == 0 {
        return vec![0; count];
    }

    let mut values = Vec::with_capacity(count);
    let mut bit_pos: usize = 0;

    for _ in 0..count {
        let mut value: u64 = 0;
        for bit_idx in 0..bit_width {
            let byte_idx = 1 + (bit_pos / 8);
            let bit_in_byte = bit_pos % 8;
            let bit = ((bytes[byte_idx] >> bit_in_byte) & 1) as u64;
            value |= bit << bit_idx;
            bit_pos += 1;
        }
        values.push(value);
    }

    values
}

fn pack_bits_u32(values: &[u32]) -> Vec<u8> {
    let mut tmp = Vec::with_capacity(values.len());
    for &v in values {
        tmp.push(v as u64);
    }
    pack_bits_u64(&tmp)
}

fn pack_bits_u8(values: &[u8]) -> Vec<u8> {
    let mut tmp = Vec::with_capacity(values.len());
    for &v in values {
        tmp.push(v as u64);
    }
    pack_bits_u64(&tmp)
}

fn unpack_bits_u32(bytes: &[u8], count: usize) -> Vec<u32> {
    unpack_bits_u64(bytes, count)
        .into_iter()
        .map(|v| v as u32)
        .collect()
}

fn unpack_bits_u8(bytes: &[u8], count: usize) -> Vec<u8> {
    unpack_bits_u64(bytes, count)
        .into_iter()
        .map(|v| v as u8)
        .collect()
}

// ============================================================================
// Dictionaries
// ============================================================================

struct StringDict {
    strings: Vec<String>,
    str_to_idx: HashMap<String, u32>,
}

impl StringDict {
    fn build(items: impl Iterator<Item = String>) -> Self {
        let mut unique: Vec<String> = items.collect();
        unique.sort();
        unique.dedup();

        let mut str_to_idx = HashMap::new();
        for (i, s) in unique.iter().enumerate() {
            str_to_idx.insert(s.clone(), i as u32);
        }

        Self {
            strings: unique,
            str_to_idx,
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_u32(&mut buf, self.strings.len() as u32);
        for s in &self.strings {
            write_u32(&mut buf, s.len() as u32);
        }
        for s in &self.strings {
            buf.extend_from_slice(s.as_bytes());
        }
        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        let mut pos = 0;
        let count = read_u32(bytes, &mut pos) as usize;
        let mut strings = Vec::with_capacity(count);
        let mut str_to_idx = HashMap::new();

        let mut lens = Vec::with_capacity(count);
        for _ in 0..count {
            let len = read_u32(bytes, &mut pos) as usize;
            lens.push(len);
        }

        for (i, len) in lens.into_iter().enumerate() {
            let s = std::str::from_utf8(&bytes[pos..pos + len])
                .unwrap()
                .to_string();
            pos += len;
            str_to_idx.insert(s.clone(), i as u32);
            strings.push(s);
        }

        Self {
            strings,
            str_to_idx,
        }
    }

    fn get_index(&self, s: &str) -> u32 {
        self.str_to_idx[s]
    }

    fn get_string(&self, index: u32) -> &str {
        &self.strings[index as usize]
    }
}

struct TypeDict {
    types: Vec<String>,
    type_to_idx: HashMap<String, u8>,
}

impl TypeDict {
    fn build(events: &[(EventKey, EventValue)]) -> Self {
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for (key, _) in events {
            *freq.entry(&key.event_type).or_insert(0) += 1;
        }

        let mut types_with_freq: Vec<_> = freq.into_iter().collect();
        types_with_freq.sort_by(|a, b| b.1.cmp(&a.1));

        let mut types = Vec::with_capacity(types_with_freq.len());
        let mut type_to_idx = HashMap::new();
        for (i, (t, _)) in types_with_freq.into_iter().enumerate() {
            types.push(t.to_string());
            type_to_idx.insert(t.to_string(), i as u8);
        }

        Self { types, type_to_idx }
    }

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_u32(&mut buf, self.types.len() as u32);
        for t in &self.types {
            write_u32(&mut buf, t.len() as u32);
            buf.extend_from_slice(t.as_bytes());
        }
        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        let mut pos = 0;
        let count = read_u32(bytes, &mut pos) as usize;
        let mut types = Vec::with_capacity(count);
        let mut type_to_idx = HashMap::new();

        for i in 0..count {
            let len = read_u32(bytes, &mut pos) as usize;
            let t = std::str::from_utf8(&bytes[pos..pos + len])
                .unwrap()
                .to_string();
            pos += len;
            type_to_idx.insert(t.clone(), i as u8);
            types.push(t);
        }

        Self { types, type_to_idx }
    }

    fn get_index(&self, event_type: &str) -> u8 {
        self.type_to_idx[event_type]
    }

    fn get_type(&self, index: u8) -> &str {
        &self.types[index as usize]
    }
}

struct RepoIdDict {
    repo_ids: Vec<u64>,
    name_variants: Vec<Vec<u32>>,
    id_to_idx: HashMap<u64, u32>,
    name_variant_lookup: HashMap<(u64, u32), u32>,
}

impl RepoIdDict {
    fn build(events: &[(EventKey, EventValue)], repo_dict: &StringDict) -> Self {
        let mut map: HashMap<u64, HashSet<u32>> = HashMap::new();
        for (_, value) in events {
            let name_idx = repo_dict.get_index(&value.repo.name);
            map.entry(value.repo.id)
                .or_insert_with(HashSet::new)
                .insert(name_idx);
        }

        let mut repo_ids: Vec<u64> = map.keys().copied().collect();
        repo_ids.sort();

        let mut name_variants = Vec::with_capacity(repo_ids.len());
        let mut id_to_idx = HashMap::with_capacity(repo_ids.len());
        let mut name_variant_lookup = HashMap::new();

        for (i, repo_id) in repo_ids.iter().enumerate() {
            id_to_idx.insert(*repo_id, i as u32);
            let mut names: Vec<u32> = map[repo_id].iter().copied().collect();
            names.sort();
            for (j, name_idx) in names.iter().enumerate() {
                name_variant_lookup.insert((*repo_id, *name_idx), j as u32);
            }
            name_variants.push(names);
        }

        Self {
            repo_ids,
            name_variants,
            id_to_idx,
            name_variant_lookup,
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_u32(&mut buf, self.repo_ids.len() as u32);
        for repo_id in &self.repo_ids {
            write_u64(&mut buf, *repo_id);
        }

        for names in &self.name_variants {
            write_u32(&mut buf, names.len() as u32);
            for name_idx in names {
                write_u32(&mut buf, *name_idx);
            }
        }

        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        let mut pos = 0;
        let count = read_u32(bytes, &mut pos) as usize;

        let mut repo_ids = Vec::with_capacity(count);
        for _ in 0..count {
            repo_ids.push(read_u64(bytes, &mut pos));
        }

        let mut name_variants = Vec::with_capacity(count);
        for _ in 0..count {
            let variant_count = read_u32(bytes, &mut pos) as usize;
            let mut names = Vec::with_capacity(variant_count);
            for _ in 0..variant_count {
                names.push(read_u32(bytes, &mut pos));
            }
            name_variants.push(names);
        }

        Self {
            repo_ids,
            name_variants,
            id_to_idx: HashMap::new(),
            name_variant_lookup: HashMap::new(),
        }
    }

    fn get_repo_id_idx(&self, repo_id: u64) -> u32 {
        self.id_to_idx[&repo_id]
    }

    fn get_name_variant_idx(&self, repo_id: u64, name_idx: u32) -> u32 {
        self.name_variant_lookup[&(repo_id, name_idx)]
    }

    fn repo_id(&self, index: u32) -> u64 {
        self.repo_ids[index as usize]
    }

    fn name_idx(&self, repo_id_idx: u32, variant_idx: u32) -> u32 {
        self.name_variants[repo_id_idx as usize][variant_idx as usize]
    }
}

// ============================================================================
// Codec implementation
// ============================================================================

pub struct XiangpengHaoCodec;

impl XiangpengHaoCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for XiangpengHaoCodec {
    fn name(&self) -> &str {
        "XiangpengHao"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let type_dict = TypeDict::build(events);
        let repo_dict = StringDict::build(events.iter().map(|(_, v)| v.repo.name.clone()));
        let repo_id_dict = RepoIdDict::build(events, &repo_dict);
        let row_groups = row_groups(events.len(), ROW_GROUP_ROWS);

        let type_dict_bytes = zstd::encode_all(type_dict.encode().as_slice(), ZSTD_LEVEL)?;
        let repo_dict_bytes = zstd::encode_all(repo_dict.encode().as_slice(), ZSTD_LEVEL)?;
        let repo_id_dict_bytes = zstd::encode_all(repo_id_dict.encode().as_slice(), ZSTD_LEVEL)?;

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);

        write_u32(&mut out, type_dict_bytes.len() as u32);
        out.extend_from_slice(&type_dict_bytes);
        write_u32(&mut out, repo_dict_bytes.len() as u32);
        out.extend_from_slice(&repo_dict_bytes);
        write_u32(&mut out, repo_id_dict_bytes.len() as u32);
        out.extend_from_slice(&repo_id_dict_bytes);

        write_u32(&mut out, row_groups.len() as u32);

        let debug = debug_enabled();
        let mut total_column_sizes = [0usize; 5];
        let mut total_row_group_bytes = 0usize;

        if debug {
            eprintln!(
                "XPH_DEBUG dict sizes: type={} repo={} repo_id={}",
                type_dict_bytes.len(),
                repo_dict_bytes.len(),
                repo_id_dict_bytes.len()
            );
        }

        for (start, end) in row_groups {
            let slice = &events[start..end];
            let row_count = slice.len();

            let mut min_id = u64::MAX;
            let mut min_ts = u64::MAX;
            let mut min_repo_id_idx = u32::MAX;

            for (key, value) in slice {
                let event_id = key.id.parse::<u64>().unwrap_or(0);
                let timestamp = parse_timestamp(&value.created_at);
                let repo_id_idx = repo_id_dict.get_repo_id_idx(value.repo.id);

                min_id = min_id.min(event_id);
                min_ts = min_ts.min(timestamp);
                min_repo_id_idx = min_repo_id_idx.min(repo_id_idx);
            }

            if min_id == u64::MAX {
                min_id = 0;
            }
            if min_ts == u64::MAX {
                min_ts = 0;
            }
            if min_repo_id_idx == u32::MAX {
                min_repo_id_idx = 0;
            }

            let mut type_indices = Vec::with_capacity(row_count);
            let mut repo_id_idx_offsets = Vec::with_capacity(row_count);
            let mut repo_name_variant_idx = Vec::with_capacity(row_count);
            let mut id_offsets = Vec::with_capacity(row_count);
            let mut ts_offsets = Vec::with_capacity(row_count);

            for (key, value) in slice {
                let type_idx = type_dict.get_index(&key.event_type);
                let repo_name_idx = repo_dict.get_index(&value.repo.name);
                let repo_id_idx = repo_id_dict.get_repo_id_idx(value.repo.id);
                let name_variant_idx =
                    repo_id_dict.get_name_variant_idx(value.repo.id, repo_name_idx);
                let event_id = key.id.parse::<u64>().unwrap_or(0);
                let timestamp = parse_timestamp(&value.created_at);

                type_indices.push(type_idx);
                repo_id_idx_offsets.push(repo_id_idx - min_repo_id_idx);
                repo_name_variant_idx.push(name_variant_idx);
                id_offsets.push(event_id - min_id);
                ts_offsets.push(timestamp - min_ts);
            }

            let mut perm: Vec<u32> = (0..row_count as u32).collect();
            perm.sort_by_key(|&i| ts_offsets[i as usize]);

            reorder_by_perm(&mut type_indices, &perm);
            reorder_by_perm(&mut repo_id_idx_offsets, &perm);
            reorder_by_perm(&mut repo_name_variant_idx, &perm);
            reorder_by_perm(&mut id_offsets, &perm);
            reorder_by_perm(&mut ts_offsets, &perm);

            let mut id_deltas = Vec::with_capacity(row_count);
            if !id_offsets.is_empty() {
                let mut prev = id_offsets[0] as i64;
                id_deltas.push(zigzag_encode(prev));
                for &id in id_offsets.iter().skip(1) {
                    let cur = id as i64;
                    id_deltas.push(zigzag_encode(cur - prev));
                    prev = cur;
                }
            }

            let mut ts_deltas = Vec::with_capacity(row_count);
            if !ts_offsets.is_empty() {
                let mut prev = ts_offsets[0];
                ts_deltas.push(prev);
                for &ts in ts_offsets.iter().skip(1) {
                    ts_deltas.push(ts - prev);
                    prev = ts;
                }
            }

            let type_indices_bytes =
                zstd::encode_all(pack_bits_u8(&type_indices).as_slice(), ZSTD_LEVEL)?;
            let repo_id_idx_offsets_bytes =
                zstd::encode_all(pack_bits_u32(&repo_id_idx_offsets).as_slice(), ZSTD_LEVEL)?;
            let repo_name_variant_idx_bytes =
                zstd::encode_all(pack_bits_u32(&repo_name_variant_idx).as_slice(), ZSTD_LEVEL)?;
            let id_deltas_bytes =
                zstd::encode_all(pack_bits_u64(&id_deltas).as_slice(), ZSTD_LEVEL)?;
            let ts_deltas_bytes =
                zstd::encode_all(pack_bits_u64(&ts_deltas).as_slice(), ZSTD_LEVEL)?;

            let section_sizes = [
                type_indices_bytes.len(),
                repo_id_idx_offsets_bytes.len(),
                repo_name_variant_idx_bytes.len(),
                id_deltas_bytes.len(),
                ts_deltas_bytes.len(),
            ];

            if debug {
                let mut row_total = 0usize;
                for (name, size) in COLUMN_NAMES.iter().zip(section_sizes.iter()) {
                    eprintln!("XPH_DEBUG row_group col {}: {}", name, size);
                    row_total += *size;
                }
                eprintln!(
                    "XPH_DEBUG row_group summary: rows={} min_id={} min_ts={} min_repo_id_idx={} bytes={}",
                    row_count,
                    min_id,
                    min_ts,
                    min_repo_id_idx,
                    row_total
                );
            }

            for i in 0..total_column_sizes.len() {
                total_column_sizes[i] += section_sizes[i];
            }
            total_row_group_bytes += section_sizes.iter().sum::<usize>();

            write_u64(&mut out, min_id);
            write_u64(&mut out, min_ts);
            write_u32(&mut out, min_repo_id_idx);
            write_u32(&mut out, row_count as u32);

            for section in [
                type_indices_bytes,
                repo_id_idx_offsets_bytes,
                repo_name_variant_idx_bytes,
                id_deltas_bytes,
                ts_deltas_bytes,
            ] {
                write_u32(&mut out, section.len() as u32);
                out.extend_from_slice(&section);
            }
        }

        if debug {
            for (name, size) in COLUMN_NAMES.iter().zip(total_column_sizes.iter()) {
                eprintln!("XPH_DEBUG total col {}: {}", name, size);
            }
            eprintln!("XPH_DEBUG total row_group bytes: {}", total_row_group_bytes);
        }

        Ok(Bytes::from(out))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Err("invalid magic".into());
        }
        pos += MAGIC.len();

        let read_section = |bytes: &[u8], pos: &mut usize| -> Result<Vec<u8>, Box<dyn Error>> {
            let len = read_u32(bytes, pos) as usize;
            let end = *pos + len;
            let compressed = &bytes[*pos..end];
            *pos = end;
            Ok(zstd::decode_all(compressed)?)
        };

        let type_dict_raw = read_section(bytes, &mut pos)?;
        let repo_dict_raw = read_section(bytes, &mut pos)?;
        let repo_id_dict_raw = read_section(bytes, &mut pos)?;

        let type_dict = TypeDict::decode(&type_dict_raw);
        let repo_dict = StringDict::decode(&repo_dict_raw);
        let repo_id_dict = RepoIdDict::decode(&repo_id_dict_raw);

        let row_group_count = read_u32(bytes, &mut pos) as usize;
        let mut events = Vec::new();

        for _ in 0..row_group_count {
            let min_id = read_u64(bytes, &mut pos);
            let min_ts = read_u64(bytes, &mut pos);
            let min_repo_id_idx = read_u32(bytes, &mut pos);

            let row_count = read_u32(bytes, &mut pos) as usize;
            let type_indices_raw = read_section(bytes, &mut pos)?;
            let repo_id_idx_offsets_raw = read_section(bytes, &mut pos)?;
            let repo_name_variant_idx_raw = read_section(bytes, &mut pos)?;
            let id_deltas_raw = read_section(bytes, &mut pos)?;
            let ts_deltas_raw = read_section(bytes, &mut pos)?;

            let type_indices = unpack_bits_u8(&type_indices_raw, row_count);
            let repo_id_idx_offsets = unpack_bits_u32(&repo_id_idx_offsets_raw, row_count);
            let repo_name_variant_idx = unpack_bits_u32(&repo_name_variant_idx_raw, row_count);
            let id_deltas = unpack_bits_u64(&id_deltas_raw, row_count);
            let ts_deltas = unpack_bits_u64(&ts_deltas_raw, row_count);

            if type_indices.len() != row_count {
                return Err("column length mismatch".into());
            }

            let mut id_offsets = Vec::with_capacity(row_count);
            if !id_deltas.is_empty() {
                let mut cur = zigzag_decode(id_deltas[0]);
                if cur < 0 {
                    return Err("negative id base".into());
                }
                id_offsets.push(cur as u64);
                for &delta in id_deltas.iter().skip(1) {
                    cur += zigzag_decode(delta);
                    if cur < 0 {
                        return Err("negative id offset".into());
                    }
                    id_offsets.push(cur as u64);
                }
            }

            let mut ts_offsets = Vec::with_capacity(row_count);
            if !ts_deltas.is_empty() {
                let mut cur = ts_deltas[0];
                ts_offsets.push(cur);
                for &delta in ts_deltas.iter().skip(1) {
                    cur += delta;
                    ts_offsets.push(cur);
                }
            }

            events.reserve(row_count);

            for i in 0..row_count {
                let type_idx = type_indices[i];
                let repo_id_idx = min_repo_id_idx + repo_id_idx_offsets[i];
                let repo_id = repo_id_dict.repo_id(repo_id_idx);
                let variant_idx = repo_name_variant_idx[i];
                if variant_idx as usize >= repo_id_dict.name_variants[repo_id_idx as usize].len() {
                    return Err("repo name variant idx out of range".into());
                }
                let repo_name_idx = repo_id_dict.name_idx(repo_id_idx, variant_idx);
                let repo_name = repo_dict.get_string(repo_name_idx).to_string();
                let repo_url = format!("https://api.github.com/repos/{}", repo_name);

                let event_type = type_dict.get_type(type_idx).to_string();
                let event_id = min_id + id_offsets[i];
                let timestamp = min_ts + ts_offsets[i];

                events.push((
                    EventKey {
                        event_type,
                        id: event_id.to_string(),
                    },
                    EventValue {
                        repo: Repo {
                            id: repo_id,
                            name: repo_name,
                            url: repo_url,
                        },
                        created_at: format_timestamp(timestamp),
                    },
                ));
            }
        }

        events.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(events)
    }
}
