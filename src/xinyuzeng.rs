//! # Xinyuzeng Codec
//!
//! Columnar layout with dictionary-encoded repo data, delta-encoded ids/timestamps,
//! and per-column zstd compression.
//!
//! Tried and reverted (worse sizes than current baseline):
//! - Split repo_names into counts/lengths/bytes streams.
//! - Global repo-name dictionary with per-repo index lists.
//! - Owner/suffix split with owner dict + suffix bytes.
//! - Reordering encode by repo-id blocks (blew up id/ts deltas).

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::{HashMap, HashSet};
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const MAGIC: &[u8; 4] = b"XYZ1";
const ZSTD_LEVEL: i32 = 22;

const COLUMN_NAMES: [&str; 8] = [
    "type_dict",
    "repo_ids",
    "repo_names",
    "type_idx",
    "repo_id_idx",
    "repo_name_variant_idx",
    "id_deltas",
    "ts_deltas",
];

fn debug_enabled() -> bool {
    std::env::var("XINYU_DEBUG").is_ok()
}

fn parse_timestamp(ts: &str) -> Result<u64, Box<dyn Error>> {
    let dt = DateTime::parse_from_rfc3339(ts)?;
    Ok(dt.timestamp() as u64)
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

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, Box<dyn Error>> {
    if *pos + 4 > bytes.len() {
        return Err("unexpected EOF".into());
    }
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[*pos..*pos + 4]);
    *pos += 4;
    Ok(u32::from_le_bytes(arr))
}

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

fn unpack_bits_u64(bytes: &[u8], count: usize) -> Result<Vec<u64>, Box<dyn Error>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if bytes.is_empty() {
        return Err("empty bitpack buffer".into());
    }
    let bit_width = bytes[0] as usize;
    if bit_width == 0 {
        return Ok(vec![0; count]);
    }

    let mut values = Vec::with_capacity(count);
    let mut bit_pos: usize = 0;

    for _ in 0..count {
        let mut value: u64 = 0;
        for bit_idx in 0..bit_width {
            let byte_idx = 1 + (bit_pos / 8);
            if byte_idx >= bytes.len() {
                return Err("bitpack overflow".into());
            }
            let bit_in_byte = bit_pos % 8;
            let bit = ((bytes[byte_idx] >> bit_in_byte) & 1) as u64;
            value |= bit << bit_idx;
            bit_pos += 1;
        }
        values.push(value);
    }

    Ok(values)
}

fn pack_bits_u32(values: &[u32]) -> Vec<u8> {
    let tmp: Vec<u64> = values.iter().map(|&v| v as u64).collect();
    pack_bits_u64(&tmp)
}

fn pack_u32_le(values: &[u32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * 4);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn unpack_bits_u32(bytes: &[u8], count: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    Ok(unpack_bits_u64(bytes, count)?
        .into_iter()
        .map(|v| v as u32)
        .collect())
}

fn unpack_u32_le(bytes: &[u8], count: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    if bytes.len() < count * 4 {
        return Err("u32 buffer too small".into());
    }
    let mut out = Vec::with_capacity(count);
    let mut pos = 0;
    for _ in 0..count {
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&bytes[pos..pos + 4]);
        pos += 4;
        out.push(u32::from_le_bytes(arr));
    }
    Ok(out)
}

fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

fn delta_encode_signed(values: &[u64]) -> Result<Vec<u64>, Box<dyn Error>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(values.len());
    let mut prev = i64::try_from(values[0])?;
    out.push(zigzag_encode(prev));
    for &v in values.iter().skip(1) {
        let cur = i64::try_from(v)?;
        out.push(zigzag_encode(cur - prev));
        prev = cur;
    }
    Ok(out)
}

fn delta_decode_signed(values: &[u64]) -> Result<Vec<u64>, Box<dyn Error>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(values.len());
    let mut cur = zigzag_decode(values[0]);
    if cur < 0 {
        return Err("negative base".into());
    }
    out.push(cur as u64);
    for &v in values.iter().skip(1) {
        cur += zigzag_decode(v);
        if cur < 0 {
            return Err("negative value".into());
        }
        out.push(cur as u64);
    }
    Ok(out)
}

fn delta_encode_unsigned(values: &[u64]) -> Vec<u64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut prev = values[0];
    out.push(prev);
    for &v in values.iter().skip(1) {
        out.push(v - prev);
        prev = v;
    }
    out
}

fn delta_decode_unsigned(values: &[u64]) -> Vec<u64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut cur = values[0];
    out.push(cur);
    for &v in values.iter().skip(1) {
        cur += v;
        out.push(cur);
    }
    out
}

struct StringDict {
    strings: Vec<String>,
    str_to_idx: HashMap<String, u32>,
}

impl StringDict {
    fn build_by_freq(events: &[(EventKey, EventValue)]) -> Self {
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for (key, _) in events {
            *freq.entry(&key.event_type).or_insert(0) += 1;
        }

        let mut types_with_freq: Vec<_> = freq.into_iter().collect();
        types_with_freq.sort_by(|a, b| b.1.cmp(&a.1));

        let mut strings = Vec::with_capacity(types_with_freq.len());
        let mut str_to_idx = HashMap::new();
        for (i, (t, _)) in types_with_freq.into_iter().enumerate() {
            strings.push(t.to_string());
            str_to_idx.insert(t.to_string(), i as u32);
        }

        Self {
            strings,
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

    fn decode(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
        let mut pos = 0;
        let count = read_u32(bytes, &mut pos)? as usize;
        let mut lens = Vec::with_capacity(count);
        for _ in 0..count {
            lens.push(read_u32(bytes, &mut pos)? as usize);
        }

        let mut strings = Vec::with_capacity(count);
        let mut str_to_idx = HashMap::new();
        for (i, len) in lens.into_iter().enumerate() {
            if pos + len > bytes.len() {
                return Err("string dict overflow".into());
            }
            let s = std::str::from_utf8(&bytes[pos..pos + len])?.to_string();
            pos += len;
            str_to_idx.insert(s.clone(), i as u32);
            strings.push(s);
        }

        Ok(Self {
            strings,
            str_to_idx,
        })
    }

    fn get_index(&self, s: &str) -> u32 {
        self.str_to_idx[s]
    }

    fn get_string(&self, index: u32) -> &str {
        &self.strings[index as usize]
    }
}

struct RepoDict {
    repo_ids: Vec<u64>,
    repo_names: Vec<Vec<String>>,
    id_to_idx: HashMap<u64, u32>,
    name_to_idx: Vec<HashMap<String, u32>>,
}

impl RepoDict {
    fn build(events: &[(EventKey, EventValue)]) -> Self {
        let mut map: HashMap<u64, HashSet<String>> = HashMap::new();
        for (_, value) in events {
            map.entry(value.repo.id)
                .or_insert_with(HashSet::new)
                .insert(value.repo.name.clone());
        }

        let mut repo_ids: Vec<u64> = map.keys().copied().collect();
        repo_ids.sort();

        let mut repo_names = Vec::with_capacity(repo_ids.len());
        let mut id_to_idx = HashMap::with_capacity(repo_ids.len());
        let mut name_to_idx = Vec::with_capacity(repo_ids.len());

        for (i, repo_id) in repo_ids.iter().enumerate() {
            id_to_idx.insert(*repo_id, i as u32);
            let mut names: Vec<String> = map
                .get(repo_id)
                .unwrap()
                .iter()
                .cloned()
                .collect();
            names.sort();
            let mut name_map = HashMap::new();
            for (j, name) in names.iter().enumerate() {
                name_map.insert(name.clone(), j as u32);
            }
            repo_names.push(names);
            name_to_idx.push(name_map);
        }

        Self {
            repo_ids,
            repo_names,
            id_to_idx,
            name_to_idx,
        }
    }

    fn encode_repo_ids(&self) -> Vec<u8> {
        let deltas = delta_encode_unsigned(&self.repo_ids);
        pack_bits_u64(&deltas)
    }

    fn encode_repo_names(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_u32(&mut buf, self.repo_names.len() as u32);
        for names in &self.repo_names {
            write_u32(&mut buf, names.len() as u32);
        }
        for names in &self.repo_names {
            for name in names {
                buf.extend_from_slice(name.as_bytes());
                buf.push(0);
            }
        }
        buf
    }

    fn decode(
        repo_count: usize,
        repo_ids_raw: &[u8],
        repo_names_raw: &[u8],
    ) -> Result<Self, Box<dyn Error>> {
        let deltas = unpack_bits_u64(repo_ids_raw, repo_count)?;
        let repo_ids = delta_decode_unsigned(&deltas);

        let mut pos = 0;
        let names_repo_count = read_u32(repo_names_raw, &mut pos)? as usize;
        if names_repo_count != repo_count {
            return Err("repo count mismatch".into());
        }
        let mut name_counts = Vec::with_capacity(names_repo_count);
        for _ in 0..repo_count {
            name_counts.push(read_u32(repo_names_raw, &mut pos)? as usize);
        }

        let mut repo_names = Vec::with_capacity(repo_count);
        let mut cursor = pos;
        for &count in &name_counts {
            let mut names = Vec::with_capacity(count);
            for _ in 0..count {
                if cursor >= repo_names_raw.len() {
                    return Err("repo names overflow".into());
                }
                let start = cursor;
                while cursor < repo_names_raw.len() && repo_names_raw[cursor] != 0 {
                    cursor += 1;
                }
                if cursor >= repo_names_raw.len() {
                    return Err("repo names missing delimiter".into());
                }
                let s = std::str::from_utf8(&repo_names_raw[start..cursor])?.to_string();
                cursor += 1; // skip delimiter
                names.push(s);
            }
            repo_names.push(names);
        }
        if cursor != repo_names_raw.len() {
            return Err("repo names trailing bytes".into());
        }

        let mut id_to_idx = HashMap::with_capacity(repo_ids.len());
        let mut name_to_idx = Vec::with_capacity(repo_ids.len());
        for (i, repo_id) in repo_ids.iter().enumerate() {
            id_to_idx.insert(*repo_id, i as u32);
            let mut name_map = HashMap::new();
            for (j, name) in repo_names[i].iter().enumerate() {
                name_map.insert(name.clone(), j as u32);
            }
            name_to_idx.push(name_map);
        }

        Ok(Self {
            repo_ids,
            repo_names,
            id_to_idx,
            name_to_idx,
        })
    }

    fn repo_id_index(&self, repo_id: u64) -> u32 {
        self.id_to_idx[&repo_id]
    }

    fn repo_name_index(&self, repo_id_idx: u32, repo_name: &str) -> u32 {
        self.name_to_idx[repo_id_idx as usize][repo_name]
    }

    fn repo_id(&self, idx: u32) -> u64 {
        self.repo_ids[idx as usize]
    }

    fn repo_name(&self, repo_id_idx: u32, name_idx: u32) -> &str {
        &self.repo_names[repo_id_idx as usize][name_idx as usize]
    }
}

pub struct XinyuzengCodec;

impl XinyuzengCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for XinyuzengCodec {
    fn name(&self) -> &str {
        "xinyuzeng"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let mut sorted_events: Vec<(EventKey, EventValue)> = events.to_vec();
        sorted_events.sort_by(|a, b| a.0.cmp(&b.0));

        let type_dict = StringDict::build_by_freq(&sorted_events);
        let repo_dict = RepoDict::build(&sorted_events);

        let type_dict_bytes = zstd::encode_all(type_dict.encode().as_slice(), ZSTD_LEVEL)?;
        let repo_ids_bytes = zstd::encode_all(repo_dict.encode_repo_ids().as_slice(), ZSTD_LEVEL)?;
        let repo_names_bytes =
            zstd::encode_all(repo_dict.encode_repo_names().as_slice(), ZSTD_LEVEL)?;

        let mut type_idx = Vec::with_capacity(sorted_events.len());
        let mut repo_id_idx = Vec::with_capacity(sorted_events.len());
        let mut repo_name_variant_idx = Vec::with_capacity(sorted_events.len());
        let mut ids = Vec::with_capacity(sorted_events.len());
        let mut timestamps = Vec::with_capacity(sorted_events.len());

        for (key, value) in &sorted_events {
            type_idx.push(type_dict.get_index(&key.event_type));
            let id_idx = repo_dict.repo_id_index(value.repo.id);
            repo_id_idx.push(id_idx);
            repo_name_variant_idx.push(repo_dict.repo_name_index(id_idx, &value.repo.name));
            let event_id = key.id.parse::<u64>()?;
            ids.push(event_id);
            let ts = parse_timestamp(&value.created_at)?;
            timestamps.push(ts);
        }

        let id_deltas = delta_encode_signed(&ids)?;
        let ts_deltas = delta_encode_signed(&timestamps)?;

        let type_idx_bytes = zstd::encode_all(pack_bits_u32(&type_idx).as_slice(), ZSTD_LEVEL)?;
        let repo_id_idx_bytes =
            zstd::encode_all(pack_u32_le(&repo_id_idx).as_slice(), ZSTD_LEVEL)?;
        let repo_name_variant_idx_bytes = zstd::encode_all(
            pack_bits_u32(&repo_name_variant_idx).as_slice(),
            ZSTD_LEVEL,
        )?;
        let id_deltas_bytes = zstd::encode_all(pack_bits_u64(&id_deltas).as_slice(), ZSTD_LEVEL)?;
        let ts_deltas_bytes = zstd::encode_all(pack_bits_u64(&ts_deltas).as_slice(), ZSTD_LEVEL)?;

        let debug = debug_enabled();
        if debug {
            let sizes = [
                type_dict_bytes.len(),
                repo_ids_bytes.len(),
                repo_names_bytes.len(),
                type_idx_bytes.len(),
                repo_id_idx_bytes.len(),
                repo_name_variant_idx_bytes.len(),
                id_deltas_bytes.len(),
                ts_deltas_bytes.len(),
            ];
            let total: usize = sizes.iter().sum();
            for (name, size) in COLUMN_NAMES.iter().zip(sizes.iter()) {
                eprintln!("XINYU_DEBUG col {}: {}", name, size);
            }
            eprintln!("XINYU_DEBUG total bytes: {}", total);
        }

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        write_u32(&mut out, sorted_events.len() as u32);
        write_u32(&mut out, type_dict.strings.len() as u32);
        write_u32(&mut out, repo_dict.repo_ids.len() as u32);

        let write_section = |buf: &mut Vec<u8>, bytes: &[u8]| {
            write_u32(buf, bytes.len() as u32);
            buf.extend_from_slice(bytes);
        };

        write_section(&mut out, &type_dict_bytes);
        write_section(&mut out, &repo_ids_bytes);
        write_section(&mut out, &repo_names_bytes);
        write_section(&mut out, &type_idx_bytes);
        write_section(&mut out, &repo_id_idx_bytes);
        write_section(&mut out, &repo_name_variant_idx_bytes);
        write_section(&mut out, &id_deltas_bytes);
        write_section(&mut out, &ts_deltas_bytes);

        Ok(Bytes::from(out))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Err("invalid magic".into());
        }
        pos += MAGIC.len();

        let row_count = read_u32(bytes, &mut pos)? as usize;
        let _type_count = read_u32(bytes, &mut pos)? as usize;
        let repo_count = read_u32(bytes, &mut pos)? as usize;

        let read_section = |bytes: &[u8], pos: &mut usize| -> Result<Vec<u8>, Box<dyn Error>> {
            let len = read_u32(bytes, pos)? as usize;
            let end = *pos + len;
            if end > bytes.len() {
                return Err("section overflow".into());
            }
            let compressed = &bytes[*pos..end];
            *pos = end;
            Ok(zstd::decode_all(compressed)?)
        };

        let type_dict_raw = read_section(bytes, &mut pos)?;
        let repo_ids_raw = read_section(bytes, &mut pos)?;
        let repo_names_raw = read_section(bytes, &mut pos)?;
        let type_idx_raw = read_section(bytes, &mut pos)?;
        let repo_id_idx_raw = read_section(bytes, &mut pos)?;
        let repo_name_variant_idx_raw = read_section(bytes, &mut pos)?;
        let id_deltas_raw = read_section(bytes, &mut pos)?;
        let ts_deltas_raw = read_section(bytes, &mut pos)?;

        let type_dict = StringDict::decode(&type_dict_raw)?;
        let repo_dict = RepoDict::decode(repo_count, &repo_ids_raw, &repo_names_raw)?;

        let type_idx = unpack_bits_u32(&type_idx_raw, row_count)?;
        let repo_id_idx = unpack_u32_le(&repo_id_idx_raw, row_count)?;
        let repo_name_variant_idx = unpack_bits_u32(&repo_name_variant_idx_raw, row_count)?;
        let id_deltas = unpack_bits_u64(&id_deltas_raw, row_count)?;
        let ts_deltas = unpack_bits_u64(&ts_deltas_raw, row_count)?;

        let ids = delta_decode_signed(&id_deltas)?;
        let timestamps = delta_decode_signed(&ts_deltas)?;

        if ids.len() != row_count || timestamps.len() != row_count {
            return Err("row count mismatch".into());
        }

        let mut events = Vec::with_capacity(row_count);
        for i in 0..row_count {
            let event_type = type_dict.get_string(type_idx[i]).to_string();
            let repo_id = repo_dict.repo_id(repo_id_idx[i]);
            let repo_name = repo_dict.repo_name(repo_id_idx[i], repo_name_variant_idx[i]);
            let repo_url = format!("https://api.github.com/repos/{}", repo_name);
            events.push((
                EventKey {
                    id: ids[i].to_string(),
                    event_type,
                },
                EventValue {
                    repo: Repo {
                        id: repo_id,
                        name: repo_name.to_string(),
                        url: repo_url,
                    },
                    created_at: format_timestamp(timestamps[i]),
                },
            ));
        }

        Ok(events)
    }
}
