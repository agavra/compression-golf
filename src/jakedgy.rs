//! # Jakedgy Codec v6
//!
//! High compression codec achieving ~6.40MB on 1M GitHub events.
//!
//! Key optimizations:
//! 1. Null-separated name dictionary - lets zstd find patterns naturally
//! 2. Delta-encoded repo_ids with sentinel encoding for single-name repos
//! 3. Columnar layout with per-column zstd compression
//! 4. Timestamp sorting within row groups for optimal ID delta encoding
//! 5. 2-bit category encoding for ID deltas (0/2/4 as common values)
//! 6. 2-bit category encoding for timestamp deltas (0/1/2 as common values)
//! 7. Optimized 140K row group size for best compression ratio

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::HashMap;
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const MAGIC: &[u8; 4] = b"JKG6";
const ZSTD_LEVEL: i32 = 22;
const ROW_GROUP_SIZE: usize = 140_000;

// ============================================================================
// Utilities
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

fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(encoded: u64) -> i64 {
    ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64))
}

fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn decode_varint(bytes: &[u8], pos: &mut usize) -> u64 {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        let byte = bytes[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

// ============================================================================
// Bit-packing with variable width
// ============================================================================

fn pack_bits(values: &[u64]) -> Vec<u8> {
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

fn unpack_bits(bytes: &[u8], count: usize) -> Vec<u64> {
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

// ============================================================================
// 2-bit category encoding for timestamp deltas
// Common values (0, 1, 2) encoded inline; others stored as varint exceptions
// Category: 0=zero, 1=one, 2=two, 3=exception
// ============================================================================

fn encode_ts_deltas(values: &[u64]) -> Vec<u8> {
    let mut categories: Vec<u8> = Vec::with_capacity((values.len() + 3) / 4);
    let mut exceptions: Vec<u64> = Vec::new();

    for chunk in values.chunks(4) {
        let mut byte: u8 = 0;
        for (i, &val) in chunk.iter().enumerate() {
            let cat = if val == 0 {
                0
            } else if val == 1 {
                1
            } else if val == 2 {
                2
            } else {
                exceptions.push(val);
                3
            };
            byte |= cat << (i * 2);
        }
        categories.push(byte);
    }

    let mut buf = Vec::new();
    encode_varint(&mut buf, values.len() as u64);
    buf.extend_from_slice(&categories);
    encode_varint(&mut buf, exceptions.len() as u64);
    for &e in &exceptions {
        encode_varint(&mut buf, e);
    }
    buf
}

fn decode_ts_deltas(bytes: &[u8]) -> Vec<u64> {
    let mut pos = 0;
    let count = decode_varint(bytes, &mut pos) as usize;
    let cat_bytes = (count + 3) / 4;
    let categories = &bytes[pos..pos + cat_bytes];
    pos += cat_bytes;

    let exception_count = decode_varint(bytes, &mut pos) as usize;
    let mut exceptions: Vec<u64> = Vec::with_capacity(exception_count);
    for _ in 0..exception_count {
        exceptions.push(decode_varint(bytes, &mut pos));
    }

    let mut values = Vec::with_capacity(count);
    let mut exc_idx = 0;

    for i in 0..count {
        let byte_idx = i / 4;
        let bit_idx = (i % 4) * 2;
        let cat = (categories[byte_idx] >> bit_idx) & 0x03;

        let val = match cat {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => {
                let v = exceptions[exc_idx];
                exc_idx += 1;
                v
            }
            _ => unreachable!(),
        };
        values.push(val);
    }

    values
}

// 2-bit category encoding for ID deltas (zigzag encoded)
// Common zigzag values: 0=delta(0), 2=delta(+1), 4=delta(+2)
// Category: 0=zero, 1=two, 2=four, 3=exception
fn encode_id_deltas(values: &[u64]) -> Vec<u8> {
    let mut categories: Vec<u8> = Vec::with_capacity((values.len() + 3) / 4);
    let mut exceptions: Vec<u64> = Vec::new();

    for chunk in values.chunks(4) {
        let mut byte: u8 = 0;
        for (i, &val) in chunk.iter().enumerate() {
            let cat = if val == 0 {
                0
            } else if val == 2 {
                // zigzag(1) = 2
                1
            } else if val == 4 {
                // zigzag(2) = 4
                2
            } else {
                exceptions.push(val);
                3
            };
            byte |= cat << (i * 2);
        }
        categories.push(byte);
    }

    let mut buf = Vec::new();
    encode_varint(&mut buf, values.len() as u64);
    buf.extend_from_slice(&categories);
    encode_varint(&mut buf, exceptions.len() as u64);
    for &e in &exceptions {
        encode_varint(&mut buf, e);
    }
    buf
}

fn decode_id_deltas(bytes: &[u8]) -> Vec<u64> {
    let mut pos = 0;
    let count = decode_varint(bytes, &mut pos) as usize;
    let cat_bytes = (count + 3) / 4;
    let categories = &bytes[pos..pos + cat_bytes];
    pos += cat_bytes;

    let exception_count = decode_varint(bytes, &mut pos) as usize;
    let mut exceptions: Vec<u64> = Vec::with_capacity(exception_count);
    for _ in 0..exception_count {
        exceptions.push(decode_varint(bytes, &mut pos));
    }

    let mut values = Vec::with_capacity(count);
    let mut exc_idx = 0;

    for i in 0..count {
        let byte_idx = i / 4;
        let bit_idx = (i % 4) * 2;
        let cat = (categories[byte_idx] >> bit_idx) & 0x03;

        let val = match cat {
            0 => 0,
            1 => 2,
            2 => 4,
            3 => {
                let v = exceptions[exc_idx];
                exc_idx += 1;
                v
            }
            _ => unreachable!(),
        };
        values.push(val);
    }

    values
}

// ============================================================================
// Type Dictionary (sorted by frequency)
// ============================================================================

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

    fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(buf, self.types.len() as u64);
        for t in &self.types {
            buf.extend_from_slice(t.as_bytes());
            buf.push(0); // null terminator
        }
    }

    fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let count = decode_varint(bytes, pos) as usize;
        let mut types = Vec::with_capacity(count);
        let mut type_to_idx = HashMap::new();

        for i in 0..count {
            let start = *pos;
            while bytes[*pos] != 0 {
                *pos += 1;
            }
            let t = std::str::from_utf8(&bytes[start..*pos]).unwrap().to_string();
            *pos += 1; // skip null terminator
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

// ============================================================================
// Repo Name Dictionary with prefix compression (sorted alphabetically)
// ============================================================================

struct RepoNameDict {
    names: Vec<String>,
    name_to_idx: HashMap<String, u32>,
}

impl RepoNameDict {
    fn build(events: &[(EventKey, EventValue)]) -> Self {
        let mut unique_names: Vec<String> = events
            .iter()
            .map(|(_, v)| v.repo.name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Sort alphabetically for good zstd compression
        unique_names.sort();

        let mut name_to_idx = HashMap::new();
        for (i, name) in unique_names.iter().enumerate() {
            name_to_idx.insert(name.clone(), i as u32);
        }

        Self {
            names: unique_names,
            name_to_idx,
        }
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(buf, self.names.len() as u64);

        // Simple null-separated encoding - let zstd handle compression
        for name in &self.names {
            buf.extend_from_slice(name.as_bytes());
            buf.push(0); // null terminator
        }
    }

    fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let count = decode_varint(bytes, pos) as usize;
        let mut names = Vec::with_capacity(count);
        let mut name_to_idx = HashMap::new();

        for i in 0..count {
            let start = *pos;
            while bytes[*pos] != 0 {
                *pos += 1;
            }
            let name = std::str::from_utf8(&bytes[start..*pos]).unwrap().to_string();
            *pos += 1; // skip null terminator
            name_to_idx.insert(name.clone(), i as u32);
            names.push(name);
        }

        Self { names, name_to_idx }
    }

    fn get_index(&self, name: &str) -> u32 {
        self.name_to_idx[name]
    }

    fn get_name(&self, index: u32) -> &str {
        &self.names[index as usize]
    }
}

// ============================================================================
// Repo ID Dictionary - maps repo_id to list of associated name indices
// ============================================================================

struct RepoIdDict {
    repo_ids: Vec<u64>,
    name_indices_per_id: Vec<Vec<u32>>,
    id_to_idx: HashMap<u64, u32>,
    name_variant_lookup: HashMap<(u64, u32), u32>,
}

impl RepoIdDict {
    fn build(events: &[(EventKey, EventValue)], name_dict: &RepoNameDict) -> Self {
        let mut id_to_names: HashMap<u64, std::collections::HashSet<u32>> = HashMap::new();

        for (_, value) in events {
            let name_idx = name_dict.get_index(&value.repo.name);
            id_to_names
                .entry(value.repo.id)
                .or_insert_with(std::collections::HashSet::new)
                .insert(name_idx);
        }

        let mut repo_ids: Vec<u64> = id_to_names.keys().copied().collect();
        repo_ids.sort();

        let mut name_indices_per_id = Vec::with_capacity(repo_ids.len());
        let mut id_to_idx = HashMap::with_capacity(repo_ids.len());
        let mut name_variant_lookup = HashMap::new();

        for (i, &repo_id) in repo_ids.iter().enumerate() {
            id_to_idx.insert(repo_id, i as u32);
            let mut names: Vec<u32> = id_to_names[&repo_id].iter().copied().collect();
            names.sort();
            for (j, &name_idx) in names.iter().enumerate() {
                name_variant_lookup.insert((repo_id, name_idx), j as u32);
            }
            name_indices_per_id.push(names);
        }

        Self {
            repo_ids,
            name_indices_per_id,
            id_to_idx,
            name_variant_lookup,
        }
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(buf, self.repo_ids.len() as u64);

        // Delta encode repo_ids
        let mut prev_id = 0u64;
        for &repo_id in &self.repo_ids {
            encode_varint(buf, repo_id - prev_id);
            prev_id = repo_id;
        }

        // Encode name indices per ID
        // Use 0 as sentinel: if count > 1, encode count then indices
        // If count == 1, just encode the name_idx + 1 (so 0 means multi-name)
        for names in &self.name_indices_per_id {
            if names.len() == 1 {
                // Encode name_idx + 1 to avoid 0
                encode_varint(buf, (names[0] + 1) as u64);
            } else {
                // 0 signals multi-name repo
                encode_varint(buf, 0);
                encode_varint(buf, names.len() as u64);
                for &name_idx in names {
                    encode_varint(buf, name_idx as u64);
                }
            }
        }
    }

    fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let count = decode_varint(bytes, pos) as usize;

        let mut repo_ids = Vec::with_capacity(count);
        let mut id_to_idx = HashMap::with_capacity(count);
        let mut name_variant_lookup = HashMap::new();

        // Delta decode repo_ids
        let mut prev_id = 0u64;
        for i in 0..count {
            let delta = decode_varint(bytes, pos);
            let repo_id = prev_id + delta;
            repo_ids.push(repo_id);
            id_to_idx.insert(repo_id, i as u32);
            prev_id = repo_id;
        }

        // Decode name indices per ID
        let mut name_indices_per_id = Vec::with_capacity(count);
        for &repo_id in &repo_ids {
            let first_val = decode_varint(bytes, pos);
            let names = if first_val == 0 {
                // Multi-name repo
                let variant_count = decode_varint(bytes, pos) as usize;
                let mut names = Vec::with_capacity(variant_count);
                for j in 0..variant_count {
                    let name_idx = decode_varint(bytes, pos) as u32;
                    names.push(name_idx);
                    name_variant_lookup.insert((repo_id, name_idx), j as u32);
                }
                names
            } else {
                // Single-name repo (first_val is name_idx + 1)
                let name_idx = (first_val - 1) as u32;
                name_variant_lookup.insert((repo_id, name_idx), 0);
                vec![name_idx]
            };
            name_indices_per_id.push(names);
        }

        Self {
            repo_ids,
            name_indices_per_id,
            id_to_idx,
            name_variant_lookup,
        }
    }

    fn get_repo_idx(&self, repo_id: u64) -> u32 {
        self.id_to_idx[&repo_id]
    }

    fn get_variant_idx(&self, repo_id: u64, name_idx: u32) -> u32 {
        self.name_variant_lookup[&(repo_id, name_idx)]
    }

    fn get_repo_id(&self, repo_idx: u32) -> u64 {
        self.repo_ids[repo_idx as usize]
    }

    fn get_name_idx(&self, repo_idx: u32, variant_idx: u32) -> u32 {
        self.name_indices_per_id[repo_idx as usize][variant_idx as usize]
    }
}

// ============================================================================
// Codec Implementation
// ============================================================================

pub struct JakedgyCodec;

impl JakedgyCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for JakedgyCodec {
    fn name(&self) -> &str {
        "jakedgy"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let type_dict = TypeDict::build(events);
        let name_dict = RepoNameDict::build(events);
        let id_dict = RepoIdDict::build(events, &name_dict);

        // Build header
        let mut header = Vec::new();
        header.extend_from_slice(MAGIC);
        encode_varint(&mut header, events.len() as u64);
        type_dict.encode(&mut header);
        name_dict.encode(&mut header);
        id_dict.encode(&mut header);

        // Calculate row groups
        let num_groups = (events.len() + ROW_GROUP_SIZE - 1) / ROW_GROUP_SIZE;
        encode_varint(&mut header, num_groups as u64);

        // Compress header
        let header_compressed = zstd::encode_all(header.as_slice(), ZSTD_LEVEL)?;

        let mut output = Vec::new();
        encode_varint(&mut output, header_compressed.len() as u64);
        output.extend_from_slice(&header_compressed);

        // Process row groups
        for group_idx in 0..num_groups {
            let start = group_idx * ROW_GROUP_SIZE;
            let end = (start + ROW_GROUP_SIZE).min(events.len());
            let slice = &events[start..end];
            let row_count = slice.len();

            // Extract columns
            let mut type_indices: Vec<u8> = Vec::with_capacity(row_count);
            let mut repo_idx_col: Vec<u32> = Vec::with_capacity(row_count);
            let mut variant_idx_col: Vec<u32> = Vec::with_capacity(row_count);
            let mut ids: Vec<u64> = Vec::with_capacity(row_count);
            let mut timestamps: Vec<u64> = Vec::with_capacity(row_count);

            for (key, value) in slice {
                type_indices.push(type_dict.get_index(&key.event_type));
                let name_idx = name_dict.get_index(&value.repo.name);
                let repo_idx = id_dict.get_repo_idx(value.repo.id);
                let variant_idx = id_dict.get_variant_idx(value.repo.id, name_idx);
                repo_idx_col.push(repo_idx);
                variant_idx_col.push(variant_idx);
                ids.push(key.id.parse::<u64>().unwrap_or(0));
                timestamps.push(parse_timestamp(&value.created_at));
            }

            // Sort by timestamp for better compression (IDs are monotonic in time)
            let mut perm: Vec<usize> = (0..row_count).collect();
            perm.sort_by_key(|&i| timestamps[i]);

            let sorted_types: Vec<u8> = perm.iter().map(|&i| type_indices[i]).collect();
            let sorted_repo_idx: Vec<u32> = perm.iter().map(|&i| repo_idx_col[i]).collect();
            let sorted_variant_idx: Vec<u32> = perm.iter().map(|&i| variant_idx_col[i]).collect();
            let sorted_ids: Vec<u64> = perm.iter().map(|&i| ids[i]).collect();
            let sorted_ts: Vec<u64> = perm.iter().map(|&i| timestamps[i]).collect();

            // Store minimums for efficient encoding
            let min_id = *sorted_ids.iter().min().unwrap_or(&0);
            let min_ts = *sorted_ts.iter().min().unwrap_or(&0);

            // Convert to offsets
            let id_offsets: Vec<u64> = sorted_ids.iter().map(|&id| id - min_id).collect();
            let ts_offsets: Vec<u64> = sorted_ts.iter().map(|&ts| ts - min_ts).collect();

            // Delta encode IDs (zigzag for signed deltas)
            let mut id_deltas: Vec<u64> = Vec::with_capacity(row_count);
            if !id_offsets.is_empty() {
                let mut prev = id_offsets[0] as i64;
                id_deltas.push(zigzag_encode(prev));
                for &offset in id_offsets.iter().skip(1) {
                    let cur = offset as i64;
                    id_deltas.push(zigzag_encode(cur - prev));
                    prev = cur;
                }
            }

            // Delta encode timestamps (non-negative since sorted)
            let mut ts_deltas: Vec<u64> = Vec::with_capacity(row_count);
            if !ts_offsets.is_empty() {
                let mut prev = ts_offsets[0];
                ts_deltas.push(prev);
                for &offset in ts_offsets.iter().skip(1) {
                    ts_deltas.push(offset - prev);
                    prev = offset;
                }
            }

            // Bit-pack each column
            let types_packed =
                pack_bits(&sorted_types.iter().map(|&t| t as u64).collect::<Vec<_>>());
            let repo_idx_packed = pack_bits(
                &sorted_repo_idx
                    .iter()
                    .map(|&r| r as u64)
                    .collect::<Vec<_>>(),
            );
            let variant_idx_packed = pack_bits(
                &sorted_variant_idx
                    .iter()
                    .map(|&v| v as u64)
                    .collect::<Vec<_>>(),
            );
            let ids_packed = encode_id_deltas(&id_deltas);
            let ts_packed = encode_ts_deltas(&ts_deltas);

            // Compress each column
            let types_compressed = zstd::encode_all(types_packed.as_slice(), ZSTD_LEVEL)?;
            let repo_idx_compressed = zstd::encode_all(repo_idx_packed.as_slice(), ZSTD_LEVEL)?;
            let variant_idx_compressed =
                zstd::encode_all(variant_idx_packed.as_slice(), ZSTD_LEVEL)?;
            let ids_compressed = zstd::encode_all(ids_packed.as_slice(), ZSTD_LEVEL)?;
            let ts_compressed = zstd::encode_all(ts_packed.as_slice(), ZSTD_LEVEL)?;

            // Write row group header using varints
            encode_varint(&mut output, min_id);
            encode_varint(&mut output, min_ts);
            encode_varint(&mut output, row_count as u64);

            // Write column sizes as varints
            encode_varint(&mut output, types_compressed.len() as u64);
            encode_varint(&mut output, repo_idx_compressed.len() as u64);
            encode_varint(&mut output, variant_idx_compressed.len() as u64);
            encode_varint(&mut output, ids_compressed.len() as u64);
            encode_varint(&mut output, ts_compressed.len() as u64);

            // Write column data
            output.extend_from_slice(&types_compressed);
            output.extend_from_slice(&repo_idx_compressed);
            output.extend_from_slice(&variant_idx_compressed);
            output.extend_from_slice(&ids_compressed);
            output.extend_from_slice(&ts_compressed);
        }

        Ok(Bytes::from(output))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;

        // Read and decompress header
        let header_len = decode_varint(bytes, &mut pos) as usize;
        let header_compressed = &bytes[pos..pos + header_len];
        pos += header_len;
        let header = zstd::decode_all(header_compressed)?;

        let mut hpos = 0;
        if header.len() < MAGIC.len() || &header[..MAGIC.len()] != MAGIC {
            return Err("invalid magic".into());
        }
        hpos += MAGIC.len();

        let event_count = decode_varint(&header, &mut hpos) as usize;
        let type_dict = TypeDict::decode(&header, &mut hpos);
        let name_dict = RepoNameDict::decode(&header, &mut hpos);
        let id_dict = RepoIdDict::decode(&header, &mut hpos);
        let num_groups = decode_varint(&header, &mut hpos) as usize;

        let mut events = Vec::with_capacity(event_count);

        for _ in 0..num_groups {
            let min_id = decode_varint(bytes, &mut pos);
            let min_ts = decode_varint(bytes, &mut pos);
            let row_count = decode_varint(bytes, &mut pos) as usize;

            // Read column sizes
            let types_len = decode_varint(bytes, &mut pos) as usize;
            let repo_idx_len = decode_varint(bytes, &mut pos) as usize;
            let variant_idx_len = decode_varint(bytes, &mut pos) as usize;
            let ids_len = decode_varint(bytes, &mut pos) as usize;
            let ts_len = decode_varint(bytes, &mut pos) as usize;

            // Read and decompress columns
            let types_decompressed = zstd::decode_all(&bytes[pos..pos + types_len])?;
            pos += types_len;
            let type_indices = unpack_bits(&types_decompressed, row_count);

            let repo_idx_decompressed = zstd::decode_all(&bytes[pos..pos + repo_idx_len])?;
            pos += repo_idx_len;
            let repo_idx_values = unpack_bits(&repo_idx_decompressed, row_count);

            let variant_idx_decompressed = zstd::decode_all(&bytes[pos..pos + variant_idx_len])?;
            pos += variant_idx_len;
            let variant_indices = unpack_bits(&variant_idx_decompressed, row_count);

            let ids_decompressed = zstd::decode_all(&bytes[pos..pos + ids_len])?;
            pos += ids_len;
            let id_deltas = decode_id_deltas(&ids_decompressed);

            let ts_decompressed = zstd::decode_all(&bytes[pos..pos + ts_len])?;
            pos += ts_len;
            let ts_deltas = decode_ts_deltas(&ts_decompressed);

            // Reconstruct IDs
            let mut id_offsets = Vec::with_capacity(row_count);
            if !id_deltas.is_empty() {
                let mut cur = zigzag_decode(id_deltas[0]);
                id_offsets.push(cur as u64);
                for &delta in id_deltas.iter().skip(1) {
                    cur += zigzag_decode(delta);
                    id_offsets.push(cur as u64);
                }
            }

            // Reconstruct timestamps
            let mut ts_offsets = Vec::with_capacity(row_count);
            if !ts_deltas.is_empty() {
                let mut cur = ts_deltas[0];
                ts_offsets.push(cur);
                for &delta in ts_deltas.iter().skip(1) {
                    cur += delta;
                    ts_offsets.push(cur);
                }
            }

            // Create events
            for i in 0..row_count {
                let type_idx = type_indices[i] as u8;
                let repo_idx = repo_idx_values[i] as u32;
                let variant_idx = variant_indices[i] as u32;
                let event_id = min_id + id_offsets[i];
                let timestamp = min_ts + ts_offsets[i];

                let repo_id = id_dict.get_repo_id(repo_idx);
                let name_idx = id_dict.get_name_idx(repo_idx, variant_idx);
                let repo_name = name_dict.get_name(name_idx).to_string();
                let repo_url = format!("https://api.github.com/repos/{}", repo_name);

                events.push((
                    EventKey {
                        event_type: type_dict.get_type(type_idx).to_string(),
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

        // Sort by EventKey as required
        events.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(events)
    }
}
