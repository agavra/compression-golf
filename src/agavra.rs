//! # Agavra Codec
//!
//! **Strategy:** Delta encoding, block bit-packing, prefix-compressed dictionaries,
//! and a combined repo tuple dictionary (lossless).
//!
//! ## How it works:
//!
//! ### 1. Type Enumeration
//! Event types are mapped to single-byte indices (0-13 for 14 types).
//! Stored once at the start, sorted by frequency.
//!
//! ### 2. String Dictionaries with Prefix Compression
//! Usernames and repo names are stored in separate sorted dictionaries.
//! Each entry stores: prefix_len (bits shared with previous) + suffix.
//! Prefix/suffix lengths are bit-packed for efficiency.
//!
//! ### 3. Repo Tuple Dictionary
//! Instead of storing (user_idx, repo_idx, repo_id) separately per event,
//! unique tuples are stored once in a dictionary. Events reference tuples
//! by index, reducing 3 fields to 1.
//!
//! ### 4. Delta Encoding
//! Event IDs, tuple indices, and timestamps are delta-encoded:
//! ```text
//! IDs:    2489651045, 2489651051, 2489651053
//! Deltas: 2489651045, +6,         +2          (much smaller!)
//! ```
//! Deltas are zigzag-encoded so negative values stay compact.
//!
//! ### 5. Block Bit-Packing
//! Events are processed in blocks of 64. For each block and field:
//! - Find the maximum bit-width needed
//! - Pack all values at that width (no wasted continuation bits)
//!
//! ## Data layout:
//!
//! ```text
//! [type_dict][user_dict][repo_dict][tuple_dict][event_count]
//! [type_markers: (position, type_idx)...]
//! [blocks of 3 packed arrays: id_delta, tuple_idx_delta, ts_delta]
//! ```
//!
//! Output is sorted by (event_type, id) for encoding, returned sorted by id.
//!

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::HashMap;
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

// ============================================================================
// Varint encoding
// ============================================================================

fn encode_varint(mut value: u64, buf: &mut Vec<u8>) {
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

// Zigzag encode (signed -> unsigned)
fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(encoded: u64) -> i64 {
    ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64))
}

// ============================================================================
// Block bit-packing: pack N values using minimum bits per field
// ============================================================================

const BLOCK_SIZE: usize = 64;

fn bits_needed(value: u64) -> u8 {
    if value == 0 {
        1 // Need at least 1 bit
    } else {
        64 - value.leading_zeros() as u8
    }
}

/// Encode a block of values with bit-packing.
/// Format: [bit_width: 1 byte][packed values]
fn encode_packed_block(values: &[u64], buf: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }

    let max_bits = values.iter().map(|&v| bits_needed(v)).max().unwrap_or(1);
    buf.push(max_bits);

    // Pack values into bytes
    let mut bit_pos: usize = 0;
    let mut current_byte: u8 = 0;

    for &value in values {
        for bit_idx in 0..max_bits {
            let bit = ((value >> bit_idx) & 1) as u8;
            current_byte |= bit << (bit_pos % 8);
            bit_pos += 1;

            if bit_pos % 8 == 0 {
                buf.push(current_byte);
                current_byte = 0;
            }
        }
    }

    // Flush remaining bits
    if bit_pos % 8 != 0 {
        buf.push(current_byte);
    }
}

/// Decode a block of values with bit-packing.
fn decode_packed_block(bytes: &[u8], pos: &mut usize, count: usize) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }

    let max_bits = bytes[*pos] as usize;
    *pos += 1;

    let mut values = Vec::with_capacity(count);
    let mut bit_pos: usize = 0;

    for _ in 0..count {
        let mut value: u64 = 0;
        for bit_idx in 0..max_bits {
            let byte_idx = *pos + (bit_pos / 8);
            let bit_in_byte = bit_pos % 8;
            let bit = ((bytes[byte_idx] >> bit_in_byte) & 1) as u64;
            value |= bit << bit_idx;
            bit_pos += 1;
        }
        values.push(value);
    }

    *pos += (bit_pos + 7) / 8; // Advance past packed bytes
    values
}

// ============================================================================
// Timestamp utilities
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

// ============================================================================
// Prefix-encoded string dictionary
// ============================================================================

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

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

    fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.strings.len() as u64, buf);

        // First pass: collect prefix/suffix lengths
        let mut prefix_lens: Vec<u64> = Vec::with_capacity(self.strings.len());
        let mut suffix_lens: Vec<u64> = Vec::with_capacity(self.strings.len());
        let mut suffixes: Vec<&str> = Vec::with_capacity(self.strings.len());
        let mut prev = String::new();

        for s in &self.strings {
            let prefix_len = common_prefix_len(s, &prev);
            let suffix = &s[prefix_len..];
            prefix_lens.push(prefix_len as u64);
            suffix_lens.push(suffix.len() as u64);
            suffixes.push(suffix);
            prev = s.clone();
        }

        // Bit-pack the lengths
        encode_packed_block(&prefix_lens, buf);
        encode_packed_block(&suffix_lens, buf);

        // Store all suffixes as UTF-8
        for suffix in suffixes {
            buf.extend_from_slice(suffix.as_bytes());
        }
    }

    fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let count = decode_varint(bytes, pos) as usize;
        if count == 0 {
            return Self {
                strings: Vec::new(),
                str_to_idx: HashMap::new(),
            };
        }

        // Decode bit-packed lengths
        let prefix_lens = decode_packed_block(bytes, pos, count);
        let suffix_lens = decode_packed_block(bytes, pos, count);

        // Reconstruct strings from UTF-8 suffixes
        let mut strings = Vec::with_capacity(count);
        let mut str_to_idx = HashMap::new();
        let mut prev = String::new();

        for i in 0..count {
            let prefix_len = prefix_lens[i] as usize;
            let suffix_len = suffix_lens[i] as usize;
            let suffix = std::str::from_utf8(&bytes[*pos..*pos + suffix_len]).unwrap();
            *pos += suffix_len;
            let s = format!("{}{}", &prev[..prefix_len], suffix);
            str_to_idx.insert(s.clone(), i as u32);
            prev = s.clone();
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

fn split_repo_name(full_name: &str) -> (&str, &str) {
    full_name.split_once('/').unwrap_or((full_name, ""))
}

// ============================================================================
// Repo tuple dictionary: (user_idx, repo_idx, repo_id) -> tuple_idx
// ============================================================================

struct RepoTupleDict {
    // For encoding: (user_idx, repo_idx, repo_id) -> tuple_idx
    tuple_to_idx: HashMap<(u32, u32, u64), u32>,
    // For decoding: tuple_idx -> (user_idx, repo_idx, repo_id)
    tuples: Vec<(u32, u32, u64)>,
}

impl RepoTupleDict {
    fn build(
        events: &[(EventKey, EventValue)],
        user_dict: &StringDict,
        repo_dict: &StringDict,
    ) -> Self {
        let mut unique: Vec<(u32, u32, u64)> = events
            .iter()
            .map(|(_, v)| {
                let (user, repo) = split_repo_name(&v.repo.name);
                (
                    user_dict.get_index(user),
                    repo_dict.get_index(repo),
                    v.repo.id,
                )
            })
            .collect();
        unique.sort();
        unique.dedup();

        let mut tuple_to_idx = HashMap::new();
        for (i, tuple) in unique.iter().enumerate() {
            tuple_to_idx.insert(*tuple, i as u32);
        }

        Self {
            tuple_to_idx,
            tuples: unique,
        }
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.tuples.len() as u64, buf);

        // Collect all components for bit-packing
        let user_idxs: Vec<u64> = self.tuples.iter().map(|(u, _, _)| *u as u64).collect();
        let repo_idxs: Vec<u64> = self.tuples.iter().map(|(_, r, _)| *r as u64).collect();
        let repo_ids: Vec<u64> = self.tuples.iter().map(|(_, _, id)| *id).collect();

        encode_packed_block(&user_idxs, buf);
        encode_packed_block(&repo_idxs, buf);
        encode_packed_block(&repo_ids, buf);
    }

    fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let count = decode_varint(bytes, pos) as usize;
        if count == 0 {
            return Self {
                tuple_to_idx: HashMap::new(),
                tuples: Vec::new(),
            };
        }

        let user_idxs = decode_packed_block(bytes, pos, count);
        let repo_idxs = decode_packed_block(bytes, pos, count);
        let repo_ids = decode_packed_block(bytes, pos, count);

        let mut tuples = Vec::with_capacity(count);
        let mut tuple_to_idx = HashMap::new();

        for i in 0..count {
            let tuple = (user_idxs[i] as u32, repo_idxs[i] as u32, repo_ids[i]);
            tuple_to_idx.insert(tuple, i as u32);
            tuples.push(tuple);
        }

        Self {
            tuple_to_idx,
            tuples,
        }
    }

    fn get_index(&self, user_idx: u32, repo_idx: u32, repo_id: u64) -> u32 {
        self.tuple_to_idx[&(user_idx, repo_idx, repo_id)]
    }

    fn get_tuple(&self, index: u32) -> (u32, u32, u64) {
        self.tuples[index as usize]
    }
}

// ============================================================================
// Type enumeration (maps event types to single-byte indices)
// ============================================================================

struct TypeEnum {
    type_to_idx: HashMap<String, u8>,
    types: Vec<String>,
}

impl TypeEnum {
    fn build(events: &[(EventKey, EventValue)]) -> Self {
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for (key, _) in events {
            *freq.entry(&key.event_type).or_insert(0) += 1;
        }

        let mut types_with_freq: Vec<_> = freq.into_iter().collect();
        types_with_freq.sort_by(|a, b| b.1.cmp(&a.1));

        let mut type_to_idx: HashMap<String, u8> = HashMap::new();
        let mut types: Vec<String> = Vec::new();
        for (i, (t, _)) in types_with_freq.into_iter().enumerate() {
            type_to_idx.insert(t.to_string(), i as u8);
            types.push(t.to_string());
        }

        Self { type_to_idx, types }
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.types.len() as u64, buf);
        for t in &self.types {
            encode_varint(t.len() as u64, buf);
            buf.extend_from_slice(t.as_bytes());
        }
    }

    fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let type_count = decode_varint(bytes, pos) as usize;
        let mut types: Vec<String> = Vec::with_capacity(type_count);
        let mut type_to_idx: HashMap<String, u8> = HashMap::new();

        for i in 0..type_count {
            let len = decode_varint(bytes, pos) as usize;
            let t = std::str::from_utf8(&bytes[*pos..*pos + len])
                .unwrap()
                .to_string();
            *pos += len;
            type_to_idx.insert(t.clone(), i as u8);
            types.push(t);
        }

        Self { type_to_idx, types }
    }

    fn get_index(&self, event_type: &str) -> u8 {
        self.type_to_idx[event_type]
    }

    fn get_type(&self, index: u8) -> &str {
        &self.types[index as usize]
    }
}

// ============================================================================
// The codec implementation
// ============================================================================

pub struct AgavraCodec;

impl AgavraCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for AgavraCodec {
    fn name(&self) -> &str {
        "agavra"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let type_enum = TypeEnum::build(events);
        let user_dict = StringDict::build(events.iter().map(|(_, v)| {
            let (user, _) = split_repo_name(&v.repo.name);
            user.to_string()
        }));
        let repo_dict = StringDict::build(events.iter().map(|(_, v)| {
            let (_, repo) = split_repo_name(&v.repo.name);
            repo.to_string()
        }));
        let tuple_dict = RepoTupleDict::build(events, &user_dict, &repo_dict);

        // Sort by event_type to group, then by id within each group
        let mut sorted: Vec<_> = events.iter().collect();
        sorted.sort_by(|a, b| (&a.0.event_type, &a.0.id).cmp(&(&b.0.event_type, &b.0.id)));

        let mut buf = Vec::new();

        type_enum.encode(&mut buf);
        user_dict.encode(&mut buf);
        repo_dict.encode(&mut buf);
        tuple_dict.encode(&mut buf);
        encode_varint(sorted.len() as u64, &mut buf);

        // Collect all deltas first (for block encoding)
        // Now only 3 fields: id_delta, tuple_idx_delta, ts_delta
        let mut all_deltas: Vec<[i64; 3]> = Vec::with_capacity(sorted.len());
        let mut type_markers: Vec<(usize, u8)> = Vec::new(); // (position, type_idx)

        let mut prev_id: u64 = 0;
        let mut prev_tuple_idx: u32 = 0;
        let mut prev_ts: u64 = 0;
        let mut current_type: Option<&str> = None;

        for (key, value) in &sorted {
            // Record type change positions
            if current_type != Some(&key.event_type) {
                type_markers.push((all_deltas.len(), type_enum.get_index(&key.event_type)));
                current_type = Some(&key.event_type);
            }

            let id: u64 = key.id.parse().unwrap_or(0);
            let delta_id = id as i64 - prev_id as i64;
            prev_id = id;

            let (user, repo) = split_repo_name(&value.repo.name);
            let user_idx = user_dict.get_index(user);
            let repo_idx = repo_dict.get_index(repo);
            let tuple_idx = tuple_dict.get_index(user_idx, repo_idx, value.repo.id);
            let delta_tuple_idx = tuple_idx as i64 - prev_tuple_idx as i64;
            prev_tuple_idx = tuple_idx;

            let ts = parse_timestamp(&value.created_at);
            let delta_ts = ts as i64 - prev_ts as i64;
            prev_ts = ts;

            all_deltas.push([delta_id, delta_tuple_idx, delta_ts]);
        }

        // Encode type markers
        encode_varint(type_markers.len() as u64, &mut buf);
        for (pos, type_idx) in &type_markers {
            encode_varint(*pos as u64, &mut buf);
            buf.push(*type_idx);
        }

        // Encode in blocks
        for chunk in all_deltas.chunks(BLOCK_SIZE) {
            // Transpose: convert from array of structs to struct of arrays
            let mut field_values: [Vec<u64>; 3] = Default::default();
            for deltas in chunk {
                for (i, &delta) in deltas.iter().enumerate() {
                    field_values[i].push(zigzag_encode(delta));
                }
            }

            // Encode each field as a packed block
            for field in &field_values {
                encode_packed_block(field, &mut buf);
            }
        }

        Ok(Bytes::from(buf))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;

        let type_enum = TypeEnum::decode(bytes, &mut pos);
        let user_dict = StringDict::decode(bytes, &mut pos);
        let repo_dict = StringDict::decode(bytes, &mut pos);
        let tuple_dict = RepoTupleDict::decode(bytes, &mut pos);
        let count = decode_varint(bytes, &mut pos) as usize;

        // Decode type markers
        let type_marker_count = decode_varint(bytes, &mut pos) as usize;
        let mut type_markers: Vec<(usize, u8)> = Vec::with_capacity(type_marker_count);
        for _ in 0..type_marker_count {
            let event_pos = decode_varint(bytes, &mut pos) as usize;
            let type_idx = bytes[pos];
            pos += 1;
            type_markers.push((event_pos, type_idx));
        }

        // Decode all blocks (now 3 fields: id_delta, tuple_idx_delta, ts_delta)
        let mut all_deltas: Vec<[i64; 3]> = Vec::with_capacity(count);
        let mut remaining = count;

        while remaining > 0 {
            let block_size = remaining.min(BLOCK_SIZE);

            // Decode each field's packed block
            let mut field_values: [Vec<u64>; 3] = Default::default();
            for field in &mut field_values {
                *field = decode_packed_block(bytes, &mut pos, block_size);
            }

            // Transpose back: struct of arrays to array of structs
            for i in 0..block_size {
                all_deltas.push([
                    zigzag_decode(field_values[0][i]),
                    zigzag_decode(field_values[1][i]),
                    zigzag_decode(field_values[2][i]),
                ]);
            }

            remaining -= block_size;
        }

        // Reconstruct events from deltas
        let mut events = Vec::with_capacity(count);
        let mut prev_id: u64 = 0;
        let mut prev_tuple_idx: u32 = 0;
        let mut prev_ts: u64 = 0;
        let mut type_marker_idx = 0;
        let mut current_type = String::new();

        for (event_idx, deltas) in all_deltas.iter().enumerate() {
            // Check for type change
            if type_marker_idx < type_markers.len() && type_markers[type_marker_idx].0 == event_idx
            {
                current_type = type_enum
                    .get_type(type_markers[type_marker_idx].1)
                    .to_string();
                type_marker_idx += 1;
            }

            let id = (prev_id as i64 + deltas[0]) as u64;
            prev_id = id;

            let tuple_idx = (prev_tuple_idx as i64 + deltas[1]) as u32;
            prev_tuple_idx = tuple_idx;

            let (user_idx, repo_idx, repo_id) = tuple_dict.get_tuple(tuple_idx);
            let user = user_dict.get_string(user_idx);
            let repo = repo_dict.get_string(repo_idx);
            let repo_name = format!("{}/{}", user, repo);
            let repo_url = format!("https://api.github.com/repos/{}", repo_name);

            let ts = (prev_ts as i64 + deltas[2]) as u64;
            prev_ts = ts;
            let created_at = format_timestamp(ts);

            events.push((
                EventKey {
                    event_type: current_type.clone(),
                    id: id.to_string(),
                },
                EventValue {
                    repo: Repo {
                        id: repo_id,
                        name: repo_name,
                        url: repo_url,
                    },
                    created_at,
                },
            ));
        }

        // Sort by id (canonical order)
        events.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(events)
    }
}
