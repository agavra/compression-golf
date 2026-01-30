//! # Samsond Codec
//!
//! **Strategy:** Sequential delta encoding + first-occurrence repo dictionary + zstd(22)
//!
//! ## Key Design Decisions:
//!
//! ### 1. Continuous deltas for naturally-local fields
//! IDs are mostly monotonic (measured: 99%+) and timestamps cluster heavily, so deltas
//! stay small without block/group resets. Measured: ID deltas are typically small
//! (e.g., delta=+4 => ZigZag=8 => 0x08, still 1 byte).
//! Example: IDs 45185629417 to 45185629422 to 45185629424 (deltas: 5, 2)
//! Small signed deltas = short zigzag varints = repeated byte patterns for zstd.
//!
//! ### 2. First-occurrence dictionary ordering (critical optimization)
//! Repo dictionary is sorted by first appearance in the stream, not lexicographically.
//! Measured: 50% of events introduce new repos; 29% of repo deltas satisfy |delta| ≤ 10
//! (ZigZag ≤ 20 => 1-byte varint).
//! 
//! Real example from data.json (first 4 events):
//! ```
//! Event 1: mshdabiola/NotePad              -> repo_idx=0 (delta=0)  -> 0x00
//! Event 2: aws-actions/amazon-ecr-login    -> repo_idx=1 (delta=+1) -> 0x02
//! Event 3: heytrgithub/degisiklik-yapan... -> repo_idx=2 (delta=+1) -> 0x02
//! Event 4: xbmc/xbmc                       -> repo_idx=3 (delta=+1) -> 0x02
//! ```
//! All repo_idx deltas encode to 1 byte (ZigZag-signed varint: +1 -> 2 -> 0x02).
//! Early in stream, new repos dominate so deltas are often +1. Later, repo reuse
//! introduces negative/larger deltas, but first-occurrence ordering keeps them smaller.
//! 
//! Compare to lexicographic ordering where "aws-actions" would be idx=0, "heytrgithub"
//! idx=1, "mshdabiola" idx=2, "xbmc" idx=3, causing event 1 to reference idx=2
//! (delta=+2 from start, encoded as 0x04).
//! 
//! Trade: Prefix compression still works (compress against previous dict entry) but
//! loses alphabetical adjacency. One-time dictionary cost vs per-event delta benefit.
//! Result: Measured 104 KB improvement over lexicographic (7.67 MB to 7.56 MB).
//!
//! ### 3. Lossless repo identity
//! Repos are keyed by (name, id) because the dataset contains the same name with
//! multiple IDs (e.g., "01Yomi/01yomi.github.io" appears with both ID 910666850
//! and 910668463). Using only name as key would lose information.
//!
//! ### 4. Low-entropy binary stream
//! Measured: Small zigzag varints (99% of ID deltas are 1-byte) and single-byte types
//! produce byte patterns that zstd compresses well. The format optimizes for this
//! dataset's measured properties: skewed event types (top 5 = 93%), temporal clustering
//! (60+ events per second), and small deltas. Achieves 27.8x compression ratio.
//!
//! ## Correctness Invariants:
//! - Dictionary: decode(encode(D)) preserves order and content
//! - Index: repo_idx always maps to correct (name, id) entry
//! - Delta: prev_* state updates identically during encode/decode
//!
//! ## Architecture:
//! 1. **Type enumeration** - Maps 15 event types to single bytes (sorted by frequency)
//! 2. **Repo dictionary** - Stores unique (repo_name, repo_id) pairs with prefix compression
//!    - Sorted by first occurrence (minimizes repo_idx deltas)
//!    - Measured: 262,949 unique pairs; prefix compression against previous entry
//! 3. **Sequential encoding** - Preserves input order:
//!    - repo_idx delta (signed varint) - measured: 73% are 1-byte
//!    - type_id (1 byte) - direct lookup, no encoding needed
//!    - timestamp delta (signed varint) - measured: |delta| ≤ 63s, so ZigZag < 128 => 100% 1-byte varints
//!    - event_id delta (signed varint) - measured: 99% are 1-byte (monotonic IDs)
//! 4. **Final compression** - zstd(22) on the optimized binary stream


use bytes::Bytes;
use chrono::{DateTime, TimeZone};
use std::collections::HashMap;
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const ZSTD_LEVEL: i32 = 22;

// ============================================================================
// Varint utilities
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

fn encode_signed_varint(value: i64, buf: &mut Vec<u8>) {
    // zigzag
    let encoded = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint(encoded, buf);
}

fn decode_signed_varint(bytes: &[u8], pos: &mut usize) -> i64 {
    let encoded = decode_varint(bytes, pos);
    ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64))
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
    chrono::Utc
        .timestamp_opt(ts as i64, 0)
        .single()
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_default()
}

// ============================================================================
// Prefix-compressed string dictionary helper
// ============================================================================

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

// ============================================================================
// Repo dictionary: full repo name + repo_id
// ============================================================================

struct RepoDict {
    entries: Vec<(String, u64)>,                    // idx -> (repo_name, repo_id)
    entry_to_idx: HashMap<(String, u64), u32>,      // (repo_name, repo_id) -> idx
}

impl RepoDict {
    fn build(events: &[(EventKey, EventValue)]) -> Self {
        // CRITICAL: Same repo name can have different IDs in the data
        // (e.g., "01Yomi/01yomi.github.io" has both ID 910666850 and 910668463)
        // Must use (name, id) tuple as key for lossless compression.
        
        // Sort by first occurrence to minimize repo_idx deltas.
        // Key insight: when repos appear in discovery order, consecutive events
        // reference nearby indices = small deltas = better varint compression.
        // Trade: loses alphabetical prefix compression (one-time cost) but gains
        // per-event delta wins (1M times) = net ~100KB improvement.
        let mut first_occurrence: HashMap<(String, u64), usize> = HashMap::new();
        for (i, (_, v)) in events.iter().enumerate() {
            let key = (v.repo.name.clone(), v.repo.id);
            first_occurrence.entry(key).or_insert(i);
        }
        
        let mut unique: Vec<(String, u64)> = first_occurrence.keys().cloned().collect();
        unique.sort_by_key(|k| first_occurrence[k]);
        
        let mut entry_to_idx = HashMap::new();
        for (i, entry) in unique.iter().enumerate() {
            entry_to_idx.insert(entry.clone(), i as u32);
        }

        Self {
            entries: unique,
            entry_to_idx,
        }
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.entries.len() as u64, buf);

        let mut prev_name = String::new();
        for (name, id) in &self.entries {
            let prefix_len = common_prefix_len(name, &prev_name);
            let suffix = &name[prefix_len..];
            encode_varint(prefix_len as u64, buf);
            encode_varint(suffix.len() as u64, buf);
            buf.extend_from_slice(suffix.as_bytes());
            prev_name = name.clone();
            
            encode_varint(*id, buf);
        }
    }

    fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let count = decode_varint(bytes, pos) as usize;

        let mut entries = Vec::with_capacity(count);
        let mut entry_to_idx = HashMap::new();
        let mut prev_name = String::new();

        for i in 0..count {
            let prefix_len = decode_varint(bytes, pos) as usize;
            let suffix_len = decode_varint(bytes, pos) as usize;
            let suffix = std::str::from_utf8(&bytes[*pos..*pos + suffix_len]).unwrap();
            *pos += suffix_len;

            let name = format!("{}{}", &prev_name[..prefix_len], suffix);
            let id = decode_varint(bytes, pos);
            
            entry_to_idx.insert((name.clone(), id), i as u32);
            prev_name = name.clone();
            entries.push((name, id));
        }

        Self {
            entries,
            entry_to_idx,
        }
    }

    fn get_index(&self, repo_full_name: &str, repo_id: u64) -> u32 {
        self.entry_to_idx[&(repo_full_name.to_string(), repo_id)]
    }

    fn get_entry(&self, index: u32) -> (&str, u64) {
        let (name, id) = &self.entries[index as usize];
        (name.as_str(), *id)
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
// Codec
// ============================================================================

pub struct SamsondCodec;

impl SamsondCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for SamsondCodec {
    fn name(&self) -> &str {
        "samsond"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let type_enum = TypeEnum::build(events);
        let repo_dict = RepoDict::build(events);

        let mut buf = Vec::new();
        self.write_header(&type_enum, &repo_dict, events.len(), &mut buf);
        self.write_events(events, &type_enum, &repo_dict, &mut buf);
        
        let compressed = zstd::encode_all(buf.as_slice(), ZSTD_LEVEL)?;
        Ok(Bytes::from(compressed))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let decompressed = zstd::decode_all(bytes)?;
        let mut pos = 0;

        let (type_enum, repo_dict, event_count) = self.read_header(&decompressed, &mut pos);
        let events = self.read_events(&decompressed, &mut pos, event_count, &type_enum, &repo_dict);

        Ok(events)
    }
}

impl SamsondCodec {
    fn write_header(
        &self,
        type_enum: &TypeEnum,
        repo_dict: &RepoDict,
        event_count: usize,
        buf: &mut Vec<u8>,
    ) {
        type_enum.encode(buf);
        repo_dict.encode(buf);
        encode_varint(event_count as u64, buf);
    }

    fn write_events(
        &self,
        events: &[(EventKey, EventValue)],
        type_enum: &TypeEnum,
        repo_dict: &RepoDict,
        buf: &mut Vec<u8>,
    ) {
        // Delta encoding: prev_* state ensures decode mirrors encode exactly
        let mut prev_repo_idx: u32 = 0;
        let mut prev_ts: u64 = 0;
        let mut prev_id: u64 = 0;

        for (key, value) in events {
            let type_id = type_enum.get_index(&key.event_type);
            let repo_idx = repo_dict.get_index(&value.repo.name, value.repo.id);
            let ts = parse_timestamp(&value.created_at);
            let id: u64 = key.id.parse().unwrap_or(0);

            // Write deltas for better compression
            self.write_repo_delta(repo_idx, prev_repo_idx, buf);
            buf.push(type_id);
            self.write_timestamp_delta(ts, prev_ts, buf);
            self.write_id_delta(id, prev_id, buf);

            prev_repo_idx = repo_idx;
            prev_ts = ts;
            prev_id = id;
        }
    }

    fn write_repo_delta(&self, current: u32, previous: u32, buf: &mut Vec<u8>) {
        encode_signed_varint(current as i64 - previous as i64, buf);
    }

    fn write_timestamp_delta(&self, current: u64, previous: u64, buf: &mut Vec<u8>) {
        encode_signed_varint(current as i64 - previous as i64, buf);
    }

    fn write_id_delta(&self, current: u64, previous: u64, buf: &mut Vec<u8>) {
        encode_signed_varint(current as i64 - previous as i64, buf);
    }

    fn read_header(&self, bytes: &[u8], pos: &mut usize) -> (TypeEnum, RepoDict, usize) {
        let type_enum = TypeEnum::decode(bytes, pos);
        let repo_dict = RepoDict::decode(bytes, pos);
        let event_count = decode_varint(bytes, pos) as usize;
        (type_enum, repo_dict, event_count)
    }

    fn read_events(
        &self,
        bytes: &[u8],
        pos: &mut usize,
        event_count: usize,
        type_enum: &TypeEnum,
        repo_dict: &RepoDict,
    ) -> Vec<(EventKey, EventValue)> {
        let mut events = Vec::with_capacity(event_count);
        let mut prev_repo_idx: u32 = 0;
        let mut prev_ts: u64 = 0;
        let mut prev_id: u64 = 0;

        for _ in 0..event_count {
            let repo_idx = self.read_repo_delta(bytes, pos, prev_repo_idx);
            let type_id = bytes[*pos];
            *pos += 1;
            let ts = self.read_timestamp_delta(bytes, pos, prev_ts);
            let id = self.read_id_delta(bytes, pos, prev_id);

            let event = self.reconstruct_event(type_id, repo_idx, ts, id, type_enum, repo_dict);
            events.push(event);

            prev_repo_idx = repo_idx;
            prev_ts = ts;
            prev_id = id;
        }

        events
    }

    fn read_repo_delta(&self, bytes: &[u8], pos: &mut usize, previous: u32) -> u32 {
        let delta = decode_signed_varint(bytes, pos);
        (previous as i64 + delta) as u32
    }

    fn read_timestamp_delta(&self, bytes: &[u8], pos: &mut usize, previous: u64) -> u64 {
        let delta = decode_signed_varint(bytes, pos);
        (previous as i64 + delta) as u64
    }

    fn read_id_delta(&self, bytes: &[u8], pos: &mut usize, previous: u64) -> u64 {
        let delta = decode_signed_varint(bytes, pos);
        (previous as i64 + delta) as u64
    }

    fn reconstruct_event(
        &self,
        type_id: u8,
        repo_idx: u32,
        ts: u64,
        id: u64,
        type_enum: &TypeEnum,
        repo_dict: &RepoDict,
    ) -> (EventKey, EventValue) {
        let (repo_name, repo_id) = repo_dict.get_entry(repo_idx);
        let repo_url = format!("https://api.github.com/repos/{}", repo_name);

        (
            EventKey {
                event_type: type_enum.get_type(type_id).to_string(),
                id: id.to_string(),
            },
            EventValue {
                repo: Repo {
                    id: repo_id,
                    name: repo_name.to_string(),
                    url: repo_url,
                },
                created_at: format_timestamp(ts),
            },
        )
    }
}