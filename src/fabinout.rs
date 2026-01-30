//! Fabinout Codec - Row-oriented store with global dictionaries and Zstd compression
//!
//! ## Compression Schema
//!
//! **Structure:**
//! ```text
//! [HEADER]
//!   - num_events: varint
//!   - repo_names dictionary: [count: varint][name1: len+bytes][name2: len+bytes]...
//!   - dates dictionary: [count: varint][date1: len+bytes][date2: len+bytes]...
//!
//! [EVENTS] (row-oriented)
//!   For each event:
//!     - id: varint (delta encoded from previous)
//!     - event_type: 1 byte (index into static EVENT_TYPES array)
//!     - repo_id: varint
//!     - repo_name_idx: varint (index into repo_names dictionary)
//!     - date_idx: varint (zigzag delta encoded)
//!
//! [COMPRESSION]
//!   - Entire buffer compressed with Zstd level 22
//! ```
//!
//! **Key optimizations:**
//! - URL field is derived from repo name (not stored)
//! - Event types use static dictionary (15 types -> 1 byte)
//! - Event IDs use delta encoding (sequential IDs -> small deltas)
//! - Date indices use zigzag delta encoding (temporal locality)
//! - Global dictionaries for repo names and dates
//! - Zstd level 22 for maximum compression

use bytes::{BufMut, Bytes, BytesMut};
use std::collections::HashMap;
use std::error::Error;
use std::io::{Read, Write};

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

pub struct FabinoutCodec;

impl FabinoutCodec {
    pub fn new() -> Self {
        Self
    }
}

// All 15 event types, ordered by frequency
const EVENT_TYPES: &[&str] = &[
    "PushEvent",
    "CreateEvent",
    "PullRequestEvent",
    "WatchEvent",
    "IssueCommentEvent",
    "DeleteEvent",
    "PullRequestReviewEvent",
    "IssuesEvent",
    "ReleaseEvent",
    "ForkEvent",
    "PullRequestReviewCommentEvent",
    "PublicEvent",
    "MemberEvent",
    "GollumEvent",
    "CommitCommentEvent",
];

// Encode integer as varint (little-endian, 7 bits per byte, MSB = continuation bit)
fn encode_varint(buf: &mut BytesMut, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.put_u8(byte);
        if value == 0 {
            break;
        }
    }
}

fn decode_varint(data: &[u8], pos: &mut usize) -> u64 {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    result
}

// Encode string with varint length prefix
fn encode_string(buf: &mut BytesMut, s: &str) {
    encode_varint(buf, s.len() as u64);
    buf.put_slice(s.as_bytes());
}

fn decode_string(data: &[u8], pos: &mut usize) -> String {
    let len = decode_varint(data, pos) as usize;
    let s = String::from_utf8(data[*pos..*pos + len].to_vec()).unwrap();
    *pos += len;
    s
}

impl EventCodec for FabinoutCodec {
    fn name(&self) -> &str {
        "fabinout"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let mut buf = BytesMut::new();

        // === PHASE 1: Build dictionaries ===

        // Repo names dictionary (repo_name -> index)
        let mut repo_names: Vec<&str> = Vec::new();
        let mut repo_name_to_idx: HashMap<&str, u32> = HashMap::new();

        // Dates dictionary (date -> index)
        let mut dates: Vec<&str> = Vec::new();
        let mut date_to_idx: HashMap<&str, u32> = HashMap::new();

        for (_key, value) in events {
            if !repo_name_to_idx.contains_key(value.repo.name.as_str()) {
                let idx = repo_names.len() as u32;
                repo_names.push(&value.repo.name);
                repo_name_to_idx.insert(&value.repo.name, idx);
            }

            if !date_to_idx.contains_key(value.created_at.as_str()) {
                let idx = dates.len() as u32;
                dates.push(&value.created_at);
                date_to_idx.insert(&value.created_at, idx);
            }
        }

        // === PHASE 2: Write dictionaries ===

        // Number of events
        encode_varint(&mut buf, events.len() as u64);

        // Repo names dictionary
        encode_varint(&mut buf, repo_names.len() as u64);
        for name in &repo_names {
            encode_string(&mut buf, name);
        }

        // Dates dictionary
        encode_varint(&mut buf, dates.len() as u64);
        for date in &dates {
            encode_string(&mut buf, date);
        }

        // === PHASE 3: Write events ===

        // Map event_type -> index
        let type_to_idx: HashMap<&str, u8> = EVENT_TYPES
            .iter()
            .enumerate()
            .map(|(i, &t)| (t, i as u8))
            .collect();

        let mut last_id: u64 = 0;
        let mut last_date_idx: i64 = 0;

        for (key, value) in events {
            // Event ID (delta encoded)
            let id: u64 = key.id.parse()?;
            let delta = id.wrapping_sub(last_id);
            encode_varint(&mut buf, delta);
            last_id = id;

            // Event type (1 byte index)
            let type_idx = type_to_idx[key.event_type.as_str()];
            buf.put_u8(type_idx);

            // Repo ID
            encode_varint(&mut buf, value.repo.id);

            // Repo name index
            let name_idx = repo_name_to_idx[value.repo.name.as_str()];
            encode_varint(&mut buf, name_idx as u64);

            // Date index (zigzag delta encoded for better compression)
            let date_idx = date_to_idx[value.created_at.as_str()] as i64;
            let date_delta = date_idx - last_date_idx;
            let date_zigzag = ((date_delta << 1) ^ (date_delta >> 63)) as u64;
            encode_varint(&mut buf, date_zigzag);
            last_date_idx = date_idx;
        }

        // Zstd compression (level 22 = maximum compression)
        let raw = buf.freeze();
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 22)?;
        encoder.write_all(&raw)?;
        let compressed = encoder.finish()?;

        Ok(Bytes::from(compressed))
    }

    fn decode(&self, data: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        // Zstd decompression
        let mut decoder = zstd::stream::Decoder::new(data)?;
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        let data = &decompressed;

        let mut pos = 0;

        // === PHASE 1: Read dictionaries ===

        let num_events = decode_varint(data, &mut pos) as usize;

        // Repo names dictionary
        let num_repo_names = decode_varint(data, &mut pos) as usize;
        let mut repo_names: Vec<String> = Vec::with_capacity(num_repo_names);
        for _ in 0..num_repo_names {
            repo_names.push(decode_string(data, &mut pos));
        }

        // Dates dictionary
        let num_dates = decode_varint(data, &mut pos) as usize;
        let mut dates: Vec<String> = Vec::with_capacity(num_dates);
        for _ in 0..num_dates {
            dates.push(decode_string(data, &mut pos));
        }

        // === PHASE 2: Read events ===

        let mut events = Vec::with_capacity(num_events);
        let mut last_id: u64 = 0;
        let mut last_date_idx: i64 = 0;

        for _ in 0..num_events {
            // Event ID (delta decoded)
            let delta = decode_varint(data, &mut pos);
            let id = last_id.wrapping_add(delta);
            last_id = id;

            // Event type
            let type_idx = data[pos] as usize;
            pos += 1;
            let event_type = EVENT_TYPES[type_idx].to_string();

            // Repo ID
            let repo_id = decode_varint(data, &mut pos);

            // Repo name (from dictionary)
            let name_idx = decode_varint(data, &mut pos) as usize;
            let repo_name = repo_names[name_idx].clone();

            // Date (zigzag delta decoded)
            let date_zigzag = decode_varint(data, &mut pos);
            let date_delta = ((date_zigzag >> 1) as i64) ^ -((date_zigzag & 1) as i64);
            let date_idx = (last_date_idx + date_delta) as usize;
            last_date_idx = date_idx as i64;
            let created_at = dates[date_idx].clone();

            // Reconstruct URL from repo name
            let repo_url = format!("https://api.github.com/repos/{}", repo_name);

            let key = EventKey {
                id: id.to_string(),
                event_type,
            };
            let value = EventValue {
                repo: Repo {
                    id: repo_id,
                    name: repo_name,
                    url: repo_url,
                },
                created_at,
            };

            events.push((key, value));
        }

        Ok(events)
    }
}
