//! # Hachikuji Codec
//!
//! **Strategy:** Repo-first grouping with nested event types.
//!
//! ## Key insight:
//! Sorting by (repo, event_type, event_id) instead of (event_type, repo, event_id)
//! reduces repo_info storage from 349k times to 262k times (~500KB savings pre-zstd).
//!
//! ## Data layout:
//!
//! ```text
//! [type_dict][user_dict][repo_dict][repo_count]
//!
//! For each repo:
//!   [repo_id: varint]
//!   [user_idx: varint]
//!   [repo_idx: varint]
//!   [event_type_count: varint]
//!
//!   For each event_type in this repo:
//!     [event_type: 1 byte]
//!     [event_count: varint]
//!
//!     For each event:
//!       [id_delta: signed varint]
//!       [timestamp_delta: signed varint]
//! ```

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::{BTreeMap, HashMap};
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

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
    let encoded = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint(encoded, buf);
}

fn decode_signed_varint(bytes: &[u8], pos: &mut usize) -> i64 {
    let encoded = decode_varint(bytes, pos);
    ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64))
}

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
        let mut prev = String::new();
        for s in &self.strings {
            let prefix_len = common_prefix_len(s, &prev);
            let suffix = &s[prefix_len..];
            encode_varint(prefix_len as u64, buf);
            encode_varint(suffix.len() as u64, buf);
            buf.extend_from_slice(suffix.as_bytes());
            prev = s.clone();
        }
    }

    fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let count = decode_varint(bytes, pos) as usize;
        let mut strings = Vec::with_capacity(count);
        let mut str_to_idx = HashMap::new();
        let mut prev = String::new();

        for i in 0..count {
            let prefix_len = decode_varint(bytes, pos) as usize;
            let suffix_len = decode_varint(bytes, pos) as usize;
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

/// Events grouped by (repo_id, user, repo_name, event_type)
struct RepoGroup {
    repo_id: u64,
    user: String,
    repo_name: String,
    /// event_type -> list of (event_id, timestamp)
    events_by_type: BTreeMap<String, Vec<(u64, u64)>>,
}

/// Key for grouping: (repo_id, user, repo_name) since same repo_id can have multiple names
#[derive(Hash, Eq, PartialEq, Clone, Ord, PartialOrd)]
struct RepoKey {
    repo_id: u64,
    user: String,
    repo_name: String,
}

fn group_events(events: &[(EventKey, EventValue)]) -> Vec<RepoGroup> {
    // Group by (repo_id, user, repo_name) -> (event_type -> events)
    let mut repo_map: HashMap<RepoKey, RepoGroup> = HashMap::new();

    for (key, value) in events {
        let repo_id = value.repo.id;
        let (user, repo_name) = split_repo_name(&value.repo.name);
        let event_id: u64 = key.id.parse().unwrap_or(0);
        let timestamp = parse_timestamp(&value.created_at);

        let repo_key = RepoKey {
            repo_id,
            user: user.to_string(),
            repo_name: repo_name.to_string(),
        };

        let group = repo_map.entry(repo_key).or_insert_with(|| {
            RepoGroup {
                repo_id,
                user: user.to_string(),
                repo_name: repo_name.to_string(),
                events_by_type: BTreeMap::new(),
            }
        });

        group
            .events_by_type
            .entry(key.event_type.clone())
            .or_insert_with(Vec::new)
            .push((event_id, timestamp));
    }

    // Sort events within each type by event_id
    for group in repo_map.values_mut() {
        for events in group.events_by_type.values_mut() {
            events.sort_by_key(|(id, _)| *id);
        }
    }

    // Sort repos by (repo_id, user, repo_name) for deterministic output
    let mut groups: Vec<_> = repo_map.into_values().collect();
    groups.sort_by(|a, b| {
        (a.repo_id, &a.user, &a.repo_name).cmp(&(b.repo_id, &b.user, &b.repo_name))
    });
    groups
}

pub struct HachikujiCodec;

impl HachikujiCodec {
    pub fn new() -> Self {
        Self
    }
}


impl EventCodec for HachikujiCodec {
    fn name(&self) -> &str {
        "Hachikuji"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        // Build dictionaries
        let type_enum = TypeEnum::build(events);
        let user_dict = StringDict::build(events.iter().map(|(_, v)| {
            let (user, _) = split_repo_name(&v.repo.name);
            user.to_string()
        }));
        let repo_dict = StringDict::build(events.iter().map(|(_, v)| {
            let (_, repo) = split_repo_name(&v.repo.name);
            repo.to_string()
        }));

        // Group events by repo
        let groups = group_events(events);

        let mut buf = Vec::new();

        // Write dictionaries
        type_enum.encode(&mut buf);
        user_dict.encode(&mut buf);
        repo_dict.encode(&mut buf);

        // Write repo count
        encode_varint(groups.len() as u64, &mut buf);

        // Write each repo group
        for group in &groups {
            // Repo info
            encode_varint(group.repo_id, &mut buf);
            encode_varint(user_dict.get_index(&group.user) as u64, &mut buf);
            encode_varint(repo_dict.get_index(&group.repo_name) as u64, &mut buf);

            // Event type count for this repo
            encode_varint(group.events_by_type.len() as u64, &mut buf);

            // Each event type
            for (event_type, events) in &group.events_by_type {
                buf.push(type_enum.get_index(event_type));
                encode_varint(events.len() as u64, &mut buf);

                // Delta-encode events within this group
                let mut prev_id: u64 = 0;
                let mut prev_ts: u64 = 0;

                for (event_id, timestamp) in events {
                    let delta_id = *event_id as i64 - prev_id as i64;
                    encode_signed_varint(delta_id, &mut buf);
                    prev_id = *event_id;

                    let delta_ts = *timestamp as i64 - prev_ts as i64;
                    encode_signed_varint(delta_ts, &mut buf);
                    prev_ts = *timestamp;
                }
            }
        }

        let compressed = zstd::encode_all(buf.as_slice(), 22)?;
        Ok(Bytes::from(compressed))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let decompressed = zstd::decode_all(bytes)?;
        let bytes = &decompressed;
        let mut pos = 0;

        // Read dictionaries
        let type_enum = TypeEnum::decode(bytes, &mut pos);
        let user_dict = StringDict::decode(bytes, &mut pos);
        let repo_dict = StringDict::decode(bytes, &mut pos);

        // Read repo count
        let repo_count = decode_varint(bytes, &mut pos) as usize;

        let mut events = Vec::new();

        // Read each repo group
        for _ in 0..repo_count {
            // Repo info
            let repo_id = decode_varint(bytes, &mut pos);
            let user_idx = decode_varint(bytes, &mut pos) as u32;
            let repo_idx = decode_varint(bytes, &mut pos) as u32;

            let user = user_dict.get_string(user_idx);
            let repo_name_part = repo_dict.get_string(repo_idx);
            let repo_name = format!("{}/{}", user, repo_name_part);
            let repo_url = format!("https://api.github.com/repos/{}", repo_name);

            // Event type count
            let type_count = decode_varint(bytes, &mut pos) as usize;

            // Each event type
            for _ in 0..type_count {
                let type_idx = bytes[pos];
                pos += 1;
                let event_type = type_enum.get_type(type_idx).to_string();

                let event_count = decode_varint(bytes, &mut pos) as usize;

                let mut prev_id: u64 = 0;
                let mut prev_ts: u64 = 0;

                for _ in 0..event_count {
                    let delta_id = decode_signed_varint(bytes, &mut pos);
                    let event_id = (prev_id as i64 + delta_id) as u64;
                    prev_id = event_id;

                    let delta_ts = decode_signed_varint(bytes, &mut pos);
                    let timestamp = (prev_ts as i64 + delta_ts) as u64;
                    prev_ts = timestamp;

                    events.push((
                        EventKey {
                            event_type: event_type.clone(),
                            id: event_id.to_string(),
                        },
                        EventValue {
                            repo: Repo {
                                id: repo_id,
                                name: repo_name.clone(),
                                url: repo_url.clone(),
                            },
                            created_at: format_timestamp(timestamp),
                        },
                    ));
                }
            }
        }

        // Sort by (event_type, event_id) to match expected output order
        events.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(events)
    }
}

