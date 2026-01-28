//! # Agavra Codec (Current Best)
//!
//! **Strategy:** Combine delta encoding, prefix compression, and type enumeration.
//!
//! **Result:** ~181.89 KB with Zstd (-22) (~91.8% smaller than naive)
//!
//! ## How it works:
//!
//! This codec combines the best ideas from multiple approaches:
//!
//! ### 1. Type Enumeration
//! Event types are mapped to single-byte indices (0-13 for 14 types).
//! The dictionary is stored once at the start, then each event uses 1 byte.
//!
//! ### 2. Delta Encoding for Numbers
//! Instead of storing absolute values, store the difference from previous:
//! ```text
//! IDs:        2489651045, 2489651051, 2489651053
//! Deltas:     2489651045, +6,         +2          (much smaller!)
//! ```
//! Uses signed varints so negative deltas are also compact.
//!
//! ### 3. Prefix Compression for Strings
//! Repo names and URLs use prefix encoding (see prefix.rs).
//! After sorting, consecutive repos often share prefixes.
//!
//! ### 4. Timestamp Delta Encoding
//! All timestamps are within 1 hour. Delta-encoded, most are 0-3 seconds
//! apart → 1 byte each instead of 24 bytes for ISO 8601 strings.
//!
//! ## Data layout:
//!
//! ```text
//! [type_dict][event_count][event1][event2]...
//!
//! Each event:
//! [type_idx: 1 byte]
//! [id_delta: signed varint]
//! [repo_id_delta: signed varint]
//! [repo_name: prefix-encoded string]
//! [repo_url: prefix-encoded string]
//! [timestamp_delta: signed varint]
//! ```
//!
//! ## Why this wins (for now):
//!
//! - Row-based layout with prefix encoding works well with Zstd
//! - Delta encoding makes numeric sequences highly compressible
//! - Type enumeration eliminates repetitive strings
//! - Sorting maximizes prefix sharing
//!
//! ## Room for improvement:
//!
//! - repo_url is STILL stored (it's derivable from repo_name!)
//! - repo_id might be derivable from repo_name dictionary index
//! - Could use arithmetic coding instead of Zstd
//! - Bit-packing the type index (4 bits) with other small fields

use bytes::Bytes;
use std::error::Error;

use crate::codec::EventCodec;
use crate::type_enum::TypeEnum;
use crate::util::{
    decode_signed_varint, decode_string_prefix, encode_signed_varint, encode_string_prefix,
    format_timestamp, parse_timestamp,
};
use crate::varint::decode_varint;
use crate::{EventKey, EventValue, Repo};

pub struct AgavraCodec;

impl EventCodec for AgavraCodec {
    fn encode(events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let type_enum = TypeEnum::build(events);

        let mut sorted: Vec<_> = events.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let mut buf = Vec::new();

        type_enum.encode(&mut buf);
        crate::varint::encode_varint(sorted.len() as u64, &mut buf);

        let mut prev_id: u64 = 0;
        let mut prev_repo_id: u64 = 0;
        let mut prev_repo_name = String::new();
        let mut prev_repo_url = String::new();
        let mut prev_ts: u64 = 0;

        for (key, value) in &sorted {
            buf.push(type_enum.get_index(&key.event_type));

            let id: u64 = key.id.parse().unwrap_or(0);
            let delta_id = id as i64 - prev_id as i64;
            encode_signed_varint(delta_id, &mut buf);
            prev_id = id;

            let delta_repo_id = value.repo.id as i64 - prev_repo_id as i64;
            encode_signed_varint(delta_repo_id, &mut buf);
            prev_repo_id = value.repo.id;

            encode_string_prefix(&value.repo.name, &prev_repo_name, &mut buf);
            encode_string_prefix(&value.repo.url, &prev_repo_url, &mut buf);
            prev_repo_name = value.repo.name.clone();
            prev_repo_url = value.repo.url.clone();

            let ts = parse_timestamp(&value.created_at);
            let delta_ts = ts as i64 - prev_ts as i64;
            encode_signed_varint(delta_ts, &mut buf);
            prev_ts = ts;
        }

        Ok(Bytes::from(buf))
    }

    fn decode(bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;

        let type_enum = TypeEnum::decode(bytes, &mut pos);
        let count = decode_varint(bytes, &mut pos) as usize;

        let mut events = Vec::with_capacity(count);
        let mut prev_id: u64 = 0;
        let mut prev_repo_id: u64 = 0;
        let mut prev_repo_name = String::new();
        let mut prev_repo_url = String::new();
        let mut prev_ts: u64 = 0;

        for _ in 0..count {
            let type_idx = bytes[pos];
            pos += 1;
            let event_type = type_enum.get_type(type_idx).to_string();

            let delta_id = decode_signed_varint(bytes, &mut pos);
            let id = (prev_id as i64 + delta_id) as u64;
            prev_id = id;

            let delta_repo_id = decode_signed_varint(bytes, &mut pos);
            let repo_id = (prev_repo_id as i64 + delta_repo_id) as u64;
            prev_repo_id = repo_id;

            let repo_name = decode_string_prefix(bytes, &mut pos, &prev_repo_name);
            let repo_url = decode_string_prefix(bytes, &mut pos, &prev_repo_url);
            prev_repo_name = repo_name.clone();
            prev_repo_url = repo_url.clone();

            let delta_ts = decode_signed_varint(bytes, &mut pos);
            let ts = (prev_ts as i64 + delta_ts) as u64;
            prev_ts = ts;
            let created_at = format_timestamp(ts);

            events.push((
                EventKey {
                    event_type,
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

        Ok(events)
    }
}
