//! # Columnar Codec
//!
//! **Strategy:** Store data in columns instead of rows, with per-column encoding.
//!
//! **Result:** ~187 KB with Zstd (-22) (~91.5% smaller than naive)
//!
//! ## How it works:
//!
//! Instead of storing events as rows:
//! ```text
//! [event1: type, id, repo_id, name, url, ts]
//! [event2: type, id, repo_id, name, url, ts]
//! ```
//!
//! Store as columns:
//! ```text
//! [all event_types] [all ids] [all repo_ids] [all names] [all urls] [all timestamps]
//! ```
//!
//! Each column uses the best encoding for its data type:
//!
//! | Column | Encoding | Why |
//! |--------|----------|-----|
//! | event_type | RLE (run-length) | Only 14 unique types, long runs after sorting |
//! | id | Delta | Sequential IDs, small deltas |
//! | repo_id | Delta | Similar repo IDs cluster together |
//! | repo_name | Dictionary | 6k unique strings, reference by index |
//! | repo_url | Dictionary | Same dictionary as name |
//! | created_at | Delta | Timestamps within 1 hour, tiny deltas |
//!
//! ## Why columnar helps:
//!
//! - Similar values are adjacent → better compression ratios
//! - Each column can use optimal encoding for its type
//! - RLE is extremely efficient for sorted categorical data
//! - Zstd can find patterns within each column more easily
//!
//! ## Room for improvement:
//!
//! - repo_url is still stored (could derive from repo_name)
//! - Dictionary isn't sorted by frequency (common strings get same-size indices)
//! - Could combine with more aggressive bit-packing

use bytes::Bytes;
use std::collections::HashMap;
use std::error::Error;

use crate::codec::EventCodec;
use crate::util::{decode_signed_varint, encode_signed_varint, format_timestamp, parse_timestamp};
use crate::varint::{decode_varint, encode_varint};
use crate::{EventKey, EventValue, Repo};

pub struct ColumnarCodec;

struct DictBuilder {
    segments: Vec<String>,
    lookup: HashMap<String, u32>,
}

impl DictBuilder {
    fn new() -> Self {
        Self {
            segments: Vec::new(),
            lookup: HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.lookup.get(s) {
            return idx;
        }
        let idx = self.segments.len() as u32;
        self.segments.push(s.to_string());
        self.lookup.insert(s.to_string(), idx);
        idx
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.segments.len() as u64, buf);
        for seg in &self.segments {
            encode_varint(seg.len() as u64, buf);
            buf.extend_from_slice(seg.as_bytes());
        }
    }
}

fn decode_dict(bytes: &[u8], pos: &mut usize) -> Vec<String> {
    let count = decode_varint(bytes, pos) as usize;
    let mut segments = Vec::with_capacity(count);
    for _ in 0..count {
        let len = decode_varint(bytes, pos) as usize;
        let seg = std::str::from_utf8(&bytes[*pos..*pos + len])
            .unwrap()
            .to_string();
        *pos += len;
        segments.push(seg);
    }
    segments
}

fn encode_delta_column(values: &[u64], buf: &mut Vec<u8>) {
    let mut prev: u64 = 0;
    for &val in values {
        let delta = val as i64 - prev as i64;
        encode_signed_varint(delta, buf);
        prev = val;
    }
}

fn decode_delta_column(bytes: &[u8], pos: &mut usize, count: usize) -> Vec<u64> {
    let mut values = Vec::with_capacity(count);
    let mut prev: u64 = 0;
    for _ in 0..count {
        let delta = decode_signed_varint(bytes, pos);
        let val = (prev as i64 + delta) as u64;
        values.push(val);
        prev = val;
    }
    values
}

fn encode_varint_column(values: &[u32], buf: &mut Vec<u8>) {
    for &val in values {
        encode_varint(val as u64, buf);
    }
}

fn decode_varint_column(bytes: &[u8], pos: &mut usize, count: usize) -> Vec<u32> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(decode_varint(bytes, pos) as u32);
    }
    values
}

fn encode_rle_column(values: &[u32], buf: &mut Vec<u8>) {
    if values.is_empty() {
        encode_varint(0, buf);
        return;
    }

    let mut runs: Vec<(u32, u64)> = Vec::new();
    let mut current_val = values[0];
    let mut current_count: u64 = 1;

    for &val in &values[1..] {
        if val == current_val {
            current_count += 1;
        } else {
            runs.push((current_val, current_count));
            current_val = val;
            current_count = 1;
        }
    }
    runs.push((current_val, current_count));

    encode_varint(runs.len() as u64, buf);

    for (val, count) in runs {
        encode_varint(val as u64, buf);
        encode_varint(count, buf);
    }
}

fn decode_rle_column(bytes: &[u8], pos: &mut usize) -> Vec<u32> {
    let num_runs = decode_varint(bytes, pos) as usize;
    let mut values = Vec::new();

    for _ in 0..num_runs {
        let val = decode_varint(bytes, pos) as u32;
        let count = decode_varint(bytes, pos) as usize;
        values.extend(std::iter::repeat_n(val, count));
    }

    values
}

impl EventCodec for ColumnarCodec {
    fn encode(events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let mut sorted: Vec<_> = events.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let mut type_dict = DictBuilder::new();
        let mut name_dict = DictBuilder::new();

        let mut col_event_type: Vec<u32> = Vec::with_capacity(sorted.len());
        let mut col_id: Vec<u64> = Vec::with_capacity(sorted.len());
        let mut col_repo_id: Vec<u64> = Vec::with_capacity(sorted.len());
        let mut col_repo_name: Vec<u32> = Vec::with_capacity(sorted.len());
        let mut col_repo_url: Vec<u32> = Vec::with_capacity(sorted.len());
        let mut col_created_at: Vec<u64> = Vec::with_capacity(sorted.len());

        for (key, value) in &sorted {
            col_event_type.push(type_dict.intern(&key.event_type));
            col_id.push(key.id.parse().unwrap_or(0));
            col_repo_id.push(value.repo.id);
            col_repo_name.push(name_dict.intern(&value.repo.name));
            col_repo_url.push(name_dict.intern(&value.repo.url));
            col_created_at.push(parse_timestamp(&value.created_at));
        }

        let mut buf = Vec::new();

        encode_varint(sorted.len() as u64, &mut buf);

        type_dict.encode(&mut buf);
        name_dict.encode(&mut buf);

        encode_rle_column(&col_event_type, &mut buf);
        encode_delta_column(&col_id, &mut buf);
        encode_delta_column(&col_repo_id, &mut buf);
        encode_varint_column(&col_repo_name, &mut buf);
        encode_varint_column(&col_repo_url, &mut buf);
        encode_delta_column(&col_created_at, &mut buf);

        Ok(Bytes::from(buf))
    }

    fn decode(bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;

        let count = decode_varint(bytes, &mut pos) as usize;

        let type_dict = decode_dict(bytes, &mut pos);
        let name_dict = decode_dict(bytes, &mut pos);

        let col_event_type = decode_rle_column(bytes, &mut pos);
        let col_id = decode_delta_column(bytes, &mut pos, count);
        let col_repo_id = decode_delta_column(bytes, &mut pos, count);
        let col_repo_name = decode_varint_column(bytes, &mut pos, count);
        let col_repo_url = decode_varint_column(bytes, &mut pos, count);
        let col_created_at = decode_delta_column(bytes, &mut pos, count);

        let mut events = Vec::with_capacity(count);
        for i in 0..count {
            let event_type = type_dict[col_event_type[i] as usize].clone();
            let id = col_id[i].to_string();
            let repo_id = col_repo_id[i];
            let repo_name = name_dict[col_repo_name[i] as usize].clone();
            let repo_url = name_dict[col_repo_url[i] as usize].clone();
            let created_at = format_timestamp(col_created_at[i]);

            events.push((
                EventKey { event_type, id },
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
