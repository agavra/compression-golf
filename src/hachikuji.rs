//! # Hachikuji Codec
//!
//! **Strategy:** Repo dictionary + global event ordering for optimal deltas.
//!
//! ## Key insights:
//! 1. Store repo info in a dictionary (once per unique repo)
//! 2. Sort events by (event_type, id) globally for small deltas
//! 3. Reference repos by index, not inline

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::HashMap;
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

        Self { strings: unique, str_to_idx }
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

        Self { strings, str_to_idx }
    }

    fn get_index(&self, s: &str) -> u32 {
        self.str_to_idx[s]
    }

    fn get_string(&self, index: u32) -> &str {
        &self.strings[index as usize]
    }
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
            let t = std::str::from_utf8(&bytes[*pos..*pos + len]).unwrap().to_string();
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

/// Unique repo tuple: (repo_id, name_idx) where name_idx indexes the name dictionary
struct RepoDict {
    tuples: Vec<(u64, u32)>,
    tuple_to_idx: HashMap<(u64, u32), u32>,
}

impl RepoDict {
    fn build(events: &[(EventKey, EventValue)], name_dict: &StringDict) -> Self {
        let mut unique: Vec<(u64, u32)> = events
            .iter()
            .map(|(_, v)| (v.repo.id, name_dict.get_index(&v.repo.name)))
            .collect();
        unique.sort();
        unique.dedup();

        let mut tuple_to_idx = HashMap::new();
        for (i, tuple) in unique.iter().enumerate() {
            tuple_to_idx.insert(*tuple, i as u32);
        }

        Self { tuples: unique, tuple_to_idx }
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.tuples.len() as u64, buf);
        let mut prev_repo_id: u64 = 0;
        for (repo_id, name_idx) in &self.tuples {
            encode_signed_varint(*repo_id as i64 - prev_repo_id as i64, buf);
            prev_repo_id = *repo_id;
            encode_varint(*name_idx as u64, buf);
        }
    }

    fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let count = decode_varint(bytes, pos) as usize;
        let mut tuples = Vec::with_capacity(count);
        let mut tuple_to_idx = HashMap::new();
        let mut prev_repo_id: u64 = 0;

        for i in 0..count {
            let repo_id = (prev_repo_id as i64 + decode_signed_varint(bytes, pos)) as u64;
            prev_repo_id = repo_id;
            let name_idx = decode_varint(bytes, pos) as u32;
            let tuple = (repo_id, name_idx);
            tuple_to_idx.insert(tuple, i as u32);
            tuples.push(tuple);
        }

        Self { tuples, tuple_to_idx }
    }

    fn get_index(&self, repo_id: u64, name_idx: u32) -> u32 {
        self.tuple_to_idx[&(repo_id, name_idx)]
    }

    fn get_tuple(&self, index: u32) -> (u64, u32) {
        self.tuples[index as usize]
    }
}

/// Simple arithmetic coder using 64-bit integers
struct ArithmeticEncoder {
    low: u64,
    high: u64,
    pending_bits: u32,
    output: Vec<u8>,
    current_byte: u8,
    bit_count: u8,
}

impl ArithmeticEncoder {
    fn new() -> Self {
        Self {
            low: 0,
            high: 0xFFFF_FFFF,
            pending_bits: 0,
            output: Vec::new(),
            current_byte: 0,
            bit_count: 0,
        }
    }

    fn write_bit(&mut self, bit: bool) {
        self.current_byte = (self.current_byte << 1) | (bit as u8);
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.output.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    fn write_bit_plus_pending(&mut self, bit: bool) {
        self.write_bit(bit);
        while self.pending_bits > 0 {
            self.write_bit(!bit);
            self.pending_bits -= 1;
        }
    }

    fn encode_symbol(&mut self, cum_freq: u64, freq: u64, total: u64) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range * (cum_freq + freq) / total) - 1;
        self.low = self.low + (range * cum_freq / total);

        loop {
            if self.high < 0x8000_0000 {
                self.write_bit_plus_pending(false);
            } else if self.low >= 0x8000_0000 {
                self.write_bit_plus_pending(true);
                self.low -= 0x8000_0000;
                self.high -= 0x8000_0000;
            } else if self.low >= 0x4000_0000 && self.high < 0xC000_0000 {
                self.pending_bits += 1;
                self.low -= 0x4000_0000;
                self.high -= 0x4000_0000;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        self.pending_bits += 1;
        if self.low < 0x4000_0000 {
            self.write_bit_plus_pending(false);
        } else {
            self.write_bit_plus_pending(true);
        }
        // Flush remaining bits
        if self.bit_count > 0 {
            self.output.push(self.current_byte << (8 - self.bit_count));
        }
        self.output
    }
}

struct ArithmeticDecoder<'a> {
    low: u64,
    high: u64,
    value: u64,
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> ArithmeticDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut decoder = Self {
            low: 0,
            high: 0xFFFF_FFFF,
            value: 0,
            data,
            byte_pos: 0,
            bit_pos: 0,
        };
        // Read initial bits
        for _ in 0..32 {
            decoder.value = (decoder.value << 1) | (decoder.read_bit() as u64);
        }
        decoder
    }

    fn read_bit(&mut self) -> bool {
        if self.byte_pos >= self.data.len() {
            return false;
        }
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        bit != 0
    }

    fn decode_symbol(&mut self, cum_freqs: &[u64], total: u64) -> usize {
        let range = self.high - self.low + 1;
        let scaled = ((self.value - self.low + 1) * total - 1) / range;

        // Binary search for symbol
        let mut lo = 0;
        let mut hi = cum_freqs.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cum_freqs[mid + 1] <= scaled {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let symbol = lo;

        // Update state
        self.high = self.low + (range * cum_freqs[symbol + 1] / total) - 1;
        self.low = self.low + (range * cum_freqs[symbol] / total);

        loop {
            if self.high < 0x8000_0000 {
                // Nothing
            } else if self.low >= 0x8000_0000 {
                self.value -= 0x8000_0000;
                self.low -= 0x8000_0000;
                self.high -= 0x8000_0000;
            } else if self.low >= 0x4000_0000 && self.high < 0xC000_0000 {
                self.value -= 0x4000_0000;
                self.low -= 0x4000_0000;
                self.high -= 0x4000_0000;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.value = (self.value << 1) | (self.read_bit() as u64);
        }

        symbol
    }
}

fn encode_arithmetic(values: &[u64], num_symbols: usize, buf: &mut Vec<u8>) {
    // Count frequencies
    let mut freqs = vec![0u64; num_symbols];
    for &v in values {
        freqs[v as usize] += 1;
    }

    // Ensure no zero frequencies (add-one smoothing)
    for f in &mut freqs {
        if *f == 0 {
            *f = 1;
        }
    }

    // Build cumulative frequencies
    let mut cum_freqs = vec![0u64; num_symbols + 1];
    for i in 0..num_symbols {
        cum_freqs[i + 1] = cum_freqs[i] + freqs[i];
    }
    let total = cum_freqs[num_symbols];

    // Write frequency table (compressed)
    encode_varint(num_symbols as u64, buf);
    for &f in &freqs {
        encode_varint(f, buf);
    }

    // Encode values
    let mut encoder = ArithmeticEncoder::new();
    for &v in values {
        let sym = v as usize;
        encoder.encode_symbol(cum_freqs[sym], freqs[sym], total);
    }
    let encoded = encoder.finish();

    encode_varint(encoded.len() as u64, buf);
    buf.extend_from_slice(&encoded);
}

fn decode_arithmetic(bytes: &[u8], pos: &mut usize, count: usize) -> Vec<u64> {
    let num_symbols = decode_varint(bytes, pos) as usize;

    // Read frequency table
    let mut freqs = Vec::with_capacity(num_symbols);
    for _ in 0..num_symbols {
        freqs.push(decode_varint(bytes, pos));
    }

    // Build cumulative frequencies
    let mut cum_freqs = vec![0u64; num_symbols + 1];
    for i in 0..num_symbols {
        cum_freqs[i + 1] = cum_freqs[i] + freqs[i];
    }
    let total = cum_freqs[num_symbols];

    // Read encoded data
    let encoded_len = decode_varint(bytes, pos) as usize;
    let encoded_data = &bytes[*pos..*pos + encoded_len];
    *pos += encoded_len;

    // Decode
    let mut decoder = ArithmeticDecoder::new(encoded_data);
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(decoder.decode_symbol(&cum_freqs, total) as u64);
    }

    values
}

fn encode_signed_varints(values: &[i64], buf: &mut Vec<u8>) {
    for &v in values {
        encode_signed_varint(v, buf);
    }
}

fn decode_signed_varints(bytes: &[u8], pos: &mut usize, count: usize) -> Vec<i64> {
    (0..count).map(|_| decode_signed_varint(bytes, pos)).collect()
}

/// Encode timestamp deltas using 2-bit categories + exceptions
/// Category: 0=zero, 1=one, 2=small (2-127), 3=large
fn encode_ts_deltas(values: &[i64], buf: &mut Vec<u8>) {
    let mut categories: Vec<u8> = Vec::with_capacity((values.len() + 3) / 4);
    let mut exceptions: Vec<i64> = Vec::new();

    for chunk in values.chunks(4) {
        let mut byte: u8 = 0;
        for (i, &val) in chunk.iter().enumerate() {
            let cat = if val == 0 {
                0
            } else if val == 1 {
                1
            } else if val >= 2 && val <= 127 {
                exceptions.push(val);
                2
            } else {
                exceptions.push(val);
                3
            };
            byte |= cat << (i * 2);
        }
        categories.push(byte);
    }

    encode_varint(values.len() as u64, buf);
    buf.extend_from_slice(&categories);
    encode_varint(exceptions.len() as u64, buf);
    for &e in &exceptions {
        encode_signed_varint(e, buf);
    }
}

fn decode_ts_deltas(bytes: &[u8], pos: &mut usize) -> Vec<i64> {
    let count = decode_varint(bytes, pos) as usize;
    let cat_bytes = (count + 3) / 4;
    let categories = &bytes[*pos..*pos + cat_bytes];
    *pos += cat_bytes;

    let exception_count = decode_varint(bytes, pos) as usize;
    let mut exceptions: Vec<i64> = Vec::with_capacity(exception_count);
    for _ in 0..exception_count {
        exceptions.push(decode_signed_varint(bytes, pos));
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
            2 | 3 => {
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
        let type_enum = TypeEnum::build(events);
        let name_dict = StringDict::build(events.iter().map(|(_, v)| v.repo.name.clone()));
        let repo_dict = RepoDict::build(events, &name_dict);

        // Sort by (event_type, id) for optimal deltas
        let mut sorted: Vec<_> = events.iter().collect();
        sorted.sort_by(|a, b| (&a.0.event_type, &a.0.id).cmp(&(&b.0.event_type, &b.0.id)));

        // Collect columnar data
        let mut type_markers: Vec<(usize, u8)> = Vec::new();
        let mut repo_idxs: Vec<u64> = Vec::with_capacity(sorted.len());
        let mut id_deltas: Vec<i64> = Vec::with_capacity(sorted.len());
        let mut ts_deltas: Vec<i64> = Vec::with_capacity(sorted.len());

        let mut prev_id: u64 = 0;
        let mut prev_ts: u64 = 0;
        let mut current_type: Option<&str> = None;

        for (key, value) in &sorted {
            if current_type != Some(&key.event_type) {
                type_markers.push((repo_idxs.len(), type_enum.get_index(&key.event_type)));
                current_type = Some(&key.event_type);
            }

            let name_idx = name_dict.get_index(&value.repo.name);
            let repo_idx = repo_dict.get_index(value.repo.id, name_idx);
            repo_idxs.push(repo_idx as u64);

            let id: u64 = key.id.parse().unwrap_or(0);
            id_deltas.push(id as i64 - prev_id as i64);
            prev_id = id;

            let ts = parse_timestamp(&value.created_at);
            ts_deltas.push(ts as i64 - prev_ts as i64);
            prev_ts = ts;
        }

        let mut buf = Vec::new();

        // Write dictionaries
        type_enum.encode(&mut buf);
        name_dict.encode(&mut buf);
        repo_dict.encode(&mut buf);

        // Write counts
        encode_varint(sorted.len() as u64, &mut buf);

        // Write type markers
        encode_varint(type_markers.len() as u64, &mut buf);
        for (pos, type_idx) in &type_markers {
            encode_varint(*pos as u64, &mut buf);
            buf.push(*type_idx);
        }

        // Write columnar event data
        let num_repos = repo_dict.tuples.len();
        encode_arithmetic(&repo_idxs, num_repos, &mut buf);
        encode_signed_varints(&id_deltas, &mut buf);
        encode_ts_deltas(&ts_deltas, &mut buf);

        let compressed = zstd::encode_all(buf.as_slice(), 22)?;
        Ok(Bytes::from(compressed))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let decompressed = zstd::decode_all(bytes)?;
        let bytes = &decompressed;
        let mut pos = 0;

        let type_enum = TypeEnum::decode(bytes, &mut pos);
        let name_dict = StringDict::decode(bytes, &mut pos);
        let repo_dict = RepoDict::decode(bytes, &mut pos);

        let event_count = decode_varint(bytes, &mut pos) as usize;

        // Read type markers
        let type_marker_count = decode_varint(bytes, &mut pos) as usize;
        let mut type_markers: Vec<(usize, u8)> = Vec::with_capacity(type_marker_count);
        for _ in 0..type_marker_count {
            let event_pos = decode_varint(bytes, &mut pos) as usize;
            let type_idx = bytes[pos];
            pos += 1;
            type_markers.push((event_pos, type_idx));
        }

        // Read columnar event data
        let repo_idxs = decode_arithmetic(bytes, &mut pos, event_count);
        let id_deltas = decode_signed_varints(bytes, &mut pos, event_count);
        let ts_deltas = decode_ts_deltas(bytes, &mut pos);

        // Reconstruct events
        let mut events = Vec::with_capacity(event_count);
        let mut prev_id: u64 = 0;
        let mut prev_ts: u64 = 0;
        let mut type_marker_idx = 0;
        let mut current_type = String::new();

        for i in 0..event_count {
            if type_marker_idx < type_markers.len() && type_markers[type_marker_idx].0 == i {
                current_type = type_enum.get_type(type_markers[type_marker_idx].1).to_string();
                type_marker_idx += 1;
            }

            let (repo_id, name_idx) = repo_dict.get_tuple(repo_idxs[i] as u32);
            let repo_name = name_dict.get_string(name_idx).to_string();
            let repo_url = format!("https://api.github.com/repos/{}", repo_name);

            let id = (prev_id as i64 + id_deltas[i]) as u64;
            prev_id = id;

            let ts = (prev_ts as i64 + ts_deltas[i]) as u64;
            prev_ts = ts;

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
                    created_at: format_timestamp(ts),
                },
            ));
        }

        events.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(events)
    }
}
