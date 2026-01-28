use chrono::{DateTime, TimeZone, Utc};

use crate::varint::{decode_varint, encode_varint};

pub fn parse_timestamp(ts: &str) -> u64 {
    DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.timestamp() as u64)
        .unwrap_or(0)
}

pub fn format_timestamp(ts: u64) -> String {
    Utc.timestamp_opt(ts as i64, 0)
        .single()
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_default()
}

pub fn common_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

pub fn encode_string_prefix(s: &str, prev: &str, buf: &mut Vec<u8>) {
    let prefix_len = common_prefix_len(s, prev);
    let suffix = &s[prefix_len..];
    encode_varint(prefix_len as u64, buf);
    encode_varint(suffix.len() as u64, buf);
    buf.extend_from_slice(suffix.as_bytes());
}

pub fn decode_string_prefix(bytes: &[u8], pos: &mut usize, prev: &str) -> String {
    let prefix_len = decode_varint(bytes, pos) as usize;
    let suffix_len = decode_varint(bytes, pos) as usize;
    let suffix = std::str::from_utf8(&bytes[*pos..*pos + suffix_len]).unwrap();
    *pos += suffix_len;
    format!("{}{}", &prev[..prefix_len], suffix)
}

pub fn encode_signed_varint(value: i64, buf: &mut Vec<u8>) {
    let encoded = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint(encoded, buf);
}

pub fn decode_signed_varint(bytes: &[u8], pos: &mut usize) -> i64 {
    let encoded = decode_varint(bytes, pos);
    ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64))
}
