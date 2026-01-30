//! Hand-rolled columnar codec with TSV mapping table for repo info.
//!
//! Data ordering:
//!   Events are sorted by event_id (ascending, parsed as u64) before storage.
//!   This ordering is critical for compression because:
//!   - event_id is sequential, so deltas between adjacent IDs are small (max 251)
//!   - created_at timestamps are correlated with event_id, so deltas stay small
//!   - Small deltas compress extremely well with zstd
//!
//!   On decode, events are re-sorted by EventKey to match expected output order.
//!
//! Strategy:
//! - Dictionary encode event_type (14 unique values)
//! - Store unique (repo_id, repo_name) pairs in a TSV mapping table (compressed once)
//! - Per event, store only an index into the mapping table
//! - Store first event_id in header, deltas as u8 (max delta is 251)
//! - Store first timestamp in header, deltas as i16 (range -2540 to +2540)
//! - Reconstruct repo.url from repo.name during decode
//! - Use zstd level 22 compression
//!
//! Binary format:
//!   Header (28 bytes, uncompressed):
//!     - first_event_id: u64
//!     - first_timestamp: i64
//!     - mapping_size: u32
//!     - num_events: u32
//!     - event_type_dict_size: u16
//!     - num_event_types: u8
//!     - _padding: u8
//!
//!   Compressed payload (zstd-22):
//!     - mapping TSV (repo_id\trepo_name\n)
//!     - event_type dict (newline-separated strings)
//!     - event_type indices: [u8; num_events]
//!     - event_id deltas: [u8; num_events]
//!     - repo_pair indices: [u32-le; num_events]
//!     - created_at deltas: [i16-le; num_events]
//!
//! Set NATE_DEBUG=1 to see column size statistics.

use bytes::Bytes;
use chrono::DateTime;
use std::collections::HashMap;
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const ZSTD_LEVEL: i32 = 22;

pub struct NatebrennandCodec;

impl NatebrennandCodec {
    pub fn new() -> Self {
        Self
    }
}

fn debug_enabled() -> bool {
    std::env::var("NATE_DEBUG").is_ok()
}

/// Print value statistics for a column: min, max, and % fitting in each bit width
fn print_value_stats_unsigned(name: &str, values: &[u64]) {
    if values.is_empty() {
        return;
    }
    let min = *values.iter().min().unwrap();
    let max = *values.iter().max().unwrap();
    let count = values.len() as f64;

    let fits_8 = values.iter().filter(|&&v| v <= u8::MAX as u64).count() as f64 / count * 100.0;
    let fits_16 = values.iter().filter(|&&v| v <= u16::MAX as u64).count() as f64 / count * 100.0;
    let fits_32 = values.iter().filter(|&&v| v <= u32::MAX as u64).count() as f64 / count * 100.0;

    eprintln!(
        "  {:<18} min={:<12} max={:<12} u8={:>5.1}% u16={:>5.1}% u32={:>5.1}%",
        name, min, max, fits_8, fits_16, fits_32
    );
}

fn print_value_stats_signed(name: &str, values: &[i64]) {
    if values.is_empty() {
        return;
    }
    let min = *values.iter().min().unwrap();
    let max = *values.iter().max().unwrap();
    let count = values.len() as f64;

    let fits_8 = values.iter().filter(|&&v| v >= i8::MIN as i64 && v <= i8::MAX as i64).count() as f64 / count * 100.0;
    let fits_16 = values.iter().filter(|&&v| v >= i16::MIN as i64 && v <= i16::MAX as i64).count() as f64 / count * 100.0;
    let fits_32 = values.iter().filter(|&&v| v >= i32::MIN as i64 && v <= i32::MAX as i64).count() as f64 / count * 100.0;

    eprintln!(
        "  {:<18} min={:<12} max={:<12} i8={:>5.1}% i16={:>5.1}% i32={:>5.1}%",
        name, min, max, fits_8, fits_16, fits_32
    );
}

fn print_column_stats(columns: &EncodedColumns, mapping_tsv_raw: usize, mapping_compressed_size: usize) {
    let num_rows = columns.event_type_indices.len();
    eprintln!("\n=== Per-Column Compressed Size Estimates ===");
    eprintln!("Total rows: {}", num_rows);
    eprintln!("{:<20} {:>10} {:>10} {:>8} {:>10}", "Column", "Raw", "Zstd", "Ratio", "B/Row");
    eprintln!("{}", "-".repeat(64));

    let mut total_raw = 0usize;
    let mut total_compressed = 0usize;

    // event_type (dict + indices)
    let dict_raw = columns.event_type_dict.len();
    let indices_raw = columns.event_type_indices.len();
    let event_type_raw = dict_raw + indices_raw;
    let mut event_type_buf = Vec::with_capacity(event_type_raw);
    event_type_buf.extend_from_slice(&columns.event_type_dict);
    event_type_buf.extend_from_slice(&columns.event_type_indices);
    let event_type_compressed = zstd::encode_all(event_type_buf.as_slice(), ZSTD_LEVEL).unwrap().len();
    total_raw += event_type_raw;
    total_compressed += event_type_compressed;
    eprintln!(
        "{:<20} {:>10} {:>10} {:>7.1}% {:>10.2}",
        "event_type",
        event_type_raw,
        event_type_compressed,
        100.0 * event_type_compressed as f64 / event_type_raw as f64,
        event_type_compressed as f64 / num_rows as f64
    );

    // event_id_delta
    let eid_raw = columns.event_id_deltas.len();
    let eid_compressed = zstd::encode_all(columns.event_id_deltas.as_slice(), ZSTD_LEVEL).unwrap().len();
    total_raw += eid_raw;
    total_compressed += eid_compressed;
    eprintln!(
        "{:<20} {:>10} {:>10} {:>7.1}% {:>10.2}",
        "event_id_delta",
        eid_raw,
        eid_compressed,
        100.0 * eid_compressed as f64 / eid_raw as f64,
        eid_compressed as f64 / num_rows as f64
    );

    // repo_pair_idx
    let repo_raw = columns.repo_pair_indices.len() * 4;
    let repo_bytes: Vec<u8> = columns.repo_pair_indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let repo_compressed = zstd::encode_all(repo_bytes.as_slice(), ZSTD_LEVEL).unwrap().len();
    total_raw += repo_raw;
    total_compressed += repo_compressed;
    eprintln!(
        "{:<20} {:>10} {:>10} {:>7.1}% {:>10.2}",
        "repo_pair_idx",
        repo_raw,
        repo_compressed,
        100.0 * repo_compressed as f64 / repo_raw as f64,
        repo_compressed as f64 / num_rows as f64
    );

    // created_at_delta
    let ts_raw = columns.created_at_deltas.len() * 2;
    let ts_bytes: Vec<u8> = columns.created_at_deltas.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ts_compressed = zstd::encode_all(ts_bytes.as_slice(), ZSTD_LEVEL).unwrap().len();
    total_raw += ts_raw;
    total_compressed += ts_compressed;
    eprintln!(
        "{:<20} {:>10} {:>10} {:>7.1}% {:>10.2}",
        "created_at_delta",
        ts_raw,
        ts_compressed,
        100.0 * ts_compressed as f64 / ts_raw as f64,
        ts_compressed as f64 / num_rows as f64
    );

    // mapping
    total_raw += mapping_tsv_raw;
    total_compressed += mapping_compressed_size;
    eprintln!(
        "{:<20} {:>10} {:>10} {:>7.1}% {:>10.2}",
        "repo mapping (TSV)",
        mapping_tsv_raw,
        mapping_compressed_size,
        100.0 * mapping_compressed_size as f64 / mapping_tsv_raw as f64,
        mapping_compressed_size as f64 / num_rows as f64
    );

    eprintln!("{}", "-".repeat(64));
    eprintln!(
        "{:<20} {:>10} {:>10} {:>7.1}% {:>10.2}",
        "TOTAL",
        total_raw,
        total_compressed,
        100.0 * total_compressed as f64 / total_raw as f64,
        total_compressed as f64 / num_rows as f64
    );

    // Value statistics
    eprintln!("\n=== Value Statistics ===");
    print_value_stats_unsigned("event_type_idx", &columns.event_type_indices.iter().map(|&v| v as u64).collect::<Vec<_>>());
    print_value_stats_unsigned("event_id_delta", &columns.event_id_deltas.iter().map(|&v| v as u64).collect::<Vec<_>>());
    print_value_stats_unsigned("repo_pair_idx", &columns.repo_pair_indices.iter().map(|&v| v as u64).collect::<Vec<_>>());
    print_value_stats_signed("created_at_delta", &columns.created_at_deltas.iter().map(|&v| v as i64).collect::<Vec<_>>());
    eprintln!();
}

fn parse_timestamp(ts: &str) -> i64 {
    DateTime::parse_from_rfc3339(ts)
        .map(|dt| dt.timestamp())
        .unwrap_or(0)
}

fn format_timestamp(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_default()
}

/// Header: first_event_id (u64) + first_timestamp (i64) + mapping_size (u32)
///       + num_events (u32) + event_type_dict_size (u16) + num_event_types (u8) + padding (u8)
struct Header {
    first_event_id: u64,
    first_timestamp: i64,
    mapping_size: u32,
    num_events: u32,
    event_type_dict_size: u16,
    num_event_types: u8,
}

const HEADER_SIZE: usize = 28; // 8 + 8 + 4 + 4 + 2 + 1 + 1

impl Header {
    fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.first_event_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.first_timestamp.to_le_bytes());
        buf[16..20].copy_from_slice(&self.mapping_size.to_le_bytes());
        buf[20..24].copy_from_slice(&self.num_events.to_le_bytes());
        buf[24..26].copy_from_slice(&self.event_type_dict_size.to_le_bytes());
        buf[26] = self.num_event_types;
        // buf[27] is padding
        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        let first_event_id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let first_timestamp = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let mapping_size = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let num_events = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        let event_type_dict_size = u16::from_le_bytes(bytes[24..26].try_into().unwrap());
        let num_event_types = bytes[26];
        Self {
            first_event_id,
            first_timestamp,
            mapping_size,
            num_events,
            event_type_dict_size,
            num_event_types,
        }
    }
}

/// Holds the encoded column data before compression
struct EncodedColumns {
    event_type_dict: Vec<u8>,       // newline-separated event type strings
    event_type_indices: Vec<u8>,    // i8 indices as bytes
    event_id_deltas: Vec<u8>,       // u8 deltas
    repo_pair_indices: Vec<u32>,    // u32 indices
    created_at_deltas: Vec<i16>,    // i16 deltas
}

/// Build TSV mapping table from events, returns (tsv_bytes, pair_to_idx mapping)
fn build_mapping_table(events: &[(EventKey, EventValue)]) -> (Vec<u8>, HashMap<(u32, String), u32>) {
    // Collect unique (repo_id, repo_name) pairs
    let mut unique_pairs: Vec<(u32, String)> = events
        .iter()
        .map(|(_, v)| (v.repo.id as u32, v.repo.name.clone()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Sort for deterministic ordering
    unique_pairs.sort();

    // Build TSV and index mapping
    let mut tsv = String::new();
    let mut pair_to_idx: HashMap<(u32, String), u32> = HashMap::new();

    for (idx, (repo_id, repo_name)) in unique_pairs.iter().enumerate() {
        tsv.push_str(&format!("{}\t{}\n", repo_id, repo_name));
        pair_to_idx.insert((*repo_id, repo_name.clone()), idx as u32);
    }

    (tsv.into_bytes(), pair_to_idx)
}

/// Parse TSV mapping table, returns Vec of (repo_id, repo_name) pairs
fn parse_mapping_table(tsv_bytes: &[u8]) -> Vec<(u32, String)> {
    let tsv = std::str::from_utf8(tsv_bytes).unwrap();
    tsv.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut parts = line.split('\t');
            let repo_id: u32 = parts.next().unwrap().parse().unwrap();
            let repo_name = parts.next().unwrap().to_string();
            (repo_id, repo_name)
        })
        .collect()
}

fn compute_u8_deltas(values: &[u64]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut deltas = Vec::with_capacity(values.len());
    deltas.push(0u8);
    for i in 1..values.len() {
        let delta = values[i] - values[i - 1];
        deltas.push(delta as u8);
    }
    deltas
}

fn compute_i16_deltas(values: &[i64]) -> Vec<i16> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut deltas = Vec::with_capacity(values.len());
    deltas.push(0i16);
    for i in 1..values.len() {
        let delta = values[i] - values[i - 1];
        deltas.push(delta as i16);
    }
    deltas
}

fn restore_u64_from_deltas(first: u64, deltas: &[u8]) -> Vec<u64> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(deltas.len());
    values.push(first);
    for i in 1..deltas.len() {
        values.push(values[i - 1] + deltas[i] as u64);
    }
    values
}

fn restore_i64_from_deltas(first: i64, deltas: &[i16]) -> Vec<i64> {
    if deltas.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(deltas.len());
    values.push(first);
    for i in 1..deltas.len() {
        values.push(values[i - 1] + deltas[i] as i64);
    }
    values
}

/// Build dictionary encoding for event types
fn build_event_type_dict(event_types: &[&str]) -> (Vec<u8>, Vec<u8>, HashMap<String, u8>) {
    // Collect unique event types and sort for deterministic ordering
    let mut unique_types: Vec<String> = event_types
        .iter()
        .map(|s| s.to_string())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique_types.sort();

    // Build dictionary: newline-separated strings
    let dict_str = unique_types.join("\n");
    let dict_bytes = dict_str.into_bytes();

    // Build string -> index mapping
    let str_to_idx: HashMap<String, u8> = unique_types
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as u8))
        .collect();

    // Encode each event type as its index
    let indices: Vec<u8> = event_types
        .iter()
        .map(|s| str_to_idx[*s])
        .collect();

    (dict_bytes, indices, str_to_idx)
}

fn encode_events(events: &[(EventKey, EventValue)]) -> Result<(Header, Vec<u8>, EncodedColumns), Box<dyn Error>> {
    // Build mapping table first
    let (mapping_tsv, pair_to_idx) = build_mapping_table(events);

    // Sort by event_id for optimal delta encoding
    let mut sorted_indices: Vec<usize> = (0..events.len()).collect();
    sorted_indices.sort_by_key(|&i| events[i].0.id.parse::<u64>().unwrap_or(0));

    // Collect values in sorted order
    let event_types: Vec<&str> = sorted_indices
        .iter()
        .map(|&i| events[i].0.event_type.as_str())
        .collect();
    let event_ids: Vec<u64> = sorted_indices
        .iter()
        .map(|&i| events[i].0.id.parse::<u64>().unwrap_or(0))
        .collect();
    let repo_pair_indices: Vec<u32> = sorted_indices
        .iter()
        .map(|&i| {
            let repo_id = events[i].1.repo.id as u32;
            let repo_name = events[i].1.repo.name.clone();
            pair_to_idx[&(repo_id, repo_name)]
        })
        .collect();
    let timestamps: Vec<i64> = sorted_indices
        .iter()
        .map(|&i| parse_timestamp(&events[i].1.created_at))
        .collect();

    // Build event type dictionary
    let (event_type_dict, event_type_indices, _) = build_event_type_dict(&event_types);
    let num_event_types = event_type_dict.split(|&b| b == b'\n').count() as u8;

    // Delta encode
    let event_id_deltas = compute_u8_deltas(&event_ids);
    let created_at_deltas = compute_i16_deltas(&timestamps);

    // Create header
    let header = Header {
        first_event_id: event_ids[0],
        first_timestamp: timestamps[0],
        mapping_size: mapping_tsv.len() as u32,
        num_events: events.len() as u32,
        event_type_dict_size: event_type_dict.len() as u16,
        num_event_types,
    };

    let columns = EncodedColumns {
        event_type_dict,
        event_type_indices,
        event_id_deltas,
        repo_pair_indices,
        created_at_deltas,
    };

    Ok((header, mapping_tsv, columns))
}

/// Decode events from raw column data
fn decode_columns(
    header: &Header,
    mapping: &[(u32, String)],
    event_type_dict: &[String],
    event_type_indices: &[u8],
    event_id_deltas: &[u8],
    repo_pair_indices: &[u32],
    created_at_deltas: &[i16],
) -> Vec<(EventKey, EventValue)> {
    // Restore values from deltas
    let event_ids = restore_u64_from_deltas(header.first_event_id, event_id_deltas);
    let timestamps = restore_i64_from_deltas(header.first_timestamp, created_at_deltas);

    (0..header.num_events as usize)
        .map(|i| {
            let event_type_idx = event_type_indices[i] as usize;
            let event_type = event_type_dict[event_type_idx].clone();

            let event_id = event_ids[i];

            // Look up repo info from mapping table
            let pair_idx = repo_pair_indices[i] as usize;
            let (repo_id, repo_name) = &mapping[pair_idx];

            let timestamp = timestamps[i];
            let repo_url = format!("https://api.github.com/repos/{}", repo_name);

            (
                EventKey {
                    id: event_id.to_string(),
                    event_type,
                },
                EventValue {
                    repo: Repo {
                        id: *repo_id as u64,
                        name: repo_name.clone(),
                        url: repo_url,
                    },
                    created_at: format_timestamp(timestamp),
                },
            )
        })
        .collect()
}

impl EventCodec for NatebrennandCodec {
    fn name(&self) -> &str {
        "natebrennand"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let (header, mapping_tsv, columns) = encode_events(events)?;

        // Compute mapping compressed size for stats
        let mapping_compressed_size = zstd::encode_all(mapping_tsv.as_slice(), ZSTD_LEVEL)?.len();

        if debug_enabled() {
            eprintln!("\n=== Header ({} bytes) ===", HEADER_SIZE);
            eprintln!("  first_event_id:       {}", header.first_event_id);
            eprintln!("  first_timestamp:      {} ({})", header.first_timestamp, format_timestamp(header.first_timestamp));
            eprintln!("  mapping_size:         {}", header.mapping_size);
            eprintln!("  num_events:           {}", header.num_events);
            eprintln!("  event_type_dict_size: {}", header.event_type_dict_size);
            eprintln!("  num_event_types:      {}", header.num_event_types);

            print_column_stats(&columns, mapping_tsv.len(), mapping_compressed_size);
        }

        // Build uncompressed payload:
        // - mapping TSV
        // - event_type dict (newline-separated)
        // - event_type indices
        // - event_id deltas
        // - repo_pair_indices (as le bytes)
        // - created_at deltas (as le bytes)
        let repo_bytes: Vec<u8> = columns.repo_pair_indices.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ts_bytes: Vec<u8> = columns.created_at_deltas.iter().flat_map(|v| v.to_le_bytes()).collect();

        let payload_size = mapping_tsv.len()
            + columns.event_type_dict.len()
            + columns.event_type_indices.len()
            + columns.event_id_deltas.len()
            + repo_bytes.len()
            + ts_bytes.len();

        let mut uncompressed = Vec::with_capacity(payload_size);
        uncompressed.extend_from_slice(&mapping_tsv);
        uncompressed.extend_from_slice(&columns.event_type_dict);
        uncompressed.extend_from_slice(&columns.event_type_indices);
        uncompressed.extend_from_slice(&columns.event_id_deltas);
        uncompressed.extend_from_slice(&repo_bytes);
        uncompressed.extend_from_slice(&ts_bytes);

        let compressed = zstd::encode_all(uncompressed.as_slice(), ZSTD_LEVEL)?;

        // Final output: header + compressed payload
        let mut buf = Vec::with_capacity(HEADER_SIZE + compressed.len());
        buf.extend_from_slice(&header.encode());
        buf.extend_from_slice(&compressed);

        if debug_enabled() {
            eprintln!("Uncompressed payload: {} bytes", payload_size);
            eprintln!("Compressed size (zstd {}): {} bytes", ZSTD_LEVEL, buf.len());
            eprintln!("Compressed bytes/row: {:.2}", buf.len() as f64 / events.len() as f64);
        }

        Ok(Bytes::from(buf))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let header = Header::decode(&bytes[0..HEADER_SIZE]);

        // Decompress
        let decompressed = zstd::decode_all(&bytes[HEADER_SIZE..])?;

        // Parse sections from decompressed payload
        let mut offset = 0;

        // 1. Mapping TSV
        let mapping_bytes = &decompressed[offset..offset + header.mapping_size as usize];
        offset += header.mapping_size as usize;
        let mapping = parse_mapping_table(mapping_bytes);

        // 2. Event type dictionary
        let dict_bytes = &decompressed[offset..offset + header.event_type_dict_size as usize];
        offset += header.event_type_dict_size as usize;
        let event_type_dict: Vec<String> = std::str::from_utf8(dict_bytes)?
            .split('\n')
            .map(|s| s.to_string())
            .collect();

        // 3. Event type indices
        let num_events = header.num_events as usize;
        let event_type_indices = &decompressed[offset..offset + num_events];
        offset += num_events;

        // 4. Event ID deltas
        let event_id_deltas = &decompressed[offset..offset + num_events];
        offset += num_events;

        // 5. Repo pair indices (u32 le)
        let repo_bytes = &decompressed[offset..offset + num_events * 4];
        offset += num_events * 4;
        let repo_pair_indices: Vec<u32> = repo_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // 6. Created at deltas (i16 le)
        let ts_bytes = &decompressed[offset..offset + num_events * 2];
        let created_at_deltas: Vec<i16> = ts_bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // Decode events
        let mut events = decode_columns(
            &header,
            &mapping,
            &event_type_dict,
            event_type_indices,
            event_id_deltas,
            &repo_pair_indices,
            &created_at_deltas,
        );

        // Sort by EventKey to match expected output
        events.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(events)
    }
}
