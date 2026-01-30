//! Hand-rolled columnar codec with per-column zstd compression.
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
//! - Store unique (repo_id, repo_name) pairs in binary mapping table (ID-sorted)
//! - Per event, store only an index into the mapping table (24-bit packed)
//! - Store first event_id in header, deltas as u8 (max delta is 251)
//! - Store first timestamp in header, deltas as varint-encoded i16
//! - Reconstruct repo.url from repo.name during decode
//! - Compress each column separately with zstd level 22
//!
//! Binary format:
//!   Header (56 bytes, uncompressed):
//!     - first_event_id: u64
//!     - first_timestamp: i64
//!     - num_events: u32
//!     - num_mapping_entries: u32
//!     - event_type_dict_size: u16 (uncompressed)
//!     - num_event_types: u8
//!     - _padding: u8
//!     - mapping_ids_compressed: u32
//!     - mapping_names_compressed: u32
//!     - dict_compressed: u32
//!     - packed_compressed: u32
//!     - id_deltas_compressed: u32
//!     - repo_compressed: u32
//!     - ts_compressed: u32
//!
//!   Per-column compressed payloads (zstd-22, concatenated):
//!     1. mapping IDs: varint delta-encoded repo_ids (sorted by ID)
//!     2. mapping names: newline-separated repo names (in ID order)
//!     3. event_type dict (newline-separated strings)
//!     4. event_type indices: 4-bit packed [(num_events+1)/2 bytes]
//!     5. event_id deltas: [u8; num_events]
//!     6. repo_pair indices: 24-bit packed [3 bytes each]
//!     7. created_at deltas: zigzag + varint encoded
//!
//! Compression techniques - what worked:
//!   - Binary mapping format: Store (repo_id, repo_name) pairs as two separate columns:
//!     IDs sorted and delta+varint encoded, names newline-separated. Sorting by repo_id
//!     makes ID deltas small (varint-compressible). Separating IDs from names allows
//!     zstd to compress each optimally. Saves 734KB (11%) vs TSV format.
//!   - Per-column zstd compression: Compress each column separately instead of
//!     concatenating then compressing. Each column has different statistical properties
//!     (text vs integers, different value ranges) that benefit from independent
//!     compression contexts. Saves ~27KB (0.4%) vs combined compression.
//!   - 24-bit packing for repo_pair_idx: Indices fit in 18 bits (max 262K repos), but
//!     byte-aligned 24-bit packing compresses better than true bit-packing because
//!     zstd works on bytes. Saves 100KB vs u32.
//!   - Varint for timestamp deltas: 99.9% of deltas fit in 1 byte after zigzag encoding.
//!     Saves 3.8KB vs fixed i16.
//!   - Delta encoding for event_id: Sorting by event_id makes deltas small (max 251),
//!     fitting in u8. Compresses to 0.35 B/row.
//!   - Dictionary encoding for event_type: 15 unique values, 4-bit packed indices
//!     (2 per byte). Compresses to 0.22 B/row.
//!   - zstd level 22: High compression ratio, acceptable encode time for batch use.
//!
//! Compression techniques - what didn't work:
//!   - TSV mapping format: Storing (repo_id\trepo_name\n) in alphabetical order.
//!     Result: 3.78 B/row. Switched to binary format (ID-sorted, delta-encoded IDs,
//!     separate names column): 3.04 B/row - saves 734KB (11%). Sorting by ID makes
//!     deltas small; separating columns lets zstd optimize each independently.
//!   - RLE for event_type: Data sorted by event_id, not event_type, so runs are
//!     short (avg 2.09, 67% length 1). RLE saves only 4.1% on raw data, and zstd
//!     already achieves 25% ratio.
//!   - Huffman encoding for event_type: Shannon entropy is 1.758 bits/symbol,
//!     theoretical minimum 220KB. Current 4-bit + zstd achieves 223KB - already
//!     within 1.5% of theoretical optimum. Huffman would give ~279KB raw, and
//!     zstd can't compress it further since it's already entropy-coded.
//!   - Frequency-ordered event_type indices: Reordering dictionary so most common
//!     types (PushEvent=68.9%) get index 0 instead of 12. Result: +4 bytes worse.
//!     zstd already handles the patterns efficiently with alphabetical ordering.
//!   - Owner/repo name splitting: Separate owner ("microsoft") from repo ("vscode")
//!     into two dictionaries, store (owner_idx, suffix_idx) per entry. Result: +350KB
//!     worse. 87.6% of owners have only 1 repo, so there's almost no sharing benefit.
//!     The overhead of separate dictionaries outweighs any gains.
//!   - Frequency-ordered repo indices: Reordering repos by frequency (most common
//!     first) improved index compression by 60KB (more values fit in smaller bytes),
//!     but hurt name compression by 90KB (lost prefix sharing). Net loss.
//!   - Varint for repo_pair_idx: Would use 2.94MB vs zstd's 2.32MB on raw u32.
//!     zstd's entropy coding already beats simple varint.
//!   - Delta encoding for repo_pair_idx: Repos are essentially random when sorted
//!     by event_id (99.6% runs of length 1). Deltas span full range (-262K to +262K),
//!     only 26% fit in i16. Would use 3.47MB vs zstd's 2.32MB.
//!   - FST (finite state transducer) for repo mapping: 7.31 B/row vs binary's 3.04 B/row.
//!     zstd's compression beats FST's space-efficient structure.
//!   - RSMarisa trie for repo names: 8.46 B/row vs binary's 3.04 B/row. The trie's
//!     structure doesn't help when the whole thing gets zstd-compressed anyway.
//!   - Combined column compression: Concatenating all columns then compressing as one
//!     zstd stream. Heterogeneous data types hurt each other when combined.
//!   - zstd parameter tuning: Tested window_log (17-30), pledged_src_size, and checksum.
//!     Default L22 settings (window_log=27, no checksum) are already optimal.
//!
//! Current results (1M events):
//!   - event_type:       0.22 B/row (4-bit packed + zstd)
//!   - event_id_delta:   0.35 B/row (u8 + zstd)
//!   - repo_pair_idx:    2.32 B/row (24-bit + zstd)
//!   - created_at_delta: 0.04 B/row (varint + zstd)
//!   - mapping IDs:      0.06 B/row (delta + varint + zstd)
//!   - mapping names:    2.98 B/row (newline-sep + zstd)
//!   - Header:           56 bytes
//!   - TOTAL:            6.00 B/row (5,996,774 bytes, -97.2% vs naive)
//!
//! Set NATE_DEBUG=1 to see column size statistics and compression experiments.

use bytes::Bytes;
use chrono::DateTime;
use std::collections::HashMap;
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const ZSTD_LEVEL: i32 = 22;


/// Pack u32 values into 24-bit (3 byte) little-endian format
fn pack_u24(values: &[u32]) -> Vec<u8> {
    let mut result = Vec::with_capacity(values.len() * 3);
    for &val in values {
        // Store low 24 bits as 3 bytes little-endian
        result.push(val as u8);
        result.push((val >> 8) as u8);
        result.push((val >> 16) as u8);
    }
    result
}

/// Unpack 24-bit (3 byte) little-endian values back to u32
fn unpack_u24(bytes: &[u8], count: usize) -> Vec<u32> {
    let mut result = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(3).take(count) {
        let val = chunk[0] as u32
            | ((chunk[1] as u32) << 8)
            | ((chunk[2] as u32) << 16);
        result.push(val);
    }
    result
}

/// Encode i16 values as zigzag + varint
fn encode_varint_i16(values: &[i16]) -> Vec<u8> {
    let mut result = Vec::with_capacity(values.len() * 2);
    for &v in values {
        // ZigZag encode: (v << 1) ^ (v >> 15)
        let zigzag = ((v as i32) << 1) ^ ((v as i32) >> 31);
        let mut val = zigzag as u32;
        loop {
            if val < 128 {
                result.push(val as u8);
                break;
            }
            result.push((val as u8 & 0x7f) | 0x80);
            val >>= 7;
        }
    }
    result
}

/// Decode zigzag + varint back to i16 values
fn decode_varint_i16(bytes: &[u8], count: usize) -> Vec<i16> {
    let mut result = Vec::with_capacity(count);
    let mut pos = 0;
    for _ in 0..count {
        let mut val: u32 = 0;
        let mut shift = 0;
        loop {
            let b = bytes[pos];
            pos += 1;
            val |= ((b & 0x7f) as u32) << shift;
            if b < 128 {
                break;
            }
            shift += 7;
        }
        // ZigZag decode: (val >> 1) ^ -(val & 1)
        let decoded = ((val >> 1) as i32) ^ -((val & 1) as i32);
        result.push(decoded as i16);
    }
    result
}


pub struct NatebrennandCodec;

impl NatebrennandCodec {
    pub fn new() -> Self {
        Self
    }
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

/// Header for per-column compressed format
/// Base fields (28 bytes):
///   - first_event_id: u64
///   - first_timestamp: i64
///   - num_events: u32
///   - num_mapping_entries: u32 (number of repo entries)
///   - event_type_dict_size: u16 (uncompressed)
///   - num_event_types: u8
///   - _padding: u8
/// Compressed sizes (28 bytes):
///   - mapping_ids_compressed: u32
///   - mapping_names_compressed: u32
///   - dict_compressed: u32
///   - packed_compressed: u32
///   - id_deltas_compressed: u32
///   - repo_compressed: u32
///   - ts_compressed: u32
struct Header {
    first_event_id: u64,
    first_timestamp: i64,
    num_events: u32,
    num_mapping_entries: u32,
    event_type_dict_size: u16,
    num_event_types: u8,
    // Compressed sizes for each column
    mapping_ids_compressed: u32,
    mapping_names_compressed: u32,
    dict_compressed: u32,
    packed_compressed: u32,
    id_deltas_compressed: u32,
    repo_compressed: u32,
    ts_compressed: u32,
}

const HEADER_SIZE: usize = 56; // 8 + 8 + 4 + 4 + 2 + 1 + 1 + (7 * 4)

impl Header {
    fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        let mut pos = 0;

        buf[pos..pos+8].copy_from_slice(&self.first_event_id.to_le_bytes());
        pos += 8;
        buf[pos..pos+8].copy_from_slice(&self.first_timestamp.to_le_bytes());
        pos += 8;
        buf[pos..pos+4].copy_from_slice(&self.num_events.to_le_bytes());
        pos += 4;
        buf[pos..pos+4].copy_from_slice(&self.num_mapping_entries.to_le_bytes());
        pos += 4;
        buf[pos..pos+2].copy_from_slice(&self.event_type_dict_size.to_le_bytes());
        pos += 2;
        buf[pos] = self.num_event_types;
        pos += 1;
        // padding byte
        pos += 1;

        // Compressed sizes
        buf[pos..pos+4].copy_from_slice(&self.mapping_ids_compressed.to_le_bytes());
        pos += 4;
        buf[pos..pos+4].copy_from_slice(&self.mapping_names_compressed.to_le_bytes());
        pos += 4;
        buf[pos..pos+4].copy_from_slice(&self.dict_compressed.to_le_bytes());
        pos += 4;
        buf[pos..pos+4].copy_from_slice(&self.packed_compressed.to_le_bytes());
        pos += 4;
        buf[pos..pos+4].copy_from_slice(&self.id_deltas_compressed.to_le_bytes());
        pos += 4;
        buf[pos..pos+4].copy_from_slice(&self.repo_compressed.to_le_bytes());
        pos += 4;
        buf[pos..pos+4].copy_from_slice(&self.ts_compressed.to_le_bytes());

        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        let mut pos = 0;

        let first_event_id = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
        pos += 8;
        let first_timestamp = i64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap());
        pos += 8;
        let num_events = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
        pos += 4;
        let num_mapping_entries = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
        pos += 4;
        let event_type_dict_size = u16::from_le_bytes(bytes[pos..pos+2].try_into().unwrap());
        pos += 2;
        let num_event_types = bytes[pos];
        pos += 2; // skip num_event_types + padding

        let mapping_ids_compressed = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
        pos += 4;
        let mapping_names_compressed = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
        pos += 4;
        let dict_compressed = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
        pos += 4;
        let packed_compressed = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
        pos += 4;
        let id_deltas_compressed = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
        pos += 4;
        let repo_compressed = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());
        pos += 4;
        let ts_compressed = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap());

        Self {
            first_event_id,
            first_timestamp,
            num_events,
            num_mapping_entries,
            event_type_dict_size,
            num_event_types,
            mapping_ids_compressed,
            mapping_names_compressed,
            dict_compressed,
            packed_compressed,
            id_deltas_compressed,
            repo_compressed,
            ts_compressed,
        }
    }
}

/// Holds the encoded column data before compression
struct EncodedColumns {
    event_type_dict: Vec<u8>,       // newline-separated event type strings
    event_type_packed: Vec<u8>,     // 4-bit packed indices (2 per byte)
    event_id_deltas: Vec<u8>,       // u8 deltas
    repo_pair_indices: Vec<u32>,    // u32 indices
    created_at_deltas: Vec<i16>,    // i16 deltas
}

/// Pack 4-bit values into bytes (2 values per byte, low nibble first)
fn pack_nibbles(values: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 1) / 2);
    for chunk in values.chunks(2) {
        let byte = chunk[0] | (chunk.get(1).copied().unwrap_or(0) << 4);
        packed.push(byte);
    }
    packed
}

/// Unpack 4-bit values from bytes
fn unpack_nibbles(packed: &[u8], count: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(count);
    for &byte in packed {
        values.push(byte & 0x0F);
        if values.len() < count {
            values.push(byte >> 4);
        }
        if values.len() >= count {
            break;
        }
    }
    values
}

/// Build binary mapping table from events
/// Returns (id_deltas_varint, names_bytes, pair_to_idx mapping)
/// Repos are sorted by repo_id for optimal delta encoding of IDs.
fn build_mapping_table(events: &[(EventKey, EventValue)]) -> (Vec<u8>, Vec<u8>, HashMap<(u32, String), u32>) {
    // Collect unique (repo_id, repo_name) pairs
    let mut unique_pairs: Vec<(u32, String)> = events
        .iter()
        .map(|(_, v)| (v.repo.id as u32, v.repo.name.clone()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Sort by repo_id for optimal delta encoding
    unique_pairs.sort_by_key(|(id, _)| *id);

    // Build index mapping and collect data
    let mut pair_to_idx: HashMap<(u32, String), u32> = HashMap::new();
    let mut names: Vec<&str> = Vec::with_capacity(unique_pairs.len());

    for (idx, (repo_id, repo_name)) in unique_pairs.iter().enumerate() {
        pair_to_idx.insert((*repo_id, repo_name.clone()), idx as u32);
        names.push(repo_name);
    }

    // Delta-encode IDs as varints
    let mut id_deltas: Vec<u8> = Vec::new();
    let mut prev_id: u32 = 0;
    for (repo_id, _) in &unique_pairs {
        let delta = repo_id - prev_id;
        // Varint encode
        let mut val = delta;
        loop {
            if val < 128 {
                id_deltas.push(val as u8);
                break;
            }
            id_deltas.push((val as u8 & 0x7f) | 0x80);
            val >>= 7;
        }
        prev_id = *repo_id;
    }

    // Names as newline-separated
    let names_bytes = names.join("\n").into_bytes();

    (id_deltas, names_bytes, pair_to_idx)
}

/// Parse binary mapping table, returns Vec of (repo_id, repo_name) pairs
fn parse_mapping_table(id_bytes: &[u8], names_bytes: &[u8], num_entries: usize) -> Vec<(u32, String)> {
    // Decode varint delta-encoded IDs
    let mut ids = Vec::with_capacity(num_entries);
    let mut pos = 0;
    let mut prev_id: u32 = 0;
    for _ in 0..num_entries {
        let mut val: u32 = 0;
        let mut shift = 0;
        loop {
            let b = id_bytes[pos];
            pos += 1;
            val |= ((b & 0x7f) as u32) << shift;
            if b < 128 {
                break;
            }
            shift += 7;
        }
        prev_id += val;
        ids.push(prev_id);
    }

    // Parse newline-separated names
    let names_str = std::str::from_utf8(names_bytes).unwrap();
    let names: Vec<&str> = names_str.split('\n').collect();

    // Combine into pairs
    ids.into_iter()
        .zip(names.into_iter())
        .map(|(id, name)| (id, name.to_string()))
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

/// Returns (base_info, mapping_ids, mapping_names, columns) where base_info is (first_event_id, first_timestamp, num_events, num_mapping_entries, num_event_types)
fn encode_events(events: &[(EventKey, EventValue)]) -> Result<((u64, i64, u32, u32, u8), Vec<u8>, Vec<u8>, EncodedColumns), Box<dyn Error>> {
    // Build mapping table first (binary format: ID deltas + names)
    let (mapping_ids, mapping_names, pair_to_idx) = build_mapping_table(events);
    let num_mapping_entries = pair_to_idx.len();

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

    let columns = EncodedColumns {
        event_type_dict,
        event_type_packed: pack_nibbles(&event_type_indices),
        event_id_deltas,
        repo_pair_indices,
        created_at_deltas,
    };

    // Return base header info (compressed sizes filled in later)
    let base_info = (
        event_ids[0],           // first_event_id
        timestamps[0],          // first_timestamp
        events.len() as u32,    // num_events
        num_mapping_entries as u32, // num_mapping_entries
        num_event_types,
    );

    Ok((base_info, mapping_ids, mapping_names, columns))
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
            let repo_url = format!("https://api.github.com/repos/{repo_name}");

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
        let (base_info, mapping_ids, mapping_names, columns) = encode_events(events)?;
        let (first_event_id, first_timestamp, num_events, num_mapping_entries, num_event_types) = base_info;

        // Convert columns to bytes
        let repo_bytes: Vec<u8> = pack_u24(&columns.repo_pair_indices);  // 24-bit packing
        let ts_bytes: Vec<u8> = encode_varint_i16(&columns.created_at_deltas);  // varint encoding

        if debug_enabled() {
            experiment_timestamp_encoding(&columns.created_at_deltas);
            experiment_repo_bit_packing(&columns.repo_pair_indices);
        }

        // Compress each column separately (mapping split into IDs and names)
        let mapping_ids_compressed = zstd::encode_all(mapping_ids.as_slice(), ZSTD_LEVEL)?;
        let mapping_names_compressed = zstd::encode_all(mapping_names.as_slice(), ZSTD_LEVEL)?;
        let dict_compressed = zstd::encode_all(columns.event_type_dict.as_slice(), ZSTD_LEVEL)?;
        let packed_compressed = zstd::encode_all(columns.event_type_packed.as_slice(), ZSTD_LEVEL)?;
        let id_deltas_compressed = zstd::encode_all(columns.event_id_deltas.as_slice(), ZSTD_LEVEL)?;
        let repo_compressed = zstd::encode_all(repo_bytes.as_slice(), ZSTD_LEVEL)?;
        let ts_compressed = zstd::encode_all(ts_bytes.as_slice(), ZSTD_LEVEL)?;

        // Build header with compressed sizes
        let header = Header {
            first_event_id,
            first_timestamp,
            num_events,
            num_mapping_entries,
            event_type_dict_size: columns.event_type_dict.len() as u16,
            num_event_types,
            mapping_ids_compressed: mapping_ids_compressed.len() as u32,
            mapping_names_compressed: mapping_names_compressed.len() as u32,
            dict_compressed: dict_compressed.len() as u32,
            packed_compressed: packed_compressed.len() as u32,
            id_deltas_compressed: id_deltas_compressed.len() as u32,
            repo_compressed: repo_compressed.len() as u32,
            ts_compressed: ts_compressed.len() as u32,
        };

        if debug_enabled() {
            eprintln!("\n=== Header ({HEADER_SIZE} bytes) ===");
            eprintln!("  first_event_id:       {}", header.first_event_id);
            eprintln!("  first_timestamp:      {} ({})", header.first_timestamp, format_timestamp(header.first_timestamp));
            eprintln!("  num_mapping_entries:  {}", header.num_mapping_entries);
            eprintln!("  num_events:           {}", header.num_events);
            eprintln!("  event_type_dict_size: {}", header.event_type_dict_size);
            eprintln!("  num_event_types:      {}", header.num_event_types);
            eprintln!("  mapping_ids_compressed:   {}", header.mapping_ids_compressed);
            eprintln!("  mapping_names_compressed: {}", header.mapping_names_compressed);
        }

        // Final output: header + compressed columns (concatenated)
        let total_compressed = mapping_ids_compressed.len()
            + mapping_names_compressed.len()
            + dict_compressed.len()
            + packed_compressed.len()
            + id_deltas_compressed.len()
            + repo_compressed.len()
            + ts_compressed.len();

        let mut buf = Vec::with_capacity(HEADER_SIZE + total_compressed);
        buf.extend_from_slice(&header.encode());
        buf.extend_from_slice(&mapping_ids_compressed);
        buf.extend_from_slice(&mapping_names_compressed);
        buf.extend_from_slice(&dict_compressed);
        buf.extend_from_slice(&packed_compressed);
        buf.extend_from_slice(&id_deltas_compressed);
        buf.extend_from_slice(&repo_compressed);
        buf.extend_from_slice(&ts_compressed);

        if debug_enabled() {
            let payload_size = mapping_ids.len()
                + mapping_names.len()
                + columns.event_type_dict.len()
                + columns.event_type_packed.len()
                + columns.event_id_deltas.len()
                + repo_bytes.len()
                + ts_bytes.len();
            eprintln!("Uncompressed payload: {payload_size} bytes");
            eprintln!("Compressed size (per-column zstd {}): {} bytes", ZSTD_LEVEL, buf.len());
            eprintln!("Compressed bytes/row: {:.2}", buf.len() as f64 / events.len() as f64);
        }

        Ok(Bytes::from(buf))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let header = Header::decode(&bytes[0..HEADER_SIZE]);

        // Decompress each column separately using sizes from header
        let mut offset = HEADER_SIZE;

        // 1. Mapping IDs (varint delta-encoded)
        let mapping_ids_compressed = &bytes[offset..offset + header.mapping_ids_compressed as usize];
        offset += header.mapping_ids_compressed as usize;
        let mapping_ids_bytes = zstd::decode_all(mapping_ids_compressed)?;

        // 2. Mapping names (newline-separated)
        let mapping_names_compressed = &bytes[offset..offset + header.mapping_names_compressed as usize];
        offset += header.mapping_names_compressed as usize;
        let mapping_names_bytes = zstd::decode_all(mapping_names_compressed)?;

        // Parse binary mapping table
        let mapping = parse_mapping_table(
            &mapping_ids_bytes,
            &mapping_names_bytes,
            header.num_mapping_entries as usize,
        );

        // 3. Event type dictionary
        let dict_compressed = &bytes[offset..offset + header.dict_compressed as usize];
        offset += header.dict_compressed as usize;
        let dict_bytes = zstd::decode_all(dict_compressed)?;
        let event_type_dict: Vec<String> = std::str::from_utf8(&dict_bytes)?
            .split('\n')
            .map(|s| s.to_string())
            .collect();

        // 4. Event type indices (4-bit packed)
        let packed_compressed = &bytes[offset..offset + header.packed_compressed as usize];
        offset += header.packed_compressed as usize;
        let packed_bytes = zstd::decode_all(packed_compressed)?;
        let num_events = header.num_events as usize;
        let event_type_indices = unpack_nibbles(&packed_bytes, num_events);

        // 5. Event ID deltas
        let id_deltas_compressed = &bytes[offset..offset + header.id_deltas_compressed as usize];
        offset += header.id_deltas_compressed as usize;
        let event_id_deltas = zstd::decode_all(id_deltas_compressed)?;

        // 6. Repo pair indices (24-bit packed)
        let repo_compressed = &bytes[offset..offset + header.repo_compressed as usize];
        offset += header.repo_compressed as usize;
        let repo_bytes = zstd::decode_all(repo_compressed)?;
        let repo_pair_indices = unpack_u24(&repo_bytes, num_events);

        // 7. Created at deltas (varint encoded)
        let ts_compressed = &bytes[offset..offset + header.ts_compressed as usize];
        let ts_bytes = zstd::decode_all(ts_compressed)?;
        let created_at_deltas = decode_varint_i16(&ts_bytes, num_events);

        // Decode events
        let mut events = decode_columns(
            &header,
            &mapping,
            &event_type_dict,
            &event_type_indices,
            &event_id_deltas,
            &repo_pair_indices,
            &created_at_deltas,
        );

        // Sort by EventKey to match expected output
        events.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(events)
    }
}

// ============================================================================
// Debug and analysis functions (enabled with NATE_DEBUG=1)
// ============================================================================

/// Compress with custom zstd parameters
fn compress_with_params(data: &[u8], level: i32, window_log: Option<u32>, pledged_size: bool, checksum: bool) -> Vec<u8> {
    use std::io::Write;
    let mut encoder = zstd::Encoder::new(Vec::new(), level).unwrap();

    if let Some(wl) = window_log {
        encoder.set_parameter(zstd::zstd_safe::CParameter::WindowLog(wl)).ok();
    }
    if pledged_size {
        encoder.set_pledged_src_size(Some(data.len() as u64)).ok();
    }
    encoder.include_checksum(checksum).ok();

    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

/// Pack u32 values into bit-packed bytes (variable bits per value) - for experiments
fn pack_bits(values: &[u32], bits_per_value: u32) -> Vec<u8> {
    let total_bits = values.len() as u64 * bits_per_value as u64;
    let num_bytes = ((total_bits + 7) / 8) as usize;
    let mut result = vec![0u8; num_bytes];

    let mut bit_pos: u64 = 0;
    for &val in values {
        // Write bits_per_value bits starting at bit_pos
        let mut remaining = bits_per_value;
        let mut v = val;
        let mut pos = bit_pos;

        while remaining > 0 {
            let byte_idx = (pos / 8) as usize;
            let bit_offset = (pos % 8) as u32;
            let bits_in_byte = std::cmp::min(remaining, 8 - bit_offset);
            let mask = ((1u32 << bits_in_byte) - 1) as u8;
            result[byte_idx] |= ((v as u8) & mask) << bit_offset;
            v >>= bits_in_byte;
            remaining -= bits_in_byte;
            pos += bits_in_byte as u64;
        }
        bit_pos += bits_per_value as u64;
    }
    result
}

/// Experiment with delta encoding repo_ids in mapping table
fn experiment_mapping_formats(mapping_tsv: &[u8]) {
    eprintln!("\n=== Mapping Format Experiment ===");

    // Parse current TSV
    let tsv_str = std::str::from_utf8(mapping_tsv).unwrap();
    let mut entries: Vec<(u32, &str)> = Vec::new();
    for line in tsv_str.lines() {
        if let Some((id_str, name)) = line.split_once('\t') {
            if let Ok(id) = id_str.parse::<u32>() {
                entries.push((id, name));
            }
        }
    }

    let current_compressed = zstd::encode_all(mapping_tsv, ZSTD_LEVEL).unwrap().len();
    eprintln!("Current TSV (alpha sorted): {} bytes compressed", current_compressed);

    // Option 1: Sort by repo_id, delta encode IDs
    let mut by_id = entries.clone();
    by_id.sort_by_key(|(id, _)| *id);

    let mut delta_tsv = String::new();
    let mut prev_id: u32 = 0;
    for (id, name) in &by_id {
        let delta = id - prev_id;
        delta_tsv.push_str(&format!("{}\t{}\n", delta, name));
        prev_id = *id;
    }
    let delta_compressed = zstd::encode_all(delta_tsv.as_bytes(), ZSTD_LEVEL).unwrap().len();
    eprintln!("ID-sorted + delta TSV:      {} bytes ({:+})",
        delta_compressed, delta_compressed as i64 - current_compressed as i64);

    // Option 2: Binary format - separate columns for IDs and names
    // IDs as delta-encoded varints, names as newline-separated
    let mut id_deltas: Vec<u32> = Vec::new();
    prev_id = 0;
    for (id, _) in &by_id {
        id_deltas.push(id - prev_id);
        prev_id = *id;
    }

    // Varint encode the deltas
    let mut id_bytes: Vec<u8> = Vec::new();
    for delta in &id_deltas {
        let mut val = *delta;
        loop {
            if val < 128 {
                id_bytes.push(val as u8);
                break;
            }
            id_bytes.push((val as u8 & 0x7f) | 0x80);
            val >>= 7;
        }
    }

    let names: Vec<&str> = by_id.iter().map(|(_, n)| *n).collect();
    let names_bytes = names.join("\n").into_bytes();

    let id_compressed = zstd::encode_all(id_bytes.as_slice(), ZSTD_LEVEL).unwrap().len();
    let names_compressed = zstd::encode_all(names_bytes.as_slice(), ZSTD_LEVEL).unwrap().len();
    let binary_total = id_compressed + names_compressed;

    eprintln!("Binary (ID-sorted):");
    eprintln!("  ID deltas (varint):       {} -> {} compressed", id_bytes.len(), id_compressed);
    eprintln!("  Names:                    {} -> {} compressed", names_bytes.len(), names_compressed);
    eprintln!("  Total:                    {} bytes ({:+})",
        binary_total, binary_total as i64 - current_compressed as i64);

    // Option 3: Keep alpha sort, but binary format (separate ID column)
    let names_alpha: Vec<&str> = entries.iter().map(|(_, n)| *n).collect();
    let names_alpha_bytes = names_alpha.join("\n").into_bytes();
    let ids_alpha: Vec<u8> = entries.iter().flat_map(|(id, _)| id.to_le_bytes()[..3].to_vec()).collect();

    let names_alpha_compressed = zstd::encode_all(names_alpha_bytes.as_slice(), ZSTD_LEVEL).unwrap().len();
    let ids_alpha_compressed = zstd::encode_all(ids_alpha.as_slice(), ZSTD_LEVEL).unwrap().len();
    let binary_alpha_total = names_alpha_compressed + ids_alpha_compressed;

    eprintln!("Binary (alpha-sorted):");
    eprintln!("  IDs (24-bit):             {} -> {} compressed", ids_alpha.len(), ids_alpha_compressed);
    eprintln!("  Names:                    {} -> {} compressed", names_alpha_bytes.len(), names_alpha_compressed);
    eprintln!("  Total:                    {} bytes ({:+})",
        binary_alpha_total, binary_alpha_total as i64 - current_compressed as i64);
}

/// Experiment with variable-width timestamp deltas
fn experiment_timestamp_encoding(deltas: &[i16]) {
    eprintln!("\n=== Timestamp Delta Encoding Experiment ===");

    let fits_i8 = deltas.iter().filter(|&&d| d >= i8::MIN as i16 && d <= i8::MAX as i16).count();
    eprintln!("Deltas fitting in i8: {} ({:.2}%)", fits_i8, fits_i8 as f64 / deltas.len() as f64 * 100.0);

    // Current: i16 (2 bytes each)
    let i16_bytes: Vec<u8> = deltas.iter().flat_map(|v| v.to_le_bytes()).collect();
    let i16_compressed = zstd::encode_all(i16_bytes.as_slice(), ZSTD_LEVEL).unwrap();
    eprintln!("Current (i16):       {} raw -> {} compressed", i16_bytes.len(), i16_compressed.len());

    // Option 1: Pack i8 values with exception list for larger values
    // Format: [i8 values where value fits, else i8::MIN as sentinel] + [exception list]
    let mut packed_i8: Vec<u8> = Vec::with_capacity(deltas.len());
    let mut exceptions: Vec<(u32, i16)> = Vec::new(); // (index, value)
    for (i, &d) in deltas.iter().enumerate() {
        if d >= i8::MIN as i16 && d <= i8::MAX as i16 {
            packed_i8.push(d as i8 as u8);
        } else {
            packed_i8.push(i8::MIN as u8); // sentinel
            exceptions.push((i as u32, d));
        }
    }
    let exception_bytes: Vec<u8> = exceptions.iter()
        .flat_map(|(idx, val)| {
            let mut v = Vec::with_capacity(6);
            v.extend_from_slice(&idx.to_le_bytes()[..3]); // 24-bit index
            v.extend_from_slice(&val.to_le_bytes()); // i16 value
            v
        })
        .collect();

    let packed_compressed = zstd::encode_all(packed_i8.as_slice(), ZSTD_LEVEL).unwrap();
    let exceptions_compressed = zstd::encode_all(exception_bytes.as_slice(), ZSTD_LEVEL).unwrap();
    let option1_total = packed_compressed.len() + exceptions_compressed.len();

    eprintln!("Option 1 (i8 + exceptions):");
    eprintln!("  i8 values:         {} -> {} compressed", packed_i8.len(), packed_compressed.len());
    eprintln!("  Exceptions ({}):  {} -> {} compressed",
        exceptions.len(), exception_bytes.len(), exceptions_compressed.len());
    eprintln!("  Total:             {} ({:+} vs i16)",
        option1_total, option1_total as i64 - i16_compressed.len() as i64);

    // Option 2: Varint encoding
    let mut varint_bytes: Vec<u8> = Vec::new();
    for &d in deltas {
        // ZigZag encode then varint
        let zigzag = ((d as i32) << 1) ^ ((d as i32) >> 31);
        let mut val = zigzag as u32;
        loop {
            if val < 128 {
                varint_bytes.push(val as u8);
                break;
            }
            varint_bytes.push((val as u8 & 0x7f) | 0x80);
            val >>= 7;
        }
    }
    let varint_compressed = zstd::encode_all(varint_bytes.as_slice(), ZSTD_LEVEL).unwrap();
    eprintln!("Option 2 (varint):   {} raw -> {} compressed ({:+} vs i16)",
        varint_bytes.len(), varint_compressed.len(),
        varint_compressed.len() as i64 - i16_compressed.len() as i64);
}

/// Experiment with bit-packing repo indices
fn experiment_repo_bit_packing(repo_indices: &[u32]) {
    eprintln!("\n=== Repo Index Bit-Packing Experiment ===");

    let max_val = *repo_indices.iter().max().unwrap_or(&0);
    let bits_needed = if max_val > 0 { 32 - max_val.leading_zeros() } else { 1 };
    eprintln!("Max value: {}, bits needed: {}", max_val, bits_needed);

    // Current: u32 (32 bits)
    let u32_bytes: Vec<u8> = repo_indices.iter().flat_map(|v| v.to_le_bytes()).collect();
    let u32_compressed = zstd::encode_all(u32_bytes.as_slice(), ZSTD_LEVEL).unwrap();
    eprintln!("Current (u32):     {} raw -> {} compressed", u32_bytes.len(), u32_compressed.len());

    // Try different bit widths
    for bits in [18, 19, 20, 24] {
        if bits >= bits_needed {
            let packed = pack_bits(repo_indices, bits);
            let compressed = zstd::encode_all(packed.as_slice(), ZSTD_LEVEL).unwrap();
            eprintln!("{}-bit packed:     {} raw -> {} compressed ({:+} vs u32)",
                bits, packed.len(), compressed.len(),
                compressed.len() as i64 - u32_compressed.len() as i64);
        }
    }

    // Also try u16 if values fit (they don't, but for comparison)
    if max_val <= u16::MAX as u32 {
        let u16_bytes: Vec<u8> = repo_indices.iter().flat_map(|v| (*v as u16).to_le_bytes()).collect();
        let u16_compressed = zstd::encode_all(u16_bytes.as_slice(), ZSTD_LEVEL).unwrap();
        eprintln!("u16:               {} raw -> {} compressed ({:+} vs u32)",
            u16_bytes.len(), u16_compressed.len(),
            u16_compressed.len() as i64 - u32_compressed.len() as i64);
    }
}

/// Experiment with different zstd compression approaches
fn experiment_compression_strategies(
    mapping_tsv: &[u8],
    event_type_dict: &[u8],
    event_type_packed: &[u8],
    event_id_deltas: &[u8],
    repo_bytes: &[u8],
    ts_bytes: &[u8],
) {
    eprintln!("\n=== Compression Strategy Experiments ===");

    // Raw sizes
    let columns: Vec<(&str, &[u8])> = vec![
        ("mapping_tsv", mapping_tsv),
        ("event_type_dict", event_type_dict),
        ("event_type_packed", event_type_packed),
        ("event_id_deltas", event_id_deltas),
        ("repo_pair_idx", repo_bytes),
        ("created_at_delta", ts_bytes),
    ];

    let total_raw: usize = columns.iter().map(|(_, d)| d.len()).sum();
    eprintln!("Total raw: {} bytes", total_raw);

    // Baseline: per-column with default settings
    let baseline: usize = columns.iter()
        .map(|(_, d)| zstd::encode_all(*d, ZSTD_LEVEL).unwrap().len())
        .sum();
    eprintln!("\nBaseline (per-column L22): {:>10} bytes", baseline);

    // Strategy 1: Window log experiments (per-column)
    eprintln!("\n1. Window log experiments (per-column):");
    for window_log in [17, 20, 23, 25, 27, 30] {
        let total: usize = columns.iter()
            .map(|(_, d)| compress_with_params(d, ZSTD_LEVEL, Some(window_log), false, false).len())
            .sum();
        eprintln!("   window_log={:<2}:          {:>10} bytes ({:+} vs baseline)",
            window_log, total, total as i64 - baseline as i64);
    }

    // Strategy 2: Pledged source size
    eprintln!("\n2. Pledged source size (per-column):");
    let with_pledged: usize = columns.iter()
        .map(|(_, d)| compress_with_params(d, ZSTD_LEVEL, None, true, false).len())
        .sum();
    eprintln!("   with pledged size:      {:>10} bytes ({:+} vs baseline)",
        with_pledged, with_pledged as i64 - baseline as i64);

    // Strategy 3: Checksum (should add overhead)
    eprintln!("\n3. Checksum overhead:");
    let with_checksum: usize = columns.iter()
        .map(|(_, d)| compress_with_params(d, ZSTD_LEVEL, None, false, true).len())
        .sum();
    eprintln!("   with checksum:          {:>10} bytes ({:+} vs baseline)",
        with_checksum, with_checksum as i64 - baseline as i64);

    // Strategy 4: Combined optimizations
    eprintln!("\n4. Combined parameter tuning:");
    for window_log in [23, 25, 27] {
        let total: usize = columns.iter()
            .map(|(_, d)| compress_with_params(d, ZSTD_LEVEL, Some(window_log), true, false).len())
            .sum();
        eprintln!("   wlog={} + pledged:      {:>10} bytes ({:+} vs baseline)",
            window_log, total, total as i64 - baseline as i64);
    }

    // Strategy 5: Per-column optimal window_log
    eprintln!("\n5. Per-column optimal window_log:");
    let mut optimized_total = 0usize;
    for (name, data) in &columns {
        let mut best_size = usize::MAX;
        let mut best_wlog = 0u32;
        for wlog in [17, 20, 23, 25, 27] {
            let size = compress_with_params(data, ZSTD_LEVEL, Some(wlog), true, false).len();
            if size < best_size {
                best_size = size;
                best_wlog = wlog;
            }
        }
        let default_size = zstd::encode_all(*data, ZSTD_LEVEL).unwrap().len();
        eprintln!("   {:<20} best wlog={:<2}: {:>8} bytes ({:+} vs default)",
            name, best_wlog, best_size, best_size as i64 - default_size as i64);
        optimized_total += best_size;
    }
    eprintln!("   Optimized total:        {:>10} bytes ({:+} vs baseline)",
        optimized_total, optimized_total as i64 - baseline as i64);

    // Strategy 6: Higher compression levels with larger window
    eprintln!("\n6. High compression with large window (single combined):");
    let mut combined = Vec::with_capacity(total_raw);
    for (_, data) in &columns {
        combined.extend_from_slice(data);
    }
    for level in [19, 22] {
        for wlog in [25, 27, 30] {
            let size = compress_with_params(&combined, level, Some(wlog), true, false).len();
            let baseline_combined = zstd::encode_all(combined.as_slice(), ZSTD_LEVEL).unwrap().len();
            eprintln!("   L{} wlog={}:             {:>10} bytes ({:+} vs L22 default)",
                level, wlog, size, size as i64 - baseline_combined as i64);
        }
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
        "  {name:<18} min={min:<12} max={max:<12} u8={fits_8:>5.1}% u16={fits_16:>5.1}% u32={fits_32:>5.1}%"
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
        "  {name:<18} min={min:<12} max={max:<12} i8={fits_8:>5.1}% i16={fits_16:>5.1}% i32={fits_32:>5.1}%"
    );
}

fn print_event_type_distribution(dict: &[u8], indices: &[u8]) {
    // Parse dictionary
    let dict_str = std::str::from_utf8(dict).unwrap();
    let event_types: Vec<&str> = dict_str.split('\n').collect();

    // Count frequencies
    let mut counts = vec![0usize; event_types.len()];
    for &idx in indices {
        counts[idx as usize] += 1;
    }

    // Sort by frequency descending
    let mut freq: Vec<(usize, &str, usize)> = counts
        .iter()
        .enumerate()
        .map(|(i, &c)| (i, event_types[i], c))
        .collect();
    freq.sort_by(|a, b| b.2.cmp(&a.2));

    eprintln!("\n=== Event Type Distribution ===");
    eprintln!("{:<5} {:<25} {:>10} {:>8}", "Idx", "Type", "Count", "%");
    eprintln!("{}", "-".repeat(52));
    let total = indices.len() as f64;
    for (idx, name, count) in &freq {
        eprintln!(
            "{:<5} {:<25} {:>10} {:>7.2}%",
            idx, name, count, *count as f64 / total * 100.0
        );
    }

    // Run-length analysis
    let mut runs: Vec<usize> = Vec::new();
    let mut current_run = 1usize;
    for i in 1..indices.len() {
        if indices[i] == indices[i - 1] {
            current_run += 1;
        } else {
            runs.push(current_run);
            current_run = 1;
        }
    }
    runs.push(current_run);

    let num_runs = runs.len();
    let avg_run = indices.len() as f64 / num_runs as f64;
    let max_run = *runs.iter().max().unwrap_or(&0);
    let runs_of_1 = runs.iter().filter(|&&r| r == 1).count();
    let runs_of_2_5 = runs.iter().filter(|&&r| (2..=5).contains(&r)).count();
    let runs_of_6_plus = runs.iter().filter(|&&r| r >= 6).count();

    eprintln!("\n=== Run-Length Analysis (event_type) ===");
    eprintln!("  Total runs:    {num_runs:>10}");
    eprintln!("  Avg run len:   {avg_run:>10.2}");
    eprintln!("  Max run len:   {max_run:>10}");
    eprintln!("  Runs of 1:     {:>10} ({:.1}%)", runs_of_1, runs_of_1 as f64 / num_runs as f64 * 100.0);
    eprintln!("  Runs of 2-5:   {:>10} ({:.1}%)", runs_of_2_5, runs_of_2_5 as f64 / num_runs as f64 * 100.0);
    eprintln!("  Runs of 6+:    {:>10} ({:.1}%)", runs_of_6_plus, runs_of_6_plus as f64 / num_runs as f64 * 100.0);

    // What would RLE cost?
    // Each run needs: 1 byte for type index + varint for length
    let rle_bytes: usize = runs.iter().map(|&r| {
        1 + if r < 128 { 1 } else if r < 16384 { 2 } else { 3 }
    }).sum();
    eprintln!("\n  RLE estimate:  {:>10} bytes (vs {} raw)", rle_bytes, indices.len());
    eprintln!("  RLE savings:   {:>10} bytes ({:.1}%)",
        indices.len() as i64 - rle_bytes as i64,
        (1.0 - rle_bytes as f64 / indices.len() as f64) * 100.0);

    // Entropy analysis (theoretical minimum bits)
    let entropy: f64 = freq.iter()
        .filter(|(_, _, c)| *c > 0)
        .map(|(_, _, c)| {
            let p = *c as f64 / total;
            -p * p.log2()
        })
        .sum();
    let entropy_bytes = (entropy * total) / 8.0;

    // Simple Huffman estimate: assign bit lengths based on frequency
    // Sort by frequency, assign codes using canonical Huffman-like approach
    let mut huffman_bits = 0.0;
    for (_, _, count) in freq.iter() {
        // Approximate bit length: ceil(log2(total/count)) but min 1
        let p = *count as f64 / total;
        let bits = if p >= 0.5 { 1.0 } else { (-p.log2()).ceil().min(8.0) };
        huffman_bits += *count as f64 * bits;
    }
    let huffman_bytes = huffman_bits / 8.0;

    eprintln!("\n=== Entropy Analysis (event_type) ===");
    eprintln!("  Shannon entropy: {entropy:.3} bits/symbol");
    eprintln!("  Theoretical min: {:>10.0} bytes ({:.2} B/row)", entropy_bytes, entropy_bytes / total);
    eprintln!("  Huffman estimate:{:>10.0} bytes ({:.2} B/row)", huffman_bytes, huffman_bytes / total);
    eprintln!("  4-bit packing:   {:>10} bytes ({:.2} B/row)", (indices.len() + 1) / 2, 0.5);
    eprintln!("  Current (zstd):  {:>10} bytes ({:.2} B/row)", 223132, 0.22);  // from debug output
}

fn print_repo_pair_idx_analysis(indices: &[u32]) {
    if indices.is_empty() {
        return;
    }

    let total = indices.len();
    let total_f = total as f64;

    // Frequency distribution
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for &idx in indices {
        *counts.entry(idx).or_insert(0) += 1;
    }

    let unique_repos = counts.len();
    let mut freq: Vec<(u32, usize)> = counts.into_iter().collect();
    freq.sort_by(|a, b| b.1.cmp(&a.1));

    eprintln!("\n=== Repo Pair Index Distribution ===");
    eprintln!("  Unique repos used: {} (of {} in mapping)", unique_repos, indices.iter().max().unwrap_or(&0) + 1);
    eprintln!("  Events per repo:   {:.2} avg", total_f / unique_repos as f64);

    // Top 10 repos
    eprintln!("\n  Top 10 repos by event count:");
    for (i, (idx, count)) in freq.iter().take(10).enumerate() {
        eprintln!("    {:>2}. idx={:<8} count={:<8} ({:.2}%)",
            i + 1, idx, count, *count as f64 / total_f * 100.0);
    }

    // Frequency buckets
    let repos_1_event = freq.iter().filter(|(_, c)| *c == 1).count();
    let repos_2_10 = freq.iter().filter(|(_, c)| *c >= 2 && *c <= 10).count();
    let repos_11_100 = freq.iter().filter(|(_, c)| *c >= 11 && *c <= 100).count();
    let repos_101_plus = freq.iter().filter(|(_, c)| *c > 100).count();

    eprintln!("\n  Repos by event count:");
    eprintln!("    1 event:     {:>8} repos ({:.1}%)", repos_1_event, repos_1_event as f64 / unique_repos as f64 * 100.0);
    eprintln!("    2-10:        {:>8} repos ({:.1}%)", repos_2_10, repos_2_10 as f64 / unique_repos as f64 * 100.0);
    eprintln!("    11-100:      {:>8} repos ({:.1}%)", repos_11_100, repos_11_100 as f64 / unique_repos as f64 * 100.0);
    eprintln!("    101+:        {:>8} repos ({:.1}%)", repos_101_plus, repos_101_plus as f64 / unique_repos as f64 * 100.0);

    // Run-length analysis
    let mut runs: Vec<usize> = Vec::new();
    let mut current_run = 1usize;
    for i in 1..indices.len() {
        if indices[i] == indices[i - 1] {
            current_run += 1;
        } else {
            runs.push(current_run);
            current_run = 1;
        }
    }
    runs.push(current_run);

    let num_runs = runs.len();
    let avg_run = total_f / num_runs as f64;
    let max_run = *runs.iter().max().unwrap_or(&0);
    let runs_of_1 = runs.iter().filter(|&&r| r == 1).count();

    eprintln!("\n=== Run-Length Analysis (repo_pair_idx) ===");
    eprintln!("  Total runs:    {num_runs:>10}");
    eprintln!("  Avg run len:   {avg_run:>10.2}");
    eprintln!("  Max run len:   {max_run:>10}");
    eprintln!("  Runs of 1:     {:>10} ({:.1}%)", runs_of_1, runs_of_1 as f64 / num_runs as f64 * 100.0);

    // Delta analysis (consecutive differences)
    let deltas: Vec<i64> = indices.windows(2)
        .map(|w| w[1] as i64 - w[0] as i64)
        .collect();

    if !deltas.is_empty() {
        let delta_min = *deltas.iter().min().unwrap();
        let delta_max = *deltas.iter().max().unwrap();
        let zeros = deltas.iter().filter(|&&d| d == 0).count();
        let fits_i8 = deltas.iter().filter(|&&d| (-128..=127).contains(&d)).count();
        let fits_i16 = deltas.iter().filter(|&&d| (-32768..=32767).contains(&d)).count();

        eprintln!("\n=== Delta Analysis (repo_pair_idx) ===");
        eprintln!("  Delta range:   {delta_min} to {delta_max}");
        eprintln!("  Zero deltas:   {:>10} ({:.1}%) [same repo]", zeros, zeros as f64 / deltas.len() as f64 * 100.0);
        eprintln!("  Fits in i8:    {:>10} ({:.1}%)", fits_i8, fits_i8 as f64 / deltas.len() as f64 * 100.0);
        eprintln!("  Fits in i16:   {:>10} ({:.1}%)", fits_i16, fits_i16 as f64 / deltas.len() as f64 * 100.0);

        // Estimate delta encoding size
        let delta_bytes: usize = deltas.iter().map(|&d| {
            if (-128..=127).contains(&d) { 1 }
            else if (-32768..=32767).contains(&d) { 2 }
            else { 4 }
        }).sum();
        eprintln!("  Delta estimate:{:>10} bytes (vs {} raw u32)", delta_bytes + 4, total * 4);
    }

    // Varint analysis for raw values
    let varint_bytes: usize = indices.iter().map(|&v| {
        if v < 128 { 1 }
        else if v < 16384 { 2 }
        else if v < 2097152 { 3 }
        else { 4 }
    }).sum();
    eprintln!("\n  Varint estimate: {:>8} bytes (vs {} raw u32)", varint_bytes, total * 4);
}

fn print_column_stats(columns: &EncodedColumns, mapping_tsv: &[u8], mapping_compressed_size: usize) {
    let num_rows = columns.event_id_deltas.len();  // 1 per event
    eprintln!("\n=== Per-Column Compressed Size Estimates ===");
    eprintln!("Total rows: {num_rows}");
    eprintln!("{:<20} {:>10} {:>10} {:>8} {:>10}", "Column", "Raw", "Zstd", "Ratio", "B/Row");
    eprintln!("{}", "-".repeat(64));

    let mut total_raw = 0usize;
    let mut total_compressed = 0usize;

    // event_type (dict + 4-bit packed indices)
    let dict_raw = columns.event_type_dict.len();
    let packed_raw = columns.event_type_packed.len();
    let event_type_raw = dict_raw + packed_raw;
    let mut event_type_buf = Vec::with_capacity(event_type_raw);
    event_type_buf.extend_from_slice(&columns.event_type_dict);
    event_type_buf.extend_from_slice(&columns.event_type_packed);
    let event_type_compressed = zstd::encode_all(event_type_buf.as_slice(), ZSTD_LEVEL).unwrap().len();
    total_raw += event_type_raw;
    total_compressed += event_type_compressed;
    eprintln!(
        "{:<20} {:>10} {:>10} {:>7.1}% {:>10.2}",
        "event_type (4-bit)",
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
    let mapping_tsv_raw = mapping_tsv.len();
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

    // Value statistics (unpack for analysis)
    let event_type_indices = unpack_nibbles(&columns.event_type_packed, num_rows);
    eprintln!("\n=== Value Statistics ===");
    print_value_stats_unsigned("event_type_idx", &event_type_indices.iter().map(|&v| v as u64).collect::<Vec<_>>());
    print_value_stats_unsigned("event_id_delta", &columns.event_id_deltas.iter().map(|&v| v as u64).collect::<Vec<_>>());
    print_value_stats_unsigned("repo_pair_idx", &columns.repo_pair_indices.iter().map(|&v| v as u64).collect::<Vec<_>>());
    print_value_stats_signed("created_at_delta", &columns.created_at_deltas.iter().map(|&v| v as i64).collect::<Vec<_>>());

    // Event type distribution and RLE analysis
    print_event_type_distribution(&columns.event_type_dict, &event_type_indices);

    // Repo pair index analysis
    print_repo_pair_idx_analysis(&columns.repo_pair_indices);
    eprintln!();
}
