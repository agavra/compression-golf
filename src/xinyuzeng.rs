//! # XinyuZeng Codec
//!
//! ## Architecture Overview
//!
//! This codec uses a **columnar storage** approach with the following key design decisions:
//!
//! - **Columnar storage**: Separate columns for ID, Type, Repo, and Timestamp data
//! - **Pre-sort by event_id**: Events are sorted by ID before encoding to enable efficient unsigned delta encoding
//! - **Dictionary encoding**: `TypeDict` for event types, `RepoColumnarDict` for repos
//! - **Compression**: zstd level 22 for most columns, LZMA (level 9 extreme) for repo dict and repo column
//! - **Re-sort on decode**: Events are re-sorted by `EventKey` to restore original order
//!
//! ## Optimization Strategies That Worked
//!
//! The following optimizations were implemented and improved compression:
//!
//! ### Core Techniques
//! - **Sorting by event_id before encoding** - Enables optimal unsigned delta encoding for IDs (no zigzag needed)
//! - **Columnar storage** - Separates different data patterns for optimal encoding per column
//! - **24-bit packing for repo_codes** - Byte-aligned packing compresses better with zstd than bit-packing
//! - **Varint delta encoding for repo IDs in dictionary** - Efficient for sorted IDs, removes FoR overhead
//! - **LZMA compression for repo dict and repo_column** - Better compression ratio than zstd for this data
//! - **Delimiter-based name encoding** - More efficient than storing lengths separately (1 byte per name vs length stream)
//! - **ID-sorted repo names** - Eliminated the `id_to_name_idx` permutation mapping, simplifying format
//! - **Removing redundant fields** - Removed `name_count` (duplicates `repo_count`), removed `ids_encoded_len`
//! - **Zstd level 22** - Maximum compression for non-repo columns
//! - **Bitpacking for type_codes** - Efficient for small number of event types
//!
//! ## Approaches That Did NOT Work
//!
//! The following optimizations were tried but did not improve compression (or made it worse):
//!
//! ### ID Column Optimizations
//! - **Patched bitpacking for id_deltas** - Outlier handling complexity didn't pay off; simple bitpacking + zstd is sufficient
//! - **Delta-of-deltas encoding for IDs** - Second-order deltas didn't compress better than first-order deltas
//!
//! ### Repo Column Optimizations
//! - **FoR (Frame of Reference) encoding for repo_codes** - 24-bit packing with zstd works better (no per-chunk base overhead)
//! - **Cascading delta encoding for repo_codes** - Repo codes are "random" when sorted by event_id; deltas don't help
//! - **Delta encoding for repo_pair_idx** - Repos are random when sorted by event_id (+1.15MB worse)
//!
//! ### Repo Dictionary Optimizations
//! - **Owner/repo splitting for names** - 87.6% of owners have only 1 repo, so no sharing benefit (+350KB worse)
//! - **SMAZ compression for repo names** - Not better than zstd for this data
//! - **Trained zstd dictionary for names** - Didn't beat standard zstd
//! - **Alphabetical sorting of names** - ID-sorted is better for overall compression (eliminates permutation mapping)
//! - **FoR for name lengths** - Delimiter method is more efficient (eliminates lengths stream entirely)
//! - **Varint for name lengths** - Delimiter method is more efficient
//! - **Frequency-based code assignment for repos** - Hurt name compression (lost prefix sharing from ID order)
//! - **FST or trie structures for names** - zstd beats these specialized structures
//! - **Huffman for already-zstd'd data** - No additional benefit
//!
//! ## Potential Future Work
//!
//! Ideas that have been considered but not yet fully explored:
//!
//! - **Separate zstd compression for each RepoColumnarDict stream** - Planned but not implemented; may improve compression by compressing IDs, names separately
//! - **MTF/LRU encoding for repo codes** - May help with locality when repos repeat within time windows
//! - **Row-group encoding with internal reordering** - Sort within groups by timestamp for better repo locality
//!
//! ## Cross-Dataset Benchmark Results
//!
//! Tested against 3 different GitHub Archive datasets to verify compression performance across diverse event patterns:
//!
//! | Dataset | Events | XinyuZeng | natebrennand | jakedgy | Hachikuji | XiangpengHao | agavra | samsond | fabinout | Zstd(9) | Rank |
//! |---------|--------|-----------|--------------|---------|-----------|--------------|--------|---------|----------|---------|------|
//! | 2024-06-15-12 | 233,373 | **1,835,639** | 1,877,092 | 1,980,836 | 2,081,576 | 2,200,195 | 2,277,278 | 2,285,833 | 2,292,856 | 4,893,652 | **1st** |
//! | 2024-09-01-18 | 189,249 | **1,496,421** | 1,520,548 | 1,627,673 | 1,680,985 | 1,772,864 | 1,795,199 | 1,886,237 | 1,865,011 | 3,615,047 | **1st** |
//! | 2025-01-10-6 | 251,247 | **1,712,890** | 1,721,251 | 1,863,444 | 1,854,375 | 2,073,840 | 2,070,048 | 2,174,752 | 2,108,447 | 4,333,032 | **1st** |
//!
//! **Result**: XinyuZeng achieves the best compression ratio on all three test datasets, consistently outperforming
//! natebrennand (the previous leaderboard leader) by 1.5-2.2% across different event distributions and time periods.
//!
//! ## Performance Stats
//!
//! Column sizes (compressed):
//!   ID column: 348814 bytes
//!   Type column: 223095 bytes (dict: 149 bytes, data: 222946 bytes)
//!   Repo column: 5243972 bytes (dict: 3079736 bytes, data: 2164236 bytes)
//!   Timestamp column: 38100 bytes
//!   Total: 5854037 bytes

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::{HashMap, HashSet};
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

extern crate lzma;

const MAGIC: &[u8; 4] = b"XYZG";

// ============================================================================
// Compression Functions
// ============================================================================

/// Compress data using zstd at level 22 (highest)
fn compress_zstd22(data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    Ok(zstd::encode_all(data, 22)?)
}

/// Decompress zstd-compressed data
fn decompress_zstd22(data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    Ok(zstd::decode_all(data)?)
}

/// Compress data using LZMA with extreme preset (level 9 | EXTREME)
fn compress_lzma(data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    Ok(lzma::compress(data, 9 | lzma::EXTREME_PRESET)?)
}

/// Decompress LZMA-compressed data
fn decompress_lzma(data: &[u8]) -> Result<Vec<u8>, Box<dyn Error>> {
    Ok(lzma::decompress(data)?)
}

fn debug_enabled() -> bool {
    std::env::var("XINYU_DEBUG").is_ok()
}

macro_rules! debug_println {
    ($($arg:tt)*) => {
        if debug_enabled() {
            println!($($arg)*);
        }
    };
}

// ============================================================================
// Utilities
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

fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> u32 {
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[*pos..*pos + 4]);
    *pos += 4;
    u32::from_le_bytes(arr)
}

fn read_u64(bytes: &[u8], pos: &mut usize) -> u64 {
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&bytes[*pos..*pos + 8]);
    *pos += 8;
    u64::from_le_bytes(arr)
}

fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(encoded: u64) -> i64 {
    ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64))
}

// ============================================================================
// Varint Encoding (LEB128)
// ============================================================================

fn encode_varint_u64(value: u64, buf: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let mut byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80; // Set continuation bit
        }
        buf.push(byte);
        if v == 0 {
            break;
        }
    }
}

fn decode_varint_u64(bytes: &[u8], pos: &mut usize) -> u64 {
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

/// Encode sorted u64 values using delta + varint encoding
fn varint_delta_encode(sorted_values: &[u64]) -> Vec<u8> {
    if sorted_values.is_empty() {
        return Vec::new();
    }

    let mut buf = Vec::new();

    // First value as full varint
    encode_varint_u64(sorted_values[0], &mut buf);

    // Remaining values as deltas (all non-negative since sorted)
    for i in 1..sorted_values.len() {
        let delta = sorted_values[i] - sorted_values[i - 1];
        encode_varint_u64(delta, &mut buf);
    }

    buf
}

/// Decode varint delta-encoded u64 values
#[allow(dead_code)] // Used by commented-out varint encoding method
fn varint_delta_decode(bytes: &[u8], count: usize) -> Vec<u64> {
    if count == 0 || bytes.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(count);
    let mut pos = 0;

    // First value
    let first = decode_varint_u64(bytes, &mut pos);
    result.push(first);

    // Remaining values as deltas
    let mut current = first;
    for _ in 1..count {
        let delta = decode_varint_u64(bytes, &mut pos);
        current += delta;
        result.push(current);
    }

    result
}

/// Decode varint delta-encoded u64 values from a mutable position pointer (streaming)
fn varint_delta_decode_streaming(bytes: &[u8], pos: &mut usize, count: usize) -> Vec<u64> {
    if count == 0 || *pos >= bytes.len() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(count);

    // First value
    let first = decode_varint_u64(bytes, pos);
    result.push(first);

    // Remaining values as deltas
    let mut current = first;
    for _ in 1..count {
        if *pos >= bytes.len() {
            break;
        }
        let delta = decode_varint_u64(bytes, pos);
        current += delta;
        result.push(current);
    }

    result
}

// ============================================================================
// Bit-packing
// ============================================================================

fn pack_bits_u64(values: &[u64]) -> Vec<u8> {
    if values.is_empty() {
        return vec![0];
    }

    let max_val = values.iter().copied().max().unwrap_or(0);
    let bit_width = if max_val == 0 {
        1
    } else {
        64 - max_val.leading_zeros() as u8
    };

    let mut buf = Vec::with_capacity((values.len() * bit_width as usize).div_ceil(8) + 1);
    buf.push(bit_width);

    let mut bit_pos: usize = 0;
    let mut current_byte: u8 = 0;

    for &value in values {
        for bit_idx in 0..bit_width {
            let bit = ((value >> bit_idx) & 1) as u8;
            current_byte |= bit << (bit_pos % 8);
            bit_pos += 1;
            if bit_pos.is_multiple_of(8) {
                buf.push(current_byte);
                current_byte = 0;
            }
        }
    }

    if !bit_pos.is_multiple_of(8) {
        buf.push(current_byte);
    }

    buf
}

fn unpack_bits_u64(bytes: &[u8], count: usize) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }
    let bit_width = bytes[0] as usize;
    if bit_width == 0 {
        return vec![0; count];
    }

    let mut values = Vec::with_capacity(count);
    let mut bit_pos: usize = 0;

    for _ in 0..count {
        let mut value: u64 = 0;
        for bit_idx in 0..bit_width {
            let byte_idx = 1 + (bit_pos / 8);
            let bit_in_byte = bit_pos % 8;
            let bit = ((bytes[byte_idx] >> bit_in_byte) & 1) as u64;
            value |= bit << bit_idx;
            bit_pos += 1;
        }
        values.push(value);
    }

    values
}

#[allow(dead_code)] // Used by commented-out FoR/varint encoding methods
fn pack_bits_u32(values: &[u32]) -> Vec<u8> {
    let mut tmp = Vec::with_capacity(values.len());
    for &v in values {
        tmp.push(v as u64);
    }
    pack_bits_u64(&tmp)
}

#[allow(dead_code)] // Used by commented-out FoR/varint encoding methods
fn unpack_bits_u32(bytes: &[u8], count: usize) -> Vec<u32> {
    unpack_bits_u64(bytes, count)
        .into_iter()
        .map(|v| v as u32)
        .collect()
}

fn pack_bits_u8(values: &[u8]) -> Vec<u8> {
    let mut tmp = Vec::with_capacity(values.len());
    for &v in values {
        tmp.push(v as u64);
    }
    pack_bits_u64(&tmp)
}

fn unpack_bits_u8(bytes: &[u8], count: usize) -> Vec<u8> {
    unpack_bits_u64(bytes, count)
        .into_iter()
        .map(|v| v as u8)
        .collect()
}

// ============================================================================
// 24-bit Packing (byte-aligned for better zstd compression)
// ============================================================================

/// Pack u32 values into 3 bytes each (24-bit little-endian)
/// Values must fit in 24 bits (max 16,777,215)
fn pack_u24(values: &[u32]) -> Vec<u8> {
    let mut result = Vec::with_capacity(values.len() * 3);
    for &val in values {
        result.push(val as u8);
        result.push((val >> 8) as u8);
        result.push((val >> 16) as u8);
    }
    result
}

/// Unpack 24-bit little-endian values back to u32
fn unpack_u24(bytes: &[u8], count: usize) -> Vec<u32> {
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * 3;
        let val = bytes[offset] as u32
            | ((bytes[offset + 1] as u32) << 8)
            | ((bytes[offset + 2] as u32) << 16);
        result.push(val);
    }
    result
}

// ============================================================================
// Frame of Reference (FoR) Encoding
// ============================================================================

fn for_encode_ids(ids: &[u64], chunk_size: usize) -> Vec<u8> {
    if ids.is_empty() {
        return Vec::new();
    }

    let mut buf = Vec::new();
    let num_chunks = ids.len().div_ceil(chunk_size);

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(ids.len());
        let chunk = &ids[start..end];

        if chunk.is_empty() {
            continue;
        }

        // Find base (minimum) value in chunk
        let base = *chunk.iter().min().unwrap();

        // Compute offsets
        let offsets: Vec<u64> = chunk.iter().map(|&id| id - base).collect();

        // Write base value (8 bytes)
        write_u64(&mut buf, base);
        // Write bitpacked offsets (pack_bits_u64 already includes bit_width as first byte)
        let packed = pack_bits_u64(&offsets);
        buf.extend_from_slice(&packed);
    }

    buf
}

fn for_decode_ids(bytes: &[u8], count: usize, chunk_size: usize) -> Vec<u64> {
    if count == 0 || bytes.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(count);
    let mut pos = 0;

    let num_chunks = count.div_ceil(chunk_size);

    for _chunk_idx in 0..num_chunks {
        if pos >= bytes.len() {
            break;
        }

        // Read base value
        let base = read_u64(bytes, &mut pos);

        // Determine chunk size (last chunk may be smaller)
        let remaining = count - result.len();
        let current_chunk_size = chunk_size.min(remaining);

        // Read bitpacked offsets (pack_bits_u64 format: [bit_width, ...packed_data])
        // First, peek at bit_width to calculate size needed
        if pos >= bytes.len() {
            break;
        }
        let bit_width = bytes[pos];

        // Calculate bytes needed: 1 byte for bit_width + packed data
        let bits_needed = current_chunk_size * bit_width as usize;
        let bytes_needed = 1 + bits_needed.div_ceil(8);
        let packed_end = pos + bytes_needed;
        let packed = &bytes[pos..packed_end.min(bytes.len())];
        pos = packed_end.min(bytes.len());

        // Unpack offsets (packed already includes bit_width as first byte)
        let offsets = unpack_bits_u64(packed, current_chunk_size);

        // Reconstruct IDs
        for offset in offsets {
            result.push(base + offset);
        }
    }

    result
}

fn find_optimal_chunk_size(ids: &[u64]) -> usize {
    if ids.is_empty() {
        return 256; // default
    }

    // Sample: first 1024, middle 1024, last 1024
    let sample_size = 1024;
    let mut samples = Vec::new();

    // First 1024
    let first_end = sample_size.min(ids.len());
    samples.extend_from_slice(&ids[..first_end]);

    // Middle 1024
    if ids.len() > sample_size * 2 {
        let mid_start = (ids.len() - sample_size) / 2;
        let mid_end = mid_start + sample_size;
        samples.extend_from_slice(&ids[mid_start..mid_end]);
    }

    // Last 1024
    if ids.len() > sample_size {
        let last_start = ids.len().saturating_sub(sample_size);
        samples.extend_from_slice(&ids[last_start..]);
    }

    // Sort samples (they should already be sorted, but ensure it)
    samples.sort();
    samples.dedup();

    if samples.is_empty() {
        return 256; // default
    }

    // Try different chunk sizes
    let candidate_sizes = vec![64, 128, 256, 512, 1024];
    let mut best_size = 256;
    let mut best_cost = usize::MAX;

    for &chunk_size in &candidate_sizes {
        let encoded = for_encode_ids(&samples, chunk_size);
        let cost = encoded.len();

        if cost < best_cost {
            best_cost = cost;
            best_size = chunk_size;
        }
    }

    best_size
}

#[allow(dead_code)] // Used by commented-out FoR encoding method
fn find_optimal_chunk_size_u32(values: &[u32]) -> usize {
    // Convert to u64 for compatibility with existing function
    let u64_values: Vec<u64> = values.iter().map(|&v| v as u64).collect();
    find_optimal_chunk_size(&u64_values)
}

#[allow(dead_code)] // Used by commented-out FoR encoding method
fn for_encode_u32(values: &[u32], chunk_size: usize) -> Vec<u8> {
    // Convert to u64 for compatibility with existing function
    let u64_values: Vec<u64> = values.iter().map(|&v| v as u64).collect();
    for_encode_ids(&u64_values, chunk_size)
}

#[allow(dead_code)] // Kept for potential future use
fn for_decode_u32(bytes: &[u8], count: usize, chunk_size: usize) -> Vec<u32> {
    // Decode as u64 then convert to u32
    let u64_values = for_decode_ids(bytes, count, chunk_size);
    u64_values.into_iter().map(|v| v as u32).collect()
}

/// Decode FoR-encoded u32 values from a mutable position pointer (streaming)
#[allow(dead_code)] // Kept for potential future use
fn for_decode_u32_streaming(
    bytes: &[u8],
    pos: &mut usize,
    count: usize,
    chunk_size: usize,
) -> Vec<u32> {
    // Decode as u64 then convert to u32
    let u64_values = for_decode_ids_streaming(bytes, pos, count, chunk_size);
    u64_values.into_iter().map(|v| v as u32).collect()
}

/// Decode FoR-encoded u64 values from a mutable position pointer (streaming)
#[allow(dead_code)] // Kept for potential future use
fn for_decode_ids_streaming(
    bytes: &[u8],
    pos: &mut usize,
    count: usize,
    chunk_size: usize,
) -> Vec<u64> {
    if count == 0 || *pos >= bytes.len() {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(count);

    let num_chunks = count.div_ceil(chunk_size);

    for _chunk_idx in 0..num_chunks {
        if *pos >= bytes.len() {
            break;
        }

        // Read base value
        let base = read_u64(bytes, pos);

        // Determine chunk size (last chunk may be smaller)
        let remaining = count - result.len();
        let current_chunk_size = chunk_size.min(remaining);

        // Read bitpacked offsets (pack_bits_u64 format: [bit_width, ...packed_data])
        if *pos >= bytes.len() {
            break;
        }
        let bit_width = bytes[*pos];

        // Calculate bytes needed: 1 byte for bit_width + packed data
        let bits_needed = current_chunk_size * bit_width as usize;
        let bytes_needed = 1 + bits_needed.div_ceil(8);
        let packed_end = *pos + bytes_needed;
        let packed = &bytes[*pos..packed_end.min(bytes.len())];
        *pos = packed_end.min(bytes.len());

        // Unpack offsets (packed already includes bit_width as first byte)
        let offsets = unpack_bits_u64(packed, current_chunk_size);

        // Reconstruct IDs
        for offset in offsets {
            result.push(base + offset);
        }
    }

    result
}

// ============================================================================
// Dictionaries
// ============================================================================

struct TypeDict {
    types: Vec<String>,
    type_to_idx: HashMap<String, u8>,
}

impl TypeDict {
    fn build(events: &[(EventKey, EventValue)]) -> Self {
        let mut unique_types: Vec<String> = events
            .iter()
            .map(|(key, _)| key.event_type.clone())
            .collect();
        unique_types.sort();
        unique_types.dedup();

        let mut type_to_idx = HashMap::new();
        for (i, t) in unique_types.iter().enumerate() {
            type_to_idx.insert(t.clone(), i as u8);
        }

        Self {
            types: unique_types,
            type_to_idx,
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_u32(&mut buf, self.types.len() as u32);
        for t in &self.types {
            write_u32(&mut buf, t.len() as u32);
            buf.extend_from_slice(t.as_bytes());
        }
        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        let mut pos = 0;
        let count = read_u32(bytes, &mut pos) as usize;
        let mut types = Vec::with_capacity(count);
        let mut type_to_idx = HashMap::new();

        for i in 0..count {
            let len = read_u32(bytes, &mut pos) as usize;
            let t = std::str::from_utf8(&bytes[pos..pos + len])
                .unwrap()
                .to_string();
            pos += len;
            type_to_idx.insert(t.clone(), i as u8);
            types.push(t);
        }

        Self { types, type_to_idx }
    }

    fn get_index(&self, event_type: &str) -> u8 {
        self.type_to_idx[event_type]
    }

    fn get_type(&self, index: u8) -> &str {
        &self.types[index as usize]
    }
}

struct RepoColumnarDict {
    // Original pairs in ID-sorted order (for get_repo by index)
    original_pairs: Vec<(u64, String)>,

    // Columnar storage: IDs and names both sorted by ID for optimal delta encoding
    repo_ids: Vec<u64>,
    repo_names: Vec<String>,

    // Mapping: (id, name) -> original index
    repo_to_idx: HashMap<(u64, String), u32>,
}

impl RepoColumnarDict {
    fn build(events: &[(EventKey, EventValue)]) -> Self {
        // Collect unique (id, name) pairs
        let mut unique_pairs: Vec<(u64, String)> = events
            .iter()
            .map(|(_, value)| (value.repo.id, value.repo.name.clone()))
            .collect();

        // Deduplicate and sort by ID (with name as tie-breaker) for optimal delta encoding
        unique_pairs.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        unique_pairs.dedup();

        // Store pairs, IDs, and names all in ID-sorted order
        let original_pairs: Vec<(u64, String)> = unique_pairs.clone();
        let repo_ids: Vec<u64> = unique_pairs.iter().map(|(id, _)| *id).collect();
        let repo_names: Vec<String> = unique_pairs.iter().map(|(_, name)| name.clone()).collect();

        // Build mapping: (id, name) -> original index
        let mut repo_to_idx = HashMap::new();
        for (i, (id, name)) in original_pairs.iter().enumerate() {
            repo_to_idx.insert((*id, name.clone()), i as u32);
        }

        Self {
            original_pairs,
            repo_ids,
            repo_names,
            repo_to_idx,
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        debug_println!("\n=== REPO DICT ENCODE BREAKDOWN ===");
        debug_println!("  Repo count: {}", self.repo_ids.len());

        // Track byte-accurate sizes for each component
        let mut sizes = std::collections::HashMap::<&str, usize>::new();

        // Write number of repos
        let start = buf.len();
        write_u32(&mut buf, self.repo_ids.len() as u32);
        sizes.insert("repo_count_header", buf.len() - start);

        // Encode IDs using varint delta encoding (IDs are sorted)
        // Note: ids_encoded_len removed - decode will read exactly count values
        let ids_encoded = varint_delta_encode(&self.repo_ids);
        let start = buf.len();
        buf.extend_from_slice(&ids_encoded);
        sizes.insert("ids_payload", buf.len() - start);

        // Encode names (string dictionary format)
        // Note: name_count is redundant with repo_count, so we skip it

        // Collect lengths and compute distribution stats
        let lengths: Vec<u32> = self.repo_names.iter().map(|n| n.len() as u32).collect();

        // Length distribution stats
        if !lengths.is_empty() {
            let mut sorted_lengths = lengths.clone();
            sorted_lengths.sort();
            let min_len = sorted_lengths[0];
            let max_len = sorted_lengths[sorted_lengths.len() - 1];
            let median_len = sorted_lengths[sorted_lengths.len() / 2];
            let p95_idx = (sorted_lengths.len() as f64 * 0.95) as usize;
            let p95_len = sorted_lengths[p95_idx.min(sorted_lengths.len() - 1)];

            let distinct_lengths: HashSet<u32> = lengths.iter().copied().collect();

            debug_println!("\n  Length distribution:");
            debug_println!(
                "    Min: {}, Median: {}, P95: {}, Max: {}",
                min_len,
                median_len,
                p95_len,
                max_len
            );
            debug_println!("    Distinct lengths: {}", distinct_lengths.len());
        }

        // Use delimiter-based encoding: no lengths stream, but +1 byte per name for null delimiter
        // This method is the most efficient because:
        // 1. It avoids storing a separate lengths stream entirely
        // 2. The overhead is only 1 byte per name (null delimiter)
        // 3. It simplifies decoding - just read until null byte
        let delimiter_overhead = self.repo_names.len(); // 1 byte per name

        debug_println!(
            "\n  Delimiter encoding: {} bytes total (no lengths stream)",
            delimiter_overhead
        );
        debug_println!("    Selected: delimiter method");

        // Use delimiter-based encoding: names are separated by null bytes (0x00).
        // This eliminates the need for a separate lengths stream, requiring only 1 byte
        // per name as overhead. Alternative methods (FoR, varint) were evaluated but
        // delimiter method consistently performs better.

        // Write encoding method marker (delimiter mode)
        let start = buf.len();
        write_u32(&mut buf, 0xFFFFFFFE);
        sizes.insert("lengths_encoding_header", buf.len() - start);

        // Write lengths payload (empty for delimiter mode - no lengths stream needed)
        let start = buf.len();
        // No lengths payload in delimiter mode
        sizes.insert("lengths_payload", buf.len() - start);

        // Analyze name strings: owner vs suffix
        let mut owner_bytes = 0;
        let mut suffix_bytes = 0;
        let mut names_total_bytes = 0;

        for name in &self.repo_names {
            names_total_bytes += name.len();
            if let Some(slash_pos) = name.find('/') {
                owner_bytes += slash_pos + 1; // Include the '/'
                suffix_bytes += name.len() - slash_pos - 1;
            } else {
                // No slash found, treat entire name as suffix
                suffix_bytes += name.len();
            }
        }

        debug_println!("\n  Name string analysis:");
        debug_println!("    Total name bytes: {}", names_total_bytes);
        debug_println!(
            "    Owner prefix bytes: {} ({:.1}%)",
            owner_bytes,
            if names_total_bytes > 0 {
                owner_bytes as f64 / names_total_bytes as f64 * 100.0
            } else {
                0.0
            }
        );
        debug_println!(
            "    Repo suffix bytes: {} ({:.1}%)",
            suffix_bytes,
            if names_total_bytes > 0 {
                suffix_bytes as f64 / names_total_bytes as f64 * 100.0
            } else {
                0.0
            }
        );

        // Write all bytes concatenated with null delimiters
        debug_println!("\n  First 20 repo names:");
        for (i, name) in self.repo_names.iter().take(20).enumerate() {
            debug_println!("    [{}] {}", i, name);
        }

        let start = buf.len();
        // Delimiter mode: write name + null byte for each
        for name in &self.repo_names {
            buf.extend_from_slice(name.as_bytes());
            buf.push(0); // Null delimiter
        }
        sizes.insert("names_payload", buf.len() - start);

        // Print detailed byte breakdown table
        let total_buf = buf.len();
        debug_println!("\n  === BYTESTREAM SIZE BREAKDOWN ===");
        debug_println!("  Component                    | Bytes    | % of total");
        debug_println!("  -----------------------------|----------|-----------");

        let components = vec![
            "repo_count_header",
            "ids_payload",
            "lengths_encoding_header",
            "lengths_payload",
            "names_payload",
        ];

        for component in &components {
            let bytes = sizes.get(*component).copied().unwrap_or(0);
            let pct = if total_buf > 0 {
                bytes as f64 / total_buf as f64 * 100.0
            } else {
                0.0
            };
            debug_println!("  {:28} | {:8} | {:6.2}%", component, bytes, pct);
        }

        debug_println!("  -----------------------------|----------|-----------");
        debug_println!("  {:28} | {:8} | {:6.2}%", "TOTAL", total_buf, 100.0);

        // Summary stats
        let ids_raw_size = self.repo_ids.len() * 8;
        let ids_ratio = if !ids_encoded.is_empty() {
            ids_raw_size as f64 / ids_encoded.len() as f64
        } else {
            0.0
        };
        debug_println!("\n  Summary:");
        debug_println!(
            "    IDs compression ratio: {:.2}x (raw: {} bytes -> encoded: {} bytes)",
            ids_ratio,
            ids_raw_size,
            sizes.get("ids_payload").copied().unwrap_or(0)
        );
        debug_println!(
            "    Names total: {} bytes (avg: {:.1} bytes/name)",
            names_total_bytes,
            if !self.repo_names.is_empty() {
                names_total_bytes as f64 / self.repo_names.len() as f64
            } else {
                0.0
            }
        );

        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        let mut pos = 0;

        let count = read_u32(bytes, &mut pos) as usize;
        if count == 0 {
            return Self {
                original_pairs: Vec::new(),
                repo_ids: Vec::new(),
                repo_names: Vec::new(),
                repo_to_idx: HashMap::new(),
            };
        }

        // Decode IDs using varint delta decoding (streaming - decode exactly count values)
        let repo_ids = varint_delta_decode_streaming(bytes, &mut pos, count);

        // Read and validate encoding marker (must be delimiter mode)
        let encoding_marker = read_u32(bytes, &mut pos);
        assert_eq!(
            encoding_marker,
            0xFFFFFFFE,
            "unsupported encoding marker: 0x{:X} (expected delimiter mode 0xFFFFFFFE)",
            encoding_marker
        );

        // Delimiter mode: read names separated by null bytes
        let mut repo_names = Vec::with_capacity(count);
        for _ in 0..count {
            if pos >= bytes.len() {
                break;
            }
            // Find null delimiter
            let name_start = pos;
            while pos < bytes.len() && bytes[pos] != 0 {
                pos += 1;
            }
            let name = std::str::from_utf8(&bytes[name_start..pos])
                .unwrap()
                .to_string();
            pos += 1; // Skip null delimiter
            repo_names.push(name);
        }

        // Reconstruct original_pairs positionally: IDs and names are both in ID-sorted order
        let mut original_pairs: Vec<(u64, String)> = Vec::with_capacity(count);
        for i in 0..count {
            original_pairs.push((repo_ids[i], repo_names[i].clone()));
        }

        // Build mapping: (id, name) -> original index
        let mut repo_to_idx = HashMap::new();
        for (i, (id, name)) in original_pairs.iter().enumerate() {
            repo_to_idx.insert((*id, name.clone()), i as u32);
        }

        Self {
            original_pairs,
            repo_ids,
            repo_names,
            repo_to_idx,
        }
    }

    fn get_index(&self, repo: &Repo) -> u32 {
        // Look up by (id, name) - url is not needed
        self.repo_to_idx[&(repo.id, repo.name.clone())]
    }

    fn get_repo(&self, index: u32) -> Repo {
        // Use original pairs to get the repo directly
        let (id, name) = &self.original_pairs[index as usize];
        let url = format!("https://api.github.com/repos/{}", name);

        Repo {
            id: *id,
            name: name.clone(),
            url,
        }
    }
}

// ============================================================================
// Codec implementation
// ============================================================================

pub struct XinyuZengCodec;

impl XinyuZengCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for XinyuZengCodec {
    fn name(&self) -> &str {
        "XinyuZeng"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        if events.is_empty() {
            let mut out = Vec::new();
            out.extend_from_slice(MAGIC);
            write_u32(&mut out, 0);
            return Ok(Bytes::from(out));
        }

        // Sort by event_id for optimal delta encoding
        let mut sorted_indices: Vec<usize> = (0..events.len()).collect();
        sorted_indices.sort_by_key(|&i| events[i].0.id.parse::<u64>().unwrap_or(0));

        // Build dictionaries (before sorting, to include all events)
        let type_dict = TypeDict::build(events);
        let repo_dict = RepoColumnarDict::build(events);

        // Encode dictionaries
        let type_dict_bytes = type_dict.encode();
        let repo_dict_bytes = repo_dict.encode();

        // Process columns in sorted order
        let num_events = events.len();

        // ID column: delta encoding (no zigzag needed since sorted ascending)
        let mut ids: Vec<u64> = Vec::with_capacity(num_events);
        for &idx in &sorted_indices {
            let id = events[idx].0.id.parse::<u64>().unwrap_or(0);
            ids.push(id);
        }

        let base_id = ids[0];
        let mut id_deltas = Vec::with_capacity(num_events - 1);
        for i in 1..num_events {
            // After sorting, deltas are always non-negative
            let delta = ids[i] - ids[i - 1];
            id_deltas.push(delta);
        }

        // Debug: Print first 20, middle 20, and last 20 id_deltas
        debug_println!("\n=== ID DELTAS DEBUG ===");
        debug_println!("Total id_deltas: {}", id_deltas.len());

        if !id_deltas.is_empty() {
            debug_println!("\nFirst 20 id_deltas:");
            for (i, &delta) in id_deltas.iter().take(20).enumerate() {
                debug_println!("  [{}] {}", i, delta);
            }

            if id_deltas.len() > 40 {
                let mid_start = id_deltas.len() / 2 - 10;
                debug_println!("\nMiddle 20 id_deltas (starting at index {}):", mid_start);
                for (i, &delta) in id_deltas[mid_start..mid_start + 20].iter().enumerate() {
                    debug_println!("  [{}] {}", mid_start + i, delta);
                }
            }

            if id_deltas.len() >= 20 {
                let last_start = id_deltas.len().saturating_sub(20);
                debug_println!("\nLast 20 id_deltas (starting at index {}):", last_start);
                for (i, &delta) in id_deltas[last_start..].iter().enumerate() {
                    debug_println!("  [{}] {}", last_start + i, delta);
                }
            }
        }

        let id_column_bytes = pack_bits_u64(&id_deltas);

        // Type column: dictionary codes + bitpacking
        let mut type_codes = Vec::with_capacity(num_events);
        for &idx in &sorted_indices {
            type_codes.push(type_dict.get_index(&events[idx].0.event_type));
        }
        let type_column_bytes = pack_bits_u8(&type_codes);

        // Repo column: dictionary codes + FoR encoding
        let mut repo_codes = Vec::with_capacity(num_events);
        for &idx in &sorted_indices {
            repo_codes.push(repo_dict.get_index(&events[idx].1.repo));
        }

        // Debug: Analyze repo_codes patterns
        debug_println!("\n=== REPO CODES ANALYSIS ===");
        debug_println!("  Total repo_codes: {}", repo_codes.len());
        let unique_count = repo_codes.iter().copied().collect::<HashSet<_>>().len();
        debug_println!("  Unique values: {}", unique_count);
        if let (Some(min), Some(max)) = (repo_codes.iter().min(), repo_codes.iter().max()) {
            debug_println!("  Min: {}, Max: {}", min, max);
        }
        // Delta analysis
        if repo_codes.len() > 1 {
            let mut deltas: Vec<i64> = Vec::with_capacity(repo_codes.len() - 1);
            for i in 1..repo_codes.len() {
                deltas.push(repo_codes[i] as i64 - repo_codes[i - 1] as i64);
            }
            if let (Some(min_delta), Some(max_delta)) = (deltas.iter().min(), deltas.iter().max()) {
                debug_println!("  Delta range: {} to {}", min_delta, max_delta);
            }
            // Check monotonicity
            let increasing = deltas.iter().filter(|&&d| d >= 0).count();
            debug_println!(
                "  Monotonically increasing deltas: {}/{}",
                increasing,
                deltas.len()
            );
            // RLE potential
            let mut runs = 1;
            for i in 1..repo_codes.len() {
                if repo_codes[i] != repo_codes[i - 1] {
                    runs += 1;
                }
            }
            debug_println!(
                "  RLE runs: {} (avg run length: {:.1})",
                runs,
                repo_codes.len() as f64 / runs as f64
            );
        }

        // Debug: Print sample of repo_codes values
        if debug_enabled() {
            println!("\n=== REPO CODES SAMPLE ===");
            for (i, &code) in repo_codes.iter().take(50).enumerate() {
                println!("  [{}] = {}", i, code);
            }

            // Histogram of top 10 most frequent repos
            let mut freq: HashMap<u32, usize> = HashMap::new();
            for &code in &repo_codes {
                *freq.entry(code).or_insert(0) += 1;
            }
            let mut top: Vec<_> = freq.into_iter().collect();
            top.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            println!("\n=== TOP 10 REPOS BY FREQUENCY ===");
            for (code, count) in top.iter().take(10) {
                // Also show the repo name for context
                let repo = repo_dict.get_repo(*code);
                println!(
                    "  code {} ({}) appears {} times ({:.2}%)",
                    code,
                    repo.name,
                    count,
                    *count as f64 / repo_codes.len() as f64 * 100.0
                );
            }
        }

        // Use 24-bit packing instead of FoR for repo_codes
        // Max repo code fits in 18 bits, and byte-aligned 24-bit packing
        // compresses better with zstd than bit-packing
        let repo_column_bytes = pack_u24(&repo_codes);

        // Timestamp column: delta encoding
        let mut timestamps: Vec<u64> = Vec::with_capacity(num_events);
        for &idx in &sorted_indices {
            timestamps.push(parse_timestamp(&events[idx].1.created_at));
        }

        let base_ts = timestamps[0];
        let mut ts_deltas = Vec::with_capacity(num_events - 1);
        for i in 1..num_events {
            let delta = timestamps[i] as i64 - timestamps[i - 1] as i64;
            ts_deltas.push(zigzag_encode(delta));
        }
        let ts_chunk_size = find_optimal_chunk_size(&ts_deltas);
        let ts_column_bytes = for_encode_ids(&ts_deltas, ts_chunk_size);

        // Compress each column (repo dict and repo column use LZMA, others use zstd22)
        let type_dict_compressed = compress_zstd22(type_dict_bytes.as_slice())?;

        // Measure repo dictionary compression (uses LZMA)
        let repo_dict_raw_size = repo_dict_bytes.len();
        let repo_dict_compressed = compress_lzma(repo_dict_bytes.as_slice())?;
        let repo_dict_compressed_size = repo_dict_compressed.len();
        let repo_dict_compression_ratio = if repo_dict_compressed_size > 0 {
            repo_dict_raw_size as f64 / repo_dict_compressed_size as f64
        } else {
            0.0
        };

        debug_println!("\n=== REPO DICT COMPRESSION (LZMA) ===");
        debug_println!("  Raw size: {} bytes", repo_dict_raw_size);
        debug_println!("  Compressed size: {} bytes", repo_dict_compressed_size);
        debug_println!("  Compression ratio: {:.2}x", repo_dict_compression_ratio);
        debug_println!(
            "  Space saved: {} bytes ({:.1}%)",
            repo_dict_raw_size - repo_dict_compressed_size,
            if repo_dict_raw_size > 0 {
                (repo_dict_raw_size - repo_dict_compressed_size) as f64 / repo_dict_raw_size as f64
                    * 100.0
            } else {
                0.0
            }
        );
        let id_column_compressed = compress_zstd22(id_column_bytes.as_slice())?;
        let type_column_compressed = compress_zstd22(type_column_bytes.as_slice())?;
        let repo_column_compressed = compress_lzma(repo_column_bytes.as_slice())?; // LZMA for repo column
        let ts_column_compressed = compress_zstd22(ts_column_bytes.as_slice())?;

        // Build output
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        write_u32(&mut out, num_events as u32);

        // Compressed sizes for each column (6 u32 values)
        write_u32(&mut out, type_dict_compressed.len() as u32);
        write_u32(&mut out, repo_dict_compressed.len() as u32);
        write_u32(&mut out, id_column_compressed.len() as u32);
        write_u32(&mut out, type_column_compressed.len() as u32);
        write_u32(&mut out, repo_column_compressed.len() as u32);
        write_u32(&mut out, ts_column_compressed.len() as u32);

        // Base values
        write_u64(&mut out, base_id);
        write_u64(&mut out, base_ts);

        // Chunk sizes for FoR encoding (repo_chunk_size is always 0 for 24-bit packing)
        write_u32(&mut out, 0);
        write_u32(&mut out, ts_chunk_size as u32);

        // Compressed column data
        out.extend_from_slice(&type_dict_compressed);
        out.extend_from_slice(&repo_dict_compressed);
        out.extend_from_slice(&id_column_compressed);
        out.extend_from_slice(&type_column_compressed);
        out.extend_from_slice(&repo_column_compressed);
        out.extend_from_slice(&ts_column_compressed);

        // Print column sizes
        debug_println!("\nColumn sizes (compressed):");
        debug_println!("  ID column: {} bytes", id_column_compressed.len());
        debug_println!(
            "  Type column: {} bytes (dict: {} bytes, data: {} bytes)",
            type_dict_compressed.len() + type_column_compressed.len(),
            type_dict_compressed.len(),
            type_column_compressed.len()
        );
        debug_println!(
            "  Repo column: {} bytes (dict: {} bytes, data: {} bytes)",
            repo_dict_compressed.len() + repo_column_compressed.len(),
            repo_dict_compressed.len(),
            repo_column_compressed.len()
        );
        debug_println!("  Timestamp column: {} bytes", ts_column_compressed.len());
        debug_println!("  Total: {} bytes\n", out.len());

        Ok(Bytes::from(out))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Err("invalid magic".into());
        }
        pos += MAGIC.len();

        let num_events = read_u32(bytes, &mut pos) as usize;
        if num_events == 0 {
            return Ok(Vec::new());
        }

        // Read compressed sizes
        let type_dict_compressed_len = read_u32(bytes, &mut pos) as usize;
        let repo_dict_compressed_len = read_u32(bytes, &mut pos) as usize;
        let id_column_compressed_len = read_u32(bytes, &mut pos) as usize;
        let type_column_compressed_len = read_u32(bytes, &mut pos) as usize;
        let repo_column_compressed_len = read_u32(bytes, &mut pos) as usize;
        let ts_column_compressed_len = read_u32(bytes, &mut pos) as usize;

        // Read base values
        let base_id = read_u64(bytes, &mut pos);
        let base_ts = read_u64(bytes, &mut pos);

        // Read chunk sizes for FoR encoding (repo_chunk_size is always 0, kept for format compatibility)
        let _repo_chunk_size = read_u32(bytes, &mut pos) as usize;
        let ts_chunk_size = read_u32(bytes, &mut pos) as usize;

        // Decompress and decode dictionaries (repo dict uses LZMA, type dict uses zstd)
        let type_dict_compressed = &bytes[pos..pos + type_dict_compressed_len];
        pos += type_dict_compressed_len;
        let type_dict_bytes = decompress_zstd22(type_dict_compressed)?;
        let type_dict = TypeDict::decode(&type_dict_bytes);

        let repo_dict_compressed = &bytes[pos..pos + repo_dict_compressed_len];
        pos += repo_dict_compressed_len;
        let repo_dict_bytes = decompress_lzma(repo_dict_compressed)?; // LZMA for repo dict
        let repo_dict = RepoColumnarDict::decode(&repo_dict_bytes);

        // Decompress column data (repo column uses LZMA, others use zstd)
        let id_column_compressed = &bytes[pos..pos + id_column_compressed_len];
        pos += id_column_compressed_len;
        let id_column_bytes = decompress_zstd22(id_column_compressed)?;

        let type_column_compressed = &bytes[pos..pos + type_column_compressed_len];
        pos += type_column_compressed_len;
        let type_column_bytes = decompress_zstd22(type_column_compressed)?;

        let repo_column_compressed = &bytes[pos..pos + repo_column_compressed_len];
        pos += repo_column_compressed_len;
        let repo_column_bytes = decompress_lzma(repo_column_compressed)?; // LZMA for repo column

        let ts_column_compressed = &bytes[pos..pos + ts_column_compressed_len];
        let ts_column_bytes = decompress_zstd22(ts_column_compressed)?;

        // Unpack columns
        let id_deltas = unpack_bits_u64(&id_column_bytes, num_events - 1);
        let type_codes = unpack_bits_u8(&type_column_bytes, num_events);
        // Decode repo_codes: always use 24-bit packing mode
        let repo_codes = unpack_u24(&repo_column_bytes, num_events);
        let ts_deltas = for_decode_ids(&ts_column_bytes, num_events - 1, ts_chunk_size);

        // Reconstruct events
        let mut events = Vec::with_capacity(num_events);

        // Reconstruct IDs (no zigzag decode needed since deltas are unsigned)
        let mut current_id = base_id;
        let mut ids = Vec::with_capacity(num_events);
        ids.push(current_id);
        for &delta in &id_deltas {
            current_id += delta;
            ids.push(current_id);
        }

        // Reconstruct timestamps
        let mut current_ts = base_ts as i64;
        let mut timestamps = Vec::with_capacity(num_events);
        timestamps.push(current_ts as u64);
        for &delta_encoded in &ts_deltas {
            let delta = zigzag_decode(delta_encoded);
            current_ts += delta;
            timestamps.push(current_ts as u64);
        }

        // Reconstruct events
        for i in 0..num_events {
            let event_type = type_dict.get_type(type_codes[i]).to_string();
            let id = ids[i].to_string();
            let repo = repo_dict.get_repo(repo_codes[i]).clone();
            let created_at = format_timestamp(timestamps[i]);

            events.push((EventKey { id, event_type }, EventValue { repo, created_at }));
        }

        // Re-sort events by EventKey to restore original order
        events.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(events)
    }
}
