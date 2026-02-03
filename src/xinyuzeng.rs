//! # Xinyuzeng Codec
//!
//! Columnar layout with dictionary-encoded repos, limited-alphabet MTF for indices,
//! byte-plane splitting, delta encoding for IDs/timestamps, and per-column zstd compression.
//!
//! Successful optimizations:
//! - Split base from id_deltas: 6,283,516 → 6,214,202 bytes (-69KB, -1.1%)
//!   id_deltas: 533,986 → 464,672 (-69KB)
//! - Remove zigzag for id_deltas: 6,214,202 → 6,098,398 bytes (-116KB, -1.9%)
//!   id_deltas: 464,672 → 348,868 (-116KB). Since IDs are sorted, deltas are
//!   always non-negative, so zigzag wastes 1 bit per value.
//! - Byte plane splitting for repo_ids: 6,098,398 → 6,006,181 bytes (-92KB, -1.5%)
//!   repo_ids: 487,627 → 395,410 (-92KB). Split 3-byte deltas into separate
//!   high/mid/low streams for better zstd compression.
//! - MTF + byte planes for repo_name_idx: 6,006,181 → 5,723,601 bytes (-283KB, -4.7%)
//!   repo_name_idx: 2,232,905 → 1,950,325 (-283KB). Move-to-Front transform
//!   exploits temporal locality, then byte planes help compress high bytes
//!   (29% of MTF values < 256, 69% < 65536).
//! - Limited-alphabet MTF: 5,723,601 → 5,784,824 bytes (+61KB, +1.1%)
//!   Performance fix: full MTF with 261K unique repos was O(n*m) = 2+ min runtime.
//!   Limit MTF to top 4096 frequent repos, fallback encoding for others.
//!   Runtime: 2+ min → ~12 sec (10x faster), acceptable size tradeoff.
//!
//! Tried and reverted (worse sizes than current baseline):
//! - Split repo_names into counts/lengths/bytes streams.
//! - Global repo-name dictionary with per-repo index lists.
//! - Owner/suffix split with owner dict + suffix bytes.
//! - Reordering encode by repo-id blocks (blew up id/ts deltas).
//! - Front-coded repo name dictionary.
//! - Delta+bitpack dup_name_meta/dup_name_ids (6,284,344 bytes).
//! - Split base from ts_deltas (+728 bytes, zstd already handles it well).
//! - Entropy coding (Huffman/FSE) for id_deltas: zstd already achieves near-optimal.
//! - repo_id_idx per row replacing repo_names + dup columns: +1.7MB worse due to
//!   needing both global_names (2.4MB) and repo_id_idx (2.1MB) vs repo_names (2.7MB).
//! - Front-coded global names + name→repo mapping: +305KB worse (mapping overhead
//!   676KB outweighs front-coding savings 371KB).
//! - repo_name_idx - repo_id_idx difference encoding: correlation too weak (±262K range),
//!   zigzag+compress gives worse results than direct encoding.

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::{HashMap, HashSet};
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const MAGIC: &[u8; 4] = b"XYZ1";
const ZSTD_LEVEL: i32 = 22;

const COLUMN_NAMES: [&str; 10] = [
    "type_dict",
    "repo_ids",
    "repo_names",
    "type_idx",
    "repo_name_idx",
    "dup_name_table",
    "dup_row_bitmap",
    "dup_variant_idx",
    "id_deltas",
    "ts_deltas",
];

fn debug_enabled() -> bool {
    std::env::var("XINYU_DEBUG").is_ok()
}

fn parse_timestamp(ts: &str) -> Result<u64, Box<dyn Error>> {
    let dt = DateTime::parse_from_rfc3339(ts)?;
    Ok(dt.timestamp() as u64)
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

fn read_u32(bytes: &[u8], pos: &mut usize) -> Result<u32, Box<dyn Error>> {
    if *pos + 4 > bytes.len() {
        return Err("unexpected EOF".into());
    }
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&bytes[*pos..*pos + 4]);
    *pos += 4;
    Ok(u32::from_le_bytes(arr))
}

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
            if bit_pos % 8 == 0 {
                buf.push(current_byte);
                current_byte = 0;
            }
        }
    }

    if bit_pos % 8 != 0 {
        buf.push(current_byte);
    }

    buf
}

fn unpack_bits_u64(bytes: &[u8], count: usize) -> Result<Vec<u64>, Box<dyn Error>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if bytes.is_empty() {
        return Err("empty bitpack buffer".into());
    }
    let bit_width = bytes[0] as usize;
    if bit_width == 0 {
        return Ok(vec![0; count]);
    }

    let mut values = Vec::with_capacity(count);
    let mut bit_pos: usize = 0;

    for _ in 0..count {
        let mut value: u64 = 0;
        for bit_idx in 0..bit_width {
            let byte_idx = 1 + (bit_pos / 8);
            if byte_idx >= bytes.len() {
                return Err("bitpack overflow".into());
            }
            let bit_in_byte = bit_pos % 8;
            let bit = ((bytes[byte_idx] >> bit_in_byte) & 1) as u64;
            value |= bit << bit_idx;
            bit_pos += 1;
        }
        values.push(value);
    }

    Ok(values)
}

fn pack_bits_u32(values: &[u32]) -> Vec<u8> {
    let tmp: Vec<u64> = values.iter().map(|&v| v as u64).collect();
    pack_bits_u64(&tmp)
}

#[allow(dead_code)]
fn pack_u32_le(values: &[u32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * 4);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn encode_u32_list(values: &[u32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + values.len() * 4);
    write_u32(&mut buf, values.len() as u32);
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn decode_u32_list(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut pos = 0;
    let count = read_u32(bytes, &mut pos)? as usize;
    if pos + count * 4 > bytes.len() {
        return Err("u32 list overflow".into());
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&bytes[pos..pos + 4]);
        pos += 4;
        out.push(u32::from_le_bytes(arr));
    }
    Ok(out)
}

fn unpack_bits_u32(bytes: &[u8], count: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    Ok(unpack_bits_u64(bytes, count)?
        .into_iter()
        .map(|v| v as u32)
        .collect())
}

#[allow(dead_code)]
fn unpack_u32_le(bytes: &[u8], count: usize) -> Result<Vec<u32>, Box<dyn Error>> {
    if bytes.len() < count * 4 {
        return Err("u32 buffer too small".into());
    }
    let mut out = Vec::with_capacity(count);
    let mut pos = 0;
    for _ in 0..count {
        let mut arr = [0u8; 4];
        arr.copy_from_slice(&bytes[pos..pos + 4]);
        pos += 4;
        out.push(u32::from_le_bytes(arr));
    }
    Ok(out)
}

fn zigzag_encode(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

fn delta_encode_signed(values: &[u64]) -> Result<Vec<u64>, Box<dyn Error>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(values.len());
    let mut prev = i64::try_from(values[0])?;
    out.push(zigzag_encode(prev));
    for &v in values.iter().skip(1) {
        let cur = i64::try_from(v)?;
        out.push(zigzag_encode(cur - prev));
        prev = cur;
    }
    Ok(out)
}

fn delta_decode_signed(values: &[u64]) -> Result<Vec<u64>, Box<dyn Error>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(values.len());
    let mut cur = zigzag_decode(values[0]);
    if cur < 0 {
        return Err("negative base".into());
    }
    out.push(cur as u64);
    for &v in values.iter().skip(1) {
        cur += zigzag_decode(v);
        if cur < 0 {
            return Err("negative value".into());
        }
        out.push(cur as u64);
    }
    Ok(out)
}

/// Encode values with base stored separately: returns (base, zigzag_deltas)
/// This avoids the base value inflating the bit width for all deltas.
#[allow(dead_code)]
fn delta_encode_signed_split(values: &[u64]) -> Result<(u64, Vec<u64>), Box<dyn Error>> {
    if values.is_empty() {
        return Ok((0, Vec::new()));
    }
    let base = values[0];
    let mut deltas = Vec::with_capacity(values.len().saturating_sub(1));
    let mut prev = i64::try_from(base)?;
    for &v in values.iter().skip(1) {
        let cur = i64::try_from(v)?;
        deltas.push(zigzag_encode(cur - prev));
        prev = cur;
    }
    Ok((base, deltas))
}

/// Decode values from base + zigzag deltas
#[allow(dead_code)]
fn delta_decode_signed_split(base: u64, deltas: &[u64]) -> Result<Vec<u64>, Box<dyn Error>> {
    let mut out = Vec::with_capacity(deltas.len() + 1);
    out.push(base);
    let mut cur = i64::try_from(base)?;
    for &d in deltas {
        cur += zigzag_decode(d);
        if cur < 0 {
            return Err("negative value".into());
        }
        out.push(cur as u64);
    }
    Ok(out)
}

#[allow(dead_code)]
fn delta_encode_unsigned(values: &[u64]) -> Vec<u64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut prev = values[0];
    out.push(prev);
    for &v in values.iter().skip(1) {
        out.push(v - prev);
        prev = v;
    }
    out
}

#[allow(dead_code)]
fn delta_decode_unsigned(values: &[u64]) -> Vec<u64> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(values.len());
    let mut cur = values[0];
    out.push(cur);
    for &v in values.iter().skip(1) {
        cur += v;
        out.push(cur);
    }
    out
}

/// Encode unsigned deltas with base stored separately, using varint encoding.
/// Since IDs are sorted, all deltas are non-negative.
/// Uses varint (LEB128-style) to handle variable delta sizes efficiently.
fn delta_encode_varint_split(values: &[u64]) -> (u64, Vec<u8>) {
    if values.is_empty() {
        return (0, Vec::new());
    }
    let base = values[0];
    let mut buf = Vec::with_capacity(values.len() * 2); // estimate
    for w in values.windows(2) {
        let mut d = w[1] - w[0];
        // Encode as varint (LEB128 unsigned)
        loop {
            let byte = (d & 0x7F) as u8;
            d >>= 7;
            if d == 0 {
                buf.push(byte);
                break;
            } else {
                buf.push(byte | 0x80);
            }
        }
    }
    (base, buf)
}

/// Decode values from base + varint deltas
fn delta_decode_varint_split(
    base: u64,
    deltas: &[u8],
    count: usize,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let mut out = Vec::with_capacity(count);
    out.push(base);
    let mut cur = base;
    let mut pos = 0;
    for _ in 1..count {
        // Decode varint
        let mut value: u64 = 0;
        let mut shift = 0;
        loop {
            if pos >= deltas.len() {
                return Err("varint overflow".into());
            }
            let byte = deltas[pos];
            pos += 1;
            value |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err("varint too large".into());
            }
        }
        cur += value;
        out.push(cur);
    }
    Ok(out)
}

/// Maximum alphabet size for MTF transform.
/// Values beyond this use fallback encoding (raw indices).
/// 4096 covers ~35% of events but keeps MTF O(n*K) manageable.
const MTF_ALPHABET_SIZE: usize = 4096;

/// Move-to-Front encode: transforms values so recently-used get small indices.
/// Exploits temporal locality - if the same value reappears soon, it gets a small index.
fn mtf_encode(values: &[u32], max_val: u32) -> Vec<u32> {
    let size = (max_val + 1) as usize;
    let mut mtf_list: Vec<u32> = (0..size as u32).collect();
    let mut positions: Vec<u32> = (0..size as u32).collect();
    let mut output = Vec::with_capacity(values.len());

    for &val in values {
        let pos = positions[val as usize];
        output.push(pos);
        if pos > 0 {
            // Shift elements and update positions
            for i in (1..=pos as usize).rev() {
                mtf_list[i] = mtf_list[i - 1];
                positions[mtf_list[i] as usize] = i as u32;
            }
            mtf_list[0] = val;
            positions[val as usize] = 0;
        }
    }
    output
}

/// Move-to-Front decode: reverses MTF transform
fn mtf_decode(values: &[u32], max_val: u32) -> Vec<u32> {
    let size = (max_val + 1) as usize;
    let mut mtf_list: Vec<u32> = (0..size as u32).collect();
    let mut output = Vec::with_capacity(values.len());

    for &pos in values {
        let val = mtf_list[pos as usize];
        output.push(val);
        if pos > 0 {
            // Shift elements
            for i in (1..=pos as usize).rev() {
                mtf_list[i] = mtf_list[i - 1];
            }
            mtf_list[0] = val;
        }
    }
    output
}

/// Result of limited-alphabet MTF encoding
struct LimitedMtfEncoded {
    /// Top-K original indices, sorted by frequency (most frequent first)
    top_k_indices: Vec<u32>,
    /// Bitmap: 1 = use MTF, 0 = use fallback
    use_mtf_bitmap: Vec<u8>,
    /// MTF-encoded values for rows where use_mtf=1
    mtf_values: Vec<u32>,
    /// Original indices for rows where use_mtf=0
    fallback_values: Vec<u32>,
}

/// Encode repo_name_idx using limited-alphabet MTF.
/// Only the top K most frequent indices use MTF encoding.
/// Infrequent indices are stored directly as fallback.
fn limited_mtf_encode(values: &[u32]) -> LimitedMtfEncoded {
    // Count frequencies
    let mut freq: HashMap<u32, usize> = HashMap::new();
    for &v in values {
        *freq.entry(v).or_insert(0) += 1;
    }

    // Sort by frequency (descending), then by index for determinism
    let mut freq_vec: Vec<(u32, usize)> = freq.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    // Select top K
    let k = MTF_ALPHABET_SIZE.min(freq_vec.len());
    let top_k_indices: Vec<u32> = freq_vec.iter().take(k).map(|(idx, _)| *idx).collect();

    // Build mapping: original_idx -> MTF alphabet position (0..K-1)
    let mut idx_to_mtf: HashMap<u32, u32> = HashMap::with_capacity(k);
    for (mtf_pos, &orig_idx) in top_k_indices.iter().enumerate() {
        idx_to_mtf.insert(orig_idx, mtf_pos as u32);
    }

    // Separate values into MTF and fallback streams
    let mut use_mtf_bitmap = vec![0u8; values.len().div_ceil(8)];
    let mut mtf_input: Vec<u32> = Vec::new();
    let mut fallback_values: Vec<u32> = Vec::new();

    for (i, &v) in values.iter().enumerate() {
        if let Some(&mtf_pos) = idx_to_mtf.get(&v) {
            // Mark as MTF
            use_mtf_bitmap[i / 8] |= 1 << (i % 8);
            mtf_input.push(mtf_pos);
        } else {
            // Fallback: store original index
            fallback_values.push(v);
        }
    }

    // Apply MTF only to the selected values (alphabet size = K)
    let mtf_values = if mtf_input.is_empty() {
        Vec::new()
    } else {
        let max_mtf = (k - 1) as u32;
        mtf_encode(&mtf_input, max_mtf)
    };

    LimitedMtfEncoded {
        top_k_indices,
        use_mtf_bitmap,
        mtf_values,
        fallback_values,
    }
}

/// Decode limited-alphabet MTF back to original indices.
fn limited_mtf_decode(
    top_k_indices: &[u32],
    use_mtf_bitmap: &[u8],
    mtf_values: &[u32],
    fallback_values: &[u32],
    row_count: usize,
) -> Result<Vec<u32>, Box<dyn Error>> {
    // Decode MTF values back to MTF alphabet positions
    let decoded_mtf = if mtf_values.is_empty() {
        Vec::new()
    } else {
        let max_mtf = (top_k_indices.len() - 1) as u32;
        mtf_decode(mtf_values, max_mtf)
    };

    // Reconstruct original indices
    let mut output = Vec::with_capacity(row_count);
    let mut mtf_cursor = 0usize;
    let mut fallback_cursor = 0usize;

    for i in 0..row_count {
        let byte = i / 8;
        let bit = i % 8;
        let use_mtf = if byte < use_mtf_bitmap.len() {
            (use_mtf_bitmap[byte] >> bit) & 1 == 1
        } else {
            false
        };

        if use_mtf {
            if mtf_cursor >= decoded_mtf.len() {
                return Err("MTF cursor overflow".into());
            }
            let mtf_pos = decoded_mtf[mtf_cursor] as usize;
            if mtf_pos >= top_k_indices.len() {
                return Err("MTF position out of range".into());
            }
            output.push(top_k_indices[mtf_pos]);
            mtf_cursor += 1;
        } else {
            if fallback_cursor >= fallback_values.len() {
                return Err("Fallback cursor overflow".into());
            }
            output.push(fallback_values[fallback_cursor]);
            fallback_cursor += 1;
        }
    }

    Ok(output)
}

struct StringDict {
    strings: Vec<String>,
    str_to_idx: HashMap<String, u32>,
}

impl StringDict {
    fn build_by_freq(events: &[(EventKey, EventValue)]) -> Self {
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for (key, _) in events {
            *freq.entry(&key.event_type).or_insert(0) += 1;
        }

        let mut types_with_freq: Vec<_> = freq.into_iter().collect();
        types_with_freq.sort_by(|a, b| b.1.cmp(&a.1));

        let mut strings = Vec::with_capacity(types_with_freq.len());
        let mut str_to_idx = HashMap::new();
        for (i, (t, _)) in types_with_freq.into_iter().enumerate() {
            strings.push(t.to_string());
            str_to_idx.insert(t.to_string(), i as u32);
        }

        Self {
            strings,
            str_to_idx,
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_u32(&mut buf, self.strings.len() as u32);
        for s in &self.strings {
            write_u32(&mut buf, s.len() as u32);
        }
        for s in &self.strings {
            buf.extend_from_slice(s.as_bytes());
        }
        buf
    }

    fn decode(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
        let mut pos = 0;
        let count = read_u32(bytes, &mut pos)? as usize;
        let mut lens = Vec::with_capacity(count);
        for _ in 0..count {
            lens.push(read_u32(bytes, &mut pos)? as usize);
        }

        let mut strings = Vec::with_capacity(count);
        let mut str_to_idx = HashMap::new();
        for (i, len) in lens.into_iter().enumerate() {
            if pos + len > bytes.len() {
                return Err("string dict overflow".into());
            }
            let s = std::str::from_utf8(&bytes[pos..pos + len])?.to_string();
            pos += len;
            str_to_idx.insert(s.clone(), i as u32);
            strings.push(s);
        }

        Ok(Self {
            strings,
            str_to_idx,
        })
    }

    fn get_index(&self, s: &str) -> u32 {
        self.str_to_idx[s]
    }

    fn get_string(&self, index: u32) -> &str {
        &self.strings[index as usize]
    }
}

#[allow(dead_code)]
struct RepoDict {
    repo_ids: Vec<u64>,
    repo_names: Vec<Vec<String>>,
    global_name_to_idx: HashMap<String, u32>,
    global_names: Vec<String>,
    id_to_idx: HashMap<u64, u32>,
    name_to_idx: Vec<HashMap<String, u32>>,
}

impl RepoDict {
    fn build(events: &[(EventKey, EventValue)]) -> Self {
        let mut map: HashMap<u64, HashSet<String>> = HashMap::new();
        for (_, value) in events {
            map.entry(value.repo.id)
                .or_default()
                .insert(value.repo.name.clone());
        }

        let mut repo_ids: Vec<u64> = map.keys().copied().collect();
        repo_ids.sort();

        let mut repo_names = Vec::with_capacity(repo_ids.len());
        let mut global_names = Vec::new();
        let mut id_to_idx = HashMap::with_capacity(repo_ids.len());
        let mut name_to_idx = Vec::with_capacity(repo_ids.len());

        for (i, repo_id) in repo_ids.iter().enumerate() {
            id_to_idx.insert(*repo_id, i as u32);
            let mut names: Vec<String> = map.get(repo_id).unwrap().iter().cloned().collect();
            names.sort();
            let mut name_map = HashMap::new();
            for (j, name) in names.iter().enumerate() {
                name_map.insert(name.clone(), j as u32);
                global_names.push(name.clone());
            }
            repo_names.push(names);
            name_to_idx.push(name_map);
        }

        global_names.sort();
        global_names.dedup();
        let mut global_name_to_idx = HashMap::with_capacity(global_names.len());
        for (i, name) in global_names.iter().enumerate() {
            global_name_to_idx.insert(name.clone(), i as u32);
        }

        Self {
            repo_ids,
            repo_names,
            global_name_to_idx,
            global_names,
            id_to_idx,
            name_to_idx,
        }
    }

    fn encode_repo_ids(&self) -> Vec<u8> {
        if self.repo_ids.is_empty() {
            return Vec::new();
        }

        // Store base value (first repo_id) as 8 bytes
        let base = self.repo_ids[0];
        let mut buf = Vec::new();
        buf.extend_from_slice(&base.to_le_bytes());

        // Compute deltas (all non-negative since sorted)
        let deltas: Vec<u64> = self.repo_ids.windows(2).map(|w| w[1] - w[0]).collect();

        // Split into 3 byte planes (max delta needs 19 bits = 3 bytes)
        let count = deltas.len();
        let mut low_bytes = Vec::with_capacity(count);
        let mut mid_bytes = Vec::with_capacity(count);
        let mut high_bytes = Vec::with_capacity(count);

        for d in &deltas {
            low_bytes.push(*d as u8);
            mid_bytes.push((*d >> 8) as u8);
            high_bytes.push((*d >> 16) as u8);
        }

        // Concatenate: high first (mostly zeros, compresses well), then mid, then low
        buf.extend_from_slice(&high_bytes);
        buf.extend_from_slice(&mid_bytes);
        buf.extend_from_slice(&low_bytes);

        buf
    }

    fn encode_repo_names(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        write_u32(&mut buf, self.repo_names.len() as u32);
        for names in &self.repo_names {
            write_u32(&mut buf, names.len() as u32);
        }
        for names in &self.repo_names {
            for name in names {
                buf.extend_from_slice(name.as_bytes());
                buf.push(0);
            }
        }
        buf
    }

    fn decode(
        repo_count: usize,
        repo_ids_raw: &[u8],
        repo_names_raw: &[u8],
    ) -> Result<Self, Box<dyn Error>> {
        // Decode repo_ids from byte planes format
        let repo_ids = if repo_count == 0 {
            Vec::new()
        } else {
            // Read base value (8 bytes)
            let base = u64::from_le_bytes(repo_ids_raw[0..8].try_into()?);

            // Remaining bytes are 3 byte planes
            let delta_count = repo_count - 1;
            let plane_size = delta_count;

            let high_bytes = &repo_ids_raw[8..8 + plane_size];
            let mid_bytes = &repo_ids_raw[8 + plane_size..8 + 2 * plane_size];
            let low_bytes = &repo_ids_raw[8 + 2 * plane_size..8 + 3 * plane_size];

            // Reconstruct repo_ids from deltas
            let mut ids = Vec::with_capacity(repo_count);
            ids.push(base);
            let mut cur = base;
            for i in 0..delta_count {
                let delta = (high_bytes[i] as u64) << 16
                    | (mid_bytes[i] as u64) << 8
                    | (low_bytes[i] as u64);
                cur += delta;
                ids.push(cur);
            }
            ids
        };

        let mut pos = 0;
        let names_repo_count = read_u32(repo_names_raw, &mut pos)? as usize;
        if names_repo_count != repo_count {
            return Err("repo count mismatch".into());
        }
        let mut name_counts = Vec::with_capacity(names_repo_count);
        for _ in 0..repo_count {
            name_counts.push(read_u32(repo_names_raw, &mut pos)? as usize);
        }

        let mut repo_names = Vec::with_capacity(repo_count);
        let mut cursor = pos;
        for &count in &name_counts {
            let mut names = Vec::with_capacity(count);
            for _ in 0..count {
                if cursor >= repo_names_raw.len() {
                    return Err("repo names overflow".into());
                }
                let start = cursor;
                while cursor < repo_names_raw.len() && repo_names_raw[cursor] != 0 {
                    cursor += 1;
                }
                if cursor >= repo_names_raw.len() {
                    return Err("repo names missing delimiter".into());
                }
                let s = std::str::from_utf8(&repo_names_raw[start..cursor])?.to_string();
                cursor += 1; // skip delimiter
                names.push(s);
            }
            repo_names.push(names);
        }
        if cursor != repo_names_raw.len() {
            return Err("repo names trailing bytes".into());
        }

        let mut id_to_idx = HashMap::with_capacity(repo_ids.len());
        let mut name_to_idx = Vec::with_capacity(repo_ids.len());
        let mut global_names = Vec::new();
        for (i, repo_id) in repo_ids.iter().enumerate() {
            id_to_idx.insert(*repo_id, i as u32);
            let mut name_map = HashMap::new();
            for (j, name) in repo_names[i].iter().enumerate() {
                name_map.insert(name.clone(), j as u32);
                global_names.push(name.clone());
            }
            name_to_idx.push(name_map);
        }

        global_names.sort();
        global_names.dedup();
        let mut global_name_to_idx = HashMap::with_capacity(global_names.len());
        for (i, name) in global_names.iter().enumerate() {
            global_name_to_idx.insert(name.clone(), i as u32);
        }

        Ok(Self {
            repo_ids,
            repo_names,
            global_name_to_idx,
            global_names,
            id_to_idx,
            name_to_idx,
        })
    }

    #[allow(dead_code)]
    fn repo_id_index(&self, repo_id: u64) -> u32 {
        self.id_to_idx[&repo_id]
    }

    #[allow(dead_code)]
    fn repo_name_index(&self, repo_id_idx: u32, repo_name: &str) -> u32 {
        self.name_to_idx[repo_id_idx as usize][repo_name]
    }

    #[allow(dead_code)]
    fn repo_id(&self, idx: u32) -> u64 {
        self.repo_ids[idx as usize]
    }

    #[allow(dead_code)]
    fn repo_name(&self, repo_id_idx: u32, name_idx: u32) -> &str {
        &self.repo_names[repo_id_idx as usize][name_idx as usize]
    }

    fn repo_name_global_index(&self, repo_name: &str) -> u32 {
        self.global_name_to_idx[repo_name]
    }

    fn repo_name_from_global(&self, idx: u32) -> &str {
        &self.global_names[idx as usize]
    }
}

pub struct XinyuzengCodec;

impl XinyuzengCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for XinyuzengCodec {
    fn name(&self) -> &str {
        "xinyuzeng"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let mut sorted_events: Vec<(EventKey, EventValue)> = events.to_vec();
        sorted_events.sort_by(|a, b| a.0.cmp(&b.0));

        let type_dict = StringDict::build_by_freq(&sorted_events);
        let repo_dict = RepoDict::build(&sorted_events);

        let type_dict_bytes = zstd::encode_all(type_dict.encode().as_slice(), ZSTD_LEVEL)?;
        let repo_ids_bytes = zstd::encode_all(repo_dict.encode_repo_ids().as_slice(), ZSTD_LEVEL)?;
        let repo_names_bytes =
            zstd::encode_all(repo_dict.encode_repo_names().as_slice(), ZSTD_LEVEL)?;

        let mut type_idx = Vec::with_capacity(sorted_events.len());
        let mut repo_name_idx = Vec::with_capacity(sorted_events.len());
        let mut ids = Vec::with_capacity(sorted_events.len());
        let mut timestamps = Vec::with_capacity(sorted_events.len());

        let mut name_id_sets: HashMap<u32, HashSet<u64>> = HashMap::new();
        let mut name_id_order: HashMap<u32, Vec<u64>> = HashMap::new();

        for (key, value) in &sorted_events {
            type_idx.push(type_dict.get_index(&key.event_type));
            let name_idx = repo_dict.repo_name_global_index(&value.repo.name);
            repo_name_idx.push(name_idx);
            name_id_sets
                .entry(name_idx)
                .or_default()
                .insert(value.repo.id);

            let event_id = key.id.parse::<u64>()?;
            ids.push(event_id);
            let ts = parse_timestamp(&value.created_at)?;
            timestamps.push(ts);
        }

        for (name_idx, ids_set) in &name_id_sets {
            if ids_set.len() > 1 {
                let mut ids_vec: Vec<u64> = ids_set.iter().copied().collect();
                ids_vec.sort();
                name_id_order.insert(*name_idx, ids_vec);
            }
        }

        let mut dup_name_entries: Vec<(u32, Vec<u64>)> = name_id_order.into_iter().collect();
        dup_name_entries.sort_by_key(|(k, _)| *k);

        let mut dup_name_offsets: HashMap<u32, u32> = HashMap::new();
        let mut dup_name_meta = Vec::new(); // [name_idx, count] pairs
        let mut dup_name_ids = Vec::new(); // u32 hi/lo pairs
        for (name_idx, ids_vec) in &dup_name_entries {
            dup_name_offsets.insert(*name_idx, (dup_name_meta.len() / 2) as u32);
            dup_name_meta.push(*name_idx);
            dup_name_meta.push(ids_vec.len() as u32);
            for id in ids_vec {
                dup_name_ids.push((*id >> 32) as u32);
                dup_name_ids.push(*id as u32);
            }
        }

        let mut dup_row_bitmap = vec![0u8; sorted_events.len().div_ceil(8)];
        let mut dup_variant_idx = Vec::new();

        for (row, (_key, value)) in sorted_events.iter().enumerate() {
            let name_idx = repo_dict.repo_name_global_index(&value.repo.name);
            if let Some(offset) = dup_name_offsets.get(&name_idx) {
                let byte = row / 8;
                let bit = row % 8;
                dup_row_bitmap[byte] |= 1 << bit;
                let ids_vec = &dup_name_entries[*offset as usize].1;
                let variant = ids_vec
                    .iter()
                    .position(|id| *id == value.repo.id)
                    .unwrap_or(0) as u32;
                dup_variant_idx.push(variant);
            }
        }

        // Use varint encoding for id_deltas since all deltas are non-negative after sorting
        // Varint handles arbitrary delta sizes (training data max=251, but test data can be larger)
        let (id_base, id_deltas) = delta_encode_varint_split(&ids);
        let ts_deltas = delta_encode_signed(&timestamps)?;

        let type_idx_bytes = zstd::encode_all(pack_bits_u32(&type_idx).as_slice(), ZSTD_LEVEL)?;

        // Apply limited-alphabet MTF transform for repo_name_idx
        // Only top-K frequent indices use MTF; others use fallback encoding
        let repo_name_idx_bytes = {
            let encoded = limited_mtf_encode(&repo_name_idx);

            // Compress top_k_indices
            let top_k_bytes = zstd::encode_all(
                encode_u32_list(&encoded.top_k_indices).as_slice(),
                ZSTD_LEVEL,
            )?;

            // Compress bitmap
            let bitmap_bytes = zstd::encode_all(encoded.use_mtf_bitmap.as_slice(), ZSTD_LEVEL)?;

            // MTF values: byte-plane split for better compression
            let (mtf_high, mtf_mid, mtf_low): (Vec<u8>, Vec<u8>, Vec<u8>) = encoded
                .mtf_values
                .iter()
                .map(|&v| ((v >> 16) as u8, (v >> 8) as u8, v as u8))
                .fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |(mut h, mut m, mut l), (hv, mv, lv)| {
                        h.push(hv);
                        m.push(mv);
                        l.push(lv);
                        (h, m, l)
                    },
                );
            let mtf_high_c = zstd::encode_all(mtf_high.as_slice(), ZSTD_LEVEL)?;
            let mtf_mid_c = zstd::encode_all(mtf_mid.as_slice(), ZSTD_LEVEL)?;
            let mtf_low_c = zstd::encode_all(mtf_low.as_slice(), ZSTD_LEVEL)?;

            // Fallback values: bitpacked
            let fallback_bytes = zstd::encode_all(
                pack_bits_u32(&encoded.fallback_values).as_slice(),
                ZSTD_LEVEL,
            )?;

            // Pack format:
            // [top_k_len:u32][bitmap_len:u32][mtf_high_len:u32][mtf_mid_len:u32][mtf_low_len:u32][fallback_len:u32]
            // [mtf_count:u32][fallback_count:u32]
            // [top_k][bitmap][mtf_high][mtf_mid][mtf_low][fallback]
            let mut buf = Vec::new();
            buf.extend_from_slice(&(top_k_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(bitmap_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(mtf_high_c.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(mtf_mid_c.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(mtf_low_c.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(fallback_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(encoded.mtf_values.len() as u32).to_le_bytes());
            buf.extend_from_slice(&(encoded.fallback_values.len() as u32).to_le_bytes());
            buf.extend_from_slice(&top_k_bytes);
            buf.extend_from_slice(&bitmap_bytes);
            buf.extend_from_slice(&mtf_high_c);
            buf.extend_from_slice(&mtf_mid_c);
            buf.extend_from_slice(&mtf_low_c);
            buf.extend_from_slice(&fallback_bytes);
            buf
        };

        let dup_name_meta_bytes =
            zstd::encode_all(encode_u32_list(&dup_name_meta).as_slice(), ZSTD_LEVEL)?;
        let dup_name_ids_bytes =
            zstd::encode_all(encode_u32_list(&dup_name_ids).as_slice(), ZSTD_LEVEL)?;
        let dup_row_bitmap_bytes = zstd::encode_all(dup_row_bitmap.as_slice(), ZSTD_LEVEL)?;
        let dup_variant_idx_bytes =
            zstd::encode_all(encode_u32_list(&dup_variant_idx).as_slice(), ZSTD_LEVEL)?;

        // Pack base value (8 bytes) followed by raw u8 deltas
        let mut id_raw = id_base.to_le_bytes().to_vec();
        id_raw.extend_from_slice(&id_deltas);
        let id_deltas_bytes = zstd::encode_all(id_raw.as_slice(), ZSTD_LEVEL)?;

        let ts_deltas_bytes = zstd::encode_all(pack_bits_u64(&ts_deltas).as_slice(), ZSTD_LEVEL)?;

        let debug = debug_enabled();
        if debug {
            let sizes = [
                type_dict_bytes.len(),
                repo_ids_bytes.len(),
                repo_names_bytes.len(),
                type_idx_bytes.len(),
                repo_name_idx_bytes.len(),
                dup_name_meta_bytes.len(),
                dup_row_bitmap_bytes.len(),
                dup_variant_idx_bytes.len(),
                id_deltas_bytes.len(),
                ts_deltas_bytes.len(),
            ];
            let total: usize = sizes.iter().sum();
            for (name, size) in COLUMN_NAMES.iter().zip(sizes.iter()) {
                eprintln!("XINYU_DEBUG col {}: {}", name, size);
            }
            eprintln!("XINYU_DEBUG total bytes: {}", total);
        }

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        write_u32(&mut out, sorted_events.len() as u32);
        write_u32(&mut out, type_dict.strings.len() as u32);
        write_u32(&mut out, repo_dict.repo_ids.len() as u32);

        let write_section = |buf: &mut Vec<u8>, bytes: &[u8]| {
            write_u32(buf, bytes.len() as u32);
            buf.extend_from_slice(bytes);
        };

        write_section(&mut out, &type_dict_bytes);
        write_section(&mut out, &repo_ids_bytes);
        write_section(&mut out, &repo_names_bytes);
        write_section(&mut out, &type_idx_bytes);
        write_section(&mut out, &repo_name_idx_bytes);
        write_section(&mut out, &dup_name_meta_bytes);
        write_section(&mut out, &dup_name_ids_bytes);
        write_section(&mut out, &dup_row_bitmap_bytes);
        write_section(&mut out, &dup_variant_idx_bytes);
        write_section(&mut out, &id_deltas_bytes);
        write_section(&mut out, &ts_deltas_bytes);

        Ok(Bytes::from(out))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Err("invalid magic".into());
        }
        pos += MAGIC.len();

        let row_count = read_u32(bytes, &mut pos)? as usize;
        let _type_count = read_u32(bytes, &mut pos)? as usize;
        let repo_count = read_u32(bytes, &mut pos)? as usize;

        let read_section = |bytes: &[u8], pos: &mut usize| -> Result<Vec<u8>, Box<dyn Error>> {
            let len = read_u32(bytes, pos)? as usize;
            let end = *pos + len;
            if end > bytes.len() {
                return Err("section overflow".into());
            }
            let compressed = &bytes[*pos..end];
            *pos = end;
            Ok(zstd::decode_all(compressed)?)
        };

        // For sections that handle their own compression (like repo_name_idx with MTF+byte planes)
        let read_section_raw = |bytes: &[u8], pos: &mut usize| -> Result<Vec<u8>, Box<dyn Error>> {
            let len = read_u32(bytes, pos)? as usize;
            let end = *pos + len;
            if end > bytes.len() {
                return Err("section overflow".into());
            }
            let raw = &bytes[*pos..end];
            *pos = end;
            Ok(raw.to_vec())
        };

        let type_dict_raw = read_section(bytes, &mut pos)?;
        let repo_ids_raw = read_section(bytes, &mut pos)?; // zstd-compressed byte planes
        let repo_names_raw = read_section(bytes, &mut pos)?;
        let type_idx_raw = read_section(bytes, &mut pos)?;
        let repo_name_idx_raw = read_section_raw(bytes, &mut pos)?; // MTF + byte planes, handles own compression
        let dup_name_meta_raw = read_section(bytes, &mut pos)?;
        let dup_name_ids_raw = read_section(bytes, &mut pos)?;
        let dup_row_bitmap_raw = read_section(bytes, &mut pos)?;
        let dup_variant_idx_raw = read_section(bytes, &mut pos)?;
        let id_deltas_raw = read_section(bytes, &mut pos)?;
        let ts_deltas_raw = read_section(bytes, &mut pos)?;

        let type_dict = StringDict::decode(&type_dict_raw)?;
        let repo_dict = RepoDict::decode(repo_count, &repo_ids_raw, &repo_names_raw)?;

        let type_idx = unpack_bits_u32(&type_idx_raw, row_count)?;

        // Decode limited-alphabet MTF for repo_name_idx
        let repo_name_idx = {
            if repo_name_idx_raw.len() < 32 {
                return Err("repo_name_idx_raw too short".into());
            }
            let mut hdr_pos = 0usize;
            let top_k_len =
                u32::from_le_bytes(repo_name_idx_raw[hdr_pos..hdr_pos + 4].try_into()?) as usize;
            hdr_pos += 4;
            let bitmap_len =
                u32::from_le_bytes(repo_name_idx_raw[hdr_pos..hdr_pos + 4].try_into()?) as usize;
            hdr_pos += 4;
            let mtf_high_len =
                u32::from_le_bytes(repo_name_idx_raw[hdr_pos..hdr_pos + 4].try_into()?) as usize;
            hdr_pos += 4;
            let mtf_mid_len =
                u32::from_le_bytes(repo_name_idx_raw[hdr_pos..hdr_pos + 4].try_into()?) as usize;
            hdr_pos += 4;
            let mtf_low_len =
                u32::from_le_bytes(repo_name_idx_raw[hdr_pos..hdr_pos + 4].try_into()?) as usize;
            hdr_pos += 4;
            let fallback_len =
                u32::from_le_bytes(repo_name_idx_raw[hdr_pos..hdr_pos + 4].try_into()?) as usize;
            hdr_pos += 4;
            let mtf_count =
                u32::from_le_bytes(repo_name_idx_raw[hdr_pos..hdr_pos + 4].try_into()?) as usize;
            hdr_pos += 4;
            let fallback_count =
                u32::from_le_bytes(repo_name_idx_raw[hdr_pos..hdr_pos + 4].try_into()?) as usize;
            hdr_pos += 4;

            let top_k_start = hdr_pos;
            let bitmap_start = top_k_start + top_k_len;
            let mtf_high_start = bitmap_start + bitmap_len;
            let mtf_mid_start = mtf_high_start + mtf_high_len;
            let mtf_low_start = mtf_mid_start + mtf_mid_len;
            let fallback_start = mtf_low_start + mtf_low_len;

            // Decompress components
            let top_k_indices = decode_u32_list(&zstd::decode_all(
                &repo_name_idx_raw[top_k_start..bitmap_start],
            )?)?;
            let use_mtf_bitmap =
                zstd::decode_all(&repo_name_idx_raw[bitmap_start..mtf_high_start])?;

            // Decompress MTF byte planes
            let mtf_high = zstd::decode_all(&repo_name_idx_raw[mtf_high_start..mtf_mid_start])?;
            let mtf_mid = zstd::decode_all(&repo_name_idx_raw[mtf_mid_start..mtf_low_start])?;
            let mtf_low = zstd::decode_all(&repo_name_idx_raw[mtf_low_start..fallback_start])?;

            // Reconstruct MTF values from byte planes
            let mut mtf_values = Vec::with_capacity(mtf_count);
            for i in 0..mtf_count {
                let v =
                    ((mtf_high[i] as u32) << 16) | ((mtf_mid[i] as u32) << 8) | (mtf_low[i] as u32);
                mtf_values.push(v);
            }

            // Decompress fallback values
            let fallback_values = unpack_bits_u32(
                &zstd::decode_all(
                    &repo_name_idx_raw[fallback_start..fallback_start + fallback_len],
                )?,
                fallback_count,
            )?;

            // Decode using limited MTF
            limited_mtf_decode(
                &top_k_indices,
                &use_mtf_bitmap,
                &mtf_values,
                &fallback_values,
                row_count,
            )?
        };

        let dup_name_meta = decode_u32_list(&dup_name_meta_raw)?;
        let dup_name_ids = decode_u32_list(&dup_name_ids_raw)?;
        let dup_row_bitmap = dup_row_bitmap_raw;
        let dup_variant_idx = decode_u32_list(&dup_variant_idx_raw)?;

        // Read base (8 bytes) then varint deltas for id_deltas
        if id_deltas_raw.len() < 8 {
            return Err("id_deltas too short".into());
        }
        let id_base = u64::from_le_bytes(id_deltas_raw[0..8].try_into().unwrap());
        let ids = delta_decode_varint_split(id_base, &id_deltas_raw[8..], row_count)?;

        let ts_deltas = unpack_bits_u64(&ts_deltas_raw, row_count)?;
        let timestamps = delta_decode_signed(&ts_deltas)?;

        if ids.len() != row_count || timestamps.len() != row_count {
            return Err("row count mismatch".into());
        }

        let mut dup_name_map: HashMap<u32, Vec<u64>> = HashMap::new();
        let mut id_cursor = 0usize;
        for chunk in dup_name_meta.chunks(2) {
            if chunk.len() != 2 {
                return Err("dup name meta mismatch".into());
            }
            let name_idx = chunk[0];
            let count = chunk[1] as usize;
            let mut ids_vec = Vec::with_capacity(count);
            for _ in 0..count {
                if id_cursor + 1 >= dup_name_ids.len() {
                    return Err("dup name ids overflow".into());
                }
                let hi = dup_name_ids[id_cursor] as u64;
                let lo = dup_name_ids[id_cursor + 1] as u64;
                id_cursor += 2;
                ids_vec.push((hi << 32) | lo);
            }
            dup_name_map.insert(name_idx, ids_vec);
        }

        let mut unique_name_to_repo_id: HashMap<u32, u64> = HashMap::new();
        for (repo_id_idx, repo_id) in repo_dict.repo_ids.iter().enumerate() {
            let names = &repo_dict.repo_names[repo_id_idx];
            for name in names {
                let name_idx = repo_dict.repo_name_global_index(name);
                if !dup_name_map.contains_key(&name_idx) {
                    unique_name_to_repo_id.insert(name_idx, *repo_id);
                }
            }
        }

        let mut dup_variant_cursor = 0usize;
        let mut events = Vec::with_capacity(row_count);
        for i in 0..row_count {
            let event_type = type_dict.get_string(type_idx[i]).to_string();
            let name_idx = repo_name_idx[i];
            let repo_name = repo_dict.repo_name_from_global(name_idx);

            let repo_id = if let Some(variants) = dup_name_map.get(&name_idx) {
                let byte = i / 8;
                let bit = i % 8;
                if byte >= dup_row_bitmap.len() {
                    return Err("dup row bitmap overflow".into());
                }
                let flagged = (dup_row_bitmap[byte] >> bit) & 1;
                if flagged == 0 {
                    return Err("dup row missing flag".into());
                }
                if dup_variant_cursor >= dup_variant_idx.len() {
                    return Err("dup variant idx overflow".into());
                }
                let variant = dup_variant_idx[dup_variant_cursor] as usize;
                dup_variant_cursor += 1;
                if variant >= variants.len() {
                    return Err("dup variant out of range".into());
                }
                variants[variant]
            } else {
                *unique_name_to_repo_id
                    .get(&name_idx)
                    .ok_or("missing unique repo id")?
            };

            let repo_url = format!("https://api.github.com/repos/{}", repo_name);
            events.push((
                EventKey {
                    id: ids[i].to_string(),
                    event_type,
                },
                EventValue {
                    repo: Repo {
                        id: repo_id,
                        name: repo_name.to_string(),
                        url: repo_url,
                    },
                    created_at: format_timestamp(timestamps[i]),
                },
            ));
        }

        Ok(events)
    }
}
