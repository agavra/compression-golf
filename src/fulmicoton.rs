/*
# Fulmicoton Codecs (5,677,291 bytes / ~97% compression):

This is obviously not a meaningful contest:
Everyone is welcome to steal ideas below, and save a few extra bytes by
adding their own. **Please consider quoting this codec though**.

I coded all of this using Claude. In fact, I mostly compete in this project
to get an idea of what LLM can do. I am bringing the ideas, and
Claude is doing the implementation.

This codec achieves 5,677,291 bytes / ~97% compression by:

1. **Columnar layout**: Events are transposed into separate column (or column families for repos)
    (event IDs, event types, timestamps, (repo name + repo id)).
    The rows are sorted by event Ids.

2. **Specialized encodings per column**:
   - Event IDs: Delta encoding + adaptive arithmetic coding.
   - Event types: ANS entropy coding (small alphabet)
   - Timestamps: The timestamps are almost sorted. There are several
events by seconds. RLE is quite efficient for this, but I went further:
I encode the small permutation required to sort the data.
After that I can "histogram encode" the result.
    I use something that I call VIPCompression a lot.
    I identify the top K most common things in a stream. I compress the stream of n elements by replacing by
    a stream of tokens in 0..K+1 representing those top elements that I compress using ANS + a sentinel element representing "others". Then I represent the stream of not so common element using a different representation.

   - Repos are dictionary encoded:
        - the repo indices indices are encoded as follows. VIP Coding (2047) + we encode
    the remaining indices over 3 bytes. The top byte is encoded using ANS. The two remaining are compressed with zstd.
        - Dictionary (we sort the repo id + repo name by repo id and we compress them in different columns)
            - repo IDs: BIC (Binary Interpolative Coding) for sorted sequences
            - repo names: We split owner and suffix from the repo names and encode them separately.
        In both case we use VIP Coding(1023). We then concatenate the remaining owner and suffix \n separated in the
        same blob of text and apply zstd on this.

I think most of the headroom left is in the repository. I suspect it will be difficult to improve event ids/types/timestamp by a lot at this point.
*/

#![allow(clippy::needless_range_loop)]

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};
use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::error::Error;

// ============================================================================
// algo.rs - permutation algorithms
// ============================================================================

pub fn identify_permutation(mut values: Vec<u64>) -> (Vec<u64>, Vec<(usize, usize)>) {
    let mut moves = Vec::new();
    let mut i = 1;
    let mut advance_count = 0;

    while i < values.len() {
        if values[i] >= values[i - 1] {
            advance_count += 1;
            i += 1;
        } else {
            let mut bubble_count = 0;
            let mut j = i;
            while j > 0 && values[j] < values[j - 1] {
                values.swap(j, j - 1);
                j -= 1;
                bubble_count += 1;
            }

            moves.push((advance_count, bubble_count));
            advance_count = 0;
            i += 1;
        }
    }

    if advance_count > 0 {
        moves.push((advance_count, 0));
    }

    (values, moves)
}

pub fn restore_permutation(mut values: Vec<u64>, moves: Vec<(usize, usize)>) -> Vec<u64> {
    let mut actions = Vec::new();
    let mut i = 1;
    for (adv, bubble) in moves {
        i += adv;
        if bubble > 0 {
            actions.push((i, bubble));
            i += 1;
        }
    }

    for (idx, bubble) in actions.into_iter().rev() {
        let src = idx - bubble;
        let dest = idx;
        values[src..=dest].rotate_left(1);
    }
    values
}

// ============================================================================
// bijection.rs - bijection traits and implementations
// ============================================================================

pub trait Bijection<A, B> {
    fn apply(&self, source: A) -> B;
    fn revert(&self, source: B) -> A;
}

pub struct ZStdBijection;

impl Bijection<Vec<u8>, Vec<u8>> for ZStdBijection {
    fn apply(&self, source: Vec<u8>) -> Vec<u8> {
        zstd::encode_all(&source[..], 22).unwrap()
    }

    fn revert(&self, source: Vec<u8>) -> Vec<u8> {
        zstd::decode_all(&source[..]).unwrap()
    }
}

pub struct VIntBijection;

impl Bijection<Vec<u64>, Vec<u8>> for VIntBijection {
    fn apply(&self, numbers: Vec<u64>) -> Vec<u8> {
        let mut buf = Vec::with_capacity(numbers.len() * 8);
        for &num in &numbers {
            let mut n = num;
            loop {
                let mut byte = (n & 0x7F) as u8;
                n >>= 7;
                if n != 0 {
                    byte |= 0x80;
                }
                buf.push(byte);
                if n == 0 {
                    break;
                }
            }
        }
        buf
    }

    fn revert(&self, bytes: Vec<u8>) -> Vec<u64> {
        let mut numbers = Vec::new();
        let mut n = 0u64;
        let mut shift = 0;
        for byte in bytes {
            n |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                numbers.push(n);
                n = 0;
                shift = 0;
            } else {
                shift += 7;
            }
        }
        numbers
    }
}

pub struct PositiveDeltaBijection;

impl Bijection<Vec<i64>, Vec<u64>> for PositiveDeltaBijection {
    fn apply(&self, source: Vec<i64>) -> Vec<u64> {
        if source.is_empty() {
            return Vec::new();
        }
        let mut deltas = Vec::with_capacity(source.len());
        let mut prev = 0;
        for &val in &source {
            let delta = val - prev;
            if delta < 0 {
                panic!("Negative delta encountered in PositiveDeltaBijection");
            }
            deltas.push(delta as u64);
            prev = val;
        }
        deltas
    }

    fn revert(&self, source: Vec<u64>) -> Vec<i64> {
        let mut original = Vec::with_capacity(source.len());
        let mut prev = 0;
        for &delta in &source {
            let val = prev + (delta as i64);
            original.push(val);
            prev = val;
        }
        original
    }
}

pub struct HistogramBijection;

impl Bijection<Vec<i64>, Vec<u64>> for HistogramBijection {
    fn apply(&self, source: Vec<i64>) -> Vec<u64> {
        if source.is_empty() {
            return vec![];
        }

        let min = source[0];
        let max = source[source.len() - 1];

        for i in 0..source.len() - 1 {
            if source[i] > source[i + 1] {
                panic!("HistogramBijection: not sorted at index {}", i);
            }
        }

        let range = (max - min + 1) as usize;
        if range > 50_000_000 {
            panic!("HistogramBijection: range too large {}", range);
        }

        let mut counts = vec![0u64; range];
        for &val in &source {
            let idx = (val - min) as usize;
            counts[idx] += 1;
        }

        let mut output = Vec::with_capacity(1 + counts.len());
        output.push(min as u64);
        output.extend(counts);
        output
    }

    fn revert(&self, source: Vec<u64>) -> Vec<i64> {
        if source.is_empty() {
            return vec![];
        }

        let min = source[0] as i64;
        let counts = &source[1..];

        let total: u64 = counts.iter().sum();
        let mut output = Vec::with_capacity(total as usize);

        let mut current_val = min;
        for &count in counts {
            for _ in 0..count {
                output.push(current_val);
            }
            current_val += 1;
        }

        output
    }
}

pub struct MonotonicPermutationBijection;

impl Bijection<Vec<i64>, Vec<u8>> for MonotonicPermutationBijection {
    fn apply(&self, source: Vec<i64>) -> Vec<u8> {
        if source.is_empty() {
            return vec![];
        }

        let min_val = *source.iter().min().unwrap();
        let values: Vec<u64> = source.iter().map(|&x| (x - min_val) as u64).collect();

        let (sorted_u64, moves) = identify_permutation(values);

        let hist = HistogramBijection;
        let ans_u64 = AnsU64Bijection;
        let zstd = ZStdBijection;

        let sorted_i64: Vec<i64> = sorted_u64.iter().map(|&x| x as i64).collect();
        let hist_u64 = hist.apply(sorted_i64);

        let sorted_bytes = ans_u64.apply(hist_u64);

        let mut moves_buf = Vec::new();
        let count = moves.len();
        write_vint_local(count, &mut moves_buf);

        for &(adv, _) in &moves {
            write_vint_local(adv, &mut moves_buf);
        }
        for &(_, bub) in &moves {
            write_vint_local(bub, &mut moves_buf);
        }

        let moves_compressed = zstd.apply(moves_buf);

        let mut result = Vec::new();
        result.extend_from_slice(&min_val.to_le_bytes());

        write_vint_local(sorted_bytes.len(), &mut result);
        result.extend(sorted_bytes);

        write_vint_local(moves_compressed.len(), &mut result);
        result.extend(moves_compressed);

        result
    }

    fn revert(&self, source: Vec<u8>) -> Vec<i64> {
        if source.is_empty() {
            return vec![];
        }

        let mut offset = 0;
        let min_val_bytes = &source[offset..offset + 8];
        let min_val = i64::from_le_bytes([
            min_val_bytes[0],
            min_val_bytes[1],
            min_val_bytes[2],
            min_val_bytes[3],
            min_val_bytes[4],
            min_val_bytes[5],
            min_val_bytes[6],
            min_val_bytes[7],
        ]);
        offset += 8;

        let mut read_part = || {
            let len = read_vint_local(&source, &mut offset);
            let part = &source[offset..offset + len];
            offset += len;
            part.to_vec()
        };

        let sorted_bytes = read_part();
        let moves_compressed = read_part();

        let hist = HistogramBijection;
        let ans_u64 = AnsU64Bijection;
        let zstd = ZStdBijection;

        let hist_u64 = ans_u64.revert(sorted_bytes);
        let sorted_i64 = hist.revert(hist_u64);
        let values: Vec<u64> = sorted_i64.iter().map(|&x| x as u64).collect();

        let moves_raw = zstd.revert(moves_compressed);
        let mut moves = Vec::new();
        let mut m_offset = 0;
        if !moves_raw.is_empty() {
            let count = read_vint_local(&moves_raw, &mut m_offset);
            let mut advances = Vec::with_capacity(count);
            for _ in 0..count {
                advances.push(read_vint_local(&moves_raw, &mut m_offset));
            }
            let mut bubbles = Vec::with_capacity(count);
            for _ in 0..count {
                bubbles.push(read_vint_local(&moves_raw, &mut m_offset));
            }

            for (adv, bub) in advances.into_iter().zip(bubbles.into_iter()) {
                moves.push((adv, bub));
            }
        }

        let restored_values = restore_permutation(values, moves);

        restored_values
            .iter()
            .map(|&x| (x as i64) + min_val)
            .collect()
    }
}

fn write_vint_local(mut n: usize, buf: &mut Vec<u8>) {
    loop {
        let mut byte = (n & 0x7F) as u8;
        n >>= 7;
        if n != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if n == 0 {
            break;
        }
    }
}

fn read_vint_local(bytes: &[u8], offset: &mut usize) -> usize {
    let mut n = 0;
    let mut shift = 0;
    loop {
        let byte = bytes[*offset];
        *offset += 1;
        n |= ((byte & 0x7F) as usize) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    n
}

/// Binary Interpolative Coding (BIC) for sorted sequences
pub struct BicBijection;

impl Bijection<Vec<u64>, Vec<u8>> for BicBijection {
    fn apply(&self, source: Vec<u64>) -> Vec<u8> {
        if source.is_empty() {
            return vec![];
        }

        let max_val = *source.last().unwrap();

        let mut writer = BitWriter::new();

        writer.write_bits(source.len() as u64, 32);
        writer.write_bits(max_val, 64);

        bic_encode(&source, 0, max_val, &mut writer);

        writer.finish()
    }

    fn revert(&self, source: Vec<u8>) -> Vec<u64> {
        if source.is_empty() {
            return vec![];
        }

        let mut reader = BitReader::new(&source);

        let n = reader.read_bits(32) as usize;
        let max_val = reader.read_bits(64);

        let mut result = vec![0u64; n];
        bic_decode(&mut result, 0, max_val, &mut reader);

        result
    }
}

fn bic_encode(values: &[u64], lo: u64, hi: u64, writer: &mut BitWriter) {
    let n = values.len();
    if n == 0 {
        return;
    }

    let mid = n / 2;
    let m = values[mid];

    let m_lo = lo + mid as u64;
    let m_hi = hi - (n - 1 - mid) as u64;

    let range = m_hi - m_lo + 1;

    if range > 1 {
        let bits = 64 - (range - 1).leading_zeros();
        writer.write_bits(m - m_lo, bits as usize);
    }

    if mid > 0 {
        bic_encode(&values[..mid], lo, m - 1, writer);
    }
    if mid + 1 < n {
        bic_encode(&values[mid + 1..], m + 1, hi, writer);
    }
}

fn bic_decode(values: &mut [u64], lo: u64, hi: u64, reader: &mut BitReader) {
    let n = values.len();
    if n == 0 {
        return;
    }

    let mid = n / 2;

    let m_lo = lo + mid as u64;
    let m_hi = hi - (n - 1 - mid) as u64;

    let range = m_hi - m_lo + 1;

    let m = if range > 1 {
        let bits = 64 - (range - 1).leading_zeros();
        m_lo + reader.read_bits(bits as usize)
    } else {
        m_lo
    };

    values[mid] = m;

    if mid > 0 {
        bic_decode(&mut values[..mid], lo, m - 1, reader);
    }
    if mid + 1 < n {
        bic_decode(&mut values[mid + 1..], m + 1, hi, reader);
    }
}

struct BitWriter {
    bytes: Vec<u8>,
    current: u64,
    bits_in_current: usize,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            bytes: Vec::new(),
            current: 0,
            bits_in_current: 0,
        }
    }

    fn write_bits(&mut self, value: u64, num_bits: usize) {
        if num_bits == 0 {
            return;
        }

        let mut value = value;
        let mut remaining = num_bits;

        while remaining > 0 {
            let space = 64 - self.bits_in_current;
            let to_write = remaining.min(space);

            let mask = if to_write >= 64 {
                u64::MAX
            } else {
                (1u64 << to_write) - 1
            };
            self.current |= (value & mask) << self.bits_in_current;
            self.bits_in_current += to_write;

            if to_write < 64 {
                value >>= to_write;
            } else {
                value = 0;
            }
            remaining -= to_write;

            while self.bits_in_current >= 8 {
                self.bytes.push(self.current as u8);
                self.current >>= 8;
                self.bits_in_current -= 8;
            }
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_in_current > 0 {
            self.bytes.push(self.current as u8);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        BitReader {
            bytes,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bits(&mut self, num_bits: usize) -> u64 {
        if num_bits == 0 {
            return 0;
        }

        let mut result = 0u64;
        let mut bits_read = 0;

        while bits_read < num_bits {
            if self.byte_pos >= self.bytes.len() {
                break;
            }

            let bits_available_in_byte = 8 - self.bit_pos;
            let bits_needed = num_bits - bits_read;
            let bits_to_read = bits_available_in_byte.min(bits_needed);

            let mask = ((1u16 << bits_to_read) - 1) as u8;
            let bits = (self.bytes[self.byte_pos] >> self.bit_pos) & mask;

            result |= (bits as u64) << bits_read;

            bits_read += bits_to_read;
            self.bit_pos += bits_to_read;

            if self.bit_pos >= 8 {
                self.bit_pos = 0;
                self.byte_pos += 1;
            }
        }

        result
    }
}

/// Adaptive arithmetic coding bijection for Vec<u64> (assumes values fit in u8)
pub struct AdaptiveMixU64Bijection;

impl Bijection<Vec<u64>, Vec<u8>> for AdaptiveMixU64Bijection {
    fn apply(&self, source: Vec<u64>) -> Vec<u8> {
        if source.is_empty() {
            return vec![];
        }

        let data: Vec<u8> = source.iter().map(|&v| v as u8).collect();

        let mut coder = AdaptiveMixCoder::new();
        let encoded = coder.encode(&data);

        let mut result = Vec::with_capacity(4 + encoded.len());
        result.extend_from_slice(&(source.len() as u32).to_le_bytes());
        result.extend(encoded);
        result
    }

    fn revert(&self, source: Vec<u8>) -> Vec<u64> {
        if source.is_empty() {
            return vec![];
        }

        let len = u32::from_le_bytes([source[0], source[1], source[2], source[3]]) as usize;
        let encoded = &source[4..];

        let mut coder = AdaptiveMixCoder::new();
        let decoded = coder.decode(encoded, len);

        decoded.iter().map(|&v| v as u64).collect()
    }
}

// ============================================================================
// ans.rs - ANS entropy encoding
// ============================================================================

pub struct AnsBijection;

const SCALE_BITS: u32 = 12;
const SCALE: u32 = 1 << SCALE_BITS;
const STATE_LOWER_BOUND: u32 = 1 << 16;

impl Bijection<Vec<u8>, Vec<u8>> for AnsBijection {
    fn apply(&self, source: Vec<u8>) -> Vec<u8> {
        if source.is_empty() {
            return vec![];
        }

        let mut counts = [0u32; 256];
        for &b in &source {
            counts[b as usize] += 1;
        }

        let mut normalized_counts = [0u16; 256];
        let total = source.len() as u64;
        let mut sum = 0u32;
        let mut max_symbol = 0;
        let mut max_count = 0;

        for i in 0..256 {
            if counts[i] > 0 {
                let mut c = (counts[i] as u64 * SCALE as u64 / total) as u32;
                if c == 0 {
                    c = 1;
                }
                normalized_counts[i] = c as u16;
                sum += c;
                if c > max_count {
                    max_count = c;
                    max_symbol = i;
                }
            }
        }

        if sum != SCALE {
            let diff = SCALE as i32 - sum as i32;
            let val = normalized_counts[max_symbol] as i32 + diff;
            normalized_counts[max_symbol] = val as u16;
        }

        let mut starts = [0u32; 256];
        let mut current_start = 0;
        for i in 0..256 {
            starts[i] = current_start;
            current_start += normalized_counts[i] as u32;
        }

        let mut stream = Vec::new();
        let mut x = STATE_LOWER_BOUND;

        for &symbol in source.iter().rev() {
            let s = symbol as usize;
            let freq = normalized_counts[s] as u32;
            let start = starts[s];

            let bound = freq << (16 + 8 - SCALE_BITS);
            while x >= bound {
                stream.push(x as u8);
                x >>= 8;
            }

            x = ((x / freq) << SCALE_BITS) + (x % freq) + start;
        }

        let x_bytes = x.to_le_bytes();
        let mut result = Vec::with_capacity(512 + 4 + 4 + stream.len());
        for &c in &normalized_counts {
            result.extend_from_slice(&c.to_le_bytes());
        }
        result.extend_from_slice(&(source.len() as u32).to_le_bytes());
        result.extend_from_slice(&x_bytes);
        result.extend(stream.iter().rev());
        result
    }

    fn revert(&self, source: Vec<u8>) -> Vec<u8> {
        if source.is_empty() {
            return vec![];
        }
        let mut cursor = 0;
        let mut normalized_counts = [0u16; 256];
        for i in 0..256 {
            let bytes = &source[cursor..cursor + 2];
            normalized_counts[i] = u16::from_le_bytes([bytes[0], bytes[1]]);
            cursor += 2;
        }

        let mut cum_freq = [0u32; 257];
        let mut sum = 0;
        for i in 0..256 {
            cum_freq[i] = sum;
            sum += normalized_counts[i] as u32;
        }
        cum_freq[256] = sum;

        let mut symbol_map = [0u8; SCALE as usize];
        for s in 0..256 {
            let start = cum_freq[s] as usize;
            let end = cum_freq[s + 1] as usize;
            for i in start..end {
                symbol_map[i] = s as u8;
            }
        }

        let len_bytes = &source[cursor..cursor + 4];
        let length =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
        cursor += 4;

        let state_bytes = &source[cursor..cursor + 4];
        let mut x = u32::from_le_bytes([
            state_bytes[0],
            state_bytes[1],
            state_bytes[2],
            state_bytes[3],
        ]);
        cursor += 4;

        let mut output = Vec::with_capacity(length);
        let stream = &source[cursor..];
        let mut stream_ptr = 0;

        for _ in 0..length {
            let slot = (x & (SCALE - 1)) as usize;
            let s = symbol_map[slot];
            output.push(s);
            let freq = normalized_counts[s as usize] as u32;
            let start = cum_freq[s as usize];
            x = freq * (x >> SCALE_BITS) + (x & (SCALE - 1)) - start;
            while x < STATE_LOWER_BOUND {
                if stream_ptr >= stream.len() {
                    break;
                }
                let byte = stream[stream_ptr] as u32;
                stream_ptr += 1;
                x = (x << 8) | byte;
            }
        }
        output
    }
}

pub struct AnsU64Bijection;

impl Bijection<Vec<u64>, Vec<u8>> for AnsU64Bijection {
    fn apply(&self, source: Vec<u64>) -> Vec<u8> {
        if source.is_empty() {
            return vec![];
        }

        let mut unique_values: Vec<u64> = source.clone();
        unique_values.sort();
        unique_values.dedup();

        if unique_values.len() > 65536 {
            panic!("AnsU64Bijection: Too many unique values for u16 rank");
        }

        let ranks: Vec<u16> = source
            .iter()
            .map(|&val| unique_values.binary_search(&val).unwrap() as u16)
            .collect();

        let alphabet_size = unique_values.len();
        let mut counts = vec![0u32; alphabet_size];
        for &r in &ranks {
            counts[r as usize] += 1;
        }

        let mut normalized_counts = vec![0u16; alphabet_size];
        let total = source.len() as u64;
        let mut sum = 0u32;
        let mut max_symbol = 0;
        let mut max_count = 0;

        for i in 0..alphabet_size {
            if counts[i] > 0 {
                let mut c = (counts[i] as u64 * SCALE as u64 / total) as u32;
                if c == 0 {
                    c = 1;
                }
                normalized_counts[i] = c as u16;
                sum += c;
                if c > max_count {
                    max_count = c;
                    max_symbol = i;
                }
            }
        }

        if sum != SCALE {
            let diff = SCALE as i32 - sum as i32;
            let val = normalized_counts[max_symbol] as i32 + diff;
            normalized_counts[max_symbol] = val as u16;
        }

        let mut starts = vec![0u32; alphabet_size];
        let mut current_start = 0;
        for i in 0..alphabet_size {
            starts[i] = current_start;
            current_start += normalized_counts[i] as u32;
        }

        let mut stream = Vec::new();
        let mut x = STATE_LOWER_BOUND;

        for &r in ranks.iter().rev() {
            let s = r as usize;
            let freq = normalized_counts[s] as u32;
            let start = starts[s];

            let bound = freq << (16 + 8 - SCALE_BITS);
            while x >= bound {
                stream.push(x as u8);
                x >>= 8;
            }
            x = ((x / freq) << SCALE_BITS) + (x % freq) + start;
        }

        let mut result = Vec::new();

        let vint = VIntBijection;
        let zstd = ZStdBijection;
        let dict_bytes = zstd.apply(vint.apply(unique_values));

        write_vint_local(dict_bytes.len(), &mut result);
        result.extend(dict_bytes);

        for &c in &normalized_counts {
            result.extend_from_slice(&c.to_le_bytes());
        }

        result.extend_from_slice(&(source.len() as u32).to_le_bytes());
        result.extend_from_slice(&x.to_le_bytes());
        result.extend(stream.iter().rev());

        result
    }

    fn revert(&self, source: Vec<u8>) -> Vec<u64> {
        if source.is_empty() {
            return vec![];
        }

        let mut offset = 0;
        let dict_len = read_vint_local(&source, &mut offset);
        let dict_bytes = &source[offset..offset + dict_len];
        offset += dict_len;

        let vint = VIntBijection;
        let zstd = ZStdBijection;
        let unique_values = vint.revert(zstd.revert(dict_bytes.to_vec()));
        let alphabet_size = unique_values.len();

        let mut normalized_counts = Vec::with_capacity(alphabet_size);
        for _ in 0..alphabet_size {
            let bytes = &source[offset..offset + 2];
            normalized_counts.push(u16::from_le_bytes([bytes[0], bytes[1]]));
            offset += 2;
        }

        let mut cum_freq = Vec::with_capacity(alphabet_size + 1);
        let mut sum = 0;
        for i in 0..alphabet_size {
            cum_freq.push(sum);
            sum += normalized_counts[i] as u32;
        }
        cum_freq.push(sum);

        let mut symbol_map = vec![0u16; SCALE as usize];
        for s in 0..alphabet_size {
            let start = cum_freq[s] as usize;
            let end = cum_freq[s + 1] as usize;
            for i in start..end {
                symbol_map[i] = s as u16;
            }
        }

        let len_bytes = &source[offset..offset + 4];
        let length =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
        offset += 4;

        let state_bytes = &source[offset..offset + 4];
        let mut x = u32::from_le_bytes([
            state_bytes[0],
            state_bytes[1],
            state_bytes[2],
            state_bytes[3],
        ]);
        offset += 4;

        let mut output = Vec::with_capacity(length);
        let stream = &source[offset..];
        let mut stream_ptr = 0;

        for _ in 0..length {
            let slot = (x & (SCALE - 1)) as usize;
            let rank = symbol_map[slot];
            output.push(unique_values[rank as usize]);

            let freq = normalized_counts[rank as usize] as u32;
            let start = cum_freq[rank as usize];

            x = freq * (x >> SCALE_BITS) + (x & (SCALE - 1)) - start;

            while x < STATE_LOWER_BOUND {
                if stream_ptr >= stream.len() {
                    break;
                }
                let byte = stream[stream_ptr] as u32;
                stream_ptr += 1;
                x = (x << 8) | byte;
            }
        }

        output
    }
}

/// Generic ANS bijection for alphabet size N
pub struct AnsGenericBijection<const N: usize>;

const SCALE_BITS_N: u32 = 14;
const SCALE_N: u32 = 1 << SCALE_BITS_N;
const STATE_LOWER_BOUND_N: u32 = 1 << 20;

impl<const N: usize> Bijection<Vec<u16>, Vec<u8>> for AnsGenericBijection<N> {
    fn apply(&self, source: Vec<u16>) -> Vec<u8> {
        if source.is_empty() {
            return vec![];
        }

        let mut counts = vec![0u32; N];
        for &s in &source {
            counts[s as usize] += 1;
        }

        let mut normalized_counts = vec![0u16; N];
        let total = source.len() as u64;
        let mut sum = 0u32;
        let mut max_symbol = 0;
        let mut max_count = 0;

        for i in 0..N {
            if counts[i] > 0 {
                let mut c = (counts[i] as u64 * SCALE_N as u64 / total) as u32;
                if c == 0 {
                    c = 1;
                }
                normalized_counts[i] = c as u16;
                sum += c;
                if c > max_count {
                    max_count = c;
                    max_symbol = i;
                }
            }
        }

        if sum != SCALE_N {
            let diff = SCALE_N as i32 - sum as i32;
            let val = normalized_counts[max_symbol] as i32 + diff;
            normalized_counts[max_symbol] = val as u16;
        }

        let mut starts = vec![0u32; N];
        let mut current_start = 0;
        for i in 0..N {
            starts[i] = current_start;
            current_start += normalized_counts[i] as u32;
        }

        let mut stream = Vec::new();
        let mut x = STATE_LOWER_BOUND_N;

        for &symbol in source.iter().rev() {
            let s = symbol as usize;
            let freq = normalized_counts[s] as u32;
            let start = starts[s];

            let bound = freq << (20 + 8 - SCALE_BITS_N);
            while x >= bound {
                stream.push(x as u8);
                x >>= 8;
            }

            x = ((x / freq) << SCALE_BITS_N) + (x % freq) + start;
        }

        let x_bytes = x.to_le_bytes();
        let mut result = Vec::with_capacity(N * 2 + 4 + 4 + stream.len());
        for &c in &normalized_counts {
            result.extend_from_slice(&c.to_le_bytes());
        }
        result.extend_from_slice(&(source.len() as u32).to_le_bytes());
        result.extend_from_slice(&x_bytes);
        result.extend(stream.iter().rev());
        result
    }

    fn revert(&self, source: Vec<u8>) -> Vec<u16> {
        if source.is_empty() {
            return vec![];
        }
        let mut cursor = 0;
        let mut normalized_counts = vec![0u16; N];
        for i in 0..N {
            let bytes = &source[cursor..cursor + 2];
            normalized_counts[i] = u16::from_le_bytes([bytes[0], bytes[1]]);
            cursor += 2;
        }

        let mut cum_freq = vec![0u32; N + 1];
        let mut sum = 0;
        for i in 0..N {
            cum_freq[i] = sum;
            sum += normalized_counts[i] as u32;
        }
        cum_freq[N] = sum;

        let mut symbol_map = vec![0u16; SCALE_N as usize];
        for s in 0..N {
            let start = cum_freq[s] as usize;
            let end = cum_freq[s + 1] as usize;
            for i in start..end {
                symbol_map[i] = s as u16;
            }
        }

        let len_bytes = &source[cursor..cursor + 4];
        let length =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
        cursor += 4;

        let state_bytes = &source[cursor..cursor + 4];
        let mut x = u32::from_le_bytes([
            state_bytes[0],
            state_bytes[1],
            state_bytes[2],
            state_bytes[3],
        ]);
        cursor += 4;

        let mut output = Vec::with_capacity(length);
        let stream = &source[cursor..];
        let mut stream_ptr = 0;

        for _ in 0..length {
            let slot = (x & (SCALE_N - 1)) as usize;
            let s = symbol_map[slot];
            output.push(s);
            let freq = normalized_counts[s as usize] as u32;
            let start = cum_freq[s as usize];
            x = freq * (x >> SCALE_BITS_N) + (x & (SCALE_N - 1)) - start;
            while x < STATE_LOWER_BOUND_N {
                if stream_ptr >= stream.len() {
                    break;
                }
                let byte = stream[stream_ptr] as u32;
                stream_ptr += 1;
                x = (x << 8) | byte;
            }
        }
        output
    }
}

// ============================================================================
// adaptive_mix.rs - Adaptive arithmetic coding
// ============================================================================

/// Frequency table for encoding/decoding
pub struct FrequencyTable {
    freqs: [u32; 256],
    cum_freqs: [u32; 257],
    total: u32,
}

impl FrequencyTable {
    const SCALE: u32 = 1 << 14;

    pub fn new() -> Self {
        let mut table = Self {
            freqs: [1; 256],
            cum_freqs: [0; 257],
            total: 256,
        };
        for i in 0..256 {
            table.cum_freqs[i + 1] = table.cum_freqs[i] + table.freqs[i];
        }
        table
    }

    pub fn update(&mut self, symbol: u8) {
        self.freqs[symbol as usize] += 1;
        self.total += 1;

        self.cum_freqs[0] = 0;
        for i in 0..256 {
            self.cum_freqs[i + 1] = self.cum_freqs[i] + self.freqs[i];
        }

        if self.total > Self::SCALE {
            self.rescale();
        }
    }

    fn rescale(&mut self) {
        self.total = 0;
        for f in &mut self.freqs {
            *f = (*f).div_ceil(2);
            self.total += *f;
        }
        self.cum_freqs[0] = 0;
        for i in 0..256 {
            self.cum_freqs[i + 1] = self.cum_freqs[i] + self.freqs[i];
        }
    }

    pub fn get_freq(&self, symbol: u8) -> u32 {
        self.freqs[symbol as usize]
    }

    pub fn get_cum_freq(&self, symbol: u8) -> u32 {
        self.cum_freqs[symbol as usize]
    }

    pub fn get_total(&self) -> u32 {
        self.total
    }

    pub fn find_symbol(&self, cum: u32) -> u8 {
        let mut lo = 0usize;
        let mut hi = 256usize;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cum_freqs[mid + 1] <= cum {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo as u8
    }
}

/// Simple range/arithmetic encoder
pub struct RangeEncoder {
    low: u64,
    high: u64,
    pending_bits: u32,
    output: Vec<u8>,
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            high: 0xFFFFFFFF,
            pending_bits: 0,
            output: Vec::new(),
        }
    }

    fn output_bit(&mut self, bit: bool) {
        if bit {
            self.output.push(1);
        } else {
            self.output.push(0);
        }
        while self.pending_bits > 0 {
            self.output.push(if bit { 0 } else { 1 });
            self.pending_bits -= 1;
        }
    }

    pub fn encode(&mut self, cum_freq: u32, freq: u32, total: u32) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range * (cum_freq + freq) as u64) / total as u64 - 1;
        self.low += (range * cum_freq as u64) / total as u64;

        loop {
            if self.high < 0x80000000 {
                self.output_bit(false);
                self.low <<= 1;
                self.high = (self.high << 1) | 1;
            } else if self.low >= 0x80000000 {
                self.output_bit(true);
                self.low = (self.low - 0x80000000) << 1;
                self.high = ((self.high - 0x80000000) << 1) | 1;
            } else if self.low >= 0x40000000 && self.high < 0xC0000000 {
                self.pending_bits += 1;
                self.low = (self.low - 0x40000000) << 1;
                self.high = ((self.high - 0x40000000) << 1) | 1;
            } else {
                break;
            }
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.pending_bits += 1;
        self.output_bit(self.low >= 0x40000000);

        let mut bytes = Vec::new();
        for chunk in self.output.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= bit << (7 - i);
            }
            bytes.push(byte);
        }
        let bit_count = self.output.len() as u32;
        let mut result = bit_count.to_le_bytes().to_vec();
        result.extend(bytes);
        result
    }
}

/// Simple range/arithmetic decoder
pub struct RangeDecoder {
    low: u64,
    high: u64,
    value: u64,
    bits: Vec<u8>,
    bit_pos: usize,
}

impl RangeDecoder {
    pub fn new(input: &[u8]) -> Self {
        if input.len() < 4 {
            return Self {
                low: 0,
                high: 0xFFFFFFFF,
                value: 0,
                bits: Vec::new(),
                bit_pos: 0,
            };
        }

        let bit_count = u32::from_le_bytes([input[0], input[1], input[2], input[3]]) as usize;

        let mut bits = Vec::with_capacity(bit_count);
        for &byte in &input[4..] {
            for i in (0..8).rev() {
                bits.push((byte >> i) & 1);
                if bits.len() >= bit_count {
                    break;
                }
            }
            if bits.len() >= bit_count {
                break;
            }
        }

        let mut decoder = Self {
            low: 0,
            high: 0xFFFFFFFF,
            value: 0,
            bits,
            bit_pos: 0,
        };

        for _ in 0..32 {
            decoder.value = (decoder.value << 1) | decoder.read_bit() as u64;
        }

        decoder
    }

    fn read_bit(&mut self) -> u8 {
        if self.bit_pos < self.bits.len() {
            let bit = self.bits[self.bit_pos];
            self.bit_pos += 1;
            bit
        } else {
            0
        }
    }

    pub fn get_freq(&self, total: u32) -> u32 {
        let range = self.high - self.low + 1;
        (((self.value - self.low + 1) * total as u64 - 1) / range) as u32
    }

    pub fn decode(&mut self, cum_freq: u32, freq: u32, total: u32) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range * (cum_freq + freq) as u64) / total as u64 - 1;
        self.low += (range * cum_freq as u64) / total as u64;

        loop {
            if self.high < 0x80000000 {
                self.low <<= 1;
                self.high = (self.high << 1) | 1;
                self.value = (self.value << 1) | self.read_bit() as u64;
            } else if self.low >= 0x80000000 {
                self.low = (self.low - 0x80000000) << 1;
                self.high = ((self.high - 0x80000000) << 1) | 1;
                self.value = ((self.value - 0x80000000) << 1) | self.read_bit() as u64;
            } else if self.low >= 0x40000000 && self.high < 0xC0000000 {
                self.low = (self.low - 0x40000000) << 1;
                self.high = ((self.high - 0x40000000) << 1) | 1;
                self.value = ((self.value - 0x40000000) << 1) | self.read_bit() as u64;
            } else {
                break;
            }
        }
    }
}

/// Adaptive arithmetic coder with order-0 model
pub struct AdaptiveMixCoder {
    freqs: FrequencyTable,
}

impl AdaptiveMixCoder {
    pub fn new() -> Self {
        Self {
            freqs: FrequencyTable::new(),
        }
    }

    pub fn encode(&mut self, data: &[u8]) -> Vec<u8> {
        let mut encoder = RangeEncoder::new();

        for &symbol in data {
            let cum_freq = self.freqs.get_cum_freq(symbol);
            let freq = self.freqs.get_freq(symbol);
            let total = self.freqs.get_total();

            encoder.encode(cum_freq, freq, total);
            self.freqs.update(symbol);
        }

        encoder.finish()
    }

    pub fn decode(&mut self, encoded: &[u8], len: usize) -> Vec<u8> {
        let mut decoder = RangeDecoder::new(encoded);
        let mut result = Vec::with_capacity(len);

        for _ in 0..len {
            let total = self.freqs.get_total();

            let freq_val = decoder.get_freq(total).min(total - 1);

            let symbol = self.freqs.find_symbol(freq_val);

            let cum_freq = self.freqs.get_cum_freq(symbol);
            let freq = self.freqs.get_freq(symbol);
            decoder.decode(cum_freq, freq, total);

            self.freqs.update(symbol);

            result.push(symbol);
        }

        result
    }
}

// ============================================================================
// columnar.rs - columnar data structures and parsing
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedEvent {
    pub id: i64,
    pub event_type: String,
    pub repo_id: u64,
    pub repo_owner: String,
    pub repo_suffix: String,
    pub created_at: i64,
}

pub struct ParseBijection;

impl ParseBijection {
    pub fn apply(&self, source: &(EventKey, EventValue)) -> ParsedEvent {
        let (key, val) = source;

        let id = key.id.parse::<i64>().expect("Failed to parse event id");
        let created_at = DateTime::parse_from_rfc3339(&val.created_at)
            .expect("Failed to parse created_at")
            .timestamp();

        let parts: Vec<&str> = val.repo.name.splitn(2, '/').collect();
        let (repo_owner, repo_suffix) = if parts.len() == 2 {
            (parts[0].to_string(), parts[1].to_string())
        } else {
            (val.repo.name.clone(), "".to_string())
        };

        ParsedEvent {
            id,
            event_type: key.event_type.clone(),
            repo_id: val.repo.id,
            repo_owner,
            repo_suffix,
            created_at,
        }
    }

    pub fn revert(&self, source: &ParsedEvent) -> (EventKey, EventValue) {
        let key = EventKey {
            id: source.id.to_string(),
            event_type: source.event_type.clone(),
        };

        let dt = Utc.timestamp_opt(source.created_at, 0).unwrap();
        let created_at = dt.format("%Y-%m-%dT%H:%M:%SZ").to_string();

        let repo_name = if source.repo_suffix.is_empty() {
            source.repo_owner.clone()
        } else {
            format!("{}/{}", source.repo_owner, source.repo_suffix)
        };

        let val = EventValue {
            repo: Repo {
                id: source.repo_id,
                name: repo_name.clone(),
                url: format!("https://api.github.com/repos/{}", repo_name),
            },
            created_at,
        };

        (key, val)
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct ColumnarEvents {
    pub event_ids: Vec<i64>,
    pub event_type_indices: Vec<u8>,
    pub dict_event_types: Vec<String>,
    pub created_ats: Vec<i64>,

    pub repo_indices: Vec<u64>,
    pub dict_repo_ids: Vec<u64>,
    pub dict_repo_owners: Vec<String>,
    pub dict_repo_suffixes: Vec<String>,
}

pub struct EventsToColumns;

impl<'a> Bijection<Cow<'a, [ParsedEvent]>, ColumnarEvents> for EventsToColumns {
    fn apply(&self, events: Cow<'a, [ParsedEvent]>) -> ColumnarEvents {
        let events = events.as_ref();
        let mut cols = ColumnarEvents::default();

        cols.event_ids.reserve(events.len());
        cols.event_type_indices.reserve(events.len());
        cols.created_ats.reserve(events.len());
        cols.repo_indices.reserve(events.len());

        let mut unique_repos: HashSet<(u64, String, String)> = HashSet::new();
        for event in events {
            unique_repos.insert((
                event.repo_id,
                event.repo_owner.clone(),
                event.repo_suffix.clone(),
            ));
        }

        let mut sorted_repos: Vec<(u64, String, String)> = unique_repos.into_iter().collect();
        sorted_repos.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });

        let mut repo_to_index = HashMap::new();
        for (i, (id, owner, suffix)) in sorted_repos.into_iter().enumerate() {
            repo_to_index.insert((id, owner.clone(), suffix.clone()), i as u64);
            cols.dict_repo_ids.push(id);
            cols.dict_repo_owners.push(owner);
            cols.dict_repo_suffixes.push(suffix);
        }

        let mut unique_event_types: HashSet<String> = HashSet::new();
        for event in events {
            unique_event_types.insert(event.event_type.clone());
        }
        let mut sorted_event_types: Vec<String> = unique_event_types.into_iter().collect();
        sorted_event_types.sort();

        let mut event_type_to_index = HashMap::new();
        for (i, et) in sorted_event_types.into_iter().enumerate() {
            if i > 255 {
                panic!("Too many event types for u8");
            }
            event_type_to_index.insert(et.clone(), i as u8);
            cols.dict_event_types.push(et);
        }

        for event in events {
            cols.event_ids.push(event.id);

            let et_idx = event_type_to_index
                .get(&event.event_type)
                .expect("Event type not found");
            cols.event_type_indices.push(*et_idx);

            cols.created_ats.push(event.created_at);

            let idx = repo_to_index
                .get(&(
                    event.repo_id,
                    event.repo_owner.clone(),
                    event.repo_suffix.clone(),
                ))
                .expect("Repo not found in dictionary");
            cols.repo_indices.push(*idx);
        }

        cols
    }

    fn revert(&self, cols: ColumnarEvents) -> Cow<'a, [ParsedEvent]> {
        let len = cols.event_ids.len();
        let mut events = Vec::with_capacity(len);

        for i in 0..len {
            let et_idx = cols.event_type_indices[i] as usize;
            let event_type = cols.dict_event_types[et_idx].clone();

            let repo_idx = cols.repo_indices[i] as usize;
            let repo_id = cols.dict_repo_ids[repo_idx];
            let repo_owner = cols.dict_repo_owners[repo_idx].clone();
            let repo_suffix = cols.dict_repo_suffixes[repo_idx].clone();

            events.push(ParsedEvent {
                id: cols.event_ids[i],
                event_type,
                repo_id,
                repo_owner,
                repo_suffix,
                created_at: cols.created_ats[i],
            });
        }

        Cow::Owned(events)
    }
}

// ============================================================================
// mod.rs - main codec implementation
// ============================================================================

pub struct FulmicotonCodec;

impl FulmicotonCodec {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for FulmicotonCodec {
    fn name(&self) -> &str {
        "fulmicoton"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let mut events = events.to_vec();
        events.sort_by(|a, b| a.0.cmp(&b.0));

        let parser = ParseBijection;
        let parsed_events: Vec<ParsedEvent> = events.iter().map(|e| parser.apply(e)).collect();

        let transformer = EventsToColumns;
        let cols = transformer.apply(Cow::Owned(parsed_events));

        let general_compression = ZStdBijection;
        let vint = VIntBijection;
        let pos_delta = PositiveDeltaBijection;
        let ans = AnsBijection;
        let perm = MonotonicPermutationBijection;

        let adaptive_mix = AdaptiveMixU64Bijection;
        let deltas = pos_delta.apply(cols.event_ids);
        let mut c_event_ids = Vec::new();
        if !deltas.is_empty() {
            write_vint(deltas[0] as usize, &mut c_event_ids);
            c_event_ids.extend(adaptive_mix.apply(deltas[1..].to_vec()));
        }
        let c_event_type_indices = ans.apply(cols.event_type_indices);
        let c_dict_event_types =
            general_compression.apply(cols.dict_event_types.join("\n").into_bytes());

        let c_created_ats = perm.apply(cols.created_ats);

        let c_repo_indices = encode_repo_indices_hybrid(&cols.repo_indices);
        let bic = BicBijection;
        let mut unique_ids: Vec<u64> = Vec::new();
        let mut dup_indices: Vec<u32> = Vec::new();
        for (i, &id) in cols.dict_repo_ids.iter().enumerate() {
            if i > 0 && id == cols.dict_repo_ids[i - 1] {
                dup_indices.push(i as u32);
            } else {
                unique_ids.push(id);
            }
        }
        let bic_encoded = bic.apply(unique_ids);
        let dup_deltas: Vec<u64> = if dup_indices.is_empty() {
            Vec::new()
        } else {
            let mut deltas = vec![dup_indices[0] as u64];
            for i in 1..dup_indices.len() {
                deltas.push((dup_indices[i] - dup_indices[i - 1]) as u64);
            }
            deltas
        };
        let dup_encoded = vint.apply(dup_deltas);
        let mut c_dict_repo_ids = Vec::new();
        c_dict_repo_ids.extend_from_slice(&(dup_indices.len() as u32).to_le_bytes());
        write_vint(dup_encoded.len(), &mut c_dict_repo_ids);
        c_dict_repo_ids.extend(dup_encoded);
        c_dict_repo_ids.extend(bic_encoded);

        let c_dict_repo_names =
            encode_repo_names_hybrid(&cols.dict_repo_owners, &cols.dict_repo_suffixes);

        let mut final_buf = Vec::new();
        let parts: Vec<Vec<u8>> = vec![
            c_event_ids,
            c_event_type_indices,
            c_dict_event_types,
            c_created_ats,
            c_repo_indices,
            c_dict_repo_ids,
            c_dict_repo_names,
        ];

        for part in parts {
            write_vint(part.len(), &mut final_buf);
            final_buf.extend_from_slice(&part);
        }

        Ok(Bytes::from(final_buf))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut offset = 0;
        let mut read_part = || {
            let len = read_vint(bytes, &mut offset);
            let part = &bytes[offset..offset + len];
            offset += len;
            part.to_vec()
        };

        let c_event_ids = read_part();
        let c_event_type_indices = read_part();
        let c_dict_event_types = read_part();
        let c_created_ats = read_part();
        let c_repo_indices = read_part();
        let c_dict_repo_ids = read_part();
        let c_dict_repo_names = read_part();

        let general_compression = ZStdBijection;
        let vint = VIntBijection;
        let pos_delta = PositiveDeltaBijection;
        let ans = AnsBijection;
        let perm = MonotonicPermutationBijection;

        let (dict_repo_owners, dict_repo_suffixes) = decode_repo_names_hybrid(&c_dict_repo_names);

        let et_names_bytes = general_compression.revert(c_dict_event_types);
        let et_names_str = String::from_utf8(et_names_bytes)?;
        let dict_event_types: Vec<String> = if et_names_str.is_empty() {
            Vec::new()
        } else {
            et_names_str.split('\n').map(|s| s.to_string()).collect()
        };

        let adaptive_mix = AdaptiveMixU64Bijection;
        let mut delta_offset = 0;
        let first_delta = read_vint(&c_event_ids, &mut delta_offset) as u64;
        let rest_deltas = adaptive_mix.revert(c_event_ids[delta_offset..].to_vec());
        let mut deltas = vec![first_delta];
        deltas.extend(rest_deltas);
        let event_ids = pos_delta.revert(deltas);

        let cols = ColumnarEvents {
            event_ids,

            event_type_indices: ans.revert(c_event_type_indices),
            dict_event_types,

            created_ats: perm.revert(c_created_ats),

            repo_indices: decode_repo_indices_hybrid(c_repo_indices),

            dict_repo_ids: {
                let mut offset = 0;
                let num_dups = u32::from_le_bytes([
                    c_dict_repo_ids[0],
                    c_dict_repo_ids[1],
                    c_dict_repo_ids[2],
                    c_dict_repo_ids[3],
                ]) as usize;
                offset += 4;
                let dup_len = read_vint(&c_dict_repo_ids, &mut offset);
                let dup_encoded = &c_dict_repo_ids[offset..offset + dup_len];
                offset += dup_len;
                let bic_encoded = c_dict_repo_ids[offset..].to_vec();

                let dup_deltas = vint.revert(dup_encoded.to_vec());
                let mut dup_indices: Vec<usize> = Vec::with_capacity(num_dups);
                let mut pos = 0u64;
                for delta in dup_deltas {
                    pos += delta;
                    dup_indices.push(pos as usize);
                }

                let unique_ids = BicBijection.revert(bic_encoded);

                let total_len = unique_ids.len() + dup_indices.len();
                let mut result = Vec::with_capacity(total_len);
                let mut unique_iter = unique_ids.into_iter();
                let dup_set: std::collections::HashSet<usize> = dup_indices.into_iter().collect();

                for i in 0..total_len {
                    if dup_set.contains(&i) {
                        result.push(*result.last().unwrap());
                    } else {
                        result.push(unique_iter.next().unwrap());
                    }
                }
                result
            },
            dict_repo_owners,
            dict_repo_suffixes,
        };

        let transformer = EventsToColumns;
        let parsed_events = transformer.revert(cols);

        let parser = ParseBijection;
        let mut events: Vec<(EventKey, EventValue)> =
            parsed_events.iter().map(|e| parser.revert(e)).collect();

        events.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(events)
    }
}

const OTHER_SYMBOL: u16 = 2047;

fn encode_repo_indices_hybrid(repo_indices: &[u64]) -> Vec<u8> {
    let mut freq_map: HashMap<u64, usize> = HashMap::new();
    for &idx in repo_indices {
        *freq_map.entry(idx).or_insert(0) += 1;
    }

    let mut freq_vec: Vec<(u64, usize)> = freq_map.into_iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let top_count = freq_vec.len().min(2047);
    let top_repos: Vec<u64> = freq_vec
        .iter()
        .take(top_count)
        .map(|(idx, _)| *idx)
        .collect();

    let mut index_to_symbol: HashMap<u64, u16> = HashMap::new();
    for (symbol, &original_idx) in top_repos.iter().enumerate() {
        index_to_symbol.insert(original_idx, symbol as u16);
    }

    let mut symbols: Vec<u16> = Vec::with_capacity(repo_indices.len());
    let mut other_indices: Vec<u64> = Vec::new();

    for &idx in repo_indices {
        if let Some(&symbol) = index_to_symbol.get(&idx) {
            symbols.push(symbol);
        } else {
            symbols.push(OTHER_SYMBOL);
            other_indices.push(idx);
        }
    }

    let ans = AnsGenericBijection::<2048>;
    let c_symbols = ans.apply(symbols);

    let zstd = ZStdBijection;
    let c_other = encode_u24_columnar(&other_indices);

    let vint = VIntBijection;
    let c_mapping = zstd.apply(vint.apply(top_repos));

    let mut result = Vec::new();
    write_vint(c_mapping.len(), &mut result);
    result.extend(c_mapping);
    write_vint(c_symbols.len(), &mut result);
    result.extend(c_symbols);
    write_vint(c_other.len(), &mut result);
    result.extend(c_other);

    result
}

fn decode_repo_indices_hybrid(data: Vec<u8>) -> Vec<u64> {
    let mut offset = 0;

    let mapping_len = read_vint(&data, &mut offset);
    let c_mapping = data[offset..offset + mapping_len].to_vec();
    offset += mapping_len;

    let symbols_len = read_vint(&data, &mut offset);
    let c_symbols = data[offset..offset + symbols_len].to_vec();
    offset += symbols_len;

    let other_len = read_vint(&data, &mut offset);
    let c_other = data[offset..offset + other_len].to_vec();

    let vint = VIntBijection;
    let zstd = ZStdBijection;
    let top_repos = vint.revert(zstd.revert(c_mapping));

    let ans = AnsGenericBijection::<2048>;
    let symbols = ans.revert(c_symbols);

    let other_indices = decode_u24_columnar(&c_other);

    let mut result = Vec::with_capacity(symbols.len());
    let mut other_iter = other_indices.into_iter();

    for symbol in symbols {
        if symbol == OTHER_SYMBOL {
            result.push(other_iter.next().expect("Missing 'other' index"));
        } else {
            result.push(top_repos[symbol as usize]);
        }
    }

    result
}

fn encode_u24_columnar(indices: &[u64]) -> Vec<u8> {
    if indices.is_empty() {
        return vec![];
    }

    let mut low_bytes = Vec::with_capacity(indices.len() * 2);
    let mut high_bytes = Vec::with_capacity(indices.len());

    for &idx in indices {
        low_bytes.push(idx as u8);
        low_bytes.push((idx >> 8) as u8);
        high_bytes.push((idx >> 16) as u8);
    }

    let zstd = ZStdBijection;
    let c_low = zstd.apply(low_bytes);
    let ans = AnsBijection;
    let c_high = ans.apply(high_bytes);

    let mut result = Vec::new();
    write_vint(c_low.len(), &mut result);
    result.extend(c_low);
    write_vint(c_high.len(), &mut result);
    result.extend(c_high);

    result
}

fn decode_u24_columnar(data: &[u8]) -> Vec<u64> {
    if data.is_empty() {
        return vec![];
    }

    let mut offset = 0;

    let len_low = read_vint(data, &mut offset);
    let c_low = &data[offset..offset + len_low];
    offset += len_low;

    let len_high = read_vint(data, &mut offset);
    let c_high = &data[offset..offset + len_high];

    let zstd = ZStdBijection;
    let low_bytes = zstd.revert(c_low.to_vec());
    let ans = AnsBijection;
    let high_bytes = ans.revert(c_high.to_vec());

    let count = high_bytes.len();
    let mut indices = Vec::with_capacity(count);
    for i in 0..count {
        let low = low_bytes[i * 2] as u64;
        let mid = low_bytes[i * 2 + 1] as u64;
        let high = high_bytes[i] as u64;
        let idx = low | (mid << 8) | (high << 16);
        indices.push(idx);
    }

    indices
}

fn write_vint(mut n: usize, buf: &mut Vec<u8>) {
    loop {
        let mut byte = (n & 0x7F) as u8;
        n >>= 7;
        if n != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if n == 0 {
            break;
        }
    }
}

fn read_vint(bytes: &[u8], offset: &mut usize) -> usize {
    let mut n = 0;
    let mut shift = 0;
    loop {
        let byte = bytes[*offset];
        *offset += 1;
        n |= ((byte & 0x7F) as usize) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    n
}

const TOP_OWNERS_COUNT: usize = 1023;
const OTHER_OWNER_SYMBOL: u16 = 1023;
const TOP_SUFFIXES_COUNT: usize = 1023;
const OTHER_SUFFIX_SYMBOL: u16 = 1023;

fn encode_repo_names_hybrid(owners: &[String], suffixes: &[String]) -> Vec<u8> {
    if suffixes.is_empty() {
        return vec![];
    }

    let mut owner_freq_map: HashMap<&str, usize> = HashMap::new();
    for s in owners {
        *owner_freq_map.entry(s.as_str()).or_insert(0) += 1;
    }

    let mut owner_freq_vec: Vec<(&str, usize)> = owner_freq_map.into_iter().collect();
    owner_freq_vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let top_owner_count = owner_freq_vec.len().min(TOP_OWNERS_COUNT);
    let top_owners: Vec<&str> = owner_freq_vec
        .iter()
        .take(top_owner_count)
        .map(|(s, _)| *s)
        .collect();

    let mut owner_to_symbol: HashMap<&str, u16> = HashMap::new();
    for (symbol, &owner) in top_owners.iter().enumerate() {
        owner_to_symbol.insert(owner, symbol as u16);
    }

    let mut owner_symbols: Vec<u16> = Vec::with_capacity(owners.len());
    let mut other_owners: Vec<&str> = Vec::new();

    for s in owners {
        if let Some(&symbol) = owner_to_symbol.get(s.as_str()) {
            owner_symbols.push(symbol);
        } else {
            owner_symbols.push(OTHER_OWNER_SYMBOL);
            other_owners.push(s.as_str());
        }
    }

    let c_owner_symbols = AnsGenericBijection::<1024>.apply(owner_symbols);

    let mut suffix_freq_map: HashMap<&str, usize> = HashMap::new();
    for s in suffixes {
        *suffix_freq_map.entry(s.as_str()).or_insert(0) += 1;
    }

    let mut suffix_freq_vec: Vec<(&str, usize)> = suffix_freq_map.into_iter().collect();
    suffix_freq_vec.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    let top_suffix_count = suffix_freq_vec.len().min(TOP_SUFFIXES_COUNT);
    let top_suffixes: Vec<&str> = suffix_freq_vec
        .iter()
        .take(top_suffix_count)
        .map(|(s, _)| *s)
        .collect();

    let mut suffix_to_symbol: HashMap<&str, u16> = HashMap::new();
    for (symbol, &suffix) in top_suffixes.iter().enumerate() {
        suffix_to_symbol.insert(suffix, symbol as u16);
    }

    let mut suffix_symbols: Vec<u16> = Vec::with_capacity(suffixes.len());
    let mut other_suffixes: Vec<&str> = Vec::new();

    for s in suffixes {
        if let Some(&symbol) = suffix_to_symbol.get(s.as_str()) {
            suffix_symbols.push(symbol);
        } else {
            suffix_symbols.push(OTHER_SUFFIX_SYMBOL);
            other_suffixes.push(s.as_str());
        }
    }

    let c_suffix_symbols = AnsGenericBijection::<1024>.apply(suffix_symbols);

    let joined_top_owners = top_owners.join("\n");
    let joined_other_owners = other_owners.join("\n");
    let joined_top_suffixes = top_suffixes.join("\n");
    let joined_other_suffixes = other_suffixes.join("\n");

    let mut combined = Vec::new();
    combined.extend_from_slice(joined_top_owners.as_bytes());
    combined.push(0);
    combined.extend_from_slice(joined_other_owners.as_bytes());
    combined.push(0);
    combined.extend_from_slice(joined_top_suffixes.as_bytes());
    combined.push(0);
    combined.extend_from_slice(joined_other_suffixes.as_bytes());

    let zstd = ZStdBijection;
    let c_strings = zstd.apply(combined);

    let mut result = Vec::new();
    write_vint(c_owner_symbols.len(), &mut result);
    result.extend(c_owner_symbols);
    write_vint(c_suffix_symbols.len(), &mut result);
    result.extend(c_suffix_symbols);
    result.extend(c_strings);

    result
}

fn decode_repo_names_hybrid(data: &[u8]) -> (Vec<String>, Vec<String>) {
    if data.is_empty() {
        return (vec![], vec![]);
    }

    let mut offset = 0;

    let owner_symbols_len = read_vint(data, &mut offset);
    let c_owner_symbols = data[offset..offset + owner_symbols_len].to_vec();
    offset += owner_symbols_len;

    let suffix_symbols_len = read_vint(data, &mut offset);
    let c_suffix_symbols = data[offset..offset + suffix_symbols_len].to_vec();
    offset += suffix_symbols_len;

    let c_strings = data[offset..].to_vec();

    let owner_symbols = AnsGenericBijection::<1024>.revert(c_owner_symbols);
    let suffix_symbols = AnsGenericBijection::<1024>.revert(c_suffix_symbols);

    let zstd = ZStdBijection;
    let strings_bytes = zstd.revert(c_strings);

    let sep_positions: Vec<usize> = strings_bytes
        .iter()
        .enumerate()
        .filter(|(_, &b)| b == 0)
        .map(|(i, _)| i)
        .collect();

    let top_owners_bytes = &strings_bytes[..sep_positions[0]];
    let other_owners_bytes = &strings_bytes[sep_positions[0] + 1..sep_positions[1]];
    let top_suffixes_bytes = &strings_bytes[sep_positions[1] + 1..sep_positions[2]];
    let other_suffixes_bytes = &strings_bytes[sep_positions[2] + 1..];

    let parse_strings = |bytes: &[u8]| -> Vec<String> {
        let s = String::from_utf8(bytes.to_vec()).unwrap();
        if s.is_empty() {
            vec![]
        } else {
            s.split('\n').map(|x| x.to_string()).collect()
        }
    };

    let top_owners = parse_strings(top_owners_bytes);
    let other_owners = parse_strings(other_owners_bytes);
    let top_suffixes = parse_strings(top_suffixes_bytes);
    let other_suffixes = parse_strings(other_suffixes_bytes);

    let mut dict_repo_owners = Vec::with_capacity(owner_symbols.len());
    let mut other_owner_iter = other_owners.into_iter();

    for symbol in owner_symbols {
        if symbol == OTHER_OWNER_SYMBOL {
            dict_repo_owners.push(other_owner_iter.next().expect("Missing 'other' owner"));
        } else {
            dict_repo_owners.push(top_owners[symbol as usize].clone());
        }
    }

    let mut dict_repo_suffixes = Vec::with_capacity(suffix_symbols.len());
    let mut other_suffix_iter = other_suffixes.into_iter();

    for symbol in suffix_symbols {
        if symbol == OTHER_SUFFIX_SYMBOL {
            dict_repo_suffixes.push(other_suffix_iter.next().expect("Missing 'other' suffix"));
        } else {
            dict_repo_suffixes.push(top_suffixes[symbol as usize].clone());
        }
    }

    (dict_repo_owners, dict_repo_suffixes)
}
