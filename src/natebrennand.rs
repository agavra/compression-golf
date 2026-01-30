//! Arrow IPC codec using dictionary encoding for low-cardinality fields.
//!
//! Strategy:
//! - Sort by event_id for optimal delta encoding
//! - Dictionary encode event_type (14 unique values)
//! - Use marisa trie for repo.name (prefix sharing for owner/repo patterns)
//! - Store repo.id as UInt64
//! - Store first event_id in header, deltas as UInt8 (max delta is 251)
//! - Store first timestamp in header, deltas as Int16 (range -2540 to +2540)
//! - Reconstruct repo.url from repo.name during decode
//! - Use Arrow IPC with zstd compression
//!
//! Set NATE_DEBUG=1 to see column size statistics.

use arrow::array::{
    Array, ArrayRef, DictionaryArray, Int16Array, RecordBatch, StringArray, UInt32Array,
    UInt8Array,
};
use arrow::datatypes::{DataType, Field, Int8Type, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use rsmarisa::base::{CacheLevel, NumTries};
use rsmarisa::grimoire::io::{Reader, Writer};
use rsmarisa::{Agent, Keyset, Trie};
// Note: Arrow IPC's built-in zstd uses level 3. We disable it and apply level 22 ourselves.
const ZSTD_LEVEL: i32 = 22;
use bytes::Bytes;
use chrono::DateTime;
use std::collections::HashSet;
use std::error::Error;
use std::io::Cursor;
use std::sync::Arc;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

pub struct NatebrennandCodec;

impl NatebrennandCodec {
    pub fn new() -> Self {
        Self
    }
}

fn debug_enabled() -> bool {
    std::env::var("NATE_DEBUG").is_ok()
}

fn print_column_stats(batch: &RecordBatch, trie_compressed_size: usize) {
    let num_rows = batch.num_rows();
    eprintln!("\n=== Per-Column Compressed Size Estimates ===");
    eprintln!("Total rows: {}", num_rows);
    eprintln!("{:<18} {:>10} {:>10} {:>8} {:>10}", "Column", "Raw", "Zstd", "Ratio", "B/Row");
    eprintln!("{}", "-".repeat(62));

    let mut total_raw = 0usize;
    let mut total_compressed = 0usize;

    for (i, field) in batch.schema().fields().iter().enumerate() {
        let col = batch.column(i);
        let raw_size = col.get_array_memory_size();

        // Serialize just this column to estimate compressed size
        let single_schema = Arc::new(Schema::new(vec![field.clone()]));
        let single_batch = RecordBatch::try_new(
            single_schema.clone(),
            vec![col.clone()],
        ).unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &single_schema).unwrap();
            writer.write(&single_batch).unwrap();
            writer.finish().unwrap();
        }
        let compressed = zstd::encode_all(buf.as_slice(), ZSTD_LEVEL).unwrap();
        let compressed_size = compressed.len();

        total_raw += raw_size;
        total_compressed += compressed_size;

        eprintln!(
            "{:<18} {:>10} {:>10} {:>7.1}% {:>10.2}",
            field.name(),
            raw_size,
            compressed_size,
            100.0 * compressed_size as f64 / raw_size as f64,
            compressed_size as f64 / num_rows as f64
        );
    }

    // Add trie as a "column"
    eprintln!(
        "{:<18} {:>10} {:>10} {:>7}  {:>10.2}",
        "repo_name (trie)",
        "-",
        trie_compressed_size,
        "-",
        trie_compressed_size as f64 / num_rows as f64
    );
    total_compressed += trie_compressed_size;

    eprintln!("{}", "-".repeat(62));
    eprintln!(
        "{:<18} {:>10} {:>10} {:>7.1}% {:>10.2}",
        "TOTAL (cols)",
        total_raw,
        total_compressed,
        100.0 * total_compressed as f64 / total_raw as f64,
        total_compressed as f64 / num_rows as f64
    );
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

fn build_schema() -> Schema {
    Schema::new(vec![
        Field::new(
            "event_type",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new(
            "event_id_delta",
            DataType::UInt8,
            false,
        ),
        Field::new(
            "repo_id",
            DataType::UInt32, // Max repo_id is ~910M, fits in u32
            false,
        ),
        Field::new(
            "repo_name_idx",
            DataType::UInt32, // Index into marisa trie
            false,
        ),
        Field::new(
            "created_at_delta",
            DataType::Int16,
            false,
        ),
    ])
}

/// Header stored outside Arrow: first event_id (u64) + first timestamp (i64) + trie_size (u32)
struct Header {
    first_event_id: u64,
    first_timestamp: i64,
    trie_size: u32,
}

const HEADER_SIZE: usize = 20; // 8 + 8 + 4

impl Header {
    fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.first_event_id.to_le_bytes());
        buf[8..16].copy_from_slice(&self.first_timestamp.to_le_bytes());
        buf[16..20].copy_from_slice(&self.trie_size.to_le_bytes());
        buf
    }

    fn decode(bytes: &[u8]) -> Self {
        let first_event_id = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let first_timestamp = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let trie_size = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        Self { first_event_id, first_timestamp, trie_size }
    }
}

/// Build a marisa trie from unique repo names and return (trie_bytes, trie)
fn build_repo_name_trie(repo_names: &[&str]) -> Result<(Vec<u8>, Trie), Box<dyn Error>> {
    // Collect unique names
    let unique_names: HashSet<&str> = repo_names.iter().copied().collect();

    // Build keyset
    let mut keyset = Keyset::new();
    for name in &unique_names {
        keyset.push_back_str(name).map_err(|e| format!("keyset push error: {:?}", e))?;
    }

    // Build trie with 1 trie + TINY cache (TextTail default is best for our data)
    let mut trie = Trie::new();
    let config = NumTries::MIN | CacheLevel::Tiny as u32;
    trie.build(&mut keyset, config as i32);

    // Serialize trie to bytes using rsmarisa's Writer
    let mut writer = Writer::from_vec(Vec::new());
    trie.write(&mut writer)?;
    let trie_bytes = writer.into_inner()?;

    Ok((trie_bytes, trie))
}

/// Look up a repo name in the trie and return its ID
fn trie_lookup(trie: &Trie, name: &str) -> Result<u32, Box<dyn Error>> {
    let mut agent = Agent::new();
    agent.set_query_str(name);
    if trie.lookup(&mut agent) {
        Ok(agent.key().id() as u32)
    } else {
        Err(format!("repo name not found in trie: {}", name).into())
    }
}

/// Reverse lookup: get repo name from trie ID
fn trie_reverse_lookup(trie: &Trie, id: u32) -> Result<String, Box<dyn Error>> {
    let mut agent = Agent::new();
    agent.set_query_id(id as usize);
    trie.reverse_lookup(&mut agent);
    Ok(agent.key().as_str().to_string())
}

fn compute_u8_deltas(values: &[u64]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut deltas = Vec::with_capacity(values.len());
    deltas.push(0u8); // First value stored in header
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
    deltas.push(0i16); // First value stored in header
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

fn encode_events(events: &[(EventKey, EventValue)]) -> Result<(Header, Vec<u8>, RecordBatch), Box<dyn Error>> {
    let schema = Arc::new(build_schema());

    // Sort by event_id for optimal delta encoding (deltas fit in u8)
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
    let repo_ids: Vec<u32> = sorted_indices
        .iter()
        .map(|&i| events[i].1.repo.id as u32)
        .collect();
    let repo_names: Vec<&str> = sorted_indices
        .iter()
        .map(|&i| events[i].1.repo.name.as_str())
        .collect();
    let timestamps: Vec<i64> = sorted_indices
        .iter()
        .map(|&i| parse_timestamp(&events[i].1.created_at))
        .collect();

    // Build marisa trie for repo names
    let (trie_bytes, trie) = build_repo_name_trie(&repo_names)?;

    // Create header with first values and trie size
    let header = Header {
        first_event_id: event_ids[0],
        first_timestamp: timestamps[0],
        trie_size: trie_bytes.len() as u32,
    };

    // Delta encode event_ids (u8) and timestamps (i16)
    let event_id_deltas = compute_u8_deltas(&event_ids);
    let timestamp_deltas = compute_i16_deltas(&timestamps);

    // Look up each repo name in trie to get its ID
    let repo_name_ids: Vec<u32> = repo_names
        .iter()
        .map(|name| trie_lookup(&trie, name).unwrap())
        .collect();

    // Build dictionary-encoded event_type array (Int8 keys - only 14 unique types)
    let event_type_array: DictionaryArray<Int8Type> = event_types.into_iter().collect();

    // Build primitive arrays
    let event_id_array = UInt8Array::from(event_id_deltas);
    let repo_id_array = UInt32Array::from(repo_ids);
    let repo_name_idx_array = UInt32Array::from(repo_name_ids);
    let timestamp_array = Int16Array::from(timestamp_deltas);

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(event_type_array) as ArrayRef,
            Arc::new(event_id_array) as ArrayRef,
            Arc::new(repo_id_array) as ArrayRef,
            Arc::new(repo_name_idx_array) as ArrayRef,
            Arc::new(timestamp_array) as ArrayRef,
        ],
    )?;

    Ok((header, trie_bytes, batch))
}

fn decode_batch(header: &Header, trie: &Trie, batch: &RecordBatch) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
    let event_type_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<DictionaryArray<Int8Type>>()
        .ok_or("Failed to downcast event_type")?;

    let event_id_delta_array = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or("Failed to downcast event_id_delta")?;

    let repo_id_array = batch
        .column(2)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or("Failed to downcast repo_id")?;

    let repo_name_idx_array = batch
        .column(3)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or("Failed to downcast repo_name_idx")?;

    let timestamp_delta_array = batch
        .column(4)
        .as_any()
        .downcast_ref::<Int16Array>()
        .ok_or("Failed to downcast created_at_delta")?;

    // Restore values from deltas
    let event_id_deltas: Vec<u8> = event_id_delta_array.values().iter().copied().collect();
    let timestamp_deltas: Vec<i16> = timestamp_delta_array.values().iter().copied().collect();
    let event_ids = restore_u64_from_deltas(header.first_event_id, &event_id_deltas);
    let timestamps = restore_i64_from_deltas(header.first_timestamp, &timestamp_deltas);

    // Get string values from dictionary arrays
    let event_type_values = event_type_array
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("Failed to get event_type values")?;

    let events: Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> = (0..batch.num_rows())
        .map(|i| {
            let event_type_idx = event_type_array.keys().value(i) as usize;
            let event_type = event_type_values.value(event_type_idx).to_string();

            let event_id = event_ids[i];
            let repo_id = repo_id_array.value(i) as u64;

            // Reverse lookup repo name from trie
            let repo_name_id = repo_name_idx_array.value(i);
            let repo_name = trie_reverse_lookup(trie, repo_name_id)?;

            let timestamp = timestamps[i];

            let repo_url = format!("https://api.github.com/repos/{}", repo_name);

            Ok((
                EventKey {
                    id: event_id.to_string(),
                    event_type,
                },
                EventValue {
                    repo: Repo {
                        id: repo_id,
                        name: repo_name,
                        url: repo_url,
                    },
                    created_at: format_timestamp(timestamp),
                },
            ))
        })
        .collect();

    events
}

impl EventCodec for NatebrennandCodec {
    fn name(&self) -> &str {
        "natebrennand"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let (header, trie_bytes, batch) = encode_events(events)?;

        // Compute trie compressed size for stats
        let trie_compressed_size = zstd::encode_all(trie_bytes.as_slice(), ZSTD_LEVEL)?.len();

        if debug_enabled() {
            print_column_stats(&batch, trie_compressed_size);
        }

        // Write Arrow IPC stream without compression (we'll apply zstd 22 ourselves)
        let mut ipc_buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut ipc_buf, &batch.schema())?;
            writer.write(&batch)?;
            writer.finish()?;
        }

        if debug_enabled() {
            let ipc_compressed = zstd::encode_all(ipc_buf.as_slice(), ZSTD_LEVEL)?;
            eprintln!("Trie: {} -> {} bytes ({:.1}%)",
                trie_bytes.len(), trie_compressed_size,
                100.0 * trie_compressed_size as f64 / trie_bytes.len() as f64);
            eprintln!("IPC:  {} -> {} bytes ({:.1}%)",
                ipc_buf.len(), ipc_compressed.len(),
                100.0 * ipc_compressed.len() as f64 / ipc_buf.len() as f64);
        }

        // Compress trie + IPC together with zstd level 22
        let mut uncompressed = Vec::with_capacity(trie_bytes.len() + ipc_buf.len());
        uncompressed.extend_from_slice(&trie_bytes);
        uncompressed.extend_from_slice(&ipc_buf);
        let compressed = zstd::encode_all(uncompressed.as_slice(), ZSTD_LEVEL)?;

        // Final output: header + compressed(trie + IPC)
        let mut buf = Vec::with_capacity(HEADER_SIZE + compressed.len());
        buf.extend_from_slice(&header.encode());
        buf.extend_from_slice(&compressed);

        if debug_enabled() {
            eprintln!("Compressed size (zstd {}): {} bytes", ZSTD_LEVEL, buf.len());
            eprintln!("Compressed bytes/row: {:.2}", buf.len() as f64 / events.len() as f64);
        }

        Ok(Bytes::from(buf))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        // Read header
        let header = Header::decode(&bytes[0..HEADER_SIZE]);

        // Decompress trie + IPC data
        let decompressed = zstd::decode_all(&bytes[HEADER_SIZE..])?;

        // Split into trie and IPC portions
        let trie_bytes = &decompressed[0..header.trie_size as usize];
        let ipc_buf = &decompressed[header.trie_size as usize..];

        // Load trie from bytes
        let mut trie = Trie::new();
        let mut reader = Reader::from_bytes(trie_bytes);
        trie.read(&mut reader)?;

        // Read Arrow IPC stream
        let cursor = Cursor::new(ipc_buf);
        let reader = StreamReader::try_new(cursor, None)?;

        let mut all_events = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            let events = decode_batch(&header, &trie, &batch)?;
            all_events.extend(events);
        }

        // Sort by EventKey to match expected output
        all_events.sort_by(|a, b| a.0.cmp(&b.0));

        Ok(all_events)
    }
}
