//! # cometkim Codec
//!
//! ## Core Strategy
//!
//! Assuming the original data is already sorted by timestamp,
//! this codec leverages the ordering to create a compact timeseries representation.
//!
//! ### Timestamp-implicit encoding
//!
//! Instead of storing the timestamp value for each event,
//! it uses the position in the event stream to infer the timestamp value,
//! and grouping events by the timestamp (per second).
//!
//! Only store the count of events per second, and the event data in order.
//!
//! It's slightly more efficent because encoding 24K interval counts is more compact
//! than encoding 1M per-event values, even with delta encoding, bit packing, etc.
//!
//! ### Out-of-order events
//!
//! There are some events that are out of order, but it's very rare (~0.2% in `data.json`).
//!
//! These are stored in separate "error batches" that are inserted back into the
//! correct position while decoding, make it possible to restore the input order.
//!
//! ### Key Optimizations
//!
//! - Use columnar layout
//! - Use varint for lengths (trivial)
//! - Use pcodec compression for numeric columns
//! - Split repo names into owner/suffix with dictionary encoding for owners (kjcao's approach)
//!
//! ## Binary Format
//!
//! ```
//! Header:
//!   base_id: varint
//!   base_timestamp: u32
//!   total_clean_events: varint
//!   interval_count: varint      // seconds span of clean stream
//!
//! Repo Dictionary:
//!   repo_dict_len: varint
//!   repo_dict_data: zstd[
//!     count: varint
//!     ids: pcodec[u64]
//!     owner_strings: [null-terminated strings]
//!     owner_indices: pcodec[u32]
//!     suffixes: [flag + optional string]
//!   ]
//!
//! Clean Timeseries:
//!   counts: varint, pcodec[u8]       // events per second
//!   types: varint, pcodec[u8]        // event type enums
//!   id_offsets: varint, pcodec[u32]  // offset from base_id
//!   repos: varint, pcodec[u32]       // repo dictionary indices
//!
//! Error Batches:
//!   batch_count: varint
//!   insert_positions: varint, pcodec[u32]
//!   event_counts: varint, pcodec[u32]
//!   types: varint, pcodec[u8]
//!   id_offsets: varint, pcodec[u32]       // offset from base_id
//!   repos: varing, pcodec[u32]
//!   ts_neg_deltas: varint, pcodec[u16]    // negative delta from insert_ts
//! ```
//!
//! Set COMETKIM_DEBUG=1 to see column size statistics.

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use pco::standalone::{simple_compress, simple_decompress};
use pco::ChunkConfig;
use std::collections::HashMap;
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const ZSTD_LEVEL: i32 = 22;

fn pco_config() -> ChunkConfig {
    let mut config = ChunkConfig::default();
    config.compression_level = 12;
    config.enable_8_bit = true; // pcodec is good for large [u8]
    config
}

fn debug_enabled() -> bool {
    std::env::var("COMETKIM_DEBUG").is_ok()
}

// GitHub events are well-known.
// All available events are listed on https://docs.github.com/en/rest/using-the-rest-api/github-event-types
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum GitHubEventType {
    CommitComment = 0,
    Create = 1,
    Delete = 2,
    Discussion = 3,
    Fork = 4,
    Gollum = 5,
    IssueComment = 6,
    Issues = 7,
    Member = 8,
    Public = 9,
    PullRequest = 10,
    PullRequestReview = 11,
    PullRequestReviewComment = 12,
    Push = 13,
    Release = 14,
    Watch = 15,
}

impl From<&str> for GitHubEventType {
    fn from(s: &str) -> Self {
        match s {
            "CommitCommentEvent" => GitHubEventType::CommitComment,
            "CreateEvent" => GitHubEventType::Create,
            "DeleteEvent" => GitHubEventType::Delete,
            "DiscussionEvent" => GitHubEventType::Discussion,
            "ForkEvent" => GitHubEventType::Fork,
            "GollumEvent" => GitHubEventType::Gollum,
            "IssueCommentEvent" => GitHubEventType::IssueComment,
            "IssuesEvent" => GitHubEventType::Issues,
            "MemberEvent" => GitHubEventType::Member,
            "PublicEvent" => GitHubEventType::Public,
            "PullRequestEvent" => GitHubEventType::PullRequest,
            "PullRequestReviewEvent" => GitHubEventType::PullRequestReview,
            "PullRequestReviewCommentEvent" => GitHubEventType::PullRequestReviewComment,
            "PushEvent" => GitHubEventType::Push,
            "ReleaseEvent" => GitHubEventType::Release,
            "WatchEvent" => GitHubEventType::Watch,
            _ => panic!("Unknown event type: {}", s),
        }
    }
}

impl From<GitHubEventType> for &'static str {
    fn from(val: GitHubEventType) -> Self {
        match val {
            GitHubEventType::CommitComment => "CommitCommentEvent",
            GitHubEventType::Create => "CreateEvent",
            GitHubEventType::Delete => "DeleteEvent",
            GitHubEventType::Discussion => "DiscussionEvent",
            GitHubEventType::Fork => "ForkEvent",
            GitHubEventType::Gollum => "GollumEvent",
            GitHubEventType::IssueComment => "IssueCommentEvent",
            GitHubEventType::Issues => "IssuesEvent",
            GitHubEventType::Member => "MemberEvent",
            GitHubEventType::Public => "PublicEvent",
            GitHubEventType::PullRequest => "PullRequestEvent",
            GitHubEventType::PullRequestReview => "PullRequestReviewEvent",
            GitHubEventType::PullRequestReviewComment => "PullRequestReviewCommentEvent",
            GitHubEventType::Push => "PushEvent",
            GitHubEventType::Release => "ReleaseEvent",
            GitHubEventType::Watch => "WatchEvent",
        }
    }
}

impl From<u8> for GitHubEventType {
    fn from(val: u8) -> Self {
        match val {
            0 => GitHubEventType::CommitComment,
            1 => GitHubEventType::Create,
            2 => GitHubEventType::Delete,
            3 => GitHubEventType::Discussion,
            4 => GitHubEventType::Fork,
            5 => GitHubEventType::Gollum,
            6 => GitHubEventType::IssueComment,
            7 => GitHubEventType::Issues,
            8 => GitHubEventType::Member,
            9 => GitHubEventType::Public,
            10 => GitHubEventType::PullRequest,
            11 => GitHubEventType::PullRequestReview,
            12 => GitHubEventType::PullRequestReviewComment,
            13 => GitHubEventType::Push,
            14 => GitHubEventType::Release,
            15 => GitHubEventType::Watch,
            _ => panic!("Unknown event type value: {}", val),
        }
    }
}

// Varint encoding/decoding helpers
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

fn parse_timestamp(ts: &str) -> Option<u32> {
    DateTime::parse_from_rfc3339(ts)
        .ok()
        .map(|dt| dt.timestamp() as u32)
}

fn format_timestamp(ts: u32) -> String {
    Utc.timestamp_opt(ts as i64, 0)
        .single()
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_default()
}

/// A parsed event for internal processing
#[derive(Clone, Debug)]
struct ParsedEvent {
    id: u64,
    event_type: u8,
    repo_idx: u32,
    timestamp: u32, // seconds since epoch
}

/// Error batch containing events that are out of timestamp order
#[derive(Clone, Debug)]
struct ErrorBatch {
    insert_position: usize,   // position in original input array
    events: Vec<ParsedEvent>, // events in this batch
}

/// Repo entry with owner/suffix split
#[derive(Clone, Debug)]
struct RepoEntry {
    id: u64,
    owner: String,
    suffix: String,
}

pub struct CometkimCodec;

impl CometkimCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for CometkimCodec {
    fn name(&self) -> &str {
        "cometkim"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        if events.is_empty() {
            return Ok(Bytes::new());
        }

        let pco_config = pco_config();

        // Step 1: Build repo dictionary with owner/suffix split
        let (repo_to_idx, repo_list) = build_repo_dictionary(events);

        // Step 2: Parse all events
        let parsed_events: Vec<ParsedEvent> = events
            .iter()
            .map(|(key, value)| {
                let id: u64 = key.id.parse().unwrap();
                let event_type = GitHubEventType::from(key.event_type.as_str()) as u8;
                let repo_key = (value.repo.id, value.repo.name.clone());
                let repo_idx = *repo_to_idx.get(&repo_key).unwrap();
                let timestamp = parse_timestamp(&value.created_at).unwrap();
                ParsedEvent {
                    id,
                    event_type,
                    repo_idx,
                    timestamp,
                }
            })
            .collect();

        // Step 3: Separate clean events and error batches
        let (clean_events, error_batches) = separate_errors(&parsed_events);

        // Step 4: Calculate base values (clean events are sorted, IDs are monotonically increasing)
        let base_id = clean_events.first().map(|e| e.id).unwrap_or(0);
        let base_timestamp = clean_events.first().map(|e| e.timestamp).unwrap_or(0);
        let max_timestamp = clean_events.last().map(|e| e.timestamp).unwrap_or(0);
        let interval_count = (max_timestamp - base_timestamp + 1) as usize;

        // Step 5: Group clean events by second
        let mut events_by_second: Vec<Vec<&ParsedEvent>> = vec![Vec::new(); interval_count];
        for event in &clean_events {
            let offset = (event.timestamp - base_timestamp) as usize;
            events_by_second[offset].push(event);
        }

        // Step 6: Encode everything with per-column compression
        let mut buf = Vec::new();

        // Header (uncompressed - small fixed size)
        encode_varint(base_id, &mut buf);
        buf.extend_from_slice(&base_timestamp.to_le_bytes());
        encode_varint(clean_events.len() as u64, &mut buf);
        encode_varint(interval_count as u64, &mut buf);

        // Repo dictionary (compressed with zstd)
        let mut repo_buf = Vec::new();
        encode_repo_dictionary(&pco_config, &repo_list, &mut repo_buf)?;
        let repo_compressed = zstd::encode_all(&repo_buf[..], ZSTD_LEVEL)?;
        encode_varint(repo_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&repo_compressed);

        // Clean timeseries columns
        // Column 1: Event counts per second
        let counts: Vec<u8> = events_by_second.iter().map(|s| s.len() as u8).collect();

        // Column 2: Type indices
        let mut all_types: Vec<u8> = Vec::new();
        for second_events in &events_by_second {
            for event in second_events {
                all_types.push(event.event_type);
            }
        }

        // Column 3: Event ID offsets (IDs are monotonically increasing, store as u32 offset from base_id)
        let mut all_id_offsets: Vec<u32> = Vec::new();
        for second_events in &events_by_second {
            for event in second_events {
                all_id_offsets.push((event.id - base_id) as u32);
            }
        }

        // Column 4: Repo indices
        let mut all_repo_indices: Vec<u32> = Vec::new();
        for second_events in &events_by_second {
            for event in second_events {
                all_repo_indices.push(event.repo_idx);
            }
        }

        // Use pcodec for numeric columns
        let counts_compressed = simple_compress(&counts, &pco_config)?;
        let types_compressed = simple_compress(&all_types, &pco_config)?;
        let ids_compressed = simple_compress(&all_id_offsets, &pco_config)?;
        let repos_compressed = simple_compress(&all_repo_indices, &pco_config)?;

        // Write columns with lengths
        encode_varint(counts_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&counts_compressed);
        encode_varint(types_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&types_compressed);
        encode_varint(ids_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&ids_compressed);
        encode_varint(repos_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&repos_compressed);

        // Error batches
        let err_start = buf.len();
        encode_error_batches(
            &pco_config,
            &error_batches,
            base_id,
            base_timestamp,
            &clean_events,
            &mut buf,
        )?;
        let err_size = buf.len() - err_start;

        if debug_enabled() {
            // Debug size breakdown
            eprintln!("=== Size Breakdown ===");
            eprintln!("Repo dict: {} bytes", repo_compressed.len());
            eprintln!("Counts: {} bytes", counts_compressed.len());
            eprintln!("Types: {} bytes", types_compressed.len());
            eprintln!("IDs: {} bytes", ids_compressed.len());
            eprintln!("Repo idx: {} bytes", repos_compressed.len());
            eprintln!("Errors: {} bytes", err_size);
        }

        Ok(Bytes::from(buf))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        let mut pos = 0;

        // Read header
        let base_id = decode_varint(bytes, &mut pos);
        let base_timestamp = u32::from_le_bytes(bytes[pos..pos + 4].try_into()?);
        pos += 4;
        let total_clean_events = decode_varint(bytes, &mut pos) as usize;
        let interval_count = decode_varint(bytes, &mut pos) as usize;

        // Read and decompress repo dictionary
        let repo_len = decode_varint(bytes, &mut pos) as usize;
        let repo_bytes = zstd::decode_all(&bytes[pos..pos + repo_len])?;
        pos += repo_len;
        let mut repo_pos = 0;
        let repo_list = decode_repo_dictionary(&repo_bytes, &mut repo_pos)?;

        // Read column data (pcodec compressed)
        let counts_len = decode_varint(bytes, &mut pos) as usize;
        let counts: Vec<u8> = simple_decompress(&bytes[pos..pos + counts_len])?;
        pos += counts_len;

        let types_len = decode_varint(bytes, &mut pos) as usize;
        let all_types: Vec<u8> = simple_decompress(&bytes[pos..pos + types_len])?;
        pos += types_len;

        let ids_len = decode_varint(bytes, &mut pos) as usize;
        let all_id_offsets: Vec<u32> = simple_decompress(&bytes[pos..pos + ids_len])?;
        pos += ids_len;

        let repos_len = decode_varint(bytes, &mut pos) as usize;
        let all_repo_indices: Vec<u32> = simple_decompress(&bytes[pos..pos + repos_len])?;
        pos += repos_len;

        // Parse clean timeseries from columns
        let mut clean_events: Vec<ParsedEvent> = Vec::with_capacity(total_clean_events);
        let mut event_pos = 0;

        for (second_offset, &count) in counts.iter().enumerate().take(interval_count) {
            let count = count as usize;

            if count > 0 {
                let timestamp = base_timestamp + second_offset as u32;

                for _ in 0..count {
                    let event_type = all_types[event_pos];
                    let id = base_id + all_id_offsets[event_pos] as u64;
                    let repo_idx = all_repo_indices[event_pos];
                    event_pos += 1;

                    clean_events.push(ParsedEvent {
                        id,
                        event_type,
                        repo_idx,
                        timestamp,
                    });
                }
            }
        }

        // Read and decompress error batches
        let error_batches =
            decode_error_batches(bytes, &mut pos, base_id, base_timestamp, &clean_events)?;

        // Reconstruct final events by inserting error batches
        let mut final_events = clean_events;

        // Process error batches in reverse order to maintain correct positions
        for batch in error_batches.into_iter().rev() {
            let insert_pos = batch.insert_position.min(final_events.len());
            for (i, event) in batch.events.into_iter().enumerate() {
                final_events.insert(insert_pos + i, event);
            }
        }

        // Convert to output format
        let result: Vec<(EventKey, EventValue)> = final_events
            .into_iter()
            .map(|event| {
                let repo_entry = &repo_list[event.repo_idx as usize];
                let repo_name = if repo_entry.suffix.is_empty() {
                    repo_entry.owner.clone()
                } else {
                    format!("{}/{}", repo_entry.owner, repo_entry.suffix)
                };
                let key = EventKey {
                    id: event.id.to_string(),
                    event_type: <&str>::from(GitHubEventType::from(event.event_type)).to_string(),
                };
                let value = EventValue {
                    repo: Repo {
                        id: repo_entry.id,
                        name: repo_name.clone(),
                        url: format!("https://api.github.com/repos/{}", repo_name),
                    },
                    created_at: format_timestamp(event.timestamp),
                };
                (key, value)
            })
            .collect();

        Ok(result)
    }
}

type RepoDict = (HashMap<(u64, String), u32>, Vec<RepoEntry>);

/// Build repo dictionary sorted by frequency for better index compression
/// with owner/suffix split for better compression
fn build_repo_dictionary(events: &[(EventKey, EventValue)]) -> RepoDict {
    // Count frequency of each repo
    let mut repo_counts: HashMap<(u64, String), usize> = HashMap::new();
    for (_, value) in events {
        let key = (value.repo.id, value.repo.name.clone());
        *repo_counts.entry(key).or_insert(0) += 1;
    }

    // Build repo entries
    let mut repo_entries: Vec<((u64, String), RepoEntry, usize)> = repo_counts
        .into_iter()
        .map(|(key, count)| {
            let parts: Vec<&str> = key.1.splitn(2, '/').collect();
            let (owner, suffix) = if parts.len() == 2 {
                (parts[0].to_string(), parts[1].to_string())
            } else {
                (key.1.clone(), String::new())
            };
            let entry = RepoEntry {
                id: key.0,
                owner,
                suffix,
            };
            (key, entry, count)
        })
        .collect();

    // Sort by frequency (descending) so frequent repos get small indices
    // (ID order saves 92KB on dict but loses 110KB on indices - frequency wins)
    repo_entries.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

    // Build index map and repo list
    let mut repo_to_idx: HashMap<(u64, String), u32> = HashMap::new();
    let mut repo_list: Vec<RepoEntry> = Vec::with_capacity(repo_entries.len());

    for (idx, (key, entry, _)) in repo_entries.into_iter().enumerate() {
        repo_to_idx.insert(key, idx as u32);
        repo_list.push(entry);
    }

    (repo_to_idx, repo_list)
}

/// Separate clean events from error batches
fn separate_errors(events: &[ParsedEvent]) -> (Vec<ParsedEvent>, Vec<ErrorBatch>) {
    let mut clean_events: Vec<ParsedEvent> = Vec::new();
    let mut error_batches: Vec<ErrorBatch> = Vec::new();

    let mut checkpoint: u32 = 0;
    let mut current_error_batch: Option<ErrorBatch> = None;

    for event in events.iter() {
        if event.timestamp < checkpoint {
            // This is an error event
            if let Some(ref mut batch) = current_error_batch {
                batch.events.push(event.clone());
            } else {
                // insert_position = number of clean events so far
                current_error_batch = Some(ErrorBatch {
                    insert_position: clean_events.len(),
                    events: vec![event.clone()],
                });
            }
        } else {
            // This is a clean event
            if let Some(batch) = current_error_batch.take() {
                error_batches.push(batch);
            }
            checkpoint = event.timestamp;
            clean_events.push(event.clone());
        }
    }

    // Don't forget the last error batch
    if let Some(batch) = current_error_batch {
        error_batches.push(batch);
    }

    (clean_events, error_batches)
}

/// Encode repo dictionary with improved owner/suffix split and preprocessing
fn encode_repo_dictionary(
    pco_config: &ChunkConfig,
    repo_list: &[RepoEntry],
    buf: &mut Vec<u8>,
) -> Result<(), Box<dyn Error>> {
    encode_varint(repo_list.len() as u64, buf);

    // Repo IDs - pcodec alone is best (492KB vs varint+zstd 761KB)
    let ids: Vec<u64> = repo_list.iter().map(|e| e.id).collect();
    let ids_compressed = simple_compress(&ids, pco_config)?;
    encode_varint(ids_compressed.len() as u64, buf);
    buf.extend_from_slice(&ids_compressed);

    // Build owner dictionary (frequency-sorted for better compression)
    let mut owner_counts: HashMap<&str, usize> = HashMap::new();
    for entry in repo_list {
        *owner_counts.entry(&entry.owner).or_insert(0) += 1;
    }
    let mut owner_list: Vec<&str> = owner_counts.keys().copied().collect();
    owner_list.sort_by(|a, b| owner_counts[b].cmp(&owner_counts[a]).then_with(|| a.cmp(b)));

    let owner_to_idx: HashMap<&str, u32> = owner_list
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as u32))
        .collect();

    // Write owner count
    encode_varint(owner_list.len() as u64, buf);

    // Write owner strings (null-terminated for efficient parsing)
    for owner in &owner_list {
        buf.extend_from_slice(owner.as_bytes());
        buf.push(0); // null terminator
    }

    // Write owner indices for each repo (pcodec compressed)
    let owner_indices: Vec<u32> = repo_list
        .iter()
        .map(|e| owner_to_idx[e.owner.as_str()])
        .collect();
    let owner_indices_compressed = simple_compress(&owner_indices, pco_config)?;
    encode_varint(owner_indices_compressed.len() as u64, buf);
    buf.extend_from_slice(&owner_indices_compressed);

    // Write suffixes with transformations (like kjcao's approach)
    for entry in repo_list {
        let owner = &entry.owner;
        let suffix = &entry.suffix;

        if suffix == owner {
            // Same as owner
            buf.push(0x01);
        } else if suffix.starts_with(owner) {
            // Starts with owner
            let remainder = &suffix[owner.len()..];
            if let Some(first_char) = remainder.chars().next() {
                if matches!(first_char, '-' | '_' | '.') {
                    buf.push(0x02);
                    buf.extend_from_slice(remainder.as_bytes());
                    buf.push(0);
                    continue;
                }
            }
            // Fall through to raw
            buf.push(0x00);
            buf.extend_from_slice(suffix.as_bytes());
            buf.push(0);
        } else {
            // Raw string
            buf.push(0x00);
            buf.extend_from_slice(suffix.as_bytes());
            buf.push(0);
        }
    }
    Ok(())
}

/// Decode repo dictionary
fn decode_repo_dictionary(bytes: &[u8], pos: &mut usize) -> Result<Vec<RepoEntry>, Box<dyn Error>> {
    let count = decode_varint(bytes, pos) as usize;

    // Decode repo IDs with pcodec
    let ids_len = decode_varint(bytes, pos) as usize;
    let repo_ids: Vec<u64> = simple_decompress(&bytes[*pos..*pos + ids_len])?;
    *pos += ids_len;

    // Read owner dictionary
    let owner_count = decode_varint(bytes, pos) as usize;
    let mut owner_list: Vec<String> = Vec::with_capacity(owner_count);
    for _ in 0..owner_count {
        let null_pos = bytes[*pos..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(bytes.len() - *pos);
        let owner = String::from_utf8(bytes[*pos..*pos + null_pos].to_vec())?;
        *pos += null_pos + 1; // Skip null terminator
        owner_list.push(owner);
    }

    // Read owner indices (pcodec compressed)
    let owner_indices_len = decode_varint(bytes, pos) as usize;
    let owner_indices_u32: Vec<u32> = simple_decompress(&bytes[*pos..*pos + owner_indices_len])?;
    *pos += owner_indices_len;
    let owner_indices: Vec<usize> = owner_indices_u32.iter().map(|&i| i as usize).collect();

    // Read suffixes with transformations
    let mut repo_list: Vec<RepoEntry> = Vec::with_capacity(count);
    for i in 0..count {
        let owner_idx = owner_indices[i];
        let owner = if owner_idx < owner_list.len() {
            owner_list[owner_idx].clone()
        } else {
            String::new()
        };

        let flag = bytes[*pos];
        *pos += 1;

        let suffix = match flag {
            0x01 => {
                // Same as owner
                owner.clone()
            }
            0x02 => {
                // Starts with owner + remainder
                let null_pos = bytes[*pos..]
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(bytes.len() - *pos);
                let remainder = String::from_utf8(bytes[*pos..*pos + null_pos].to_vec())?;
                *pos += null_pos + 1;
                format!("{}{}", owner, remainder)
            }
            _ => {
                // Raw string
                let null_pos = bytes[*pos..]
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(bytes.len() - *pos);
                let raw = String::from_utf8(bytes[*pos..*pos + null_pos].to_vec())?;
                *pos += null_pos + 1;
                raw
            }
        };

        repo_list.push(RepoEntry {
            id: repo_ids[i],
            owner,
            suffix,
        });
    }

    Ok(repo_list)
}

fn encode_error_batches(
    pco_config: &ChunkConfig,
    batches: &[ErrorBatch],
    base_id: u64,
    base_timestamp: u32,
    clean_events: &[ParsedEvent],
    buf: &mut Vec<u8>,
) -> Result<(), Box<dyn Error>> {
    encode_varint(batches.len() as u64, buf);
    if batches.is_empty() {
        return Ok(());
    }

    // Collect all batch metadata
    let mut insert_positions: Vec<u32> = Vec::new();
    let mut event_counts: Vec<u32> = Vec::new();

    // Collect all event data (one entry per event)
    let mut all_types: Vec<u8> = Vec::new();
    let mut all_id_offsets: Vec<u32> = Vec::new();
    let mut all_repos: Vec<u32> = Vec::new();
    let mut all_ts_neg_deltas: Vec<u16> = Vec::new();

    for batch in batches {
        insert_positions.push(batch.insert_position as u32);
        event_counts.push(batch.events.len() as u32);

        // Get the timestamp at insert position (or base if at start)
        let insert_ts = if batch.insert_position > 0 && batch.insert_position <= clean_events.len()
        {
            clean_events[batch.insert_position - 1].timestamp
        } else {
            base_timestamp
        };

        for event in &batch.events {
            all_types.push(event.event_type);
            all_id_offsets.push((event.id - base_id) as u32);
            all_repos.push(event.repo_idx);
            // Store absolute negative delta (max ~2500)
            let neg_delta = insert_ts.saturating_sub(event.timestamp);
            all_ts_neg_deltas.push(neg_delta as u16);
        }
    }

    let insert_pos_comp = simple_compress(&insert_positions, pco_config)?;
    let event_counts_comp = simple_compress(&event_counts, pco_config)?;
    let types_comp = simple_compress(&all_types, pco_config)?;
    let ids_comp = simple_compress(&all_id_offsets, pco_config)?;
    let repos_comp = simple_compress(&all_repos, pco_config)?;
    let ts_comp = simple_compress(&all_ts_neg_deltas, pco_config)?;

    encode_varint(insert_pos_comp.len() as u64, buf);
    buf.extend_from_slice(&insert_pos_comp);
    encode_varint(event_counts_comp.len() as u64, buf);
    buf.extend_from_slice(&event_counts_comp);
    encode_varint(types_comp.len() as u64, buf);
    buf.extend_from_slice(&types_comp);
    encode_varint(ids_comp.len() as u64, buf);
    buf.extend_from_slice(&ids_comp);
    encode_varint(repos_comp.len() as u64, buf);
    buf.extend_from_slice(&repos_comp);
    encode_varint(ts_comp.len() as u64, buf);
    buf.extend_from_slice(&ts_comp);

    Ok(())
}

fn decode_error_batches(
    bytes: &[u8],
    pos: &mut usize,
    base_id: u64,
    base_timestamp: u32,
    clean_events: &[ParsedEvent],
) -> Result<Vec<ErrorBatch>, Box<dyn Error>> {
    let batch_count = decode_varint(bytes, pos) as usize;
    if batch_count == 0 {
        return Ok(Vec::new());
    }

    // Read all compressed sections
    let insert_pos_len = decode_varint(bytes, pos) as usize;
    let insert_positions: Vec<u32> = simple_decompress(&bytes[*pos..*pos + insert_pos_len])?;
    *pos += insert_pos_len;

    let event_counts_len = decode_varint(bytes, pos) as usize;
    let event_counts: Vec<u32> = simple_decompress(&bytes[*pos..*pos + event_counts_len])?;
    *pos += event_counts_len;

    let types_len = decode_varint(bytes, pos) as usize;
    let all_types: Vec<u8> = simple_decompress(&bytes[*pos..*pos + types_len])?;
    *pos += types_len;

    let ids_len = decode_varint(bytes, pos) as usize;
    let all_id_offsets: Vec<u32> = simple_decompress(&bytes[*pos..*pos + ids_len])?;
    *pos += ids_len;

    let repos_len = decode_varint(bytes, pos) as usize;
    let all_repos: Vec<u32> = simple_decompress(&bytes[*pos..*pos + repos_len])?;
    *pos += repos_len;

    let ts_len = decode_varint(bytes, pos) as usize;
    let all_ts_neg_deltas: Vec<u16> = simple_decompress(&bytes[*pos..*pos + ts_len])?;
    *pos += ts_len;

    // Reconstruct batches
    let mut batches: Vec<ErrorBatch> = Vec::with_capacity(batch_count);
    let mut event_offset: usize = 0;

    for batch_idx in 0..batch_count {
        let insert_position = insert_positions[batch_idx] as usize;
        let event_count = event_counts[batch_idx] as usize;

        // Get the timestamp at insert position (or base if at start)
        let insert_ts = if insert_position > 0 && insert_position <= clean_events.len() {
            clean_events[insert_position - 1].timestamp
        } else {
            base_timestamp
        };

        // Reconstruct events
        let mut events: Vec<ParsedEvent> = Vec::with_capacity(event_count);
        for _ in 0..event_count {
            events.push(ParsedEvent {
                id: base_id + all_id_offsets[event_offset] as u64,
                event_type: all_types[event_offset],
                repo_idx: all_repos[event_offset],
                timestamp: insert_ts - all_ts_neg_deltas[event_offset] as u32,
            });
            event_offset += 1;
        }

        batches.push(ErrorBatch {
            insert_position,
            events,
        });
    }

    Ok(batches)
}
