//! # cometkim Codec
//!
//! **Strategy:**
//! Assuming all data is correctly ordered by `created_at` field.
//! It mostly is. There are exceptions in data but its rate is pretty low. (0.1218% in the sample data)
//!
//! Push abnormal items into separate block.
//! The decoder uses these blocks to perform post-processing for error correction.
//!
//! **Binary Format:**
//!
//! Header:
//!   base_id: u64
//!   base_timestamp: u32 (seconds since epoch)
//!   total_clean_events: varint
//!   interval_count: varint (seconds span of clean stream)
//!
//! Repo Dictionary:
//!   count: varint
//!   repo_id_deltas: [signed_varint; count]  // first is absolute
//!   name_lengths: [varint; count]
//!   names_concat: [u8]
//!
//! Clean Timeseries (per second):
//!   event_count: u8
//!   if event_count > 0:
//!     type_indices: [u8; event_count]
//!     id_deltas: [signed_varint; event_count]
//!     repo_indices: [varint; event_count]
//!
//! Error Batches:
//!   batch_count: varint
//!   for each batch:
//!     insert_position: varint
//!     total_event_count: varint
//!     interval_count: varint
//!     timestamp_offsets: [varint; interval_count]
//!     events_per_interval: [varint; interval_count]
//!     type_indices: [u8; total_event_count]
//!     event_ids: [u64; total_event_count]  // absolute IDs
//!     repo_indices: [varint; total_event_count]

use bytes::Bytes;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::HashMap;
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

const ZSTD_LEVEL: i32 = 22;

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

fn encode_signed_varint(value: i64, buf: &mut Vec<u8>) {
    let encoded = ((value << 1) ^ (value >> 63)) as u64;
    encode_varint(encoded, buf);
}

fn decode_signed_varint(bytes: &[u8], pos: &mut usize) -> i64 {
    let encoded = decode_varint(bytes, pos);
    ((encoded >> 1) as i64) ^ (-((encoded & 1) as i64))
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

/// Grouped events by timestamp within an error batch
struct ErrorBatchInterval {
    timestamp_offset: u32, // offset from base_timestamp
    event_count: usize,
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

        // Step 1: Build repo dictionary
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

        // Step 4: Calculate base values
        let base_id = clean_events.iter().map(|e| e.id).min().unwrap_or(0);
        let base_timestamp = clean_events.iter().map(|e| e.timestamp).min().unwrap_or(0);
        let max_timestamp = clean_events.iter().map(|e| e.timestamp).max().unwrap_or(0);
        let interval_count = if max_timestamp >= base_timestamp {
            (max_timestamp - base_timestamp + 1) as usize
        } else {
            0
        };

        // Step 5: Group clean events by second
        let mut events_by_second: Vec<Vec<&ParsedEvent>> = vec![Vec::new(); interval_count];
        for event in &clean_events {
            let offset = (event.timestamp - base_timestamp) as usize;
            events_by_second[offset].push(event);
        }

        // Step 6: Encode everything with per-column zstd compression
        let mut buf = Vec::new();

        // Header (uncompressed - small fixed size)
        buf.extend_from_slice(&base_id.to_le_bytes());
        buf.extend_from_slice(&base_timestamp.to_le_bytes());
        encode_varint(clean_events.len() as u64, &mut buf);
        encode_varint(interval_count as u64, &mut buf);

        // Repo dictionary (compressed as a block)
        let mut repo_buf = Vec::new();
        encode_repo_dictionary(&repo_list, &mut repo_buf);
        let repo_compressed = zstd::encode_all(&repo_buf[..], ZSTD_LEVEL)?;
        encode_varint(repo_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&repo_compressed);

        // Clean timeseries - encode each column separately for better compression
        // Column 1: Event counts per second
        let mut counts_buf = Vec::new();
        for second_events in &events_by_second {
            counts_buf.push(second_events.len() as u8);
        }

        // Column 2: Type indices (all events, flattened)
        let mut types_buf = Vec::new();
        for second_events in &events_by_second {
            for event in second_events {
                types_buf.push(event.event_type);
            }
        }

        // Column 3: ID deltas (all events, flattened)
        let mut ids_buf = Vec::new();
        let mut prev_id = base_id as i64;
        for second_events in &events_by_second {
            for event in second_events {
                let delta = event.id as i64 - prev_id;
                encode_signed_varint(delta, &mut ids_buf);
                prev_id = event.id as i64;
            }
        }

        // Column 4: Repo indices (all events, flattened)
        let mut repos_buf = Vec::new();
        for second_events in &events_by_second {
            for event in second_events {
                encode_varint(event.repo_idx as u64, &mut repos_buf);
            }
        }

        // Compress each column with zstd
        let counts_compressed = zstd::encode_all(&counts_buf[..], ZSTD_LEVEL)?;
        let types_compressed = zstd::encode_all(&types_buf[..], ZSTD_LEVEL)?;
        let ids_compressed = zstd::encode_all(&ids_buf[..], ZSTD_LEVEL)?;
        let repos_compressed = zstd::encode_all(&repos_buf[..], ZSTD_LEVEL)?;

        // Write compressed columns with lengths
        encode_varint(counts_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&counts_compressed);
        encode_varint(types_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&types_compressed);
        encode_varint(ids_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&ids_compressed);
        encode_varint(repos_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&repos_compressed);

        // Error batches (compressed as a block)
        let mut err_buf = Vec::new();
        encode_varint(error_batches.len() as u64, &mut err_buf);
        for batch in &error_batches {
            encode_error_batch(batch, base_id, base_timestamp, &mut err_buf);
        }
        let err_compressed = zstd::encode_all(&err_buf[..], ZSTD_LEVEL)?;
        encode_varint(err_compressed.len() as u64, &mut buf);
        buf.extend_from_slice(&err_compressed);

        Ok(Bytes::from(buf))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }

        let mut pos = 0;

        // Read header (uncompressed)
        let base_id = u64::from_le_bytes(bytes[pos..pos + 8].try_into()?);
        pos += 8;
        let base_timestamp = u32::from_le_bytes(bytes[pos..pos + 4].try_into()?);
        pos += 4;
        let total_clean_events = decode_varint(bytes, &mut pos) as usize;
        let interval_count = decode_varint(bytes, &mut pos) as usize;

        // Read and decompress repo dictionary block
        let repo_len = decode_varint(bytes, &mut pos) as usize;
        let repo_compressed = &bytes[pos..pos + repo_len];
        pos += repo_len;
        let repo_bytes = zstd::decode_all(repo_compressed)?;
        let mut repo_pos = 0;
        let repo_list = decode_repo_dictionary(&repo_bytes, &mut repo_pos)?;

        // Read and decompress column data
        let counts_len = decode_varint(bytes, &mut pos) as usize;
        let counts_bytes = zstd::decode_all(&bytes[pos..pos + counts_len])?;
        pos += counts_len;

        let types_len = decode_varint(bytes, &mut pos) as usize;
        let types_bytes = zstd::decode_all(&bytes[pos..pos + types_len])?;
        pos += types_len;

        let ids_len = decode_varint(bytes, &mut pos) as usize;
        let ids_bytes = zstd::decode_all(&bytes[pos..pos + ids_len])?;
        pos += ids_len;

        let repos_len = decode_varint(bytes, &mut pos) as usize;
        let repos_bytes = zstd::decode_all(&bytes[pos..pos + repos_len])?;
        pos += repos_len;

        // Parse clean timeseries from columns
        let mut clean_events: Vec<ParsedEvent> = Vec::with_capacity(total_clean_events);
        let mut prev_id = base_id as i64;
        let mut types_pos = 0;
        let mut ids_pos = 0;
        let mut repos_pos = 0;

        for (second_offset, &count_byte) in counts_bytes.iter().enumerate().take(interval_count) {
            let count = count_byte as usize;

            if count > 0 {
                let timestamp = base_timestamp + second_offset as u32;

                for _ in 0..count {
                    let event_type = types_bytes[types_pos];
                    types_pos += 1;

                    let delta = decode_signed_varint(&ids_bytes, &mut ids_pos);
                    prev_id += delta;
                    let id = prev_id as u64;

                    let repo_idx = decode_varint(&repos_bytes, &mut repos_pos) as u32;

                    clean_events.push(ParsedEvent {
                        id,
                        event_type,
                        repo_idx,
                        timestamp,
                    });
                }
            }
        }

        // Read and decompress error batches block
        let err_len = decode_varint(bytes, &mut pos) as usize;
        let err_compressed = &bytes[pos..pos + err_len];
        let err_bytes = zstd::decode_all(err_compressed)?;
        let mut err_pos = 0;

        let batch_count = decode_varint(&err_bytes, &mut err_pos) as usize;
        let mut error_batches: Vec<ErrorBatch> = Vec::with_capacity(batch_count);

        for _ in 0..batch_count {
            let batch = decode_error_batch(&err_bytes, &mut err_pos, base_id, base_timestamp)?;
            error_batches.push(batch);
        }

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
                let (repo_id, repo_name) = &repo_list[event.repo_idx as usize];
                let key = EventKey {
                    id: event.id.to_string(),
                    event_type: <&str>::from(GitHubEventType::from(event.event_type)).to_string(),
                };
                let value = EventValue {
                    repo: Repo {
                        id: *repo_id,
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

type RepoDict = (HashMap<(u64, String), u32>, Vec<(u64, String)>);

/// Build repo dictionary sorted by repo_id for delta compression
fn build_repo_dictionary(events: &[(EventKey, EventValue)]) -> RepoDict {
    let mut repo_set: HashMap<(u64, String), u32> = HashMap::new();

    for (_, value) in events {
        let key = (value.repo.id, value.repo.name.clone());
        repo_set.entry(key).or_insert(0);
    }

    // Sort by repo_id for better delta compression
    let mut repo_list: Vec<(u64, String)> = repo_set.keys().cloned().collect();
    repo_list.sort_by_key(|(id, _)| *id);

    // Build index map
    let repo_to_idx: HashMap<(u64, String), u32> = repo_list
        .iter()
        .enumerate()
        .map(|(idx, key)| (key.clone(), idx as u32))
        .collect();

    (repo_to_idx, repo_list)
}

/// Separate clean events from error batches
/// insert_position is tracked as the number of clean events before the error batch
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

/// Encode repo dictionary to buffer
fn encode_repo_dictionary(repo_list: &[(u64, String)], buf: &mut Vec<u8>) {
    encode_varint(repo_list.len() as u64, buf);

    // Repo ID deltas (first is absolute, rest are deltas)
    let mut prev_id: i64 = 0;
    for (repo_id, _) in repo_list {
        let delta = *repo_id as i64 - prev_id;
        encode_signed_varint(delta, buf);
        prev_id = *repo_id as i64;
    }

    // Name lengths
    for (_, name) in repo_list {
        encode_varint(name.len() as u64, buf);
    }

    // Names concatenated
    for (_, name) in repo_list {
        buf.extend_from_slice(name.as_bytes());
    }
}

/// Decode repo dictionary from buffer
fn decode_repo_dictionary(
    bytes: &[u8],
    pos: &mut usize,
) -> Result<Vec<(u64, String)>, Box<dyn Error>> {
    let count = decode_varint(bytes, pos) as usize;

    // Decode repo ID deltas
    let mut repo_ids: Vec<u64> = Vec::with_capacity(count);
    let mut prev_id: i64 = 0;
    for _ in 0..count {
        let delta = decode_signed_varint(bytes, pos);
        prev_id += delta;
        repo_ids.push(prev_id as u64);
    }

    // Decode name lengths
    let mut name_lengths: Vec<usize> = Vec::with_capacity(count);
    for _ in 0..count {
        name_lengths.push(decode_varint(bytes, pos) as usize);
    }

    // Decode names
    let mut repo_list: Vec<(u64, String)> = Vec::with_capacity(count);
    for i in 0..count {
        let name_len = name_lengths[i];
        let name = String::from_utf8(bytes[*pos..*pos + name_len].to_vec())?;
        *pos += name_len;
        repo_list.push((repo_ids[i], name));
    }

    Ok(repo_list)
}

/// Encode error batch to buffer
fn encode_error_batch(batch: &ErrorBatch, base_id: u64, base_timestamp: u32, buf: &mut Vec<u8>) {
    encode_varint(batch.insert_position as u64, buf);
    encode_varint(batch.events.len() as u64, buf);

    // Group events by timestamp
    let mut intervals: Vec<ErrorBatchInterval> = Vec::new();
    let mut current_ts: Option<u32> = None;

    for event in &batch.events {
        if current_ts == Some(event.timestamp) {
            intervals.last_mut().unwrap().event_count += 1;
        } else {
            intervals.push(ErrorBatchInterval {
                timestamp_offset: event.timestamp.saturating_sub(base_timestamp),
                event_count: 1,
            });
            current_ts = Some(event.timestamp);
        }
    }

    // Write interval info
    encode_varint(intervals.len() as u64, buf);
    for interval in &intervals {
        encode_varint(interval.timestamp_offset as u64, buf);
    }
    for interval in &intervals {
        encode_varint(interval.event_count as u64, buf);
    }

    // Write columnar data
    // Type indices
    for event in &batch.events {
        buf.push(event.event_type);
    }

    // Event ID deltas (using signed varint, first delta is from base_id)
    let mut prev_id = base_id as i64;
    for event in &batch.events {
        let delta = event.id as i64 - prev_id;
        encode_signed_varint(delta, buf);
        prev_id = event.id as i64;
    }

    // Repo indices
    for event in &batch.events {
        encode_varint(event.repo_idx as u64, buf);
    }
}

/// Decode error batch from buffer
fn decode_error_batch(
    bytes: &[u8],
    pos: &mut usize,
    base_id: u64,
    base_timestamp: u32,
) -> Result<ErrorBatch, Box<dyn Error>> {
    let insert_position = decode_varint(bytes, pos) as usize;
    let total_event_count = decode_varint(bytes, pos) as usize;

    // Read interval info
    let interval_count = decode_varint(bytes, pos) as usize;
    let mut timestamp_offsets: Vec<u32> = Vec::with_capacity(interval_count);
    for _ in 0..interval_count {
        timestamp_offsets.push(decode_varint(bytes, pos) as u32);
    }
    let mut events_per_interval: Vec<usize> = Vec::with_capacity(interval_count);
    for _ in 0..interval_count {
        events_per_interval.push(decode_varint(bytes, pos) as usize);
    }

    // Read columnar data
    // Type indices
    let type_indices: Vec<u8> = bytes[*pos..*pos + total_event_count].to_vec();
    *pos += total_event_count;

    // Event ID deltas (signed varint, first delta is from base_id)
    let mut event_ids: Vec<u64> = Vec::with_capacity(total_event_count);
    let mut prev_id = base_id as i64;
    for _ in 0..total_event_count {
        let delta = decode_signed_varint(bytes, pos);
        prev_id += delta;
        event_ids.push(prev_id as u64);
    }

    // Repo indices
    let mut repo_indices: Vec<u32> = Vec::with_capacity(total_event_count);
    for _ in 0..total_event_count {
        repo_indices.push(decode_varint(bytes, pos) as u32);
    }

    // Reconstruct events with timestamps
    let mut events: Vec<ParsedEvent> = Vec::with_capacity(total_event_count);
    let mut event_idx = 0;
    for (interval_idx, &count) in events_per_interval.iter().enumerate() {
        let timestamp = base_timestamp + timestamp_offsets[interval_idx];
        for _ in 0..count {
            events.push(ParsedEvent {
                id: event_ids[event_idx],
                event_type: type_indices[event_idx],
                repo_idx: repo_indices[event_idx],
                timestamp,
            });
            event_idx += 1;
        }
    }

    Ok(ErrorBatch {
        insert_position,
        events,
    })
}
