/*!
This codec implements a multi-layer compression strategy for GitHub timeline events,
combining data restructuring, delta encoding, backreferences, and entropy coding.
Test with `cargo run -r -- --codec lifthrasiir`.

It is no secret that a more powerful entropy coding can replace lots of context modeling;
the winner would probably have to use something like PAQ- or cmix-derived coders.
This codec intentionally forgoes such "shortcuts" in favor of explicit data transformations,
'cause it is more fun and interesting!

The codec was primarily built with GLM-4.7 and Claude Code harness with human feedback.
No existing codec was referenced during development; all ideas are original,
though they do tend to converge towards the local optimum for the obvious reasons.

## Core Principles

### 1. Struct of Arrays (SoA) Layout

Rather than storing events as an array of structs, the codec splits data into separate
parallel arrays (id_deltas, ts_deltas, types, backrefs, repo_ids, repo_users, repo_names).
This groups similar data types together, enabling better compression through locality
and simpler per-type encoding strategies.

Some fields underwent an additional restructuring, particularly ts_deltas and types
which are combined into a single bit-packed field. Most values of ts_deltas are zeroes,
so by storing the low 4 bits alongside the type (which fits in 4 bits), we mostly retain
the compressability of the original types array. Higher bits are stored separately as varints.

### 2. Delta Encoding

- **IDs**: Store only the difference from previous event (monotonically increasing)
- **Timestamps**: Use an adaptive delta scheme referencing the maximum of the two previous
  timestamps to handle one-off out-of-order events while maintaining small deltas

### 3. Combined Backreference Encoding

A single integer encodes both repository and owner references in three cases:
- `0`: New repository + new owner (emits both to string tables)
- `2n+1`: New repository + existing owner with index n
- `2n+2`: Existing repository with index n

This eliminates redundant repository/owner strings while allowing efficient reuse
of previously seen entities.

### 4. Case Compression

Eliminates uppercase letters in strings using two markers:
- `^` (caret): Next character is uppercase (shift)
- `$` (dollar): Latch mode for runs of consecutive uppercase, continues til non-alphanumeric

This is particularly effective for repository names following mixed-case conventions.

### 5. Bidirectional Encoding

When a repository name contains its owner name, the owner substring is replaced with `@`.
During decoding, `@` is substituted back with the actual owner string. This handles cases
like `tensorflow/tensorflow` efficiently.

Additionally, two special cases are recognized where the repository name is extremely
predictable: if the name after replacement is exactly `@`, it is stored as an empty string;
if it is `@.github.io`, it is stored as `@` (should be unambiguous due to the former case).

### 6. Transposed Encoding

For log-scaled multi-byte integers (repo_ids), bytes are stored transposed:
all first bytes, then all second bytes, etc. This groups similar bit-positions together
for better entropy compression.

### 7. Entropy Coding

- **Varint**: Custom leading-ones length prefix encoding for integers
- **Zigzag**: Converts signed deltas to unsigned for varint encoding
- **Zstandard (-22)**: Final entropy compression layer, providing deduplication and FSE coding

Fun fact: Using lpaq1 instead places this codec to the first place at the time of writing,
at 5,054,116 bytes! (I have confirmed that the fulmicoton codec can't be easily modified
to take advantage of a superior entropy coder.)
*/

use bytes::Bytes;
use chrono::{TimeZone, Utc};
use std::collections::HashMap;
use std::error::Error;
use std::io::{Cursor, Read, Write};

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

pub struct LifthrasiirCodec;

impl LifthrasiirCodec {
    pub fn new() -> Self {
        Self
    }
}

const EVENT_TYPES: &[&str] = &[
    "CommitCommentEvent",
    "CreateEvent",
    "DeleteEvent",
    "DiscussionEvent",
    "ForkEvent",
    "GollumEvent",
    "IssueCommentEvent",
    "IssuesEvent",
    "MemberEvent",
    "PublicEvent",
    "PullRequestEvent",
    "PullRequestReviewCommentEvent",
    "PullRequestReviewEvent",
    "PushEvent",
    "ReleaseEvent",
    "WatchEvent",
];

fn get_type_index(type_str: &str) -> Option<u8> {
    EVENT_TYPES
        .iter()
        .position(|&t| t == type_str)
        .map(|i| i as u8)
}

fn get_type_from_index(index: u8) -> Option<&'static str> {
    EVENT_TYPES.get(index as usize).copied()
}

/// Encode string with case compression (shift + latch):
/// - Lowercase by default
/// - ^ (caret) = next character is uppercase (shift)
/// - $ (dollar) = 2+ consecutive uppercase letters when NOT followed by lowercase letter
fn encode_case(input: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if !c.is_alphabetic() {
            result.push(c);
            i += 1;
            continue;
        }

        if c.is_uppercase() {
            // Count consecutive uppercase letters
            let mut run_len = 1;
            while i + run_len < chars.len() && chars[i + run_len].is_uppercase() {
                run_len += 1;
            }

            // Check if next char after uppercase run is a lowercase letter
            let next_is_lower = i + run_len < chars.len() && chars[i + run_len].is_lowercase();

            // Use latch for 2+ consecutive uppercase when NOT followed by lowercase
            if run_len >= 2 && !next_is_lower {
                result.push('$');
                for _ in 0..run_len {
                    result.extend(chars[i].to_lowercase());
                    i += 1;
                }
            } else {
                // Use shift for each uppercase char
                for _ in 0..run_len {
                    result.push('^');
                    result.extend(chars[i].to_lowercase());
                    i += 1;
                }
            }
        } else {
            result.push(c);
            i += 1;
        }
    }

    result
}

/// Decode string with case compression
fn decode_case(input: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '^' && i + 1 < chars.len() {
            result.extend(chars[i + 1].to_uppercase());
            i += 2;
        } else if chars[i] == '$' {
            i += 1;
            // Convert consecutive lowercase letters to uppercase (until non-letter)
            while i < chars.len() && chars[i].is_lowercase() {
                result.extend(chars[i].to_uppercase());
                i += 1;
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

impl EventCodec for LifthrasiirCodec {
    fn name(&self) -> &str {
        "lifthrasiir"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        // Sort events by key before encoding (to match expected output).
        // Note that this is not strictly necessary for correctness,
        // the following code does handle non-monotonic ids just fine.
        // Comment out and replace `sorted_events` with `events`
        // in the main `codecs` vector to test with unsorted input.
        let mut events = events.to_vec();
        events.sort_by(|a, b| a.0.cmp(&b.0));
        let events = &events[..];

        let num_events = events.len();

        // Collect data into separate arrays (struct of arrays)
        let mut id_deltas: Vec<i64> = Vec::with_capacity(num_events);
        let mut ts_deltas: Vec<i64> = Vec::with_capacity(num_events);
        let mut types: Vec<u8> = Vec::with_capacity(num_events);
        let mut combined_backrefs: Vec<u64> = Vec::with_capacity(num_events);
        let mut repo_ids: Vec<u32> = Vec::new();
        let mut repo_users: Vec<u8> = Vec::new(); // owner part
        let mut repo_names: Vec<u8> = Vec::new(); // repo name part

        // Calculate timestamp deltas
        let mut timestamps: Vec<i64> = Vec::with_capacity(num_events);
        for (_, value) in events {
            let ts: chrono::DateTime<Utc> = value.created_at.parse()?;
            timestamps.push(ts.timestamp());
        }

        let mut ts_delta_list: Vec<i64> = Vec::with_capacity(num_events);
        let mut prev_ts: i64 = 0;
        let mut prev_prev_ts: i64 = 0;
        for (i, ts) in timestamps.iter().enumerate() {
            if i == 0 {
                ts_delta_list.push(ts - prev_ts);
            } else if i == 1 {
                ts_delta_list.push(ts - prev_ts);
                prev_prev_ts = prev_ts;
            } else {
                let max_prev = prev_ts.max(prev_prev_ts);
                ts_delta_list.push(ts - max_prev);
                prev_prev_ts = prev_ts;
            }
            prev_ts = *ts;
        }

        // Process events
        let mut repos: Vec<(u64, String)> = Vec::new();
        let mut repo_to_idx: HashMap<(u64, &str), u32> = HashMap::new();
        let mut owners: Vec<String> = Vec::new();
        let mut owner_to_idx: HashMap<&str, u32> = HashMap::new();
        let mut prev_id: u64 = 0;

        for (i, (key, value)) in events.iter().enumerate() {
            // ID delta
            let id: u64 = key.id.parse()?;
            let id_delta = (id as i64) - (prev_id as i64);
            prev_id = id;
            id_deltas.push(id_delta);

            // Timestamp delta
            ts_deltas.push(ts_delta_list[i]);

            // Event type
            let type_idx = get_type_index(&key.event_type).ok_or("Unknown event type")?;
            types.push(type_idx);

            // Combined backref encoding:
            // 0 = new repo + new owner
            // 2n+1 = new repo + existing owner of index n
            // 2n+2 = existing repo of index n
            let repo_key = (value.repo.id, value.repo.name.as_str());
            if let Some(&idx) = repo_to_idx.get(&repo_key) {
                // Existing repo: encode as 2 * idx + 2
                combined_backrefs.push((2 * idx + 2) as u64);
            } else {
                let idx = repos.len() as u32;
                repos.push((value.repo.id, value.repo.name.clone()));
                repo_to_idx.insert((value.repo.id, value.repo.name.as_str()), idx);

                repo_ids.push(value.repo.id.try_into()?);

                // Split name into owner and repo name
                let parts: Vec<&str> = value.repo.name.splitn(2, '/').collect();
                assert_eq!(parts.len(), 2, "Invalid repo name format");
                let owner = parts[0];
                let name = parts[1];

                // Check if new or existing owner
                if let Some(&owner_idx) = owner_to_idx.get(owner) {
                    // New repo + existing owner: encode as 2 * owner_idx + 1
                    combined_backrefs.push((2 * owner_idx + 1) as u64);
                } else {
                    // New repo + new owner: encode as 0
                    let owner_idx = owners.len() as u32;
                    owners.push(owner.to_string());
                    owner_to_idx.insert(owner, owner_idx);
                    combined_backrefs.push(0);

                    // Store owner with case compression
                    let case_encoded = encode_case(owner);
                    repo_users.extend_from_slice(case_encoded.as_bytes());
                    repo_users.push(0);
                }

                // Check if name is a substring of owner for bidirectional encoding
                let mut processed_name = if name.contains(owner) && !owner.is_empty() {
                    name.replace(owner, "@")
                } else {
                    name.to_string()
                };

                // Special mapping: empty string instead of `@`, `@` instead of `@.github.io`
                if processed_name == "@" {
                    processed_name = String::new();
                } else if processed_name == "@.github.io" {
                    processed_name = "@".to_string();
                }

                // Apply case compression to name
                let case_encoded = encode_case(&processed_name);

                repo_names.extend_from_slice(case_encoded.as_bytes());
                repo_names.push(0);
            }
        }

        // Now encode each array
        let mut cursor = Cursor::new(Vec::new());

        // Header
        cursor.write_all(&(num_events as u32).to_le_bytes())?;
        cursor.write_all(&(repo_ids.len() as u32).to_le_bytes())?;

        // Section 1: id_delta
        // For non-positive deltas: signal with 0 byte, then encode -delta as varint
        // For positive deltas: encode directly as varint
        for id_delta in &id_deltas {
            if *id_delta <= 0 {
                cursor.write_all(&[0])?;
                write_varint(&mut cursor, (-*id_delta) as u64)?;
            } else {
                write_varint(&mut cursor, *id_delta as u64)?;
            }
        }

        // Section 2: Combined types + ts_delta_low
        // Pack types (4 bits) + low 4 bits of zigzag(ts_delta) into single byte
        // Then store remaining high bits of zigzag(ts_delta) as varint
        let mut ts_delta_highs: Vec<u64> = Vec::new();
        for (ty, ts_delta) in types.iter().zip(ts_deltas.iter()) {
            let zg = zigzag(*ts_delta);
            // Type in high 4 bits, low 4 bits of zg in low 4 bits
            let combined = (*ty << 4) | ((zg & 0x0F) as u8);
            cursor.write_all(&[combined])?;
            // High bits
            ts_delta_highs.push(zg >> 4);
        }

        // Section 3: ts_delta_high (only for non-zero high bits, 99.9% are zero)
        for high in &ts_delta_highs {
            write_varint(&mut cursor, *high)?;
        }

        // Section 4: combined_backrefs
        for backref in &combined_backrefs {
            write_varint(&mut cursor, *backref)?;
        }

        // Section 5: repo_ids (transposed by bytes)
        // Group all byte 0s, then all byte 1s, etc. for better compression
        for byte_idx in 0..4 {
            for repo_id in &repo_ids {
                let bytes = repo_id.to_le_bytes();
                cursor.write_all(&[bytes[byte_idx]])?;
            }
        }

        // Section 6: repo_users (null-terminated)
        cursor.write_all(&repo_users)?;

        // Section 7: repo_names (null-terminated)
        cursor.write_all(&repo_names)?;

        // Compress with zstd
        let compressed = zstd::encode_all(cursor.into_inner().as_slice(), 22)?;
        Ok(Bytes::from(compressed))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        // Decompress with zstd
        let decompressed = zstd::decode_all(bytes)?;
        let mut cursor = Cursor::new(&decompressed[..]);

        // Read header
        let mut buf = [0u8; 4];
        cursor.read_exact(&mut buf)?;
        let num_events = u32::from_le_bytes(buf) as usize;

        cursor.read_exact(&mut buf)?;
        let num_new_repos = u32::from_le_bytes(buf) as usize;

        // Read Section 1: id_delta
        let mut id_deltas: Vec<i64> = Vec::with_capacity(num_events);
        for _ in 0..num_events {
            let first = read_varint(&mut cursor)? as i64;
            if first == 0 {
                // Non-positive delta: read -delta as varint and negate
                let neg_delta = read_varint(&mut cursor)? as i64;
                id_deltas.push(-neg_delta);
            } else {
                // Positive delta
                id_deltas.push(first);
            }
        }

        // Read Section 2+3: Combined types + ts_delta_low
        let mut types: Vec<u8> = Vec::with_capacity(num_events);
        let mut ts_deltas: Vec<i64> = Vec::with_capacity(num_events);

        // Read combined bytes and extract types + low 4 bits of zigzag
        let mut combined_bytes: Vec<u8> = vec![0; num_events];
        cursor.read_exact(&mut combined_bytes)?;
        for combined in combined_bytes {
            let ty = combined >> 4;
            let low = (combined & 0x0F) as u64;
            types.push(ty);

            // Read high bits
            let high = read_varint(&mut cursor)?;
            // Reconstruct zigzag value
            let zg = (high << 4) | low;
            ts_deltas.push(unzigzag(zg));
        }

        // Read Section 4: combined_backrefs
        let mut combined_backrefs: Vec<u64> = Vec::with_capacity(num_events);
        for _ in 0..num_events {
            combined_backrefs.push(read_varint(&mut cursor)?);
        }

        // Read Section 5: repo_ids (transposed by bytes)
        let mut repo_ids: Vec<u32> = Vec::with_capacity(num_new_repos);
        let mut bytes_matrix: Vec<[u8; 4]> = vec![[0u8; 4]; num_new_repos];

        for byte_idx in 0..4 {
            for i in 0..num_new_repos {
                let mut byte = [0u8; 1];
                cursor.read_exact(&mut byte)?;
                bytes_matrix[i][byte_idx] = byte[0];
            }
        }

        for i in 0..num_new_repos {
            repo_ids.push(u32::from_le_bytes(bytes_matrix[i]));
        }

        // Count unique owners (combined_backref == 0 means new owner)
        let num_new_owners = combined_backrefs.iter().filter(|&&r| r == 0).count();

        // Read Section 6: repo_users (null-terminated, with case compression)
        let mut repo_users: Vec<String> = Vec::with_capacity(num_new_owners);
        for _ in 0..num_new_owners {
            let mut user_bytes = Vec::new();
            loop {
                let mut byte = [0u8; 1];
                cursor.read_exact(&mut byte)?;
                if byte[0] == 0 {
                    break;
                }
                user_bytes.push(byte[0]);
            }
            let user_str = String::from_utf8(user_bytes)?;
            repo_users.push(decode_case(&user_str));
        }

        // Read Section 7: repo_names (null-terminated, with case compression)
        // Only for new repos (repo_backref == 0)
        let mut repo_names: Vec<String> = Vec::with_capacity(num_new_repos);
        for _ in 0..num_new_repos {
            let mut name_bytes = Vec::new();
            loop {
                let mut byte = [0u8; 1];
                cursor.read_exact(&mut byte)?;
                if byte[0] == 0 {
                    break;
                }
                name_bytes.push(byte[0]);
            }
            let name_str = String::from_utf8(name_bytes)?;
            repo_names.push(decode_case(&name_str));
        }

        // Reconstruct events
        let mut repos: Vec<(u64, String)> = Vec::new();
        let mut owners: Vec<String> = Vec::new();
        let mut events = Vec::with_capacity(num_events);
        let mut prev_id: u64 = 0;
        let mut prev_ts: i64 = 0;
        let mut prev_prev_ts: i64 = 0;
        let mut repo_id_idx = 0;
        let mut owner_idx = 0;
        let mut name_idx = 0;

        for i in 0..num_events {
            // Reconstruct id
            let id = (prev_id as i64 + id_deltas[i]) as u64;
            prev_id = id;

            // Reconstruct timestamp
            let ts_delta = ts_deltas[i];
            let ts = if i == 0 {
                ts_delta
            } else if i == 1 {
                let ts_val = prev_ts + ts_delta;
                prev_prev_ts = prev_ts;
                ts_val
            } else {
                let max_prev = prev_ts.max(prev_prev_ts);
                let ts_val = ts_delta + max_prev;
                prev_prev_ts = prev_ts;
                ts_val
            };
            prev_ts = ts;

            let created_at = Utc
                .timestamp_opt(ts, 0)
                .single()
                .ok_or("Invalid timestamp")?
                .format("%Y-%m-%dT%H:%M:%SZ")
                .to_string();

            // Get type
            let type_idx = types[i];
            let event_type = get_type_from_index(type_idx).ok_or("Invalid type index")?;

            // Decode combined_backref
            let combined = combined_backrefs[i];

            let (repo_id, repo_name) = if combined == 0 {
                // New repo + new owner
                let id = repo_ids[repo_id_idx];
                repo_id_idx += 1;

                // Get owner
                if owner_idx >= repo_users.len() {
                    return Err(format!(
                        "owner_idx {} >= repo_users.len() {} at event {}",
                        owner_idx,
                        repo_users.len(),
                        i
                    )
                    .into());
                }
                let owner = repo_users[owner_idx].clone();
                owner_idx += 1;
                owners.push(owner.clone());

                // Get name
                let name = repo_names[name_idx].clone();
                name_idx += 1;

                // Undo special mapping
                let name = if name.is_empty() {
                    "@".to_string()
                } else if name == "@" {
                    "@.github.io".to_string()
                } else {
                    name
                };

                // If name has '@', replace it with owner
                let full_name = format!("{}/{}", owner, name.replace('@', &owner));

                repos.push((id as u64, full_name));
                repos.last().unwrap().clone()
            } else if combined.is_multiple_of(2) {
                // Existing repo
                let idx = (combined / 2 - 1) as usize;
                if idx >= repos.len() {
                    return Err(format!(
                        "Invalid combined_backref {} at event {}, repos.len()={}",
                        combined,
                        i,
                        repos.len()
                    )
                    .into());
                }
                repos[idx].clone()
            } else {
                // New repo + existing owner
                let id = repo_ids[repo_id_idx];
                repo_id_idx += 1;

                // Get existing owner
                let owner_idx_decoded = (combined / 2) as usize;
                if owner_idx_decoded >= owners.len() {
                    return Err(format!(
                        "Invalid owner index {} from combined_backref {} at event {}, owners.len()={}",
                        owner_idx_decoded, combined, i, owners.len()
                    )
                    .into());
                }
                let owner = owners[owner_idx_decoded].clone();

                // Get name
                let name = repo_names[name_idx].clone();
                name_idx += 1;

                // Undo special mapping
                let name = if name.is_empty() {
                    "@".to_string()
                } else if name == "@" {
                    "@.github.io".to_string()
                } else {
                    name
                };

                // If name has '@', replace it with owner
                let full_name = format!("{}/{}", owner, name.replace('@', &owner));

                repos.push((id as u64, full_name));
                repos.last().unwrap().clone()
            };

            let repo_url = format!("https://api.github.com/repos/{}", repo_name);

            let key = EventKey {
                id: id.to_string(),
                event_type: event_type.to_string(),
            };

            let value = EventValue {
                repo: Repo {
                    id: repo_id,
                    name: repo_name,
                    url: repo_url,
                },
                created_at,
            };

            events.push((key, value));
        }

        Ok(events)
    }
}

/// Zigzag encode: maps negative numbers to odd, positive to even
/// 0 → 0, -1 → 1, 1 → 2, -2 → 3, 2 → 4, ...
fn zigzag(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Zigzag decode
fn unzigzag(n: u64) -> i64 {
    let n = n as i64;
    (n >> 1) ^ -(n & 1)
}

/// Write varint with leading-ones length encoding:
/// 0xxxxxxx (1 byte), 10xxxxxx yyyyyyyy (2 bytes), 110xxxxx yyyyyyyy zzzzzzzz (3 bytes), etc.
/// For 9 bytes (values >= 2^56): 11111111 yyyyyyyy zzzzzzzz ...
fn write_varint<W: Write>(writer: &mut W, value: u64) -> Result<(), std::io::Error> {
    // Determine number of bytes needed
    let num_bytes = if value < (1 << 7) {
        1
    } else if value < (1 << 14) {
        2
    } else if value < (1 << 21) {
        3
    } else if value < (1 << 28) {
        4
    } else if value < (1 << 35) {
        5
    } else if value < (1 << 42) {
        6
    } else if value < (1 << 49) {
        7
    } else if value < (1 << 56) {
        8
    } else {
        9
    };

    // Write bytes
    if num_bytes == 9 {
        // Special case: first byte is all 1s (0xFF)
        writer.write_all(&[0xFF])?;
        // Write remaining 8 bytes of data
        for i in 0..8 {
            let shift = (7 - i) * 8;
            let byte = (value >> shift) as u8;
            writer.write_all(&[byte])?;
        }
    } else {
        for i in 0..num_bytes {
            let shift = (num_bytes - 1 - i) * 8;
            let mut byte = (value >> shift) as u8;

            if i == 0 {
                // First byte: clear leading bits, then set prefix (leading 1s followed by 0)
                byte &= (1u8 << (8 - num_bytes)) - 1; // Clear leading bits
                let prefix: u8 = [
                    0b00000000, 0b10000000, 0b11000000, 0b11100000, 0b11110000, 0b11111000,
                    0b11111100, 0b11111110,
                ][num_bytes - 1];
                byte |= prefix;
            }

            writer.write_all(&[byte])?;
        }
    }

    Ok(())
}

/// Read varint with leading-ones length encoding
fn read_varint<R: Read>(reader: &mut R) -> Result<u64, std::io::Error> {
    // Read first byte to determine length
    let mut first_byte = [0u8; 1];
    reader.read_exact(&mut first_byte)?;
    let first_byte = first_byte[0];

    // Count leading 1s to get num_bytes
    let num_bytes = if first_byte == 0xFF {
        9
    } else {
        first_byte.leading_ones() as usize + 1
    };

    let mut result = (first_byte & (0xFF >> num_bytes)) as u64;

    // Read remaining bytes
    for _ in 1..num_bytes {
        let mut byte = [0u8; 1];
        reader.read_exact(&mut byte)?;
        result = (result << 8) | (byte[0] as u64);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Repo;

    #[test]
    fn test_varint() {
        let values = vec![0, 1, 127, 128, 16383, 16384, 1000000];

        for value in values {
            let mut buf = Vec::new();
            write_varint(&mut buf, value).unwrap();
            let mut cursor = Cursor::new(&buf);
            assert_eq!(read_varint(&mut cursor).unwrap(), value);
        }
    }

    #[test]
    fn test_codec_basic() {
        let events = vec![
            (
                EventKey {
                    id: "100".to_string(),
                    event_type: "PushEvent".to_string(),
                },
                EventValue {
                    repo: Repo {
                        id: 123,
                        name: "test/repo".to_string(),
                        url: "https://api.github.com/repos/test/repo".to_string(),
                    },
                    created_at: "2025-01-01T00:00:00Z".to_string(),
                },
            ),
            (
                EventKey {
                    id: "110".to_string(),
                    event_type: "WatchEvent".to_string(),
                },
                EventValue {
                    repo: Repo {
                        id: 456,
                        name: "other/repo".to_string(),
                        url: "https://api.github.com/repos/other/repo".to_string(),
                    },
                    created_at: "2025-01-01T01:00:00Z".to_string(),
                },
            ),
        ];

        let codec = LifthrasiirCodec::new();
        let encoded = codec.encode(&events).expect("Encode failed");
        println!("Encoded {} bytes", encoded.len());

        let decoded = codec.decode(&encoded).expect("Decode failed");

        assert_eq!(events.len(), decoded.len());
        for (i, (expected, actual)) in events.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(expected, actual, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_case_encoding_specific() {
        // Test specific case from actual data
        let name = "NotePad";
        let processed = name.replace("mshdabiola", "@"); // owner shouldn't be in this case
        let encoded = encode_case(&processed);
        let decoded = decode_case(&encoded);
        let reconstructed = decoded.replace('@', "mshdabiola");

        assert_eq!(
            name, reconstructed,
            "Roundtrip failed: {} -> {} (processed={}, encoded={}, decoded={})",
            name, reconstructed, processed, encoded, decoded
        );
    }

    #[test]
    fn test_owner_replacement_with_case() {
        // Test the full flow: owner/name -> @ replacement -> case encoding -> decode -> @ replacement
        let owner = "mshdabiola";
        let name = "NotePad";

        let processed = name.replace(owner, "@");
        let encoded = encode_case(&processed);
        let decoded = decode_case(&encoded);
        let reconstructed = decoded.replace('@', owner);

        assert_eq!(
            name, reconstructed,
            "Failed: {} != {} (processed={}, encoded={}, decoded={})",
            name, reconstructed, processed, encoded, decoded
        );
    }

    #[test]
    fn test_case_encoding_debug() {
        // Test with specific problematic cases
        let test_cases = vec![
            ("facebook", "react"),
            ("facebook", "React"),
            ("OpenAI", "gpt-3"),
            ("OpenAI", "GPT-3"),
            ("tensorflow", "tensorflow"),
            ("microsoft", "TypeScript"),
        ];

        for (owner, name) in test_cases {
            let processed = name.replace(owner, "@");
            let encoded = encode_case(&processed);
            let decoded = decode_case(&encoded);
            let reconstructed = decoded.replace('@', owner);

            println!(
                "Owner: {}, Name: {}, Processed: {}, Encoded: {}, Decoded: {}, Reconstructed: {}",
                owner, name, processed, encoded, decoded, reconstructed
            );

            assert_eq!(
                name, reconstructed,
                "Failed for {}/{}: {} != {}",
                owner, name, name, reconstructed
            );
        }
    }

    #[test]
    fn test_case_encoding() {
        // Test case encoding/decoding
        let test_cases = vec![
            "react",
            "React",
            "Redux",
            "TensorFlow",
            "OpenAI",
            "GitHub",
            "FastAPI",
            "GraphQL",
            "TypeScript",
            "JavaScript",
            "pytest",
            "rust-lang",
            "test-repo-123",
            "API",
            "CNN",
        ];

        for input in test_cases {
            let encoded = encode_case(input);
            let decoded = decode_case(&encoded);
            assert_eq!(
                input, decoded,
                "Failed for '{}': encoded as '{}', decoded as '{}'",
                input, encoded, decoded
            );
        }
    }
}
