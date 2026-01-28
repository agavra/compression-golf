use bytes::Bytes;
use std::error::Error;

use crate::codec::EventCodec;
use crate::util::{decode_string_prefix, encode_string_prefix, format_timestamp, parse_timestamp};
use crate::varint::{decode_varint, encode_varint};
use crate::{EventKey, EventValue, Repo};

pub struct PrefixCodec;

impl EventCodec for PrefixCodec {
    fn encode(events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let mut sorted: Vec<_> = events.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let mut buf = Vec::new();
        encode_varint(sorted.len() as u64, &mut buf);

        let mut prev_type = String::new();
        let mut prev_id = String::new();
        let mut prev_repo_name = String::new();
        let mut prev_repo_url = String::new();

        for (key, value) in &sorted {
            encode_string_prefix(&key.event_type, &prev_type, &mut buf);
            encode_string_prefix(&key.id, &prev_id, &mut buf);
            prev_type = key.event_type.clone();
            prev_id = key.id.clone();

            encode_varint(value.repo.id, &mut buf);
            encode_string_prefix(&value.repo.name, &prev_repo_name, &mut buf);
            encode_string_prefix(&value.repo.url, &prev_repo_url, &mut buf);
            prev_repo_name = value.repo.name.clone();
            prev_repo_url = value.repo.url.clone();

            let ts = parse_timestamp(&value.created_at);
            encode_varint(ts, &mut buf);
        }

        Ok(Bytes::from(buf))
    }

    fn decode(bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;
        let count = decode_varint(bytes, &mut pos) as usize;

        let mut events = Vec::with_capacity(count);
        let mut prev_type = String::new();
        let mut prev_id = String::new();
        let mut prev_repo_name = String::new();
        let mut prev_repo_url = String::new();

        for _ in 0..count {
            let event_type = decode_string_prefix(bytes, &mut pos, &prev_type);
            let id = decode_string_prefix(bytes, &mut pos, &prev_id);
            prev_type = event_type.clone();
            prev_id = id.clone();

            let repo_id = decode_varint(bytes, &mut pos);
            let repo_name = decode_string_prefix(bytes, &mut pos, &prev_repo_name);
            let repo_url = decode_string_prefix(bytes, &mut pos, &prev_repo_url);
            prev_repo_name = repo_name.clone();
            prev_repo_url = repo_url.clone();

            let ts = decode_varint(bytes, &mut pos);
            let created_at = format_timestamp(ts);

            events.push((
                EventKey { event_type, id },
                EventValue {
                    repo: Repo {
                        id: repo_id,
                        name: repo_name,
                        url: repo_url,
                    },
                    created_at,
                },
            ));
        }

        Ok(events)
    }
}
