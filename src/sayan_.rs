use bytes::Bytes;
use std::collections::HashMap;
use std::error::Error;
use std::io::{Read, Write};

use crate::codec::EventCodec;
use crate::{EventKey, EventValue, Repo};

pub struct Sayan_Codec;

impl Sayan_Codec {
    pub fn new() -> Self { Self }
}

fn encode_varint(value: u64, out: &mut Vec<u8>) {
    let mut v = value;
    loop {
        if v < 128 { out.push(v as u8); break; }
        out.push((v as u8 & 0x7F) | 0x80);
        v >>= 7;
    }
}

fn decode_varints(bytes: &[u8], count: usize) -> Vec<u64> {
    let mut result = Vec::with_capacity(count);
    let mut pos = 0;
    for _ in 0..count {
        let mut value: u64 = 0;
        let mut shift = 0;
        loop {
            let b = bytes[pos]; pos += 1;
            value |= ((b & 0x7F) as u64) << shift;
            if b < 128 { break; }
            shift += 7;
        }
        result.push(value);
    }
    result
}

fn zigzag_encode(v: i64) -> u64 { ((v << 1) ^ (v >> 63)) as u64 }
fn zigzag_decode(v: u64) -> i64 { ((v >> 1) as i64) ^ -((v & 1) as i64) }

fn parse_timestamp(ts: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(ts).map(|dt| dt.timestamp()).unwrap_or(0)
}

fn format_timestamp(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
        .unwrap_or_default()
}

fn compress_brotli(data: &[u8], quality: u32) -> Vec<u8> {
    let mut output = Vec::new();
    let mut writer = brotli::CompressorWriter::new(&mut output, 4096, quality, 22);
    writer.write_all(data).unwrap();
    drop(writer);
    output
}

fn compress_bzip2(data: &[u8]) -> Vec<u8> {
    use bzip2::write::BzEncoder;
    let mut encoder = BzEncoder::new(Vec::new(), bzip2::Compression::best());
    encoder.write_all(data).unwrap();
    encoder.finish().unwrap()
}

fn compress_zopfli(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    zopfli::compress(zopfli::Options::default(), zopfli::Format::Deflate, data, &mut output).unwrap();
    output
}

fn decompress_brotli(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    brotli::Decompressor::new(data, 4096).read_to_end(&mut output).unwrap();
    output
}

fn decompress_zopfli(data: &[u8]) -> Vec<u8> {
    use flate2::read::DeflateDecoder;
    let mut output = Vec::new();
    DeflateDecoder::new(data).read_to_end(&mut output).unwrap();
    output
}

fn decompress_bzip2(data: &[u8]) -> Vec<u8> {
    use bzip2::read::BzDecoder;
    let mut output = Vec::new();
    BzDecoder::new(data).read_to_end(&mut output).unwrap();
    output
}

impl EventCodec for Sayan_Codec {
    fn name(&self) -> &str { "Sayan-" }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let mut repo_set: HashMap<(u64, &str), ()> = HashMap::new();
        for (_, v) in events { repo_set.insert((v.repo.id, v.repo.name.as_str()), ()); }
        let mut repo_list: Vec<(u64, &str)> = repo_set.into_keys().collect();
        repo_list.sort_by_key(|(id, _)| *id);
        
        let repo_to_idx: HashMap<(u64, &str), u32> = repo_list.iter().enumerate()
            .map(|(i, (id, name))| ((*id, *name), i as u32)).collect();

        let mut event_type_set: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (key, _) in events { event_type_set.insert(&key.event_type); }
        let mut event_type_list: Vec<&str> = event_type_set.into_iter().collect();
        event_type_list.sort();
        let event_type_to_idx: HashMap<&str, u8> = event_type_list.iter().enumerate()
            .map(|(i, &t)| (t, i as u8)).collect();

        let mut sorted_indices: Vec<usize> = (0..events.len()).collect();
        sorted_indices.sort_by_key(|&i| events[i].0.id.parse::<u64>().unwrap_or(0));

        let mut event_ids: Vec<u64> = Vec::with_capacity(events.len());
        let mut event_types: Vec<u8> = Vec::with_capacity(events.len());
        let mut repo_indices: Vec<u32> = Vec::with_capacity(events.len());
        let mut timestamps: Vec<i64> = Vec::with_capacity(events.len());

        for &idx in &sorted_indices {
            let (key, value) = &events[idx];
            event_ids.push(key.id.parse().unwrap_or(0));
            event_types.push(event_type_to_idx[key.event_type.as_str()]);
            repo_indices.push(repo_to_idx[&(value.repo.id, value.repo.name.as_str())]);
            timestamps.push(parse_timestamp(&value.created_at));
        }
        
        let event_type_dict = event_type_list.join("\n");

        let mut repo_ids_data: Vec<u8> = Vec::new();
        let mut prev_id: u64 = 0;
        for (id, _) in &repo_list {
            encode_varint(*id - prev_id, &mut repo_ids_data);
            prev_id = *id;
        }
        let repo_names = repo_list.iter().map(|(_, n)| *n).collect::<Vec<_>>().join("\n");

        let first_id = event_ids[0];
        let mut id_deltas: Vec<u8> = Vec::new();
        for i in 1..event_ids.len() { encode_varint(event_ids[i] - event_ids[i-1], &mut id_deltas); }

        let first_ts = timestamps[0];
        let mut ts_deltas: Vec<u8> = Vec::new();
        for i in 1..timestamps.len() { encode_varint(zigzag_encode(timestamps[i] - timestamps[i-1]), &mut ts_deltas); }

        let mut types_packed: Vec<u8> = Vec::with_capacity((events.len() + 1) / 2);
        for chunk in event_types.chunks(2) {
            types_packed.push(chunk[0] | (chunk.get(1).copied().unwrap_or(0) << 4));
        }

        let repo_idx_packed: Vec<u8> = repo_indices.iter()
            .flat_map(|&idx| [idx as u8, (idx >> 8) as u8, (idx >> 16) as u8])
            .collect();

        let repo_ids_c = compress_brotli(&repo_ids_data, 10);
        let repo_names_c = compress_brotli(repo_names.as_bytes(), 11);
        let id_deltas_c = compress_brotli(&id_deltas, 10);
        let types_c = compress_zopfli(&types_packed);
        let repo_idx_c = compress_bzip2(&repo_idx_packed);
        let ts_deltas_c = compress_bzip2(&ts_deltas);
        let event_type_dict_c = compress_brotli(event_type_dict.as_bytes(), 11);

        let mut output = Vec::new();
        output.extend_from_slice(&(events.len() as u32).to_le_bytes());
        output.extend_from_slice(&(repo_list.len() as u32).to_le_bytes());
        output.extend_from_slice(&(event_type_list.len() as u8).to_le_bytes());
        output.extend_from_slice(&first_id.to_le_bytes());
        output.extend_from_slice(&first_ts.to_le_bytes());
        
        output.extend_from_slice(&(event_type_dict_c.len() as u16).to_le_bytes());
        output.extend_from_slice(&event_type_dict_c);
        
        for col in [&repo_ids_c, &repo_names_c, &id_deltas_c, &types_c, &repo_idx_c, &ts_deltas_c] {
            output.extend_from_slice(&(col.len() as u32).to_le_bytes());
            output.extend_from_slice(col);
        }

        Ok(Bytes::from(output))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let mut pos = 0;
        let num_events = u32::from_le_bytes(bytes[pos..pos+4].try_into()?) as usize; pos += 4;
        let num_repos = u32::from_le_bytes(bytes[pos..pos+4].try_into()?) as usize; pos += 4;
        let _num_event_types = bytes[pos] as usize; pos += 1;
        let first_id = u64::from_le_bytes(bytes[pos..pos+8].try_into()?); pos += 8;
        let first_ts = i64::from_le_bytes(bytes[pos..pos+8].try_into()?); pos += 8;
        
        let len = u16::from_le_bytes(bytes[pos..pos+2].try_into()?) as usize; pos += 2;
        let event_type_dict_data = decompress_brotli(&bytes[pos..pos+len]); pos += len;
        let event_type_dict_str = std::str::from_utf8(&event_type_dict_data)?;
        let event_type_list: Vec<&str> = event_type_dict_str.split('\n').collect();
        
        let len = u32::from_le_bytes(bytes[pos..pos+4].try_into()?) as usize; pos += 4;
        let repo_ids_data = decompress_brotli(&bytes[pos..pos+len]); pos += len;
        
        let len = u32::from_le_bytes(bytes[pos..pos+4].try_into()?) as usize; pos += 4;
        let repo_names_data = decompress_brotli(&bytes[pos..pos+len]); pos += len;
        
        let len = u32::from_le_bytes(bytes[pos..pos+4].try_into()?) as usize; pos += 4;
        let id_deltas_data = decompress_brotli(&bytes[pos..pos+len]); pos += len;
        
        let len = u32::from_le_bytes(bytes[pos..pos+4].try_into()?) as usize; pos += 4;
        let types_data = decompress_zopfli(&bytes[pos..pos+len]); pos += len;
        
        let len = u32::from_le_bytes(bytes[pos..pos+4].try_into()?) as usize; pos += 4;
        let repo_idx_data = decompress_bzip2(&bytes[pos..pos+len]); pos += len;
        
        let len = u32::from_le_bytes(bytes[pos..pos+4].try_into()?) as usize; pos += 4;
        let ts_deltas_data = decompress_bzip2(&bytes[pos..pos+len]);
        
        let repo_id_deltas = decode_varints(&repo_ids_data, num_repos);
        let mut repo_ids = Vec::with_capacity(num_repos);
        let mut prev: u64 = 0;
        for d in repo_id_deltas { prev += d; repo_ids.push(prev); }
        let repo_names: Vec<&str> = std::str::from_utf8(&repo_names_data)?.split('\n').collect();
        let repos: Vec<(u64, &str)> = repo_ids.into_iter().zip(repo_names).collect();
        
        let id_deltas = decode_varints(&id_deltas_data, num_events - 1);
        let mut event_ids = vec![first_id];
        for d in id_deltas { event_ids.push(event_ids.last().unwrap() + d); }
        
        let mut event_types = Vec::with_capacity(num_events);
        for &b in &types_data {
            event_types.push(b & 0x0F);
            if event_types.len() < num_events { event_types.push(b >> 4); }
        }
        
        let repo_indices: Vec<u32> = repo_idx_data.chunks(3)
            .map(|c| c[0] as u32 | ((c[1] as u32) << 8) | ((c[2] as u32) << 16))
            .collect();
        
        let ts_deltas = decode_varints(&ts_deltas_data, num_events - 1);
        let mut timestamps = vec![first_ts];
        for d in ts_deltas { timestamps.push(timestamps.last().unwrap() + zigzag_decode(d)); }
        
        let mut events = Vec::with_capacity(num_events);
        for i in 0..num_events {
            let (repo_id, repo_name) = repos[repo_indices[i] as usize];
            events.push((
                EventKey { id: event_ids[i].to_string(), event_type: event_type_list[event_types[i] as usize].to_string() },
                EventValue {
                    repo: Repo { id: repo_id, name: repo_name.to_string(), url: format!("https://api.github.com/repos/{}", repo_name) },
                    created_at: format_timestamp(timestamps[i]),
                },
            ));
        }
        events.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(events)
    }
}
