use std::collections::HashMap;

use crate::varint::{decode_varint, encode_varint};
use crate::{EventKey, EventValue};

pub struct TypeEnum {
    pub type_to_idx: HashMap<String, u8>,
    pub types: Vec<String>,
}

impl TypeEnum {
    pub fn build(events: &[(EventKey, EventValue)]) -> Self {
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for (key, _) in events {
            *freq.entry(&key.event_type).or_insert(0) += 1;
        }

        let mut types_with_freq: Vec<_> = freq.into_iter().collect();
        types_with_freq.sort_by(|a, b| b.1.cmp(&a.1));

        let mut type_to_idx: HashMap<String, u8> = HashMap::new();
        let mut types: Vec<String> = Vec::new();
        for (i, (t, _)) in types_with_freq.into_iter().enumerate() {
            type_to_idx.insert(t.to_string(), i as u8);
            types.push(t.to_string());
        }

        Self { type_to_idx, types }
    }

    pub fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.types.len() as u64, buf);
        for t in &self.types {
            encode_varint(t.len() as u64, buf);
            buf.extend_from_slice(t.as_bytes());
        }
    }

    pub fn decode(bytes: &[u8], pos: &mut usize) -> Self {
        let type_count = decode_varint(bytes, pos) as usize;
        let mut types: Vec<String> = Vec::with_capacity(type_count);
        let mut type_to_idx: HashMap<String, u8> = HashMap::new();

        for i in 0..type_count {
            let len = decode_varint(bytes, pos) as usize;
            let t = std::str::from_utf8(&bytes[*pos..*pos + len])
                .unwrap()
                .to_string();
            *pos += len;
            type_to_idx.insert(t.clone(), i as u8);
            types.push(t);
        }

        Self { type_to_idx, types }
    }

    pub fn get_index(&self, event_type: &str) -> u8 {
        self.type_to_idx[event_type]
    }

    pub fn get_type(&self, index: u8) -> &str {
        &self.types[index as usize]
    }
}
