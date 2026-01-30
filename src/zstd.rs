//! # Zstd Codec
//!
//! **Strategy:** Plain JSON serialization compressed with Zstd.
//!
//! This applies Zstd compression to the naive JSON representation.

use bytes::Bytes;
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue};

pub struct ZstdCodec {
    level: i32,
    name: String,
}

impl ZstdCodec {
    pub fn new(level: i32) -> Self {
        Self {
            level,
            name: format!("Zstd({})", level),
        }
    }
}

impl EventCodec for ZstdCodec {
    fn name(&self) -> &str {
        &self.name
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let json = serde_json::to_vec(events)?;
        let compressed = zstd::encode_all(json.as_slice(), self.level)?;
        Ok(Bytes::from(compressed))
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let decompressed = zstd::decode_all(bytes)?;
        let events = serde_json::from_slice(&decompressed)?;
        Ok(events)
    }
}
