use bytes::Bytes;
use std::error::Error;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue};

pub struct NaiveCodec;

impl EventCodec for NaiveCodec {
    fn encode(events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let json = serde_json::to_vec(events)?;
        Ok(Bytes::from(json))
    }

    fn decode(bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let events = serde_json::from_slice(bytes)?;
        Ok(events)
    }
}
