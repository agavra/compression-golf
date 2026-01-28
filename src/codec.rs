use bytes::Bytes;
use std::error::Error;

use crate::{EventKey, EventValue};

pub trait EventCodec {
    fn encode(events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>>;
    fn decode(bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>>;
}
