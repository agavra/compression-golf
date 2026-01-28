use bytes::Bytes;
use std::error::Error;
use std::marker::PhantomData;

use crate::codec::EventCodec;
use crate::{EventKey, EventValue};

pub struct ZstdCodec<C: EventCodec> {
    _phantom: PhantomData<C>,
}

impl<C: EventCodec> EventCodec for ZstdCodec<C> {
    fn encode(events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        let inner = C::encode(events)?;
        let compressed = zstd::encode_all(inner.as_ref(), 3)?;
        Ok(Bytes::from(compressed))
    }

    fn decode(bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        let decompressed = zstd::decode_all(bytes)?;
        C::decode(&decompressed)
    }
}
