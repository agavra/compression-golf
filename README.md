# compression-golf

**Can you beat 9,407,052 bytes?**

A compression challenge: encode 1,000,000 GitHub events into the smallest possible binary format.

## Leaderboard

| Rank | Who                                | Size (Bytes) |
|------|------------------------------------|--------------|
| 1    | [agavra](src/agavra.rs)            | 9,407,052    |
| 2    | *[Zstd(22)](src/zstd.rs)*          | 11,917,798   |
| 3    | *[Zstd(9)](src/zstd.rs)*           | 17,869,403   |
|      | *[Naive (baseline)](src/naive.rs)* | 210,727,389  |

*[Submit a PR](https://github.com/agavra/compression-golf/pulls) to claim your spot!*

## The Challenge

Your codec must:

1. Implement the `EventCodec` trait
2. Perfectly reconstruct the original data (lossless)
3. Beat the Naive codec (210,727,389 bytes)

## Quick Start

```bash
git clone https://github.com/agavra/compression-golf
cd compression-golf
gunzip -k data.json.gz  # decompress the dataset
cargo run --release
```

The dataset is distributed as `data.json.gz` to keep the repo size manageable.

## How to Compete

1. Fork this repo
2. Create `src/yourname.rs` implementing `EventCodec`
3. Add it to `main.rs` (see [Adding Your Codec](#adding-your-codec))
4. Run `cargo run --release` to verify it beats the current best
5. **Submit a PR** to claim your spot on the leaderboard

## The Data

Each of the 11,351 events contains:

```rust
pub struct EventKey {
    pub id: String,          // numeric string, e.g., "2489651045"
    pub event_type: String,  // 14 unique types (e.g., "PushEvent", "WatchEvent")
}

pub struct EventValue {
    pub repo: Repo,
    pub created_at: String,  // ISO 8601, e.g., "2015-01-01T15:00:00Z"
}

pub struct Repo {
    pub id: u64,             // 6,181 unique repos
    pub name: String,        // e.g., "owner/repo"
    pub url: String,         // e.g., "https://api.github.com/repos/owner/repo"
}
```

## The Interface

```rust
pub trait EventCodec {
    fn name(&self) -> &str;
    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>>;
    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>>;
}
```

## Adding Your Codec

1. Create `src/yourname.rs`:

```rust
use bytes::Bytes;
use std::error::Error;
use crate::codec::EventCodec;
use crate::{EventKey, EventValue};

pub struct YournameCodec;

impl YournameCodec {
    pub fn new() -> Self {
        Self
    }
}

impl EventCodec for YournameCodec {
    fn name(&self) -> &str {
        "yourname"
    }

    fn encode(&self, events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        todo!()
    }

    fn decode(&self, bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        todo!()
    }
}
```

2. Add to `src/main.rs`:

```rust
mod yourname;
use yourname::YournameCodec;
```

3. Add your codec to the `codecs` vec in `main()`:

```rust
let codecs: Vec<(Box<dyn EventCodec>, &[(EventKey, EventValue)])> = vec![
    // ... existing codecs ...
    (Box::new(YournameCodec::new()), &sorted_events),
];
```

## Rules

- Codec must be deterministic
- No external data or pretrained models
- Must compile with stable Rust
- Decode must produce byte-identical output to sorted input

## Resources

- [Strategies for beating the current best](#) *(blog post coming soon)*
- [GitHub Archive](https://www.gharchive.org/) — source of the dataset

## License

MIT
