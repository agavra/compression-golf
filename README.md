# bit-golf

**Can you beat 190.14 KB?**

A compression challenge: encode 11,351 GitHub events into the smallest possible binary format.

## Leaderboard

| Rank | Who | Size | Approach |
|------|-----|------|----------|
| 🥇 | [agavra](https://github.com/agavra) | 190.14 KB | delta + prefix + zstd |
| 🥈 | *[Columnar example](src/columnar.rs)* | 200.03 KB | columnar + dict + rle + zstd |
| 🥉 | *[Prefix example](src/prefix.rs)* | 202.05 KB | prefix coding + zstd |
|    | *Naive (baseline)* | 2.16 MB | JSON serialization |

*[Submit a PR](https://github.com/agavra/bit-golf/pulls) to claim your spot!*

## The Challenge

Your codec must:

1. Implement the `EventCodec` trait
2. Perfectly reconstruct the original data (lossless)
3. Beat the Naive compression (2.16MB)

I supsect the **theoretical minimum is ~150 KB**, which means there's still 
~20% left on the table!

## Quick Start

```bash
git clone https://github.com/agavra/bit-golf
cd bit-golf
cargo run --release
```

The dataset (`data.json`) is included in the repo — no extra downloads needed.

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
    pub event_type: String,  // 14 unique types (e.g., "PushEvent", "WatchEvent")
    pub id: String,          // numeric string, e.g., "2489651045"
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
    fn encode(events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>>;
    fn decode(bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>>;
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

impl EventCodec for YournameCodec {
    fn encode(events: &[(EventKey, EventValue)]) -> Result<Bytes, Box<dyn Error>> {
        todo!()
    }

    fn decode(bytes: &[u8]) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
        todo!()
    }
}
```

2. Add to `src/main.rs`:

```rust
mod yourname;
use yourname::YournameCodec;
```

3. Add benchmark in `main()`:

```rust
let encoded = ZstdCodec::<YournameCodec>::encode(&events)?;
print_row("yourname", encoded.len(), baseline);
let decoded = ZstdCodec::<YournameCodec>::decode(&encoded)?;
assert_eq!(sorted_events, decoded);
```

## Rules

- Codec must be deterministic
- No external data or pretrained models
- Must compile with stable Rust
- `ZstdCodec<T>` wrapper is allowed and encouraged
- Decode must produce byte-identical output to sorted input

## Resources

- [Strategies for beating the current best](#) *(blog post coming soon)*
- [GitHub Archive](https://www.gharchive.org/) — source of the dataset

## License

MIT
