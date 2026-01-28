use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

mod agavra;
mod codec;
mod columnar;
mod compressed;
mod naive;
mod prefix;
mod type_enum;
mod util;
mod varint;

use agavra::AgavraCodec;
use codec::EventCodec;
use columnar::ColumnarCodec;
use compressed::ZstdCodec;
use naive::NaiveCodec;
use prefix::PrefixCodec;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EventKey {
    pub event_type: String,
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventValue {
    pub repo: Repo,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Repo {
    pub id: u64,
    pub name: String,
    pub url: String,
}

#[derive(Deserialize)]
struct RawGitHubEvent {
    id: String,
    #[serde(rename = "type")]
    event_type: String,
    repo: Repo,
    created_at: String,
}

fn load_events(path: &str) -> Result<Vec<(EventKey, EventValue)>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let raw: RawGitHubEvent = serde_json::from_str(&line)?;
        let key = EventKey {
            event_type: raw.event_type,
            id: raw.id,
        };
        let value = EventValue {
            repo: raw.repo,
            created_at: raw.created_at,
        };
        events.push((key, value));
    }

    Ok(events)
}

fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;

    let bytes_f = bytes as f64;
    if bytes_f >= MB {
        format!("{:.2} MB", bytes_f / MB)
    } else if bytes_f >= KB {
        format!("{:.2} KB", bytes_f / KB)
    } else {
        format!("{} B", bytes)
    }
}

fn print_row(name: &str, size: usize, baseline: usize) {
    let improvement = if baseline > 0 {
        ((baseline as f64 - size as f64) / baseline as f64) * 100.0
    } else {
        0.0
    };

    let improvement_str = if improvement > 0.0 {
        format!("-{:.1}%", improvement)
    } else if improvement < 0.0 {
        format!("+{:.1}%", -improvement)
    } else {
        "baseline".to_string()
    };

    println!(
        "│ {:<22} │ {:>10} │ {:>10} │",
        name,
        format_bytes(size),
        improvement_str
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = "data.json";
    let events = load_events(path)?;
    println!("Loaded {} events\n", events.len());

    let mut sorted_events: Vec<_> = events.clone();
    sorted_events.sort_by(|a, b| a.0.cmp(&b.0));

    // Table header
    println!("┌────────────────────────┬────────────┬────────────┐");
    println!("│ Codec                  │       Size │ vs Naive   │");
    println!("├────────────────────────┼────────────┼────────────┤");

    // Naive codec (baseline)
    let encoded = NaiveCodec::encode(&events)?;
    let baseline = encoded.len();
    print_row("Naive", encoded.len(), baseline);
    let decoded = NaiveCodec::decode(&encoded)?;
    assert_eq!(events, decoded);

    // Prefix + Zstd (example)
    let encoded = ZstdCodec::<PrefixCodec>::encode(&events)?;
    print_row("Prefix + Zstd", encoded.len(), baseline);
    let decoded = ZstdCodec::<PrefixCodec>::decode(&encoded)?;
    assert_eq!(sorted_events, decoded);

    // Columnar + Zstd (example)
    let encoded = ZstdCodec::<ColumnarCodec>::encode(&events)?;
    print_row("Columnar + Zstd", encoded.len(), baseline);
    let decoded = ZstdCodec::<ColumnarCodec>::decode(&encoded)?;
    assert_eq!(sorted_events, decoded);

    // Agavra + Zstd (current best)
    let encoded = ZstdCodec::<AgavraCodec>::encode(&events)?;
    print_row("agavra", encoded.len(), baseline);
    let decoded = ZstdCodec::<AgavraCodec>::decode(&encoded)?;
    assert_eq!(sorted_events, decoded);

    println!("└────────────────────────┴────────────┴────────────┘");
    println!("\nAll verifications passed");

    Ok(())
}
