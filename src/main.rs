use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

mod agavra;
mod codec;
mod hachikuji;
mod naive;
mod zstd;

use agavra::AgavraCodec;
use codec::EventCodec;
use hachikuji::HachikujiCodec;
use naive::NaiveCodec;
use zstd::ZstdCodec;

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

    // Baseline for comparison
    let naive = NaiveCodec::new();
    let baseline = naive.encode(&events)?.len();

    let codecs: Vec<(Box<dyn EventCodec>, &[(EventKey, EventValue)])> = vec![
        (Box::new(NaiveCodec::new()), &events),
        (Box::new(ZstdCodec::new()), &events),
        (Box::new(AgavraCodec::new()), &sorted_events),
        (Box::new(HachikujiCodec::new()), &sorted_events),
    ];

    for (codec, expected) in codecs {
        let encoded = codec.encode(&events)?;
        print_row(codec.name(), encoded.len(), baseline);
        let decoded = codec.decode(&encoded)?;
        assert_eq!(expected, &decoded);
    }

    println!("└────────────────────────┴────────────┴────────────┘");
    println!("\nAll verifications passed");

    Ok(())
}
