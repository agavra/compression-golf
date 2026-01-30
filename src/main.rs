use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

mod agavra;
mod codec;
mod fabinout;
mod hachikuji;
mod jakedgy;
mod naive;
mod samsond;
mod natebrennand;
mod xiangpenghao;
mod zstd;

use agavra::AgavraCodec;
use codec::EventCodec;
use fabinout::FabinoutCodec;
use hachikuji::HachikujiCodec;
use jakedgy::JakedgyCodec;
use naive::NaiveCodec;
use samsond::SamsondCodec;
use natebrennand::NatebrennandCodec;
use xiangpenghao::XiangpengHaoCodec;
use zstd::ZstdCodec;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EventKey {
    pub id: String,
    pub event_type: String,
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
    let s = bytes.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
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
        "│ {:<22} │ {:>14} │ {:>10} │",
        name,
        format_bytes(size),
        improvement_str
    );
}

fn assert_events_eq(
    name: &str,
    expected: &[(EventKey, EventValue)],
    actual: &[(EventKey, EventValue)],
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{}: length mismatch (expected {}, got {})",
        name,
        expected.len(),
        actual.len()
    );
    let mismatches: Vec<_> = expected
        .iter()
        .zip(actual.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .take(3)
        .collect();
    assert!(
        mismatches.is_empty(),
        "{}: {} mismatches, first at index {}",
        name,
        expected
            .iter()
            .zip(actual.iter())
            .filter(|(a, b)| a != b)
            .count(),
        mismatches[0].0
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut path = "data.json".to_string();
    let mut codec_filter: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--codec" => {
                if i + 1 < args.len() {
                    codec_filter = Some(args[i + 1].to_lowercase());
                    i += 1;
                }
            }
            arg if !arg.starts_with('-') => {
                path = arg.to_string();
            }
            _ => {}
        }
        i += 1;
    }
    let events = load_events(&path)?;
    println!("Loaded {} events from {}\n", events.len(), path);

    let mut sorted_events: Vec<_> = events.clone();
    sorted_events.sort_by(|a, b| a.0.cmp(&b.0));

    // Table header
    println!("┌────────────────────────┬────────────────┬────────────┐");
    println!("│ Codec                  │           Size │ vs Naive   │");
    println!("├────────────────────────┼────────────────┼────────────┤");

    // Baseline for comparison
    let naive = NaiveCodec::new();
    let baseline = naive.encode(&events)?.len();

    let codecs: Vec<(Box<dyn EventCodec>, &[(EventKey, EventValue)])> = vec![
        (Box::new(NaiveCodec::new()), &events),
        (Box::new(ZstdCodec::new(9)), &events),
        // (Box::new(ZstdCodec::new(22)), &events), // commented out b/c it takes long to run
        (Box::new(AgavraCodec::new()), &sorted_events),
        (Box::new(FabinoutCodec::new()), &events),
        (Box::new(HachikujiCodec::new()), &sorted_events),
        (Box::new(XiangpengHaoCodec::new()), &sorted_events),
        (Box::new(SamsondCodec::new()), &events),
        (Box::new(JakedgyCodec::new()), &sorted_events),
        (Box::new(NatebrennandCodec::new()), &sorted_events),
    ];

    for (codec, expected) in codecs {
        // Skip if filter is set and doesn't match (always run Naive for baseline)
        if let Some(ref filter) = codec_filter {
            let name_lower = codec.name().to_lowercase();
            if !name_lower.contains(filter) && !name_lower.contains("naive") {
                continue;
            }
        }

        let encoded = codec.encode(&events)?;
        print_row(codec.name(), encoded.len(), baseline);
        let decoded = codec.decode(&encoded)?;
        assert_events_eq(codec.name(), expected, &decoded);
    }

    println!("└────────────────────────┴────────────────┴────────────┘");
    println!("\nAll verifications passed");

    Ok(())
}
