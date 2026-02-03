# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds the Rust implementation. Each codec lives in its own file (e.g., `src/fulmicoton.rs`, `src/naive.rs`).
- `src/codec.rs` defines the shared `EventCodec` trait and core types.
- `src/main.rs` wires codecs together and runs the benchmark.
- `data.json.gz` is the training dataset (decompress to `data.json` when needed).
- `target/` contains build artifacts.

## Build, Test, and Development Commands
- `gunzip -k data.json.gz` decompresses the training dataset without deleting the `.gz` file.
- `cargo run --release` builds and runs all codecs against the dataset.
- `cargo run --release -- --codec yourname` runs only a single codec by name.
- `cargo run --release -- path/to/data.json` runs against a custom dataset.

## Coding Style & Naming Conventions
- Follow standard Rust 2021 formatting (4-space indentation, rustfmt defaults).
- Codec files are named `src/<github-username>.rs` and expose `<Name>Codec` with `new()`.
- Keep deterministic, lossless encode/decode logic and avoid external data.

## Testing Guidelines
- There are no dedicated unit tests; correctness is validated by `cargo run --release` which encodes and decodes the dataset.
- When adding a codec, verify round-trip integrity and confirm compressed size improves or is tracked.

## Commit & Pull Request Guidelines
- Commit messages are short and descriptive (e.g., `Add fulmicoton codec`).
- PRs should add a single file `src/<your-github-username>.rs` and the minimal `src/main.rs` import/wiring.
- Include the resulting compressed size in the PR description.
- Before committing, ensure there are no clippy warnings by running `cargo clippy --release -- -D warnings`.

## Agent-Specific Instructions
- When iterating on or optimizing a codec, continuously monitor compression ratios in detail (e.g., total size and per-column sizes if your format tracks them) to validate whether changes improve results.
- Prefer small, incremental changes and re-run `cargo run --release -- --codec <name>` after each change to confirm impact.
- For `xinyuzeng`, always run `cargo run --release -- --codec xinyuzeng` after each change and include the latest size in the commit message.
- When determining prior `xinyuzeng` sizes for comparison, check the git log entries for recent size-tagged commits.
- Record every unsuccessful edit effort in the top comment of `src/xinyuzeng.rs`.
- Before proposing aggressive codec changes, add debug samples and distribution stats for the largest columns (currently `repo_id_idx` and `repo_names`) and use those diagnostics to guide strategy selection.
