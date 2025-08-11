# Show available commands
default:
    @just --list

# Set up Python development environment
setup-python:
    uv venv .venv
    @echo "Activate the virtual environment with: source .venv/bin/activate"

# Install evaluator dependencies
install-evaluator:
    cd evaluator && uv sync --dev

# Install miner dependencies
install-miner:
    cd miner && uv sync --dev

# Build the Rust validator
build-validator:
    cargo build

# Build the Rust validator in release mode
build-validator-release:
    cargo build --release

# Run the Rust validator
run-validator:
    cargo run

# Watch and rebuild validator on changes
watch-validator:
    cargo watch -x build

# Run Rust tests
test-rust:
    cargo test

# Format Rust code
fmt-rust:
    cargo fmt

# Lint Rust code
lint-rust:
    cargo clippy --workspace --all-targets --all-features -- --deny warnings

# Run Python linting and formatting
lint-python:
    cd evaluator && ruff check .
    cd miner && ruff check .

# Format Python code
fmt-python:
    cd evaluator && ruff format .
    cd miner && ruff format .

# Run all Python tests
test-python:
    cd evaluator && pytest
    cd miner && pytest

# Clean build artifacts
clean:
    cargo clean
    find . -name "__pycache__" -type d -exec rm -rf {} +
    find . -name "*.pyc" -delete

# Full setup for new development environment
setup: setup-python install-evaluator install-miner build-validator
    @echo "Development environment setup complete!"
    @echo "Activate the virtual environment with: source .venv/bin/activate"
