{
  description = "Storb RL Subnet Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-utils.url = "github:numtide/flake-utils";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    rust-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      overlays = [(import rust-overlay)];
      pkgs = import nixpkgs {
        inherit system overlays;
      };

      # Rust toolchain
      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        extensions = ["rust-src" "rust-analyzer"];
      };

      # Python environment (just the interpreter, no package management)
      pythonEnv = pkgs.python313;
    in {
      devShells.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # Rust development
          rustToolchain
          cargo-watch
          cargo-edit

          # Python and uv for package management
          pythonEnv
          uv

          # System dependencies needed for Rust compilation
          pkg-config
          openssl

          # General development tools
          git
          just
        ];

        shellHook = ''
          echo "Storb RL Development Environment"
          echo ""
          echo "  - Rust version: $(rustc --version)"
          echo "  - Python version: $(python --version)"
          echo "  - uv version: $(uv --version)"
          echo ""
          echo "💡 To set up Python environment:"
          echo "  uv venv .venv"
          echo "  source .venv/bin/activate"
          echo "  cd evaluator && uv sync --dev  # for evaluator"
          echo "  cd miner && uv sync --dev      # for miner"
          echo ""
          echo "🔧 To build Rust validator:"
          echo "  cargo build"
          echo ""
        '';

        # Environment variables
        RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
        PKG_CONFIG_PATH = "${pkgs.openssl.dev}/lib/pkgconfig";
      };

      # Formatter for nix files
      formatter = pkgs.alejandra;
    });
}
