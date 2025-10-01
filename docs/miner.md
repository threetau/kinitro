---
section: 'Get Started'
---

# Miner
### Setting up environment variables
Copy the `.env.miner.example` file to `.env` and fill in the required environment variables:
```bash
cp .env.miner.example .env
```

### Configuration
Copy the example configuration file, and edit it to include parameters like your Hugging Face submission repo, Bittensor wallet location, etc.
```bash
cp config/miner.toml.example miner.toml
```
Edit `miner.toml` to set your desired parameters.

### Uploading your agent
To upload your agent to the Kinitro platform, use the following command:
```bash
python -m miner upload
```
This command will package your agent, upload it to the specified Hugging Face repository.

### Committing submission info to the blockchain
After uploading your agent, you need to commit the submission information to the Bittensor blockchain. Use the following command:
```bash
python -m miner commit --config miner.toml
```
This command will create a new submission on the blockchain, linking to the uploaded agent.

#### Chain Commitment Version
You can specify the chain commitment version in your configuration file or via command line:
- In `miner.toml`: Set `chain_commitment_version = "1.0"`
- Via CLI: Use `--chain-commitment-version "1.0"`

The default version is "1.0" if not specified.
