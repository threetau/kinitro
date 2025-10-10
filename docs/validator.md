---
section: 'Get Started'
---

# Validator

Validators are responsible for evaluating the performance of miner-submitted agents on a variety of tasks.

## Setup

### Setting up environment variables
Copy the `.env.validator.example` file to `.env` and fill in the required environment variables:
```bash
cp .env.validator.example .env
```

You will need to create an R2 bucket and set the relevant environment variables. This is required for storing some evaluation data. For more information please refer to Cloudflare's [R2 documentation](https://developers.cloudflare.com/r2/buckets/). 

You will need to set `KINITRO_API_KEY` to obtain access to the Kinitro backend. Please contact us on our [discord channel](https://discord.gg/96SdmpeMqG) for access.

### Configuration
To configure the validator websocket app, you will need to create a configuration file. You can start by copying the example configuration file:

```bash
cp config/validator.toml.example validator.toml
```
Edit `validator.toml` to set your desired parameters, such as the Bittensor wallet to use, the backend websocket URL, and other settings.

You will also need to set up the evaluator configuration file. You can start by copying the example configuration file:

```bash
cp config/evaluator.toml.example evaluator.toml
```
Edit `evaluator.toml` to set your desired parameters, such as the PostgreSQL database connection string, R2 credentials, and logging intervals.

### Setting up database
The validator requires a PostgreSQL database for queuing evaluation jobs and results.

To set up the database, you can either:

1. **Reset the database** (drops and recreates the database with all migrations):
   ```bash
   chmod +x ./scripts/reset_validator_db.sh
   ./scripts/reset_validator_db.sh
   ```

2. **Run migrations only** (on an existing database):
   ```bash
   chmod +x ./scripts/migrate_validator_db.sh
   ./scripts/migrate_validator_db.sh
   ```

The migration script will check if the database exists and run Alembic migrations to bring it up to date. It will also ensure the pgq extension is installed if needed.

### Running the Websocket app
The websocket app will connect to the Kinitro backend, listen for evaluation jobs, and forward them to the evaluator to execute.
Once your configuration file is set up, you can run the validator using the following command:
```bash
python -m validator --config validator.toml
```

### Running the Evaluator
The evaluator is responsible for executing the evaluation jobs received from the websocket app. It will run the agents in the specified environments and log the results to the database.
To start the evaluator, use the following command:
```bash
python -m evaluator.orchestrator --config evaluator.toml
```
