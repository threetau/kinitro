# Kinitro Backend Database Migrations

This directory contains Alembic migrations for the Kinitro Backend database.

## Setup

1. **Configure Database URL**:
   Set the environment variable `KINITRO_BACKEND_DATABASE_URL` or update the URL in `alembic.ini`:
   ```bash
   export KINITRO_BACKEND_DATABASE_URL="postgresql://user:password@localhost/kinitro_backend"
   ```

2. **Install Dependencies**:
   ```bash
   pip install alembic sqlalchemy asyncpg
   ```

## Running Migrations

### Apply Migrations
```bash
# From the backend directory
cd src/backend
alembic upgrade head
```

### Generate New Migration
```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "description of changes"

# Create empty migration file
alembic revision -m "description of changes"
```

### Migration History
```bash
# Show current revision
alembic current

# Show migration history
alembic history --verbose

# Show pending migrations
alembic show head
```

### Downgrade
```bash
# Downgrade to previous migration
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade <revision_id>

# Downgrade all (drop all tables)
alembic downgrade base
```

## Migration Files

- `env.py` - Alembic environment configuration
- `script.py.mako` - Template for generating new migration files
- `versions/001_initial_backend_schema.py` - Initial schema with all backend tables

## Database Schema

The backend database includes:

### Core Tables
- **competitions** - Competition definitions with benchmarks and points
- **miner_submissions** - Miner submissions from blockchain commitments
- **backend_evaluation_jobs** - Jobs created and broadcast to validators
- **backend_evaluation_results** - Results received from validators

### Infrastructure Tables
- **validator_connections** - Validator connection tracking and stats
- **backend_state** - Service state persistence (singleton table)

### Key Features
- **Foreign Key Constraints** - Maintain data integrity
- **Check Constraints** - Validate data ranges and business rules
- **Indexes** - Optimize common query patterns
- **Timestamps** - Track creation and modification times
- **Unique Constraints** - Prevent duplicate data

## Environment Variables

- `KINITRO_BACKEND_DATABASE_URL` - PostgreSQL connection string
- Default: `postgresql://user:password@localhost/kinitro_backend`

## Notes

- The backend database is separate from the validator database
- Each has its own migration history and schema
- Use different database names to avoid conflicts
- Always backup production data before running migrations