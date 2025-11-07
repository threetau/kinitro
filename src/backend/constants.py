DEFAULT_MAX_COMMITMENT_LOOKBACK = 360
DEFAULT_CHAIN_SYNC_INTERVAL = 30
MAX_WORKERS = 4
HEARTBEAT_INTERVAL = 30

#  Yield control every N blocks to prevent blocking WebSocket connections
CHAIN_SCAN_YIELD_INTERVAL = 2

# API Pagination constants
DEFAULT_PAGE_LIMIT = 100
MAX_PAGE_LIMIT = 1000
MIN_PAGE_LIMIT = 1

BACKEND_PORT = 8080

# Competition scoring thresholds
DEFAULT_MIN_AVG_REWARD = 0.0
DEFAULT_WIN_MARGIN_PCT = 0.05  # 5% margin required to win
DEFAULT_MIN_SUCCESS_RATE = 1.0  # Min. success rate to be considered a leader
DEFAULT_OWNER_UID = 4
DEFAULT_BURN_PCT = 0.98

# Scoring and weight setting intervals (in seconds)
SCORE_EVALUATION_INTERVAL = 300  # 5 minutes
WEIGHT_BROADCAST_INTERVAL = 600  # 10 minutes

# Task startup delays (in seconds)
SCORE_EVALUATION_STARTUP_DELAY = 10
WEIGHT_BROADCAST_STARTUP_DELAY = 10

# Evaluation job timeout (in seconds)
EVAL_JOB_TIMEOUT = 7200  # 2 hours

# WebSocket initial state data limit
INITIAL_STATE_DATA_LIMIT = 50  # Default limit for initial state data queries

# Hold-out system defaults
DEFAULT_SUBMISSION_HOLDOUT_SECONDS = 300  # 5 minutes private window
SUBMISSION_UPLOAD_URL_TTL_SECONDS = 600  # Presigned PUT validity (10 minutes)
SUBMISSION_DOWNLOAD_URL_TTL_SECONDS = 21600  # Signed GET validity (6 hours)
HOLDOUT_RELEASE_SCAN_INTERVAL = 300  # Poll every 5 minutes
SUBMISSION_RELEASE_URL_TTL_SECONDS = (
    604800  # Default release URL validity - set to max (7 days)
)
SUBMISSION_SIGNATURE_MAX_AGE_SECONDS = 300  # Signature freshness window
