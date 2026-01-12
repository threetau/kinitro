from datetime import timedelta

DEFAULT_MAX_COMMITMENT_LOOKBACK = 360
DEFAULT_CHAIN_SYNC_INTERVAL = timedelta(seconds=30)
MAX_WORKERS = 4
HEARTBEAT_INTERVAL = timedelta(seconds=30)

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

# Scoring and weight setting intervals
SCORE_EVALUATION_INTERVAL = timedelta(minutes=5)
WEIGHT_BROADCAST_INTERVAL = timedelta(minutes=10)

# Task startup delays
SCORE_EVALUATION_STARTUP_DELAY = timedelta(seconds=10)
WEIGHT_BROADCAST_STARTUP_DELAY = timedelta(seconds=10)

# Evaluation job timeout
EVAL_JOB_TIMEOUT = timedelta(hours=2)

# WebSocket initial state data limit
INITIAL_STATE_DATA_LIMIT = 50  # Default limit for initial state data queries

# Hold-out system defaults
DEFAULT_SUBMISSION_HOLDOUT = timedelta(minutes=5)  # 5 minutes private window
SUBMISSION_UPLOAD_URL_TTL = timedelta(minutes=10)  # Presigned PUT validity
SUBMISSION_DOWNLOAD_URL_TTL = timedelta(hours=6)  # Signed GET validity
HOLDOUT_RELEASE_SCAN_INTERVAL = timedelta(minutes=5)  # Poll every 5 minutes
SUBMISSION_RELEASE_URL_TTL = timedelta(days=7)  # URL validity (max is 7 days)
SUBMISSION_SIGNATURE_MAX_AGE = timedelta(minutes=5)  # Signature freshness window
