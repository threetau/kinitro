from datetime import timedelta

WAIT_TIME = timedelta(seconds=5)
PROCESS_JOB_WAIT_TIME = timedelta(seconds=1)
QUEUE_MAXSIZE = 100
EVAL_TIMEOUT = timedelta(hours=2)
RAY_WAIT_TIMEOUT = timedelta(seconds=0.1)
MIN_CONCURRENT_JOBS = 4
RESOURCE_BACKOFF_SECONDS = timedelta(seconds=15)
POD_LOG_TAIL_LINES = 200
