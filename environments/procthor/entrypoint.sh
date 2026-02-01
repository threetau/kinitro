#!/usr/bin/env bash
# Entrypoint script for ProcTHOR container
# Starts Xvfb before the main application

# Start Xvfb for headless rendering
DISPLAY=${DISPLAY:-:99}
export DISPLAY

# Check if Xvfb is already running
if ! xdpyinfo -display $DISPLAY >/dev/null 2>&1; then
    echo "Starting Xvfb on display $DISPLAY..."
    Xvfb $DISPLAY -screen 0 1024x768x24 -ac &
    sleep 1
fi

# Execute the command passed to the container
exec "$@"
