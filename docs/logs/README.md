Logs collection for experiments and architectures
===============================================

This folder contains tools and an example config to fetch training, benchmark,
and system logs from remote Vast.ai instances and keep a local, timestamped
archive. Use this to guard experimental claims with raw logs and reproducible
artifacts.

Quick start
-----------

1. Create a Python virtualenv and install requirements:

   python -m venv .venv-logs
   .venv-logs/bin/pip install -r docs/logs/requirements-logs.txt

2. Edit `docs/logs/config_example.json` (or copy to `docs/logs/config.json`) and set your instance SSH/IP and paths to logs.

3. Run the fetcher:

   python scripts/fetch_vast_logs.py --config docs/logs/config.json

Optional: set `use_vast_api` true and provide `vast_api_key` to auto-discover instances.

Storage layout
--------------

logs/vast/<label>-<UTC timestamp>/*

Each folder contains the raw files fetched from the remote instance. Keep these
directories in the `docs/logs` branch so they don't clutter the main code branch;
they can be referenced from documentation as evidence for claims.

Branching policy
-----------------

Create a branch `docs/logs` (this repository includes a utilities commit on that
branch). The branch is intended to store fetched logs and small utilities only.
When an architecture is selected, rebase or merge the relevant docs into `master`
and keep `docs/logs` as an archival branch.

Streaming live logs
-------------------

If you prefer a live stream instead of periodic fetches, use `scripts/stream_vast_logs.py`.

Examples:

1) Stream a single remote log to a local file (direct SSH):

   python scripts/stream_vast_logs.py --host 34.12.34.56 --user root --key ~/.ssh/id_rsa --remote /root/train_logs/run.log --local logs/live/run.log

2) Stream logs from an instance defined in your config:

   python scripts/stream_vast_logs.py --config docs/logs/config.json --label instance-1 --remote /root/train_logs/run.log

The streamer uses `tail -F` on the remote side and will reconnect automatically on transient failures. Press Ctrl+C to stop.

