#!/usr/bin/env python3
"""stream_vast_logs.py

Stream active logs from remote machines (Vast.ai instances) via SSH and
append them to local files in real time. Use this when you want a live copy
of training logs as they are produced.

Usage:
  python scripts/stream_vast_logs.py --host 1.2.3.4 --user root --key ~/.ssh/id_rsa --remote /root/train_logs/run.log
  python scripts/stream_vast_logs.py --config docs/logs/config_example.json --label instance-1 --remote /root/train_logs/run.log

The script requires `paramiko`.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Any, List

try:
    # load .env if present so users can keep keys in a local file (NOT committed)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    import paramiko
except Exception:
    print("Missing dependency: paramiko. Install with: pip install paramiko")
    raise


def tail_remote_to_file(host: str, port: int, username: str, key_path: str, remote_path: str, local_path: str, reconnect: bool = True):
    """Connect to host via SSH and stream remote_path using 'tail -F' into local_path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    while True:
        try:
            print(f"Connecting to {host}:{port} as {username}; tailing {remote_path}")
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            pkey = None
            if key_path:
                try:
                    pkey = paramiko.RSAKey.from_private_key_file(os.path.expanduser(key_path))
                except Exception:
                    try:
                        pkey = paramiko.Ed25519Key.from_private_key_file(os.path.expanduser(key_path))
                    except Exception:
                        pkey = None
            if pkey:
                client.connect(hostname=host, port=port, username=username, pkey=pkey, timeout=20)
            else:
                client.connect(hostname=host, port=port, username=username, timeout=20)

            # Use -n 0 to ensure tail outputs from end and -F to follow name changes
            cmd = f'tail -n +0 -F {remote_path}'
            stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)

            with open(local_path, 'ab') as lf:
                while True:
                    line = stdout.readline()
                    if not line:
                        # check if channel is closed
                        if stdout.channel.exit_status_ready():
                            break
                        time.sleep(0.2)
                        continue
                    # write bytes (stdout.readline() is str)
                    try:
                        b = line.encode('utf-8', errors='replace')
                    except Exception:
                        b = bytes(line)
                    lf.write(b)
                    lf.flush()
                    # also print to local stdout for live view
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except Exception:
                        pass

            client.close()
            if not reconnect:
                break
            print(f"Connection to {host} closed; reconnecting in 5s...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Interrupted by user; stopping tail")
            try:
                client.close()
            except Exception:
                pass
            break
        except Exception as e:
            print(f"Error while tailing {remote_path} on {host}: {e}")
            if not reconnect:
                break
            print("Reconnecting in 5s...")
            time.sleep(5)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_from_config(cfg_path: str, label: str | None, remote: str | None, dest_base: str | None):
    cfg = load_config(cfg_path)
    instances = cfg.get('instances', [])
    if label:
        instances = [i for i in instances if i.get('label') == label]
    if not instances:
        print('No instances found in config (or label mismatch).')
        return
    threads: List[threading.Thread] = []
    for inst in instances:
        host = inst.get('host')
        port = int(inst.get('port', 22))
        username = inst.get('username', cfg.get('ssh_username', 'root'))
        key_path = inst.get('key_path', cfg.get('ssh_key_path'))
        remote_paths = [remote] if remote else inst.get('remote_paths', cfg.get('remote_paths', ['/root/train_logs/run.log']))
        for r in remote_paths:
            ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            base = dest_base or cfg.get('local_logs_dir', 'logs/stream')
            local_fn = os.path.join(base, f"{inst.get('label','inst')}-{os.path.basename(r)}.log")
            t = threading.Thread(target=tail_remote_to_file, args=(host, port, username, key_path, r, local_fn), daemon=True)
            t.start()
            threads.append(t)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Stopping all tails...')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', help='Direct host to stream from')
    p.add_argument('--port', type=int, default=22)
    p.add_argument('--user', default='root')
    p.add_argument('--key', help='Path to private key')
    p.add_argument('--remote', help='Remote file path to tail')
    p.add_argument('--local', help='Local path to write combined log')
    p.add_argument('--config', help='Config JSON (alternative to host/user/key)')
    p.add_argument('--label', help='Filter instances in config by label')
    p.add_argument('--dest-base', help='Base directory to write logs (overrides config)')
    args = p.parse_args()

    if args.config:
        run_from_config(args.config, args.label, args.remote, args.dest_base)
        return

    if not args.host or not args.remote:
        p.error('Either --config or both --host and --remote must be provided')

    local_path = args.local or os.path.join('logs', f"{args.host.replace(':','_')}-{os.path.basename(args.remote)}.log")
    try:
        tail_remote_to_file(args.host, args.port, args.user, args.key, args.remote, local_path)
    except KeyboardInterrupt:
        print('Interrupted, exiting')


if __name__ == '__main__':
    main()
