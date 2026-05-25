#!/usr/bin/env python3
"""fetch_vast_logs.py

Fetch training / benchmark logs from remote Vast.ai instances via SSH/SFTP
and store them under a local logs directory organized by instance label and
timestamp. Supports getting instance connection info from a local config file
or querying the Vast.ai API if you provide an API key.

Usage:
  python scripts/fetch_vast_logs.py --config docs/logs/config_example.json

Config fields (see docs/logs/config_example.json):
 - local_logs_dir: where to save logs (default ./logs)
 - instances: array of objects with keys: label, host, port, username, key_path, remote_paths
 - vast_api_key: (optional) if provided and use_vast_api true, will query Vast.ai API to populate instances

The script uses paramiko for SFTP. Install dependencies with:
  pip install -r docs/logs/requirements-logs.txt
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Any

try:
    import paramiko
except Exception:
    print("Missing dependency: paramiko. Install with: pip install paramiko")
    raise


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sftp_fetch(host: str, port: int, username: str, key_path: str, remote_paths: List[str], dest_dir: str, timeout: int = 30):
    os.makedirs(dest_dir, exist_ok=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        pkey = None
        if key_path:
            try:
                pkey = paramiko.RSAKey.from_private_key_file(key_path)
            except Exception:
                # try Ed25519
                try:
                    pkey = paramiko.Ed25519Key.from_private_key_file(key_path)
                except Exception:
                    pkey = None
        if pkey:
            client.connect(hostname=host, port=port, username=username, pkey=pkey, timeout=timeout)
        else:
            client.connect(hostname=host, port=port, username=username, timeout=timeout)

        sftp = client.open_sftp()
        fetched = []
        for rpath in remote_paths:
            try:
                # if rpath is a directory, walk it; otherwise attempt to download file
                try:
                    attr = sftp.stat(rpath)
                    # it's a file
                    rem_files = [rpath]
                except IOError:
                    # try to list directory
                    rem_files = []
                    try:
                        for entry in sftp.listdir_attr(rpath):
                            rem_files.append(os.path.join(rpath, entry.filename))
                    except Exception:
                        # treat as glob pattern by running a remote ls via SSH
                        stdin, stdout, stderr = client.exec_command(f"ls -1 {rpath}")
                        lines = stdout.read().decode().splitlines()
                        rem_files = [ln.strip() for ln in lines if ln.strip()]

                for rf in rem_files:
                    basename = os.path.basename(rf)
                    local_path = os.path.join(dest_dir, basename)
                    try:
                        sftp.get(rf, local_path)
                        fetched.append(local_path)
                    except Exception:
                        # skip file if cannot fetch
                        print(f"Warning: failed to fetch {rf} from {host}:{port}")
            except Exception:
                print(f"Warning: remote path processing failed for {rpath} on {host}")
        sftp.close()
        client.close()
        return fetched
    except Exception as e:
        traceback.print_exc()
        try:
            client.close()
        except Exception:
            pass
        return []


def fetch_from_instance(inst: Dict[str, Any], base_dest: str) -> Dict[str, Any]:
    label = inst.get('label') or inst.get('id') or inst.get('host')
    host = inst.get('host')
    port = int(inst.get('port', 22))
    username = inst.get('username', 'root')
    key_path = inst.get('key_path') or os.path.expanduser('~/.ssh/id_rsa')
    remote_paths = inst.get('remote_paths', ['/var/log/'])

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    dest_dir = os.path.join(base_dest, f"{label}-{ts}")
    print(f"Fetching from {label} ({host}:{port}) -> {dest_dir}")
    try:
        files = sftp_fetch(host, port, username, key_path, remote_paths, dest_dir)
        return {'label': label, 'host': host, 'fetched': files}
    except Exception as e:
        return {'label': label, 'host': host, 'error': str(e)}


def run_parallel(instances: List[Dict[str, Any]], base_dest: str, workers: int = 4):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_from_instance, inst, base_dest): inst for inst in instances}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                r = {'error': str(e)}
            results.append(r)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', '-c', required=True, help='Path to JSON config file')
    p.add_argument('--dest', '-d', help='Local base directory for logs (overrides config)')
    p.add_argument('--workers', '-j', type=int, default=4)
    args = p.parse_args()

    cfg = load_config(args.config)
    base_dest = args.dest or cfg.get('local_logs_dir', 'logs')
    use_vast = cfg.get('use_vast_api', False)
    instances = cfg.get('instances', [])

    # Optionally query Vast.ai API to get instances (requires API key)
    if use_vast:
        vast_key = cfg.get('vast_api_key') or os.environ.get('VAST_API_KEY')
        if not vast_key:
            print('Vast.ai API key not found in config or VAST_API_KEY env var. Falling back to configured instances.')
        else:
            try:
                import requests
                headers = {'Authorization': f'Bearer {vast_key}'}
                # This uses the public Vast.ai API (subject to change). We fetch instances and map to ssh info.
                resp = requests.get('https://api.vast.ai/v0/binstances', headers=headers, timeout=30)
                resp.raise_for_status()
                j = resp.json()
                # Map the API results to instances list expected by the script
                instances = []
                for it in j.get('bnodes', []) + j.get('binstances', []) if isinstance(j, dict) else j:
                    # best-effort extract
                    host = it.get('ip') or it.get('ssh_addr') or it.get('ssh_ip')
                    port = it.get('ssh_port') or 22
                    label = it.get('label') or str(it.get('id') or it.get('binstance_id'))
                    username = cfg.get('ssh_username', 'root')
                    key = cfg.get('ssh_key_path')
                    instances.append({'label': label, 'host': host, 'port': port, 'username': username, 'key_path': key, 'remote_paths': cfg.get('remote_paths', ['/var/log/', '/root/.cache/'])})
            except Exception:
                print('Vast.ai query failed; falling back to config instances')

    if not instances:
        print('No instances configured. Edit the config file and re-run.')
        sys.exit(1)

    os.makedirs(base_dest, exist_ok=True)
    res = run_parallel(instances, base_dest, workers=args.workers)
    print('\nSummary:')
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
