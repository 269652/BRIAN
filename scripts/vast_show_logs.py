#!/usr/bin/env python3
"""vast_show_logs.py

Fetch stdout/stderr logs for a Vast.ai instance.

Priority order:
 1. If the `vastai` CLI is installed, shell out to `vastai show logs <id>` (supports streaming/follow).
 2. Fallback: call the Vast.ai API to request logs (best-effort; API surface may vary).

Usage:
  python scripts/vast_show_logs.py --instance-id 37240129 --dest logs/vast/37240129.log
  python scripts/vast_show_logs.py --label neuroslm-full --follow

The script reads VAST_API key from the environment or from a local .env file (VAST_API or VAST_AI).
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def has_cli() -> bool:
    return shutil.which('vastai') is not None


def run_cli(instance_id: str, follow: bool, dest: str | None):
    # Candidate CLI subcommands to try. We'll probe each with --help to see which is accepted.
    candidates = [
        ['logs'],
        ['show', 'logs'],
        ['logs', 'get'],
        ['get', 'logs'],
        ['show', 'instance', 'logs'],
    ]

    def probe(prefix: List[str]) -> bool:
        probe_cmd = ['vastai'] + prefix + ['--help']
        try:
            p = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return p.returncode == 0
        except FileNotFoundError:
            raise
        except Exception:
            return False

    working_prefix = None
    for pref in candidates:
        try:
            if probe(pref):
                working_prefix = pref
                break
        except FileNotFoundError:
            raise

    if working_prefix is None:
        # fall back to a simple attempt; let the CLI produce its own error
        working_prefix = ['logs']

    cmd = ['vastai'] + working_prefix + [str(instance_id)]

    # Some versions of the vastai CLI don't support a `--follow` flag on the logs subcommand.
    # Probe the chosen subcommand's help text to see if --follow is supported; if not, omit it.
    supports_follow = False
    try:
        help_p = subprocess.run(['vastai'] + working_prefix + ['--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        help_text = (help_p.stdout or '') + (help_p.stderr or '')
        supports_follow = '--follow' in help_text
    except Exception:
        # If probing fails, conservatively assume follow is not supported
        supports_follow = False

    if follow and not supports_follow:
        print('Note: the vastai CLI on this system does not support --follow for this command; fetching once without follow.')
        follow = False

    if follow:
        cmd.append('--follow')

    print('Running CLI:', ' '.join(cmd))

    if dest:
        # ensure parent directory exists so open() doesn't fail
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # stream into file and stdout
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p, open(dest, 'a', encoding='utf-8') as f:
            try:
                for line in p.stdout:
                    f.write(line)
                    f.flush()
                    sys.stdout.write(line)
            except KeyboardInterrupt:
                p.terminate()
                raise
    else:
        subprocess.run(cmd)


def api_fetch(instance_id: str, dest: str | None):
    try:
        import requests
    except Exception:
        print('requests not installed. Install with: pip install requests')
        return
    key = os.environ.get('VAST_API_KEY') or os.environ.get('VAST_AI')
    if not key:
        print('VAST API key not found in env (.env). Set VAST_API_KEY or VAST_AI.')
        return
    url = f'https://api.vast.ai/v0/instances/{instance_id}/logs'
    print('Requesting via API:', url)
    try:
        r = requests.get(url, headers={'Authorization': 'Bearer ' + key}, stream=True, timeout=30)
        r.raise_for_status()
        if dest:
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print('Saved logs to', dest)
        else:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    sys.stdout.buffer.write(chunk)
    except Exception as e:
        print('API fetch failed:', e)


def list_instances_by_label(label: str) -> List[Dict[str, Any]]:
    try:
        import requests
    except Exception:
        print('requests not installed. Install with: pip install requests')
        return []
    key = os.environ.get('VAST_API_KEY') or os.environ.get('VAST_AI')
    if not key:
        print('VAST API key not found in env (.env). Set VAST_API_KEY or VAST_AI.')
        return []
    url = 'https://api.vast.ai/v0/binstances'
    try:
        r = requests.get(url, headers={'Authorization': 'Bearer ' + key}, timeout=30)
        r.raise_for_status()
        j = r.json()
        items = []
        if isinstance(j, dict):
            items = j.get('binstances') or j.get('bnodes') or j.get('results') or []
        else:
            items = j
        # filter by label heuristics
        matches = []
        for it in items:
            lbl = it.get('label') or it.get('title') or it.get('job_label') or it.get('name')
            if not lbl:
                continue
            if label in lbl or lbl in label:
                matches.append(it)
        return matches
    except Exception as e:
        print('Failed to list instances via API:', e)
        return []


def pick_instance_id_from_item(it: Dict[str, Any]) -> str | None:
    for k in ('id','binstance_id','bnode_id','instance_id'):
        if k in it:
            return str(it[k])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--instance-id', help='Vast.ai instance id')
    p.add_argument('--label', help='Label substring to match against instances')
    p.add_argument('--follow', action='store_true', help='Stream/follow logs (CLI only)')
    p.add_argument('--dest', help='Local file to write logs to')
    args = p.parse_args()

    inst_id = args.instance_id
    if not inst_id and args.label:
        matches = list_instances_by_label(args.label)
        if not matches:
            print('No matching instances found for label', args.label)
            return
        it = matches[0]
        inst_id = pick_instance_id_from_item(it)
        print('Selected instance id', inst_id, 'label', it.get('label'))

    if not inst_id:
        print('Instance id is required (--instance-id or --label to auto-resolve)')
        return

    if has_cli():
        try:
            run_cli(inst_id, args.follow, args.dest)
            return
        except Exception as e:
            print('CLI invocation failed, falling back to API:', e)

    # fallback to API fetch
    api_fetch(inst_id, args.dest)


if __name__ == '__main__':
    import argparse
    main()
