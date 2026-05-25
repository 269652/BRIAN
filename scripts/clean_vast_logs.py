#!/usr/bin/env python3
"""clean_vast_logs.py

Scan log files under logs/vast and remove obvious corruption lines such as
CLI usage errors or Python codec error messages (e.g., 'charmap codec can't encode').
Creates a .bak of each cleaned file.
"""
import re
from pathlib import Path


ROOT = Path('logs') / 'vast'

PATTERNS = [
    re.compile(r"^usage: vastai\.exe.*", re.IGNORECASE),
    re.compile(r"^vastai\.exe: error:.*", re.IGNORECASE),
    re.compile(r"charmap' codec can't encode characters", re.IGNORECASE),
    re.compile(r"Error response from daemon: No such container", re.IGNORECASE),
]


def clean_file(p):
    text = p.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    new_lines = []
    removed = 0
    for L in lines:
        if any(pat.search(L) for pat in PATTERNS):
            removed += 1
            continue
        new_lines.append(L)
    if removed:
        bak = p.with_suffix(p.suffix + '.bak')
        p.rename(bak)
        p.write_text('\n'.join(new_lines), encoding='utf-8')
        print(f'Cleaned {p} (removed {removed} lines); backup -> {bak}')


def main():
    if not ROOT.exists():
        print('No logs/vast directory found')
        return
    for p in sorted(ROOT.glob('*.log')):
        try:
            clean_file(p)
        except Exception as e:
            print('Failed to clean', p, e)


if __name__ == '__main__':
    main()
