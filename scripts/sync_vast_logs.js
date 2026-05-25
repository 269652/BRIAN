#!/usr/bin/env node
const { spawnSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const INTERVAL_SECONDS = process.env.SYNC_LOGS_INTERVAL || 30; // default poll interval
const DEST_DIR = process.env.SYNC_LOGS_DEST || path.join('logs', 'vast');

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function listInstancesCLI() {
  // Try to get JSON output first (preferred). Some vastai versions support
  // `vastai show instances-v1 --json` or `vastai show instances --json`.
  const tryCmds = [
    ['show', 'instances-v1', '--json'],
    ['show', 'instances-v1', '--json', '--raw'],
    ['show', 'instances', '--json'],
    ['show', 'instances', '--raw', '--json'],
  ];

  for (const args of tryCmds) {
    try {
      const out = spawnSync('vastai', args, { encoding: 'utf8' });
      if (out.error) continue;
      const txt = (out.stdout || '') + '\n' + (out.stderr || '');
      try {
        const j = JSON.parse(txt);
        const items = Array.isArray(j) ? j : (j.binstances || j.results || j.instances || j);
        const results = [];
        if (Array.isArray(items)) {
          for (const it of items) {
            const iid = it.id || it.instance_id || it.binstance_id || it.bnode_id || it.bnode || it.bnode_id;
            const lbl = it.label || it.title || it.name || it.job_label || '';
            if (iid) results.push({ id: String(iid), label: String(lbl || '') });
          }
          if (results.length) return results;
        }
      } catch (e) {
        // not JSON, fall through to next method
      }
    } catch (e) {
      // ignore and try next
    }
  }

  // Fallback: parse the tabular output from `vastai show instances` for IDs only.
  try {
    const out = spawnSync('vastai', ['show', 'instances'], { encoding: 'utf8' });
    if (out.error) throw out.error;
    const txt = (out.stdout || '') + '\n' + (out.stderr || '');
    const lines = txt.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const results = [];
    // Look for lines starting with an index and an ID column
    for (const l of lines) {
      const m = l.match(/^\s*#?\s*(\d+)\s+(\d+)\s+/);
      if (m) {
        const id = m[2];
        results.push({ id: id, label: '' });
      }
    }
    return results;
  } catch (e) {
    return [];
  }
}

function fetchLogsOnce(instanceId, destPath) {
  ensureDir(path.dirname(destPath));
  // Run vastai logs <id> and write to destPath
  const p = spawnSync('vastai', ['logs', String(instanceId)], { encoding: 'utf8' });
  if (p.error) {
    console.error('Error running vastai for', instanceId, p.error);
    return;
  }
  fs.writeFileSync(destPath, p.stdout || p.stderr || '', { encoding: 'utf8' });
  console.log('Fetched', instanceId, '->', destPath);
}

function syncAll() {
  ensureDir(DEST_DIR);
  let instances = [];
  try {
    instances = listInstancesCLI();
  } catch (e) {
    console.error('Failed to list instances', e);
    return;
  }
  for (const it of instances) {
    const id = it.id;
    const fname = `${id}.log`;
    const dest = path.join(DEST_DIR, fname);
    try {
      fetchLogsOnce(id, dest);
    } catch (e) {
      console.error('Failed to fetch logs for', id, e);
    }
  }
}

function main() {
  ensureDir(DEST_DIR);
  console.log('Starting sync: logs ->', DEST_DIR, 'interval', INTERVAL_SECONDS, 's');
  syncAll();
  setInterval(syncAll, INTERVAL_SECONDS * 1000);
}

main();
