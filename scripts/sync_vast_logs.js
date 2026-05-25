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
  // Try JSON output first (preferred): instances-v1 is paginated and newer.
  const tryCmds = [
    ['show', 'instances-v1', '--json'],
    ['show', 'instances', '--json'],
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
        // not JSON, try next
      }
    } catch (e) {
      // ignore and try next
    }
  }

  // Fallback: parse the tabular output from `vastai show instances`.
  try {
    const out = spawnSync('vastai', ['show', 'instances'], { encoding: 'utf8' });
    if (out.error) throw out.error;
    const txt = (out.stdout || '') + '\n' + (out.stderr || '');
    const lines = txt.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const results = [];
    // Look for lines that contain an ID in the second column
    for (const l of lines) {
      const m = l.match(/^\s*#?\s*(\d+)\s+(\d+)\s+/);
      if (m) {
        const id = m[2];
        results.push({ id: id, label: '' });
      }
    }
    if (results.length) return results;
    // if parsing failed, print raw output for debugging
    console.log('Debug: raw `vastai show instances` output:\n' + txt);
    return [];
  } catch (e) {
    console.log('Debug: failed to run `vastai show instances`: ', e && e.message);
    return [];
  }
}

function fetchLogsOnce(instanceId, destPath) {
  ensureDir(path.dirname(destPath));
  // Run vastai logs <id> and write to destPath
  const p = spawnSync('vastai', ['logs', String(instanceId)], { encoding: 'utf8' });
  if (p.error) {
    console.error('Error running vastai for', instanceId, p.error && p.error.message);
    return;
  }
  // Only write stdout to the file if the command exited successfully (status 0)
  if (p.status === 0) {
    const outtxt = p.stdout || '';
    fs.writeFileSync(destPath, outtxt, { encoding: 'utf8' });
    console.log('Fetched', instanceId, '->', destPath, `(${Buffer.byteLength(outtxt, 'utf8')} bytes)`);
  } else {
    console.error('vastai logs', instanceId, 'exited with code', p.status, '; stderr excerpts:');
    const stderr = (p.stderr || '').trim();
    console.error(stderr.split(/\r?\n/).slice(0,5).join('\n'));
  }
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
