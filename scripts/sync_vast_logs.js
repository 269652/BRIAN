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
  // run `vastai show instances` and parse the ID and Label
  const out = spawnSync('vastai', ['show', 'instances'], { encoding: 'utf8' });
  if (out.error) throw out.error;
  const txt = (out.stdout || '') + '\n' + (out.stderr || '');
  const lines = txt.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const instances = [];
  const re = /^\d+\s+(\d+)\s+\S+\s+\S+\s+(.*?)\s+ssh/;
  for (const l of lines) {
    const m = l.match(/^\s*#?\s*(\d+)\s+(\d+)\s+/);
    if (m) {
      const id = m[2];
      // try to extract label in following lines
      instances.push({ id, label: '' });
    }
  }
  return instances;
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
