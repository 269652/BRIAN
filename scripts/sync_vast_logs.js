#!/usr/bin/env node
const { spawnSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const INTERVAL_SECONDS = Number(process.env.SYNC_LOGS_INTERVAL || 30); // default poll interval
const DEST_DIR = process.env.SYNC_LOGS_DEST || path.join('logs', 'vast');

const SAFE_ENV = Object.assign({}, process.env, { PAGER: 'cat', TERM: 'dumb', VAST_NO_PROMPT: '1' });

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function listInstancesCLI() {
  // Try JSON output first (preferred): instances-v1 is paginated and newer.
  const tryCmds = [
    ['show', 'instances-v1', '--json'],
    ['show', 'instances', '--json'],
  ];

  // Build a safe env for non-interactive runs (avoid pagers and prompts)
  const safeEnv = Object.assign({}, process.env, { PAGER: 'cat', TERM: 'dumb', VAST_NO_PROMPT: '1' });

  for (const args of tryCmds) {
    try {
      const out = spawnSync('vastai', args, { encoding: 'utf8', env: safeEnv, input: '' });
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
  const out = spawnSync('vastai', ['show', 'instances'], { encoding: 'utf8', env: safeEnv, input: '' });
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
  const p = spawnSync('vastai', ['logs', String(instanceId)], { encoding: 'utf8', env: SAFE_ENV, input: '' });
  if (p.error) {
    console.error('Error running vastai for', instanceId, p.error && p.error.message);
    return;
  }

  const exitCode = Number.isFinite(p.status) ? p.status : 0;
  if (exitCode !== 0) {
    console.error(`vastai logs ${instanceId} exited with code ${exitCode}. stderr (trimmed):`);
    const stderr = (p.stderr || '').trim();
    console.error(stderr.split(/\r?\n/).slice(0, 10).join('\n'));
    return;
  }

  const outtxt = p.stdout || '';
  // Write/append logic: if file exists and its content is a prefix of new output, append only the tail
  let appended = 0;
  try {
    if (fs.existsSync(destPath)) {
      const existing = fs.readFileSync(destPath, { encoding: 'utf8' });
      if (outtxt.startsWith(existing)) {
        const tail = outtxt.slice(existing.length);
        if (tail.length > 0) {
          fs.appendFileSync(destPath, tail, { encoding: 'utf8' });
          appended = Buffer.byteLength(tail, 'utf8');
        }
      } else {
        // If existing content isn't a prefix, overwrite (avoids duplicating or corrupting files)
        fs.writeFileSync(destPath, outtxt, { encoding: 'utf8' });
        appended = Buffer.byteLength(outtxt, 'utf8');
      }
    } else {
      ensureDir(path.dirname(destPath));
      fs.writeFileSync(destPath, outtxt, { encoding: 'utf8' });
      appended = Buffer.byteLength(outtxt, 'utf8');
    }
    const total = fs.existsSync(destPath) ? fs.statSync(destPath).size : appended;
    console.log(`Fetched ${instanceId} -> ${destPath} (appended ${appended} bytes, total ${total} bytes)`);
  } catch (e) {
    console.error('Failed to write log for', instanceId, e && e.message);
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
