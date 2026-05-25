#!/usr/bin/env node
const { spawnSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');

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


function loadDotEnv() {
  const envPath = path.join(process.cwd(), '.env');
  if (!fs.existsSync(envPath)) return;
  const txt = fs.readFileSync(envPath, { encoding: 'utf8' });
  for (const line of txt.split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z0-9_]+)=(.*)$/);
    if (!m) continue;
    const k = m[1];
    let v = m[2] || '';
    // strip quotes
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) {
      v = v.slice(1, -1);
    }
    if (!process.env[k]) process.env[k] = v;
  }
}


function listInstancesAPI(callback) {
  const key = process.env.VAST_API_KEY || process.env.VAST_AI;
  if (!key) return callback(new Error('VAST_API_KEY or VAST_AI not set'), null);
  const opts = {
    hostname: 'api.vast.ai',
    path: '/v0/binstances',
    method: 'GET',
    headers: { 'Authorization': 'Bearer ' + key }
  };
  const req = https.request(opts, (res) => {
    let body = '';
    res.setEncoding('utf8');
    res.on('data', (d) => body += d);
    res.on('end', () => {
      try {
        const j = JSON.parse(body);
        const items = Array.isArray(j) ? j : (j.binstances || j.results || j.instances || []);
        const results = [];
        if (Array.isArray(items)) {
          for (const it of items) {
            const iid = it.id || it.instance_id || it.binstance_id || it.bnode_id || it.bnode || it.bnode_id;
            const lbl = it.label || it.title || it.name || it.job_label || '';
            if (iid) results.push({ id: String(iid), label: String(lbl || '') });
          }
        }
        callback(null, results);
      } catch (e) {
        callback(e, null);
      }
    });
  });
  req.on('error', (err) => callback(err, null));
  req.end();
}


function fetchLogsAPI(instanceId, destPath, cb) {
  const key = process.env.VAST_API_KEY || process.env.VAST_AI;
  if (!key) return cb(new Error('VAST_API_KEY or VAST_AI not set'));
  const opts = {
    hostname: 'api.vast.ai',
    path: `/v0/instances/${instanceId}/logs`,
    method: 'GET',
    headers: { 'Authorization': 'Bearer ' + key }
  };
  const req = https.request(opts, (res) => {
    if (res.statusCode !== 200) {
      let errBody = '';
      res.setEncoding('utf8');
      res.on('data', d => errBody += d);
      res.on('end', () => cb(new Error(`status ${res.statusCode}: ${errBody}`)));
      return;
    }
    ensureDir(path.dirname(destPath));
    const ws = fs.createWriteStream(destPath, { encoding: 'utf8' });
    let written = 0;
    res.on('data', (chunk) => {
      ws.write(chunk);
      written += Buffer.byteLength(chunk, 'utf8');
    });
    res.on('end', () => {
      ws.end();
      cb(null, written);
    });
  });
  req.on('error', (err) => cb(err));
  req.end();
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

function listInstancesAPIAsync() {
  return new Promise((resolve, reject) => {
    listInstancesAPI((err, res) => {
      if (err) return reject(err);
      resolve(res);
    });
  });
}

function fetchLogsAPIAsync(instanceId, destPath) {
  return new Promise((resolve, reject) => {
    fetchLogsAPI(instanceId, destPath, (err, written) => {
      if (err) return reject(err);
      resolve(written);
    });
  });
}

async function syncAll() {
  ensureDir(DEST_DIR);
  loadDotEnv();

  // detect if vastai CLI is available
  let hasCLI = true;
  try {
    const r = spawnSync('vastai', ['--version'], { encoding: 'utf8', env: SAFE_ENV, input: '' });
    if (r.error) hasCLI = false;
  } catch (e) {
    hasCLI = false;
  }

  let instances = [];
  if (hasCLI) {
    try {
      instances = listInstancesCLI();
      console.log(`Found ${instances.length} instances via CLI`);
    } catch (e) {
      console.error('CLI listing failed, falling back to API:', e && e.message);
      hasCLI = false;
    }
  }

  if (!hasCLI) {
    try {
      instances = await listInstancesAPIAsync();
      console.log(`Found ${instances.length} instances via API`);
    } catch (e) {
      console.error('API listing failed:', e && e.message);
      return;
    }
  }

  for (const it of instances) {
    const id = it.id;
    const fname = `${id}.log`;
    const dest = path.join(DEST_DIR, fname);
    try {
      if (hasCLI) {
        fetchLogsOnce(id, dest);
      } else {
        const bytes = await fetchLogsAPIAsync(id, dest);
        console.log(`Fetched ${id} -> ${dest} (${bytes} bytes)`);
      }
    } catch (e) {
      console.error('Failed to fetch logs for', id, e && e.message);
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
