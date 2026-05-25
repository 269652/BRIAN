#!/usr/bin/env node
/*
 * sync_vast_logs.js — fetch vast.ai instance stdout logs into logs/vast/.
 *
 * Usage:
 *   npm run sync:logs              # daemon: poll every SYNC_LOGS_INTERVAL seconds (default 30)
 *   npm run sync:logs -- --once    # one-shot: fetch once and exit
 *   npm run sync:logs -- --instance 12345  # one-shot for a single id
 *
 * Env:
 *   VAST_API_KEY or VAST_AI    (read from .env if present)
 *   SYNC_LOGS_INTERVAL         poll interval seconds (default 30)
 *   SYNC_LOGS_DEST             output directory (default logs/vast)
 *
 * Naming:
 *   logs/vast/<id>__<label>.log   when an instance has a label
 *   logs/vast/<id>.log            when it doesn't
 *
 * Discovery order for instance list:
 *   1. `vastai show instances --raw`     (JSON, includes labels — preferred)
 *   2. `vastai show instances`           (text, fragile parsing — fallback)
 *   3. https://api.vast.ai/v0/binstances (API, requires key — last resort)
 *
 * Fetching is always via `vastai logs <id>` (no API fallback for log content
 * because the public API surface for log streaming is undocumented). The
 * fetched output is appended to the destination file if the existing content
 * is a prefix of the new output; otherwise the file is rewritten.
 */

'use strict';

const { spawnSync, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');

// ── argv ────────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const ONCE = argv.includes('--once');
const SINGLE_INSTANCE = (() => {
  const i = argv.indexOf('--instance');
  return i >= 0 && argv[i + 1] ? argv[i + 1] : null;
})();

// ── env / paths ─────────────────────────────────────────────────────────
const INTERVAL_SECONDS = Number(process.env.SYNC_LOGS_INTERVAL || 30);
const DEST_DIR = process.env.SYNC_LOGS_DEST || path.join('logs', 'vast');
const SAFE_ENV = Object.assign({}, process.env, {
  PAGER: 'cat',
  TERM: 'dumb',
  VAST_NO_PROMPT: '1',
});

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function loadDotEnv() {
  const envPath = path.join(process.cwd(), '.env');
  if (!fs.existsSync(envPath)) return;
  const txt = fs.readFileSync(envPath, 'utf8');
  for (const line of txt.split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Za-z0-9_]+)=(.*)$/);
    if (!m) continue;
    const k = m[1];
    let v = m[2] || '';
    if ((v.startsWith('"') && v.endsWith('"')) ||
        (v.startsWith("'") && v.endsWith("'"))) {
      v = v.slice(1, -1);
    }
    if (!process.env[k]) process.env[k] = v;
  }
}

// ── vastai CLI discovery ────────────────────────────────────────────────
function resolveVastBin() {
  try {
    const cmd = process.platform === 'win32' ? 'where vastai' : 'which vastai';
    const out = execSync(cmd, { encoding: 'utf8', shell: true });
    const first = out.split(/\r?\n/)[0].trim();
    if (first) return first;
  } catch (_) {}
  return 'vastai';
}
let VAST_BIN = null;

function hasVastCLI() {
  if (!VAST_BIN) VAST_BIN = resolveVastBin();
  const r = spawnSync(VAST_BIN, ['--version'], { encoding: 'utf8', env: SAFE_ENV, shell: true });
  return !r.error && (r.status === 0 || r.status === null);
}

// ── instance listing ────────────────────────────────────────────────────
function sanitizeLabel(lbl) {
  if (!lbl) return '';
  return String(lbl)
    .trim().toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9\-._]/g, '');
}

function listInstancesRawJSON() {
  // Preferred: `vastai show instances --raw` returns JSON with labels.
  try {
    const txt = execSync(`${VAST_BIN} show instances --raw`, {
      encoding: 'utf8', env: SAFE_ENV, shell: true,
    });
    // Strip any leading non-JSON noise (some CLI versions prefix banners).
    const starts = [txt.indexOf('['), txt.indexOf('{')].filter(i => i >= 0);
    if (!starts.length) return null;
    const jsonText = txt.slice(Math.min(...starts));
    const parsed = JSON.parse(jsonText);
    const items = Array.isArray(parsed) ? parsed
      : (parsed.instances || parsed.binstances || parsed.results || []);
    const out = [];
    for (const it of items) {
      const id = it.id ?? it.instance_id ?? it.binstance_id;
      if (id == null) continue;
      const label = it.label ?? it.title ?? it.job_label ?? it.name ?? '';
      out.push({ id: String(id), label: String(label) });
    }
    return out;
  } catch (e) {
    return null;
  }
}

function listInstancesTextCLI() {
  // Fallback: parse the human-readable table. Schema:
  //   ID    Machine  Status   ...   Label
  // Take the FIRST integer column as the instance id; take the trailing
  // non-empty whitespace-separated token as the label (best-effort — the
  // CLI's column count drifts across versions, so we keep this minimal).
  try {
    const txt = execSync(`${VAST_BIN} show instances`, {
      encoding: 'utf8', env: SAFE_ENV, shell: true,
    });
    const lines = txt.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const out = [];
    for (const l of lines) {
      // Skip header line (starts with "ID" or contains "Machine")
      if (/^ID\s/i.test(l) || /Machine/i.test(l) && !/^\d/.test(l)) continue;
      const m = l.match(/^(\d+)\s+/);
      if (!m) continue;
      const id = m[1];
      // Heuristic: label is the last whitespace-token if it's not a pure number/timestamp
      const tokens = l.split(/\s+/);
      let label = '';
      for (let i = tokens.length - 1; i >= 1; i--) {
        const t = tokens[i];
        if (!t) continue;
        if (/^[0-9.:T-]+$/.test(t)) continue;          // skip timestamps/sizes
        if (/^[0-9.]+$/.test(t)) continue;             // skip pure numbers
        if (/^(running|stopped|loading|exited|offline)$/i.test(t)) continue;
        label = t;
        break;
      }
      out.push({ id, label });
    }
    return out;
  } catch (_) {
    return null;
  }
}

function listInstancesAPI() {
  const key = process.env.VAST_API_KEY || process.env.VAST_AI;
  if (!key) return Promise.resolve(null);
  return new Promise((resolve) => {
    const req = https.request({
      hostname: 'api.vast.ai',
      path: '/v0/binstances',
      method: 'GET',
      headers: { Authorization: 'Bearer ' + key },
    }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', d => body += d);
      res.on('end', () => {
        try {
          const j = JSON.parse(body);
          const items = Array.isArray(j) ? j : (j.binstances || j.results || j.instances || []);
          const out = [];
          for (const it of items) {
            const id = it.id ?? it.instance_id ?? it.binstance_id;
            if (id == null) continue;
            const label = it.label ?? it.title ?? it.name ?? '';
            out.push({ id: String(id), label: String(label) });
          }
          resolve(out);
        } catch (_) {
          resolve(null);
        }
      });
    });
    req.on('error', () => resolve(null));
    req.end();
  });
}

async function discoverInstances() {
  if (SINGLE_INSTANCE) return [{ id: SINGLE_INSTANCE, label: '' }];

  if (hasVastCLI()) {
    const raw = listInstancesRawJSON();
    if (raw && raw.length) return raw;
    const text = listInstancesTextCLI();
    if (text && text.length) return text;
  }
  const api = await listInstancesAPI();
  return api || [];
}

// ── log fetching ────────────────────────────────────────────────────────
function destPathFor(inst) {
  const safe = sanitizeLabel(inst.label);
  const name = safe ? `${inst.id}__${safe}.log` : `${inst.id}.log`;
  return path.join(DEST_DIR, name);
}

function fetchAndWrite(inst) {
  const dest = destPathFor(inst);
  let txt;
  try {
    txt = execSync(`${VAST_BIN} logs ${inst.id}`, {
      encoding: 'utf8', env: SAFE_ENV, shell: true,
      maxBuffer: 64 * 1024 * 1024,  // 64 MB cap
    });
  } catch (e) {
    // vastai logs returns non-zero for stopped instances; treat as soft error.
    console.error(`  ✗ ${inst.id}: vastai logs failed (${e.message.split('\n')[0]})`);
    return { id: inst.id, status: 'fetch-failed' };
  }
  if (!txt) {
    return { id: inst.id, status: 'empty' };
  }
  ensureDir(path.dirname(dest));
  let appended = 0;
  let totalAfter = 0;
  try {
    if (fs.existsSync(dest)) {
      const existing = fs.readFileSync(dest, 'utf8');
      if (txt.startsWith(existing)) {
        const tail = txt.slice(existing.length);
        if (tail.length > 0) {
          fs.appendFileSync(dest, tail, 'utf8');
          appended = Buffer.byteLength(tail, 'utf8');
        }
      } else {
        // Existing file diverged from current capture (e.g. instance was
        // restarted and stdout history was reset). Rewrite from scratch.
        fs.writeFileSync(dest, txt, 'utf8');
        appended = Buffer.byteLength(txt, 'utf8');
      }
    } else {
      fs.writeFileSync(dest, txt, 'utf8');
      appended = Buffer.byteLength(txt, 'utf8');
    }
    totalAfter = fs.statSync(dest).size;
  } catch (e) {
    console.error(`  ✗ ${inst.id}: write failed (${e.message})`);
    return { id: inst.id, status: 'write-failed' };
  }
  const label = inst.label ? ` [${sanitizeLabel(inst.label)}]` : '';
  console.log(`  ✓ ${inst.id}${label}: +${appended} B (total ${totalAfter} B) → ${dest}`);
  return { id: inst.id, dest, appended, totalAfter };
}

// ── orchestration ───────────────────────────────────────────────────────
async function syncAll() {
  ensureDir(DEST_DIR);
  loadDotEnv();

  if (!hasVastCLI()) {
    console.error('vastai CLI not on PATH; install it (pip install vastai). Skipping this run.');
    return;
  }

  const instances = await discoverInstances();
  if (!instances.length) {
    console.log(`[${new Date().toISOString()}] no vast instances active.`);
    return;
  }

  console.log(`[${new Date().toISOString()}] syncing ${instances.length} instance(s):`);
  for (const inst of instances) fetchAndWrite(inst);
}

function main() {
  ensureDir(DEST_DIR);

  if (ONCE || SINGLE_INSTANCE) {
    syncAll().catch(e => {
      console.error('syncAll failed:', e?.stack || e);
      process.exit(1);
    });
    return;
  }

  console.log(`sync:logs daemon → ${DEST_DIR} every ${INTERVAL_SECONDS}s ` +
              `(Ctrl-C to stop; --once for one-shot)`);

  const runOnce = async () => {
    try { await syncAll(); }
    catch (e) { console.error('syncAll failed:', e?.message || e); }
  };

  setInterval(() => runOnce(), INTERVAL_SECONDS * 1000);
  runOnce();
}

process.on('unhandledRejection', (reason) => {
  console.error('Unhandled rejection:', reason?.stack || reason);
});
process.on('uncaughtException', (err) => {
  console.error('Uncaught exception:', err?.stack || err);
});

main();
