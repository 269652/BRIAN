#!/usr/bin/env node
/*
 * sync_vast_logs.js — fetch vast.ai instance stdout logs into logs/vast/.
 *
 * Usage:
 *   npm run sync:logs                       # daemon: poll every SYNC_LOGS_INTERVAL seconds
 *   npm run sync:logs -- --once             # one-shot: fetch once and exit
 *   npm run sync:logs -- --instance 12345   # one-shot for a single id
 *   npm run sync:logs -- --no-analyze       # suppress claude-code auto-analysis
 *
 * Env:
 *   VAST_API_KEY or VAST_AI    (read from .env if present)
 *   SYNC_LOGS_INTERVAL         poll interval seconds (default 30)
 *   SYNC_LOGS_DEST             output directory (default logs/vast)
 *   SYNC_LOGS_NO_ANALYZE=1     same as --no-analyze
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
 * Auto-analysis (default ON when `claude` CLI is on PATH):
 *   After each successful fetch, if the log contains a training/eval
 *   end-marker AND no analysis md under logs/analyzed/ cites this
 *   instance id, the script invokes `claude --dangerously-skip-permissions
 *   -p "..."` to apply logs/analyzed/INSTRUCTIONS.md headlessly — rename
 *   the log, write the analysis md, update FINDINGS.md per SOP rules,
 *   commit + push docs/logs.
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
const NO_ANALYZE = argv.includes('--no-analyze') ||
                   process.env.SYNC_LOGS_NO_ANALYZE === '1';

// ── env / paths ─────────────────────────────────────────────────────────
const INTERVAL_SECONDS = Number(process.env.SYNC_LOGS_INTERVAL || 30);
const DEST_DIR = process.env.SYNC_LOGS_DEST || path.join('logs', 'vast');
const SAFE_ENV = Object.assign({}, process.env, {
  PAGER: 'cat',
  TERM: 'dumb',
  VAST_NO_PROMPT: '1',
  // BRIAN training logs contain unicode (⚡ Φ λ ★ ✓) that crashes the
  // vastai CLI on Windows with the default cp1252 codepage:
  //   'charmap' codec can't encode characters in position N-M
  // Force UTF-8 on both Python I/O streams to keep the subprocess alive.
  PYTHONIOENCODING: 'utf-8',
  PYTHONUTF8: '1',
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
  try {
    const txt = execSync(`${VAST_BIN} show instances --raw`, {
      encoding: 'utf8', env: SAFE_ENV, shell: true,
    });
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
  try {
    const txt = execSync(`${VAST_BIN} show instances`, {
      encoding: 'utf8', env: SAFE_ENV, shell: true,
    });
    const lines = txt.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const out = [];
    for (const l of lines) {
      if (/^ID\s/i.test(l) || /Machine/i.test(l) && !/^\d/.test(l)) continue;
      const m = l.match(/^(\d+)\s+/);
      if (!m) continue;
      const id = m[1];
      const tokens = l.split(/\s+/);
      let label = '';
      for (let i = tokens.length - 1; i >= 1; i--) {
        const t = tokens[i];
        if (!t) continue;
        if (/^[0-9.:T-]+$/.test(t)) continue;
        if (/^[0-9.]+$/.test(t)) continue;
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
  return { id: inst.id, dest, appended, totalAfter, txt };
}

// ── end-of-run detection + claude code auto-analysis ────────────────────

const END_MARKERS = [
  '[train] done.',
  '✓ training reached target',
  '── OOD eval done ──',
  '[ood] results saved to',
];

function hasEndMarker(text) {
  return END_MARKERS.some(m => text.includes(m));
}

function hasClaudeCLI() {
  const r = spawnSync('claude', ['--version'], { encoding: 'utf8', shell: true });
  return !r.error && (r.status === 0 || r.status === null);
}

function alreadyAnalyzed(inst, dest) {
  // Two ways a log can be "already analyzed":
  //   1. The file was renamed off its raw id form (rename is part of the SOP).
  //   2. An analysis md under logs/analyzed/ cites this instance id in its
  //      frontmatter ("**Instance id:** <id>").
  const base = path.basename(dest, '.log');
  if (!/^\d+(__|$)/.test(base)) return true;

  const analyzedDir = path.join('logs', 'analyzed');
  if (!fs.existsSync(analyzedDir)) return false;
  const idStr = String(inst.id);
  for (const f of fs.readdirSync(analyzedDir)) {
    if (!f.endsWith('.md')) continue;
    let content;
    try {
      content = fs.readFileSync(path.join(analyzedDir, f), 'utf8');
    } catch (_) { continue; }
    if (content.includes(`**Instance id:** ${idStr}`)) return true;
  }
  return false;
}

function triggerClaudeAnalysis(inst, dest) {
  if (NO_ANALYZE) return;
  if (!hasClaudeCLI()) {
    console.log(`  · ${inst.id}: end marker detected but claude CLI not on PATH; ` +
                `install claude code (https://docs.anthropic.com/en/docs/claude-code) ` +
                `to enable auto-analysis, or run with --no-analyze to silence`);
    return;
  }
  const logBasename = path.basename(dest);
  const prompt =
    `Apply logs/analyzed/INSTRUCTIONS.md end-to-end to logs/vast/${logBasename}. ` +
    `Read INSTRUCTIONS.md first for the full SOP, then execute every step (1 ` +
    `through 9) without asking for confirmation: classify the role, extract ` +
    `evidence with line citations, pick the descriptive name per §4, git mv the ` +
    `raw log, write logs/analyzed/<descriptive-name>.md from the template in §6, ` +
    `update docs/FINDINGS.md only if there is a real insight per §7 rules, then ` +
    `git add the analyzed pair (and FINDINGS.md if changed) and commit + push ` +
    `origin docs/logs per §9. Do not touch unrelated files; do not edit ` +
    `results/*.json; do not edit docs/architecture.md.`;
  console.log(`  → ${inst.id}: end marker found, launching claude code analysis…`);
  const r = spawnSync('claude', ['--dangerously-skip-permissions', '-p', prompt], {
    stdio: 'inherit', shell: true, env: process.env,
  });
  if (r.status !== 0) {
    console.error(`  ✗ ${inst.id}: claude exited ${r.status}`);
  } else {
    console.log(`  ✓ ${inst.id}: analysis dispatched`);
  }
}

function maybeAnalyze(inst, fetchResult) {
  if (!fetchResult || fetchResult.status === 'fetch-failed' ||
      fetchResult.status === 'empty' || fetchResult.status === 'write-failed') {
    return;
  }
  if (!hasEndMarker(fetchResult.txt)) return;
  if (alreadyAnalyzed(inst, fetchResult.dest)) return;
  triggerClaudeAnalysis(inst, fetchResult.dest);
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
  for (const inst of instances) {
    const r = fetchAndWrite(inst);
    maybeAnalyze(inst, r);
  }
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
              `(Ctrl-C to stop; --once for one-shot; --no-analyze to skip auto-analysis)`);

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
