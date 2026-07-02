/* app.js – GestureFly 4.0 renderer logic */

const API = 'http://127.0.0.1:5001';

const GESTURE_CLASSES = [
  'TAKEOFF', 'LAND', 'UP', 'DOWN',
  'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD',
  'CLOCKWISE_90', 'ANTICLOCKWISE_90',
];

const GESTURE_ICONS = {
  TAKEOFF: '🛫', LAND: '🛬', UP: '⬆', DOWN: '⬇',
  LEFT: '⬅', RIGHT: '➡', FORWARD: '🔼', BACKWARD: '🔽',
  CLOCKWISE_90: '↻', ANTICLOCKWISE_90: '↺',
};

const MODEL_NAMES = ['LR', 'KNN', 'SVM', 'RF', 'XGB', 'ANN'];

// ── State ──────────────────────────────────────────────────────────────────
let selectedClass  = null;
let targetCount    = 50;
let isCapturing    = false;
let predictSource  = null;   // EventSource for /predict_stream
let autoFly        = false;
let droneConnected = false;
let isTraining     = false;
let lastConsensus  = null;
let cmdCooldown    = false;
let lastSentCmd    = null;
let pendingCmd     = null;   // gesture waiting for 2-s stabilisation
let pendingCmdTimer = null;  // setTimeout handle
let batteryPollInterval = null;  // setInterval handle for battery polling

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Tab switching ──────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    tab.classList.add('active');
    $(`tab-${tab.dataset.tab}`).classList.add('active');
    if (tab.dataset.tab === 'train')  loadTrainMeta();
    if (tab.dataset.tab === 'deploy') startPredictStream();
  });
});

$('btn-goto-train').addEventListener('click', () => {
  document.querySelector('[data-tab="train"]').click();
});
$('btn-goto-deploy').addEventListener('click', () => {
  document.querySelector('[data-tab="deploy"]').click();
});

// ── Build class grid ───────────────────────────────────────────────────────
function buildClassGrid() {
  const grid = $('class-grid');
  GESTURE_CLASSES.forEach(cls => {
    const btn = document.createElement('button');
    btn.className   = 'class-btn';
    btn.dataset.cls = cls;
    btn.innerHTML   = `<div style="font-size:18px">${GESTURE_ICONS[cls]}</div>${cls.replace('_', ' ')}`;
    btn.addEventListener('click', () => selectClass(cls));
    grid.appendChild(btn);
  });
}

function selectClass(cls) {
  selectedClass = cls;
  document.querySelectorAll('.class-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.cls === cls);
  });
  $('btn-capture').disabled = false;
  $('capture-status').textContent = `Class "${cls}" selected. Show gesture + press Space.`;
}

// ── Capture ────────────────────────────────────────────────────────────────
async function captureOne() {
  if (!selectedClass || isCapturing) return;
  const target = parseInt($('target-count').value) || 50;

  isCapturing = true;
  $('btn-capture').disabled = true;

  try {
    const res  = await fetch(`${API}/capture`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: selectedClass }),
    });
    const data = await res.json();

    if (data.error) {
      $('capture-status').textContent = `⚠ ${data.error}`;
    } else {
      await refreshCounts();
      const current = parseInt(
        document.querySelector(`.count-val[data-cls="${selectedClass}"]`)?.textContent || '0'
      );
      const left = Math.max(0, target - current);
      $('capture-status').textContent =
        `✓ Captured [${selectedClass}] — ${current}/${target} samples  (${left} to go)`;
    }
  } catch (e) {
    $('capture-status').textContent = `Error: ${e.message}`;
  }

  isCapturing = false;
  $('btn-capture').disabled = false;
}

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && document.querySelector('#tab-collect.active')) {
    e.preventDefault();
    captureOne();
  }
});
$('btn-capture').addEventListener('click', captureOne);

$('btn-clear').addEventListener('click', async () => {
  if (!confirm('Clear ALL collected samples?')) return;
  await fetch(`${API}/clear_samples`, { method: 'POST' });
  await refreshCounts();
  $('capture-status').textContent = 'All samples cleared.';
});

// ── Counts ─────────────────────────────────────────────────────────────────
function buildCountsGrid() {
  const grid = $('counts-grid');
  GESTURE_CLASSES.forEach(cls => {
    const row = document.createElement('div');
    row.className = 'count-row';
    row.innerHTML = `
      <span class="count-label">${GESTURE_ICONS[cls]} ${cls.replace(/_/g, ' ')}</span>
      <div class="count-bar-wrap"><div class="count-bar" data-cls="${cls}" style="width:0%"></div></div>
      <span class="count-val" data-cls="${cls}">0</span>
      <button class="btn-delete-cls" data-cls="${cls}" title="Delete ${cls} samples">✕</button>`;
    grid.appendChild(row);
  });

  $('counts-grid').addEventListener('click', async e => {
    const btn = e.target.closest('.btn-delete-cls');
    if (!btn) return;
    const cls = btn.dataset.cls;
    if (!confirm(`Delete all samples for "${cls}"?`)) return;
    await deleteClass(cls);
  });
}

async function deleteClass(cls) {
  try {
    const res  = await fetch(`${API}/clear_class`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: cls }),
    });
    const data = await res.json();
    await refreshCounts();
    $('capture-status').textContent =
      data.ok ? `✓ Deleted ${data.removed} samples for [${cls}]` : `⚠ ${data.error}`;
  } catch (e) {
    $('capture-status').textContent = `Error: ${e.message}`;
  }
}

async function refreshCounts() {
  try {
    const res    = await fetch(`${API}/samples`);
    const counts = await res.json();
    const target = parseInt($('target-count').value) || 50;
    let   total  = 0;
    for (const [cls, n] of Object.entries(counts)) {
      const valEl = document.querySelector(`.count-val[data-cls="${cls}"]`);
      const barEl = document.querySelector(`.count-bar[data-cls="${cls}"]`);
      if (valEl) valEl.textContent = n;
      if (barEl) barEl.style.width = Math.min(100, (n / target) * 100) + '%';
      total += n;
    }
    $('total-samples').textContent = `Total: ${total} samples`;
  } catch (_) {}
}

// Poll counts every 3 s while on collect tab
setInterval(() => {
  if (document.querySelector('#tab-collect.active')) refreshCounts();
}, 3000);

// ── Train ──────────────────────────────────────────────────────────────────
async function loadTrainMeta() {
  try {
    const res    = await fetch(`${API}/samples`);
    const counts = await res.json();
    const total  = Object.values(counts).reduce((a, b) => a + b, 0);
    const active = Object.entries(counts).filter(([, n]) => n > 0).length;
    $('train-meta').textContent =
      `${total} samples across ${active} classes ready for training.`;
  } catch (_) {}
}

$('btn-train').addEventListener('click', () => {
  const log      = $('train-log');
  const resWrap  = $('results-table-wrap');
  const tbody    = $('results-tbody');
  const banner   = $('best-model-banner');

  log.textContent = '';
  resWrap.classList.add('hidden');
  tbody.innerHTML = '';
  $('btn-train').disabled = true;

  isTraining = true;
  setModelBadge('training');

  const results = [];
  const src = new EventSource(`${API}/train`);

  src.onmessage = e => {
    const { msg } = JSON.parse(e.data);
    log.textContent += msg + '\n';
    log.scrollTop = log.scrollHeight;

    // Parse result lines like "  SVM: CV=1.0000 ± 0.0000  ✓ saved"
    const m = msg.match(/^\s+(\w+):\s+CV=([\d.]+)\s+±\s+([\d.]+)/);
    if (m) results.push({ name: m[1], acc: parseFloat(m[2]), std: parseFloat(m[3]) });

    if (msg === 'DONE') {
      src.close();
      $('btn-train').disabled = false;
      isTraining = false;
      setModelBadge('ready');

      // Build results table
      results.sort((a, b) => b.acc - a.acc);
      results.forEach((r, i) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><span class="clf-badge">${r.name}</span></td>
          <td>${(r.acc * 100).toFixed(2)}%</td>
          <td>± ${(r.std * 100).toFixed(2)}%</td>
          <td><div class="acc-bar-wrap"><div class="acc-bar" style="width:${r.acc * 100}%"></div></div></td>`;
        if (i === 0) tr.style.color = 'var(--accent2)';
        tbody.appendChild(tr);
      });

      if (results.length) {
        const best = results[0];
        banner.textContent = `🏆 Best: ${best.name}  —  ${(best.acc * 100).toFixed(2)}% CV accuracy`;
      }
      resWrap.classList.remove('hidden');
    }
  };

  src.onerror = () => {
    log.textContent += '\n[Connection closed]\n';
    src.close();
    $('btn-train').disabled = false;
    isTraining = false;
    checkServer();
  };
});

// ── Deploy – Inference stream ──────────────────────────────────────────────
function buildInferencePanel() {
  const panel = $('inference-panel');
  panel.innerHTML = '';
  MODEL_NAMES.forEach(name => {
    const row = document.createElement('div');
    row.className   = 'inf-row';
    row.id          = `inf-${name}`;
    row.innerHTML   = `
      <span class="inf-name">${name}</span>
      <span class="inf-label" id="inf-label-${name}">—</span>
      <div class="inf-bar-wrap"><div class="inf-bar" id="inf-bar-${name}" style="width:0%"></div></div>
      <span class="inf-conf"  id="inf-conf-${name}"></span>`;
    panel.appendChild(row);
  });

  // Consensus row
  const crow = document.createElement('div');
  crow.id        = 'inf-consensus';
  crow.className = 'inf-row consensus';
  crow.innerHTML = `
    <span class="inf-name" style="color:var(--accent2)">VOTE</span>
    <span class="inf-label" id="inf-consensus-label" style="color:var(--accent2);font-size:15px">—</span>
    <span style="font-size:11px;color:var(--text2)" id="inf-vote-count"></span>`;
  panel.appendChild(crow);
}

function startPredictStream() {
  if (predictSource) { predictSource.close(); predictSource = null; }

  predictSource = new EventSource(`${API}/predict_stream`);
  predictSource.onmessage = e => {
    const data = JSON.parse(e.data);
    if (!data.hand) {
      MODEL_NAMES.forEach(n => {
        const lbl = $(`inf-label-${n}`);
        if (lbl) lbl.textContent = '—';
        const bar = $(`inf-bar-${n}`);
        if (bar) bar.style.width = '0%';
        const row = $(`inf-${n}`);
        if (row) row.classList.remove('consensus');
      });
      const cl = $('inf-consensus-label');
      if (cl) cl.textContent = 'No hand';
      lastSentCmd = null;
      pendingCmd  = null;
      if (pendingCmdTimer) { clearTimeout(pendingCmdTimer); pendingCmdTimer = null; }
      return;
    }

    const votes = {};
    for (const [name, { label, confidence }] of Object.entries(data.predictions)) {
      const lbl = $(`inf-label-${name}`);
      if (lbl) lbl.textContent = `${GESTURE_ICONS[label] || ''} ${label}`;
      const bar = $(`inf-bar-${name}`);
      if (bar) bar.style.width = (confidence * 100) + '%';
      const conf = $(`inf-conf-${name}`);
      if (conf) conf.textContent = (confidence * 100).toFixed(0) + '%';
      votes[label] = (votes[label] || 0) + 1;
    }

    // Consensus
    const sorted  = Object.entries(votes).sort((a, b) => b[1] - a[1]);
    const [topLbl, topCnt] = sorted[0] || ['—', 0];
    lastConsensus = topLbl;

    const cl = $('inf-consensus-label');
    if (cl) cl.textContent = `${GESTURE_ICONS[topLbl] || ''} ${topLbl}`;
    const vc = $('inf-vote-count');
    if (vc) vc.textContent = `${topCnt}/6 agree`;

    // Highlight rows that agree
    MODEL_NAMES.forEach(n => {
      const pred = data.predictions[n];
      const row  = $(`inf-${n}`);
      if (row && pred) row.classList.toggle('consensus', pred.label === topLbl);
    });

    // Auto-fly — 2-second stabilisation before sending command
    if (autoFly && droneConnected && topLbl !== pendingCmd) {
      // Gesture changed — restart the stabilisation timer
      if (pendingCmdTimer) { clearTimeout(pendingCmdTimer); pendingCmdTimer = null; }
      pendingCmd = topLbl;
      if (topLbl !== lastSentCmd) {
        pendingCmdTimer = setTimeout(() => {
          pendingCmdTimer = null;
          pendingCmd = null;
          lastSentCmd = topLbl;
          sendDroneCommand(topLbl, false);
        }, 1000);
      }
    }
  };
}

// ── Deploy – Tello ─────────────────────────────────────────────────────────

// Start polling battery level while drone is connected
function startBatteryPoll() {
  if (batteryPollInterval) clearInterval(batteryPollInterval);
  batteryPollInterval = setInterval(async () => {
    if (!droneConnected) {
      if (batteryPollInterval) clearInterval(batteryPollInterval);
      batteryPollInterval = null;
      return;
    }
    try {
      const res  = await fetch(`${API}/tello/battery`);
      const data = await res.json();
      if (data.ok && data.battery >= 0) {
        $('drone-battery').textContent = `🔋 ${data.battery}%`;
      }
    } catch (_) {
      // Silently ignore errors during polling
    }
  }, 2000); // Poll every 2 seconds
}

$('btn-connect').addEventListener('click', async () => {
  logCmd('Connecting to Tello…', 'info');
  $('btn-connect').disabled = true;
  try {
    const res  = await fetch(`${API}/tello/connect`, { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      droneConnected = true;
      $('drone-dot').className   = 'drone-dot on';
      $('drone-label').textContent = 'Connected';
      $('drone-battery').textContent = `🔋 ${data.battery}%`;
      $('btn-disconnect').disabled = false;
      $('tello-feed').src = `${API}/tello/stream`;
      $('tello-feed').classList.remove('tello-placeholder');
      logCmd(`Connected — battery ${data.battery}%`, 'ok');
      startBatteryPoll();  // Start polling battery level
    } else {
      logCmd(`Connect failed: ${data.error}`, 'err');
      $('btn-connect').disabled = false;
    }
  } catch (e) {
    logCmd(`Error: ${e.message}`, 'err');
    $('btn-connect').disabled = false;
  }
});

$('btn-disconnect').addEventListener('click', async () => {
  await fetch(`${API}/tello/disconnect`, { method: 'POST' });
  droneConnected = false;
  if (batteryPollInterval) clearInterval(batteryPollInterval);
  batteryPollInterval = null;
  $('drone-dot').className   = 'drone-dot off';
  $('drone-label').textContent = 'Disconnected';
  $('drone-battery').textContent = '';
  $('btn-connect').disabled    = false;
  $('btn-disconnect').disabled = true;
  $('ai-result-panel').classList.add('hidden');
  $('tello-feed').src = '';
  $('tello-feed').classList.add('tello-placeholder');
  logCmd('Disconnected', 'info');
});

$('auto-fly-toggle').addEventListener('change', e => {
  autoFly = e.target.checked;
  logCmd(autoFly ? 'Auto-fly ENABLED' : 'Auto-fly DISABLED', 'info');
});

// ── AI Scene Capture & Analysis ────────────────────────────────────────────
$('btn-ai-capture').addEventListener('click', async () => {
  const btn = $('btn-ai-capture');
  btn.disabled = true;
  btn.textContent = '⏳ Analyzing…';
  logCmd('Capturing frame for Ollama…', 'info');

  try {
    // 1. Get Ollama config from server
    const cfgRes = await fetch(`${API}/config`);
    const cfg    = await cfgRes.json();
    const ollamaHost  = cfg.ollama_host  || 'http://localhost:11434';
    const ollamaModel = cfg.ollama_model || 'llava';

    // 2. Grab snapshot from server (drone cam or webcam)
    const snapRes = await fetch(`${API}/tello/snapshot`);
    const snap    = await snapRes.json();
    if (!snap.ok) {
      logCmd(`AI error: ${snap.error}`, 'err');
      btn.disabled = false;
      btn.textContent = '🔍 Analyze Scene (AI)';
      return;
    }

    logCmd(`Sending ${snap.source} frame to Ollama (${ollamaModel})…`, 'info');

    // 3. Call Ollama local REST API
    const ollamaRes = await fetch(`${ollamaHost}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model:  ollamaModel,
          prompt: 'You are a drone\'s AI vision system. Describe concisely what you see in this image captured from a drone camera. Focus on the scene, objects, people, and any notable features. Keep it under 3 sentences.',
          images: [snap.image_b64],
          stream: false,
        }),
      }
    );

    const ollamaData = await ollamaRes.json();
    if (!ollamaRes.ok) {
      const errMsg = ollamaData?.error || ollamaRes.statusText;
      logCmd(`AI error: ${errMsg}`, 'err');
      btn.disabled = false;
      btn.textContent = '🔍 Analyze Scene (AI)';
      return;
    }

    const description = ollamaData.response?.trim() || '(no response)';

    // 4. Show result
    $('ai-result-text').textContent = description;
    $('ai-result-panel').classList.remove('hidden');
    logCmd(`AI: ${description}`, 'ok');

    // 5. Speak it
    if ('speechSynthesis' in window) {
      // Reset speech synthesis state
      window.speechSynthesis.cancel();
      
      // Small delay to ensure state reset
      setTimeout(() => {
        // Ensure not paused
        if (window.speechSynthesis.paused) {
          window.speechSynthesis.resume();
        }
        
        const utter = new SpeechSynthesisUtterance(description);
        utter.rate   = 1.0;
        utter.pitch  = 1.0;
        utter.volume = 1.0;
        
        // Handle potential errors
        utter.onerror = (event) => {
          logCmd(`Voice error: ${event.error}`, 'warn');
        };
        
        window.speechSynthesis.speak(utter);
      }, 100);
    }

  } catch (e) {
    logCmd(`AI error: ${e.message}`, 'err');
  }

  btn.disabled = false;
  btn.textContent = '🔍 Analyze Scene (AI)';
});

document.querySelectorAll('.btn-drone').forEach(btn => {
  btn.addEventListener('click', () => sendDroneCommand(btn.dataset.cmd, true));
});

async function sendDroneCommand(cmd, manual = false) {
  if (!droneConnected && !manual) return;
  if (cmdCooldown) return;
  cmdCooldown = true;
  setTimeout(() => { cmdCooldown = false; }, 1500);

  try {
    const res  = await fetch(`${API}/tello/command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cmd }),
    });
    const data = await res.json();
    logCmd(`${manual ? '[manual]' : '[auto]'} ${cmd} → ${data.ok ? 'OK' : data.error}`,
           data.ok ? 'ok' : 'err');
  } catch (e) {
    logCmd(`${cmd} → Error: ${e.message}`, 'err');
  }
}

// ── Command log ────────────────────────────────────────────────────────────
function logCmd(msg, type = 'info') {
  const log  = $('cmd-log');
  const entry = document.createElement('div');
  entry.className = `log-entry ${type}`;
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  log.prepend(entry);
  while (log.children.length > 100) log.removeChild(log.lastChild);
}

// ── Model badge helper ─────────────────────────────────────────────────────
function setModelBadge(state) {
  const mel = $('model-status');
  if (!mel) return;
  if (state === 'training') {
    mel.innerHTML = '<span class="badge-spinner"></span> Training…';
    mel.className = 'badge badge-training';
  } else if (state === 'ready') {
    mel.textContent = '◉ Model Ready';
    mel.className   = 'badge badge-ok';
  } else {
    mel.textContent = '◉ Not Trained';
    mel.className   = 'badge badge-err';
  }
}

// ── Server health poll ─────────────────────────────────────────────────────
async function checkServer() {
  try {
    const res  = await fetch(`${API}/status`);
    const data = await res.json();
    const ok   = res.ok;
    const el   = $('server-status');
    el.textContent = ok ? '● Server OK' : '● Server Error';
    el.className   = `badge ${ok ? 'badge-ok' : 'badge-err'}`;

    // Don't overwrite the badge while training is in progress
    if (!isTraining) {
      setModelBadge(data.model_ready ? 'ready' : 'idle');
    }
  } catch (_) {
    const el = $('server-status');
    el.textContent = '● Server Offline';
    el.className   = 'badge badge-err';
  }
}
setInterval(checkServer, 5000);

// ── Init ───────────────────────────────────────────────────────────────────
buildClassGrid();
buildCountsGrid();
buildInferencePanel();
refreshCounts();
checkServer();
