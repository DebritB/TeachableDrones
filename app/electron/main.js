const { app, BrowserWindow, ipcMain, shell } = require('electron');
const path  = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

// ── Python server ──────────────────────────────────────────────────────────

function startPythonServer() {
  const root    = path.join(__dirname, '..', '..');
  const venvPy  = path.join(root, 'venv', 'Scripts', 'python.exe');
  const server  = path.join(root, 'server.py');

  pythonProcess = spawn(venvPy, [server], {
    cwd: root,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  pythonProcess.stdout.on('data', d => process.stdout.write('[py] ' + d));
  pythonProcess.stderr.on('data', d => process.stderr.write('[py] ' + d));
  pythonProcess.on('exit', code => console.log('[py] Server exited:', code));

  console.log('[electron] Python server started, PID:', pythonProcess.pid);
}

function stopPythonServer() {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
}

// ── Wait for Flask to be ready ─────────────────────────────────────────────

async function waitForServer(url, retries = 20, delay = 500) {
  const http = require('http');
  for (let i = 0; i < retries; i++) {
    await new Promise(r => setTimeout(r, delay));
    try {
      await new Promise((resolve, reject) => {
        http.get(url, res => {
          if (res.statusCode < 500) resolve(); else reject();
        }).on('error', reject);
      });
      return true;
    } catch (_) { /* retry */ }
  }
  return false;
}

// ── Main window ────────────────────────────────────────────────────────────

async function createWindow() {
  startPythonServer();

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: '#0f0f1a',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload:          path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration:  false,
    },
    icon: path.join(__dirname, '..', 'renderer', 'assets', 'icon.png'),
  });

  const ready = await waitForServer('http://127.0.0.1:5000/status');
  if (!ready) {
    mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'error.html'));
    return;
  }

  mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));

  mainWindow.on('closed', () => { mainWindow = null; });
}

// ── App lifecycle ──────────────────────────────────────────────────────────

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  stopPythonServer();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', stopPythonServer);

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
