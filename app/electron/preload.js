const { contextBridge } = require('electron');

// Expose nothing — renderer talks directly to Flask on localhost via fetch/SSE
// This preload exists as a placeholder for future IPC if needed.
contextBridge.exposeInMainWorld('gestureFly', {
  version: '4.0.0',
});
