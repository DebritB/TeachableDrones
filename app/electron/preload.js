const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('gestureFly', {
  version: '4.0.0',
  platform: process.platform,   // 'darwin' | 'win32' | 'linux'
});
