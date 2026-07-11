// ── Pointer Control ───────────────────────────────────────────────────────────
// Sends pointer state over ws://localhost:9998
// The local Python process listens here and starts/stops its detection loop.

import { createSocket } from './websocket.js';

const POINTER_URL = 'ws://localhost:9998';

let sock         = null;
let pointerState = false;

function ensureConnected() {
  if (!sock) {
    sock = createSocket(POINTER_URL, {
      onOpen()  { console.log('[pointer] control socket ready'); },
      onClose() { console.log('[pointer] control socket closed'); },
    });
  }
}

/**
 * Set pointer active state and broadcast to backend.
 * @param {boolean} active
 */
export function setPointer(active) {
  ensureConnected();
  pointerState = active;
  sock.send({ type: 'pointer', active });
  console.log(`[pointer] ${active ? 'ON' : 'OFF'}`);
}

/**
 * Toggle pointer and return new state.
 * @returns {boolean} new state
 */
export function togglePointer() {
  setPointer(!pointerState);
  return pointerState;
}

export function getPointerState() {
  return pointerState;
}