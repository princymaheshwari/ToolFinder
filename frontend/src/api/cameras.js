// ── Camera Feed Manager ───────────────────────────────────────────────────────
// Each camera is a WebSocket that pushes JPEG blobs.
// Ports 9999, 10000, 10001, 10002 map to camera slots 0–3.

import { createSocket } from './websocket.js';

export const CAMERA_PORTS = [9999, 10000, 10001, 10002];
export const MAX_CAMERAS  = CAMERA_PORTS.length;

// Active sockets keyed by slot index (0–3)
const activeSockets = {};

/**
 * Subscribe a camera slot to its WebSocket stream.
 * Calls onFrame(objectURL) each time a blob arrives.
 * Calls onStatus('connected'|'disconnected') on state changes.
 *
 * @param {number}   slot       — 0–3
 * @param {function} onFrame    — (objectURL: string) => void
 * @param {function} onStatus   — ('connected'|'disconnected') => void
 */
export function subscribeCamera(slot, onFrame, onStatus) {
  if (activeSockets[slot]) unsubscribeCamera(slot); // clean up first

  const port = CAMERA_PORTS[slot];
  const url  = `ws://localhost:${port}`;

  const sock = createSocket(url, {
    onOpen()  { onStatus?.('connected'); },
    onClose() { onStatus?.('disconnected'); },
    onMessage(event) {
      const url = URL.createObjectURL(event.data);
      onFrame(url);
    },
  }, 'blob');

  activeSockets[slot] = sock;
}

/**
 * Unsubscribe and permanently close a camera slot.
 */
export function unsubscribeCamera(slot) {
  activeSockets[slot]?.close();
  delete activeSockets[slot];
}

/**
 * Returns true if a slot currently has an active socket.
 */
export function isCameraActive(slot) {
  return !!activeSockets[slot];
}