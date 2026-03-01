// ── Detection API ─────────────────────────────────────────────────────────────
// Combines:
//   • ws://<BACKEND_HOST>:8000/ws/detection — receives detection result JSON
//   • http://<BACKEND_HOST>:8000            — REST base for HTTP requests
//
// Set BACKEND_HOST to the IP of the machine running server.py.

import { createSocket } from './websocket.js';

const BACKEND_HOST      = import.meta.env.VITE_BACKEND_HOST;
const DETECTION_WS_URL  = `ws://${BACKEND_HOST}:8000/ws/detection`;
const REST_BASE_URL     = `http://${BACKEND_HOST}:8000`;

let resultSocket = null;

// ── WebSocket: receive detection results ──────────────────────────────────────

/**
 * Start listening for detection results from the backend.
 * @param {function} onResult — (result: object) => void
 */
export function listenForResults(onResult) {
  if (resultSocket) return; // already listening

  resultSocket = createSocket(DETECTION_WS_URL, {
    onMessage(event) {
      try {
        const result = JSON.parse(event.data);
        onResult(result);
      } catch (e) {
        console.error('[detection] Failed to parse result JSON', e);
      }
    },
  }, 'arraybuffer');
}

// ── REST helpers ──────────────────────────────────────────────────────────────

/**
 * Send a transcript + optional tool hints to the backend for processing.
 * The backend handles Whisper / Gemini / Modal detection.
 *
 * @param {string}   transcript — raw speech text
 * @param {string[]} tools      — optional list of tool names detected client-side
 * @returns {Promise<object>}
 */
export async function sendTranscript(transcript, tools = []) {
  return post('/detect', { transcript, tools });
}

/**
 * Generic GET helper.
 * @param {string} path — e.g. "/status"
 */
export async function get(path) {
  const res = await fetch(`${REST_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

/**
 * Generic POST helper.
 * @param {string} path   — e.g. "/detect"
 * @param {object} body   — will be JSON stringified
 */
export async function post(path, body = {}) {
  const res = await fetch(`${REST_BASE_URL}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json();
}