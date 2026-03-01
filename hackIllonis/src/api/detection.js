// ── Detection API ─────────────────────────────────────────────────────────────
// All communication via HTTP REST.
// Set VITE_BACKEND_HOST in your .env to the IP of the machine running server.py.

const BACKEND_HOST = import.meta.env.VITE_BACKEND_HOST;
const REST_BASE_URL = `http://${BACKEND_HOST}:8000`;

// ── REST helpers ──────────────────────────────────────────────────────────────

/**
 * Send a transcript to the backend for processing.
 * Returns the detection result directly from the POST response.
 *
 * @param {string}   transcript — raw speech text
 * @param {string[]} tools      — optional list of tool names
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
 * @param {string} path — e.g. "/detect"
 * @param {object} body — will be JSON stringified
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