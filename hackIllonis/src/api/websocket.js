// ── Generic WebSocket Manager ─────────────────────────────────────────────────
// Wraps a WebSocket with auto-reconnect, typed message handlers, and a clean
// send() that queues messages if the socket isn't open yet.

const RECONNECT_DELAY_MS = 2000;

/**
 * Creates a managed WebSocket connection.
 *
 * @param {string}   url         — e.g. "ws://localhost:9999"
 * @param {object}   handlers    — { onOpen, onMessage, onClose, onError }
 * @param {string}   binaryType  — "blob" | "arraybuffer" (default: "blob")
 * @returns {{ send, close, isOpen }}
 */
export function createSocket(url, handlers = {}, binaryType = 'blob') {
  let socket   = null;
  let alive    = true;   // false after close() is called manually
  let queue    = [];     // messages buffered before socket is open

  function connect() {
    socket             = new WebSocket(url);
    socket.binaryType  = binaryType;

    socket.onopen = () => {
      console.log(`[ws] connected → ${url}`);
      handlers.onOpen?.();
      // Flush queued messages
      queue.forEach(msg => socket.send(msg));
      queue = [];
    };

    socket.onmessage = (event) => {
      handlers.onMessage?.(event);
    };

    socket.onclose = () => {
      console.log(`[ws] closed → ${url}`);
      handlers.onClose?.();
      if (alive) setTimeout(connect, RECONNECT_DELAY_MS);
    };

    socket.onerror = (err) => {
      console.error(`[ws] error → ${url}`, err);
      handlers.onError?.(err);
      socket.close();
    };
  }

  connect();

  return {
    /** Send a string or object (auto JSON-stringified) */
    send(data) {
      const payload = typeof data === 'string' ? data : JSON.stringify(data);
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(payload);
      } else {
        queue.push(payload);
      }
    },
    /** Permanently close — no reconnect */
    close() {
      alive = false;
      socket?.close();
    },
    isOpen() {
      return socket?.readyState === WebSocket.OPEN;
    },
  };
}