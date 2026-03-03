// ── Header Component ──────────────────────────────────────────────────────────
/**
 * Renders the top header bar and mounts pointer toggle logic.
 * @param {HTMLElement} root — element to append the header into
 */
export function mountHeader(root) {
  const header = document.createElement('header');
  header.className = 'animate-fade-up border-b border-neutral-800 px-6 py-4 flex items-center justify-between';
  header.innerHTML = `
    <div class="flex items-center gap-3">
      <div class="animate-pulse-dot w-2 h-2 rounded-full bg-emerald-400"></div>
      <span class="title text-2xl tracking-widest text-white">TOOLFINDER</span>
    </div>

    <span class="text-xs text-neutral-500 hidden md:block tracking-widest">
      MAKERSPACE VISION ASSISTANT
    </span>

    <div class="flex items-center gap-3">
      <div id="conn-badge" class="text-xs px-2 py-1 border border-neutral-700 rounded-sm text-neutral-500 tracking-widest">
        READY
      </div>
      <button
        id="pointer-btn"
        class="text-xs px-3 py-1.5 border rounded-sm tracking-widest transition-colors duration-200 border-neutral-700 text-neutral-400 hover:border-neutral-500"
      >
        POINTER OFF
      </button>
    </div>
  `;

  root.appendChild(header);
}

/**
 * Update the connection status badge from anywhere.
 * @param {'connected'|'disconnected'|'listening'} status
 */
export function setHeaderStatus(status) {
  const badge = document.getElementById('conn-badge');
  if (!badge) return;

  const map = {
    connected:    ['CONNECTED',   'border-emerald-400 text-emerald-400'],
    disconnected: ['DISCONNECTED','border-neutral-700 text-neutral-500'],
    listening:    ['LISTENING',   'border-yellow-400 text-yellow-400'],
    detecting:    ['DETECTING…',  'border-blue-400 text-blue-400'],
  };

  const [label, cls] = map[status] ?? ['READY', 'border-neutral-700 text-neutral-500'];
  badge.textContent = label;
  badge.className   = `text-xs px-2 py-1 border rounded-sm tracking-widest ${cls}`;
}