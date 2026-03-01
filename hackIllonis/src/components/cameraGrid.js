// ── Camera Grid Component ─────────────────────────────────────────────────────
// Renders up to 4 camera feed slots. Each slot can be added, removed, or
// switched to a different port. Subscribes via the cameras API.

import { subscribeCamera, unsubscribeCamera, isCameraActive, CAMERA_PORTS, MAX_CAMERAS } from '../api/cameras.js';

const SLOT_COUNT = MAX_CAMERAS; // max 4 slots rendered

/**
 * Mount the camera grid into the given root element.
 * @param {HTMLElement} root
 */
export function mountCameraGrid(root) {
  const section = document.createElement('section');
  section.className = 'animate-fade-up-2';
  section.innerHTML = `
    <div class="flex items-center justify-between mb-4">
      <span class="title text-lg tracking-widest text-white">CAMERA FEEDS</span>
      <div class="flex gap-2">
        <button id="add-camera-btn"
          class="text-[10px] px-2 py-1 border border-neutral-700 rounded-sm text-neutral-400 hover:border-emerald-400 hover:text-emerald-400 tracking-widest transition-colors">
          + ADD FEED
        </button>
      </div>
    </div>
    <div id="camera-grid" class="grid gap-3 grid-cols-1"></div>
  `;

  root.appendChild(section);

  const grid    = section.querySelector('#camera-grid');
  const addBtn  = section.querySelector('#add-camera-btn');

  // Track which port each slot is showing
  const slotPorts = {}; // slotIndex → port index

  // Start with 1 feed open
  addSlot(grid, slotPorts, 0);
  updateGridLayout(grid, Object.keys(slotPorts).length);

  addBtn.addEventListener('click', () => {
    const activeSlots = Object.keys(slotPorts).length;
    if (activeSlots >= SLOT_COUNT) return;
    const nextSlot = getNextFreeSlot(slotPorts);
    addSlot(grid, slotPorts, nextSlot);
    updateGridLayout(grid, Object.keys(slotPorts).length);
    if (Object.keys(slotPorts).length >= SLOT_COUNT) {
      addBtn.disabled   = true;
      addBtn.className += ' opacity-30 cursor-not-allowed';
    }
  });
}

// ── Slot creation ─────────────────────────────────────────────────────────────

function addSlot(grid, slotPorts, slotIndex) {
  // Pick the first port not already in use
  const usedPorts = Object.values(slotPorts);
  const portIndex = CAMERA_PORTS.findIndex((_, i) => !usedPorts.includes(i));
  if (portIndex === -1) return;

  slotPorts[slotIndex] = portIndex;

  const el = document.createElement('div');
  el.id        = `slot-${slotIndex}`;
  el.className = 'relative border border-neutral-800 bg-neutral-900 rounded-sm overflow-hidden ring-1 ring-emerald-400/10';
  el.innerHTML = slotHTML(slotIndex, portIndex);
  grid.appendChild(el);

  // Corner accents
  attachCorners(el);

  // Subscribe to stream
  startStream(slotIndex, portIndex, el);

  // Controls
  wireSlotControls(el, slotIndex, portIndex, grid, slotPorts);
}

function slotHTML(slotIndex, portIndex) {
  const port = CAMERA_PORTS[portIndex];
  return `
    <!-- top bar -->
    <div class="flex items-center justify-between px-3 py-1.5 border-b border-neutral-800">
      <span class="text-[10px] text-neutral-500 tracking-widest">CAM ${slotIndex + 1} — :${port}</span>
      <div class="flex items-center gap-2">
        <span id="status-${slotIndex}" class="animate-blink text-[10px] tracking-widest text-neutral-500">● CONNECTING</span>
        <!-- port switcher -->
        <select id="port-select-${slotIndex}"
          class="text-[10px] bg-neutral-800 border border-neutral-700 rounded-sm text-neutral-400 px-1 py-0.5 tracking-widest cursor-pointer">
          ${CAMERA_PORTS.map((p, i) => `<option value="${i}" ${i === portIndex ? 'selected' : ''}>:${p}</option>`).join('')}
        </select>
        <!-- remove -->
        <button id="remove-${slotIndex}"
          class="text-[10px] text-neutral-600 hover:text-red-400 tracking-widest transition-colors px-1">✕</button>
      </div>
    </div>
    <!-- stream -->
    <div class="relative bg-black">
      <img id="img-${slotIndex}" class="block w-full object-cover bg-black" alt="Camera ${slotIndex + 1}" style="aspect-ratio:4/3" />
      <div id="overlay-${slotIndex}"
        class="absolute inset-0 flex flex-col items-center justify-center bg-black gap-2">
        <svg class="w-8 h-8 text-neutral-700" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round"
            d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5 5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z" />
        </svg>
        <span class="text-[10px] text-neutral-600 tracking-widest">NO SIGNAL</span>
      </div>
    </div>
    <!-- bottom bar -->
    <div class="flex items-center justify-between px-3 py-1.5 border-t border-neutral-800">
      <span id="frames-${slotIndex}" class="text-[10px] text-neutral-600 tracking-widest">FRAMES: 0</span>
      <span id="ts-${slotIndex}" class="text-[10px] text-neutral-600 tracking-widest"></span>
    </div>
  `;
}

// ── Stream management ─────────────────────────────────────────────────────────

function startStream(slotIndex, portIndex, el) {
  let frames = 0;

  const img      = el.querySelector(`#img-${slotIndex}`);
  const overlay  = el.querySelector(`#overlay-${slotIndex}`);
  const status   = el.querySelector(`#status-${slotIndex}`);
  const framesEl = el.querySelector(`#frames-${slotIndex}`);
  const tsEl     = el.querySelector(`#ts-${slotIndex}`);

  // Clock
  const clock = setInterval(() => {
    tsEl.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });
  }, 1000);
  el._clock = clock;

  subscribeCamera(
    slotIndex,
    (url) => {
      img.onload = () => URL.revokeObjectURL(url);
      img.src    = url;
      overlay.style.display = 'none';
      frames++;
      framesEl.textContent = `FRAMES: ${frames}`;
    },
    (s) => {
      if (s === 'connected') {
        status.textContent = '● LIVE';
        status.className   = 'animate-blink text-[10px] tracking-widest text-emerald-400';
      } else {
        status.textContent = '● NO SIGNAL';
        status.className   = 'animate-blink text-[10px] tracking-widest text-neutral-500';
        overlay.style.display = 'flex';
      }
    }
  );
}

function stopStream(slotIndex, el) {
  unsubscribeCamera(slotIndex);
  clearInterval(el._clock);
}

// ── Slot controls ─────────────────────────────────────────────────────────────

function wireSlotControls(el, slotIndex, portIndex, grid, slotPorts) {
  // Port switcher
  const select = el.querySelector(`#port-select-${slotIndex}`);
  select.addEventListener('change', () => {
    const newPortIndex = parseInt(select.value, 10);
    slotPorts[slotIndex] = newPortIndex;
    stopStream(slotIndex, el);

    // Update top bar label
    const label = el.querySelector(`.text-\\[10px\\].text-neutral-500.tracking-widest`);
    if (label) label.textContent = `CAM ${slotIndex + 1} — :${CAMERA_PORTS[newPortIndex]}`;

    // Reset frame counter
    el.querySelector(`#frames-${slotIndex}`).textContent = 'FRAMES: 0';
    el.querySelector(`#overlay-${slotIndex}`).style.display = 'flex';

    startStream(slotIndex, newPortIndex, el);
  });

  // Remove button
  const removeBtn = el.querySelector(`#remove-${slotIndex}`);
  removeBtn.addEventListener('click', () => {
    stopStream(slotIndex, el);
    delete slotPorts[slotIndex];
    el.remove();
    updateGridLayout(grid, Object.keys(slotPorts).length);

    // Re-enable add button
    const addBtn = document.getElementById('add-camera-btn');
    if (addBtn) {
      addBtn.disabled  = false;
      addBtn.className = addBtn.className.replace(' opacity-30 cursor-not-allowed', '');
    }
  });
}

// ── Layout helpers ────────────────────────────────────────────────────────────

function updateGridLayout(grid, count) {
  const cols = count <= 1 ? 'grid-cols-1'
             : count === 2 ? 'grid-cols-2'
             : 'grid-cols-2';
  grid.className = `grid gap-3 ${cols}`;
}

function getNextFreeSlot(slotPorts) {
  for (let i = 0; i < SLOT_COUNT; i++) {
    if (!(i in slotPorts)) return i;
  }
  return SLOT_COUNT - 1;
}

function attachCorners(el) {
  const corners = [
    'absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-emerald-400/50 pointer-events-none z-10',
    'absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-emerald-400/50 pointer-events-none z-10',
    'absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-emerald-400/50 pointer-events-none z-10',
    'absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-emerald-400/50 pointer-events-none z-10',
  ];
  corners.forEach(cls => {
    const div = document.createElement('div');
    div.className = cls;
    el.appendChild(div);
  });
}