// ── Results Component ─────────────────────────────────────────────────────────
// Displays the annotated detection image returned by the backend.
// Expected result shape: { image: "<base64 jpeg>" | null, ... }

import { listenForResults } from '../api/detection.js';
import { setHeaderStatus }  from './header.js';

/**
 * Mount the results panel into root and start listening for detections.
 * @param {HTMLElement} root
 */
export function mountResults(root) {
  const section = document.createElement('section');
  section.id        = 'results-panel';
  section.className = 'animate-fade-up-3';
  section.innerHTML = `
    <div class="relative border border-neutral-800 bg-neutral-900 rounded-sm overflow-hidden ring-1 ring-emerald-400/10">

      <!-- Corner accents -->
      <div class="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-emerald-400/50 pointer-events-none z-10"></div>
      <div class="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-emerald-400/50 pointer-events-none z-10"></div>
      <div class="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-emerald-400/50 pointer-events-none z-10"></div>
      <div class="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-emerald-400/50 pointer-events-none z-10"></div>

      <!-- Top bar -->
      <div class="flex items-center justify-between px-4 py-2 border-b border-neutral-800">
        <span class="title text-lg tracking-widest text-white">DETECTIONS</span>
        <span id="result-badge" class="animate-blink text-[10px] tracking-widest text-neutral-500">● AWAITING</span>
      </div>

      <!-- Image area -->
      <div class="relative bg-black" style="aspect-ratio:4/3">
        <img id="result-image"
          class="block w-full h-full object-contain hidden"
          alt="Detection result" />

        <!-- Placeholder shown before first result -->
        <div id="result-placeholder"
          class="absolute inset-0 flex flex-col items-center justify-center gap-3">
          <svg class="w-10 h-10 text-neutral-700" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round"
              d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
          <span class="text-[10px] text-neutral-600 tracking-widest">AWAITING DETECTION</span>
        </div>
      </div>

      <!-- Bottom bar -->
      <div class="flex items-center justify-between px-4 py-2 border-t border-neutral-800">
        <span id="result-ts" class="text-[10px] text-neutral-600 tracking-widest">—</span>
        <span id="result-count" class="text-[10px] text-neutral-600 tracking-widest">—</span>
      </div>

    </div>
  `;

  root.appendChild(section);

  listenForResults((result) => {
    renderResult(section, result);
    setHeaderStatus('connected');
  });
}

// ── Render ────────────────────────────────────────────────────────────────────

function renderResult(section, result) {
  const imgEl       = section.querySelector('#result-image');
  const placeholder = section.querySelector('#result-placeholder');
  const badge       = section.querySelector('#result-badge');
  const countEl     = section.querySelector('#result-count');
  const tsEl        = section.querySelector('#result-ts');

  const ts    = new Date().toLocaleTimeString('en-GB', { hour12: false });
  const count = result.count ?? result.detections?.length ?? 0;

  tsEl.textContent = ts;

  if (result.image) {
    imgEl.src = `data:image/jpeg;base64,${result.image}`;
    imgEl.classList.remove('hidden');
    placeholder.style.display = 'none';

    badge.textContent = '● LIVE';
    badge.className   = 'animate-blink text-[10px] tracking-widest text-emerald-400';

    countEl.textContent = count > 0 ? `${count} FOUND` : 'NONE FOUND';
    countEl.className   = count > 0
      ? 'text-[10px] tracking-widest text-emerald-400'
      : 'text-[10px] tracking-widest text-red-400';
  } else {
    badge.textContent = '● NO IMAGE';
    badge.className   = 'text-[10px] tracking-widest text-neutral-500';
    countEl.textContent = '—';
  }
}