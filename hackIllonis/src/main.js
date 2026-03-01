// ── main.js — Entry Point ─────────────────────────────────────────────────────

import './style.css';

const tw = document.createElement('script');
tw.src   = 'https://cdn.tailwindcss.com';
document.head.appendChild(tw);

document.title = 'ToolFinder';

import { mountHeader }     from './components/header.js';
import { mountCameraGrid } from './components/cameraGrid.js';
import { mountSpeech }     from './components/speech.js';
import { mountResults }    from './components/results.js';
import { initSwap, performSwap } from './components/swap.js';

// ── Layout ────────────────────────────────────────────────────────────────────
document.body.className = 'bg-[#0a0a0a] text-neutral-200 min-h-screen overflow-x-hidden';

mountHeader(document.body);

const scanline = document.createElement('div');
scanline.className = 'pointer-events-none fixed inset-0 z-50';
scanline.style.background = 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.06) 2px, rgba(0,0,0,0.06) 4px)';
document.body.appendChild(scanline);

const main = document.createElement('main');
main.className = 'max-w-7xl mx-auto px-4 py-8 grid grid-cols-1 xl:grid-cols-[1fr_360px] gap-6';
document.body.appendChild(main);

const left = document.createElement('div');
left.id = 'col-left';
left.className = 'flex flex-col gap-6';
main.appendChild(left);

const right = document.createElement('div');
right.id = 'col-right';
right.className = 'flex flex-col gap-6';
main.appendChild(right);

// Mount and capture stable node refs
mountCameraGrid(left);
const cameraPanel = left.firstElementChild;

mountResults(right);
const resultsPanel = document.getElementById('results-panel');

mountSpeech(right);

// Register panels with swap controller
initSwap({ camera: cameraPanel, results: resultsPanel, left, right });

// ── Swap button ───────────────────────────────────────────────────────────────

const swapBtn = document.createElement('button');
swapBtn.className = [
  'fixed bottom-6 right-6 z-50',
  'text-[10px] tracking-widest px-3 py-2',
  'border border-neutral-700 rounded-sm',
  'bg-neutral-900 text-neutral-400',
  'hover:border-emerald-400 hover:text-emerald-400',
  'transition-colors duration-200',
].join(' ');
swapBtn.textContent = '⇄ SWAP VIEW';
swapBtn.addEventListener('click', performSwap);
document.body.appendChild(swapBtn);