// ── Speech Component ──────────────────────────────────────────────────────────
// Fixed-height card. Live area shows interim/final text, with a rotating hint
// when idle. Hint rotates every 5s but pauses for 20s after the user speaks.

import { sendTranscript } from '../api/detection.js';
import { performSwap }   from './swap.js';
import { setHeaderStatus } from './header.js';

const HINTS = [
  '"where is my hammer"',
  '"find the red screwdriver"',
  '"locate the drill"',
  '"where are the pliers"',
  '"find the large wrench"',
  '"where is the tape measure"',
  '"locate the blue clamp"',
  '"find the yellow level"',
  '"where is the chisel"',
  '"find the saw"',
];

const HINT_INTERVAL_MS  = 5000;
const SPEECH_SILENCE_MS = 20000;

export function mountSpeech(root) {
  const section = document.createElement('section');
  section.className = 'animate-fade-up-3';
  section.innerHTML = `
    <div class="border border-neutral-800 bg-neutral-900 rounded-sm overflow-hidden ring-1 ring-emerald-400/10 flex flex-col h-[380px]">

      <!-- Header — fixed -->
      <div class="flex-shrink-0 flex items-center justify-between px-4 py-2 border-b border-neutral-800">
        <span class="title text-lg tracking-widest text-white">VOICE QUERY</span>
        <span id="speech-status" class="text-[10px] tracking-widest text-neutral-500">● IDLE</span>
      </div>

      <!-- Live area — hint when idle, transcript when speaking -->
      <div class="flex-shrink-0 px-4 py-4 border-b border-neutral-800">
        <p id="transcript-display"
          class="text-sm h-[44px] overflow-hidden leading-5 text-neutral-600 tracking-wide italic transition-all duration-300">
          ${HINTS[0]}
        </p>
      </div>

      <!-- Send status — always-visible slot -->
      <div class="flex-shrink-0 px-4 py-2 border-b border-neutral-800">
        <div id="send-status-bar" class="px-3 py-1.5 rounded-sm border border-neutral-800 bg-neutral-800/40 text-[10px] tracking-widest flex items-center justify-between">
          <span id="send-status-text" class="text-neutral-600">READY</span>
          <span id="send-status-time" class="text-neutral-700"></span>
        </div>
      </div>

      <!-- Query log — scrollable, fills remaining space -->
      <div class="flex flex-col min-h-0 flex-1">
        <div class="flex-shrink-0 flex items-center justify-between px-4 py-1.5 border-b border-neutral-800">
          <span class="text-[10px] text-neutral-600 tracking-widest">QUERY LOG</span>
          <button id="clear-log" class="text-[10px] text-neutral-600 hover:text-red-400 tracking-widest transition-colors">CLEAR</button>
        </div>
        <ul id="log-list" class="divide-y divide-neutral-800 overflow-y-auto flex-1">
          <li id="log-empty" class="px-4 py-3 text-[10px] text-neutral-700 tracking-widest text-center">NO QUERIES YET</li>
        </ul>
      </div>

      <!-- Controls — fixed -->
      <div class="flex-shrink-0 px-4 py-3 border-t border-neutral-800 flex items-center gap-3">
        <button id="speech-toggle"
          class="text-xs px-3 py-1.5 border border-neutral-700 rounded-sm text-neutral-400 hover:border-emerald-400 hover:text-emerald-400 tracking-widest transition-colors duration-200">
          START LISTENING
        </button>
        <span id="speech-error" class="text-[10px] text-red-400 tracking-widest hidden"></span>
      </div>

    </div>
  `;

  root.appendChild(section);

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showError(section, 'Web Speech API not supported. Use Chrome or Edge.');
    return;
  }

  // Shared state for hint rotation
  const state = {
    lastSpeechAt: 0,   // timestamp of last speech activity
    hintIndex: 0,
    isListening: false,
  };

  startHintRotation(section, state);

  const recognition = buildRecognition(SpeechRecognition, section, state);
  wireToggle(section, recognition, state);
  wireClearLog(section);
}

// ── Hint rotation ─────────────────────────────────────────────────────────────

function startHintRotation(section, state) {
  const display = section.querySelector('#transcript-display');

  setInterval(() => {
    // Only rotate if user hasn't spoken recently and mic isn't active
    const silentLongEnough = (Date.now() - state.lastSpeechAt) >= SPEECH_SILENCE_MS;
    if (!state.isListening && silentLongEnough) {
      state.hintIndex = (state.hintIndex + 1) % HINTS.length;
      display.style.opacity = '0';
      setTimeout(() => {
        display.textContent = HINTS[state.hintIndex];
        display.className   = 'text-sm h-[44px] overflow-hidden leading-5 text-neutral-600 tracking-wide italic transition-all duration-300';
        display.style.opacity = '1';
      }, 300);
    }
  }, HINT_INTERVAL_MS);
}

// ── Recognition ───────────────────────────────────────────────────────────────

function buildRecognition(SpeechRecognition, section, state) {
  const recognition          = new SpeechRecognition();
  recognition.continuous     = true;
  recognition.interimResults = true;
  recognition.lang           = 'en-US';

  const display   = section.querySelector('#transcript-display');
  const statusEl  = section.querySelector('#speech-status');
  const toggleBtn = section.querySelector('#speech-toggle');

  recognition.onstart = () => {
    state.isListening = true;
    setStatus(statusEl, 'listening');
    setHeaderStatus('listening');
    toggleBtn.textContent = 'STOP LISTENING';
    toggleBtn.className   = 'text-xs px-3 py-1.5 border border-red-400 rounded-sm text-red-400 tracking-widest transition-colors duration-200';
    display.textContent   = '…';
    display.className     = 'text-sm h-[44px] overflow-hidden leading-5 text-neutral-600 tracking-wide italic';
    display.style.opacity = '1';
  };

  recognition.onend = () => {
    state.isListening = false;
    setStatus(statusEl, 'idle');
    setHeaderStatus('connected');
    toggleBtn.textContent = 'START LISTENING';
    toggleBtn.className   = 'text-xs px-3 py-1.5 border border-neutral-700 rounded-sm text-neutral-400 hover:border-emerald-400 hover:text-emerald-400 tracking-widest transition-colors duration-200';
    // Show current hint immediately on stop
    display.textContent = HINTS[state.hintIndex];
    display.className   = 'text-sm h-[44px] overflow-hidden leading-5 text-neutral-600 tracking-wide italic';
  };

  recognition.onresult = (event) => {
    state.lastSpeechAt = Date.now(); // reset silence timer on any speech
    let interim = '';
    let final   = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) final += t;
      else interim += t;
    }

    if (interim) {
      display.textContent = interim;
      display.className   = 'text-sm h-[44px] overflow-hidden leading-5 text-neutral-500 tracking-wide italic';
      display.style.opacity = '1';
    }

    if (final) {
      const text = final.trim().toLowerCase();
      display.textContent = `"${text}"`;
      display.className   = 'text-sm h-[44px] overflow-hidden leading-5 text-neutral-200 tracking-wide';

      // Voice command: "swap" triggers layout swap, does not send to backend
      if (/\bswap\b/.test(text)) {
        performSwap();
        setStatus(statusEl, 'listening');
        return;
      }

      setStatus(statusEl, 'sending');
      setHeaderStatus('detecting');
      dispatchToBackend(text, section, state, statusEl);
    }
  };

  recognition.onerror = (event) => {
    console.error('[speech] error:', event.error);
    showError(section, `MIC ERROR: ${event.error.toUpperCase()}`);
  };

  return recognition;
}

// ── Toggle ────────────────────────────────────────────────────────────────────

function wireToggle(section, recognition, state) {
  const btn = section.querySelector('#speech-toggle');
  let active = false;
  btn.addEventListener('click', () => {
    if (!active) { recognition.start(); active = true; }
    else         { recognition.stop();  active = false; }
  });
}

function wireClearLog(section) {
  section.querySelector('#clear-log').addEventListener('click', () => {
    const list = section.querySelector('#log-list');
    list.innerHTML = '<li id="log-empty" class="px-4 py-3 text-[10px] text-neutral-700 tracking-widest text-center">NO QUERIES YET</li>';
  });
}

// ── Backend dispatch ──────────────────────────────────────────────────────────

async function dispatchToBackend(transcript, section, state, statusEl) {
  setSendBar(section, 'sending');

  try {
    console.log(`[speech] → "${transcript}"`);
    const result = await sendTranscript(transcript);
    console.log('[speech] ✓', result);
    setSendBar(section, 'sent');
    setStatus(statusEl, 'listening');
    setHeaderStatus('connected');
    addToLog(section, transcript, 'ok');
  } catch (err) {
    console.error('[speech] ✗', err);
    setSendBar(section, 'error');
    setStatus(statusEl, 'listening');
    setHeaderStatus('listening');
    addToLog(section, transcript, 'error');
  }
}

// ── Send status bar ───────────────────────────────────────────────────────────

function setSendBar(section, state) {
  const bar    = section.querySelector('#send-status-bar');
  const textEl = section.querySelector('#send-status-text');
  const timeEl = section.querySelector('#send-status-time');
  const ts     = new Date().toLocaleTimeString('en-GB', { hour12: false });

  const map = {
    sending: { text: '⟳ SENDING…',   cls: 'text-yellow-400',  border: 'border-yellow-400/30 bg-yellow-400/5' },
    sent:    { text: '✓ SENT',        cls: 'text-emerald-400', border: 'border-emerald-400/30 bg-emerald-400/5' },
    error:   { text: '✗ SEND FAILED', cls: 'text-red-400',     border: 'border-red-400/30 bg-red-400/5' },
  };

  const { text, cls, border } = map[state];
  bar.className      = `px-3 py-1.5 rounded-sm border text-[10px] tracking-widest flex items-center justify-between ${border}`;
  textEl.textContent = text;
  textEl.className   = cls;
  timeEl.textContent = ts;

  if (state !== 'sending') {
    setTimeout(() => {
      bar.className      = 'px-3 py-1.5 rounded-sm border border-neutral-800 bg-neutral-800/40 text-[10px] tracking-widest flex items-center justify-between';
      textEl.textContent = 'READY';
      textEl.className   = 'text-neutral-600';
      timeEl.textContent = '';
    }, 3000);
  }
}

// ── Query log ─────────────────────────────────────────────────────────────────

function addToLog(section, transcript, status) {
  const list  = section.querySelector('#log-list');
  const empty = section.querySelector('#log-empty');
  if (empty) empty.remove();

  const ts       = new Date().toLocaleTimeString('en-GB', { hour12: false });
  const dotColor = status === 'ok' ? 'bg-emerald-400' : 'bg-red-400';
  const dotTitle = status === 'ok' ? 'Sent' : 'Failed';

  const li = document.createElement('li');
  li.className = 'px-4 py-2 flex items-center justify-between gap-2';
  li.innerHTML = `
    <div class="flex items-center gap-2 min-w-0">
      <div class="w-1.5 h-1.5 rounded-full flex-shrink-0 ${dotColor}" title="${dotTitle}"></div>
      <span class="text-xs text-neutral-300 tracking-wide truncate">"${transcript}"</span>
    </div>
    <span class="text-[10px] text-neutral-600 tracking-widest flex-shrink-0">${ts}</span>
  `;
  list.insertBefore(li, list.firstChild);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function setStatus(el, state) {
  const map = {
    idle:     ['● IDLE',      'text-[10px] tracking-widest text-neutral-500'],
    listening:['● LISTENING', 'text-[10px] tracking-widest text-yellow-400 animate-blink'],
    sending:  ['● SENDING',   'text-[10px] tracking-widest text-yellow-400 animate-blink'],
  };
  const [label, cls] = map[state] ?? map.idle;
  el.textContent = label;
  el.className   = cls;
}

function showError(section, msg) {
  const el = section.querySelector('#speech-error');
  if (!el) return;
  el.textContent = msg;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 4000);
}