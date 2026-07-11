// ── Swap Controller ───────────────────────────────────────────────────────────
// Holds the single source of truth for the swap state and exposes
// performSwap() so both the button and voice command can trigger it.

let swapped     = false;
let cameraPanel = null;
let resultsPanel = null;
let leftCol     = null;
let rightCol    = null;

/**
 * Must be called once after panels are mounted.
 */
export function initSwap({ camera, results, left, right }) {
  cameraPanel  = camera;
  resultsPanel = results;
  leftCol      = left;
  rightCol     = right;
}

/**
 * Toggle swap state. Safe to call repeatedly from anywhere.
 */
export function performSwap() {
  swapped = !swapped;

  if (swapped) {
    rightCol.insertBefore(cameraPanel,  rightCol.firstElementChild);
    leftCol.insertBefore(resultsPanel,  leftCol.firstElementChild);
  } else {
    leftCol.insertBefore(cameraPanel,   leftCol.firstElementChild);
    rightCol.insertBefore(resultsPanel, rightCol.firstElementChild);
  }
}