// ============================================================
//  Training Web Worker – Evolution Strategy (OpenAI-style ES)
//  Maintains a center genome (mu), samples noisy perturbations,
//  evaluates them, and updates mu using fitness-weighted gradients.
// ============================================================

const g_acc = 9.81;
const PHYSICS_DT = 0.02;
const PHYSICS_SUB = 4;
const MAX_FORCE = 60;
const NN_IN = 8;
const NN_HID1 = 24;
const NN_HID2 = 16;
const NN_OUT = 1;
const GENOME_LEN = (NN_IN + 1) * NN_HID1 + (NN_HID1 + 1) * NN_HID2 + (NN_HID2 + 1) * NN_OUT;

// --- ES hyperparameters ---
const ES_LR = 0.05;           // learning rate
const ES_SIGMA_INIT = 0.1;    // initial noise std
const ES_SIGMA_MIN = 0.01;    // minimum noise
const ES_SIGMA_MAX = 0.5;     // maximum noise
const ES_DECAY = 0.005;       // weight decay (L2 regularization)
const EVAL_RUNS = 3;

let popSize = 300;
let mu = null;                 // center genome
let sigma = ES_SIGMA_INIT;     // current noise scale
let epsilons = [];             // noise vectors for current gen
let pop = [];
let gen = 0;
let evalIdx = 0;
let bestFit = -Infinity;
let allBestFit = -Infinity;
let allBestGenome = null;
let fitHistory = [];
let avgFitHistory = [];
let currentPreset = 0;
let running = true;
let totalEvals = 0;
let trainStartTime = 0;
let staleGens = 0;
let lastBestFit = -Infinity;
let lastStatusTime = 0;

const PRESETS = [
  { M: 2.0, m1: 0.5, m2: 0.5, L1: 0.5, L2: 0.5, track: 3.0, fric: 0.1, th1: 0.15, th2: 0.15, steps: 800 },
  { M: 2.0, m1: 0.8, m2: 0.5, L1: 0.5, L2: 0.5, track: 3.0, fric: 0.1, th1: 0.35, th2: 0.35, steps: 800 },
  { M: 2.5, m1: 0.4, m2: 1.8, L1: 0.4, L2: 0.55, track: 3.5, fric: 0.1, th1: 0.2, th2: 0.2, steps: 900 },
  { M: 3.0, m1: 0.5, m2: 0.5, L1: 0.4, L2: 0.4, track: 4.0, fric: 0.05, th1: Math.PI, th2: Math.PI, steps: 1500 },
  { M: 3.0, m1: 0.35, m2: 0.3, L1: 0.7, L2: 0.6, track: 4.0, fric: 0.1, th1: 0.25, th2: 0.25, steps: 800 },
];

// --- Simplified physics (Semi-implicit Euler, fast) ---
function simDerivs(Mc, m1, m2, L1, L2, fric, x, xd, t1, t1d, t2, t2d, F) {
  let s1 = Math.sin(t1), c1 = Math.cos(t1);
  let s2 = Math.sin(t2), c2 = Math.cos(t2);
  let s12 = Math.sin(t1 - t2), c12 = Math.cos(t1 - t2);

  // 3x3 system: [xdd, t1dd, t2dd]
  let a00 = Mc + m1 + m2, a01 = (m1 + m2) * L1 * c1, a02 = m2 * L2 * c2;
  let a10 = a01, a11 = (m1 + m2) * L1 * L1, a12 = m2 * L1 * L2 * c12;
  let a20 = a02, a21 = a12, a22 = m2 * L2 * L2;

  let b0 = F - fric * xd + (m1 + m2) * L1 * t1d * t1d * s1 + m2 * L2 * t2d * t2d * s2;
  let b1 = (m1 + m2) * g_acc * L1 * s1 - m2 * L1 * L2 * t2d * t2d * s12;
  let b2 = m2 * g_acc * L2 * s2 + m2 * L1 * L2 * t1d * t1d * s12;

  // Solve 3x3 via Cramer's rule (faster than Gaussian for small fixed size)
  let det = a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20);
  if (Math.abs(det) < 1e-12) return [0, 0, 0];
  let invDet = 1 / det;
  let xdd  = (b0*(a11*a22-a12*a21) - a01*(b1*a22-a12*b2) + a02*(b1*a21-a11*b2)) * invDet;
  let t1dd = (a00*(b1*a22-a12*b2) - b0*(a10*a22-a12*a20) + a02*(a10*b2-b1*a20)) * invDet;
  let t2dd = (a00*(a11*b2-b1*a21) - a01*(a10*b2-b1*a20) + b0*(a10*a21-a11*a20)) * invDet;
  return [xdd, t1dd, t2dd];
}

function evalAgent(genome, preset) {
  let p = preset;

  // Parse genome inline for speed (2 hidden layers)
  let gi = 0;
  let w1 = new Float64Array(NN_HID1 * NN_IN);
  let b1 = new Float64Array(NN_HID1);
  for (let j = 0; j < NN_HID1; j++) {
    for (let k = 0; k < NN_IN; k++) w1[j * NN_IN + k] = genome[gi++];
    b1[j] = genome[gi++];
  }
  let w2 = new Float64Array(NN_HID2 * NN_HID1);
  let b2 = new Float64Array(NN_HID2);
  for (let j = 0; j < NN_HID2; j++) {
    for (let k = 0; k < NN_HID1; k++) w2[j * NN_HID1 + k] = genome[gi++];
    b2[j] = genome[gi++];
  }
  let w3 = new Float64Array(NN_HID2);
  for (let k = 0; k < NN_HID2; k++) w3[k] = genome[gi++];
  let b3out = genome[gi++];

  let hid1 = new Float64Array(NN_HID1);
  let hid2 = new Float64Array(NN_HID2);
  let totalR = 0;
  let dt = PHYSICS_DT;

  // Average over multiple runs to reduce noise
  for (let run = 0; run < EVAL_RUNS; run++) {
    let x = (Math.random() - 0.5) * 0.1;
    let xd = 0;
    let t1 = p.th1 + (Math.random() - 0.5) * 0.1;
    let t1d = 0;
    let t2 = p.th2 + (Math.random() - 0.5) * 0.1;
    let t2d = 0;
    let runR = 0;

    for (let step = 0; step < p.steps; step++) {
      if (Math.abs(x) > p.track) break;
      let ct1 = Math.cos(t1), ct2 = Math.cos(t2);
      if (ct1 < -0.5 && ct2 < -0.5) break;

      // NN forward pass (inlined, 2 hidden layers)
      let inp0 = x / p.track;
      let inp1 = xd * 0.15;
      let inp2 = Math.sin(t1), inp3 = ct1;
      let inp4 = t1d * 0.08;
      let inp5 = Math.sin(t2), inp6 = ct2;
      let inp7 = t2d * 0.08;

      for (let j = 0; j < NN_HID1; j++) {
        let s = b1[j];
        let base = j * NN_IN;
        s += w1[base]*inp0 + w1[base+1]*inp1 + w1[base+2]*inp2 + w1[base+3]*inp3;
        s += w1[base+4]*inp4 + w1[base+5]*inp5 + w1[base+6]*inp6 + w1[base+7]*inp7;
        hid1[j] = Math.tanh(s);
      }
      for (let j = 0; j < NN_HID2; j++) {
        let s = b2[j];
        let base = j * NN_HID1;
        for (let k = 0; k < NN_HID1; k++) s += w2[base + k] * hid1[k];
        hid2[j] = Math.tanh(s);
      }
      let out = b3out;
      for (let k = 0; k < NN_HID2; k++) out += w3[k] * hid2[k];
      out = Math.tanh(out);

      let force = out * MAX_FORCE;
      force = force < -MAX_FORCE ? -MAX_FORCE : (force > MAX_FORCE ? MAX_FORCE : force);

      // Semi-implicit Euler with substeps
      let sub_dt = dt / PHYSICS_SUB;
      for (let sub = 0; sub < PHYSICS_SUB; sub++) {
        let acc = simDerivs(p.M, p.m1, p.m2, p.L1, p.L2, p.fric, x, xd, t1, t1d, t2, t2d, force);
        xd += acc[0] * sub_dt;
        t1d += acc[1] * sub_dt;
        t2d += acc[2] * sub_dt;
        x += xd * sub_dt;
        t1 += t1d * sub_dt;
        t2 += t2d * sub_dt;
      }

      // Reward: upright + centered + calm
      runR += (1 + ct1) / 2 + (1 + ct2) / 2
            - 0.01 * x * x
            - 0.0003 * (t1d * t1d + t2d * t2d);
    }
    totalR += runR;
  }
  return totalR / EVAL_RUNS;
}

function gaussRand() {
  return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

// --- ES: Initialize center genome + first population ---
function initPop() {
  mu = new Float64Array(GENOME_LEN);
  for (let i = 0; i < GENOME_LEN; i++) mu[i] = (Math.random() - 0.5) * 1.0;
  sigma = ES_SIGMA_INIT;
  samplePopulation();
  gen = 0; evalIdx = 0;
  bestFit = -Infinity;
  allBestFit = -Infinity;
  allBestGenome = null;
  fitHistory = [];
  avgFitHistory = [];
  totalEvals = 0;
  staleGens = 0;
  lastBestFit = -Infinity;
  trainStartTime = performance.now();
  lastStatusTime = 0;
}

// --- ES: Sample population around mu ---
function samplePopulation() {
  pop = [];
  epsilons = [];
  for (let i = 0; i < popSize; i++) {
    let eps = new Float64Array(GENOME_LEN);
    let g = new Float64Array(GENOME_LEN);
    for (let j = 0; j < GENOME_LEN; j++) {
      eps[j] = gaussRand();
      g[j] = mu[j] + sigma * eps[j];
    }
    epsilons.push(eps);
    pop.push({ g, fit: 0 });
  }
}

function getTopGenomes(n) {
  let sorted = pop.slice().sort((a, b) => b.fit - a.fit);
  let top = [];
  for (let i = 0; i < Math.min(n, sorted.length); i++) {
    if (sorted[i].fit > 0) top.push({ genome: Array.from(sorted[i].g), fit: sorted[i].fit });
  }
  return top;
}

function sendStatus() {
  self.postMessage({
    type: "status",
    gen: gen,
    evalIdx: evalIdx,
    popSize: popSize,
    bestFit: bestFit,
    allBestFit: allBestFit,
    fitHistory: fitHistory.slice(-100),
    avgFitHistory: avgFitHistory.slice(-100),
    totalEvals: totalEvals,
    elapsed: performance.now() - trainStartTime,
    topGenomes: getTopGenomes(5),
  });
  lastStatusTime = performance.now();
}

// --- ES: Update mu using rank-based fitness-weighted gradients ---
function nextGen() {
  // Sort by fitness to compute ranks
  let indices = [];
  for (let i = 0; i < popSize; i++) indices.push(i);
  indices.sort((a, b) => pop[a].fit - pop[b].fit); // ascending

  // Rank-based utility: worst=-0.5, best=+0.5
  let utility = new Float64Array(popSize);
  for (let rank = 0; rank < popSize; rank++) {
    utility[indices[rank]] = rank / (popSize - 1) - 0.5;
  }

  // Compute fitness stats for logging
  let topIdx = indices[popSize - 1];
  let topFit = pop[topIdx].fit;
  let avgFit = 0;
  for (let i = 0; i < popSize; i++) avgFit += pop[i].fit;
  avgFit /= popSize;
  fitHistory.push(topFit);
  avgFitHistory.push(avgFit);

  if (topFit > bestFit) bestFit = topFit;
  if (topFit > allBestFit) {
    allBestFit = topFit;
    allBestGenome = new Float64Array(pop[topIdx].g);
    self.postMessage({
      type: "best",
      genome: Array.from(allBestGenome),
      fit: allBestFit,
      gen: gen,
    });
  }

  // Track stagnation for sigma adaptation
  if (topFit > lastBestFit + 0.5) {
    staleGens = 0;
    lastBestFit = topFit;
  } else {
    staleGens++;
  }

  // Update mu: gradient = sum(utility[i] * epsilon[i]) / (popSize * sigma)
  for (let j = 0; j < GENOME_LEN; j++) {
    let grad = 0;
    for (let i = 0; i < popSize; i++) {
      grad += utility[i] * epsilons[i][j];
    }
    mu[j] += ES_LR / (popSize * sigma) * grad;
    // Weight decay
    mu[j] *= (1 - ES_DECAY);
  }

  // Sigma adaptation: increase if stale, decrease if improving
  if (staleGens > 15) {
    sigma = Math.min(ES_SIGMA_MAX, sigma * 1.05);
  } else if (staleGens === 0) {
    sigma = Math.max(ES_SIGMA_MIN, sigma * 0.97);
  }

  // Sample new population around updated mu
  samplePopulation();
  gen++;
}

function trainLoop() {
  if (!running) return;
  let preset = PRESETS[currentPreset];

  // Check if generation is complete before evaluating
  if (evalIdx >= popSize) {
    nextGen();
    evalIdx = 0;
    sendStatus();
  }

  // Evaluate a batch of agents
  let batchSize = Math.min(10, popSize - evalIdx);
  for (let i = 0; i < batchSize; i++) {
    pop[evalIdx].fit = evalAgent(pop[evalIdx].g, preset);
    totalEvals++;
    evalIdx++;
  }

  // Send progress updates every ~50ms so UI stays responsive
  let now = performance.now();
  if (now - lastStatusTime > 50) sendStatus();

  // Yield back and continue
  setTimeout(trainLoop, 0);
}

// --- Message handler ---
self.onmessage = function(e) {
  let msg = e.data;
  if (msg.type === "start") {
    currentPreset = msg.preset || 0;
    popSize = msg.popSize || 300;
    running = true;
    initPop();
    trainLoop();
  }
  if (msg.type === "preset") {
    currentPreset = msg.preset;
    initPop();
  }
  if (msg.type === "popSize") {
    popSize = msg.popSize;
    initPop();
  }
  if (msg.type === "stop") {
    running = false;
  }
};
