// ============================================================
//  PENDULUM BALANCE – p5.js neuroevolution simulation
//  A neural network learns to balance an inverted double
//  pendulum on a cart using a genetic algorithm.
//  Training runs in a Web Worker; this file handles display only.
// ============================================================

// --- Physics ---
const g_acc = 9.81;
const PX_PER_M = 160;
const PHYSICS_DT = 0.02;
const PHYSICS_SUB = 4;
const MAX_FORCE = 60;

// --- NN architecture ---
const NN_IN = 8;
const NN_HID1 = 24;
const NN_HID2 = 16;
const NN_OUT = 1;
const GENOME_LEN = (NN_IN + 1) * NN_HID1 + (NN_HID1 + 1) * NN_HID2 + (NN_HID2 + 1) * NN_OUT;

// --- GA state (received from worker) ---
let popSize = 300;
let gen = 0;
let evalIdx = 0;
let allBestGenome = null;
let allBestFit = -Infinity;
let bestFit = -Infinity;
let fitHistory = [];
let avgFitHistory = [];
let totalEvals = 0;
let trainElapsed = 0;
let topGenomes = [];

// --- Display ---
let dispSim = null;
let dispNet = null;
let dispTrail = [];
const TRAIL_LEN = 250;
let dispStep = 0;

// --- Secondary pendulums (top agents from population) ---
let secSims = [];   // array of { sim: CartDP, net: NNet }
let showAll = true;  // toggle with 'A' key

let paused = false;
let showTrails = true;
let showGrid = true;
let showNN = true;
let currentPreset = 0;
let time = 0;

// --- Training activity tracking ---
let lastWorkerMsgTime = 0;
let prevEvalIdx = 0;
let evalsPerSec = 0;
let lastEvalCountTime = 0;

// --- Worker ---
let worker = null;

// --- Audio ---
let audioCtx = null;
let sfxEnabled = true;
let audioStarted = false;
let masterGain = null;
let balanceOsc = null;
let balanceGain = null;
let balanceFilter = null;

// --- Recording ---
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;

// --- Presets ---
const PRESETS = [
  { name: "Easy Balance", M: 2.0, m1: 0.5, m2: 0.5, L1: 0.5, L2: 0.5,
    track: 3.0, fric: 0.1, th1: 0.15, th2: 0.15, steps: 800,
    col1: [255, 100, 60], col2: [60, 160, 255] },
  { name: "Standard", M: 2.0, m1: 0.8, m2: 0.5, L1: 0.5, L2: 0.5,
    track: 3.0, fric: 0.1, th1: 0.35, th2: 0.35, steps: 800,
    col1: [120, 255, 120], col2: [255, 80, 180] },
  { name: "Heavy Tip", M: 2.5, m1: 0.4, m2: 1.8, L1: 0.4, L2: 0.55,
    track: 3.5, fric: 0.1, th1: 0.2, th2: 0.2, steps: 900,
    col1: [255, 200, 50], col2: [50, 200, 255] },
  { name: "Swing Up", M: 3.0, m1: 0.5, m2: 0.5, L1: 0.4, L2: 0.4,
    track: 4.0, fric: 0.05, th1: Math.PI, th2: Math.PI, steps: 1500,
    col1: [255, 160, 40], col2: [100, 180, 255] },
  { name: "Long Arms", M: 3.0, m1: 0.35, m2: 0.3, L1: 0.7, L2: 0.6,
    track: 4.0, fric: 0.1, th1: 0.25, th2: 0.25, steps: 800,
    col1: [255, 70, 70], col2: [70, 255, 200] },
];

// ============================================================
//  Neural Network
// ============================================================
class NNet {
  constructor(genome) {
    this.g = genome || new Float64Array(GENOME_LEN);
    this.parse();
    this.lastH1 = new Float64Array(NN_HID1);
    this.lastH2 = new Float64Array(NN_HID2);
    this.lastO = [0];
    this.lastIn = new Float64Array(NN_IN);
  }
  parse() {
    let i = 0, g = this.g;
    this.w1 = []; this.b1 = [];
    for (let j = 0; j < NN_HID1; j++) {
      this.w1[j] = [];
      for (let k = 0; k < NN_IN; k++) this.w1[j][k] = g[i++];
      this.b1[j] = g[i++];
    }
    this.w2 = []; this.b2 = [];
    for (let j = 0; j < NN_HID2; j++) {
      this.w2[j] = [];
      for (let k = 0; k < NN_HID1; k++) this.w2[j][k] = g[i++];
      this.b2[j] = g[i++];
    }
    this.w3 = []; this.b3 = [];
    for (let j = 0; j < NN_OUT; j++) {
      this.w3[j] = [];
      for (let k = 0; k < NN_HID2; k++) this.w3[j][k] = g[i++];
      this.b3[j] = g[i++];
    }
  }
  forward(inp) {
    this.lastIn = inp;
    for (let j = 0; j < NN_HID1; j++) {
      let s = this.b1[j];
      for (let k = 0; k < NN_IN; k++) s += this.w1[j][k] * inp[k];
      this.lastH1[j] = Math.tanh(s);
    }
    for (let j = 0; j < NN_HID2; j++) {
      let s = this.b2[j];
      for (let k = 0; k < NN_HID1; k++) s += this.w2[j][k] * this.lastH1[k];
      this.lastH2[j] = Math.tanh(s);
    }
    for (let j = 0; j < NN_OUT; j++) {
      let s = this.b3[j];
      for (let k = 0; k < NN_HID2; k++) s += this.w3[j][k] * this.lastH2[k];
      this.lastO[j] = Math.tanh(s);
    }
    return this.lastO;
  }
}


// ============================================================
//  Cart + Double Pendulum Physics
//  theta = 0 is upright, theta = PI is hanging
//  3x3 mass matrix solved via Gaussian elimination + RK4
// ============================================================
class CartDP {
  constructor(p) {
    this.Mc = p.M; this.m1 = p.m1; this.m2 = p.m2;
    this.L1 = p.L1; this.L2 = p.L2;
    this.track = p.track; this.fric = p.fric;
    this.col1 = p.col1; this.col2 = p.col2;
    this.reset(p.th1, p.th2);
  }
  reset(t1, t2, pert = 0.05) {
    this.x = (Math.random() - 0.5) * 0.1;
    this.xd = 0;
    this.t1 = t1 + (Math.random() - 0.5) * pert * 2;
    this.t1d = 0;
    this.t2 = t2 + (Math.random() - 0.5) * pert * 2;
    this.t2d = 0;
    this.dead = false;
  }
  state() {
    return [
      this.x / this.track,
      this.xd * 0.15,
      Math.sin(this.t1), Math.cos(this.t1),
      this.t1d * 0.08,
      Math.sin(this.t2), Math.cos(this.t2),
      this.t2d * 0.08
    ];
  }
  derivs(st, F) {
    let [x, xd, t1, t1d, t2, t2d] = st;
    let { Mc, m1, m2, L1, L2, fric } = this;
    let s1 = Math.sin(t1), c1 = Math.cos(t1);
    let s2 = Math.sin(t2), c2 = Math.cos(t2);
    let s12 = Math.sin(t1 - t2), c12 = Math.cos(t1 - t2);

    let A = [
      [Mc + m1 + m2,          (m1 + m2) * L1 * c1,   m2 * L2 * c2],
      [(m1 + m2) * L1 * c1,   (m1 + m2) * L1 * L1,   m2 * L1 * L2 * c12],
      [m2 * L2 * c2,          m2 * L1 * L2 * c12,     m2 * L2 * L2]
    ];
    let b = [
      F - fric * xd + (m1 + m2) * L1 * t1d * t1d * s1 + m2 * L2 * t2d * t2d * s2,
      (m1 + m2) * g_acc * L1 * s1 - m2 * L1 * L2 * t2d * t2d * s12,
      m2 * g_acc * L2 * s2 + m2 * L1 * L2 * t1d * t1d * s12
    ];
    let acc = solve3(A, b);
    return [xd, acc[0], t1d, acc[1], t2d, acc[2]];
  }
  step(force) {
    force = Math.max(-MAX_FORCE, Math.min(MAX_FORCE, force));
    let sub_dt = PHYSICS_DT / PHYSICS_SUB;
    // Semi-implicit Euler with substeps — must match worker physics exactly
    for (let s = 0; s < PHYSICS_SUB; s++) {
      let acc = this.derivsFlat(force);
      this.xd  += acc[0] * sub_dt;
      this.t1d += acc[1] * sub_dt;
      this.t2d += acc[2] * sub_dt;
      this.x   += this.xd * sub_dt;
      this.t1  += this.t1d * sub_dt;
      this.t2  += this.t2d * sub_dt;
    }
    if (Math.abs(this.x) > this.track) { this.dead = true; this.xd = 0; this.x = Math.sign(this.x) * this.track; }
  }
  derivsFlat(F) {
    let { Mc, m1, m2, L1, L2, fric } = this;
    let s1 = Math.sin(this.t1), c1 = Math.cos(this.t1);
    let s2 = Math.sin(this.t2), c2 = Math.cos(this.t2);
    let s12 = Math.sin(this.t1 - this.t2), c12 = Math.cos(this.t1 - this.t2);

    let a00 = Mc + m1 + m2, a01 = (m1 + m2) * L1 * c1, a02 = m2 * L2 * c2;
    let a10 = a01, a11 = (m1 + m2) * L1 * L1, a12 = m2 * L1 * L2 * c12;
    let a20 = a02, a21 = a12, a22 = m2 * L2 * L2;

    let b0 = F - fric * this.xd + (m1 + m2) * L1 * this.t1d * this.t1d * s1 + m2 * L2 * this.t2d * this.t2d * s2;
    let b1 = (m1 + m2) * g_acc * L1 * s1 - m2 * L1 * L2 * this.t2d * this.t2d * s12;
    let b2 = m2 * g_acc * L2 * s2 + m2 * L1 * L2 * this.t1d * this.t1d * s12;

    let det = a00*(a11*a22-a12*a21) - a01*(a10*a22-a12*a20) + a02*(a10*a21-a11*a20);
    if (Math.abs(det) < 1e-12) return [0, 0, 0];
    let invDet = 1 / det;
    let xdd  = (b0*(a11*a22-a12*a21) - a01*(b1*a22-a12*b2) + a02*(b1*a21-a11*b2)) * invDet;
    let t1dd = (a00*(b1*a22-a12*b2) - b0*(a10*a22-a12*a20) + a02*(a10*b2-b1*a20)) * invDet;
    let t2dd = (a00*(a11*b2-b1*a21) - a01*(a10*b2-b1*a20) + b0*(a10*a21-a11*a20)) * invDet;
    return [xdd, t1dd, t2dd];
  }
  pixelPos() {
    let cx = this.x * PX_PER_M;
    let b1x = cx + this.L1 * PX_PER_M * Math.sin(this.t1);
    let b1y = -this.L1 * PX_PER_M * Math.cos(this.t1);
    let b2x = b1x + this.L2 * PX_PER_M * Math.sin(this.t2);
    let b2y = b1y - this.L2 * PX_PER_M * Math.cos(this.t2);
    return { cx, b1x, b1y, b2x, b2y };
  }
  reward() {
    return (1 + Math.cos(this.t1)) / 2 + (1 + Math.cos(this.t2)) / 2 - 0.005 * this.x * this.x;
  }
}

function solve3(A, b) {
  let a = [
    [A[0][0], A[0][1], A[0][2], b[0]],
    [A[1][0], A[1][1], A[1][2], b[1]],
    [A[2][0], A[2][1], A[2][2], b[2]]
  ];
  for (let c = 0; c < 3; c++) {
    let mx = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(a[r][c]) > Math.abs(a[mx][c])) mx = r;
    [a[c], a[mx]] = [a[mx], a[c]];
    if (Math.abs(a[c][c]) < 1e-12) return [0, 0, 0];
    for (let r = c + 1; r < 3; r++) {
      let f = a[r][c] / a[c][c];
      for (let j = c; j < 4; j++) a[r][j] -= f * a[c][j];
    }
  }
  let x = [0, 0, 0];
  for (let i = 2; i >= 0; i--) {
    x[i] = a[i][3];
    for (let j = i + 1; j < 3; j++) x[i] -= a[i][j] * x[j];
    x[i] /= a[i][i];
  }
  return x;
}

// ============================================================
//  Worker communication
// ============================================================
function initWorker() {
  if (worker) worker.terminate();
  worker = new Worker('worker.js');
  worker.onmessage = function(e) {
    let msg = e.data;
    lastWorkerMsgTime = millis();

    if (msg.type === "best") {
      allBestGenome = new Float64Array(msg.genome);
      allBestFit = msg.fit;
      gen = msg.gen;
      resetDisplay();
    }
    if (msg.type === "status") {
      let prevIdx = evalIdx;
      gen = msg.gen;
      evalIdx = msg.evalIdx;
      popSize = msg.popSize;
      bestFit = msg.bestFit;
      allBestFit = msg.allBestFit;
      fitHistory = msg.fitHistory;
      avgFitHistory = msg.avgFitHistory;
      totalEvals = msg.totalEvals || 0;
      trainElapsed = msg.elapsed || 0;

      // Track evals per second using totalEvals
      let now = millis();
      if (now - lastEvalCountTime > 1000) {
        evalsPerSec = (totalEvals - prevEvalIdx) / ((now - lastEvalCountTime) / 1000);
        prevEvalIdx = totalEvals;
        lastEvalCountTime = now;
      }

      // Update secondary pendulums from top genomes
      if (msg.topGenomes) {
        topGenomes = msg.topGenomes;
        rebuildSecondary();
      }
    }
  };
}

// ============================================================
//  Display simulation (runs best agent in real-time)
// ============================================================
function resetDisplay() {
  let p = PRESETS[currentPreset];
  dispSim = new CartDP(p);
  dispNet = new NNet(allBestGenome || new Float64Array(GENOME_LEN));
  dispTrail = [];
  dispStep = 0;
  rebuildSecondary();
}

function rebuildSecondary() {
  let p = PRESETS[currentPreset];
  secSims = [];
  for (let i = 0; i < topGenomes.length; i++) {
    let tg = topGenomes[i];
    let sim = new CartDP(p);
    // Sync initial state close to main sim
    if (dispSim) {
      sim.x = dispSim.x; sim.xd = dispSim.xd;
      sim.t1 = dispSim.t1; sim.t1d = dispSim.t1d;
      sim.t2 = dispSim.t2; sim.t2d = dispSim.t2d;
    }
    let net = new NNet(new Float64Array(tg.genome));
    secSims.push({ sim, net, step: 0 });
  }
}



function stepDisplay() {
  if (!dispSim) return;
  if (dispStep > PRESETS[currentPreset].steps) {
    resetDisplay();
    return;
  }
  let s = dispSim.state();
  let out = dispNet.forward(s);
  dispSim.step(out[0] * MAX_FORCE);
  let pos = dispSim.pixelPos();
  dispTrail.push({ x: pos.b2x, y: pos.b2y });
  if (dispTrail.length > TRAIL_LEN) dispTrail.shift();
  dispStep++;

  // Step secondary pendulums
  let maxSteps = PRESETS[currentPreset].steps;
  for (let i = 0; i < secSims.length; i++) {
    let sc = secSims[i];
    if (sc.step > maxSteps || sc.sim.dead) continue;
    let ss = sc.sim.state();
    let so = sc.net.forward(ss);
    sc.sim.step(so[0] * MAX_FORCE);
    sc.step++;
  }
}

// ============================================================
//  Drawing
// ============================================================
function drawPendulum() {
  if (!dispSim) return;
  let p = PRESETS[currentPreset];
  let pos = dispSim.pixelPos();
  let pivotY = height * 0.6;

  push();
  translate(width * 0.42, pivotY);

  // Track
  let trackPx = p.track * PX_PER_M;
  stroke(255, 30);
  strokeWeight(2);
  line(-trackPx, 0, trackPx, 0);
  // Track limits
  stroke(255, 50, 50, 60);
  strokeWeight(1);
  line(-trackPx, -15, -trackPx, 15);
  line(trackPx, -15, trackPx, 15);

  // Grid
  if (showGrid) {
    stroke(255, 12);
    strokeWeight(0.5);
    for (let gx = -trackPx; gx <= trackPx; gx += 50) line(gx, -300, gx, 50);
    for (let gy = -300; gy <= 50; gy += 50) line(-trackPx, gy, trackPx, gy);
  }

  // Secondary pendulums (ghost style)
  if (showAll && secSims.length > 0) {
    let ghostColors = [
      [255, 150, 50], [50, 255, 180], [200, 100, 255], [255, 255, 80], [100, 200, 255]
    ];
    for (let gi = 0; gi < secSims.length; gi++) {
      let sc = secSims[gi];
      let gp = sc.sim.pixelPos();
      let gc = ghostColors[gi % ghostColors.length];
      let ga = 40; // ghost alpha

      // Ghost cart
      noStroke();
      fill(gc[0], gc[1], gc[2], 15);
      rect(gp.cx - 20, -6, 40, 12, 3);

      // Ghost rod 1
      stroke(gc[0], gc[1], gc[2], ga);
      strokeWeight(1.5);
      line(gp.cx, 0, gp.b1x, gp.b1y);

      // Ghost rod 2
      stroke(gc[0], gc[1], gc[2], ga * 0.8);
      strokeWeight(1.2);
      line(gp.b1x, gp.b1y, gp.b2x, gp.b2y);

      // Ghost bobs
      noStroke();
      fill(gc[0], gc[1], gc[2], ga + 20);
      ellipse(gp.b1x, gp.b1y, 6, 6);
      ellipse(gp.b2x, gp.b2y, 6, 6);
    }
  }

  // Trail
  if (showTrails && dispTrail.length > 1) {
    noFill();
    for (let i = 1; i < dispTrail.length; i++) {
      let t = i / dispTrail.length;
      stroke(p.col2[0], p.col2[1], p.col2[2], t * 100);
      strokeWeight(t * 2.5);
      line(dispTrail[i - 1].x, dispTrail[i - 1].y, dispTrail[i].x, dispTrail[i].y);
    }
  }

  // Cart
  noStroke();
  fill(255, 255, 255, 30);
  rect(pos.cx - 25, -8, 50, 16, 4);
  fill(255, 255, 255, 80);
  rect(pos.cx - 22, -5, 44, 10, 3);

  // Rod 1
  stroke(255, 255, 255, 80);
  strokeWeight(2.5);
  line(pos.cx, 0, pos.b1x, pos.b1y);

  // Rod 2
  stroke(255, 255, 255, 60);
  strokeWeight(2);
  line(pos.b1x, pos.b1y, pos.b2x, pos.b2y);

  // Bob 1 glow
  noStroke();
  let r1 = map(p.m1, 0.2, 2, 7, 18);
  for (let i = 5; i >= 1; i--) {
    fill(p.col1[0], p.col1[1], p.col1[2], (6 - i) * 8);
    ellipse(pos.b1x, pos.b1y, r1 + i * 10, r1 + i * 10);
  }
  fill(p.col1[0], p.col1[1], p.col1[2], 200);
  ellipse(pos.b1x, pos.b1y, r1 * 2, r1 * 2);
  fill(255, 255, 255, 180);
  ellipse(pos.b1x, pos.b1y, r1 * 0.6, r1 * 0.6);

  // Bob 2 glow
  let r2 = map(p.m2, 0.2, 2, 7, 18);
  for (let i = 5; i >= 1; i--) {
    fill(p.col2[0], p.col2[1], p.col2[2], (6 - i) * 8);
    ellipse(pos.b2x, pos.b2y, r2 + i * 10, r2 + i * 10);
  }
  fill(p.col2[0], p.col2[1], p.col2[2], 200);
  ellipse(pos.b2x, pos.b2y, r2 * 2, r2 * 2);
  fill(255, 255, 255, 180);
  ellipse(pos.b2x, pos.b2y, r2 * 0.6, r2 * 0.6);

  // Pivot
  fill(255, 255, 255, 100);
  ellipse(pos.cx, 0, 6, 6);

  pop();
}

function drawNNVis() {
  if (!showNN || !dispNet) return;

  // Panel dimensions
  let panelW = 280;
  let panelH = 360;
  let px = width - panelW - 20;
  let py = 20;
  let pad = 12;

  // Panel background
  noStroke();
  fill(8, 10, 22, 210);
  rect(px, py, panelW, panelH, 8);
  stroke(255, 255, 255, 15);
  strokeWeight(1);
  noFill();
  rect(px, py, panelW, panelH, 8);

  // Title
  noStroke();
  fill(255, 255, 255, 50);
  textFont("monospace");
  textSize(9);
  textAlign(CENTER, TOP);
  text("NEURAL NETWORK  8-24-16-1", px + panelW / 2, py + 8);

  // Layout inside panel: 4 columns
  let innerX = px + pad;
  let innerY = py + 26;
  let usableW = panelW - pad * 2 - 16;
  let layerGap = usableW / 3;
  let nodeR = 4;

  let inLabels = ["x", "v", "s1", "c1", "w1", "s2", "c2", "w2"];
  let inY = [], h1Y = [], h2Y = [], outY = [];
  let inX = innerX + 12;
  let h1X = innerX + 12 + layerGap;
  let h2X = innerX + 12 + layerGap * 2;
  let outX = innerX + 12 + layerGap * 3;

  let h1Spacing = 13;
  let totalH = (NN_HID1 - 1) * h1Spacing;
  let inSpacing = totalH / (NN_IN - 1);
  let h2Spacing = totalH / (NN_HID2 - 1);
  let inStart = innerY;
  let h1Start = innerY;
  let h2Start = innerY;
  let outYpos = innerY + totalH / 2;

  for (let i = 0; i < NN_IN; i++) inY.push(inStart + i * inSpacing);
  for (let i = 0; i < NN_HID1; i++) h1Y.push(h1Start + i * h1Spacing);
  for (let i = 0; i < NN_HID2; i++) h2Y.push(h2Start + i * h2Spacing);
  outY.push(outYpos);

  // Connections: input -> hidden1
  for (let j = 0; j < NN_HID1; j++) {
    for (let i = 0; i < NN_IN; i++) {
      let w = dispNet.w1[j][i];
      let alpha = constrain(Math.abs(w) * 35, 2, 45);
      let sw = constrain(Math.abs(w) * 0.7, 0.12, 1.3);
      if (w > 0) stroke(80, 180, 255, alpha);
      else stroke(255, 80, 80, alpha);
      strokeWeight(sw);
      line(inX + nodeR, inY[i], h1X - nodeR, h1Y[j]);
    }
  }
  // Connections: hidden1 -> hidden2
  for (let j = 0; j < NN_HID2; j++) {
    for (let i = 0; i < NN_HID1; i++) {
      let w = dispNet.w2[j][i];
      let alpha = constrain(Math.abs(w) * 35, 2, 45);
      let sw = constrain(Math.abs(w) * 0.7, 0.12, 1.3);
      if (w > 0) stroke(80, 180, 255, alpha);
      else stroke(255, 80, 80, alpha);
      strokeWeight(sw);
      line(h1X + nodeR, h1Y[i], h2X - nodeR, h2Y[j]);
    }
  }
  // Connections: hidden2 -> output
  for (let j = 0; j < NN_HID2; j++) {
    let w = dispNet.w3[0][j];
    let alpha = constrain(Math.abs(w) * 50, 3, 70);
    let sw = constrain(Math.abs(w) * 1, 0.2, 2);
    if (w > 0) stroke(80, 180, 255, alpha);
    else stroke(255, 80, 80, alpha);
    strokeWeight(sw);
    line(h2X + nodeR, h2Y[j], outX - nodeR, outY[0]);
  }

  noStroke();
  textFont("monospace");
  textSize(7);
  textAlign(RIGHT, CENTER);

  // Input nodes
  for (let i = 0; i < NN_IN; i++) {
    let v = dispNet.lastIn ? dispNet.lastIn[i] : 0;
    let bright = constrain(map(Math.abs(v), 0, 1, 60, 220), 60, 220);
    fill(bright, bright, bright, 180);
    ellipse(inX, inY[i], nodeR * 2, nodeR * 2);
    fill(255, 255, 255, 50);
    text(inLabels[i], inX - nodeR - 1, inY[i]);
  }

  // Hidden1 nodes
  textAlign(CENTER, CENTER);
  for (let j = 0; j < NN_HID1; j++) {
    let v = dispNet.lastH1 ? dispNet.lastH1[j] : 0;
    let r = v > 0 ? 60 : 255;
    let g = v > 0 ? 200 : 80;
    let b = v > 0 ? 255 : 80;
    let alpha = constrain(Math.abs(v) * 200 + 40, 40, 220);
    fill(r, g, b, alpha);
    ellipse(h1X, h1Y[j], nodeR * 2, nodeR * 2);
  }

  // Hidden2 nodes
  for (let j = 0; j < NN_HID2; j++) {
    let v = dispNet.lastH2 ? dispNet.lastH2[j] : 0;
    let r = v > 0 ? 60 : 255;
    let g = v > 0 ? 200 : 80;
    let b = v > 0 ? 255 : 80;
    let alpha = constrain(Math.abs(v) * 200 + 40, 40, 220);
    fill(r, g, b, alpha);
    ellipse(h2X, h2Y[j], nodeR * 2, nodeR * 2);
  }

  // Output node
  let ov = dispNet.lastO ? dispNet.lastO[0] : 0;
  let oColor = ov > 0 ? [80, 255, 130] : [255, 130, 80];
  fill(oColor[0], oColor[1], oColor[2], 200);
  ellipse(outX, outY[0], nodeR * 3, nodeR * 3);
  fill(255, 255, 255, 100);
  textSize(6);
  text("F", outX, outY[0]);
}

function drawFitnessGraph() {
  let panelW = 280;
  let gw = panelW;
  let gh = 140;
  let gx = width - panelW - 20;
  let gy = 390;

  // Panel background
  noStroke();
  fill(8, 10, 22, 210);
  rect(gx, gy, gw, gh, 8);
  stroke(255, 255, 255, 15);
  strokeWeight(1);
  noFill();
  rect(gx, gy, gw, gh, 8);

  noStroke();
  fill(255, 255, 255, 50);
  textFont("monospace");
  textSize(9);
  textAlign(CENTER, TOP);
  text("FITNESS / GENERATION", gx + gw / 2, gy + 8);

  if (fitHistory.length < 2) {
    fill(255, 255, 255, 30);
    textSize(10);
    textAlign(CENTER, CENTER);
    text("Waiting for data...", gx + gw / 2, gy + gh / 2 + 8);

    // Show a loading animation
    let isTraining = (millis() - lastWorkerMsgTime) < 500;
    if (isTraining) {
      let dotCount = Math.floor(frameCount / 20) % 4;
      fill(80, 255, 120, 60);
      text("Training" + ".".repeat(dotCount), gx + gw / 2, gy + gh / 2 + 24);
    }
    return;
  }

  let pad = 14;
  let chartX = gx + pad + 20; // Extra room for y-axis labels
  let chartY = gy + 24;
  let chartW = gw - pad * 2 - 20;
  let chartH = gh - 50;

  let data = fitHistory.slice(-80);
  let avg = avgFitHistory.slice(-80);
  let allData = data.concat(avg);
  let minR = Math.min(...allData);
  let maxR = Math.max(...allData);
  if (maxR - minR < 1) maxR = minR + 1;

  // Y-axis labels
  textSize(7);
  textAlign(RIGHT, CENTER);
  fill(255, 255, 255, 40);
  text(maxR.toFixed(0), chartX - 4, chartY);
  text(((maxR + minR) / 2).toFixed(0), chartX - 4, chartY + chartH / 2);
  text(minR.toFixed(0), chartX - 4, chartY + chartH);

  // Horizontal grid lines
  stroke(255, 255, 255, 8);
  strokeWeight(0.5);
  line(chartX, chartY, chartX + chartW, chartY);
  line(chartX, chartY + chartH / 2, chartX + chartW, chartY + chartH / 2);
  line(chartX, chartY + chartH, chartX + chartW, chartY + chartH);

  // Average line (filled area)
  noStroke();
  fill(255, 255, 255, 8);
  beginShape();
  for (let i = 0; i < avg.length; i++) {
    let px = chartX + (i / (avg.length - 1)) * chartW;
    let py = chartY + chartH - ((avg[i] - minR) / (maxR - minR)) * chartH;
    vertex(px, py);
  }
  vertex(chartX + chartW, chartY + chartH);
  vertex(chartX, chartY + chartH);
  endShape(CLOSE);

  // Average line
  noFill();
  stroke(255, 255, 255, 50);
  strokeWeight(1);
  beginShape();
  for (let i = 0; i < avg.length; i++) {
    let px = chartX + (i / (avg.length - 1)) * chartW;
    let py = chartY + chartH - ((avg[i] - minR) / (maxR - minR)) * chartH;
    vertex(px, py);
  }
  endShape();

  // Best line
  stroke(120, 255, 120, 180);
  strokeWeight(1.5);
  beginShape();
  for (let i = 0; i < data.length; i++) {
    let px = chartX + (i / (data.length - 1)) * chartW;
    let py = chartY + chartH - ((data[i] - minR) / (maxR - minR)) * chartH;
    vertex(px, py);
  }
  endShape();

  // Mark the current best point
  if (data.length > 0) {
    let lastX = chartX + chartW;
    let lastY = chartY + chartH - ((data[data.length - 1] - minR) / (maxR - minR)) * chartH;
    noStroke();
    fill(120, 255, 120, 200);
    ellipse(lastX, lastY, 5, 5);
  }

  // X-axis generation labels
  noStroke();
  fill(255, 255, 255, 35);
  textSize(7);
  textAlign(CENTER, TOP);
  let startGen = Math.max(0, gen - data.length);
  text("Gen " + startGen, chartX, chartY + chartH + 4);
  text("Gen " + gen, chartX + chartW, chartY + chartH + 4);

  // Legend
  textSize(8);
  textAlign(LEFT, BOTTOM);
  fill(120, 255, 120, 150);
  ellipse(gx + pad + 2, gy + gh - 6, 4, 4);
  text("BEST: " + data[data.length - 1].toFixed(0), gx + pad + 8, gy + gh - 2);
  fill(255, 255, 255, 60);
  let avgX = gx + gw / 2 + 10;
  ellipse(avgX - 6, gy + gh - 6, 4, 4);
  text("AVG: " + avg[avg.length - 1].toFixed(0), avgX, gy + gh - 2);

  // Improvement indicator
  if (data.length >= 2) {
    let delta = data[data.length - 1] - data[data.length - 2];
    if (Math.abs(delta) > 0.1) {
      textAlign(RIGHT, BOTTOM);
      if (delta > 0) {
        fill(80, 255, 120, 120);
        text("+" + delta.toFixed(1), gx + gw - pad, gy + gh - 2);
      } else {
        fill(255, 80, 80, 80);
        text(delta.toFixed(1), gx + gw - pad, gy + gh - 2);
      }
    }
  }
}

function drawHUD() {
  let px = 20;
  let py = 20;
  let panelW = 280;

  // --- Title bar ---
  noStroke();
  fill(8, 10, 22, 210);
  rect(px, py, panelW, 42, 8);
  stroke(255, 255, 255, 15);
  strokeWeight(1);
  noFill();
  rect(px, py, panelW, 42, 8);

  noStroke();
  fill(255, 255, 255, 220);
  textFont("monospace");
  textSize(16);
  textAlign(LEFT, CENTER);
  text("PENDULUM BALANCE", px + 14, py + 21);

  // --- Training status indicator ---
  let isTraining = (millis() - lastWorkerMsgTime) < 500;
  let trainY = py + 52;
  let trainH = 32;

  noStroke();
  fill(8, 10, 22, 210);
  rect(px, trainY, panelW, trainH, 8);
  stroke(255, 255, 255, 15);
  strokeWeight(1);
  noFill();
  rect(px, trainY, panelW, trainH, 8);

  noStroke();
  textSize(11);
  textAlign(LEFT, CENTER);

  if (isTraining) {
    // Pulsing green dot
    let pulse = 150 + Math.sin(frameCount * 0.15) * 105;
    fill(80, 255, 120, pulse);
    ellipse(px + 18, trainY + trainH / 2, 8, 8);
    // Glow ring
    noFill();
    stroke(80, 255, 120, pulse * 0.3);
    strokeWeight(2);
    ellipse(px + 18, trainY + trainH / 2, 14, 14);
    noStroke();
    fill(80, 255, 120, 220);
    text("TRAINING", px + 30, trainY + trainH / 2);
  } else {
    fill(255, 60, 60, 180);
    ellipse(px + 18, trainY + trainH / 2, 8, 8);
    fill(255, 60, 60, 180);
    text("IDLE", px + 30, trainY + trainH / 2);
  }

  // Generation progress bar
  let barX = px + 110;
  let barW = panelW - 120;
  let barH = 6;
  let barY = trainY + trainH / 2 - barH / 2;
  let prog = popSize > 0 ? evalIdx / popSize : 0;

  fill(255, 255, 255, 15);
  rect(barX, barY, barW, barH, 3);
  if (isTraining) {
    fill(80, 255, 120, 120);
  } else {
    fill(255, 255, 255, 40);
  }
  rect(barX, barY, barW * prog, barH, 3);

  fill(255, 255, 255, 80);
  textSize(8);
  textAlign(RIGHT, CENTER);
  text(`${evalIdx}/${popSize}`, barX + barW, barY + barH + 8);

  // --- Stats panel ---
  let sp = trainY + trainH + 10;
  let statLines = 10;
  let spH = 17 * statLines + 20;

  noStroke();
  fill(8, 10, 22, 210);
  rect(px, sp, panelW, spH, 8);
  stroke(255, 255, 255, 15);
  strokeWeight(1);
  noFill();
  rect(px, sp, panelW, spH, 8);

  let tx = px + 14;
  let ty = sp + 14;
  let lh = 17;
  noStroke();
  textFont("monospace");
  textSize(11);
  textAlign(LEFT, TOP);

  fill(255, 255, 255, 60);
  text("PRESET", tx, ty);
  fill(255, 255, 255, 180);
  text(PRESETS[currentPreset].name, tx + 90, ty);
  ty += lh;

  fill(255, 255, 255, 60);
  text("GENERATION", tx, ty);
  fill(255, 255, 255, 180);
  text(gen, tx + 90, ty);
  ty += lh;

  fill(255, 255, 255, 60);
  text("POP SIZE", tx, ty);
  fill(255, 255, 255, 140);
  text(popSize, tx + 90, ty);
  ty += lh;

  fill(255, 255, 255, 60);
  text("BEST GEN", tx, ty);
  fill(120, 255, 120, 180);
  text(bestFit > -Infinity ? bestFit.toFixed(1) : "---", tx + 90, ty);
  ty += lh;

  fill(255, 255, 255, 60);
  text("BEST ALL", tx, ty);
  fill(255, 200, 50, 200);
  text(allBestFit > -Infinity ? allBestFit.toFixed(1) : "---", tx + 90, ty);
  ty += lh;

  fill(255, 255, 255, 60);
  text("TOTAL EVALS", tx, ty);
  fill(255, 255, 255, 140);
  text(totalEvals.toLocaleString(), tx + 90, ty);
  ty += lh;

  fill(255, 255, 255, 60);
  text("EVALS/SEC", tx, ty);
  fill(isTraining ? [80, 200, 255][0] : 255, isTraining ? [80, 200, 255][1] : 255, isTraining ? [80, 200, 255][2] : 255, 160);
  text(evalsPerSec.toFixed(1), tx + 90, ty);
  ty += lh;

  fill(255, 255, 255, 60);
  text("ELAPSED", tx, ty);
  fill(255, 255, 255, 140);
  let secs = Math.floor(trainElapsed / 1000);
  let mins = Math.floor(secs / 60);
  let hrs = Math.floor(mins / 60);
  let timeStr = hrs > 0
    ? `${hrs}h ${mins % 60}m ${secs % 60}s`
    : mins > 0
      ? `${mins}m ${secs % 60}s`
      : `${secs}s`;
  text(timeStr, tx + 90, ty);
  ty += lh;

  if (dispSim) {
    let bal = (1 + Math.cos(dispSim.t1)) / 2 + (1 + Math.cos(dispSim.t2)) / 2;
    fill(255, 255, 255, 60);
    text("BALANCE", tx, ty);
    let bt = bal / 2;
    fill(lerp(255, 120, bt), lerp(80, 255, bt), lerp(80, 120, bt), 180);
    text(bal.toFixed(2) + " / 2.00", tx + 90, ty);
    ty += lh;

    fill(255, 255, 255, 60);
    text("CART POS", tx, ty);
    fill(255, 255, 255, 140);
    text(dispSim.x.toFixed(2) + "m", tx + 90, ty);
    ty += lh;
  } else {
    ty += lh * 2;
  }

  // --- Key hints (bottom-left) ---
  drawKeyHints();

  // --- Recording indicator ---
  if (isRecording) {
    let recX = width / 2;
    let recY = 20;
    noStroke();
    fill(8, 10, 22, 210);
    rect(recX - 36, recY, 72, 26, 6);
    let pulse = 200 + Math.sin(frameCount * 0.1) * 55;
    fill(255, 40, 40, pulse);
    ellipse(recX - 16, recY + 13, 10, 10);
    fill(255, 255, 255, 180);
    textFont("monospace");
    textSize(11);
    textAlign(LEFT, CENTER);
    text("REC", recX - 4, recY + 13);
  }

  // --- Paused overlay ---
  if (paused) {
    fill(5, 5, 15, 120);
    noStroke();
    rect(0, 0, width, height);
    textAlign(CENTER, CENTER);
    textFont("monospace");
    textSize(32);
    fill(255, 255, 255, 200);
    text("PAUSED", width / 2, height / 2);
    textSize(12);
    fill(255, 255, 255, 80);
    text("Press SPACE to resume", width / 2, height / 2 + 30);
  }
}

// ============================================================
//  Audio
// ============================================================
function initAudio() {
  if (audioStarted) return;
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  audioStarted = true;

  masterGain = audioCtx.createGain();
  masterGain.gain.value = 0.35;
  masterGain.connect(audioCtx.destination);

  // Ambient drone
  let droneG = audioCtx.createGain();
  droneG.gain.value = 0.05;
  let droneF = audioCtx.createBiquadFilter();
  droneF.type = "lowpass"; droneF.frequency.value = 140; droneF.Q.value = 1;
  droneF.connect(droneG); droneG.connect(masterGain);
  [28, 28.1, 42, 56].forEach(f => {
    let o = audioCtx.createOscillator(); o.type = "sawtooth"; o.frequency.value = f;
    o.connect(droneF); o.start();
  });
  let lfo = audioCtx.createOscillator(); lfo.type = "sine"; lfo.frequency.value = 0.06;
  let lfoG = audioCtx.createGain(); lfoG.gain.value = 40;
  lfo.connect(lfoG); lfoG.connect(droneF.frequency); lfo.start();

  // Balance tone
  balanceOsc = audioCtx.createOscillator();
  balanceOsc.type = "sine"; balanceOsc.frequency.value = 120;
  balanceFilter = audioCtx.createBiquadFilter();
  balanceFilter.type = "lowpass"; balanceFilter.frequency.value = 200; balanceFilter.Q.value = 5;
  balanceGain = audioCtx.createGain(); balanceGain.gain.value = 0;
  balanceOsc.connect(balanceFilter); balanceFilter.connect(balanceGain); balanceGain.connect(masterGain);
  balanceOsc.start();
}

function updateAudio() {
  if (!audioStarted || !sfxEnabled || paused || !dispSim) return;
  let now = audioCtx.currentTime;
  let bal = (1 + Math.cos(dispSim.t1)) / 2 + (1 + Math.cos(dispSim.t2)) / 2; // 0-2
  let t = bal / 2; // 0-1
  balanceGain.gain.linearRampToValueAtTime(t * 0.08, now + 0.05);
  balanceOsc.frequency.linearRampToValueAtTime(100 + t * 400, now + 0.05);
  balanceFilter.frequency.linearRampToValueAtTime(150 + t * 800, now + 0.05);
}

function playImpact() {
  if (!audioStarted || !sfxEnabled) return;
  let now = audioCtx.currentTime;
  let o = audioCtx.createOscillator(); o.type = "sine";
  o.frequency.setValueAtTime(80, now); o.frequency.exponentialRampToValueAtTime(25, now + 0.4);
  let g = audioCtx.createGain();
  g.gain.setValueAtTime(0.3, now); g.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
  o.connect(g); g.connect(masterGain); o.start(now); o.stop(now + 0.5);

  let bufSz = audioCtx.sampleRate * 0.15;
  let buf = audioCtx.createBuffer(1, bufSz, audioCtx.sampleRate);
  let d = buf.getChannelData(0);
  for (let i = 0; i < bufSz; i++) d[i] = (Math.random() * 2 - 1) * 0.5;
  let ns = audioCtx.createBufferSource(); ns.buffer = buf;
  let nf = audioCtx.createBiquadFilter(); nf.type = "lowpass"; nf.frequency.value = 200;
  let ng = audioCtx.createGain();
  ng.gain.setValueAtTime(0.15, now); ng.gain.exponentialRampToValueAtTime(0.001, now + 0.2);
  ns.connect(nf); nf.connect(ng); ng.connect(masterGain); ns.start(now); ns.stop(now + 0.2);
}

// ============================================================
//  Key Hints
// ============================================================
function drawKeyHints() {
  let hints = [
    ["SPACE", "Pause / Resume"],
    ["R", "Reset"],
    ["1-5", "Switch Preset"],
    ["T", "Toggle Trail" + (showTrails ? "" : "  [off]")],
    ["G", "Toggle Grid" + (showGrid ? "" : "  [off]")],
    ["N", "Toggle NN" + (showNN ? "" : "  [off]")],
    ["A", "Toggle All Agents" + (showAll ? "" : "  [off]")],
    ["M", "Toggle Audio" + (sfxEnabled ? "" : "  [off]")],
    ["V", "Record"],
  ];

  let lh = 16;
  let panelH = hints.length * lh + 16;
  let panelW = 200;
  let px = 20;
  let py = height - panelH - 20;

  noStroke();
  fill(8, 10, 22, 180);
  rect(px, py, panelW, panelH, 8);
  stroke(255, 255, 255, 10);
  strokeWeight(1);
  noFill();
  rect(px, py, panelW, panelH, 8);

  textFont("monospace");
  textSize(10);
  noStroke();
  let ty = py + 10;

  for (let h of hints) {
    fill(255, 255, 255, 90);
    textAlign(RIGHT, TOP);
    text(h[0], px + 50, ty);
    fill(255, 255, 255, 45);
    textAlign(LEFT, TOP);
    text(h[1], px + 58, ty);
    ty += lh;
  }
}

// ============================================================
//  Recording
// ============================================================
function startRecording() {
  let stream = document.querySelector('canvas').captureStream(60);
  mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 8000000 });
  recordedChunks = [];
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    let blob = new Blob(recordedChunks, { type: 'video/webm' });
    let url = URL.createObjectURL(blob);
    let a = document.createElement('a');
    a.href = url; a.download = `PendulumBalance_${PRESETS[currentPreset].name.replace(/\s+/g, '_')}_${Date.now()}.webm`;
    a.click(); URL.revokeObjectURL(url);
  };
  mediaRecorder.start(); isRecording = true;
}
function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
  isRecording = false;
}

// ============================================================
//  Setup & Draw
// ============================================================
function setup() {
  createCanvas(windowWidth, windowHeight);
  textFont("monospace");
  initWorker();
  loadPreset(currentPreset);
}

function loadPreset(idx) {
  currentPreset = idx;
  gen = 0; evalIdx = 0;
  bestFit = -Infinity;
  allBestFit = -Infinity;
  allBestGenome = null;
  fitHistory = []; avgFitHistory = [];
  totalEvals = 0;
  trainElapsed = 0;
  evalsPerSec = 0;
  lastEvalCountTime = millis();
  topGenomes = [];
  resetDisplay();
  worker.postMessage({ type: "start", preset: idx, popSize: popSize });
  playImpact();
}

function draw() {
  background(5, 5, 15);

  if (!paused) {
    stepDisplay();
    updateAudio();
    time += 1 / 60;
  }

  drawPendulum();
  drawNNVis();
  drawFitnessGraph();
  drawHUD();
}

// ============================================================
//  Input
// ============================================================
function keyPressed() {
  if (!audioStarted) initAudio();

  if (key === " ") paused = !paused;
  if (key === "r" || key === "R") loadPreset(currentPreset);
  if (key === "t" || key === "T") showTrails = !showTrails;
  if (key === "g" || key === "G") showGrid = !showGrid;
  if (key === "n" || key === "N") showNN = !showNN;
  if (key === "a" || key === "A") showAll = !showAll;
  if (key === "m" || key === "M") {
    sfxEnabled = !sfxEnabled;
    if (masterGain) masterGain.gain.value = sfxEnabled ? 0.35 : 0;
  }
  if (key === "v" || key === "V") {
    if (isRecording) stopRecording(); else startRecording();
  }
  if (key >= "1" && key <= "5") loadPreset(int(key) - 1);
}

function mousePressed() { if (!audioStarted) initAudio(); }
function windowResized() { resizeCanvas(windowWidth, windowHeight); }
