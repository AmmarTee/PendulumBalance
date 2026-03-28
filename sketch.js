// ============================================================
//  PENDULUM BALANCE – p5.js neuroevolution simulation
//  A neural network learns to balance an inverted double
//  pendulum on a cart using a genetic algorithm
// ============================================================

// --- Physics ---
const g_acc = 9.81;
const PX_PER_M = 160;
const PHYSICS_DT = 0.02;
const PHYSICS_SUB = 4;
const MAX_FORCE = 60;

// --- NN architecture ---
const NN_IN = 8;   // x, xd, sin1, cos1, w1, sin2, cos2, w2
const NN_HID = 12;
const NN_OUT = 1;
const GENOME_LEN = (NN_IN + 1) * NN_HID + (NN_HID + 1) * NN_OUT; // 121

// --- GA ---
const POP_SIZE = 100;
const ELITE = 10;
const MUT_RATE = 0.15;
const MUT_STD = 0.35;

// --- State ---
let pop = [];
let gen = 0;
let evalIdx = 0;
let bestGenome = null;
let bestFit = -Infinity;
let allBestGenome = null;
let allBestFit = -Infinity;
let fitHistory = [];
let avgFitHistory = [];
let evalsPerFrame = 10;

let dispSim = null;
let dispNet = null;
let dispTrail = [];
const TRAIL_LEN = 250;
let dispStep = 0;

let paused = false;
let speedMultiplier = 1;
let showTrails = true;
let showGrid = true;
let showNN = true;
let currentPreset = 0;
let time = 0;

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
    this.g = genome || NNet.rand();
    this.parse();
    this.lastH = new Float64Array(NN_HID);
    this.lastO = [0];
    this.lastIn = new Float64Array(NN_IN);
  }
  parse() {
    let i = 0, g = this.g;
    this.w1 = []; this.b1 = [];
    for (let j = 0; j < NN_HID; j++) {
      this.w1[j] = [];
      for (let k = 0; k < NN_IN; k++) this.w1[j][k] = g[i++];
      this.b1[j] = g[i++];
    }
    this.w2 = []; this.b2 = [];
    for (let j = 0; j < NN_OUT; j++) {
      this.w2[j] = [];
      for (let k = 0; k < NN_HID; k++) this.w2[j][k] = g[i++];
      this.b2[j] = g[i++];
    }
  }
  forward(inp) {
    this.lastIn = inp;
    for (let j = 0; j < NN_HID; j++) {
      let s = this.b1[j];
      for (let k = 0; k < NN_IN; k++) s += this.w1[j][k] * inp[k];
      this.lastH[j] = Math.tanh(s);
    }
    for (let j = 0; j < NN_OUT; j++) {
      let s = this.b2[j];
      for (let k = 0; k < NN_HID; k++) s += this.w2[j][k] * this.lastH[k];
      this.lastO[j] = Math.tanh(s);
    }
    return this.lastO;
  }
  static rand() {
    let g = new Float64Array(GENOME_LEN);
    for (let i = 0; i < GENOME_LEN; i++) g[i] = (Math.random() - 0.5) * 2;
    return g;
  }
  static cross(a, b) {
    let c = new Float64Array(GENOME_LEN);
    let pt = Math.floor(Math.random() * GENOME_LEN);
    for (let i = 0; i < GENOME_LEN; i++) c[i] = i < pt ? a[i] : b[i];
    return c;
  }
  static mutate(g) {
    let m = new Float64Array(g);
    for (let i = 0; i < GENOME_LEN; i++) {
      if (Math.random() < MUT_RATE) m[i] += gaussRand() * MUT_STD;
    }
    return m;
  }
}

function gaussRand() {
  return Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
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
      -(m1 + m2) * g_acc * L1 * s1 - m2 * L1 * L2 * t2d * t2d * s12,
      -m2 * g_acc * L2 * s2 + m2 * L1 * L2 * t1d * t1d * s12
    ];
    let acc = solve3(A, b);
    return [xd, acc[0], t1d, acc[1], t2d, acc[2]];
  }
  step(force) {
    force = Math.max(-MAX_FORCE, Math.min(MAX_FORCE, force));
    let dt = PHYSICS_DT / PHYSICS_SUB;
    for (let s = 0; s < PHYSICS_SUB; s++) {
      let st = [this.x, this.xd, this.t1, this.t1d, this.t2, this.t2d];
      let k1 = this.derivs(st, force);
      let k2 = this.derivs(st.map((v, i) => v + 0.5 * dt * k1[i]), force);
      let k3 = this.derivs(st.map((v, i) => v + 0.5 * dt * k2[i]), force);
      let k4 = this.derivs(st.map((v, i) => v + dt * k3[i]), force);
      this.x   += (dt / 6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
      this.xd  += (dt / 6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
      this.t1  += (dt / 6) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
      this.t1d += (dt / 6) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]);
      this.t2  += (dt / 6) * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4]);
      this.t2d += (dt / 6) * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5]);
    }
    if (Math.abs(this.x) > this.track) { this.dead = true; this.xd = 0; this.x = Math.sign(this.x) * this.track; }
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
//  Genetic Algorithm
// ============================================================
function initPop() {
  pop = [];
  for (let i = 0; i < POP_SIZE; i++) pop.push({ g: NNet.rand(), fit: 0 });
  gen = 0; evalIdx = 0;
  bestFit = -Infinity; bestGenome = null;
  allBestFit = -Infinity; allBestGenome = null;
  fitHistory = []; avgFitHistory = [];
}

function evalAgent(genome) {
  let p = PRESETS[currentPreset];
  let sim = new CartDP(p);
  let net = new NNet(genome);
  let totalR = 0;
  for (let i = 0; i < p.steps; i++) {
    if (sim.dead) break;
    let s = sim.state();
    let out = net.forward(s);
    sim.step(out[0] * MAX_FORCE);
    totalR += sim.reward();
  }
  return totalR;
}

function trainBatch(n) {
  for (let i = 0; i < n; i++) {
    if (evalIdx >= POP_SIZE) {
      nextGen();
      evalIdx = 0;
    }
    pop[evalIdx].fit = evalAgent(pop[evalIdx].g);
    evalIdx++;
  }
}

function nextGen() {
  pop.sort((a, b) => b.fit - a.fit);
  let topFit = pop[0].fit;
  let avgFit = pop.reduce((s, a) => s + a.fit, 0) / POP_SIZE;
  fitHistory.push(topFit);
  avgFitHistory.push(avgFit);

  if (topFit > bestFit) { bestFit = topFit; bestGenome = new Float64Array(pop[0].g); }
  if (topFit > allBestFit) { allBestFit = topFit; allBestGenome = new Float64Array(pop[0].g); resetDisplay(); }

  // Breed next generation
  let newPop = [];
  // Elitism
  for (let i = 0; i < ELITE; i++) newPop.push({ g: new Float64Array(pop[i].g), fit: 0 });
  // Breed rest
  while (newPop.length < POP_SIZE) {
    let a = tournament(3);
    let b = tournament(3);
    let child = NNet.cross(pop[a].g, pop[b].g);
    child = NNet.mutate(child);
    newPop.push({ g: child, fit: 0 });
  }
  pop = newPop;
  gen++;
  bestFit = -Infinity;
}

function tournament(k) {
  let best = Math.floor(Math.random() * POP_SIZE);
  for (let i = 1; i < k; i++) {
    let c = Math.floor(Math.random() * POP_SIZE);
    if (pop[c].fit > pop[best].fit) best = c;
  }
  return best;
}

// ============================================================
//  Display simulation (runs best agent in real-time)
// ============================================================
function resetDisplay() {
  let p = PRESETS[currentPreset];
  dispSim = new CartDP(p);
  dispNet = new NNet(allBestGenome || NNet.rand());
  dispTrail = [];
  dispStep = 0;
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
  translate(width * 0.38, pivotY);

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
  let nx = width - 230;
  let ny = 80;
  let layerGap = 85;
  let nodeR = 8;

  let inLabels = ["x", "v", "s1", "c1", "w1", "s2", "c2", "w2"];
  let inY = [], hidY = [], outY = [];
  let inX = nx, hidX = nx + layerGap, outX = nx + layerGap * 2;

  let inSpacing = 22;
  let hidSpacing = 24;
  let inStart = ny;
  let hidStart = ny + (NN_IN * inSpacing - NN_HID * hidSpacing) / 2;
  let outYpos = ny + (NN_IN * inSpacing) / 2;

  for (let i = 0; i < NN_IN; i++) inY.push(inStart + i * inSpacing);
  for (let i = 0; i < NN_HID; i++) hidY.push(hidStart + i * hidSpacing);
  outY.push(outYpos);

  // Connections: input -> hidden
  for (let j = 0; j < NN_HID; j++) {
    for (let i = 0; i < NN_IN; i++) {
      let w = dispNet.w1[j][i];
      let alpha = constrain(Math.abs(w) * 80, 5, 120);
      let sw = constrain(Math.abs(w) * 1.5, 0.3, 2.5);
      if (w > 0) stroke(80, 180, 255, alpha);
      else stroke(255, 80, 80, alpha);
      strokeWeight(sw);
      line(inX + nodeR, inY[i], hidX - nodeR, hidY[j]);
    }
  }
  // Connections: hidden -> output
  for (let j = 0; j < NN_HID; j++) {
    let w = dispNet.w2[0][j];
    let alpha = constrain(Math.abs(w) * 80, 5, 120);
    let sw = constrain(Math.abs(w) * 1.5, 0.3, 2.5);
    if (w > 0) stroke(80, 180, 255, alpha);
    else stroke(255, 80, 80, alpha);
    strokeWeight(sw);
    line(hidX + nodeR, hidY[j], outX - nodeR, outY[0]);
  }

  noStroke();
  textFont("monospace");
  textSize(9);
  textAlign(RIGHT, CENTER);

  // Input nodes
  for (let i = 0; i < NN_IN; i++) {
    let v = dispNet.lastIn ? dispNet.lastIn[i] : 0;
    let bright = constrain(map(Math.abs(v), 0, 1, 60, 220), 60, 220);
    fill(bright, bright, bright, 180);
    ellipse(inX, inY[i], nodeR * 2, nodeR * 2);
    fill(255, 255, 255, 70);
    text(inLabels[i], inX - nodeR - 3, inY[i]);
  }

  // Hidden nodes
  textAlign(CENTER, CENTER);
  for (let j = 0; j < NN_HID; j++) {
    let v = dispNet.lastH ? dispNet.lastH[j] : 0;
    let r = v > 0 ? 60 : 255;
    let g = v > 0 ? 200 : 80;
    let b = v > 0 ? 255 : 80;
    let alpha = constrain(Math.abs(v) * 200 + 40, 40, 220);
    fill(r, g, b, alpha);
    ellipse(hidX, hidY[j], nodeR * 2, nodeR * 2);
  }

  // Output node
  let ov = dispNet.lastO ? dispNet.lastO[0] : 0;
  let oColor = ov > 0 ? [80, 255, 130] : [255, 130, 80];
  fill(oColor[0], oColor[1], oColor[2], 200);
  ellipse(outX, outY[0], nodeR * 2.5, nodeR * 2.5);
  fill(255, 255, 255, 100);
  textSize(8);
  text("F", outX, outY[0]);

  // Label
  fill(255, 255, 255, 50);
  textSize(10);
  textAlign(CENTER, TOP);
  text("NEURAL NETWORK", nx + layerGap, ny - 20);
}

function drawFitnessGraph() {
  let gw = 200, gh = 70;
  let gx = width - gw - 30;
  let gy = height - gh - 30;

  fill(10, 12, 25, 200);
  stroke(255, 25);
  strokeWeight(1);
  rect(gx, gy, gw, gh, 4);

  if (fitHistory.length < 2) return;

  let data = fitHistory.slice(-80);
  let avg = avgFitHistory.slice(-80);
  let allData = data.concat(avg);
  let minR = Math.min(...allData);
  let maxR = Math.max(...allData);
  if (maxR - minR < 1) maxR = minR + 1;

  // Average line
  noFill();
  stroke(255, 255, 255, 40);
  strokeWeight(1);
  beginShape();
  for (let i = 0; i < avg.length; i++) {
    let px = gx + 10 + (i / (avg.length - 1)) * (gw - 20);
    let py = gy + gh - 10 - ((avg[i] - minR) / (maxR - minR)) * (gh - 20);
    vertex(px, py);
  }
  endShape();

  // Best line
  stroke(120, 255, 120, 150);
  strokeWeight(1.5);
  beginShape();
  for (let i = 0; i < data.length; i++) {
    let px = gx + 10 + (i / (data.length - 1)) * (gw - 20);
    let py = gy + gh - 10 - ((data[i] - minR) / (maxR - minR)) * (gh - 20);
    vertex(px, py);
  }
  endShape();

  noStroke();
  fill(255, 255, 255, 60);
  textFont("monospace");
  textSize(9);
  textAlign(LEFT, TOP);
  text("Fitness / Gen", gx + 6, gy + 4);
}

function drawHUD() {
  fill(255, 255, 255, 200);
  noStroke();
  textFont("monospace");
  textSize(22);
  textAlign(LEFT, TOP);
  text("PENDULUM BALANCE", 30, 25);

  textSize(14);
  fill(255, 255, 255, 120);
  text(`Preset: ${PRESETS[currentPreset].name}`, 30, 55);

  textSize(13);
  fill(255, 255, 255, 100);
  text(`Generation: ${gen}`, 30, 78);

  let yOff = 105;
  textSize(12);

  fill(255, 255, 255, 120);
  text(`Population: ${POP_SIZE}  |  Eval: ${evalIdx}/${POP_SIZE}`, 30, yOff); yOff += 20;
  fill(120, 255, 120, 150);
  text(`Best (gen): ${bestFit > -Infinity ? bestFit.toFixed(1) : "---"}`, 30, yOff); yOff += 18;
  fill(255, 200, 50, 150);
  text(`Best (all): ${allBestFit > -Infinity ? allBestFit.toFixed(1) : "---"}`, 30, yOff); yOff += 18;

  if (dispSim) {
    let bal = ((1 + Math.cos(dispSim.t1)) / 2 + (1 + Math.cos(dispSim.t2)) / 2).toFixed(2);
    fill(255, 255, 255, 100);
    text(`Balance: ${bal} / 2.00`, 30, yOff); yOff += 18;
    text(`Cart: ${dispSim.x.toFixed(2)}m`, 30, yOff); yOff += 18;
  }

  fill(255, 255, 255, 80);
  textSize(11);
  text(`Genome: ${GENOME_LEN} params  |  NN: ${NN_IN}-${NN_HID}-${NN_OUT}`, 30, yOff);

  // Buttons drawn separately
  drawButtonBar();

  if (isRecording) {
    textAlign(RIGHT, TOP);
    let pulse = 200 + Math.sin(frameCount * 0.1) * 55;
    fill(255, 40, 40, pulse);
    noStroke();
    ellipse(width - 35, 30, 14, 14);
    fill(255, 255, 255, 180);
    textSize(13);
    text("REC", width - 48, 22);
  }

  if (paused) {
    textAlign(CENTER, CENTER);
    textSize(28);
    fill(255, 255, 255, 150);
    text("PAUSED", width / 2, height / 2);
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
//  Button UI
// ============================================================
const BTN_H = 28;
const BTN_GAP = 4;
const BTN_Y_OFF = 48;

function getBtns() {
  return [
    { label: paused ? "PLAY" : "PAUSE", action: () => { paused = !paused; }, w: 46 },
    { label: "RESET", action: () => { loadPreset(currentPreset); }, w: 44 },
    { label: "1", action: () => loadPreset(0), w: 24, hi: currentPreset === 0 },
    { label: "2", action: () => loadPreset(1), w: 24, hi: currentPreset === 1 },
    { label: "3", action: () => loadPreset(2), w: 24, hi: currentPreset === 2 },
    { label: "4", action: () => loadPreset(3), w: 24, hi: currentPreset === 3 },
    { label: "5", action: () => loadPreset(4), w: 24, hi: currentPreset === 4 },
    { label: "TRAIL", action: () => { showTrails = !showTrails; }, w: 42, hi: showTrails },
    { label: "GRID", action: () => { showGrid = !showGrid; }, w: 38, hi: showGrid },
    { label: "NN", action: () => { showNN = !showNN; }, w: 28, hi: showNN },
    { label: "SFX", action: () => { sfxEnabled = !sfxEnabled; if (masterGain) masterGain.gain.value = sfxEnabled ? 0.35 : 0; }, w: 34, hi: sfxEnabled },
    { label: "-", action: () => { speedMultiplier = max(speedMultiplier / 2, 0.25); }, w: 24 },
    { label: `${speedMultiplier}x`, action: () => {}, w: 38 },
    { label: "+", action: () => { speedMultiplier = min(speedMultiplier * 2, 16); }, w: 24 },
    { label: isRecording ? "STOP" : "REC", action: () => { if (isRecording) stopRecording(); else startRecording(); }, w: 38, hi: isRecording },
  ];
}

function drawButtonBar() {
  let btns = getBtns();
  let totalW = btns.reduce((s, b) => s + b.w + BTN_GAP, -BTN_GAP);
  let sx = (width - totalW) / 2;
  let y = height - BTN_Y_OFF;

  noStroke();
  fill(10, 12, 25, 190);
  rect(sx - 10, y - 6, totalW + 20, BTN_H + 12, 8);

  let x = sx;
  textFont("monospace");
  textSize(10);
  textAlign(CENTER, CENTER);

  for (let b of btns) {
    if (b.hi) {
      fill(255, 255, 255, 25);
      stroke(255, 255, 255, 60);
    } else {
      fill(255, 255, 255, 8);
      stroke(255, 255, 255, 25);
    }
    strokeWeight(1);
    rect(x, y, b.w, BTN_H, 4);

    noStroke();
    fill(255, 255, 255, b.hi ? 200 : 130);
    text(b.label, x + b.w / 2, y + BTN_H / 2);
    x += b.w + BTN_GAP;
  }
}

function handleBtnClick(mx, my) {
  if (!audioStarted) initAudio();
  let btns = getBtns();
  let totalW = btns.reduce((s, b) => s + b.w + BTN_GAP, -BTN_GAP);
  let sx = (width - totalW) / 2;
  let y = height - BTN_Y_OFF;
  let x = sx;
  for (let b of btns) {
    if (mx >= x && mx <= x + b.w && my >= y && my <= y + BTN_H) {
      b.action();
      return true;
    }
    x += b.w + BTN_GAP;
  }
  return false;
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
  pixelDensity(1);
  textFont("monospace");
  loadPreset(currentPreset);
}

function loadPreset(idx) {
  currentPreset = idx;
  initPop();
  resetDisplay();
  playImpact();
}

function draw() {
  background(5, 5, 15);

  if (!paused) {
    let batch = Math.round(evalsPerFrame * speedMultiplier);
    trainBatch(batch);
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
  if (key === "m" || key === "M") {
    sfxEnabled = !sfxEnabled;
    if (masterGain) masterGain.gain.value = sfxEnabled ? 0.35 : 0;
  }
  if (key === "=" || key === "+") speedMultiplier = min(speedMultiplier * 2, 16);
  if (key === "-" || key === "_") speedMultiplier = max(speedMultiplier / 2, 0.25);
  if (key === "v" || key === "V") {
    if (isRecording) stopRecording(); else startRecording();
  }
  if (key >= "1" && key <= "5") loadPreset(int(key) - 1);
}

function mousePressed() { if (!audioStarted) initAudio(); handleBtnClick(mouseX, mouseY); }
function windowResized() { resizeCanvas(windowWidth, windowHeight); }
