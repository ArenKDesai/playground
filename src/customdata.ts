/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
 * User-defined regression datasets. The target can come from either a
 * formula f(x, y) or from CSV rows of (x, y, target) / (x, target).
 *
 * Formulas are parsed by a small recursive-descent parser instead of
 * eval/new Function, since they are shared through the URL hash.
 */

import {Example2D} from "./dataset";

/** Half-width of the input domain used by all the playground datasets. */
const RADIUS = 6;
/** CSV inputs are rescaled to [-CSV_RADIUS, CSV_RADIUS]. */
const CSV_RADIUS = 5.8;
/** Larger CSV files are subsampled to keep training interactive. */
export const MAX_CSV_POINTS = 3000;

export const DEFAULT_FORMULA = "x^2 - y^2";

export type Source = "formula" | "csv";

// ---------------------------------------------------------------------------
// Expression parsing.
// ---------------------------------------------------------------------------

export type Expr = (x: number, y: number) => number;

let FUNCTIONS: {[name: string]: (...args: number[]) => number} = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan,
  asin: Math.asin, acos: Math.acos, atan: Math.atan,
  sinh: x => (Math.exp(x) - Math.exp(-x)) / 2,
  cosh: x => (Math.exp(x) + Math.exp(-x)) / 2,
  tanh: x => {
    // Avoid Infinity / Infinity for large |x|.
    if (x > 20) { return 1; }
    if (x < -20) { return -1; }
    let e2x = Math.exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  },
  sigmoid: x => 1 / (1 + Math.exp(-x)),
  exp: Math.exp, log: Math.log, sqrt: Math.sqrt, abs: Math.abs,
  sign: x => x > 0 ? 1 : x < 0 ? -1 : 0,
  floor: Math.floor, ceil: Math.ceil, round: Math.round,
  min: Math.min, max: Math.max, pow: Math.pow, atan2: Math.atan2,
  hypot: (a, b) => Math.sqrt(a * a + b * b),
  step: x => x >= 0 ? 1 : 0
};

let ARITY: {[name: string]: number} = {
  min: 2, max: 2, pow: 2, atan2: 2, hypot: 2
};

let CONSTANTS: {[name: string]: number} = {
  pi: Math.PI, e: Math.E
};

export let FUNCTION_NAMES = Object.keys(FUNCTIONS);

type Token = {type: "num" | "id" | "op", value: string, pos: number};

function tokenize(src: string): Token[] {
  let tokens: Token[] = [];
  let i = 0;
  while (i < src.length) {
    let c = src.charAt(i);
    if (/\s/.test(c)) {
      i++;
    } else if (/[0-9.]/.test(c)) {
      let m = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(src.slice(i));
      if (m == null) {
        throw new Error(`Bad number at position ${i + 1}`);
      }
      tokens.push({type: "num", value: m[0], pos: i});
      i += m[0].length;
    } else if (/[a-zA-Z_]/.test(c)) {
      let m = /^[a-zA-Z_][a-zA-Z_0-9]*/.exec(src.slice(i));
      tokens.push({type: "id", value: m[0].toLowerCase(), pos: i});
      i += m[0].length;
    } else if (src.substr(i, 2) === "**") {
      tokens.push({type: "op", value: "^", pos: i});
      i += 2;
    } else if ("+-*/^(),%".indexOf(c) >= 0) {
      tokens.push({type: "op", value: c, pos: i});
      i++;
    } else {
      throw new Error(`Unexpected character "${c}" at position ${i + 1}`);
    }
  }
  return tokens;
}

/**
 * Compiles a formula in x and y into a function.
 * Grammar:
 *   expr   := term (('+' | '-') term)*
 *   term   := unary (('*' | '/' | '%') unary)*
 *   unary  := ('-' | '+') unary | power
 *   power  := atom ('^' unary)?          (right-associative)
 *   atom   := number | x | y | const | fn '(' expr (',' expr)* ')' | '(' expr ')'
 * Implicit multiplication such as "2x" or "2(x+1)" is also accepted.
 */
export function compileFormula(src: string): Expr {
  let tokens = tokenize(src);
  if (tokens.length === 0) {
    throw new Error("Formula is empty");
  }
  let pos = 0;

  function peek(): Token { return tokens[pos]; }
  function isOp(value: string): boolean {
    let t = peek();
    return t != null && t.type === "op" && t.value === value;
  }
  function expect(value: string) {
    if (!isOp(value)) {
      let t = peek();
      throw new Error(t == null ? `Expected "${value}" at end of formula` :
          `Expected "${value}" at position ${t.pos + 1}`);
    }
    pos++;
  }

  function parseExpr(): Expr {
    let left = parseTerm();
    while (isOp("+") || isOp("-")) {
      let op = tokens[pos++].value;
      let a = left, b = parseTerm();
      left = op === "+" ? (x, y) => a(x, y) + b(x, y) :
          (x, y) => a(x, y) - b(x, y);
    }
    return left;
  }

  function startsAtom(): boolean {
    let t = peek();
    return t != null && (t.type === "num" || t.type === "id" ||
        (t.type === "op" && t.value === "("));
  }

  function parseTerm(): Expr {
    let left = parseUnary();
    while (true) {
      let op: string;
      if (isOp("*") || isOp("/") || isOp("%")) {
        op = tokens[pos++].value;
      } else if (startsAtom()) {
        op = "*";  // Implicit multiplication.
      } else {
        break;
      }
      let a = left, b = parseUnary();
      if (op === "*") {
        left = (x, y) => a(x, y) * b(x, y);
      } else if (op === "/") {
        left = (x, y) => a(x, y) / b(x, y);
      } else {
        left = (x, y) => a(x, y) % b(x, y);
      }
    }
    return left;
  }

  function parseUnary(): Expr {
    if (isOp("-")) {
      pos++;
      let a = parseUnary();
      return (x, y) => -a(x, y);
    }
    if (isOp("+")) {
      pos++;
      return parseUnary();
    }
    return parsePower();
  }

  function parsePower(): Expr {
    let base = parseAtom();
    if (isOp("^")) {
      pos++;
      let exponent = parseUnary();
      return (x, y) => Math.pow(base(x, y), exponent(x, y));
    }
    return base;
  }

  function parseAtom(): Expr {
    let t = peek();
    if (t == null) {
      throw new Error("Formula ends unexpectedly");
    }
    pos++;
    if (t.type === "num") {
      let v = parseFloat(t.value);
      return () => v;
    }
    if (t.type === "op") {
      if (t.value === "(") {
        let inner = parseExpr();
        expect(")");
        return inner;
      }
      throw new Error(`Unexpected "${t.value}" at position ${t.pos + 1}`);
    }
    // Identifier.
    let name = t.value;
    if (name === "x") {
      return x => x;
    }
    if (name === "y") {
      return (x, y) => y;
    }
    if (CONSTANTS.hasOwnProperty(name)) {
      let v = CONSTANTS[name];
      return () => v;
    }
    if (FUNCTIONS.hasOwnProperty(name)) {
      let fn = FUNCTIONS[name];
      expect("(");
      let args: Expr[] = [parseExpr()];
      while (isOp(",")) {
        pos++;
        args.push(parseExpr());
      }
      expect(")");
      let arity = ARITY.hasOwnProperty(name) ? ARITY[name] : 1;
      if (args.length !== arity) {
        throw new Error(`${name}() takes ${arity} argument` +
            `${arity === 1 ? "" : "s"}, got ${args.length}`);
      }
      if (arity === 1) {
        let a = args[0];
        return (x, y) => fn(a(x, y));
      }
      let a0 = args[0], a1 = args[1];
      return (x, y) => fn(a0(x, y), a1(x, y));
    }
    throw new Error(`Unknown name "${t.value}" at position ${t.pos + 1}. ` +
        `Use x, y, pi, e or a function like sin(...)`);
  }

  let result = parseExpr();
  if (pos < tokens.length) {
    let t = tokens[pos];
    throw new Error(`Unexpected "${t.value}" at position ${t.pos + 1}`);
  }
  return result;
}

// ---------------------------------------------------------------------------
// CSV parsing.
// ---------------------------------------------------------------------------

export type CsvData = {
  /** Points already rescaled into the playground domain. */
  points: Example2D[];
  /** True when the CSV had only one input column. */
  oneDimensional: boolean;
  numRows: number;
  numSkipped: number;
  columns: string[];
  xRange: [number, number];
  yRange: [number, number];
  targetRange: [number, number];
};

function splitRow(line: string): string[] {
  let delim: RegExp;
  if (line.indexOf("\t") >= 0) {
    delim = /\t/;
  } else if (line.indexOf(";") >= 0) {
    delim = /;/;
  } else if (line.indexOf(",") >= 0) {
    delim = /,/;
  } else {
    delim = /\s+/;
  }
  return line.trim().split(delim).map(s => s.trim().replace(/^"(.*)"$/, "$1"));
}

function isNumeric(s: string): boolean {
  return s !== "" && isFinite(+s);
}

function range(values: number[]): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (let v of values) {
    if (v < lo) { lo = v; }
    if (v > hi) { hi = v; }
  }
  return [lo, hi];
}

/** Linearly maps [lo, hi] onto [-r, r]; constant columns map to 0. */
function rescaler(r: [number, number], radius: number): (v: number) => number {
  let [lo, hi] = r;
  if (hi - lo < 1e-12) {
    return () => 0;
  }
  return v => ((v - lo) / (hi - lo) * 2 - 1) * radius;
}

/**
 * Parses CSV text with columns (x, y, target) or (x, target). A header row is
 * optional. When there are more than three columns, the first two are used as
 * inputs and the last one as the target. Throws on unusable input.
 */
export function parseCsv(text: string): CsvData {
  let lines = text.split(/\r\n|\r|\n/)
      .filter(l => l.trim() !== "" && l.trim().charAt(0) !== "#");
  if (lines.length === 0) {
    throw new Error("No data rows found");
  }
  let columns: string[] = null;
  let first = splitRow(lines[0]);
  if (!first.every(isNumeric)) {
    columns = first;
    lines = lines.slice(1);
  }
  let rows: number[][] = [];
  let numSkipped = 0;
  let width = -1;
  for (let line of lines) {
    let cells = splitRow(line);
    if (!cells.every(isNumeric)) {
      numSkipped++;
      continue;
    }
    if (width === -1) {
      width = cells.length;
    }
    if (cells.length !== width) {
      numSkipped++;
      continue;
    }
    rows.push(cells.map(Number));
  }
  if (rows.length < 2) {
    throw new Error("Need at least 2 numeric rows");
  }
  if (width < 2) {
    throw new Error("Need at least 2 columns: x, target (or x, y, target)");
  }
  let oneDimensional = width === 2;
  if (columns == null || columns.length !== width) {
    columns = oneDimensional ? ["x", "target"] : ["x", "y", "target"];
  }
  let xs = rows.map(r => r[0]);
  let ys = oneDimensional ? null : rows.map(r => r[1]);
  let ts = rows.map(r => r[width - 1]);
  let xRange = range(xs);
  let yRange: [number, number] = oneDimensional ? null : range(ys);
  let targetRange = range(ts);
  let sx = rescaler(xRange, CSV_RADIUS);
  let sy = oneDimensional ? null : rescaler(yRange, CSV_RADIUS);
  let st = rescaler(targetRange, 1);
  let points: Example2D[] = rows.map((r, i) => ({
    x: sx(xs[i]),
    // 1-D data has no y; the network only gets x.
    y: oneDimensional ? 0 : sy(ys[i]),
    label: st(ts[i])
  }));
  return {
    points, oneDimensional, numRows: rows.length, numSkipped,
    columns: oneDimensional ? [columns[0], columns[width - 1]] :
        [columns[0], columns[1], columns[width - 1]],
    xRange, yRange, targetRange
  };
}

/** An example CSV: electricity price ($/MWh) by hour ending. */
export function exampleLmpCsv(): string {
  let lmp = [21.48, 19.79, 20.21, 20.19, 22.06, 30.67, 45.71, 39.28, 28.2,
      30.06, 32.3, 30.8, 31.65, 33.17, 30.93, 33.38, 40.58, 112.65, 160.21,
      70.62, 48.31, 34.83, 29.69, 26.98];
  return ["hour_ending,lmp"].concat(lmp.map((v, i) => `${i + 1},${v}`))
      .join("\n");
}

// ---------------------------------------------------------------------------
// The dataset generator.
// ---------------------------------------------------------------------------


/**
 * Data whose target only depends on x is shown as a 1-D plot: the vertical
 * axis of the output heatmap shows the target, mapped by this factor from
 * [-1, 1].
 */
export const DISPLAY_RADIUS = CSV_RADIUS;

let source: Source = "formula";
let formula: FormulaInfo = prepareFormula(DEFAULT_FORMULA);
let csvData: CsvData = null;

/** Range of f over a grid on the domain, used to rescale to [-1, 1]. */
function sampleRange(f: Expr): [number, number] {
  let lo = Infinity, hi = -Infinity;
  let n = 60;
  for (let i = 0; i <= n; i++) {
    for (let j = 0; j <= n; j++) {
      let v = f(-RADIUS + 2 * RADIUS * i / n, -RADIUS + 2 * RADIUS * j / n);
      if (isFinite(v)) {
        if (v < lo) { lo = v; }
        if (v > hi) { hi = v; }
      }
    }
  }
  if (lo === Infinity) {
    throw new Error("Formula is not finite anywhere on the [-6, 6] grid");
  }
  return [lo, hi];
}

/** True when f(x, y) does not depend on y on a sample grid. */
function ignoresY(f: Expr): boolean {
  let n = 24;
  for (let i = 0; i <= n; i++) {
    let x = -RADIUS + 2 * RADIUS * i / n;
    let v0 = f(x, 0);
    for (let j = 0; j <= n; j++) {
      let v = f(x, -RADIUS + 2 * RADIUS * j / n);
      if (isFinite(v) !== isFinite(v0) ||
          (isFinite(v) && Math.abs(v - v0) > 1e-9 * (1 + Math.abs(v0)))) {
        return false;
      }
    }
  }
  return true;
}

export type FormulaInfo = {
  fn: Expr,
  range: [number, number],
  oneDimensional: boolean
};

/** Compiles and validates a formula, returning it with its output range. */
export function prepareFormula(src: string): FormulaInfo {
  let fn = compileFormula(src);
  return {fn, range: sampleRange(fn), oneDimensional: ignoresY(fn)};
}

export function setFormula(info: FormulaInfo) {
  source = "formula";
  formula = info;
}

export function setCsv(data: CsvData) {
  source = "csv";
  csvData = data;
}

export function getSource(): Source {
  return source;
}

/** Whether the active custom dataset only depends on x. */
export function isOneDimensional(): boolean {
  return source === "csv" && csvData != null ? csvData.oneDimensional :
      formula.oneDimensional;
}

/** Moves 1-D points so their height shows the target value. */
export function toDisplayPoints(points: Example2D[], oneDimensional: boolean):
    Example2D[] {
  if (!oneDimensional) {
    return points;
  }
  return points.map(p => ({x: p.x, y: p.label * DISPLAY_RADIUS,
      label: p.label}));
}

/**
 * Maps the playground domain [-6, 6] back to original units, given that
 * [lo, hi] was rescaled to [-radius, radius].
 */
function originalDomain(r: [number, number], radius: number):
    [number, number] {
  if (r == null || r[1] - r[0] < 1e-12) {
    return null;
  }
  let toOrig = (v: number) => r[0] + (v / radius + 1) / 2 * (r[1] - r[0]);
  return [toOrig(-RADIUS), toOrig(RADIUS)];
}

/**
 * Axis domains (in original units) for the output heatmap of the active
 * custom dataset. null means the default [-6, 6] axis.
 */
export function axisDomains(): AxisInfo {
  if (source === "csv" && csvData != null) {
    let c = csvData.columns;
    return {
      x: originalDomain(csvData.xRange, CSV_RADIUS),
      y: csvData.oneDimensional ?
          originalDomain(csvData.targetRange, DISPLAY_RADIUS) :
          originalDomain(csvData.yRange, CSV_RADIUS),
      xLabel: c[0],
      yLabel: csvData.oneDimensional ? c[c.length - 1] : c[1]
    };
  }
  let oneD = formula.oneDimensional;
  return {
    x: null,
    y: oneD ? originalDomain(formula.range, DISPLAY_RADIUS) : null,
    xLabel: "x",
    yLabel: oneD ? "f(x)" : "y"
  };
}

function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

/**
 * Generates points for an arbitrary formula / CSV source. For 1-D data the
 * y input is always 0, so the network can only use x.
 */
export function generate(src: Source, info: FormulaInfo, csv: CsvData,
    numSamples: number, noise: number): Example2D[] {
  let points: Example2D[] = [];
  if (src === "formula") {
    let scale = rescaler(info.range, 1);
    for (let i = 0; i < numSamples; i++) {
      let x = randUniform(-RADIUS, RADIUS);
      let y = info.oneDimensional ? 0 : randUniform(-RADIUS, RADIUS);
      let noiseX = randUniform(-RADIUS, RADIUS) * noise;
      let noiseY = info.oneDimensional ? 0 :
          randUniform(-RADIUS, RADIUS) * noise;
      let v = info.fn(x + noiseX, y + noiseY);
      if (!isFinite(v)) {
        continue;
      }
      let label = Math.max(-1, Math.min(1, scale(v)));
      points.push({x, y, label});
    }
    return points;
  }
  let rows = csv.points;
  if (rows.length > MAX_CSV_POINTS) {
    // Random subsample via a partial Fisher-Yates shuffle of the indices.
    let indices = rows.map((_, i) => i);
    for (let i = 0; i < MAX_CSV_POINTS; i++) {
      let j = i + Math.floor(Math.random() * (indices.length - i));
      let tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
    }
    rows = indices.slice(0, MAX_CSV_POINTS).map(i => csv.points[i]);
  }
  for (let p of rows) {
    let label = p.label + randUniform(-2, 2) * noise;
    points.push({x: p.x, y: p.y, label: Math.max(-1, Math.min(1, label))});
  }
  return points;
}

// ---------------------------------------------------------------------------
// Built-in example: locational marginal prices (LMPs) by hour of day.
// ---------------------------------------------------------------------------

/** Price range ($/MWh) that is mapped onto the [-1, 1] target range. */
export const LMP_PRICE_RANGE: [number, number] = [-30, 190];
const LMP_HOUR_RANGE: [number, number] = [0, 24];

function bump(h: number, center: number, width: number): number {
  return Math.exp(-Math.pow(h - center, 2) / (2 * width * width));
}

/**
 * An exaggerated "duck curve": cheap overnight power, a morning ramp,
 * negative prices when solar floods the grid at midday, and a scarcity spike
 * when the sun sets and demand peaks. `spike` scales the evening peak.
 */
export function lmpCurve(hour: number, spike = 1): number {
  return 28 - 6 * bump(hour, 3, 2) + 30 * bump(hour, 7.5, 1.3) -
      45 * bump(hour, 13, 2.2) + spike * 120 * bump(hour, 19, 1.3);
}

/** Simulated real-time LMPs sampled at random times over many days. */
export function regressLmp(numSamples: number, noise: number): Example2D[] {
  let toX = rescaler(LMP_HOUR_RANGE, CSV_RADIUS);
  let toLabel = rescaler(LMP_PRICE_RANGE, 1);
  let points: Example2D[] = [];
  for (let i = 0; i < numSamples; i++) {
    let hour = randUniform(0, 24);
    // Some days the evening peak is tamer, some days it is brutal.
    let price = lmpCurve(hour, randUniform(0.7, 1.3)) +
        randUniform(-4, 4) + randUniform(-60, 60) * noise;
    let label = Math.max(-1, Math.min(1, toLabel(price)));
    points.push({x: toX(hour), y: 0, label});
  }
  return points;
}

export type AxisInfo = {
  x: [number, number],
  y: [number, number],
  xLabel: string,
  yLabel: string
};

export const LMP_AXES: AxisInfo = {
  x: originalDomain(LMP_HOUR_RANGE, CSV_RADIUS),
  y: originalDomain(LMP_PRICE_RANGE, DISPLAY_RADIUS),
  xLabel: "Hour of day",
  yLabel: "LMP ($/MWh)"
};

/** DataGenerator for the "Custom" regression thumbnail. */
export function regressCustom(numSamples: number, noise: number):
    Example2D[] {
  if (source === "csv" && csvData != null) {
    return generate("csv", null, csvData, numSamples, noise);
  }
  return generate("formula", formula, null, numSamples, noise);
}
