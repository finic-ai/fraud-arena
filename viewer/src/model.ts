// The replay format (fraud_arena/replay.py) and what the viewer derives from it.

export type FlowKind = "g" | "f" | "c" | "x";
export type Flow = [string, string, FlowKind, number, number];
export type RunName = "game" | "reference";

export interface RingDay {
  attempts: number;
  successes: number;
  blocked: number;
  loss_observed: number;
  loss_background: number;
  by_node: Record<string, [number, number, number, number]>;
}

export interface Harness {
  workflow?: { cases?: number; actions?: Record<string, number> };
  queue?: { looks?: number; case_looks?: number; decided?: number };
  grayson?: number;
}

export interface DayFrame {
  t: "day";
  day: number;
  run: RunName;
  src: string;
  flows: Flow[];
  rings: Record<string, RingDay>;
  decisions?: Record<string, Record<string, Record<string, [number, number]>>>;
  harness?: Record<string, Harness>;
}

export interface Changes {
  changed: boolean;
  digest?: string | null;
  rules?: number;
  models?: number;
  model_kinds?: string[];
  workflow?: boolean;
  grayson?: string[];
  notes?: string[];
  review?: string[];
  keys?: string[];
  plan?: Record<string, unknown>;
}

export interface Turn {
  player: string;
  side: "defender" | "attacker";
  ok: boolean;
  error: string | null;
  seconds: number;
  waits: number;
  review: boolean | null;
  changes: Changes;
}

export interface BoundaryFrame {
  t: "boundary";
  day: number;
  index: number;
  turns: Turn[];
}

export interface DefenderScore {
  composite: number;
  losses_prevented: number;
  added_friction: number;
  mule_flow_prevented: number;
  displacement: number;
}

export interface AttackerScore {
  score: number;
  score_of_scripted_ring: number;
  vs_scripted: number;
}

export interface ScoreFrame {
  t: "score";
  day: number;
  exact: boolean;
  final?: boolean;
  defenders: Record<string, DefenderScore>;
  attackers: Record<string, AttackerScore>;
}

export interface NodeInfo {
  id: string;
  kind: "institution" | "background" | "merchants" | "web" | "fraudsters" | "ring";
  label: string;
  player?: string | null;
}

export interface Player {
  name: string;
  side: "defender" | "attacker";
  target: string | number;
  runner: string;
  model: string;
}

export interface Meta {
  t: "meta";
  format: string;
  status: "running" | "finished";
  source: string;
  revision: number;
  world: { id?: string; path?: string | null; sha256?: string; commit?: string };
  epoch: string;
  config: Record<string, string>;
  players: Player[];
  boundaries: number[];
  start_day: number;
  end_day: number;
  settle_days: number;
  nodes: NodeInfo[];
  synthetic?: boolean;
  synthetic_note?: string;
  notes?: string[];
}

export type Frame = Meta | DayFrame | BoundaryFrame | ScoreFrame | { t: "rollback"; day: number } | { t: "end"; day: number };

export interface Point {
  day: number;
  value: number;
}

export interface Series {
  player: string;
  side: "defender" | "attacker";
  exact: Point[];
  proxy: Point[];
}

const ADVERSE = new Set(["decline", "reject", "deny", "refuse", "step_up", "hold", "standard", "review", "raise_alert",
  "block", "restrict", "close_account"]);

export function isAdverse(action: string): boolean {
  return ADVERSE.has(action);
}

export class Replay {
  meta: Meta | null = null;
  game = new Map<number, DayFrame>();
  ref = new Map<number, DayFrame>();
  boundaries = new Map<number, BoundaryFrame>();
  scores = new Map<number, ScoreFrame>();
  ended = false;
  version = 0;
  private cache = new Map<string, unknown>();

  add(f: Frame): void {
    this.version++;
    this.cache.clear();
    switch (f.t) {
      case "meta":
        this.meta = f;
        break;
      case "day":
        (f.run === "reference" ? this.ref : this.game).set(f.day, f);
        break;
      case "boundary":
        this.boundaries.set(f.day, f);
        break;
      case "score":
        this.scores.set(f.day, f);
        break;
      case "rollback":
        for (const d of [...this.game.keys()]) if (d >= f.day) this.game.delete(d);
        break;
      case "end":
        this.ended = true;
        break;
    }
  }

  addText(text: string): void {
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      try {
        this.add(JSON.parse(line) as Frame);
      } catch {
        // a line still being written
      }
    }
  }

  get start(): number {
    return this.meta?.start_day ?? 0;
  }

  get end(): number {
    return this.meta?.end_day ?? this.start + 1;
  }

  /** The last day with data for the game run (the live edge while the match runs). */
  get lastDay(): number {
    let m = this.start - 1;
    for (const d of this.game.keys()) m = Math.max(m, d);
    return m;
  }

  hasReference(): boolean {
    return this.ref.size > 0;
  }

  node(id: string): NodeInfo | undefined {
    return this.meta?.nodes.find((n) => n.id === id);
  }

  player(name: string): Player | undefined {
    return this.meta?.players.find((p) => p.name === name);
  }

  private memo<T>(key: string, make: () => T): T {
    if (!this.cache.has(key)) this.cache.set(key, make());
    return this.cache.get(key) as T;
  }

  /** Each ring's totals from the start through `day` (inclusive) in one run. */
  ringTotals(run: RunName, day: number): Map<string, RingDay> {
    return this.memo(`rings:${run}:${day}`, () => {
      const out = new Map<string, RingDay>();
      const frames = run === "game" ? this.game : this.ref;
      for (let d = this.start; d <= day; d++) {
        const f = frames.get(d);
        if (!f) continue;
        for (const [r, x] of Object.entries(f.rings || {})) {
          const t = out.get(r) ?? { attempts: 0, successes: 0, blocked: 0, loss_observed: 0, loss_background: 0, by_node: {} };
          t.attempts += x.attempts;
          t.successes += x.successes;
          t.blocked += x.blocked;
          t.loss_observed += x.loss_observed;
          t.loss_background += x.loss_background;
          for (const [n, v] of Object.entries(x.by_node || {})) {
            const b = t.by_node[n] ?? [0, 0, 0, 0];
            for (let i = 0; i < 4; i++) b[i] += v[i];
            t.by_node[n] = b;
          }
          out.set(r, t);
        }
      }
      return out;
    });
  }

  /** An institution's decisions in [from, to]: decision -> action -> [on good, on fraud]. */
  decisionMix(inst: string, from: number, to: number, run: RunName = "game"): Record<string, Record<string, [number, number]>> {
    return this.memo(`mix:${run}:${inst}:${from}:${to}`, () => {
      const out: Record<string, Record<string, [number, number]>> = {};
      const frames = run === "game" ? this.game : this.ref;
      for (let d = from; d <= to; d++) {
        const dec = frames.get(d)?.decisions?.[inst];
        if (!dec) continue;
        for (const [k, acts] of Object.entries(dec)) {
          const o = (out[k] ??= {});
          for (const [a, v] of Object.entries(acts)) {
            const c = (o[a] ??= [0, 0]);
            c[0] += v[0];
            c[1] += v[1];
          }
        }
      }
      return out;
    });
  }

  harness(player: string, from: number, to: number): { cases: number; grayson: number; queue: number; actions: Record<string, number> } {
    const out = { cases: 0, grayson: 0, queue: 0, actions: {} as Record<string, number> };
    for (let d = from; d <= to; d++) {
      const h = this.game.get(d)?.harness?.[player];
      if (!h) continue;
      out.cases += h.workflow?.cases ?? 0;
      out.grayson += h.grayson ?? 0;
      out.queue += h.queue?.decided ?? 0;
      for (const [k, v] of Object.entries(h.workflow?.actions ?? {})) out.actions[k] = (out.actions[k] ?? 0) + v;
    }
    return out;
  }

  /** Flows of one day, or summed over [from, to]. */
  flows(run: RunName, from: number, to: number = from): Flow[] {
    return this.memo(`flows:${run}:${from}:${to}`, () => {
      const frames = run === "game" ? this.game : this.ref;
      if (from === to) return frames.get(from)?.flows ?? [];
      const acc = new Map<string, Flow>();
      for (let d = from; d <= to; d++) {
        for (const f of frames.get(d)?.flows ?? []) {
          const k = `${f[0]}|${f[1]}|${f[2]}`;
          const a = acc.get(k);
          if (a) {
            a[3] += f[3];
            a[4] += f[4];
          } else acc.set(k, [f[0], f[1], f[2], f[3], f[4]]);
        }
      }
      return [...acc.values()];
    });
  }

  /** Score series per player: the exact scores (at boundaries and the end), and while the match runs a proxy from
   * the incidents (losses at the observed institutions, game against reference). */
  series(): Series[] {
    return this.memo("series", () => {
      const players = this.meta?.players ?? [];
      const out: Series[] = [];
      const days = [...this.scores.keys()].sort((a, b) => a - b);
      const last = this.lastDay;
      for (const p of players) {
        const exact: Point[] = [];
        if (days.length) exact.push({ day: this.start, value: 0 });
        for (const d of days) {
          const s = this.scores.get(d)!;
          const v = p.side === "defender" ? s.defenders[p.name]?.composite : s.attackers[p.name]?.vs_scripted;
          if (v !== undefined) exact.push({ day: d, value: v });
        }
        const proxy: Point[] = [];
        if (this.hasReference()) {
          let acc = 0;
          proxy.push({ day: this.start, value: 0 });
          for (let d = this.start; d <= last; d++) {
            const g = this.game.get(d);
            const r = this.ref.get(d);
            if (!g || !r) break;
            if (p.side === "defender") {
              const inst = String(p.target);
              acc += lossAt(r, inst) - lossAt(g, inst);
            } else {
              const ring = `R${p.target}`;
              acc += (g.rings?.[ring]?.loss_observed ?? 0) - (r.rings?.[ring]?.loss_observed ?? 0);
            }
            proxy.push({ day: d + 1, value: acc });
          }
        }
        out.push({ player: p.name, side: p.side, exact, proxy });
      }
      return out;
    });
  }

  /** The value of a series at `day`: linear between known points, held after the last. */
  static at(points: Point[], day: number): number | null {
    if (!points.length || day < points[0].day) return null;
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1];
      const b = points[i];
      if (day <= b.day) return a.value + ((b.value - a.value) * (day - a.day)) / Math.max(1e-9, b.day - a.day);
    }
    return points[points.length - 1].value;
  }

  /** The plan an attacker had in force on `day` (its latest applied plan). */
  planAt(player: string, day: number): Record<string, unknown> | null {
    let plan: Record<string, unknown> | null = null;
    for (const b of [...this.boundaries.keys()].sort((x, y) => x - y)) {
      if (b > day) break;
      const t = this.boundaries.get(b)!.turns.find((t) => t.player === player);
      if (t?.changes.plan) plan = t.changes.plan;
    }
    return plan;
  }
}

function lossAt(f: DayFrame, inst: string): number {
  let s = 0;
  for (const r of Object.values(f.rings || {})) s += r.by_node?.[inst]?.[3] ?? 0;
  return s;
}

export function money(v: number, signed = false): string {
  const a = Math.abs(v);
  const s = a >= 1e6 ? `${(a / 1e6).toFixed(2)}M` : a >= 1e4 ? `${(a / 1e3).toFixed(1)}k` : a >= 1e3 ? `${(a / 1e3).toFixed(2)}k` : a.toFixed(0);
  const sign = v < 0 ? "-" : signed && v > 0 ? "+" : "";
  return `${sign}$${s}`;
}
