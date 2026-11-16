// The economy map: nodes, the edges that carried money in the last week, and money moving as particles.
import { Application, Container, Graphics, Sprite, Text, Texture } from "pixi.js";
import { Flow, FlowKind, NodeInfo, Replay, RunName, money } from "./model";

export const COLORS = {
  bg: 0x0b0f14,
  genuine: 0x56b4e9,
  fraud: 0xe69f00,
  caught: 0xf0e442,
  friction: 0xcc79a7,
  node: 0x16202b,
  nodeLine: 0x8a9bb0,
  inst: 0xdfe7f0,
  ring: 0xe69f00,
  text: 0xdfe7f0,
  muted: 0x7d8b9c,
  good: 0x4fd1a5,
  bad: 0xff8a80,
};

const KIND_COLOR: Record<FlowKind, number> = { g: COLORS.genuine, f: COLORS.fraud, c: COLORS.caught, x: COLORS.friction };

interface Particle {
  sprite: Sprite;
  pane: Pane;
  from: string;
  to: string;
  kind: FlowKind;
  t0: number;
  dur: number;
  bend: number;
}

interface NodeView {
  info: NodeInfo;
  box: Container;
  shape: Graphics;
  badge: Text;
  x: number;
  y: number;
  r: number;
}

interface Pane {
  run: RunName;
  root: Container;
  edges: Graphics;
  bursts: Graphics;
  particles: Container;
  nodes: Map<string, NodeView>;
  title: Text;
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Where each node sits, in [-1, 1] coordinates of its pane. */
function position(n: NodeInfo, rings: string[]): [number, number] {
  const fixed: Record<string, [number, number]> = {
    brindlemark: [-0.38, -0.32], dovrex: [0.08, -0.32], hopcrate: [-0.38, 0.3], kellmont: [0.08, 0.3],
    background: [-0.86, -0.02], merchants: [-0.15, 0.84], web: [-0.15, -0.86], fraudsters: [0.5, 0.86],
  };
  if (fixed[n.id]) return fixed[n.id];
  const i = rings.indexOf(n.id);
  const k = Math.max(1, rings.length);
  const t = k === 1 ? 0.5 : i / (k - 1);
  const y = -0.8 + 1.45 * t;
  return [0.66 + 0.2 * Math.cos((t - 0.5) * Math.PI * 0.9), y];
}

export class MapScene {
  app = new Application();
  private panes: Pane[] = [];
  private pool: Sprite[] = [];
  private live: Particle[] = [];
  private dot!: Texture;
  private ringTex!: Texture;
  private burstsList: { pane: Pane; x: number; y: number; t0: number; color: number; dur: number }[] = [];
  focus: string | null = null;
  ghost = false;
  onFocus: (id: string | null) => void = () => {};
  private now = 0;
  private lastSpawnDay = -1;
  private shownDay = -1;
  private badges = new Map<string, { text: string; good: boolean | null }>();

  constructor(private host: HTMLElement, public replay: Replay, private reduced: boolean) {}

  async init(): Promise<void> {
    await this.app.init({
      background: COLORS.bg,
      resizeTo: this.host,
      antialias: true,
      autoDensity: true,
      resolution: Math.min(2, window.devicePixelRatio || 1),
      preference: "webgl",
    });
    this.app.canvas.setAttribute("aria-hidden", "true");
    this.host.appendChild(this.app.canvas);
    this.dot = this.app.renderer.generateTexture(new Graphics().circle(0, 0, 8).fill({ color: 0xffffff }));
    this.ringTex = this.app.renderer.generateTexture(new Graphics().circle(0, 0, 8).stroke({ width: 3, color: 0xffffff }));
    this.app.stage.eventMode = "static";
    this.app.stage.on("pointertap", (e) => {
      if (e.target === this.app.stage) this.onFocus(null);
    });
    this.app.ticker.add((t) => this.tick(t.deltaMS));
    this.app.renderer.on("resize", () => this.build());
    this.build();
  }

  destroy(): void {
    this.app.destroy(true, { children: true });
  }

  /** Lays the panes out again (size, ghost or focus changed, or new nodes). */
  build(): void {
    for (const p of this.panes) p.root.destroy({ children: true });
    this.panes = [];
    this.live = [];
    this.pool = [];
    this.burstsList = [];
    const W = this.app.screen.width;
    const H = this.app.screen.height;
    const runs: RunName[] = this.ghost && this.replay.hasReference() ? ["game", "reference"] : ["game"];
    const w = W / runs.length;
    for (const [i, run] of runs.entries()) this.panes.push(this.pane(run, i * w, 0, w, H, runs.length > 1));
    if (this.shownDay >= 0) {
      for (const p of this.panes) this.drawEdges(p, this.shownDay);
      this.setBadges(this.badges);
      this.applyFocus();
    }
  }

  private pane(run: RunName, x: number, y: number, w: number, h: number, titled: boolean): Pane {
    const root = new Container();
    root.position.set(x, y);
    this.app.stage.addChild(root);
    if (titled && run === "reference") {
      const sep = new Graphics().moveTo(0, 12).lineTo(0, h - 12).stroke({ width: 1, color: 0x2a3644 });
      root.addChild(sep);
    }
    const edges = new Graphics();
    const particles = new Container();
    const bursts = new Graphics();
    root.addChild(edges, particles, bursts);
    const title = new Text({
      text: titled ? (run === "game" ? "Game" : "Reference: built-in rules, scripted rings") : "",
      style: { fill: COLORS.muted, fontSize: 12, fontFamily: "system-ui, sans-serif" },
    });
    title.position.set(12, 8);
    root.addChild(title);
    const pane: Pane = { run, root, edges, bursts, particles, nodes: new Map(), title, x, y, w, h };
    const nodes = this.replay.meta?.nodes ?? [];
    const rings = nodes.filter((n) => n.kind === "ring").map((n) => n.id);
    const scale = Math.max(0.55, Math.min(1.2, Math.min(w, h) / 620));
    const mx = Math.max(56, w * 0.08);
    const my = Math.max(40, h * 0.09);
    for (const n of nodes) {
      const [px, py] = position(n, rings);
      const nx = mx + ((px + 1) / 2) * (w - 2 * mx);
      const ny = my + ((py + 1) / 2) * (h - 2 * my);
      pane.nodes.set(n.id, this.node(pane, n, nx, ny, scale));
    }
    return pane;
  }

  private node(pane: Pane, n: NodeInfo, x: number, y: number, s: number): NodeView {
    const box = new Container();
    box.position.set(x, y);
    const shape = new Graphics();
    let r = 18 * s;
    if (n.kind === "institution") {
      r = 26 * s;
      shape.roundRect(-r, -r, 2 * r, 2 * r, 8 * s).fill({ color: COLORS.node }).stroke({ width: 2, color: COLORS.inst });
    } else if (n.kind === "ring" || n.kind === "fraudsters") {
      r = (n.kind === "ring" ? 17 : 13) * s;
      shape.poly([0, -r * 1.25, r * 1.25, 0, 0, r * 1.25, -r * 1.25, 0]).fill({ color: 0x1d160a })
        .stroke({ width: n.player ? 3 : 1.5, color: COLORS.ring, alpha: n.player ? 1 : 0.7 });
    } else {
      r = 22 * s;
      shape.circle(0, 0, r).fill({ color: COLORS.node }).stroke({ width: 1.5, color: COLORS.nodeLine });
    }
    box.addChild(shape);
    const short = this.reduced && n.kind !== "ring" ? n.label.split(" ")[0] : n.label;
    const label = new Text({
      text: short,
      style: { fill: COLORS.text, fontSize: Math.round((n.kind === "institution" ? 13 : 11) * Math.max(0.85, s)), fontFamily: "system-ui, sans-serif", fontWeight: n.kind === "institution" ? "600" : "400" },
    });
    label.anchor.set(0.5, 0);
    label.position.set(0, r * 1.3 + 2);
    box.addChild(label);
    const badge = new Text({ text: "", style: { fill: COLORS.muted, fontSize: Math.round(11 * Math.max(0.85, s)), fontFamily: "system-ui, sans-serif" } });
    badge.anchor.set(0.5, 0);
    badge.position.set(0, r * 1.3 + 18 * Math.max(0.85, s));
    box.addChild(badge);
    box.eventMode = "static";
    box.cursor = "pointer";
    box.on("pointertap", () => this.onFocus(this.focus === n.id ? null : n.id));
    pane.root.addChild(box);
    return { info: n, box, shape, badge, x, y, r };
  }

  /** Puts the map at `day`: edges of the week up to it, badges, and (when playing) the day's money in motion. */
  setDay(day: number, playing: boolean, daySeconds: number): void {
    const d = Math.floor(day);
    this.shownDay = d;
    for (const p of this.panes) this.drawEdges(p, d);
    if (playing && d !== this.lastSpawnDay) {
      for (const p of this.panes) this.spawn(p, d, daySeconds);
    }
    this.lastSpawnDay = d;
    this.applyFocus();
  }

  setBadges(values: Map<string, { text: string; good: boolean | null }>): void {
    this.badges = values;
    for (const p of this.panes) {
      if (p.run !== "game") continue;
      for (const [id, v] of p.nodes) {
        const b = values.get(id);
        v.badge.text = b?.text ?? "";
        v.badge.style.fill = b?.good == null ? COLORS.muted : b.good ? COLORS.good : COLORS.bad;
      }
    }
  }

  private applyFocus(): void {
    for (const p of this.panes) {
      const near = new Set<string>();
      if (this.focus) {
        near.add(this.focus);
        for (const f of this.replay.flows(p.run, Math.max(this.replay.start, this.lastSpawnDay - 6), this.lastSpawnDay)) {
          if (f[0] === this.focus) near.add(f[1]);
          if (f[1] === this.focus) near.add(f[0]);
        }
      }
      for (const [id, v] of p.nodes) {
        v.box.alpha = !this.focus || near.has(id) ? 1 : 0.28;
        v.shape.scale.set(this.focus === id ? 1.15 : 1);
      }
    }
  }

  private drawEdges(p: Pane, d: number): void {
    const g = p.edges;
    g.clear();
    const from = Math.max(this.replay.start, d - 6);
    const flows = this.replay.flows(p.run, from, d);
    const agg = new Map<string, { s: string; t: string; fraud: boolean; n: number }>();
    for (const [s, t, k, n] of flows) {
      if (s === t || n <= 0) continue;
      const fraud = k !== "g";
      const key = `${s}|${t}|${fraud}`;
      const e = agg.get(key) ?? { s, t, fraud, n: 0 };
      e.n += n;
      agg.set(key, e);
    }
    for (const e of [...agg.values()].sort((a, b) => Number(a.fraud) - Number(b.fraud))) {
      const a = p.nodes.get(e.s);
      const b = p.nodes.get(e.t);
      if (!a || !b) continue;
      const touches = !this.focus || e.s === this.focus || e.t === this.focus;
      const [cx, cy] = control(a, b, e.fraud ? 0.16 : 0.1);
      g.moveTo(a.x, a.y).quadraticCurveTo(cx, cy, b.x, b.y).stroke({
        width: Math.min(7, 0.6 + Math.log10(1 + e.n) * (e.fraud ? 1.6 : 1.1)),
        color: e.fraud ? COLORS.fraud : COLORS.genuine,
        alpha: (e.fraud ? 0.4 : 0.24) * (touches ? 1 : 0.2),
      });
    }
  }

  private spawn(p: Pane, d: number, daySeconds: number): void {
    const flows = this.replay.flows(p.run, d);
    const budget = this.reduced ? 90 : 320;
    const wanted = flows.map((f) => particlesFor(f));
    const total = wanted.reduce((a, b) => a + b, 0);
    const k = total > budget ? budget / total : 1;
    const dur = Math.max(0.5, Math.min(2.4, daySeconds * 1.4));
    for (const [i, f] of flows.entries()) {
      const [s, t, kind, n, amt] = f;
      if (s === t) {
        if (kind === "x" || kind === "c") {
          const v = p.nodes.get(t);
          if (v) for (let j = 0; j < Math.min(3, Math.ceil(Math.log2(1 + n))); j++) {
            this.burstsList.push({ pane: p, x: v.x, y: v.y, t0: this.now + Math.random() * daySeconds, color: KIND_COLOR[kind], dur: 0.9 });
          }
        }
        continue;
      }
      const count = Math.max(kind === "g" ? 0 : 1, Math.round(wanted[i] * k));
      const size = kind === "g" ? 0.3 : 0.34 + 0.12 * Math.log10(1 + amt / Math.max(1, n) / 50);
      for (let j = 0; j < count; j++) {
        const sp = this.pool.pop() ?? new Sprite(this.dot);
        sp.texture = kind === "c" ? this.ringTex : this.dot;
        sp.anchor.set(0.5);
        sp.tint = KIND_COLOR[kind];
        sp.scale.set(Math.min(0.75, size) * (this.reduced ? 0.8 : 1));
        sp.visible = false;
        p.particles.addChild(sp);
        this.live.push({ sprite: sp, pane: p, from: s, to: t, kind, t0: this.now + Math.random() * daySeconds, dur: dur * (0.85 + Math.random() * 0.3), bend: kind === "g" ? 0.1 : 0.16 });
      }
    }
  }

  private tick(ms: number): void {
    this.now += ms / 1000;
    const keep: Particle[] = [];
    for (const q of this.live) {
      const a = q.pane.nodes.get(q.from);
      const b = q.pane.nodes.get(q.to);
      const u = (this.now - q.t0) / q.dur;
      if (!a || !b || u >= 1) {
        if (a && b && q.kind === "c") {
          const [x, y] = along(a, b, q.bend, 0.78);
          this.burstsList.push({ pane: q.pane, x, y, t0: this.now, color: COLORS.caught, dur: 0.7 });
        }
        q.sprite.visible = false;
        q.sprite.removeFromParent();
        this.pool.push(q.sprite);
        continue;
      }
      if (u < 0) {
        keep.push(q);
        continue;
      }
      // Caught fraud stops short of the institution it tried to get through.
      const v = q.kind === "c" ? Math.min(u / 0.8, 1) * 0.78 : u;
      const [x, y] = along(a, b, q.bend, v);
      q.sprite.position.set(x, y);
      q.sprite.visible = true;
      const touches = !this.focus || q.from === this.focus || q.to === this.focus;
      const base = q.kind === "g" ? 0.55 : 0.95;
      q.sprite.alpha = base * (touches ? 1 : 0.12) * (q.kind === "c" && u > 0.8 ? 1 - (u - 0.8) / 0.2 : 1);
      keep.push(q);
    }
    this.live = keep;
    const bursts = this.burstsList.filter((b) => this.now - b.t0 < b.dur);
    this.burstsList = bursts;
    for (const p of this.panes) p.bursts.clear();
    for (const b of bursts) {
      const u = (this.now - b.t0) / b.dur;
      if (u < 0) continue;
      const g = b.pane.bursts;
      const r = 6 + 16 * u;
      g.circle(b.x, b.y, r).stroke({ width: 2, color: b.color, alpha: 1 - u });
      if (b.color === COLORS.caught) {
        const k = 5 + 3 * u;
        g.moveTo(b.x - k, b.y - k).lineTo(b.x + k, b.y + k).moveTo(b.x + k, b.y - k).lineTo(b.x - k, b.y + k).stroke({ width: 2, color: b.color, alpha: 1 - u });
      }
    }
  }

  clearMotion(): void {
    for (const q of this.live) {
      q.sprite.visible = false;
      q.sprite.removeFromParent();
      this.pool.push(q.sprite);
    }
    this.live = [];
    this.burstsList = [];
    this.lastSpawnDay = -1;
  }
}

function particlesFor(f: Flow): number {
  const [, , k, n] = f;
  if (k === "g") return Math.ceil(Math.log2(1 + n) * 1.5);
  return Math.min(14, Math.ceil(Math.log2(1 + n) * 2) + 1);
}

function control(a: { x: number; y: number }, b: { x: number; y: number }, bend: number): [number, number] {
  const mx = (a.x + b.x) / 2;
  const my = (a.y + b.y) / 2;
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  return [mx - dy * bend, my + dx * bend];
}

function along(a: NodeView, b: NodeView, bend: number, u: number): [number, number] {
  const [cx, cy] = control(a, b, bend);
  const v = 1 - u;
  return [v * v * a.x + 2 * v * u * cx + u * u * b.x, v * v * a.y + 2 * v * u * cy + u * u * b.y];
}

export function badgeText(v: number | null): { text: string; good: boolean | null } {
  if (v == null) return { text: "", good: null };
  return { text: money(v, true), good: v > 0.5 ? true : v < -0.5 ? false : null };
}
