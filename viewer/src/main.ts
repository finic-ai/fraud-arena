// Fraud Arena match viewer. Mounts into every `<div id="fraud-arena-viewer" data-replay="URL">` (or any element with
// `data-fraud-arena-viewer`); `data-live="URL"` follows a running match through `fraud-arena watch`'s poll endpoint.
import { BoundaryFrame, Player, Replay, Series, Turn, isAdverse, money } from "./model";
import { MapScene, badgeText } from "./scene";

const DEF_COLORS = ["#56b4e9", "#4fd1a5", "#8fb8ff", "#b9e769"];
const ATT_COLORS = ["#e69f00", "#ff7f50", "#cc79a7", "#f0e442"];
const DASHES = ["", "6 3", "2 3", "10 3 2 3"];
const SPEEDS = [0.5, 1, 2, 4, 8];

function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, string> = {}, ...kids: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const k of kids) e.append(k);
  return e;
}

function esc(s: unknown): string {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

class Viewer {
  replay = new Replay();
  scene: MapScene | null = null;
  day = 0;
  playing = false;
  speed = 1;
  following = true;
  reduced: boolean;
  private file = "";
  private offset = 0;
  private last = performance.now();
  private shownDay = -1;
  private shownVersion = -1;
  private ui: Record<string, HTMLElement> = {};

  constructor(private root: HTMLElement, private url: string, private live: string | null) {
    this.reduced = window.matchMedia("(max-width: 760px)").matches;
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.playing = !calm;
    this.layout();
  }

  private layout(): void {
    const r = this.root;
    r.classList.add("fav");
    r.tabIndex = 0;
    r.setAttribute("role", "application");
    r.setAttribute("aria-label", "Fraud Arena match replay");
    r.innerHTML = "";
    const u = this.ui;
    u.title = h("div", { class: "fav-title" }, "Loading the match…");
    u.badges = h("div", { class: "fav-badges" });
    u.map = h("div", { class: "fav-map" });
    u.focusSel = h("select", { "aria-label": "Focus on a node" });
    u.ghost = h("input", { type: "checkbox", id: `fav-ghost-${Math.random().toString(36).slice(2)}` });
    const ghostLabel = h("label", { class: "fav-ghost" }, u.ghost, " Reference run side by side");
    ghostLabel.setAttribute("for", u.ghost.id);
    u.ghostLabel = ghostLabel;
    u.scores = h("section", { class: "fav-card", "aria-label": "Scores" });
    u.mix = h("section", { class: "fav-card", "aria-label": "Decisions" });
    u.rings = h("section", { class: "fav-card", "aria-label": "Ring take" });
    u.focus = h("section", { class: "fav-card fav-focus", "aria-live": "polite" });
    u.play = h("button", { class: "fav-play", type: "button", "aria-label": "Play" });
    u.speed = h("select", { "aria-label": "Playback speed" });
    for (const s of SPEEDS) u.speed.append(h("option", { value: String(s) }, `${s}× day/s`));
    (u.speed as HTMLSelectElement).value = "1";
    u.dayLabel = h("div", { class: "fav-day", "aria-live": "off" });
    u.range = h("input", { type: "range", class: "fav-range", "aria-label": "Simulated day", step: "1" });
    u.ticks = h("div", { class: "fav-ticks", "aria-hidden": "true" });
    u.markers = h("div", { class: "fav-markers" });
    u.liveBtn = h("button", { class: "fav-live", type: "button", hidden: "" }, "● Live");
    u.dialog = h("dialog", { class: "fav-dialog" });
    const legend = h("div", { class: "fav-legend", "aria-hidden": "true" });
    legend.innerHTML = `<span><i class="sw g"></i>genuine</span><span><i class="sw f"></i>fraud through</span>`
      + `<span><i class="sw c"></i>fraud stopped</span><span><i class="sw x"></i>good customer stopped</span>`
      + `<span><i class="sh inst"></i>institution</span><span><i class="sh ring"></i>ring</span>`;
    r.append(
      h("header", { class: "fav-head" }, u.title, u.badges),
      h("div", { class: "fav-body" },
        h("div", { class: "fav-stage" }, u.map, legend, h("div", { class: "fav-tools" }, h("label", {}, "Focus ", u.focusSel), ghostLabel)),
        h("aside", { class: "fav-side" }, u.scores, u.focus, u.mix, u.rings)),
      h("footer", { class: "fav-timeline" },
        h("div", { class: "fav-controls" }, u.play, u.speed, u.dayLabel, u.liveBtn),
        h("div", { class: "fav-track" }, u.markers, u.range, u.ticks)),
      u.dialog,
    );
    u.play.addEventListener("click", () => this.setPlaying(!this.playing));
    u.speed.addEventListener("change", () => (this.speed = Number((u.speed as HTMLSelectElement).value)));
    u.range.addEventListener("input", () => {
      this.day = Number((u.range as HTMLInputElement).value);
      this.following = false;
      this.scene?.clearMotion();
      this.render(true);
    });
    u.focusSel.addEventListener("change", () => this.setFocus((u.focusSel as HTMLSelectElement).value || null));
    u.ghost.addEventListener("change", () => {
      if (this.scene) {
        this.scene.ghost = (u.ghost as HTMLInputElement).checked;
        this.scene.build();
        this.render(true);
      }
    });
    u.liveBtn.addEventListener("click", () => {
      this.following = true;
      this.day = this.replay.lastDay;
      this.setPlaying(true);
    });
    r.addEventListener("keydown", (e) => {
      if ((e.target as HTMLElement).closest("select, input, button, dialog")) return;
      if (e.key === " ") {
        e.preventDefault();
        this.setPlaying(!this.playing);
      } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        this.following = false;
        this.day = Math.max(this.replay.start, Math.min(this.maxDay(), Math.floor(this.day) + (e.key === "ArrowRight" ? 1 : -1)));
        this.scene?.clearMotion();
        this.render(true);
      } else if (e.key === "Escape") this.setFocus(null);
    });
    this.setPlaying(this.playing);
  }

  async start(): Promise<void> {
    try {
      if (this.live) await this.poll();
      else this.replay.addText(await (await fetch(this.url)).text());
    } catch (e) {
      this.ui.title.textContent = `Can't load the replay: ${e}`;
      return;
    }
    if (!this.replay.meta) {
      this.ui.title.textContent = "Waiting for the match to start…";
    }
    this.day = this.live ? Math.max(this.replay.start, this.replay.lastDay) : this.replay.start;
    if (!this.live) this.following = false;
    this.scene = new MapScene(this.ui.map, this.replay, this.reduced);
    await this.scene.init();
    this.scene.onFocus = (id) => this.setFocus(id);
    this.onMeta();
    this.applyHash();
    if (this.live) window.setInterval(() => this.poll().catch(() => undefined), 2000);
    const loop = (t: number) => {
      const dt = Math.min(0.25, (t - this.last) / 1000);
      this.last = t;
      if (this.playing) {
        const max = this.maxDay();
        this.day = Math.min(max + 0.999, this.day + dt * this.speed);
        if (this.live && this.day >= max) this.following = true;
        if (!this.live && this.day >= this.replay.end - 0.001) this.setPlaying(false);
      }
      this.render(false);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  /** Deep links: `#day=63&focus=R2&ghost=1&paused=1&turns=35`. */
  private applyHash(): void {
    const q = new URLSearchParams(window.location.hash.slice(1));
    if (q.has("day")) {
      this.day = Math.max(this.replay.start, Math.min(this.maxDay(), Number(q.get("day"))));
      this.following = false;
    }
    if (q.get("paused") === "1") this.setPlaying(false);
    if (q.get("ghost") === "1" && this.scene && this.replay.hasReference() && !this.reduced) {
      (this.ui.ghost as HTMLInputElement).checked = true;
      this.scene.ghost = true;
      this.scene.build();
    }
    if (q.get("focus")) this.setFocus(q.get("focus"));
    const b = this.replay.boundaries.get(Number(q.get("turns")));
    if (b) this.openTurns(b);
    this.render(true);
  }

  private maxDay(): number {
    return this.replay.ended || !this.live ? this.replay.end - 1 : Math.max(this.replay.start, this.replay.lastDay);
  }

  private async poll(): Promise<void> {
    const r = await fetch(`${this.live}?offset=${this.offset}&file=${encodeURIComponent(this.file)}`, { cache: "no-store" });
    const p = (await r.json()) as { file: string; offset: number; text: string; reset: boolean };
    if (p.reset) {
      const hadMeta = !!this.replay.meta;
      this.replay = new Replay();
      if (this.scene) this.scene.replay = this.replay;
      this.replay.addText(p.text);
      if (hadMeta || this.replay.meta) this.onMeta();
    } else if (p.text) {
      const before = this.replay.meta;
      this.replay.addText(p.text);
      if (!before && this.replay.meta) this.onMeta();
    }
    this.file = p.file;
    this.offset = p.offset;
    if (this.following && !this.playing) this.day = this.maxDay();
  }

  private onMeta(): void {
    const m = this.replay.meta;
    if (!m || !this.scene) return;
    const world = m.world?.id ?? m.world?.path?.split("/").pop() ?? "match";
    const models = (side: string) => [...new Set(m.players.filter((p) => p.side === side).map((p) => `${p.runner}:${p.model}`))].join(", ") || "scripted";
    this.ui.title.innerHTML = `<b>${esc(world)}</b> · seed ${esc(m.config.seed)} · <span class="def">defenders ${esc(models("defender"))}</span> vs <span class="att">attackers ${esc(models("attacker"))}</span>`;
    this.ui.badges.innerHTML = (m.status === "finished" && this.replay.ended ? `<span class="b">Finished</span>` : `<span class="b live">Live</span>`)
      + (m.synthetic ? `<span class="b syn" title="${esc(m.synthetic_note)}">Synthetic demo</span>` : "")
      + (m.source === "live" && m.status !== "finished" ? `<span class="b">running totals are estimates</span>` : "");
    const sel = this.ui.focusSel as HTMLSelectElement;
    sel.innerHTML = `<option value="">Everything</option>` + m.nodes.map((n) => `<option value="${esc(n.id)}">${esc(n.label)}</option>`).join("");
    const r = this.ui.range as HTMLInputElement;
    r.min = String(m.start_day);
    r.max = String(m.end_day - 1);
    this.ui.ghostLabel.hidden = !this.replay.hasReference() || this.reduced;
    this.drawTicks();
    this.scene.build();
    this.render(true);
  }

  private setPlaying(on: boolean): void {
    this.playing = on;
    this.ui.play.textContent = on ? "❚❚" : "▶";
    this.ui.play.setAttribute("aria-label", on ? "Pause" : "Play");
    if (on && !this.live && this.day >= this.replay.end - 1.001) this.day = this.replay.start;
  }

  private setFocus(id: string | null): void {
    if (!this.scene) return;
    this.scene.focus = id;
    (this.ui.focusSel as HTMLSelectElement).value = id ?? "";
    this.render(true);
  }

  private drawTicks(): void {
    const m = this.replay.meta;
    if (!m) return;
    const span = Math.max(1, m.end_day - 1 - m.start_day);
    const pct = (d: number) => `${((d - m.start_day) / span) * 100}%`;
    this.ui.ticks.innerHTML = m.boundaries.map((b, i) => `<span class="tick${i === m.boundaries.length - 1 ? " end" : ""}" style="left:${pct(b)}"></span>`).join("")
      + (m.settle_days ? `<span class="settle" style="left:${pct(m.boundaries[m.boundaries.length - 1])};right:0">settling</span>` : "");
    this.ui.markers.innerHTML = "";
    for (const b of [...this.replay.boundaries.values()].sort((x, y) => x.day - y.day)) {
      const changed = b.turns.filter((t) => t.changes?.changed).length;
      const errors = b.turns.filter((t) => t.error).length;
      const btn = h("button", { type: "button", class: `mk${changed ? " ch" : ""}${errors ? " err" : ""}`, style: `left:${pct(b.day)}`, "aria-label": `Turns at day ${b.day}: ${changed} changed, ${errors} with errors` }, `W${b.index + 1}`);
      btn.addEventListener("click", () => this.openTurns(b));
      this.ui.markers.append(btn);
    }
  }

  private openTurns(b: BoundaryFrame): void {
    const d = this.ui.dialog as HTMLDialogElement;
    const rows = b.turns.map((t) => turnHtml(t, this.replay.player(t.player))).join("");
    d.innerHTML = `<form method="dialog"><header><h3>Week ${b.index + 1} · day ${b.day} · players' turns</h3><button aria-label="Close">✕</button></header><div class="turns">${rows}</div></form>`;
    if (typeof d.showModal === "function") d.showModal();
    else d.setAttribute("open", "");
  }

  private render(force: boolean): void {
    const m = this.replay.meta;
    if (!m || !this.scene) return;
    const d = Math.max(m.start_day, Math.min(m.end_day - 1, Math.floor(this.day)));
    const changedDay = d !== this.shownDay;
    const changedData = this.replay.version !== this.shownVersion;
    if (!force && !changedDay && !changedData) return;
    if (changedData) {
      this.drawTicks();
      this.ui.ghostLabel.hidden = !this.replay.hasReference() || this.reduced;
      const finished = this.replay.ended && m.status === "finished";
      this.ui.liveBtn.hidden = !this.live || finished;
    }
    this.shownDay = d;
    this.shownVersion = this.replay.version;
    (this.ui.range as HTMLInputElement).value = String(d);
    this.scene.setDay(this.day, this.playing, 1 / this.speed);
    this.ui.dayLabel.innerHTML = dayLabel(m.epoch, d, m.boundaries, m.settle_days, this.replay.lastDay, !!this.live && !this.replay.ended);
    const series = this.replay.series();
    const badges = new Map<string, { text: string; good: boolean | null }>();
    for (const s of series) {
      const p = this.replay.player(s.player)!;
      const v = Replay.at(s.exact.length > 1 ? s.exact : s.proxy, d + 1);
      badges.set(p.side === "defender" ? String(p.target) : `R${p.target}`, badgeText(v));
    }
    this.scene.setBadges(badges);
    this.ui.scores.innerHTML = scoresHtml(series, this.replay, d, this.reduced);
    this.ui.mix.innerHTML = mixHtml(this.replay, d, this.scene.focus);
    this.ui.rings.innerHTML = ringsHtml(this.replay, d, this.scene.focus);
    this.ui.focus.innerHTML = focusHtml(this.replay, d, this.scene.focus);
    this.ui.focus.hidden = !this.scene.focus;
    for (const el of this.ui.rings.querySelectorAll<HTMLElement>("[data-node]")) {
      el.addEventListener("click", () => this.setFocus(el.dataset.node!));
    }
  }
}

function dayLabel(epoch: string, d: number, bounds: number[], settle: number, last: number, live: boolean): string {
  const date = new Date(Date.parse(epoch) + d * 86400000);
  const ds = date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
  const end = bounds[bounds.length - 1];
  let phase: string;
  if (d >= end) phase = `settling (${d - end + 1}/${settle})`;
  else {
    const i = bounds.findIndex((b, j) => d >= b && d < (bounds[j + 1] ?? Infinity));
    phase = `week ${i + 1} of ${bounds.length - 1}`;
  }
  const behind = live && d < last ? ` · <span class="muted">${last - d} days behind live</span>` : "";
  return `<b>Day ${d}</b> · ${ds} · ${phase}${behind}`;
}

function turnHtml(t: Turn, p: Player | undefined): string {
  const c = t.changes ?? { changed: false };
  const who = `<b>${esc(t.player)}</b> <span class="muted">${esc(p ? `${p.runner}:${p.model}` : "")}</span>`;
  const status = t.error ? `<span class="err">✕ refused/error</span>` : c.changed ? `<span class="ok">✓ changed</span>` : `<span class="muted">no change</span>`;
  let body = "";
  if (t.side === "defender") {
    if (c.changed) {
      const parts = [`${c.rules ?? 0} rules`, `${c.models ?? 0} models${c.model_kinds?.length ? ` (${c.model_kinds.join(", ")})` : ""}`];
      if (c.workflow) parts.push("workflow.py");
      if (c.grayson?.length) parts.push(`Grayson at ${c.grayson.join(", ")}`);
      body += `<div>${esc(parts.join(" · "))} · bundle ${esc(c.digest)}</div>`;
      if (c.notes?.length) body += `<blockquote>${c.notes.map(esc).join("<br>")}</blockquote>`;
    } else if (c.digest) body += `<div class="muted">bundle ${esc(c.digest)} stays</div>`;
    if (c.review?.length) body += `<div class="muted">review: ${esc(c.review[0])}</div>`;
  } else {
    if (c.changed) body += `<div>plan changed: ${esc((c.keys ?? []).join(", "))}</div>`;
    if (c.plan) body += `<pre>${esc(JSON.stringify(c.plan, null, 1)).slice(0, 1200)}</pre>`;
  }
  if (t.error) body += `<div class="err">${esc(t.error)}</div>`;
  return `<div class="turn ${t.side}"><div class="who">${who} ${status} <span class="muted">${Math.round(t.seconds)}s</span></div>${body}</div>`;
}

function playerColor(series: Series[], name: string): [string, string] {
  const s = series.find((x) => x.player === name);
  const same = series.filter((x) => x.side === s?.side);
  const i = same.findIndex((x) => x.player === name);
  const colors = s?.side === "attacker" ? ATT_COLORS : DEF_COLORS;
  return [colors[i % colors.length], DASHES[i % DASHES.length]];
}

function scoresHtml(series: Series[], rp: Replay, d: number, reduced: boolean): string {
  const m = rp.meta!;
  const groups: ["defender" | "attacker", string, string][] = [
    ["defender", "Defenders", "losses prevented − friction, vs the reference run"],
    ["attacker", "Attackers", "ring's take at the observed institutions, vs the scripted ring"],
  ];
  let out = "";
  for (const [side, title, sub] of groups) {
    const ss = series.filter((s) => s.side === side);
    if (!ss.length) continue;
    const W = 320;
    const H = reduced ? 110 : 130;
    const pad = { l: 44, r: 8, t: 8, b: 16 };
    const x0 = m.start_day;
    const x1 = m.end_day;
    const pts = ss.flatMap((s) => (s.exact.length > 1 ? s.exact : s.proxy));
    const lo = Math.min(0, ...pts.map((p) => p.value));
    const hi = Math.max(lo + 1, 0, ...pts.map((p) => p.value));
    const X = (v: number) => pad.l + ((v - x0) / Math.max(1, x1 - x0)) * (W - pad.l - pad.r);
    const Y = (v: number) => pad.t + (1 - (v - lo) / Math.max(1e-9, hi - lo)) * (H - pad.t - pad.b);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)} score over time">`;
    for (const b of m.boundaries) svg += `<line x1="${X(b)}" x2="${X(b)}" y1="${pad.t}" y2="${H - pad.b}" class="grid"/>`;
    svg += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(0)}" y2="${Y(0)}" class="zero"/>`;
    svg += `<text x="${pad.l - 4}" y="${Y(hi) + 4}" class="ax" text-anchor="end">${money(hi)}</text>`;
    if (lo < 0) svg += `<text x="${pad.l - 4}" y="${Y(lo)}" class="ax" text-anchor="end">${money(lo)}</text>`;
    let legend = "";
    for (const s of ss) {
      const [color, dash] = playerColor(series, s.player);
      const exact = s.exact.length > 1;
      const pts2 = exact ? s.exact : s.proxy;
      if (pts2.length) {
        const path = pts2.map((p, i) => `${i ? "L" : "M"}${X(p.day).toFixed(1)},${Y(p.value).toFixed(1)}`).join("");
        svg += `<path d="${path}" stroke="${color}" stroke-dasharray="${exact ? dash : "3 3"}" class="ln"/>`;
        if (exact) for (const p of pts2.slice(1)) svg += `<circle cx="${X(p.day)}" cy="${Y(p.value)}" r="2.4" fill="${color}"/>`;
      }
      const v = Replay.at(pts2, d + 1);
      const p = rp.player(s.player);
      legend += `<li><svg width="22" height="8" aria-hidden="true"><line x1="0" x2="22" y1="4" y2="4" stroke="${color}" stroke-width="2.5" stroke-dasharray="${exact ? dash : "3 3"}"/></svg>`
        + `<span class="nm">${esc(s.player.replace(/^(defender|attacker)-/, ""))}</span> <span class="muted">${esc(p?.model ?? "")}</span>`
        + `<span class="val ${v != null && v < 0 ? "neg" : "pos"}">${v == null ? "–" : money(v, true)}${exact ? "" : "*"}</span></li>`;
    }
    svg += `<line x1="${X(d + 1)}" x2="${X(d + 1)}" y1="${pad.t}" y2="${H - pad.b}" class="cursor"/></svg>`;
    out += `<h4>${title} <small>${esc(sub)}</small></h4>${svg}<ul class="legend">${legend}</ul>`;
  }
  const anyProxy = series.some((s) => s.exact.length <= 1 && s.proxy.length);
  if (anyProxy) out += `<p class="note">* estimate from incident losses until the exact scores are computed at the end of the match.</p>`;
  else if (rp.scores.size) out += `<p class="note">Exact scores at each boundary (no settling yet); the last point is the final score.</p>`;
  return out;
}

function mixHtml(rp: Replay, d: number, focus: string | null): string {
  const m = rp.meta!;
  const insts = m.nodes.filter((n) => n.kind === "institution" && (!focus || focus === n.id || !m.nodes.some((x) => x.id === focus && x.kind === "institution")));
  const from = Math.max(m.start_day, d - 6);
  let out = `<h4>Decisions <small>days ${from}–${d}</small></h4>`;
  let any = false;
  for (const n of insts) {
    const mix = rp.decisionMix(n.id, from, d);
    const hs = n.player ? rp.harness(n.player, from, d) : null;
    const decs = Object.entries(mix).sort();
    if (!decs.length && !(hs && (hs.cases || hs.grayson || hs.queue))) continue;
    any = true;
    out += `<div class="inst"><div class="iname">${esc(n.label)}</div>`;
    for (const [dec, acts] of decs) {
      const total = Object.values(acts).reduce((a, v) => a + v[0] + v[1], 0);
      const segs = Object.entries(acts).sort((a, b) => Number(isAdverse(a[0])) - Number(isAdverse(b[0])) || b[1][0] + b[1][1] - (a[1][0] + a[1][1]));
      const bar = segs.map(([a, v]) => {
        const w = ((v[0] + v[1]) / Math.max(1, total)) * 100;
        const cls = isAdverse(a) ? "adv" : "pass";
        return `<span class="seg ${cls}" style="width:${w}%" title="${esc(a)}: ${v[0] + v[1]} (${v[1]} fraud)"></span>`;
      }).join("");
      const adv = segs.filter(([a]) => isAdverse(a)).map(([a, v]) => `${esc(a)} ${v[0] + v[1]} <span class="fr">${v[1]} fraud</span> / <span class="gd">${v[0]} good</span>`).join(" · ");
      out += `<div class="dec"><div class="dname">${esc(dec.replace(/^[a-z]+_/, ""))} <span class="muted">${total}</span></div><div class="bar">${bar}</div>${adv ? `<div class="adv-line">${adv}</div>` : ""}</div>`;
    }
    if (hs && (hs.cases || hs.grayson || hs.queue)) {
      out += `<div class="harness">workflow ${hs.cases} cases · Grayson ${hs.grayson} asks · queue ${hs.queue} decided</div>`;
    }
    out += `</div>`;
  }
  if (!any && rp.meta?.source === "live") out += `<p class="muted">Decision counts come with the exact replay when the match ends; live, only workflow and Grayson work shows here.</p>`;
  else if (!any) out += `<p class="muted">No decisions in this window.</p>`;
  return out;
}

function ringsHtml(rp: Replay, d: number, focus: string | null): string {
  const m = rp.meta!;
  const g = rp.ringTotals("game", d);
  const r = rp.hasReference() ? rp.ringTotals("reference", d) : null;
  const rows = [...g.entries()].sort((a, b) => (b[1].loss_observed + b[1].loss_background) - (a[1].loss_observed + a[1].loss_background));
  const max = Math.max(1, ...rows.map(([, v]) => v.loss_observed + v.loss_background), ...(r ? [...r.values()].map((v) => v.loss_observed + v.loss_background) : []));
  let out = `<h4>Ring take <small>incident losses to day ${d}; tick = reference run</small></h4><ul class="rings">`;
  for (const [k, v] of rows) {
    const node = m.nodes.find((n) => n.id === k);
    const tot = v.loss_observed + v.loss_background;
    const ref = r?.get(k);
    const refTot = ref ? ref.loss_observed + ref.loss_background : null;
    const obsW = (v.loss_observed / max) * 100;
    const bgW = (v.loss_background / max) * 100;
    out += `<li class="${focus === k ? "on" : ""}"><button type="button" data-node="${esc(k)}" class="rname">${esc(node?.label ?? k)}${node?.player ? ' <span class="ai">agent</span>' : ""}</button>`
      + `<div class="rbar"><span class="obs" style="width:${obsW}%"></span><span class="bgd" style="width:${bgW}%"></span>${refTot != null ? `<span class="ghost" style="left:${(refTot / max) * 100}%"></span>` : ""}</div>`
      + `<div class="rnum">${money(tot)} <span class="muted">${v.attempts} tries · ${v.successes} ok · ${v.blocked} blocked</span></div></li>`;
  }
  out += `</ul><p class="note"><span class="sw obs"></span> at the observed institutions <span class="sw bgd"></span> elsewhere</p>`;
  return out;
}

function focusHtml(rp: Replay, d: number, focus: string | null): string {
  if (!focus) return "";
  const n = rp.node(focus);
  if (!n) return "";
  let out = `<h4>${esc(n.label)} <small>${esc(n.kind)}</small></h4>`;
  const p = n.player ? rp.player(n.player) : undefined;
  if (p) out += `<div>Played by <b>${esc(p.name)}</b> <span class="muted">${esc(p.runner)}:${esc(p.model)}</span></div>`;
  else if (n.kind === "ring") out += `<div class="muted">Scripted ring (no agent)</div>`;
  else if (n.kind === "institution") out += `<div class="muted">Built-in rules (no agent)</div>`;
  if (p) {
    const bs = [...rp.boundaries.values()].filter((b) => b.day <= d).sort((a, b) => b.day - a.day);
    const last = bs.map((b) => [b, b.turns.find((t) => t.player === p.name)] as const).find(([, t]) => t && (t.changes?.changed || t.error));
    if (last) out += `<div class="turns compact"><div class="muted">Last change, day ${last[0].day}:</div>${turnHtml(last[1]!, p)}</div>`;
    if (p.side === "attacker") {
      const plan = rp.planAt(p.name, d);
      if (plan) out += `<div class="muted">Plan in force</div><pre>${esc(JSON.stringify(plan, null, 1)).slice(0, 800)}</pre>`;
    }
  }
  const flows = rp.flows("game", Math.max(rp.start, d - 6), d).filter((f) => (f[0] === focus || f[1] === focus) && f[0] !== f[1]);
  const sum = (k: string, dir: 0 | 1) => flows.filter((f) => f[2] === k && f[dir] === focus).reduce((a, f) => a + f[4], 0);
  out += `<div class="flows">Last 7 days: in ${money(sum("g", 1))} genuine, ${money(sum("f", 1))} fraud · out ${money(sum("g", 0))} genuine, ${money(sum("f", 0))} fraud</div>`;
  return out;
}

export function mount(el: HTMLElement): Viewer | null {
  const url = el.dataset.replay;
  if (!url || el.dataset.favMounted) return null;
  el.dataset.favMounted = "1";
  const v = new Viewer(el, url, el.dataset.live ?? null);
  void v.start();
  return v;
}

function mountAll(): void {
  document.querySelectorAll<HTMLElement>("#fraud-arena-viewer, [data-fraud-arena-viewer]").forEach((el) => mount(el));
}

(window as unknown as { FraudArenaViewer: unknown }).FraudArenaViewer = { mount };
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountAll);
else mountAll();
