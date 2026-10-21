// The app's one simulation: its model inputs, the Sim, and the frame loop.
//
// Model edits (CTCFs, rates, strengths, the number of LEFs) build a new model
// and carry the LEFs over at the current time, so the run continues rather
// than restarting. Restart places the LEFs afresh with the seed. Resizing the
// lattice starts over from the default layout at the new size.
//
// The visible view drives what each frame does (see `ViewDriver`): it may ask
// for samples at fixed simulation times, independent of playback speed.

import {
  createLattice,
  createModel,
  createWorld,
  DIRECTION,
  type Extrusion,
  type Lattice,
  type LoadingSites,
  MOBILITY,
  type Model,
  modelFromJson,
  modelToJson,
  populate,
  type Rates,
  Sim,
} from "lefsim";

import { type Frame, setLattice, snapshotOf } from "../render.ts";

export const DEFAULT_N_SITES = 4_000;
export const MIN_SITES = 100;
export const MAX_SITES = 100_000;
export const MAX_LEFS = 700;
const CAPTURED = MOBILITY.indexOf("Captured");
const MOVING = MOBILITY.indexOf("Moving");
const LEFT = DIRECTION.indexOf("Left");
/** Cap on events per frame when no view samples, so a burst can't stall the page. */
const MAX_EVENTS_PER_FRAME = 40_000;
/** Wall-clock time per frame for a view's sampling, in ms; past it, playback slows. */
const SAMPLE_BUDGET_MS = 10;
/** The default density of loop extruders: 200 per 10,000 sites (the cr139 M5_v3 model). */
const LEFS_PER_SITE = 200 / 10_000;

/** A CTCF; `left` means it captures from the left (its chevron points left). */
export interface Ctcf {
  site: number;
  left: boolean;
  /** The probability, in [0, 1], that a landing leg interacts. */
  strength: number;
}

/** The default CTCFs, as fractions of the lattice: `>` `>` `<`, a domain with a second right-facing CTCF inside. */
const DEFAULT_LAYOUT: readonly { at: number; left: boolean }[] = [
  { at: 0.25, left: false },
  { at: 0.5, left: false },
  { at: 0.75, left: true },
];

/**
 * The cr139 zoom window (win3730: hg38 chr4:37,300,000–38,100,000) at
 * 200 bp/site, N=4000 — the region the user's 6×4 figure displays. CTCFs from
 * m5_v3_staticPol's `cr139.pkl`, keeping only sites inside the window and
 * rebased by −2990 (the window's left edge in full-region 200 bp units).
 *
 * Direction mapping (from the sim's translocator code): its
 * `CTCF_left_positions` stall the *leftward-moving* leg, so in lefsim terms
 * they capture legs arriving from the right — `left: false`, drawn `>`, like
 * the green "CTCF L" track in the user's figures. Its `CTCF_right_positions`
 * stall the rightward-moving leg — `left: true`, drawn `<`.
 */
const CR139_CTCF_LEFT_POSITIONS =
  "760 761 762 763 764 765 766 767 768 769 770 1510 1511 1512 1513 1514 1515 1516 1517 1518 1519 1520 2508 2509 2510 2511 2512 2513 2514 2515 2516 2517 2518 2519 2520 3386 3387 3388 3389 3390 3391 3800 3801 3802 3803 3804 3805 3806 3807 3808 3809 3810 3811 3812 3813 3814 3815 3816 3817 3818 3819 3820 3821 3822 3823 3824 3825 3826 3827 3828 3829 3830 3831 3832 3833 3834 3835 3836 3837 3838 3839 3840 3841 3842 3843 3844 3845 3846 3847 3848 3849 3850";
const CR139_CTCF_RIGHT_POSITIONS =
  "416 417 418 419 420 421 448 449 450 451 452 453 1331 1332 1333 1334 1335 1336 2460 2461 2462 2463 2464 2465 2466 2467 2468 2469 2470 2508 2509 2510 2511 2512 2513 2514 2515 2516 2517 2518 2519 2520 3380 3381 3382 3383 3384 3385 3386 3387 3388 3389 3390 3673 3674 3675 3676 3677 3678 3679";

/** The five M5_v3 Pol II genes with a TSS inside the zoom window (rebased by −2990). */
export const CR139_GENES: readonly number[] = [769, 1931, 2633, 2955, 3301];

/** The M5_v3 NIPBL loading sites inside the zoom window (4921–4925, rebased by −2990). */
export const CR139_LOADING: readonly number[] = [1931, 1932, 1933, 1934, 1935];

function cr139Ctcfs(strength: number): Ctcf[] {
  const parse = (sites: string, left: boolean) => sites.split(" ").map((s) => ({ site: Number(s), left, strength }));
  return [...parse(CR139_CTCF_LEFT_POSITIONS, false), ...parse(CR139_CTCF_RIGHT_POSITIONS, true)];
}

export const DEFAULT_STRENGTH = 1.0;
/** The M5_v3 rates: pause 0.5 (step 0.5), life 200 (respawn 5e-4); stalling
 *  never extends lifetime, so all three respawn rates are equal. */
export const DEFAULT_RATES: Readonly<Rates> = { step: 0.5, respawn: 0.0005, respawnBlocked: 0.0005, respawnCaptured: 0.0005 };

/** A complete setup for a run: what a tour scene loads (see tour/tour.ts). */
export interface Preset {
  nSites: number;
  nLefs: number;
  ctcfs: { site: number; left: boolean }[];
  /** One strength for every CTCF. */
  strength: number;
  rates: Rates;
  seed: number;
  /** Playback speed, in simulation time units per second. */
  speed: number;
}

/** What the visible view asks of the run each frame. */
export interface ViewDriver {
  /** Sample every `sampleDt` of simulation time, by calling `sample` once the sim has reached each sample time. */
  sampleDt?: number;
  sample?(time: number, sim: Sim): void;
  /** Once per animation frame (and after any change while paused), with the current state. */
  frame?(frame: Frame, sim: Sim): void;
  /** The model was edited ("edit") or the run restarted ("restart"). A new lattice size remounts the views instead. */
  changed?(kind: "edit" | "restart"): void;
}

/** Markers a view draws on the stage, which can be dragged along the lattice. */
export interface StageMarkers {
  sites: number[];
  move(index: number, site: number): void;
}

/** The most LEFs a lattice can hold: the spawn rule needs 2n - 1 plain sites for n LEFs. */
export function maxLefsFor(nSites: number, nCtcfs: number): number {
  return Math.max(0, Math.min(MAX_LEFS, Math.floor((nSites - nCtcfs + 1) / 2)));
}

function defaultCtcfs(nSites: number, strength: number): Ctcf[] {
  // At the default size the cr139 layout; any other size gets the generic one.
  if (nSites === DEFAULT_N_SITES) return cr139Ctcfs(strength);
  return DEFAULT_LAYOUT.map((c) => ({ site: Math.round(c.at * (nSites - 1)), left: c.left, strength }));
}

/** The setup the app opens with, and returns to when the tour finishes. */
export const DEFAULT_PRESET: Readonly<Preset> = {
  nSites: DEFAULT_N_SITES,
  nLefs: 80,
  ctcfs: defaultCtcfs(DEFAULT_N_SITES, DEFAULT_STRENGTH).map(({ site, left }) => ({ site, left })),
  strength: DEFAULT_STRENGTH,
  rates: DEFAULT_RATES,
  seed: 1,
  speed: 100,
};

export class Run {
  // Model inputs.
  nSites = $state(DEFAULT_N_SITES);
  nLefs = $state(DEFAULT_PRESET.nLefs);
  seed = $state(DEFAULT_PRESET.seed);
  /** The extrusion leg-rule: two-sided, or one-sided with anchored legs. */
  extrusion = $state<Extrusion>("one_sided");
  /** Pol II genes (TSS sites; all load at `polKin`), empty for no Pol II. */
  genes = $state<number[]>([...CR139_GENES]);
  /** The loading rate of every Pol II gene. */
  polKin = $state(0.001);
  /** The probability that an active leg meeting an RNAP switches legs. */
  polSwitch = $state(1);
  /** The RNAP unloading rate (mean dwell 1/off). */
  polOff = $state(0.002);
  /** LEF loading (NIPBL) sites: each is `loadingWeight`x as likely to receive a (re)spawning LEF as a plain site. */
  loadingSites = $state<number[]>([...CR139_LOADING]);
  /** A loading site's spawn weight relative to a plain site (50 = the M5_v3 preference). */
  loadingWeight = $state(50);
  rates = $state<Rates>({ ...DEFAULT_RATES });
  ctcfs = $state<Ctcf[]>(defaultCtcfs(DEFAULT_N_SITES, DEFAULT_STRENGTH));
  /** One strength for every CTCF (`strength`), or each its own. */
  strengthLocked = $state(true);
  /** The shared strength while locked, and the strength new CTCFs get. */
  strength = $state(DEFAULT_STRENGTH);
  /** The model last opened from a file, to go back to. */
  loaded = $state.raw<Model | undefined>(undefined);

  // Playback.
  playing = $state(false);
  /** Playback speed as log10 of simulation time units per second. */
  speedLog = $state(Math.log10(DEFAULT_PRESET.speed));

  // Readouts.
  frame = $state.raw<Frame | undefined>(undefined);
  eventsPerSecond = $state(0);
  /** The last frame couldn't keep up with the speed (event cap or sampling budget). */
  behind = $state(false);
  /** Bumped on every model change and restart, so the stage redraws. */
  revision = $state(0);

  /** Set by the visible view. */
  markers = $state.raw<StageMarkers | undefined>(undefined);

  sim: Sim | undefined;
  #driver: ViewDriver | undefined;
  #nextSample = 0;
  #rebuilds = 0;
  #lastNow: number | undefined;
  #raf = 0;
  #rateWindow: { t: number; events: number }[] = [];

  get speed(): number {
    return Math.round(10 ** this.speedLog);
  }

  get time(): number {
    return this.frame?.time ?? 0;
  }

  get held(): number {
    return this.frame ? this.frame.held.reduce((a, h) => a + h, 0) : 0;
  }

  get maxLefs(): number {
    return maxLefsFor(this.nSites, this.ctcfs.length);
  }

  /** The range of CTCF strengths, for display while unlocked. */
  get strengthRange(): [number, number] {
    if (this.ctcfs.length === 0) return [this.strength, this.strength];
    const s = this.ctcfs.map((c) => c.strength);
    return [Math.min(...s), Math.max(...s)];
  }

  lattice(): Lattice {
    return createLattice(
      this.nSites,
      this.ctcfs.map((c) => ({ site: c.site, capturesFrom: c.left ? "left" : "right", strength: c.strength })),
    );
  }

  /** The Pol II config as a model option, or undefined when gene-less. */
  pol2Option(): { pol2: NonNullable<Model["pol2"]> } | Record<string, never> {
    if (!this.genes.length) return {};
    return {
      pol2: {
        genes: this.genes.map((site) => ({ site, kin: this.polKin })),
        switchProb: this.polSwitch,
        off: this.polOff,
      },
    };
  }

  /** Set the Pol II config; restarts the run (placement and RNAPs start over). */
  setPol2(genes: number[], kin: number, switchProb: number, off: number) {
    this.genes = genes;
    this.polKin = kin;
    this.polSwitch = switchProb;
    this.polOff = off;
    this.restart();
  }

  /** Add a Pol II gene (TSS) at `site`; false if one is there already. */
  addGene(site: number): boolean {
    if (this.genes.includes(site)) return false;
    this.setPol2([...this.genes, site], this.polKin, this.polSwitch, this.polOff);
    return true;
  }

  /** Move gene `i` to `site`; false if another gene is there. */
  moveGene(i: number, site: number): boolean {
    if (this.genes[i] === site) return true;
    if (this.genes.includes(site)) return false;
    const genes = [...this.genes];
    genes[i] = site;
    this.setPol2(genes, this.polKin, this.polSwitch, this.polOff);
    return true;
  }

  removeGene(i: number) {
    const genes = [...this.genes];
    genes.splice(i, 1);
    this.setPol2(genes, this.polKin, this.polSwitch, this.polOff);
  }

  /** The loading-sites config as a populate/Sim option, or nothing when empty (uniform = the M5_v3u arm). */
  loadingOption(): { loading: LoadingSites } | Record<string, never> {
    if (!this.loadingSites.length) return {};
    return { loading: { sites: [...this.loadingSites], weight: this.loadingWeight } };
  }

  /** Set the loading-sites config; restarts the run (placement starts over). */
  setLoading(sites: number[], weight: number) {
    this.loadingSites = sites;
    this.loadingWeight = weight;
    this.restart();
  }

  /** Add a loading site at `site`; false if one is there already. */
  addLoadingSite(site: number): boolean {
    if (this.loadingSites.includes(site)) return false;
    this.setLoading([...this.loadingSites, site], this.loadingWeight);
    return true;
  }

  /** Move loading site `i` to `site`; false if another is there. */
  moveLoadingSite(i: number, site: number): boolean {
    if (this.loadingSites[i] === site) return true;
    if (this.loadingSites.includes(site)) return false;
    const sites = [...this.loadingSites];
    sites[i] = site;
    this.setLoading(sites, this.loadingWeight);
    return true;
  }

  removeLoadingSite(i: number) {
    const sites = [...this.loadingSites];
    sites.splice(i, 1);
    this.setLoading(sites, this.loadingWeight);
  }

  /** A new run: the LEFs placed afresh with the seed, on the current model. */
  restart() {
    const lattice = this.lattice();
    this.nLefs = Math.min(Math.max(0, Math.round(this.nLefs)), this.maxLefs);
    this.#start(
      createModel(
        populate(lattice, this.nLefs, { seed: this.seed, extrusion: this.extrusion, ...this.loadingOption() }),
        { ...this.rates },
        this.pol2Option(),
      ),
    );
  }

  /**
   * Start a fresh run of `model` (from a restart or a file). A new lattice size
   * detaches the visible view first: it's remounted for the new size, and the
   * new one attaches itself, so the old one never sees a lattice it wasn't
   * built for.
   */
  #start(model: Model) {
    if (this.sim && this.sim.lattice.nSites !== model.world.lattice.nSites) this.#driver = undefined;
    this.sim?.free();
    this.#rebuilds = 0;
    this.sim = new Sim(model, { seed: this.seed, extrusion: this.extrusion, ...this.loadingOption() });
    setLattice(model.world.lattice, this.genes);
    this.#rateWindow = [];
    this.#nextSample = this.sim.time + (this.#driver?.sampleDt ?? 0);
    this.revision++;
    this.#driver?.changed?.("restart");
    this.#publish();
  }

  /**
   * The same LEFs on the current model, continuing from the current time, with
   * LEFs added or dropped to match `nLefs`. A captured leg stays captured only
   * if a CTCF that can hold it is still at its site.
   */
  rebuild() {
    const old = this.sim;
    if (!old) return;
    // Copy the state first: the next call into wasm can detach its views.
    const st = old.state();
    const time = old.time;
    let leftPos = Array.from(st.leftPos);
    let rightPos = Array.from(st.rightPos);
    const leftMob = Array.from(st.leftMobility);
    const rightMob = Array.from(st.rightMobility);
    const capturesLeftAt = new Map(this.ctcfs.map((c) => [c.site, c.left]));
    const captured = (mob: number, site: number, leftLeg: boolean) =>
      mob === CAPTURED && capturesLeftAt.get(site) === !leftLeg ? CAPTURED : MOVING;
    let leftMobility = leftMob.map((m, i) => captured(m, leftPos[i]!, true));
    let rightMobility = rightMob.map((m, i) => captured(m, rightPos[i]!, false));

    // Match the LEF count: drop from the end, or add LEFs at free plain sites.
    const n = Math.min(Math.max(0, Math.round(this.nLefs)), this.maxLefs);
    this.nLefs = n;
    if (n < leftPos.length) {
      leftPos = leftPos.slice(0, n);
      rightPos = rightPos.slice(0, n);
      leftMobility = leftMobility.slice(0, n);
      rightMobility = rightMobility.slice(0, n);
    } else if (n > leftPos.length) {
      const taken = new Set([...leftPos, ...rightPos, ...this.ctcfs.map((c) => c.site)]);
      while (leftPos.length < n && taken.size < this.nSites) {
        const s = Math.floor(Math.random() * this.nSites);
        if (taken.has(s)) continue;
        taken.add(s);
        leftPos.push(s);
        rightPos.push(s);
        leftMobility.push(MOVING);
        rightMobility.push(MOVING);
      }
    }

    const lattice = this.lattice();
    const world = createWorld(lattice, { leftPos, rightPos, leftMobility, rightMobility });
    this.sim = new Sim(createModel(world, { ...this.rates }, this.pol2Option()), {
      seed: this.seed + ++this.#rebuilds,
      startTime: time,
      extrusion: this.extrusion,
      ...this.loadingOption(),
    });
    old.free();
    setLattice(lattice, this.genes);
    this.revision++;
    this.#driver?.changed?.("edit");
    this.#publish();
  }

  // ---------------------------------------------------------------- edits

  /** Add a CTCF at `site`, with the current strength; false if one is there already. */
  addCtcf(site: number, left: boolean): boolean {
    if (this.ctcfs.some((c) => c.site === site)) return false;
    this.ctcfs.push({ site, left, strength: this.strength });
    this.rebuild();
    return true;
  }

  /**
   * Add a CTCF in the middle of the widest stretch without one (the lattice's
   * ends count as edges), so repeated adds spread evenly. Its direction pairs
   * it into a domain where it can: facing left (`<`) to close a domain opened
   * by a left neighbor facing right; else facing right (`>`) to open one
   * closed by a right neighbor facing left; else facing left if it's the
   * first from the lattice's start, and right otherwise. False if no stretch
   * has room.
   */
  addCtcfEvenly(): boolean {
    const sorted = [...this.ctcfs].sort((a, b) => a.site - b.site);
    const edges = [-1, ...sorted.map((c) => c.site), this.nSites];
    let best = -1;
    for (let k = 0; k < edges.length - 1; k++) {
      const gap = edges[k + 1]! - edges[k]!;
      if (gap >= 2 && (best < 0 || gap > edges[best + 1]! - edges[best]!)) best = k;
    }
    if (best < 0) return false;
    const site = Math.floor((edges[best]! + edges[best + 1]!) / 2);
    const before = best > 0 ? sorted[best - 1] : undefined;
    const after = sorted[best];
    let left: boolean;
    if (before && !before.left)
      left = true; // close the domain `before` opens
    else if (after?.left)
      left = false; // open a domain `after` closes
    else left = before === undefined; // first from the start: the start and it bound a domain
    return this.addCtcf(site, left);
  }

  /** Remove the most recently added CTCF (the last listed, for a default or opened layout). */
  removeLastCtcf() {
    if (this.ctcfs.length) this.removeCtcf(this.ctcfs.length - 1);
  }

  /** Move CTCF `i` to `site`; false if another CTCF is there. */
  moveCtcf(i: number, site: number): boolean {
    if (this.ctcfs[i]?.site === site) return true;
    if (this.ctcfs.some((c) => c.site === site)) return false;
    this.ctcfs[i]!.site = site;
    this.rebuild();
    return true;
  }

  /** Turn CTCF `i` to capture from the other side. */
  flipCtcf(i: number) {
    this.ctcfs[i]!.left = !this.ctcfs[i]!.left;
    this.rebuild();
  }

  /** Set CTCF `i`'s own strength (only while strengths are unlocked). */
  setCtcfStrength(i: number, s: number) {
    if (this.strengthLocked || !(s >= 0 && s <= 1)) return;
    this.ctcfs[i]!.strength = s;
    this.rebuild();
  }

  removeCtcf(i: number) {
    this.ctcfs.splice(i, 1);
    this.rebuild();
  }

  clearCtcfs() {
    this.ctcfs = [];
    this.rebuild();
  }

  /** Set the rates; a value that isn't positive and finite is ignored. */
  setRates(rates: Partial<Rates>) {
    for (const [k, v] of Object.entries(rates) as [keyof Rates, number][]) {
      if (Number.isFinite(v) && v > 0) this.rates[k] = v;
    }
    this.rebuild();
  }

  /** Set the shared strength (and, while locked, every CTCF's). */
  setStrength(s: number) {
    if (!(s >= 0 && s <= 1)) return;
    this.strength = s;
    if (this.strengthLocked) {
      for (const c of this.ctcfs) c.strength = s;
      this.rebuild();
    }
  }

  /** Lock the strengths (equalizing every CTCF to the shared strength) or unlock them. */
  setStrengthLocked(locked: boolean) {
    this.strengthLocked = locked;
    if (locked) {
      for (const c of this.ctcfs) c.strength = this.strength;
      this.rebuild();
    }
  }

  setLefCount(n: number) {
    this.nLefs = Math.min(Math.max(0, Math.round(n)), this.maxLefs);
    this.rebuild();
  }

  /** A new lattice size: the default layout at that size, with LEFs at the default density, run afresh. */
  resize(nSites: number) {
    const n = Math.max(MIN_SITES, Math.min(MAX_SITES, Math.round(nSites)));
    this.nSites = n;
    this.ctcfs = defaultCtcfs(n, this.strength);
    this.strengthLocked = true;
    for (const c of this.ctcfs) c.strength = this.strength;
    this.genes = this.genes.filter((g) => g >= 1 && g <= n - 2);
    this.loadingSites = this.loadingSites.filter((s) => s >= 1 && s <= n - 2);
    this.nLefs = Math.max(1, Math.round(n * LEFS_PER_SITE));
    this.loaded = undefined;
    this.restart();
  }

  /** Set up `preset` and run it afresh: its lattice, CTCFs (with one shared strength), LEFs, rates, seed and speed. */
  applyPreset(preset: Preset) {
    this.nSites = Math.max(MIN_SITES, Math.min(MAX_SITES, Math.round(preset.nSites)));
    this.strength = preset.strength;
    this.strengthLocked = true;
    this.ctcfs = preset.ctcfs.map((c) => ({ ...c, strength: preset.strength }));
    this.nLefs = preset.nLefs;
    this.rates = { ...preset.rates };
    this.seed = preset.seed;
    this.speedLog = Math.log10(Math.max(1, preset.speed));
    this.loaded = undefined;
    this.restart();
  }

  // ---------------------------------------------------------------- files

  /** The current state as a model document (lefsim.model JSON): the world now, with its rates. */
  toJson(): string {
    if (!this.sim) throw new Error("no run to save");
    return modelToJson(this.sim.model(), 2);
  }

  /** Open a model document: its lattice, CTCFs, LEFs and rates, run from where it was saved. Throws on an invalid one. */
  open(json: string) {
    const model = modelFromJson(json);
    this.loaded = model;
    this.#apply(model);
  }

  /** Go back to the model last opened from a file. */
  reloadOpened() {
    if (this.loaded) this.#apply(this.loaded);
  }

  #apply(model: Model) {
    const { lattice, lefs } = model.world;
    const { site, capturesFrom, strength } = lattice.ctcfs;
    this.nSites = lattice.nSites;
    this.ctcfs = Array.from(site, (s, i) => ({ site: s, left: capturesFrom[i] === LEFT, strength: strength[i]! }));
    const strengths = this.ctcfs.map((c) => c.strength);
    this.strengthLocked = strengths.every((s) => s === strengths[0]);
    if (strengths.length) this.strength = this.strengthLocked ? strengths[0]! : Math.max(...strengths);
    this.nLefs = lefs.leftPos.length;
    this.rates = { ...model.rates };
    this.genes = []; // model files carry no Pol II config
    this.loadingSites = []; // nor a loading-sites config
    this.#start(model);
  }

  // ---------------------------------------------------------------- views

  /** The visible view's driver (or none). */
  setDriver(driver: ViewDriver | undefined) {
    this.#driver = driver;
    if (this.sim && driver?.sampleDt) this.#nextSample = this.sim.time + driver.sampleDt;
    if (this.sim && this.frame) driver?.frame?.(this.frame, this.sim);
  }

  // ---------------------------------------------------------------- playback

  setPlaying(on: boolean) {
    this.playing = on;
    this.#lastNow = undefined;
    this.#rateWindow = [];
    if (on && !this.#raf) this.#raf = requestAnimationFrame(this.#tick);
  }

  #tick = (now: number) => {
    this.#raf = 0;
    const sim = this.sim;
    if (!this.playing || !sim) return;
    const dt = this.#lastNow === undefined ? 0 : Math.min((now - this.#lastNow) / 1000, 0.1);
    this.#lastNow = now;
    const target = sim.time + this.speed * dt;
    const d = this.#driver;
    if (d?.sampleDt && d.sample) {
      // Sample at fixed simulation times, within the frame's budget; if it
      // runs out, stop at the last sample so the samples never skip time.
      const deadline = performance.now() + SAMPLE_BUDGET_MS;
      let behind = false;
      while (this.#nextSample <= target) {
        if (performance.now() > deadline) {
          behind = true;
          break;
        }
        sim.runUntil(this.#nextSample);
        d.sample(this.#nextSample, sim);
        this.#nextSample += d.sampleDt;
      }
      if (!behind) sim.runUntil(target);
      this.behind = behind;
    } else {
      const report = sim.runUntil(target, { maxEvents: MAX_EVENTS_PER_FRAME });
      this.behind = report.stopped === "reached_event_count";
    }
    this.#publish(now);
    this.#raf = requestAnimationFrame(this.#tick);
  };

  /** Copy the current state into `frame`, update the readouts, and hand it to the view. */
  #publish(now = performance.now()) {
    const sim = this.sim;
    if (!sim) return;
    const frame = snapshotOf(sim.state(), sim.rnapState());
    this.frame = frame;
    this.#rateWindow.push({ t: now, events: sim.nEvents });
    while (this.#rateWindow.length > 2 && now - this.#rateWindow[0]!.t > 1000) this.#rateWindow.shift();
    if (this.#rateWindow.length > 1) {
      const a = this.#rateWindow[0]!;
      const b = this.#rateWindow.at(-1)!;
      this.eventsPerSecond = b.t > a.t ? ((b.events - a.events) / (b.t - a.t)) * 1000 : 0;
    }
    this.#driver?.frame?.(frame, sim);
  }
}
