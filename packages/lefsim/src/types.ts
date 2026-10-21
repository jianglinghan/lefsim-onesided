/**
 * Shared constants and data shapes, used by both `Sim` and `Replayer`. Kept separate so neither module imports the other.
 */

/** Mobility names, indexed by the codes in {@link LiveState} mobility arrays. */
export const MOBILITY = ["Moving", "Captured", "Anchored"] as const;
/** A mobility name: `"Moving"`, `"Captured"` or `"Anchored"`. */
export type Mobility = (typeof MOBILITY)[number];

/** The extrusion leg-rule: which legs of a LEF may extrude. */
export type Extrusion = "two_sided" | "one_sided";

/** Direction names, indexed by the codes in {@link Ctcfs} `capturesFrom`. */
export const DIRECTION = ["Left", "Right"] as const;

/** Site-kind names, indexed by the core's site-kind codes. */
export const SITE_KIND = ["Plain", "Ctcf"] as const;

/** Row-kind names, indexed by the codes in a {@link DeltaChunk}'s `kind`. */
export const ROW_KIND = ["Step", "Respawn", "Keyframe", "Evict", "Switch", "RnapLoad", "RnapOff"] as const;

/** The `kind` code of keyframe rows. */
export const KEYFRAME = 2;

/**
 * A chunk of event deltas: one row per leg changed by each event, sorted by
 * time. Rows carry absolute values (the leg's new position and mobility), so
 * applying them in order is an idempotent overwrite. A keyframed chunk opens
 * with one `Keyframe` row per leg holding the full state.
 *
 * This plain-columns shape is what {@link Replayer} consumes, whatever the
 * source (a local {@link Sim}, or later a network stream).
 */
export interface DeltaChunk {
  /** When each row's event fired. */
  time: Float64Array;
  /** The leg each row changes: `2 * lef_id + side`, side 0 left and 1 right. */
  legId: Uint32Array;
  /** The leg's position after the event. */
  pos: Uint32Array;
  /** {@link MOBILITY} codes. */
  mobility: Uint8Array;
  /** {@link ROW_KIND} codes. */
  kind: Uint8Array;
  /** The chunk describes the state through this time, even past its last row. */
  end: number;
}

/**
 * Every LEF's current state, indexed by `lef_id`.
 *
 * The arrays are views into WebAssembly memory. **They are valid only until
 * the next call into the Sim** (or any Sim): wasm memory can grow, which
 * detaches them. Copy what you need to keep (`arr.slice()`).
 */
export interface LiveState {
  /** The sim's current time. */
  time: number;
  /** Each LEF's left leg position. */
  leftPos: Uint32Array;
  /** Each LEF's right leg position. */
  rightPos: Uint32Array;
  /** Each LEF's left leg mobility, as {@link MOBILITY} codes. */
  leftMobility: Uint8Array;
  /** Each LEF's right leg mobility, as {@link MOBILITY} codes. */
  rightMobility: Uint8Array;
}

/**
 * Grid snapshots of every LEF, snapshot-major: `time` has one entry per
 * snapshot, and every other column has `nLefs` entries per snapshot, so
 * snapshot `s`, LEF `i` is at index `s * nLefs + i`.
 */
export interface SnapshotColumns {
  /** LEFs per snapshot. */
  nLefs: number;
  /** Each snapshot's time. */
  time: Float64Array;
  /** Left leg positions, `nLefs` per snapshot. */
  leftPos: Uint32Array;
  /** Right leg positions, `nLefs` per snapshot. */
  rightPos: Uint32Array;
  /** Left leg mobilities, as {@link MOBILITY} codes, `nLefs` per snapshot. */
  leftMobility: Uint8Array;
  /** Right leg mobilities, as {@link MOBILITY} codes, `nLefs` per snapshot. */
  rightMobility: Uint8Array;
}

/** A chunk of grid snapshots, from `Sim.recordSnapshotChunk`. */
export interface SnapshotChunk extends SnapshotColumns {
  /** The sim's time after the call. */
  end: number;
  /** Why the call returned. */
  stopped: StopReason;
}

/**
 * Why a run call returned. `"recorder_full"` comes only from the chunk
 * methods and streams: the chunk filled first; call again for the next.
 */
export type StopReason = "reached_time" | "reached_event_count" | "exhausted" | "recorder_full";

/** Event rates per unit time. {@link createModel} checks that they're positive and finite. */
export interface Rates {
  /** Step rate of a moving, unblocked leg. */
  step: number;
  /** Basal respawn rate: a LEF with both legs free to move. */
  respawn: number;
  /**
   * Respawn rate of a LEF with either leg blocked (by a leg ahead or the
   * lattice's end), unless the other is captured.
   */
  respawnBlocked: number;
  /** Respawn rate of a LEF with a captured leg, whatever the other leg does. */
  respawnCaptured: number;
}

/** The built-in spawn rules: where a LEF lands when it (re)spawns. */
export type SpawnRule = "uniform_non_ctcf";

/**
 * Preferential LEF loading sites (e.g. NIPBL loading): on a (re)spawn, plain
 * lattice sites have weight 1 and each of these sites has weight `weight`, so
 * a `weight` of 50 makes a loading site 50x as likely to receive a LEF. Only
 * *where* LEFs land is weighted; the respawn rate is unchanged.
 */
export interface LoadingSites {
  /** The loading sites: integers in [0, nSites), each on a plain (non-CTCF) site. */
  sites: number[];
  /** A loading site's weight relative to a plain site: finite, at least 1. */
  weight: number;
}

/** A lattice's CTCFs as columns. `capturesFrom` holds {@link DIRECTION} codes. */
export interface Ctcfs {
  /** Each CTCF's site. */
  site: Uint32Array;
  /** Each CTCF's capture side, as {@link DIRECTION} codes. */
  capturesFrom: Uint8Array;
  /** Each CTCF's strength: the probability, in [0, 1], that a landing leg interacts. */
  strength: Float64Array;
}

/** A 1D lattice: its size and its CTCFs. Create with `createLattice`. */
export interface Lattice {
  /** Number of sites. */
  readonly nSites: number;
  /** The CTCFs, as columns. */
  readonly ctcfs: Ctcfs;
}

/**
 * The LEF table: one row per LEF, row `i` being LEF `i`. Mobility columns
 * hold {@link MOBILITY} codes. The same columns as a {@link LiveState}.
 */
export interface Lefs {
  /** Each LEF's left leg position. */
  leftPos: Uint32Array;
  /** Each LEF's right leg position. */
  rightPos: Uint32Array;
  /** Each LEF's left leg mobility, as {@link MOBILITY} codes. */
  leftMobility: Uint8Array;
  /** Each LEF's right leg mobility, as {@link MOBILITY} codes. */
  rightMobility: Uint8Array;
}

/** The state of a simulation at one moment. Create with `createWorld` or `populate`. */
export interface World {
  /** The lattice. */
  readonly lattice: Lattice;
  /** The LEFs on it. */
  readonly lefs: Lefs;
}

/** A complete model: everything a run needs except a seed. Create with `createModel`. */
export interface Model {
  /** The starting state. */
  readonly world: World;
  /** The event rates. */
  readonly rates: Rates;
  /** Where LEFs land when they respawn. */
  readonly spawnRule: SpawnRule;
  /**
   * Pol II genes, if any: stationary RNAPs load at each gene's TSS and block
   * it; a one-sided LEF's active leg meeting an RNAP switches legs with
   * probability `switchProb`. Absent means no Pol II.
   */
  readonly pol2?: Pol2;
}

/** The Pol II configuration: genes, the switch probability, the unload rate. */
export interface Pol2 {
  /** The genes RNAPs load at. */
  genes: { site: number; kin: number }[];
  /** Switch probability on encounter, in [0, 1]. */
  switchProb: number;
  /** RNAP unloading rate (mean dwell 1/off). */
  off: number;
}
