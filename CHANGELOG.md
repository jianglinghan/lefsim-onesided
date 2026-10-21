# Changelog

All three packages (the `lefsim` crate, the Python wheel and the npm package)
share one version. While the version is 0.x, a minor release may break the API
in any language *or change seeded trajectories*; a patch release does neither.
Model JSON files carry their own format version.

## Unreleased

Added, all opt-in (seeded default trajectories unchanged):

- One-sided extrusion (`Extrusion::OneSided`, wasm/TS `extrusion` option).
- Pol II (batch-D static rule): `Pol2`/`Gene`/`Rnap` on the world, the Switch,
  RnapLoad and RnapOff actions, wasm/TS `pol2` model option and
  `Sim.rnapState`.
- Preferential LEF loading sites: `WeightedSpawn` and
  `Sim::with_spawn_extrusion_from` in Rust; the `loading` option on `populate`
  and `new Sim` in wasm/TS. Not yet exposed in the Python bindings.

## 0.1.0 - 2026-10-16

First release: the Rust core, Python bindings (Arrow output, Parquet event logs,
replay) and the WebAssembly/TypeScript package.
