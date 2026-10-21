//! M5_v3 (static Pol II) reproduction: the cr139 gene set from the SOP,
//! scaled onto a 500-site lattice, with one-sided LEFs that switch legs at
//! both CTCF captures and stationary RNAPs (switch probability 1).
//!
//! Model settings follow sop/cr139_6x4_models.md §5 (batch D, staticPol):
//! one-sided extrusion, CTCF capture = certain (strength 1), pause 0.5
//! (step rate 0.5), RNAP dwell mean 500 steps (off 0.002), per-gene loading
//! rates as in the SOP table. Loading is uniform (the m5v3u arm); the
//! lattice is a single replica.

use lefsim::*;

/// The seven cr139 genes: (TSS in 200 bp lattice coords, kin per step).
/// Scaled by 1/20 onto the 500-site lattice.
const GENES: &[(u32, f64)] = &[
    (136, 2.56e-4),  // NWD2
    (188, 6.75e-4),  // C4orf19
    (246, 5.0e-3),   // RELL1 (the strong loading site)
    (281, 5.0e-3),   // PGM2
    (297, 1.0e-3),   // TBC1D1
    (315, 9.78e-4),  // PTTG2
    (490, 5.0e-3),   // KLF3
];

fn build() -> Sim {
    let mut sites = vec![Site::plain(); 500];
    // Two convergent CTCF pairs bounding "TADs".
    for (l, r) in [(100, 160), (330, 400)] {
        sites[l] = Site::ctcf(Direction::Right, 1.0);
        sites[r] = Site::ctcf(Direction::Left, 1.0);
    }

    let mut world = populate_with(
        Lattice::new(sites).expect("valid lattice"),
        40,
        &SpawnRule::UniformNonCtcf,
        3,
        Extrusion::OneSided,
    )
    .expect("populated");
    world
        .set_pol2(Pol2 {
            genes: GENES
                .iter()
                .map(|&(tss, kin)| Gene { tss, kin })
                .collect(),
            switch_prob: 1.0,
            off: 0.002,
        })
        .expect("valid Pol II config");

    let model = Model::new(
        world,
        Rates {
            step: 0.5, // pause 0.5
            respawn: 5e-4,
            respawn_blocked: 5e-3,
            respawn_captured: 5e-4,
        },
        SpawnRule::UniformNonCtcf,
    )
    .expect("valid model");
    Sim::with_extrusion(model, 42, Extrusion::OneSided)
}

fn describe(sim: &Sim, action: Action) -> String {
    match action {
        Action::Step { leg } => format!("Step leg {}", sim.leg_id(leg).unwrap()),
        Action::Respawn { lef } => format!("Respawn lef {}", sim.lef_id(lef).unwrap()),
        Action::Evict { leg } => format!("Evict leg {}", sim.leg_id(leg).unwrap()),
        Action::Switch { leg } => format!("Switch leg {}", sim.leg_id(leg).unwrap()),
        Action::RnapLoad { gene } => format!("RnapLoad gene {gene}"),
        Action::RnapOff { .. } => "RnapOff".to_string(),
    }
}

fn main() {
    let mut sim = build();
    let mut switches_at_rnap = 0;
    let mut loads = 0;
    let mut offs = 0;
    for _ in 0..3000 {
        let Some((t, action)) = sim.step() else { break };
        match action {
            Action::RnapLoad { .. } => loads += 1,
            Action::RnapOff { .. } => offs += 1,
            Action::Switch { .. } => switches_at_rnap += 1,
            _ => {}
        }
        if !matches!(action, Action::Step { .. }) {
            let rnaps: Vec<u32> = sim.world().rnap_sites();
            println!(
                "t={t:>9.2}  {:20}  rnaps={rnaps:?}",
                describe(&sim, action)
            );
        }
    }
    println!("\n{loads} RNAP loads, {offs} unloads, {switches_at_rnap} leg switches at RNAPs");
}
