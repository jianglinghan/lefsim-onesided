//! One-sided smoke example: the same convergent-CTCF TAD as `single_lef`,
//! but the LEF extrudes one-sidedly. It loads with one leg anchored, grows
//! its loop from the active leg alone, and each CTCF hit switches the roles:
//! the hitting leg anchors at the barrier and the partner takes over.
//! Respawns are made negligibly rare so the ping-pong plays out.

use lefsim::*;

fn build() -> Sim {
    let mut sites = vec![Site::plain(); 100];
    sites[30] = Site::ctcf(Direction::Right, 1.0);
    sites[70] = Site::ctcf(Direction::Left, 1.0);

    let world = populate_with(
        Lattice::new(sites).expect("valid lattice"),
        1,
        &SpawnRule::UniformNonCtcf,
        3,
        Extrusion::OneSided,
    )
    .expect("populated");
    let model = Model::new(
        world,
        Rates {
            step: 1.0,
            respawn: 1e-9,
            respawn_blocked: 1e-9,
            respawn_captured: 1e-9,
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
    }
}

fn main() {
    let mut sim = build();
    let ((l, lm), (r, rm)) = sim.world().lef_state(0).unwrap();
    println!("t=0.0000  init             pos=({l}, {r})  mob=({lm:?}, {rm:?})");
    for _ in 0..80 {
        let Some((t, action)) = sim.step() else { break };
        let ((l, lm), (r, rm)) = sim.world().lef_state(0).unwrap();
        // One line per event; a switch shows as (Moving, Anchored) flipping.
        println!(
            "t={t:>8.4}  {:15}  pos=({l}, {r})  mob=({lm:?}, {rm:?})",
            describe(&sim, action)
        );
    }
}
