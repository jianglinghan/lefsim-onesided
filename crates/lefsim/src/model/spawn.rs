//! Spawn rules: where a LEF lands when it (re)spawns.
//!
//! [`Spawn`] is what a rule can do; [`SpawnRule`] lists the built-in rules as
//! plain data; [`populate`] uses a rule to build a starting world.

use rand::{Rng, SeedableRng};

use super::SimRng;
use super::action::Extrusion;
use super::lattice::{Lattice, Position, SiteKind};
use super::world::World;

/// A spawn rule: pick sites, and say whether it always can.
///
/// Must stay object-safe (dyn-compatible): the engine takes `&dyn Spawn` so
/// that FFI wrappers can hold a concrete, non-generic simulation type. Don't
/// add generic methods or `Self`-returning methods.
pub trait Spawn {
    /// Pick a site for a (re)spawning LEF. The LEF itself is not in
    /// `world`'s occupancy while this runs.
    fn sample(&self, world: &World, rng: &mut SimRng) -> Position;

    /// Whether this rule can always place a LEF when `n_lefs` LEFs share
    /// `lattice`, whatever their positions. `sample` may loop forever or
    /// panic if this fails.
    fn check(&self, lattice: &Lattice, n_lefs: usize) -> Result<(), SpawnError>;
}

/// The built-in spawn rules.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "rule", rename_all = "snake_case"))]
pub enum SpawnRule {
    /// Uniform among `Plain` sites that no leg occupies.
    #[default]
    UniformNonCtcf,
    /// Per-site loading weights: a LEF loads at a site with probability
    /// proportional to its weight (high-weight sites are NIPBL-rich loading
    /// preferences; weight 1 is the background). Sampling is rejection over
    /// free plain sites, so a run never deadlocks; once every high-weight
    /// site is taken, loading falls back to effectively uniform among the
    /// free plain sites.
    WeightedSites { weights: Vec<f64> },
}

impl Spawn for SpawnRule {
    fn sample(&self, world: &World, rng: &mut SimRng) -> Position {
        match self {
            // By rejection. Fine when occupancy << lattice length; precompute
            // a free-site list otherwise.
            SpawnRule::UniformNonCtcf => {
                let n = world.lattice.len() as u32;
                loop {
                    let i = rng.gen_range(0..n);
                    let kind = world.lattice.sites[i as usize].kind;
                    if matches!(kind, SiteKind::Plain) && !world.occupied(i) {
                        return i;
                    }
                }
            }
            SpawnRule::WeightedSites { weights } => {
                let n = world.lattice.len() as u32;
                let max_w = weights.iter().copied().fold(0.0, f64::max);
                // Cap the attempts so a lattice whose high-weight sites are
                // all taken (or absent) falls back to uniform placement
                // instead of spinning.
                let cap = 100 * world.lattice.len() + 1;
                for _ in 0..cap {
                    let i = rng.gen_range(0..n);
                    let kind = world.lattice.sites[i as usize].kind;
                    if !matches!(kind, SiteKind::Plain) || world.occupied(i) {
                        continue;
                    }
                    let w = weights.get(i as usize).copied().unwrap_or(0.0);
                    if w > 0.0 && rng.gen_bool((w / max_w).min(1.0)) {
                        return i;
                    }
                }
                // Fallback: uniform among free plain sites.
                loop {
                    let i = rng.gen_range(0..n);
                    let kind = world.lattice.sites[i as usize].kind;
                    if matches!(kind, SiteKind::Plain) && !world.occupied(i) {
                        return i;
                    }
                }
            }
        }
    }

    fn check(&self, lattice: &Lattice, n_lefs: usize) -> Result<(), SpawnError> {
        match self {
            // The other n - 1 LEFs cover at most 2n - 2 sites, so 2n - 1
            // plain sites always leave one free.
            SpawnRule::UniformNonCtcf => {
                let needed = (2 * n_lefs).saturating_sub(1);
                let n_plain = lattice.n_plain();
                if n_plain < needed {
                    return Err(SpawnError::NotEnoughSites {
                        n_lefs,
                        needed,
                        n_plain,
                    });
                }
                Ok(())
            }
            // Same plain-site budget; concentration beyond it degrades to
            // uniform loading rather than failing.
            SpawnRule::WeightedSites { .. } => {
                let needed = (2 * n_lefs).saturating_sub(1);
                let n_plain = lattice.n_plain();
                if n_plain < needed {
                    return Err(SpawnError::NotEnoughSites {
                        n_lefs,
                        needed,
                        n_plain,
                    });
                }
                Ok(())
            }
        }
    }
}

/// Why a spawn rule can't place the LEFs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpawnError {
    /// Not enough plain sites for `n_lefs` LEFs: the rule needs `needed`.
    NotEnoughSites {
        n_lefs: usize,
        needed: usize,
        n_plain: usize,
    },
}

impl std::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpawnError::NotEnoughSites {
                n_lefs,
                needed,
                n_plain,
            } => write!(
                f,
                "too many LEFs: {n_lefs} LEFs need at least {needed} non-CTCF sites, \
                 the lattice has {n_plain}"
            ),
        }
    }
}

impl std::error::Error for SpawnError {}

/// A world with `n` LEFs placed one at a time by `rule`, exactly as respawns
/// place them: each LEF lands at a sampled site (rolling for CTCF capture,
/// or drawing an active leg under [`Extrusion::OneSided`]) before the next
/// site is sampled. `seed` drives only this placement.
pub fn populate(
    lattice: Lattice,
    n: usize,
    rule: &dyn Spawn,
    seed: u64,
) -> Result<World, SpawnError> {
    populate_with(lattice, n, rule, seed, Extrusion::TwoSided)
}

/// [`populate`] under a chosen [`Extrusion`] leg-rule, so one-sided worlds
/// start with each LEF's passive leg anchored.
pub fn populate_with(
    lattice: Lattice,
    n: usize,
    rule: &dyn Spawn,
    seed: u64,
    extrusion: Extrusion,
) -> Result<World, SpawnError> {
    rule.check(&lattice, n)?;
    let mut rng = SimRng::seed_from_u64(seed);
    let mut world = World::new(lattice);
    for _ in 0..n {
        place_lef(&mut world, rule, &mut rng, extrusion);
    }
    Ok(world)
}

/// Sample a site with `rule` and land a new LEF there.
pub(crate) fn place_lef(
    world: &mut World,
    rule: &dyn Spawn,
    rng: &mut SimRng,
    extrusion: Extrusion,
) {
    let pos = rule.sample(world, rng);
    world.spawn_lef(pos, rng, extrusion);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Ctcf, Direction, Mobility};

    #[test]
    fn uniform_check_needs_2n_minus_1_plain_sites() {
        let rule = SpawnRule::UniformNonCtcf;
        let ctcf = Ctcf {
            site: 0,
            captures_from: Direction::Left,
            strength: 1.0,
        };
        let lattice = Lattice::with_ctcfs(6, &[ctcf]).unwrap(); // 5 plain sites
        assert_eq!(rule.check(&lattice, 0), Ok(()));
        assert_eq!(rule.check(&lattice, 3), Ok(()));
        assert_eq!(
            rule.check(&lattice, 4),
            Err(SpawnError::NotEnoughSites {
                n_lefs: 4,
                needed: 7,
                n_plain: 5
            })
        );
    }

    #[test]
    fn populate_places_valid_lefs_reproducibly() {
        let ctcfs: Vec<_> = (0..10)
            .map(|i| Ctcf {
                site: 10 * i + 5,
                captures_from: Direction::Left,
                strength: 1.0,
            })
            .collect();
        let lattice = Lattice::with_ctcfs(100, &ctcfs).unwrap();
        let world = populate(lattice.clone(), 20, &SpawnRule::UniformNonCtcf, 3).unwrap();

        assert_eq!(world.n_lefs(), 20);
        let states = world.lef_states();
        for i in 0..20 {
            let ((l, lm), (r, rm)) = states.get(i).unwrap();
            assert_eq!(l, r, "both legs land on one site");
            assert!(matches!(lattice.sites()[l as usize].kind, SiteKind::Plain));
            assert_eq!((lm, rm), (Mobility::Moving, Mobility::Moving));
        }
        // A rebuild from the table passes validation, so no site is shared.
        assert_eq!(
            World::from_lef_states(lattice.clone(), &states).as_ref(),
            Ok(&world)
        );

        let again = populate(lattice.clone(), 20, &SpawnRule::UniformNonCtcf, 3).unwrap();
        assert_eq!(again, world);
        let other = populate(lattice, 20, &SpawnRule::UniformNonCtcf, 4).unwrap();
        assert_ne!(other, world);
    }

    #[test]
    fn populate_checks_the_rule_first() {
        let err = populate(Lattice::plain(3).unwrap(), 3, &SpawnRule::UniformNonCtcf, 0);
        assert_eq!(
            err,
            Err(SpawnError::NotEnoughSites {
                n_lefs: 3,
                needed: 5,
                n_plain: 3
            })
        );
    }

    #[test]
    fn weighted_sites_sample_follows_weights() {
        use rand::SeedableRng;
        let n = 100;
        let mut weights = vec![1.0; n];
        weights[50] = 50.0; // a single NIPBL-rich hot site
        let rule = SpawnRule::WeightedSites { weights };
        // One LEF occupying site 0: 98 free plain sites at weight 1.
        let mut world = World::new(Lattice::plain(n).unwrap());
        world
            .add_lef((0, Mobility::Moving), (0, Mobility::Moving))
            .unwrap();
        let mut rng = SimRng::seed_from_u64(11);
        let trials = 20_000;
        let mut hits = 0;
        for _ in 0..trials {
            if rule.sample(&world, &mut rng) == 50 {
                hits += 1;
            }
        }
        // Rejection sampling lands on a site with probability ∝ its weight:
        // P(hot) = 50 / (98 * 1 + 50).
        let p = 50.0 / 148.0;
        let frac = hits as f64 / trials as f64;
        let sigma = (p * (1.0 - p) / trials as f64).sqrt();
        assert!(
            (frac - p).abs() < 4.0 * sigma,
            "hot-site fraction {frac}, want {p} (4σ {sigma:.4})"
        );
    }

    #[test]
    fn weighted_sites_reproduce_and_report_errors() {
        let n = 1000;
        let mut weights = vec![1.0; n];
        for i in [100, 200, 300, 400, 500] {
            weights[i] = 50.0;
        }
        let lattice = Lattice::plain(n).unwrap();
        let rule = SpawnRule::WeightedSites { weights };
        assert_eq!(rule.check(&lattice, 20), Ok(()));

        let world = populate(lattice.clone(), 20, &rule, 7).expect("populated");
        let again = populate(lattice.clone(), 20, &rule, 7).unwrap();
        assert_eq!(again, world);

        let tight = Lattice::plain(6).unwrap();
        assert_eq!(
            rule.check(&tight, 4),
            Err(SpawnError::NotEnoughSites {
                n_lefs: 4,
                needed: 7,
                n_plain: 6
            })
        );
    }

    #[test]
    fn weighted_sites_fall_back_when_hot_sites_are_full() {
        // One hot site, more LEFs than it can hold: the rest load elsewhere
        // instead of deadlocking.
        let n = 200;
        let mut weights = vec![1.0; n];
        weights[50] = 50.0;
        let lattice = Lattice::plain(n).unwrap();
        let rule = SpawnRule::WeightedSites { weights };
        let world = populate(lattice, 20, &rule, 3).expect("populated");
        assert_eq!(world.n_lefs(), 20);
    }

    #[test]
    fn weighted_sites_check_needs_enough_plain_sites() {
        let weights = vec![50.0; 6];
        let lattice = Lattice::plain(6).unwrap();
        let rule = SpawnRule::WeightedSites { weights };
        assert_eq!(
            rule.check(&lattice, 4),
            Err(SpawnError::NotEnoughSites {
                n_lefs: 4,
                needed: 7,
                n_plain: 6
            })
        );
    }

    #[test]
    fn populate_fills_a_tight_lattice() {
        // Exactly 2n - 1 plain sites is enough.
        let world = populate(Lattice::plain(7).unwrap(), 4, &SpawnRule::UniformNonCtcf, 9).unwrap();
        assert_eq!(world.n_lefs(), 4);
    }
}
