//! Spawn rules: where a LEF lands when it (re)spawns.
//!
//! [`Spawn`] is what a rule can do; [`SpawnRule`] lists the built-in rules as
//! plain data; [`WeightedSpawn`] adds preferential loading sites; [`populate`]
//! uses a rule to build a starting world.

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
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "rule", rename_all = "snake_case"))]
pub enum SpawnRule {
    /// Uniform among `Plain` sites that no leg occupies.
    #[default]
    UniformNonCtcf,
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
        }
    }
}

/// Why a [`WeightedSpawn`] can't be built.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum WeightsError {
    /// The loading ratio must be finite and at least 1.
    BadRatio(f64),
    /// A loading site is off the lattice, or on a CTCF site (which never spawns).
    BadSite(Position),
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightsError::BadRatio(r) => write!(f, "loading ratio must be finite and >= 1, got {r}"),
            WeightsError::BadSite(s) => write!(f, "loading site {s} is off the lattice or a CTCF site"),
        }
    }
}

impl std::error::Error for WeightsError {}

/// A spawn rule with preferential loading sites (e.g. NIPBL loading in the
/// user's M5_v3 model): every plain site has weight 1, and each site in
/// `sites` has weight `ratio` (50 reproduces the 0.05-vs-0.001 preference).
/// Sampling draws a weighted site over the plain ones and redraws while it is
/// occupied, so the placement *rate* is unchanged — only *where* LEFs land.
///
/// Not a [`SpawnRule`], as it carries data; a `Sim` takes it through
/// [`Sim::with_spawn_extrusion_from`](crate::Sim::with_spawn_extrusion_from).
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedSpawn {
    /// Per-site weight: 0 where no LEF can land, 1 on plain sites, `ratio` on
    /// loading sites.
    weights: Vec<f64>,
    /// Sum of `weights`; the draw range.
    total: f64,
}

impl WeightedSpawn {
    /// Plain sites of a lattice of `n_sites` (CTCFs at `ctcf_sites`) get
    /// weight 1; each site in `sites` gets weight `ratio`.
    pub fn loading(
        n_sites: u32,
        ctcf_sites: &[Position],
        sites: &[Position],
        ratio: f64,
    ) -> Result<Self, WeightsError> {
        if !(ratio.is_finite() && ratio >= 1.0) {
            return Err(WeightsError::BadRatio(ratio));
        }
        let mut weights = vec![1.0; n_sites as usize];
        for &c in ctcf_sites {
            if let Some(w) = weights.get_mut(c as usize) {
                *w = 0.0;
            }
        }
        let mut total = weights.iter().sum();
        for &s in sites {
            let w = weights.get_mut(s as usize).ok_or(WeightsError::BadSite(s))?;
            if *w == 0.0 {
                return Err(WeightsError::BadSite(s));
            }
            if *w == 1.0 {
                // First time this site is listed; duplicates keep the ratio.
                total += ratio - 1.0;
                *w = ratio;
            }
        }
        Ok(Self { weights, total })
    }
}

impl Spawn for WeightedSpawn {
    fn sample(&self, world: &World, rng: &mut SimRng) -> Position {
        loop {
            let mut u = rng.gen_range(0.0..self.total);
            for (i, &w) in self.weights.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                if u < w {
                    if !world.occupied(i as Position) {
                        return i as Position;
                    }
                    break; // occupied: redraw
                }
                u -= w;
            }
            // Falling off the end (rounding) also redraws.
        }
    }

    fn check(&self, lattice: &Lattice, n_lefs: usize) -> Result<(), SpawnError> {
        // As UniformNonCtcf, but only plain sites with a positive weight count.
        let needed = (2 * n_lefs).saturating_sub(1);
        let n_plain = lattice
            .sites
            .iter()
            .zip(self.weights.iter().chain(std::iter::repeat(&0.0)))
            .filter(|(s, w)| matches!(s.kind, SiteKind::Plain) && **w > 0.0)
            .count();
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
    fn weighted_spawn_fills_loading_sites_first_reproducibly() {
        // 1000 plain sites, 10 of them loading at 50x: each draw lands on a
        // loading site with probability 500 / (990 + 500) ~= 0.34, so the 10
        // loading sites are snapped up within the first placements.
        let lattice = Lattice::plain(1000).unwrap();
        let sites: Vec<Position> = (100..110).collect();
        let rule = WeightedSpawn::loading(1000, &[], &sites, 50.0).unwrap();
        let world = populate(lattice.clone(), 200, &rule, 3).unwrap();
        let hits = world
            .lef_states()
            .left_pos
            .iter()
            .filter(|&&p| (100..110).contains(&p))
            .count();
        assert_eq!(hits, 10, "every loading site should hold a LEF");

        let again = populate(lattice.clone(), 200, &rule, 3).unwrap();
        assert_eq!(again, world);
        let other = populate(lattice, 200, &rule, 4).unwrap();
        assert_ne!(other, world);
    }

    #[test]
    fn weighted_spawn_validates_and_checks_capacity() {
        let ctcf = Ctcf {
            site: 5,
            captures_from: Direction::Left,
            strength: 1.0,
        };
        let lattice = Lattice::with_ctcfs(10, &[ctcf]).unwrap();
        assert_eq!(
            WeightedSpawn::loading(10, &[5], &[99], 50.0),
            Err(WeightsError::BadSite(99))
        );
        assert_eq!(
            WeightedSpawn::loading(10, &[5], &[5], 50.0),
            Err(WeightsError::BadSite(5))
        );
        assert_eq!(
            WeightedSpawn::loading(10, &[5], &[2], 0.5).map(|_| ()),
            Err(WeightsError::BadRatio(0.5))
        );
        // 9 spawnable sites: 5 LEFs need 9 (just fits), 6 need 11.
        let rule = WeightedSpawn::loading(10, &[5], &[], 50.0).unwrap();
        assert_eq!(rule.check(&lattice, 5), Ok(()));
        assert_eq!(
            rule.check(&lattice, 6),
            Err(SpawnError::NotEnoughSites {
                n_lefs: 6,
                needed: 11,
                n_plain: 9
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
    fn populate_fills_a_tight_lattice() {
        // Exactly 2n - 1 plain sites is enough.
        let world = populate(Lattice::plain(7).unwrap(), 4, &SpawnRule::UniformNonCtcf, 9).unwrap();
        assert_eq!(world.n_lefs(), 4);
    }
}
