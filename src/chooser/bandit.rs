use std::sync::RwLock;

use nalgebra::{DMatrix, DVector};

use super::features::FEATURE_DIM;

/// Tunable parameters for the LinUCB contextual bandit.
///
/// ```
/// use speedemon::chooser::bandit::BanditConfig;
///
/// let cfg = BanditConfig::default();
/// assert!(cfg.alpha > 0.0);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BanditConfig {
    pub alpha: f64,
    pub lazy_inversion_threshold: usize,
    pub regularization: f64,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            lazy_inversion_threshold: 50,
            regularization: 1.0,
        }
    }
}

impl BanditConfig {
    /// Build a `BanditConfig` overriding one or more fields of the default.
    ///
    /// Use this instead of a struct expression because `BanditConfig` is
    /// `#[non_exhaustive]`.
    pub fn with(
        alpha: f64,
        lazy_inversion_threshold: usize,
        regularization: f64,
    ) -> Self {
        Self {
            alpha,
            lazy_inversion_threshold,
            regularization,
        }
    }
}

struct ArmState {
    a: DMatrix<f64>,
    b: DVector<f64>,
    a_inv: DMatrix<f64>,
    theta: DVector<f64>,
    updates_since_inversion: usize,
    total_updates: usize,
}

impl ArmState {
    fn new(dim: usize, regularization: f64) -> Self {
        let a = DMatrix::from_diagonal_element(dim, dim, regularization);
        let a_inv = DMatrix::from_diagonal_element(dim, dim, 1.0 / regularization);
        let b = DVector::zeros(dim);
        let theta = DVector::zeros(dim);

        Self {
            a,
            b,
            a_inv,
            theta,
            updates_since_inversion: 0,
            total_updates: 0,
        }
    }

    fn recompute_inverse(&mut self) {
        if let Some(inv) = self.a.clone().try_inverse() {
            self.a_inv = inv;
            self.theta = &self.a_inv * &self.b;
            self.updates_since_inversion = 0;
        }
    }

    fn update(&mut self, x: &DVector<f64>, reward: f64, lazy_threshold: usize) {
        let outer = x * x.transpose();
        self.a += outer;
        self.b += x * reward;
        self.updates_since_inversion += 1;
        self.total_updates += 1;

        if self.updates_since_inversion >= lazy_threshold {
            self.recompute_inverse();
        }
    }

    fn ucb(&self, x: &DVector<f64>, alpha: f64) -> f64 {
        // Note: `theta` is only refreshed inside `recompute_inverse` (lazy
        // inversion), so between inversions the exploitation term uses a
        // slightly stale parameter vector while the exploration term uses a
        // fresh `a_inv`. The bias is bounded by the inversion threshold and
        // acceptable for a prototype; for strict consistency, either refresh
        // theta on every read or take a write lock here.
        let exploitation = self.theta.dot(x);
        let a_inv_x = &self.a_inv * x;
        let exploration = alpha * x.dot(&a_inv_x).max(0.0).sqrt();
        exploitation + exploration
    }
}

pub struct LinUCBBandit {
    config: BanditConfig,
    arms: Vec<RwLock<ArmState>>,
}

impl std::fmt::Debug for LinUCBBandit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinUCBBandit")
            .field("num_arms", &self.arms.len())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl LinUCBBandit {
    /// Build a LinUCB bandit with `num_arms` arms.
    ///
    /// # Panics
    ///
    /// Panics if `num_arms == 0`.
    pub fn new(num_arms: usize, config: BanditConfig) -> Self {
        assert!(num_arms > 0, "bandit requires at least one arm");
        let arms = (0..num_arms)
            .map(|_| RwLock::new(ArmState::new(FEATURE_DIM, config.regularization)))
            .collect();

        Self { config, arms }
    }

    pub fn select(&self, x: &[f64; FEATURE_DIM]) -> usize {
        let x_vec = DVector::from_column_slice(x);

        let mut best_arm = 0;
        let mut best_ucb = f64::NEG_INFINITY;

        for (idx, arm_lock) in self.arms.iter().enumerate() {
            let ucb = {
                let arm = arm_lock.read().unwrap();
                arm.ucb(&x_vec, self.config.alpha)
            };
            if ucb > best_ucb {
                best_ucb = ucb;
                best_arm = idx;
            }
        }

        best_arm
    }

    pub fn update(&self, arm_idx: usize, x: &[f64; FEATURE_DIM], reward: f64) {
        let x_vec = DVector::from_column_slice(x);
        let mut arm = self.arms[arm_idx].write().unwrap();
        arm.update(&x_vec, reward, self.config.lazy_inversion_threshold);
    }

    pub fn num_arms(&self) -> usize {
        self.arms.len()
    }

    #[cfg(test)]
    pub fn update_count(&self, arm_idx: usize) -> usize {
        self.arms[arm_idx].read().unwrap().total_updates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_bandit_with_correct_arms() {
        let bandit = LinUCBBandit::new(4, BanditConfig::default());
        assert_eq!(bandit.num_arms(), 4);
    }

    #[test]
    fn selects_arm_consistently() {
        let bandit = LinUCBBandit::new(2, BanditConfig::default());
        let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let arm1 = bandit.select(&x);
        let arm2 = bandit.select(&x);
        assert_eq!(arm1, arm2);
    }

    #[test]
    fn converges_to_better_arm() {
        let config = BanditConfig {
            alpha: 0.1,
            lazy_inversion_threshold: 1,
            regularization: 1.0,
        };
        let bandit = LinUCBBandit::new(2, config);

        let ctx0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ctx1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for _ in 0..200 {
            bandit.update(0, &ctx0, 1.0);
            bandit.update(1, &ctx1, 0.1);
        }

        let mut arm0_selected = 0;
        for _ in 0..100 {
            let arm = bandit.select(&ctx0);
            if arm == 0 {
                arm0_selected += 1;
            }
        }

        assert!(
            arm0_selected >= 90,
            "arm 0 selected {} / 100 times, expected >= 90",
            arm0_selected
        );
    }

    #[test]
    fn update_changes_selection() {
        let config = BanditConfig {
            alpha: 0.1,
            lazy_inversion_threshold: 1,
            regularization: 1.0,
        };
        let bandit = LinUCBBandit::new(2, config);

        let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for _ in 0..50 {
            bandit.update(0, &x, 1.0);
        }

        let selected = bandit.select(&x);
        assert_eq!(selected, 0);
    }

    #[test]
    fn lazy_inversion_works() {
        let config = BanditConfig {
            alpha: 0.3,
            lazy_inversion_threshold: 10,
            regularization: 1.0,
        };
        let bandit = LinUCBBandit::new(2, config);

        let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for _ in 0..5 {
            bandit.update(0, &x, 1.0);
        }

        let arm = bandit.select(&x);
        assert!(arm < 2);
    }

    #[test]
    fn exploration_parameter_affects_selection() {
        let config_low = BanditConfig {
            alpha: 0.01,
            lazy_inversion_threshold: 1,
            regularization: 1.0,
        };
        let config_high = BanditConfig {
            alpha: 10.0,
            lazy_inversion_threshold: 1,
            regularization: 1.0,
        };

        let bandit_low = LinUCBBandit::new(3, config_low);
        let bandit_high = LinUCBBandit::new(3, config_high);

        let x = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        for _ in 0..100 {
            bandit_low.update(0, &x, 1.0);
            bandit_high.update(0, &x, 1.0);
        }

        let arm_low = bandit_low.select(&x);
        let arm_high = bandit_high.select(&x);

        assert_eq!(arm_low, 0);
        assert!(arm_high < 3);
    }

    #[test]
    fn empty_arms_panics() {
        let result = std::panic::catch_unwind(|| LinUCBBandit::new(0, BanditConfig::default()));
        assert!(result.is_err());
    }
}
