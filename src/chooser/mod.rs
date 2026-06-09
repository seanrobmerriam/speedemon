pub mod bandit;
pub mod features;
pub mod observer;
pub mod r#trait;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;

use bandit::{BanditConfig, LinUCBBandit};
use features::{FeatureConfig, FeatureExtractor};
use observer::{spawn_observer, ObserverMessage, RewardConfig, RewardEvent};
use r#trait::{Decision, RateLimiter, RequestContext};

pub struct AlgorithmChooser {
    algorithms: Vec<Arc<dyn RateLimiter>>,
    bandit: Arc<LinUCBBandit>,
    features: Arc<FeatureExtractor>,
    reward_tx: mpsc::Sender<ObserverMessage>,
    event_counter: AtomicU64,
}

impl AlgorithmChooser {
    pub fn check(&self, ctx: &RequestContext) -> Decision {
        let x = self.features.extract(ctx);
        let arm = self.bandit.select(&x);
        let t0 = Instant::now();
        let dec = self.algorithms[arm].check(ctx);
        let lat = t0.elapsed().as_nanos() as u64;

        let event_id = self.event_counter.fetch_add(1, Ordering::Relaxed);
        let _ = self.reward_tx.try_send(ObserverMessage::Event(RewardEvent {
            event_id,
            arm_idx: arm,
            context: x,
            decision: dec,
            latency_ns: lat,
            false_positive: None,
            false_negative: None,
        }));

        dec
    }

    pub fn signal_false_positive(&self, event_id: u64, is_fp: bool) {
        let _ = self.reward_tx.try_send(ObserverMessage::SignalFalsePositive {
            event_id,
            is_fp,
        });
    }

    pub fn signal_false_negative(&self, event_id: u64, is_fn: bool) {
        let _ = self.reward_tx.try_send(ObserverMessage::SignalFalseNegative {
            event_id,
            is_fn,
        });
    }

    pub fn bandit(&self) -> &Arc<LinUCBBandit> {
        &self.bandit
    }

    pub fn features(&self) -> &Arc<FeatureExtractor> {
        &self.features
    }

    pub fn num_algorithms(&self) -> usize {
        self.algorithms.len()
    }

    pub fn algorithm_name(&self, idx: usize) -> Option<&'static str> {
        self.algorithms.get(idx).map(|a| a.name())
    }
}

pub struct AlgorithmChooserBuilder {
    algorithms: Vec<Arc<dyn RateLimiter>>,
    bandit_config: BanditConfig,
    feature_config: FeatureConfig,
    reward_config: RewardConfig,
}

impl AlgorithmChooserBuilder {
    pub fn new() -> Self {
        Self {
            algorithms: Vec::new(),
            bandit_config: BanditConfig::default(),
            feature_config: FeatureConfig::default(),
            reward_config: RewardConfig::default(),
        }
    }

    pub fn add_algorithm(mut self, limiter: Arc<dyn RateLimiter>) -> Self {
        self.algorithms.push(limiter);
        self
    }

    pub fn bandit_config(mut self, config: BanditConfig) -> Self {
        self.bandit_config = config;
        self
    }

    pub fn feature_config(mut self, config: FeatureConfig) -> Self {
        self.feature_config = config;
        self
    }

    pub fn reward_config(mut self, config: RewardConfig) -> Self {
        self.reward_config = config;
        self
    }

    pub fn build(self) -> (AlgorithmChooser, tokio::task::JoinHandle<()>) {
        assert!(
            !self.algorithms.is_empty(),
            "at least one algorithm required"
        );

        let bandit = Arc::new(LinUCBBandit::new(self.algorithms.len(), self.bandit_config));
        let features = Arc::new(FeatureExtractor::new(self.feature_config));

        let (tx, rx) = mpsc::channel(10000);
        let handle = spawn_observer(bandit.clone(), self.reward_config, rx);

        let chooser = AlgorithmChooser {
            algorithms: self.algorithms,
            bandit,
            features,
            reward_tx: tx,
            event_counter: AtomicU64::new(0),
        };

        (chooser, handle)
    }
}

impl Default for AlgorithmChooserBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BucketConfig, WindowConfig};
    use std::time::Duration;
    use r#trait::{
        ClientClass, FixedWindowAdapter, LeakyBucketAdapter, SlidingWindowAdapter,
        TokenBucketAdapter,
    };

    fn make_ctx(client_id: u64) -> RequestContext {
        RequestContext {
            client_id,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            endpoint_hash: 0,
            client_class: ClientClass::ApiKey,
            in_flight: 0,
        }
    }

    #[tokio::test]
    async fn chooser_with_single_algorithm() {
        let (chooser, handle) = AlgorithmChooserBuilder::new()
            .add_algorithm(Arc::new(TokenBucketAdapter::new(BucketConfig::new(
                10, 1.0,
            ))))
            .build();

        let ctx = make_ctx(1);
        let dec = chooser.check(&ctx);
        assert_eq!(dec, Decision::Allow);

        drop(chooser);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn chooser_with_multiple_algorithms() {
        let (chooser, handle) = AlgorithmChooserBuilder::new()
            .add_algorithm(Arc::new(TokenBucketAdapter::new(BucketConfig::new(
                100, 10.0,
            ))))
            .add_algorithm(Arc::new(LeakyBucketAdapter::new(BucketConfig::new(
                100, 10.0,
            ))))
            .add_algorithm(Arc::new(SlidingWindowAdapter::new(WindowConfig::new(
                100,
                Duration::from_secs(1),
            ))))
            .add_algorithm(Arc::new(FixedWindowAdapter::new(WindowConfig::new(
                100,
                Duration::from_secs(1),
            ))))
            .build();

        assert_eq!(chooser.num_algorithms(), 4);

        for i in 0..50 {
            let ctx = make_ctx(i);
            let dec = chooser.check(&ctx);
            assert_eq!(dec, Decision::Allow);
        }

        drop(chooser);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn chooser_bandit_learns() {
        // Asymmetric setup: token_bucket allows generously (goodput reward
        // ~1.0); fixed_window with capacity 1 over 60s throttles almost
        // every subsequent request for the same client (goodput ~0.0).
        // After enough events, the bandit should overwhelmingly prefer
        // arm 0 (token_bucket).
        let config = BanditConfig {
            alpha: 0.05,
            lazy_inversion_threshold: 1,
            regularization: 1.0,
        };

        let (chooser, handle) = AlgorithmChooserBuilder::new()
            .add_algorithm(Arc::new(TokenBucketAdapter::new(BucketConfig::new(
                1000, 1000.0,
            ))))
            .add_algorithm(Arc::new(FixedWindowAdapter::new(WindowConfig::new(
                1,
                Duration::from_secs(60),
            ))))
            .bandit_config(config)
            .build();

        for i in 0..200u64 {
            let ctx = make_ctx(i % 10);
            chooser.check(&ctx);
        }

        tokio::time::sleep(Duration::from_millis(100)).await;

        let mut arm0_count = 0;
        let probes = 50;
        for i in 200..(200 + probes) {
            let ctx = make_ctx(i);
            let x = chooser.features().extract(&ctx);
            let arm = chooser.bandit().select(&x);
            if arm == 0 {
                arm0_count += 1;
            }
        }

        assert!(
            arm0_count >= probes * 4 / 5,
            "expected arm 0 (token_bucket) selected >= 80% after training, got {}/{}",
            arm0_count,
            probes
        );

        drop(chooser);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn chooser_algorithm_names() {
        let (chooser, handle) = AlgorithmChooserBuilder::new()
            .add_algorithm(Arc::new(TokenBucketAdapter::new(BucketConfig::new(
                10, 1.0,
            ))))
            .add_algorithm(Arc::new(LeakyBucketAdapter::new(BucketConfig::new(
                10, 1.0,
            ))))
            .build();

        assert_eq!(chooser.algorithm_name(0), Some("token_bucket"));
        assert_eq!(chooser.algorithm_name(1), Some("leaky_bucket"));
        assert_eq!(chooser.algorithm_name(2), None);

        drop(chooser);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn chooser_signal_false_positive() {
        let (chooser, handle) = AlgorithmChooserBuilder::new()
            .add_algorithm(Arc::new(TokenBucketAdapter::new(BucketConfig::new(
                10, 1.0,
            ))))
            .build();

        let ctx = make_ctx(1);
        chooser.check(&ctx);

        chooser.signal_false_positive(0, true);

        tokio::time::sleep(Duration::from_millis(10)).await;

        drop(chooser);
        let _ = handle.await;
    }

    #[tokio::test]
    async fn chooser_signal_false_negative() {
        let (chooser, handle) = AlgorithmChooserBuilder::new()
            .add_algorithm(Arc::new(TokenBucketAdapter::new(BucketConfig::new(
                10, 1.0,
            ))))
            .build();

        let ctx = make_ctx(1);
        chooser.check(&ctx);

        chooser.signal_false_negative(0, true);

        tokio::time::sleep(Duration::from_millis(10)).await;

        drop(chooser);
        let _ = handle.await;
    }
}
