use std::collections::VecDeque;
use std::sync::Arc;

use dashmap::DashMap;

use super::r#trait::RequestContext;

#[cfg(test)]
use super::r#trait::ClientClass;

pub const FEATURE_DIM: usize = 7;

/// Tunable parameters for the request feature extractor.
///
/// ```
/// use speedemon::chooser::features::FeatureConfig;
///
/// let cfg = FeatureConfig::default();
/// assert!(cfg.rate_ceiling > 0.0);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FeatureConfig {
    pub rate_ceiling: f64,
    pub max_concurrency: f64,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            rate_ceiling: 1000.0,
            max_concurrency: 100.0,
        }
    }
}

impl FeatureConfig {
    /// Build a `FeatureConfig` overriding one or more fields of the default.
    ///
    /// Use this instead of a struct expression because `FeatureConfig` is
    /// `#[non_exhaustive]`.
    pub fn with(rate_ceiling: f64, max_concurrency: f64) -> Self {
        Self {
            rate_ceiling,
            max_concurrency,
        }
    }
}

#[derive(Debug)]
struct ClientState {
    timestamps: VecDeque<u64>,
}

impl ClientState {
    fn new() -> Self {
        Self {
            timestamps: VecDeque::new(),
        }
    }

    fn record(&mut self, timestamp_ns: u64) {
        self.timestamps.push_back(timestamp_ns);
        let one_sec_ago = timestamp_ns.saturating_sub(1_000_000_000);
        while let Some(front) = self.timestamps.front() {
            if *front < one_sec_ago {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }
    }

    fn rate_1s(&self, timestamp_ns: u64) -> f64 {
        let one_sec_ago = timestamp_ns.saturating_sub(1_000_000_000);
        self.timestamps
            .iter()
            .filter(|&&ts| ts >= one_sec_ago)
            .count() as f64
    }

    fn burst_coefficient(&self) -> f64 {
        if self.timestamps.len() < 10 {
            return 1.0;
        }

        let inter_arrivals: Vec<f64> = self
            .timestamps
            .iter()
            .skip(1)
            .zip(self.timestamps.iter())
            .map(|(&curr, &prev)| (curr.saturating_sub(prev)) as f64 / 1_000_000_000.0)
            .collect();

        if inter_arrivals.is_empty() {
            return 1.0;
        }

        let mean = inter_arrivals.iter().sum::<f64>() / inter_arrivals.len() as f64;
        if mean == 0.0 {
            return 1.0;
        }

        let variance = inter_arrivals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / inter_arrivals.len() as f64;

        variance / mean
    }
}

#[derive(Debug)]
pub struct FeatureExtractor {
    config: FeatureConfig,
    client_states: Arc<DashMap<u64, ClientState>>,
}

impl FeatureExtractor {
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config,
            client_states: Arc::new(DashMap::new()),
        }
    }

    pub fn extract(&self, ctx: &RequestContext) -> [f64; FEATURE_DIM] {
        let mut state = self
            .client_states
            .entry(ctx.client_id)
            .or_insert_with(ClientState::new);
        state.record(ctx.timestamp_ns);

        let rate = state.rate_1s(ctx.timestamp_ns);
        let burst = state.burst_coefficient();

        let rate_signal = (rate / self.config.rate_ceiling).min(1.0);
        let queue_pressure = (ctx.in_flight as f64 / self.config.max_concurrency).min(1.0);
        let client_class = ctx.client_class.ordinal();

        let minutes = ((ctx.timestamp_ns / 60_000_000_000) % 1440) as f64;
        let time_angle = minutes / 1440.0 * 2.0 * std::f64::consts::PI;
        let time_sin = time_angle.sin();
        let time_cos = time_angle.cos();

        // endpoint_hash is uniform-random per endpoint, so this feature is
        // effectively noise from the bandit's perspective. Replace with a
        // learned embedding or a small fixed cardinality (e.g. route group)
        // when endpoint-level learning is needed.
        let endpoint_norm = (ctx.endpoint_hash as f64) / (u64::MAX as f64);

        [
            rate_signal,
            burst,
            queue_pressure,
            client_class,
            time_sin,
            time_cos,
            endpoint_norm,
        ]
    }

    pub fn reset(&self, client_id: u64) {
        self.client_states.remove(&client_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(client_id: u64, timestamp_ns: u64) -> RequestContext {
        RequestContext {
            client_id,
            timestamp_ns,
            endpoint_hash: 0,
            client_class: ClientClass::ApiKey,
            in_flight: 0,
        }
    }

    #[test]
    fn extracts_rate_signal() {
        let extractor = FeatureExtractor::new(FeatureConfig {
            rate_ceiling: 10.0,
            max_concurrency: 100.0,
        });

        let base_ts = 1_000_000_000_000u64;
        for i in 0..5 {
            let ctx = make_ctx(1, base_ts + i * 100_000_000);
            extractor.extract(&ctx);
        }

        let ctx = make_ctx(1, base_ts + 500_000_000);
        let features = extractor.extract(&ctx);
        assert!(features[0] > 0.0);
        assert!(features[0] <= 1.0);
    }

    #[test]
    fn burst_coefficient_default_for_few_samples() {
        let extractor = FeatureExtractor::new(FeatureConfig::default());

        let ctx = make_ctx(1, 1_000_000_000_000);
        let features = extractor.extract(&ctx);
        assert_eq!(features[1], 1.0);
    }

    #[test]
    fn burst_coefficient_computed_after_10_samples() {
        let extractor = FeatureExtractor::new(FeatureConfig::default());

        let base_ts = 1_000_000_000_000u64;
        for i in 0..11 {
            let ctx = make_ctx(1, base_ts + i * 100_000_000);
            extractor.extract(&ctx);
        }

        let ctx = make_ctx(1, base_ts + 1_100_000_000);
        let features = extractor.extract(&ctx);
        assert!(features[1] >= 0.0);
    }

    #[test]
    fn burst_coefficient_is_unitless_at_high_rate() {
        let extractor = FeatureExtractor::new(FeatureConfig::default());

        let base_ts = 1_000_000_000_000u64;
        for i in 0u64..200 {
            let jitter = (i * 17) % 500;
            let ctx = make_ctx(1, base_ts + i * 1_000_000 + jitter * 1_000);
            extractor.extract(&ctx);
        }

        let ctx = make_ctx(1, base_ts + 200_000_000);
        let features = extractor.extract(&ctx);

        assert!(
            features[1] < 100.0,
            "burst coefficient {} should be O(1) for sane feature scaling, not in millions",
            features[1]
        );
    }

    #[test]
    fn queue_pressure_scales_with_in_flight() {
        let extractor = FeatureExtractor::new(FeatureConfig {
            rate_ceiling: 1000.0,
            max_concurrency: 10.0,
        });

        let ctx = RequestContext {
            client_id: 1,
            timestamp_ns: 1_000_000_000_000,
            endpoint_hash: 0,
            client_class: ClientClass::ApiKey,
            in_flight: 5,
        };
        let features = extractor.extract(&ctx);
        assert!((features[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn queue_pressure_caps_at_one() {
        let extractor = FeatureExtractor::new(FeatureConfig {
            rate_ceiling: 1000.0,
            max_concurrency: 10.0,
        });

        let ctx = RequestContext {
            client_id: 1,
            timestamp_ns: 1_000_000_000_000,
            endpoint_hash: 0,
            client_class: ClientClass::ApiKey,
            in_flight: 100,
        };
        let features = extractor.extract(&ctx);
        assert_eq!(features[2], 1.0);
    }

    #[test]
    fn client_class_encoding() {
        let extractor = FeatureExtractor::new(FeatureConfig::default());

        let ctx_api = RequestContext {
            client_id: 1,
            timestamp_ns: 1_000_000_000_000,
            endpoint_hash: 0,
            client_class: ClientClass::ApiKey,
            in_flight: 0,
        };
        let ctx_int = RequestContext {
            client_id: 2,
            timestamp_ns: 1_000_000_000_000,
            endpoint_hash: 0,
            client_class: ClientClass::Internal,
            in_flight: 0,
        };
        let ctx_anon = RequestContext {
            client_id: 3,
            timestamp_ns: 1_000_000_000_000,
            endpoint_hash: 0,
            client_class: ClientClass::Anonymous,
            in_flight: 0,
        };

        assert_eq!(extractor.extract(&ctx_api)[3], 0.0);
        assert_eq!(extractor.extract(&ctx_int)[3], 0.5);
        assert_eq!(extractor.extract(&ctx_anon)[3], 1.0);
    }

    #[test]
    fn time_encoding_wraps() {
        let extractor = FeatureExtractor::new(FeatureConfig::default());

        let ctx1 = RequestContext {
            client_id: 1,
            timestamp_ns: 0,
            endpoint_hash: 0,
            client_class: ClientClass::ApiKey,
            in_flight: 0,
        };
        let ctx2 = RequestContext {
            client_id: 2,
            timestamp_ns: 86_400_000_000_000,
            endpoint_hash: 0,
            client_class: ClientClass::ApiKey,
            in_flight: 0,
        };

        let f1 = extractor.extract(&ctx1);
        let f2 = extractor.extract(&ctx2);

        assert!((f1[4] - f2[4]).abs() < 0.01);
        assert!((f1[5] - f2[5]).abs() < 0.01);
    }

    #[test]
    fn endpoint_hash_normalized() {
        let extractor = FeatureExtractor::new(FeatureConfig::default());

        let ctx = RequestContext {
            client_id: 1,
            timestamp_ns: 0,
            endpoint_hash: u64::MAX / 2,
            client_class: ClientClass::ApiKey,
            in_flight: 0,
        };
        let features = extractor.extract(&ctx);
        assert!((features[6] - 0.5).abs() < 0.01);
    }

    #[test]
    fn independent_clients() {
        let extractor = FeatureExtractor::new(FeatureConfig::default());

        let ctx1 = make_ctx(1, 1_000_000_000_000);
        let ctx2 = make_ctx(2, 1_000_000_000_000);

        extractor.extract(&ctx1);
        extractor.extract(&ctx1);
        extractor.extract(&ctx1);

        let f1 = extractor.extract(&ctx1);
        let f2 = extractor.extract(&ctx2);

        assert!(f1[0] > f2[0]);
    }

    #[test]
    fn reset_clears_state() {
        let extractor = FeatureExtractor::new(FeatureConfig::default());

        let ctx = make_ctx(1, 1_000_000_000_000);
        extractor.extract(&ctx);
        extractor.extract(&ctx);

        extractor.reset(1);

        let f = extractor.extract(&ctx);
        assert!(f[0] <= 1.0);
    }
}
