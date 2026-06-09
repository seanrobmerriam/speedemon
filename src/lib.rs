pub mod chooser;
pub mod fixed_window;
pub mod leaky_bucket;
pub mod sliding_window;
pub mod token_bucket;
pub mod types;

pub use fixed_window::FixedWindow;
pub use leaky_bucket::LeakyBucket;
pub use sliding_window::SlidingWindow;
pub use token_bucket::TokenBucket;
pub use types::{Algorithm, BucketConfig, RateLimitResult, RateLimiter, WindowConfig};

use std::sync::Arc;

pub struct RateLimiterSuite {
    limiters: Vec<Arc<dyn RateLimiter>>,
}

impl RateLimiterSuite {
    pub fn builder() -> RateLimiterSuiteBuilder {
        RateLimiterSuiteBuilder {
            limiters: Vec::new(),
        }
    }

    pub fn check(&self, key: &str) -> RateLimitResult {
        self.check_n(key, 1)
    }

    pub fn check_n(&self, key: &str, n: u64) -> RateLimitResult {
        let mut most_restrictive: Option<RateLimitResult> = None;

        for limiter in &self.limiters {
            let result = limiter.check_n(key, n);

            if !result.allowed {
                return result;
            }

            match &most_restrictive {
                Some(current) if result.remaining < current.remaining => {
                    most_restrictive = Some(result);
                }
                None => most_restrictive = Some(result),
                _ => {}
            }
        }

        most_restrictive.unwrap_or(RateLimitResult::allowed(0, 0, Algorithm::TokenBucket))
    }

    pub fn reset(&self, key: &str) {
        for limiter in &self.limiters {
            limiter.reset(key);
        }
    }

    pub fn limiters(&self) -> &[Arc<dyn RateLimiter>] {
        &self.limiters
    }
}

pub struct RateLimiterSuiteBuilder {
    limiters: Vec<Arc<dyn RateLimiter>>,
}

impl RateLimiterSuiteBuilder {
    pub fn token_bucket(mut self, config: BucketConfig) -> Self {
        self.limiters.push(Arc::new(TokenBucket::new(config)));
        self
    }

    pub fn leaky_bucket(mut self, config: BucketConfig) -> Self {
        self.limiters.push(Arc::new(LeakyBucket::new(config)));
        self
    }

    pub fn sliding_window(mut self, config: WindowConfig) -> Self {
        self.limiters.push(Arc::new(SlidingWindow::new(config)));
        self
    }

    pub fn fixed_window(mut self, config: WindowConfig) -> Self {
        self.limiters.push(Arc::new(FixedWindow::new(config)));
        self
    }

    pub fn add_limiter(mut self, limiter: Arc<dyn RateLimiter>) -> Self {
        self.limiters.push(limiter);
        self
    }

    pub fn build(self) -> RateLimiterSuite {
        RateLimiterSuite {
            limiters: self.limiters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn suite_with_single_limiter() {
        let suite = RateLimiterSuite::builder()
            .token_bucket(BucketConfig::new(3, 1.0))
            .build();

        assert!(suite.check("client1").allowed);
        assert!(suite.check("client1").allowed);
        assert!(suite.check("client1").allowed);
        assert!(!suite.check("client1").allowed);
    }

    #[test]
    fn suite_most_restrictive_wins() {
        let suite = RateLimiterSuite::builder()
            .token_bucket(BucketConfig::new(10, 1.0))
            .fixed_window(WindowConfig::new(3, Duration::from_secs(60)))
            .build();

        assert!(suite.check("client1").allowed);
        assert!(suite.check("client1").allowed);
        assert!(suite.check("client1").allowed);

        let result = suite.check("client1");
        assert!(!result.allowed);
        assert_eq!(result.algorithm, Algorithm::FixedWindow);
    }

    #[test]
    fn suite_resets_all_limiters() {
        let suite = RateLimiterSuite::builder()
            .token_bucket(BucketConfig::new(2, 0.1))
            .fixed_window(WindowConfig::new(2, Duration::from_secs(60)))
            .build();

        assert!(suite.check("client1").allowed);
        assert!(suite.check("client1").allowed);
        assert!(!suite.check("client1").allowed);

        suite.reset("client1");

        assert!(suite.check("client1").allowed);
        assert!(suite.check("client1").allowed);
    }

    #[test]
    fn suite_all_four_algorithms() {
        let suite = RateLimiterSuite::builder()
            .token_bucket(BucketConfig::new(100, 10.0))
            .leaky_bucket(BucketConfig::new(100, 10.0))
            .sliding_window(WindowConfig::new(100, Duration::from_secs(1)))
            .fixed_window(WindowConfig::new(100, Duration::from_secs(1)))
            .build();

        assert_eq!(suite.limiters().len(), 4);

        for _ in 0..50 {
            assert!(suite.check("client1").allowed);
        }
    }

    #[test]
    fn suite_check_n_across_limiters() {
        let suite = RateLimiterSuite::builder()
            .token_bucket(BucketConfig::new(20, 1.0))
            .fixed_window(WindowConfig::new(10, Duration::from_secs(60)))
            .build();

        let result = suite.check_n("client1", 5);
        assert!(result.allowed);

        let result = suite.check_n("client1", 6);
        assert!(!result.allowed);
    }

    #[test]
    fn suite_empty_returns_default() {
        let suite = RateLimiterSuite::builder().build();
        let result = suite.check("client1");
        assert!(result.allowed);
    }
}
