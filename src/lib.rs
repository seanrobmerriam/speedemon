//! # speedemon
//!
//! A modern, thread-safe rate limiting library for Rust featuring four
//! industry-standard algorithms with composable multi-limiter support and a
//! contextual bandit-driven algorithm chooser.
//!
//! ## Algorithms
//!
//! - [`TokenBucket`] — burst-tolerant, refills at a fixed rate.
//! - [`LeakyBucket`] — constant outflow, queue-based.
//! - [`SlidingWindow`] — rolling window of request timestamps.
//! - [`FixedWindow`] — counter per fixed time bucket.
//!
//! All four implement the [`RateLimiter`] trait and can be combined into a
//! [`RateLimiterSuite`] via the builder, or fed to [`chooser::AlgorithmChooser`]
//! to let a contextual bandit pick the best algorithm per request.
//!
//! ## Quick start — a single limiter
//!
//! ```
//! use std::time::Duration;
//! use speedemon::{TokenBucket, RateLimiter, BucketConfig};
//!
//! // 5-request burst capacity, refills 1 token / second.
//! let limiter = TokenBucket::new(BucketConfig::new(5, 1.0));
//!
//! assert!(limiter.check("client1").allowed);
//! assert!(limiter.check("client1").allowed);
//! ```
//!
//! ## Composing multiple limiters
//!
//! [`RateLimiterSuite`] applies every limiter to every key and reports the
//! most restrictive answer:
//!
//! ```
//! use std::time::Duration;
//! use speedemon::{RateLimiterSuite, RateLimiter, BucketConfig, WindowConfig};
//!
//! let suite = RateLimiterSuite::builder()
//!     .token_bucket(BucketConfig::new(100, 10.0))
//!     .fixed_window(WindowConfig::new(10, Duration::from_secs(60)))
//!     .build();
//!
//! let result = suite.check("client1");
//! assert!(result.allowed);
//! ```
//!
//! ## Adaptive algorithm selection
//!
//! The [`chooser::AlgorithmChooser`] uses a LinUCB contextual bandit to pick
//! the best limiter per [`chooser::RequestContext`], with a Tokio-based
//! observer that learns from observed decisions and (optional) false-positive
//! and false-negative signals:
//!
//! ```
//! use std::sync::Arc;
//! use speedemon::chooser::{
//!     AlgorithmChooserBuilder, RateLimiter as ChooserLimiter,
//!     TokenBucketAdapter, FixedWindowAdapter, Decision, RequestContext,
//!     ClientClass,
//! };
//! use speedemon::{BucketConfig, WindowConfig};
//! use std::time::Duration;
//!
//! # async fn doc() {
//! let (chooser, _handle) = AlgorithmChooserBuilder::new()
//!     .add_algorithm(Arc::new(TokenBucketAdapter::new(BucketConfig::new(
//!         100, 10.0,
//!     ))))
//!     .add_algorithm(Arc::new(FixedWindowAdapter::new(WindowConfig::new(
//!         100, Duration::from_secs(60),
//!     ))))
//!     .build();
//!
//! let ctx = RequestContext {
//!     client_id: 1,
//!     timestamp_ns: 0,
//!     endpoint_hash: 0,
//!     client_class: ClientClass::ApiKey,
//!     in_flight: 0,
//! };
//! assert_eq!(chooser.check(&ctx), Decision::Allow);
//! # }
//! ```
//!
//! ## Error handling
//!
//! Configuration constructors come in two flavors:
//!
//! - [`BucketConfig::try_new`] / [`WindowConfig::try_new`] return a
//!   [`Result`] so callers can handle invalid parameters.
//! - [`BucketConfig::new`] / [`WindowConfig::new`] panic on invalid input —
//!   convenient for tests and at the top of `fn main`. For `const` contexts,
//!   use [`BucketConfig::from_raw`] / [`WindowConfig::from_raw`] after
//!   validating the inputs yourself.

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
pub use types::{Algorithm, BucketConfig, ConfigError, RateLimitResult, RateLimiter, WindowConfig};

use std::sync::Arc;

/// A composition of independent rate limiters that all see every check.
///
/// On every call, the suite invokes every limiter and returns the most
/// restrictive [`RateLimitResult`] (smallest `remaining`, or the first
/// denial).
///
/// ```
/// use std::time::Duration;
/// use speedemon::{RateLimiterSuite, RateLimiter, BucketConfig, WindowConfig};
///
/// let suite = RateLimiterSuite::builder()
///     .token_bucket(BucketConfig::new(100, 10.0))
///     .fixed_window(WindowConfig::new(5, Duration::from_secs(60)))
///     .build();
///
/// // The first 5 succeed; the 6th is denied by fixed_window even though
/// // token_bucket still has plenty of headroom.
/// for _ in 0..5 {
///     assert!(suite.check("client1").allowed);
/// }
/// assert!(!suite.check("client1").allowed);
/// ```
#[derive(Debug, Default)]
pub struct RateLimiterSuite {
    limiters: Vec<Arc<dyn RateLimiter>>,
}

impl RateLimiterSuite {
    pub fn builder() -> RateLimiterSuiteBuilder {
        RateLimiterSuiteBuilder::default()
    }

    /// A suite with no limiters installed.
    ///
    /// An empty suite always reports `allowed = true` and `remaining = 0`
    /// for every check.
    pub fn new() -> Self {
        Self::default()
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

impl AsRef<[Arc<dyn RateLimiter>]> for RateLimiterSuite {
    fn as_ref(&self) -> &[Arc<dyn RateLimiter>] {
        &self.limiters
    }
}

#[derive(Debug, Default)]
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
