use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::types::{Algorithm, BucketConfig, RateLimitResult, RateLimiter};

#[derive(Debug)]
struct BucketState {
    tokens: f64,
    last_refill: Instant,
}

impl BucketState {
    fn new(capacity: u64) -> Self {
        Self {
            tokens: capacity as f64,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self, config: &BucketConfig) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * config.refill_rate;
        self.tokens = (self.tokens + new_tokens).min(config.capacity as f64);
        self.last_refill = now;
    }

    fn try_consume(&mut self, config: &BucketConfig, n: u64) -> RateLimitResult {
        self.refill(config);

        if self.tokens >= n as f64 {
            self.tokens -= n as f64;
            RateLimitResult::allowed(
                self.tokens.floor() as u64,
                config.capacity,
                Algorithm::TokenBucket,
            )
        } else {
            let deficit = n as f64 - self.tokens;
            let retry_after = Duration::from_secs_f64(deficit / config.refill_rate);
            RateLimitResult::denied(
                self.tokens.floor() as u64,
                config.capacity,
                retry_after,
                Algorithm::TokenBucket,
            )
        }
    }
}

/// Token-bucket rate limiter.
///
/// Allows bursts up to `capacity`, refilling at `refill_rate` tokens per
/// second. Each successful check decrements one token; denied checks do not
/// consume tokens.
///
/// ```
/// use speedemon::{TokenBucket, RateLimiter, BucketConfig};
///
/// let limiter = TokenBucket::new(BucketConfig::new(3, 1.0));
/// for _ in 0..3 {
///     assert!(limiter.check("client1").allowed);
/// }
/// assert!(!limiter.check("client1").allowed);
/// ```
#[derive(Debug)]
pub struct TokenBucket {
    config: BucketConfig,
    buckets: Mutex<HashMap<String, BucketState>>,
}

impl TokenBucket {
    pub fn new(config: BucketConfig) -> Self {
        Self {
            config,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    fn with_bucket<F>(&self, key: &str, f: F) -> RateLimitResult
    where
        F: FnOnce(&mut BucketState) -> RateLimitResult,
    {
        let mut buckets = self.buckets.lock().unwrap();
        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| BucketState::new(self.config.capacity));
        f(bucket)
    }
}

impl RateLimiter for TokenBucket {
    fn check(&self, key: &str) -> RateLimitResult {
        self.check_n(key, 1)
    }

    fn check_n(&self, key: &str, n: u64) -> RateLimitResult {
        self.with_bucket(key, |bucket| bucket.try_consume(&self.config, n))
    }

    fn reset(&self, key: &str) {
        let mut buckets = self.buckets.lock().unwrap();
        buckets.remove(key);
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::TokenBucket
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_requests_within_capacity() {
        let limiter = TokenBucket::new(BucketConfig::new(5, 1.0));
        for _ in 0..5 {
            assert!(limiter.check("client1").allowed);
        }
    }

    #[test]
    fn denies_when_exhausted() {
        let limiter = TokenBucket::new(BucketConfig::new(2, 1.0));
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        let result = limiter.check("client1");
        assert!(!result.allowed);
        assert!(result.retry_after.is_some());
    }

    #[test]
    fn refills_over_time() {
        let limiter = TokenBucket::new(BucketConfig::new(2, 100.0));
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        assert!(!limiter.check("client1").allowed);

        std::thread::sleep(Duration::from_millis(20));

        assert!(limiter.check("client1").allowed);
    }

    #[test]
    fn check_n_consumes_multiple() {
        let limiter = TokenBucket::new(BucketConfig::new(10, 1.0));
        let result = limiter.check_n("client1", 5);
        assert!(result.allowed);
        assert_eq!(result.remaining, 5);

        let result = limiter.check_n("client1", 6);
        assert!(!result.allowed);
    }

    #[test]
    fn independent_keys() {
        let limiter = TokenBucket::new(BucketConfig::new(1, 1.0));
        assert!(limiter.check("a").allowed);
        assert!(limiter.check("b").allowed);
        assert!(!limiter.check("a").allowed);
        assert!(!limiter.check("b").allowed);
    }

    #[test]
    fn reset_clears_bucket() {
        let limiter = TokenBucket::new(BucketConfig::new(1, 0.1));
        assert!(limiter.check("client1").allowed);
        assert!(!limiter.check("client1").allowed);

        limiter.reset("client1");
        assert!(limiter.check("client1").allowed);
    }

    #[test]
    fn burst_respects_capacity() {
        let limiter = TokenBucket::new(BucketConfig::new(10, 5.0));
        let result = limiter.check_n("client1", 10);
        assert!(result.allowed);
        assert_eq!(result.remaining, 0);
    }

    #[test]
    fn retry_after_is_positive_on_denial() {
        let limiter = TokenBucket::new(BucketConfig::new(1, 2.0));
        assert!(limiter.check("client1").allowed);
        let result = limiter.check("client1");
        assert!(!result.allowed);
        assert!(result.retry_after.unwrap() > Duration::ZERO);
    }
}
