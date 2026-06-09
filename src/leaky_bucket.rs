use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::types::{Algorithm, BucketConfig, RateLimitResult, RateLimiter};

struct LeakyState {
    queue: VecDeque<Instant>,
    last_leak: Instant,
}

impl LeakyState {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            last_leak: Instant::now(),
        }
    }

    fn leak(&mut self, config: &BucketConfig) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_leak).as_secs_f64();
        let leaked = (elapsed * config.refill_rate) as u64;

        if leaked > 0 {
            let drain_count = leaked.min(self.queue.len() as u64) as usize;
            self.queue.drain(..drain_count);
            self.last_leak = now;
        }
    }

    fn try_enqueue(&mut self, config: &BucketConfig, n: u64) -> RateLimitResult {
        self.leak(config);

        let available = config.capacity.saturating_sub(self.queue.len() as u64);

        if n <= available {
            let now = Instant::now();
            for _ in 0..n {
                self.queue.push_back(now);
            }
            let remaining = config.capacity - self.queue.len() as u64;
            RateLimitResult::allowed(remaining, config.capacity, Algorithm::LeakyBucket)
        } else {
            let retry_after = if self.queue.is_empty() {
                Duration::from_secs_f64(n as f64 / config.refill_rate)
            } else {
                let oldest = self.queue.front().unwrap();
                let age = Instant::now().duration_since(*oldest);
                let time_to_drain =
                    Duration::from_secs_f64(self.queue.len() as f64 / config.refill_rate);
                time_to_drain.saturating_sub(age)
            };
            let remaining = config.capacity.saturating_sub(self.queue.len() as u64);
            RateLimitResult::denied(
                remaining,
                config.capacity,
                retry_after,
                Algorithm::LeakyBucket,
            )
        }
    }
}

pub struct LeakyBucket {
    config: BucketConfig,
    buckets: Mutex<HashMap<String, LeakyState>>,
}

impl LeakyBucket {
    pub fn new(config: BucketConfig) -> Self {
        Self {
            config,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    fn with_bucket<F>(&self, key: &str, f: F) -> RateLimitResult
    where
        F: FnOnce(&mut LeakyState) -> RateLimitResult,
    {
        let mut buckets = self.buckets.lock().unwrap();
        let state = buckets
            .entry(key.to_string())
            .or_insert_with(LeakyState::new);
        f(state)
    }
}

impl RateLimiter for LeakyBucket {
    fn check(&self, key: &str) -> RateLimitResult {
        self.check_n(key, 1)
    }

    fn check_n(&self, key: &str, n: u64) -> RateLimitResult {
        self.with_bucket(key, |state| state.try_enqueue(&self.config, n))
    }

    fn reset(&self, key: &str) {
        let mut buckets = self.buckets.lock().unwrap();
        buckets.remove(key);
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::LeakyBucket
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_requests_within_capacity() {
        let limiter = LeakyBucket::new(BucketConfig::new(5, 1.0));
        for _ in 0..5 {
            assert!(limiter.check("client1").allowed);
        }
    }

    #[test]
    fn denies_when_full() {
        let limiter = LeakyBucket::new(BucketConfig::new(3, 1.0));
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        let result = limiter.check("client1");
        assert!(!result.allowed);
        assert!(result.retry_after.is_some());
    }

    #[test]
    fn leaks_over_time() {
        let limiter = LeakyBucket::new(BucketConfig::new(2, 100.0));
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        assert!(!limiter.check("client1").allowed);

        std::thread::sleep(Duration::from_millis(20));

        assert!(limiter.check("client1").allowed);
    }

    #[test]
    fn check_n_consumes_multiple() {
        let limiter = LeakyBucket::new(BucketConfig::new(10, 1.0));
        let result = limiter.check_n("client1", 5);
        assert!(result.allowed);
        assert_eq!(result.remaining, 5);

        let result = limiter.check_n("client1", 6);
        assert!(!result.allowed);
    }

    #[test]
    fn independent_keys() {
        let limiter = LeakyBucket::new(BucketConfig::new(1, 1.0));
        assert!(limiter.check("a").allowed);
        assert!(limiter.check("b").allowed);
        assert!(!limiter.check("a").allowed);
        assert!(!limiter.check("b").allowed);
    }

    #[test]
    fn reset_clears_state() {
        let limiter = LeakyBucket::new(BucketConfig::new(1, 0.1));
        assert!(limiter.check("client1").allowed);
        assert!(!limiter.check("client1").allowed);

        limiter.reset("client1");
        assert!(limiter.check("client1").allowed);
    }

    #[test]
    fn constant_rate_no_burst() {
        let limiter = LeakyBucket::new(BucketConfig::new(10, 10.0));
        let result = limiter.check_n("client1", 10);
        assert!(result.allowed);
        assert_eq!(result.remaining, 0);

        let result = limiter.check("client1");
        assert!(!result.allowed);
    }
}
