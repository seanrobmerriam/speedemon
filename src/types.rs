use std::fmt;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    TokenBucket,
    LeakyBucket,
    SlidingWindow,
    FixedWindow,
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Algorithm::TokenBucket => write!(f, "token_bucket"),
            Algorithm::LeakyBucket => write!(f, "leaky_bucket"),
            Algorithm::SlidingWindow => write!(f, "sliding_window"),
            Algorithm::FixedWindow => write!(f, "fixed_window"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitResult {
    pub allowed: bool,
    pub remaining: u64,
    pub limit: u64,
    pub retry_after: Option<Duration>,
    pub algorithm: Algorithm,
}

impl RateLimitResult {
    pub fn allowed(remaining: u64, limit: u64, algorithm: Algorithm) -> Self {
        Self {
            allowed: true,
            remaining,
            limit,
            retry_after: None,
            algorithm,
        }
    }

    pub fn denied(
        remaining: u64,
        limit: u64,
        retry_after: Duration,
        algorithm: Algorithm,
    ) -> Self {
        Self {
            allowed: false,
            remaining,
            limit,
            retry_after: Some(retry_after),
            algorithm,
        }
    }
}

pub trait RateLimiter: Send + Sync {
    fn check(&self, key: &str) -> RateLimitResult;

    fn check_n(&self, key: &str, n: u64) -> RateLimitResult;

    fn reset(&self, key: &str);

    fn algorithm(&self) -> Algorithm;
}

#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub capacity: u64,
    pub refill_rate: f64,
}

impl BucketConfig {
    pub fn new(capacity: u64, refill_rate: f64) -> Self {
        assert!(capacity > 0, "capacity must be greater than 0");
        assert!(refill_rate > 0.0, "refill_rate must be greater than 0.0");
        Self {
            capacity,
            refill_rate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub max_requests: u64,
    pub window_size: Duration,
}

impl WindowConfig {
    pub fn new(max_requests: u64, window_size: Duration) -> Self {
        assert!(max_requests > 0, "max_requests must be greater than 0");
        assert!(
            !window_size.is_zero(),
            "window_size must be greater than zero"
        );
        Self {
            max_requests,
            window_size,
        }
    }
}
