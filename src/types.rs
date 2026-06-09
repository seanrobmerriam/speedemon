use std::fmt;
use std::time::Duration;

use thiserror::Error;

/// Errors returned by [`BucketConfig::try_new`] and [`WindowConfig::try_new`]
/// when the supplied parameters are not usable.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ConfigError {
    /// A capacity / `max_requests` of zero was supplied.
    #[error("`{field}` must be greater than 0")]
    ZeroCapacity {
        /// Name of the offending field, e.g. `"capacity"` or `"max_requests"`.
        field: &'static str,
    },
    /// A refill rate of zero was supplied.
    #[error("`refill_rate` must be greater than 0")]
    ZeroRefillRate,
    /// A `refill_rate` was not a finite number.
    #[error("`refill_rate` must be finite (got {value})")]
    NonFiniteRate {
        /// The value that was supplied.
        value: f64,
    },
    /// A window size of zero was supplied.
    #[error("`window_size` must be greater than zero")]
    ZeroWindowSize,
}

/// Identifies the rate-limiting algorithm that produced a [`RateLimitResult`].
///
/// ```
/// use speedemon::Algorithm;
/// assert_eq!(Algorithm::TokenBucket.to_string(), "token_bucket");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
#[non_exhaustive]
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

/// A thread-safe, per-key rate limiter.
///
/// Implementors are expected to be cheap to share (`Send + Sync`) and to
/// maintain per-key state internally.
///
/// ```
/// use speedemon::{TokenBucket, RateLimiter, BucketConfig};
///
/// let limiter = TokenBucket::new(BucketConfig::new(2, 1.0));
/// let r = limiter.check("client1");
/// assert!(r.allowed);
/// assert_eq!(r.remaining, 1);
/// ```
pub trait RateLimiter: Send + Sync + std::fmt::Debug {
    /// Check whether one request for `key` is allowed. Convenience wrapper
    /// around [`RateLimiter::check_n`] with `n = 1`.
    fn check(&self, key: &str) -> RateLimitResult;

    /// Check whether `n` requests for `key` are allowed. On denial the
    /// returned [`RateLimitResult::retry_after`] indicates how long the
    /// caller should wait before retrying.
    fn check_n(&self, key: &str, n: u64) -> RateLimitResult;

    /// Drop all state for `key`, allowing it to start fresh on the next
    /// call.
    fn reset(&self, key: &str);

    /// The algorithm implemented by this limiter. Used by the chooser and
    /// surfaced on [`RateLimitResult`].
    fn algorithm(&self) -> Algorithm;
}

/// Configuration for the token-bucket and leaky-bucket algorithms.
///
/// Construct via [`BucketConfig::try_new`] (returns a [`Result`]) or
/// [`BucketConfig::new`] (panics on invalid input).
///
/// ```
/// use speedemon::BucketConfig;
///
/// // 5-request burst, refills 2 tokens per second.
/// let cfg = BucketConfig::new(5, 2.0);
/// assert_eq!(cfg.capacity, 5);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BucketConfig {
    pub capacity: u64,
    pub refill_rate: f64,
}

impl BucketConfig {
    /// Build a [`BucketConfig`] without validation.
    ///
    /// Callers are responsible for ensuring `capacity > 0` and
    /// `refill_rate > 0` and finite. Prefer [`BucketConfig::try_new`] or
    /// [`BucketConfig::new`] unless you have already validated the inputs.
    pub const fn from_raw(capacity: u64, refill_rate: f64) -> Self {
        Self {
            capacity,
            refill_rate,
        }
    }

    /// Build a [`BucketConfig`], returning a [`ConfigError`] on invalid input.
    pub fn try_new(capacity: u64, refill_rate: f64) -> Result<Self, ConfigError> {
        if capacity == 0 {
            return Err(ConfigError::ZeroCapacity { field: "capacity" });
        }
        if refill_rate == 0.0 {
            return Err(ConfigError::ZeroRefillRate);
        }
        if !refill_rate.is_finite() {
            return Err(ConfigError::NonFiniteRate { value: refill_rate });
        }
        Ok(Self {
            capacity,
            refill_rate,
        })
    }

    /// Build a [`BucketConfig`], panicking on invalid input.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`, `refill_rate <= 0.0`, or `refill_rate` is
    /// not finite.
    pub fn new(capacity: u64, refill_rate: f64) -> Self {
        Self::try_new(capacity, refill_rate)
            .expect("BucketConfig::new: invalid parameters")
    }
}

/// Configuration for the fixed-window and sliding-window algorithms.
///
/// ```
/// use std::time::Duration;
/// use speedemon::WindowConfig;
///
/// let cfg = WindowConfig::new(100, Duration::from_secs(60));
/// assert_eq!(cfg.max_requests, 100);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WindowConfig {
    pub max_requests: u64,
    pub window_size: Duration,
}

impl WindowConfig {
    pub const fn from_raw(max_requests: u64, window_size: Duration) -> Self {
        Self {
            max_requests,
            window_size,
        }
    }

    pub fn try_new(max_requests: u64, window_size: Duration) -> Result<Self, ConfigError> {
        if max_requests == 0 {
            return Err(ConfigError::ZeroCapacity {
                field: "max_requests",
            });
        }
        if window_size.is_zero() {
            return Err(ConfigError::ZeroWindowSize);
        }
        Ok(Self {
            max_requests,
            window_size,
        })
    }

    /// Build a [`WindowConfig`], panicking on invalid input.
    ///
    /// # Panics
    ///
    /// Panics if `max_requests == 0` or `window_size` is zero.
    pub fn new(max_requests: u64, window_size: Duration) -> Self {
        Self::try_new(max_requests, window_size).expect("WindowConfig::new: invalid parameters")
    }
}

