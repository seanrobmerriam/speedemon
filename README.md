# Rate Limiter Suite

A modern, thread-safe rate limiting library for Rust featuring four industry-standard algorithms with composable multi-limiter support.

## Features

- **Four Rate Limiting Algorithms**
  - Token Bucket — allows controlled bursts with sustained refill
  - Leaky Bucket — smooths traffic at a constant rate
  - Sliding Window — precise per-request tracking with rolling windows
  - Fixed Window — efficient time-bucketed counting
- **Composable Suite** — combine multiple limiters; most restrictive wins
- **AlgorithmChooser** — a LinUCB contextual bandit that picks the best
  algorithm per request from a pool of limiters and learns from
  outcomes fed back asynchronously; 4 µs p99 end-to-end
- **Thread-Safe** — all limiters and the chooser implement `Send + Sync`
- **Per-Key Isolation** — independent rate limits for each client/tenant/IP
- **Informative Results** — every check returns remaining quota, limits, and retry-after duration

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rate-limiter-suite = "0.1.0"
```

## Quick Start

```rust
use rate_limiter_suite::{RateLimiterSuite, BucketConfig, WindowConfig};
use std::time::Duration;

fn main() {
    let suite = RateLimiterSuite::builder()
        .token_bucket(BucketConfig::new(100, 10.0))  // 100 burst, 10 req/s refill
        .fixed_window(WindowConfig::new(1000, Duration::from_secs(3600)))  // 1000 req/hour
        .build();

    let result = suite.check("client_abc");

    if result.allowed {
        println!("Request allowed, {} remaining", result.remaining);
    } else {
        println!("Rate limited by {:?}", result.algorithm);
        if let Some(retry) = result.retry_after {
            println!("Retry after {:?}", retry);
        }
    }
}
```

## Algorithms

### Token Bucket

Allows controlled bursts up to bucket capacity with continuous token refill.

```rust
use rate_limiter_suite::{TokenBucket, BucketConfig, RateLimiter};

let limiter = TokenBucket::new(BucketConfig::new(
    100,   // capacity (burst size)
    10.0,  // refill rate (tokens per second)
));

let result = limiter.check("client1");
```

**Use case**: API rate limiting with burst tolerance

### Leaky Bucket

Processes requests at a constant rate, smoothing traffic spikes.

```rust
use rate_limiter_suite::{LeakyBucket, BucketConfig, RateLimiter};

let limiter = LeakyBucket::new(BucketConfig::new(
    50,   // queue capacity
    5.0,  // drain rate (requests per second)
));

let result = limiter.check("client1");
```

**Use case**: Outbound traffic shaping, protecting downstream services

### Sliding Window

Tracks individual request timestamps for precise rolling window enforcement.

```rust
use rate_limiter_suite::{SlidingWindow, WindowConfig, RateLimiter};
use std::time::Duration;

let limiter = SlidingWindow::new(WindowConfig::new(
    100,  // max requests
    Duration::from_secs(60),  // window size
));

let result = limiter.check("client1");
```

**Use case**: Strict per-second fairness, high-accuracy limits

### Fixed Window

Counts requests within fixed time buckets (e.g., per minute, per hour).

```rust
use rate_limiter_suite::{FixedWindow, WindowConfig, RateLimiter};
use std::time::Duration;

let limiter = FixedWindow::new(WindowConfig::new(
    1000,  // max requests
    Duration::from_secs(3600),  // 1 hour window
));

let result = limiter.check("client1");
```

**Use case**: Daily/hourly quotas, coarse-grained limits

## Composing Multiple Limiters

The `RateLimiterSuite` evaluates requests against all configured limiters. The most restrictive result wins:

```rust
use rate_limiter_suite::{RateLimiterSuite, BucketConfig, WindowConfig};
use std::time::Duration;

let suite = RateLimiterSuite::builder()
    // Per-client burst limit
    .token_bucket(BucketConfig::new(100, 10.0))
    // Per-client hourly quota
    .fixed_window(WindowConfig::new(1000, Duration::from_secs(3600)))
    // Global sliding window
    .sliding_window(WindowConfig::new(10000, Duration::from_secs(60)))
    .build();

let result = suite.check("client_abc");

if !result.allowed {
    println!("Blocked by: {:?}", result.algorithm);
}
```

## API Reference

### Core Trait

```rust
pub trait RateLimiter: Send + Sync {
    fn check(&self, key: &str) -> RateLimitResult;
    fn check_n(&self, key: &str, n: u64) -> RateLimitResult;
    fn reset(&self, key: &str);
    fn algorithm(&self) -> Algorithm;
}
```

### RateLimitResult

```rust
pub struct RateLimitResult {
    pub allowed: bool,
    pub remaining: u64,
    pub limit: u64,
    pub retry_after: Option<Duration>,
    pub algorithm: Algorithm,
}
```

### Configuration

**BucketConfig** (Token Bucket, Leaky Bucket):
- `capacity: u64` — maximum tokens/queue size
- `refill_rate: f64` — tokens added per second

**WindowConfig** (Sliding Window, Fixed Window):
- `max_requests: u64` — maximum requests per window
- `window_size: Duration` — window duration

## Advanced Usage

### Consuming Multiple Tokens

```rust
let result = limiter.check_n("client1", 5);  // consume 5 tokens
```

### Resetting Limits

```rust
limiter.reset("client1");  // clear all state for this key
suite.reset("client1");    // reset across all limiters
```

### Accessing Individual Limiters

```rust
let suite = RateLimiterSuite::builder()
    .token_bucket(BucketConfig::new(100, 10.0))
    .build();

for limiter in suite.limiters() {
    println!("Algorithm: {:?}", limiter.algorithm());
}
```

## Testing

Run the test suite:

```bash
cargo test
```

Run with coverage:

```bash
cargo tarpaulin
```

All tests pass, including:
- Algorithm-specific behavior tests
- Time-based refill/drain verification
- Multi-limiter composition
- Thread safety under concurrent load
- LinUCB bandit convergence (`converges_to_better_arm`)
- Feature extractor unit tests
- Reward observer (event processing, deferred false-positive signals, TTL expiry)
- End-to-end integration tests (`scenario_a_burst_detection`, `scenario_b_mixed_clients`)

Run the criterion benchmarks:

```bash
cargo bench
```

## AlgorithmChooser (contextual bandit)

The `AlgorithmChooser` is a meta-layer that picks the best legacy limiter
per request using a **disjoint LinUCB contextual bandit** and learns from
outcomes fed back asynchronously by a background reward observer.

### Why?

Different traffic patterns are best served by different algorithms. A
bursty client hits the token bucket's burst capacity; a steady client
gets cleaner enforcement from a leaky bucket. The chooser learns which
arm yields the highest reward for each context and routes accordingly.

### Quick start

```rust
use std::sync::Arc;
use rate_limiter_suite::chooser::r#trait::{
    TokenBucketAdapter, LeakyBucketAdapter, SlidingWindowAdapter, FixedWindowAdapter,
    ClientClass, RequestContext,
};
use rate_limiter_suite::chooser::{AlgorithmChooserBuilder, Decision};
use rate_limiter_suite::types::{BucketConfig, WindowConfig};
use std::time::Duration;

let (chooser, _handle) = AlgorithmChooserBuilder::new()
    .add_algorithm(Arc::new(TokenBucketAdapter::new(BucketConfig::new(100, 10.0))))
    .add_algorithm(Arc::new(LeakyBucketAdapter::new(BucketConfig::new(100, 10.0))))
    .add_algorithm(Arc::new(SlidingWindowAdapter::new(
        WindowConfig::new(1000, Duration::from_secs(60))
    )))