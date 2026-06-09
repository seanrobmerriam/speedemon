use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::types::{Algorithm, RateLimitResult, RateLimiter, WindowConfig};

#[derive(Debug)]
struct FixedWindowState {
    count: u64,
    window_start: Instant,
}

impl FixedWindowState {
    fn new() -> Self {
        Self {
            count: 0,
            window_start: Instant::now(),
        }
    }

    fn maybe_reset(&mut self, window_size: Duration) {
        let elapsed = self.window_start.elapsed();
        if elapsed >= window_size {
            self.count = 0;
            self.window_start = Instant::now();
        }
    }

    fn try_admit(&mut self, config: &WindowConfig, n: u64) -> RateLimitResult {
        self.maybe_reset(config.window_size);

        if self.count + n <= config.max_requests {
            self.count += n;
            let remaining = config.max_requests - self.count;
            RateLimitResult::allowed(remaining, config.max_requests, Algorithm::FixedWindow)
        } else {
            let elapsed = self.window_start.elapsed();
            let retry_after = config.window_size.saturating_sub(elapsed);
            let remaining = config.max_requests.saturating_sub(self.count);
            RateLimitResult::denied(
                remaining,
                config.max_requests,
                retry_after,
                Algorithm::FixedWindow,
            )
        }
    }
}

/// Fixed-window counter rate limiter.
///
/// Each key has a counter that resets at the start of each `window_size`
/// interval. Allows up to `max_requests` per window.
///
/// ```
/// use std::time::Duration;
/// use speedemon::{FixedWindow, RateLimiter, WindowConfig};
///
/// let limiter = FixedWindow::new(WindowConfig::new(2, Duration::from_secs(60)));
/// assert!(limiter.check("client1").allowed);
/// assert!(limiter.check("client1").allowed);
/// assert!(!limiter.check("client1").allowed);
/// ```
#[derive(Debug)]
pub struct FixedWindow {
    config: WindowConfig,
    windows: Mutex<HashMap<String, FixedWindowState>>,
}

impl FixedWindow {
    pub fn new(config: WindowConfig) -> Self {
        Self {
            config,
            windows: Mutex::new(HashMap::new()),
        }
    }

    fn with_window<F>(&self, key: &str, f: F) -> RateLimitResult
    where
        F: FnOnce(&mut FixedWindowState) -> RateLimitResult,
    {
        let mut windows = self.windows.lock().unwrap();
        let state = windows
            .entry(key.to_string())
            .or_insert_with(FixedWindowState::new);
        f(state)
    }
}

impl RateLimiter for FixedWindow {
    fn check(&self, key: &str) -> RateLimitResult {
        self.check_n(key, 1)
    }

    fn check_n(&self, key: &str, n: u64) -> RateLimitResult {
        self.with_window(key, |state| state.try_admit(&self.config, n))
    }

    fn reset(&self, key: &str) {
        let mut windows = self.windows.lock().unwrap();
        windows.remove(key);
    }

    fn algorithm(&self) -> Algorithm {
        Algorithm::FixedWindow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_requests_within_limit() {
        let limiter = FixedWindow::new(WindowConfig::new(5, Duration::from_secs(1)));
        for _ in 0..5 {
            assert!(limiter.check("client1").allowed);
        }
    }

    #[test]
    fn denies_when_exceeded() {
        let limiter = FixedWindow::new(WindowConfig::new(3, Duration::from_secs(1)));
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        let result = limiter.check("client1");
        assert!(!result.allowed);
        assert!(result.retry_after.is_some());
    }

    #[test]
    fn resets_after_window() {
        let limiter = FixedWindow::new(WindowConfig::new(2, Duration::from_millis(50)));
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        assert!(!limiter.check("client1").allowed);

        std::thread::sleep(Duration::from_millis(60));

        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
    }

    #[test]
    fn check_n_consumes_multiple() {
        let limiter = FixedWindow::new(WindowConfig::new(10, Duration::from_secs(1)));
        let result = limiter.check_n("client1", 5);
        assert!(result.allowed);
        assert_eq!(result.remaining, 5);

        let result = limiter.check_n("client1", 6);
        assert!(!result.allowed);
    }

    #[test]
    fn independent_keys() {
        let limiter = FixedWindow::new(WindowConfig::new(1, Duration::from_secs(1)));
        assert!(limiter.check("a").allowed);
        assert!(limiter.check("b").allowed);
        assert!(!limiter.check("a").allowed);
        assert!(!limiter.check("b").allowed);
    }

    #[test]
    fn reset_clears_window() {
        let limiter = FixedWindow::new(WindowConfig::new(1, Duration::from_secs(60)));
        assert!(limiter.check("client1").allowed);
        assert!(!limiter.check("client1").allowed);

        limiter.reset("client1");
        assert!(limiter.check("client1").allowed);
    }

    #[test]
    fn remaining_decrements_correctly() {
        let limiter = FixedWindow::new(WindowConfig::new(5, Duration::from_secs(1)));
        let r1 = limiter.check("client1");
        assert_eq!(r1.remaining, 4);
        let r2 = limiter.check("client1");
        assert_eq!(r2.remaining, 3);
    }
}
