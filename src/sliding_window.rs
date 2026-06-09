use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::types::{Algorithm, RateLimitResult, RateLimiter, WindowConfig};

struct WindowState {
    timestamps: VecDeque<Instant>,
}

impl WindowState {
    fn new() -> Self {
        Self {
            timestamps: VecDeque::new(),
        }
    }

    fn prune(&mut self, window_size: Duration) {
        let cutoff = Instant::now() - window_size;
        while let Some(front) = self.timestamps.front() {
            if *front <= cutoff {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }
    }

    fn try_admit(&mut self, config: &WindowConfig, n: u64) -> RateLimitResult {
        self.prune(config.window_size);

        let current_count = self.timestamps.len() as u64;

        if current_count + n <= config.max_requests {
            let now = Instant::now();
            for _ in 0..n {
                self.timestamps.push_back(now);
            }
            let remaining = config.max_requests - self.timestamps.len() as u64;
            RateLimitResult::allowed(remaining, config.max_requests, Algorithm::SlidingWindow)
        } else {
            let retry_after = if let Some(oldest) = self.timestamps.front() {
                let age = Instant::now().duration_since(*oldest);
                config.window_size.saturating_sub(age)
            } else {
                config.window_size
            };
            let remaining = config.max_requests.saturating_sub(current_count);
            RateLimitResult::denied(
                remaining,
                config.max_requests,
                retry_after,
                Algorithm::SlidingWindow,
            )
        }
    }
}

pub struct SlidingWindow {
    config: WindowConfig,
    windows: Mutex<HashMap<String, WindowState>>,
}

impl SlidingWindow {
    pub fn new(config: WindowConfig) -> Self {
        Self {
            config,
            windows: Mutex::new(HashMap::new()),
        }
    }

    fn with_window<F>(&self, key: &str, f: F) -> RateLimitResult
    where
        F: FnOnce(&mut WindowState) -> RateLimitResult,
    {
        let mut windows = self.windows.lock().unwrap();
        let state = windows
            .entry(key.to_string())
            .or_insert_with(WindowState::new);
        f(state)
    }
}

impl RateLimiter for SlidingWindow {
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
        Algorithm::SlidingWindow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_requests_within_limit() {
        let limiter = SlidingWindow::new(WindowConfig::new(5, Duration::from_secs(1)));
        for _ in 0..5 {
            assert!(limiter.check("client1").allowed);
        }
    }

    #[test]
    fn denies_when_exceeded() {
        let limiter = SlidingWindow::new(WindowConfig::new(3, Duration::from_secs(1)));
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        let result = limiter.check("client1");
        assert!(!result.allowed);
        assert!(result.retry_after.is_some());
    }

    #[test]
    fn window_slides_over_time() {
        let limiter = SlidingWindow::new(WindowConfig::new(2, Duration::from_millis(50)));
        assert!(limiter.check("client1").allowed);
        assert!(limiter.check("client1").allowed);
        assert!(!limiter.check("client1").allowed);

        std::thread::sleep(Duration::from_millis(60));

        assert!(limiter.check("client1").allowed);
    }

    #[test]
    fn check_n_consumes_multiple() {
        let limiter = SlidingWindow::new(WindowConfig::new(10, Duration::from_secs(1)));
        let result = limiter.check_n("client1", 5);
        assert!(result.allowed);
        assert_eq!(result.remaining, 5);

        let result = limiter.check_n("client1", 6);
        assert!(!result.allowed);
    }

    #[test]
    fn independent_keys() {
        let limiter = SlidingWindow::new(WindowConfig::new(1, Duration::from_secs(1)));
        assert!(limiter.check("a").allowed);
        assert!(limiter.check("b").allowed);
        assert!(!limiter.check("a").allowed);
        assert!(!limiter.check("b").allowed);
    }

    #[test]
    fn reset_clears_window() {
        let limiter = SlidingWindow::new(WindowConfig::new(1, Duration::from_secs(60)));
        assert!(limiter.check("client1").allowed);
        assert!(!limiter.check("client1").allowed);

        limiter.reset("client1");
        assert!(limiter.check("client1").allowed);
    }

    #[test]
    fn old_requests_expire() {
        let limiter = SlidingWindow::new(WindowConfig::new(2, Duration::from_millis(30)));
        assert!(limiter.check("client1").allowed);
        std::thread::sleep(Duration::from_millis(15));
        assert!(limiter.check("client1").allowed);
        assert!(!limiter.check("client1").allowed);

        std::thread::sleep(Duration::from_millis(20));

        assert!(limiter.check("client1").allowed);
    }
}
