use crate::fixed_window::FixedWindow;
use crate::leaky_bucket::LeakyBucket;
use crate::sliding_window::SlidingWindow;
use crate::token_bucket::TokenBucket;
use crate::types::{BucketConfig, RateLimiter as LegacyRateLimiter, WindowConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientClass {
    ApiKey,
    Internal,
    Anonymous,
}

impl ClientClass {
    pub fn ordinal(&self) -> f64 {
        match self {
            ClientClass::ApiKey => 0.0,
            ClientClass::Internal => 0.5,
            ClientClass::Anonymous => 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestContext {
    pub client_id: u64,
    pub timestamp_ns: u64,
    pub endpoint_hash: u64,
    pub client_class: ClientClass,
    pub in_flight: u64,
}

impl RequestContext {
    pub fn key(&self) -> String {
        format!("client_{}", self.client_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
    Throttle { delay_ms: u64 },
}

impl Decision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Decision::Allow)
    }
}

pub trait RateLimiter: Send + Sync {
    fn check(&self, ctx: &RequestContext) -> Decision;
    fn name(&self) -> &'static str;
}

pub struct TokenBucketAdapter {
    inner: TokenBucket,
}

impl TokenBucketAdapter {
    pub fn new(config: BucketConfig) -> Self {
        Self {
            inner: TokenBucket::new(config),
        }
    }
}

impl RateLimiter for TokenBucketAdapter {
    fn check(&self, ctx: &RequestContext) -> Decision {
        let result = self.inner.check(&ctx.key());
        map_result(result, "token_bucket")
    }

    fn name(&self) -> &'static str {
        "token_bucket"
    }
}

pub struct LeakyBucketAdapter {
    inner: LeakyBucket,
}

impl LeakyBucketAdapter {
    pub fn new(config: BucketConfig) -> Self {
        Self {
            inner: LeakyBucket::new(config),
        }
    }
}

impl RateLimiter for LeakyBucketAdapter {
    fn check(&self, ctx: &RequestContext) -> Decision {
        let result = self.inner.check(&ctx.key());
        map_result(result, "leaky_bucket")
    }

    fn name(&self) -> &'static str {
        "leaky_bucket"
    }
}

pub struct SlidingWindowAdapter {
    inner: SlidingWindow,
}

impl SlidingWindowAdapter {
    pub fn new(config: WindowConfig) -> Self {
        Self {
            inner: SlidingWindow::new(config),
        }
    }
}

impl RateLimiter for SlidingWindowAdapter {
    fn check(&self, ctx: &RequestContext) -> Decision {
        let result = self.inner.check(&ctx.key());
        map_result(result, "sliding_window")
    }

    fn name(&self) -> &'static str {
        "sliding_window"
    }
}

pub struct FixedWindowAdapter {
    inner: FixedWindow,
}

impl FixedWindowAdapter {
    pub fn new(config: WindowConfig) -> Self {
        Self {
            inner: FixedWindow::new(config),
        }
    }
}

impl RateLimiter for FixedWindowAdapter {
    fn check(&self, ctx: &RequestContext) -> Decision {
        let result = self.inner.check(&ctx.key());
        map_result(result, "fixed_window")
    }

    fn name(&self) -> &'static str {
        "fixed_window"
    }
}

#[inline]
fn map_result(
    result: crate::types::RateLimitResult,
    _name: &'static str,
) -> Decision {
    if result.allowed {
        Decision::Allow
    } else if let Some(retry) = result.retry_after {
        Decision::Throttle {
            delay_ms: retry.as_millis() as u64,
        }
    } else {
        Decision::Deny
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_ctx(client_id: u64) -> RequestContext {
        RequestContext {
            client_id,
            timestamp_ns: 0,
            endpoint_hash: 0,
            client_class: ClientClass::ApiKey,
            in_flight: 0,
        }
    }

    #[test]
    fn token_bucket_adapter_allows() {
        let adapter = TokenBucketAdapter::new(BucketConfig::new(5, 1.0));
        let ctx = make_ctx(1);
        assert_eq!(adapter.check(&ctx), Decision::Allow);
        assert_eq!(adapter.name(), "token_bucket");
    }

    #[test]
    fn token_bucket_adapter_throttles() {
        let adapter = TokenBucketAdapter::new(BucketConfig::new(1, 1.0));
        let ctx = make_ctx(1);
        assert_eq!(adapter.check(&ctx), Decision::Allow);
        match adapter.check(&ctx) {
            Decision::Throttle { delay_ms } => assert!(delay_ms > 0),
            other => panic!("expected Throttle, got {:?}", other),
        }
    }

    #[test]
    fn leaky_bucket_adapter_allows() {
        let adapter = LeakyBucketAdapter::new(BucketConfig::new(5, 1.0));
        let ctx = make_ctx(1);
        assert_eq!(adapter.check(&ctx), Decision::Allow);
        assert_eq!(adapter.name(), "leaky_bucket");
    }

    #[test]
    fn sliding_window_adapter_allows() {
        let adapter = SlidingWindowAdapter::new(WindowConfig::new(5, Duration::from_secs(1)));
        let ctx = make_ctx(1);
        assert_eq!(adapter.check(&ctx), Decision::Allow);
        assert_eq!(adapter.name(), "sliding_window");
    }

    #[test]
    fn fixed_window_adapter_allows() {
        let adapter = FixedWindowAdapter::new(WindowConfig::new(5, Duration::from_secs(1)));
        let ctx = make_ctx(1);
        assert_eq!(adapter.check(&ctx), Decision::Allow);
        assert_eq!(adapter.name(), "fixed_window");
    }

    #[test]
    fn client_class_ordinal() {
        assert_eq!(ClientClass::ApiKey.ordinal(), 0.0);
        assert_eq!(ClientClass::Internal.ordinal(), 0.5);
        assert_eq!(ClientClass::Anonymous.ordinal(), 1.0);
    }
}
