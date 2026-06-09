use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use rate_limiter_suite::chooser::bandit::{BanditConfig, LinUCBBandit};
use rate_limiter_suite::chooser::features::{FeatureConfig, FeatureExtractor, FEATURE_DIM};
use rate_limiter_suite::chooser::r#trait::{
    ClientClass, Decision, FixedWindowAdapter, LeakyBucketAdapter, RateLimiter, RequestContext,
    SlidingWindowAdapter, TokenBucketAdapter,
};
use rate_limiter_suite::chooser::AlgorithmChooserBuilder;
use rate_limiter_suite::types::{BucketConfig, WindowConfig};

fn warm_request_context(client_id: u64, ts_ns: u64) -> RequestContext {
    RequestContext {
        client_id,
        timestamp_ns: ts_ns,
        endpoint_hash: 0xCAFEBABEu64,
        client_class: ClientClass::ApiKey,
        in_flight: 4,
    }
}

/// Build the chooser on a dedicated tokio runtime so the observer task can
/// be spawned, then drop the runtime handle (the observer task keeps running
/// on a worker thread for the lifetime of the chooser).
fn build_warm_chooser() -> rate_limiter_suite::chooser::AlgorithmChooser {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("tokio runtime");
    let _guard = rt.enter();

    let algorithms: Vec<Arc<dyn RateLimiter>> = vec![
        Arc::new(TokenBucketAdapter::new(BucketConfig::new(1_000_000, 1_000.0))),
        Arc::new(LeakyBucketAdapter::new(BucketConfig::new(1_000_000, 1_000.0))),
        Arc::new(SlidingWindowAdapter::new(WindowConfig::new(
            1_000_000,
            Duration::from_secs(60),
        ))),
        Arc::new(FixedWindowAdapter::new(WindowConfig::new(
            1_000_000,
            Duration::from_secs(60),
        ))),
    ];

    let config = BanditConfig {
        alpha: 0.3,
        lazy_inversion_threshold: 50,
        regularization: 1.0,
    };

    let (chooser, _handle) = rt.block_on(async {
        let (c, h) = AlgorithmChooserBuilder::new()
            .add_algorithm(algorithms[0].clone())
            .add_algorithm(algorithms[1].clone())
            .add_algorithm(algorithms[2].clone())
            .add_algorithm(algorithms[3].clone())
            .bandit_config(config)
            .feature_config(FeatureConfig {
                rate_ceiling: 1_000_000.0,
                max_concurrency: 1_000_000.0,
            })
            .build();
        (c, h)
    });

    // Warm up the bandit so the inverse is cached and the per-client ring
    // is populated.
    for i in 0..2_000u64 {
        let ctx = warm_request_context(7, 1_000_000_000_000 + i * 1_000_000);
        let _: Decision = chooser.check(&ctx);
    }
    chooser
}

fn chooser_hot_path(c: &mut Criterion) {
    let chooser = build_warm_chooser();
    let mut i: u64 = 0;
    c.bench_function("chooser_hot_path", |b| {
        b.iter(|| {
            i += 1;
            let ctx = warm_request_context(7, 2_000_000_000_000 + i * 1_000_000);
            std::hint::black_box(chooser.check(&ctx))
        })
    });

    // Custom p99 measurement: run 10_000 iterations and report p50/p99/p999.
    let mut samples: Vec<u128> = Vec::with_capacity(10_000);
    for k in 0..10_000u64 {
        let ctx = warm_request_context(7, 3_000_000_000_000 + k * 1_000_000);
        let start = std::time::Instant::now();
        let _ = chooser.check(&ctx);
        samples.push(start.elapsed().as_nanos());
    }
    samples.sort_unstable();
    let p = |q: f64| -> u128 {
        let idx = ((samples.len() as f64 - 1.0) * q) as usize;
        samples[idx]
    };
    eprintln!(
        "chooser_hot_path percentiles (ns): p50={} p90={} p99={} p999={} max={}",
        p(0.50),
        p(0.90),
        p(0.99),
        p(0.999),
        *samples.last().unwrap()
    );
}

fn feature_extraction(c: &mut Criterion) {
    let extractor = FeatureExtractor::new(FeatureConfig {
        rate_ceiling: 1_000_000.0,
        max_concurrency: 1_000.0,
    });
    // Populate the per-client ring once.
    for i in 0..20u64 {
        let _ = extractor.extract(&warm_request_context(7, 1_000_000_000_000 + i * 1_000_000));
    }
    let mut i: u64 = 0;
    c.bench_function("feature_extraction", |b| {
        b.iter(|| {
            i += 1;
            let ctx = warm_request_context(7, 1_000_000_000_000 + i * 1_000_000);
            std::hint::black_box(extractor.extract(&ctx))
        })
    });
}

fn bandit_select(c: &mut Criterion) {
    let bandit = LinUCBBandit::new(
        4,
        BanditConfig {
            alpha: 0.3,
            lazy_inversion_threshold: 50,
            regularization: 1.0,
        },
    );
    // Warm up so the inverse is cached.
    for _ in 0..200 {
        let x: [f64; FEATURE_DIM] = [0.05, 0.1, 0.0, 0.5, 0.3, 0.4, 0.1];
        bandit.update(0, &x, 1.0);
        bandit.update(1, &x, 0.3);
        bandit.update(2, &x, 0.7);
        bandit.update(3, &x, -0.2);
    }
    let x: [f64; FEATURE_DIM] = [0.05, 0.1, 0.0, 0.5, 0.3, 0.4, 0.1];
    c.bench_function("bandit_select", |b| {
        b.iter(|| std::hint::black_box(bandit.select(&x)))
    });
}

criterion_group!(benches, chooser_hot_path, feature_extraction, bandit_select);
criterion_main!(benches);
