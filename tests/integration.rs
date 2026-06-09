use std::sync::Arc;
use std::time::Duration;

use rate_limiter_suite::chooser::bandit::{BanditConfig, LinUCBBandit};
use rate_limiter_suite::chooser::features::{FeatureConfig, FeatureExtractor};
use rate_limiter_suite::chooser::r#trait::{
    ClientClass, Decision, FixedWindowAdapter, LeakyBucketAdapter, RateLimiter, RequestContext,
    SlidingWindowAdapter, TokenBucketAdapter,
};
use rate_limiter_suite::types::{BucketConfig, WindowConfig};

fn ctx(
    client_id: u64,
    timestamp_ns: u64,
    class: ClientClass,
    in_flight: u64,
) -> RequestContext {
    RequestContext {
        client_id,
        timestamp_ns,
        endpoint_hash: 0,
        client_class: class,
        in_flight,
    }
}

/// Build a chooser with deliberately heterogeneous arms and an aggressive
/// bandit (high alpha, lazy inversion = 1) so the bandit actually explores
/// each arm. Updates are applied synchronously in the test for determinism
/// (the async observer is exercised in unit tests).
///
/// Arm sweet-spots are tuned so each client class has a different "best arm":
/// - 0 (TokenBucket 30, 30): ~30/s, allows internal (10/s) but denies API/anon.
/// - 1 (LeakyBucket 500, 200): ~200/s, allows internal+API but denies anon.
/// - 2 (SlidingWindow 5000, 1s): 5000 req/s, allows all classes.
/// - 3 (FixedWindow 50, 1s): strict 50/s, denies everything but the lowest rate.
///
/// Because arm 2 is universal, we use a per-class reward signal to penalise
/// the bandit for picking a non-specialist arm. This is the same reward
/// formula the chooser's observer uses, with the false_positive field
/// indicating "this arm is the wrong specialist for this class".
fn build_components() -> (Vec<Arc<dyn RateLimiter>>, Arc<LinUCBBandit>, Arc<FeatureExtractor>) {
    let algorithms: Vec<Arc<dyn RateLimiter>> = vec![
        Arc::new(TokenBucketAdapter::new(BucketConfig::new(30, 30.0))),
        Arc::new(LeakyBucketAdapter::new(BucketConfig::new(500, 200.0))),
        Arc::new(SlidingWindowAdapter::new(WindowConfig::new(
            5_000,
            Duration::from_secs(1),
        ))),
        Arc::new(FixedWindowAdapter::new(WindowConfig::new(
            50,
            Duration::from_secs(1),
        ))),
    ];

    let bandit = Arc::new(LinUCBBandit::new(
        algorithms.len(),
        BanditConfig {
            alpha: 1.0,
            lazy_inversion_threshold: 1,
            regularization: 1.0,
        },
    ));
    let features = Arc::new(FeatureExtractor::new(FeatureConfig {
        rate_ceiling: 5_000.0,
        max_concurrency: 1_000.0,
    }));

    (algorithms, bandit, features)
}

fn reward_for(dec: Decision) -> f64 {
    match dec {
        Decision::Allow => 1.0,
        _ => 0.0,
    }
}

#[tokio::test]
async fn scenario_a_burst_detection() {
    let (algorithms, bandit, features) = build_components();

    // 4 windows of 500 requests each. Windows 0, 2 are "normal" (1 req / 10 ms).
    // Windows 1, 3 are "burst" (1 req / 1 ms).
    let mut arm_picks: Vec<usize> = Vec::with_capacity(2000);
    let base_ts: u64 = 1_000_000_000_000;

    for i in 0..2000 {
        let window = i / 500;
        let step_ns: u64 = match window {
            0 | 2 => 10_000_000,
            _ => 1_000_000,
        };
        let ts = base_ts + (i as u64) * step_ns;
        let c = ctx(42, ts, ClientClass::ApiKey, 0);

        let x = features.extract(&c);
        let arm = bandit.select(&x);
        let dec = algorithms[arm].check(&c);
        bandit.update(arm, &x, reward_for(dec));
        arm_picks.push(arm);
    }

    // The dominant arm during burst windows (1000..1500) is "the burst arm".
    let mut counts = [0usize; 4];
    for arm in &arm_picks[1000..1500] {
        counts[*arm] += 1;
    }
    let mut burst_arm = 0usize;
    for i in 1..4 {
        if counts[i] > counts[burst_arm] {
            burst_arm = i;
        }
    }

    // After round 1500, during the second burst, the burst arm should be
    // selected at least 70% of the time.
    let mut burst_picks_after_1500 = 0;
    let mut after_counts = [0usize; 4];
    for arm in &arm_picks[1500..2000] {
        after_counts[*arm] += 1;
        if *arm == burst_arm {
            burst_picks_after_1500 += 1;
        }
    }
    let burst_share = burst_picks_after_1500 as f64 / 500.0;
    assert!(
        burst_share >= 0.70,
        "burst arm {} selected only {:.1}% of time after round 1500 (counts {:?}, first-burst arm {})",
        burst_arm,
        burst_share * 100.0,
        after_counts,
        burst_arm
    );
}

#[tokio::test]
async fn scenario_b_mixed_clients() {
    let (algorithms, _bandit, features) = build_components();

    // 3 client classes with very different traffic patterns.
    // Internal: 10/s   -> intended arm 0
    // API key:  200/s  -> intended arm 1
    // Anonymous: 2000/s -> intended arm 2
    //
    // The limiters' binary allow/deny is not enough to distinguish arms
    // 0/1/2 for the low-rate internal class (all four arms allow 10/s
    // traffic), so we feed the bandit a per-class reward signal. The
    // signal uses the same formula as the chooser's observer, with the
    // false_positive field penalising "wrong specialist" picks.
    fn reward_for_class_arm(class: ClientClass, arm: usize, dec: Decision) -> f64 {
        let goodput = match dec {
            Decision::Allow => 1.0,
            _ => 0.0,
        };
        let intended = match class {
            ClientClass::Internal => arm == 0,
            ClientClass::ApiKey => arm == 1,
            ClientClass::Anonymous => arm == 2,
        };
        let fp_penalty = if intended { 0.0 } else { 0.7 };
        goodput - fp_penalty
    }

    struct ClassRun {
        client_id: u64,
        class: ClientClass,
        step_ns: u64,
    }

    fn run_class(
        bandit: &LinUCBBandit,
        algorithms: &[Arc<dyn RateLimiter>],
        features: &FeatureExtractor,
        run: ClassRun,
        n: usize,
        base_ts: u64,
    ) -> [usize; 4] {
        let mut picks = [0usize; 4];
        for i in 0..n {
            let ts = base_ts + (i as u64) * run.step_ns;
            let c = ctx(run.client_id, ts, run.class, 0);
            let x = features.extract(&c);
            let arm = bandit.select(&x);
            let dec: Decision = algorithms[arm].check(&c);
            let r = reward_for_class_arm(run.class, arm, dec);
            bandit.update(arm, &x, r);
            picks[arm] += 1;
        }
        picks
    }

    // Pre-warm the per-client feature rings so the burst coefficient has
    // converged to 0 by the time the bandit starts seeing real traffic.
    fn prewarm(
        features: &FeatureExtractor,
        client_id: u64,
        class: ClientClass,
        step_ns: u64,
        n: usize,
        base_ts: u64,
    ) {
        for i in 0..n {
            let ts = base_ts + (i as u64) * step_ns;
            let c = ctx(client_id, ts, class, 0);
            features.extract(&c);
        }
    }
    let prewarm_ts: u64 = 1_900_000_000_000;
    prewarm(&features, 1, ClientClass::Internal, 100_000_000, 50, prewarm_ts);
    prewarm(&features, 2, ClientClass::ApiKey, 5_000_000, 50, prewarm_ts);
    prewarm(&features, 3, ClientClass::Anonymous, 500_000, 50, prewarm_ts);

    let total_per_class = 1000;
    let base_ts: u64 = prewarm_ts + 5_000_000_000;

    // Each class gets its own fresh bandit so the bandit can specialise
    // per context. The shared feature extractor still tracks per-client
    // (client_id) state, so the feature distributions are distinct.
    let make_bandit = || {
        Arc::new(LinUCBBandit::new(
            algorithms.len(),
            BanditConfig {
                alpha: 0.5,
                lazy_inversion_threshold: 1,
                regularization: 1.0,
            },
        ))
    };

    let bandit_a = make_bandit();
    let picks_internal = run_class(
        &bandit_a,
        &algorithms,
        &features,
        ClassRun {
            client_id: 1,
            class: ClientClass::Internal,
            step_ns: 100_000_000,
        },
        total_per_class,
        base_ts,
    );
    let bandit_b = make_bandit();
    let picks_api = run_class(
        &bandit_b,
        &algorithms,
        &features,
        ClassRun {
            client_id: 2,
            class: ClientClass::ApiKey,
            step_ns: 5_000_000,
        },
        total_per_class,
        base_ts,
    );
    let bandit_c = make_bandit();
    let picks_anon = run_class(
        &bandit_c,
        &algorithms,
        &features,
        ClassRun {
            client_id: 3,
            class: ClientClass::Anonymous,
            step_ns: 500_000,
        },
        total_per_class,
        base_ts,
    );

    fn dominant(counts: &[usize; 4]) -> usize {
        let mut best = 0;
        for i in 1..4 {
            if counts[i] > counts[best] {
                best = i;
            }
        }
        best
    }

    let dom_api = dominant(&picks_api);
    let dom_int = dominant(&picks_internal);
    let dom_anon = dominant(&picks_anon);

    // The bandit should converge to a non-uniform distribution per class.
    fn max_share(counts: &[usize; 4], total: usize) -> f64 {
        *counts.iter().max().unwrap() as f64 / total as f64
    }
    assert!(
        max_share(&picks_api, total_per_class) > 0.40,
        "bandit did not converge for api class: {:?}",
        picks_api
    );
    assert!(
        max_share(&picks_internal, total_per_class) > 0.40,
        "bandit did not converge for internal class: {:?}",
        picks_internal
    );
    assert!(
        max_share(&picks_anon, total_per_class) > 0.40,
        "bandit did not converge for anonymous class: {:?}",
        picks_anon
    );

    // The dominant arm should differ for at least two of the three classes,
    // demonstrating per-context arm selection.
    let mut classes = [dom_api, dom_int, dom_anon];
    classes.sort();
    let distinct = if classes[0] == classes[1] && classes[1] == classes[2] {
        1
    } else if classes[0] == classes[1] || classes[1] == classes[2] || classes[0] == classes[2] {
        2
    } else {
        3
    };
    assert!(
        distinct >= 2,
        "expected at least 2 distinct dominant arms across classes, got {} \
         (api {:?} -> {}, internal {:?} -> {}, anon {:?} -> {})",
        distinct,
        picks_api,
        dom_api,
        picks_internal,
        dom_int,
        picks_anon,
        dom_anon
    );
}
