use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use super::bandit::LinUCBBandit;
use super::features::FEATURE_DIM;
use super::r#trait::Decision;

#[derive(Debug, Clone)]
pub struct RewardConfig {
    pub goodput_weight: f64,
    pub false_positive_weight: f64,
    pub latency_weight: f64,
    pub latency_ceiling_ns: f64,
    pub event_ttl: Duration,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            goodput_weight: 1.0,
            false_positive_weight: 2.0,
            latency_weight: 0.1,
            latency_ceiling_ns: 10_000_000.0,
            event_ttl: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RewardEvent {
    pub event_id: u64,
    pub arm_idx: usize,
    pub context: [f64; FEATURE_DIM],
    pub decision: Decision,
    pub latency_ns: u64,
    pub false_positive: Option<bool>,
}

impl RewardEvent {
    pub fn compute_reward(&self, config: &RewardConfig) -> f64 {
        let goodput = if self.decision.is_allowed() { 1.0 } else { 0.0 };
        let fp = self.false_positive.unwrap_or(false) as i32 as f64;
        let latency_penalty = self.latency_ns as f64 / config.latency_ceiling_ns;

        config.goodput_weight * goodput
            - config.false_positive_weight * fp
            - config.latency_weight * latency_penalty
    }
}

struct PendingEvent {
    event: RewardEvent,
    created_at: Instant,
}

pub struct RewardObserver {
    bandit: Arc<LinUCBBandit>,
    config: RewardConfig,
    pending: HashMap<u64, PendingEvent>,
}

impl RewardObserver {
    fn new(bandit: Arc<LinUCBBandit>, config: RewardConfig) -> Self {
        Self {
            bandit,
            config,
            pending: HashMap::new(),
        }
    }

    fn process_event(&mut self, event: RewardEvent) {
        let reward = event.compute_reward(&self.config);
        self.bandit.update(event.arm_idx, &event.context, reward);
        self.pending.insert(
            event.event_id,
            PendingEvent {
                event,
                created_at: Instant::now(),
            },
        );
    }

    fn signal_false_positive(&mut self, event_id: u64, is_fp: bool) {
        if let Some(pending) = self.pending.remove(&event_id) {
            let mut event = pending.event;
            event.false_positive = Some(is_fp);
            let reward = event.compute_reward(&self.config);
            self.bandit.update(event.arm_idx, &event.context, reward);
        }
    }

    fn expire_old_events(&mut self) {
        let now = Instant::now();
        let expired: Vec<u64> = self
            .pending
            .iter()
            .filter(|(_, p)| now.duration_since(p.created_at) >= self.config.event_ttl)
            .map(|(id, _)| *id)
            .collect();

        for event_id in expired {
            if let Some(pending) = self.pending.remove(&event_id) {
                let mut event = pending.event;
                if event.false_positive.is_none() {
                    event.false_positive = Some(false);
                    let reward = event.compute_reward(&self.config);
                    self.bandit.update(event.arm_idx, &event.context, reward);
                }
            }
        }
    }
}

pub enum ObserverMessage {
    Event(RewardEvent),
    SignalFalsePositive { event_id: u64, is_fp: bool },
    Shutdown,
}

pub fn spawn_observer(
    bandit: Arc<LinUCBBandit>,
    config: RewardConfig,
    mut rx: mpsc::Receiver<ObserverMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut observer = RewardObserver::new(bandit, config.clone());
        let mut expire_interval = tokio::time::interval(config.event_ttl / 10);

        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Some(ObserverMessage::Event(event)) => {
                            observer.process_event(event);
                        }
                        Some(ObserverMessage::SignalFalsePositive { event_id, is_fp }) => {
                            observer.signal_false_positive(event_id, is_fp);
                        }
                        Some(ObserverMessage::Shutdown) | None => {
                            break;
                        }
                    }
                }
                _ = expire_interval.tick() => {
                    observer.expire_old_events();
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_for_allowed_request() {
        let config = RewardConfig::default();
        let event = RewardEvent {
            event_id: 1,
            arm_idx: 0,
            context: [0.0; FEATURE_DIM],
            decision: Decision::Allow,
            latency_ns: 1_000_000,
            false_positive: Some(false),
        };
        let reward = event.compute_reward(&config);
        assert!(reward > 0.0);
    }

    #[test]
    fn reward_for_denied_request() {
        let config = RewardConfig::default();
        let event = RewardEvent {
            event_id: 1,
            arm_idx: 0,
            context: [0.0; FEATURE_DIM],
            decision: Decision::Deny,
            latency_ns: 1_000_000,
            false_positive: Some(false),
        };
        let reward = event.compute_reward(&config);
        assert!(reward < 1.0);
    }

    #[test]
    fn false_positive_penalty() {
        let config = RewardConfig::default();
        let event_no_fp = RewardEvent {
            event_id: 1,
            arm_idx: 0,
            context: [0.0; FEATURE_DIM],
            decision: Decision::Allow,
            latency_ns: 1_000_000,
            false_positive: Some(false),
        };
        let event_fp = RewardEvent {
            event_id: 2,
            arm_idx: 0,
            context: [0.0; FEATURE_DIM],
            decision: Decision::Allow,
            latency_ns: 1_000_000,
            false_positive: Some(true),
        };
        let reward_no_fp = event_no_fp.compute_reward(&config);
        let reward_fp = event_fp.compute_reward(&config);
        assert!(reward_no_fp > reward_fp);
    }

    #[test]
    fn latency_penalty_scales() {
        let config = RewardConfig::default();
        let event_fast = RewardEvent {
            event_id: 1,
            arm_idx: 0,
            context: [0.0; FEATURE_DIM],
            decision: Decision::Allow,
            latency_ns: 1_000_000,
            false_positive: Some(false),
        };
        let event_slow = RewardEvent {
            event_id: 2,
            arm_idx: 0,
            context: [0.0; FEATURE_DIM],
            decision: Decision::Allow,
            latency_ns: 10_000_000,
            false_positive: Some(false),
        };
        let reward_fast = event_fast.compute_reward(&config);
        let reward_slow = event_slow.compute_reward(&config);
        assert!(reward_fast > reward_slow);
    }

    #[test]
    fn throttle_decision_not_allowed() {
        let event = RewardEvent {
            event_id: 1,
            arm_idx: 0,
            context: [0.0; FEATURE_DIM],
            decision: Decision::Throttle { delay_ms: 100 },
            latency_ns: 1_000_000,
            false_positive: Some(false),
        };
        assert!(!event.decision.is_allowed());
    }

    #[tokio::test]
    async fn observer_processes_events() {
        let bandit = Arc::new(LinUCBBandit::new(
            2,
            super::super::bandit::BanditConfig::default(),
        ));
        let config = RewardConfig::default();
        let (tx, rx) = mpsc::channel(100);

        let handle = spawn_observer(bandit.clone(), config, rx);

        let event = RewardEvent {
            event_id: 1,
            arm_idx: 0,
            context: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            decision: Decision::Allow,
            latency_ns: 1_000_000,
            false_positive: Some(false),
        };

        tx.send(ObserverMessage::Event(event)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        tx.send(ObserverMessage::Shutdown).await.unwrap();

        handle.await.unwrap();
    }

    #[tokio::test]
    async fn observer_handles_false_positive_signal() {
        let bandit = Arc::new(LinUCBBandit::new(
            2,
            super::super::bandit::BanditConfig::default(),
        ));
        let config = RewardConfig::default();
        let (tx, rx) = mpsc::channel(100);

        let handle = spawn_observer(bandit.clone(), config, rx);

        let event = RewardEvent {
            event_id: 42,
            arm_idx: 0,
            context: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            decision: Decision::Allow,
            latency_ns: 1_000_000,
            false_positive: None,
        };

        tx.send(ObserverMessage::Event(event)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        tx.send(ObserverMessage::SignalFalsePositive {
            event_id: 42,
            is_fp: true,
        })
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        tx.send(ObserverMessage::Shutdown).await.unwrap();
        handle.await.unwrap();
    }
}
