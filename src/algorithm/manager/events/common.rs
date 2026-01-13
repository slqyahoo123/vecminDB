// 通用事件功能

use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

/// 事件系统配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventSystemConfig {
    pub max_events: usize,
    pub max_subscribers: usize,
    pub event_timeout: Duration,
    pub batch_size: usize,
    pub enable_persistence: bool,
    pub enable_metrics: bool,
}

impl Default for EventSystemConfig {
    fn default() -> Self {
        Self {
            max_events: 10000,
            max_subscribers: 1000,
            event_timeout: Duration::from_secs(30),
            batch_size: 100,
            enable_persistence: true,
            enable_metrics: true,
        }
    }
}

/// 事件系统统计信息
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct EventSystemStats {
    pub total_events_published: u64,
    pub total_events_processed: u64,
    pub total_subscribers: u64,
    pub active_subscribers: u64,
    pub failed_deliveries: u64,
    pub average_processing_time: Duration,
    pub last_event_time: Option<SystemTime>,
}

/// 订阅者信息
#[derive(Clone)]
pub struct SubscriberInfo {
    pub id: String,
    pub callback: std::sync::Arc<dyn crate::event::EventCallback + Send + Sync>,
    pub created_at: SystemTime,
    pub last_activity: SystemTime,
} 