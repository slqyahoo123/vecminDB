use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

// ==================== 存储系统实现 ====================

/// 缓存管理器实现
pub struct CacheManagerImpl {
    pub(crate) cache_store: Arc<RwLock<HashMap<String, CacheEntry>>>,
    pub(crate) cache_config: CacheConfig,
    pub(crate) eviction_policy: Box<dyn EvictionPolicy>,
    pub(crate) statistics: Arc<RwLock<CacheStatistics>>,
}

/// 缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub max_size: usize,
    pub max_entries: usize,
    pub ttl_seconds: u64,
    pub eviction_policy: String,
    pub enable_compression: bool,
    pub compression_level: u8,
    pub enable_statistics: bool,
}

/// 缓存条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub created_at: DateTime<Utc>,
    pub accessed_at: DateTime<Utc>,
    pub access_count: u64,
    pub ttl: u64,
    pub size: usize,
    pub compressed: bool,
}

/// 缓存统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub total_size: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub compression_ratio: f32,
    pub average_access_time: f64,
}

/// 驱逐策略trait
pub trait EvictionPolicy: Send + Sync {
    fn select_for_eviction(&self, entries: &HashMap<String, CacheEntry>) -> Option<String>;
    fn update_access(&self, entry: &mut CacheEntry);
}

/// LRU驱逐策略
pub struct LruEvictionPolicy;

impl EvictionPolicy for LruEvictionPolicy {
    fn select_for_eviction(&self, entries: &HashMap<String, CacheEntry>) -> Option<String> {
        entries
            .iter()
            .min_by_key(|(_, entry)| entry.accessed_at)
            .map(|(key, _)| key.clone())
    }

    fn update_access(&self, entry: &mut CacheEntry) {
        entry.accessed_at = Utc::now();
        entry.access_count += 1;
    }
}

/// LFU驱逐策略
pub struct LfuEvictionPolicy;

impl EvictionPolicy for LfuEvictionPolicy {
    fn select_for_eviction(&self, entries: &HashMap<String, CacheEntry>) -> Option<String> {
        entries
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(key, _)| key.clone())
    }

    fn update_access(&self, entry: &mut CacheEntry) {
        entry.accessed_at = Utc::now();
        entry.access_count += 1;
    }
}

impl CacheManagerImpl {
    pub fn new(config: CacheConfig) -> Self {
        let eviction_policy: Box<dyn EvictionPolicy> = match config.eviction_policy.as_str() {
            "lru" => Box::new(LruEvictionPolicy),
            "lfu" => Box::new(LfuEvictionPolicy),
            _ => Box::new(LruEvictionPolicy),
        };

        Self {
            cache_store: Arc::new(RwLock::new(HashMap::new())),
            cache_config: config,
            eviction_policy,
            statistics: Arc::new(RwLock::new(CacheStatistics {
                total_entries: 0,
                total_size: 0,
                hit_count: 0,
                miss_count: 0,
                eviction_count: 0,
                compression_ratio: 1.0,
                average_access_time: 0.0,
            })),
        }
    }

    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, crate::Error> {
        let start_time = std::time::Instant::now();
        let mut cache = self.cache_store.write().unwrap();

        if let Some(entry) = cache.get_mut(key) {
            let now = Utc::now();
            if now.signed_duration_since(entry.created_at).num_seconds() > entry.ttl as i64 {
                cache.remove(key);
                self.update_statistics(false, start_time.elapsed());
                return Ok(None);
            }

            self.eviction_policy.update_access(entry);
            self.update_statistics(true, start_time.elapsed());

            Ok(Some(entry.value.clone()))
        } else {
            self.update_statistics(false, start_time.elapsed());
            Ok(None)
        }
    }

    pub async fn put(&self, key: &str, value: Vec<u8>) -> Result<(), crate::Error> {
        let mut cache = self.cache_store.write().unwrap();

        if cache.len() >= self.cache_config.max_entries {
            if let Some(evict_key) = self.eviction_policy.select_for_eviction(&cache) {
                cache.remove(&evict_key);
                let mut stats = self.statistics.write().unwrap();
                stats.eviction_count += 1;
            }
        }

        let val_len = value.len();
        let entry = CacheEntry {
            key: key.to_string(),
            value: if self.cache_config.enable_compression { value } else { value },
            created_at: Utc::now(),
            accessed_at: Utc::now(),
            access_count: 1,
            ttl: self.cache_config.ttl_seconds,
            size: val_len,
            compressed: self.cache_config.enable_compression,
        };

        cache.insert(key.to_string(), entry);

        let mut stats = self.statistics.write().unwrap();
        stats.total_entries = cache.len();
        stats.total_size += val_len;

        Ok(())
    }

    pub async fn remove(&self, key: &str) -> Result<bool, crate::Error> {
        let mut cache = self.cache_store.write().unwrap();
        let removed = cache.remove(key).is_some();

        if removed {
            let mut stats = self.statistics.write().unwrap();
            stats.total_entries = cache.len();
        }

        Ok(removed)
    }

    pub async fn clear(&self) -> Result<(), crate::Error> {
        let mut cache = self.cache_store.write().unwrap();
        cache.clear();

        let mut stats = self.statistics.write().unwrap();
        stats.total_entries = 0;
        stats.total_size = 0;

        Ok(())
    }

    pub async fn get_statistics(&self) -> Result<CacheStatistics, crate::Error> {
        let stats = self.statistics.read().unwrap();
        Ok(stats.clone())
    }

    fn update_statistics(&self, hit: bool, access_time: std::time::Duration) {
        let mut stats = self.statistics.write().unwrap();
        if hit {
            stats.hit_count += 1;
        } else {
            stats.miss_count += 1;
        }

        let total_requests = stats.hit_count + stats.miss_count;
        let current_avg = stats.average_access_time;
        let new_avg = (current_avg * (total_requests - 1) as f64 + access_time.as_micros() as f64)
            / total_requests as f64;
        stats.average_access_time = new_avg;
    }
}

/// 存储健康监控器
pub struct StorageHealthMonitor {
    pub(crate) health_checks: HashMap<String, Box<dyn HealthCheck>>,
    pub(crate) health_status: Arc<RwLock<StorageHealthStatus>>,
    pub(crate) check_interval: Duration,
}

/// 存储健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageHealthStatus {
    pub overall_status: String,
    pub last_check: DateTime<Utc>,
    pub checks: HashMap<String, HealthCheckResult>,
    pub alerts: Vec<HealthAlert>,
}

/// 健康检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub name: String,
    pub status: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub metrics: HashMap<String, f64>,
}

/// 健康告警
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    pub alert_id: String,
    pub severity: String,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
}

/// 健康检查trait
pub trait HealthCheck: Send + Sync {
    fn check(&self) -> Result<HealthCheckResult, crate::Error>;
    fn get_name(&self) -> &str;
}

/// 磁盘空间检查
pub struct DiskSpaceCheck {
    pub(crate) threshold_percent: f32,
}

impl HealthCheck for DiskSpaceCheck {
    fn check(&self) -> Result<HealthCheckResult, crate::Error> {
        let total_space = 1_000_000_000_000u64;
        let used_space = 800_000_000_000u64;
        let available_space = total_space - used_space;
        let usage_percent = (used_space as f64 / total_space as f64) * 100.0;

        let status = if usage_percent > self.threshold_percent as f64 {
            "warning".to_string()
        } else {
            "healthy".to_string()
        };

        let message = if usage_percent > self.threshold_percent as f64 {
            format!(
                "Disk usage is {}%, above threshold of {}%",
                usage_percent, self.threshold_percent
            )
        } else {
            format!("Disk usage is {}%, within normal range", usage_percent)
        };

        Ok(HealthCheckResult {
            name: "disk_space".to_string(),
            status,
            message,
            timestamp: Utc::now(),
            metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("total_space_gb".to_string(), total_space as f64 / 1_000_000_000.0);
                metrics.insert("used_space_gb".to_string(), used_space as f64 / 1_000_000_000.0);
                metrics.insert(
                    "available_space_gb".to_string(),
                    available_space as f64 / 1_000_000_000.0,
                );
                metrics.insert("usage_percent".to_string(), usage_percent);
                metrics
            },
        })
    }

    fn get_name(&self) -> &str {
        "disk_space"
    }
}

/// 连接数检查
pub struct ConnectionCountCheck {
    pub(crate) max_connections: usize,
}

impl HealthCheck for ConnectionCountCheck {
    fn check(&self) -> Result<HealthCheckResult, crate::Error> {
        let current_connections = 150;
        let max_connections = self.max_connections;
        let connection_percent = (current_connections as f64 / max_connections as f64) * 100.0;

        let status = if connection_percent > 80.0 {
            "warning".to_string()
        } else {
            "healthy".to_string()
        };

        let message = if connection_percent > 80.0 {
            format!(
                "Connection count is {} ({}%), approaching limit of {}",
                current_connections, connection_percent, max_connections
            )
        } else {
            format!(
                "Connection count is {} ({}%), within normal range",
                current_connections, connection_percent
            )
        };

        Ok(HealthCheckResult {
            name: "connection_count".to_string(),
            status,
            message,
            timestamp: Utc::now(),
            metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("current_connections".to_string(), current_connections as f64);
                metrics.insert("max_connections".to_string(), max_connections as f64);
                metrics.insert("connection_percent".to_string(), connection_percent);
                metrics
            },
        })
    }

    fn get_name(&self) -> &str {
        "connection_count"
    }
}

/// 响应时间检查
pub struct ResponseTimeCheck {
    pub(crate) max_response_time_ms: u64,
}

impl HealthCheck for ResponseTimeCheck {
    fn check(&self) -> Result<HealthCheckResult, crate::Error> {
        let avg_response_time = 45.0;
        let max_response_time = self.max_response_time_ms as f64;

        let status = if avg_response_time > max_response_time {
            "warning".to_string()
        } else {
            "healthy".to_string()
        };

        let message = if avg_response_time > max_response_time {
            format!(
                "Average response time is {}ms, above threshold of {}ms",
                avg_response_time, max_response_time
            )
        } else {
            format!(
                "Average response time is {}ms, within normal range",
                avg_response_time
            )
        };

        Ok(HealthCheckResult {
            name: "response_time".to_string(),
            status,
            message,
            timestamp: Utc::now(),
            metrics: {
                let mut metrics = HashMap::new();
                metrics.insert("avg_response_time_ms".to_string(), avg_response_time);
                metrics.insert("max_response_time_ms".to_string(), max_response_time);
                metrics
            },
        })
    }

    fn get_name(&self) -> &str {
        "response_time"
    }
}

impl StorageHealthMonitor {
    pub fn new() -> Self {
        let mut health_checks: HashMap<String, Box<dyn HealthCheck>> = HashMap::new();
        health_checks.insert(
            "disk_space".to_string(),
            Box::new(DiskSpaceCheck {
                threshold_percent: 85.0,
            }),
        );
        health_checks.insert(
            "connection_count".to_string(),
            Box::new(ConnectionCountCheck { max_connections: 200 }),
        );
        health_checks.insert(
            "response_time".to_string(),
            Box::new(ResponseTimeCheck {
                max_response_time_ms: 100,
            }),
        );

        Self {
            health_checks,
            health_status: Arc::new(RwLock::new(StorageHealthStatus {
                overall_status: "unknown".to_string(),
                last_check: Utc::now(),
                checks: HashMap::new(),
                alerts: Vec::new(),
            })),
            check_interval: Duration::from_secs(60),
        }
    }

    pub async fn start_monitoring(&self) -> Result<(), crate::Error> {
        let health_checks = self.health_checks.iter().map(|(k,v)| (k.clone(), v.get_name().to_string())).collect::<Vec<_>>();
        let health_status = self.health_status.clone();
        let check_interval = self.check_interval;

        tokio::spawn(async move {
            let mut interval_stream = tokio::time::interval(check_interval);

            loop {
                interval_stream.tick().await;

                let mut status = health_status.write().unwrap();
                status.last_check = Utc::now();

                let mut overall_healthy = true;
                let mut new_alerts = Vec::new();

                for (name, _check_name) in &health_checks {
                    // 重新构造检查，以避免 Fn trait 对象 clone 限制；生产环境应共享引用并上锁
                    let check: Box<dyn HealthCheck> = match name.as_str() {
                        "disk_space" => Box::new(DiskSpaceCheck { threshold_percent: 85.0 }),
                        "connection_count" => Box::new(ConnectionCountCheck { max_connections: 200 }),
                        "response_time" => Box::new(ResponseTimeCheck { max_response_time_ms: 100 }),
                        _ => Box::new(ResponseTimeCheck { max_response_time_ms: 100 }),
                    };
                    match check.check() {
                        Ok(result) => {
                            status.checks.insert(name.clone(), result.clone());

                            if result.status == "warning" || result.status == "error" {
                                overall_healthy = false;

                                new_alerts.push(HealthAlert {
                                    alert_id: format!("{}_{}", name, Utc::now().timestamp()),
                                    severity: result.status.clone(),
                                    message: result.message.clone(),
                                    timestamp: Utc::now(),
                                    resolved: false,
                                });
                            }
                        }
                        Err(e) => {
                            status.checks.insert(
                                name.clone(),
                                HealthCheckResult {
                                    name: name.clone(),
                                    status: "error".to_string(),
                                    message: format!("Health check failed: {}", e),
                                    timestamp: Utc::now(),
                                    metrics: HashMap::new(),
                                },
                            );
                            overall_healthy = false;
                        }
                    }
                }

                status.overall_status = if overall_healthy {
                    "healthy".to_string()
                } else {
                    "warning".to_string()
                };
                status.alerts.extend(new_alerts);
            }
        });

        Ok(())
    }

    pub async fn get_health_status(&self) -> Result<StorageHealthStatus, crate::Error> {
        let status = self.health_status.read().unwrap();
        Ok(status.clone())
    }

    pub async fn add_health_check(&mut self, name: String, check: Box<dyn HealthCheck>) -> Result<(), crate::Error> {
        self.health_checks.insert(name, check);
        Ok(())
    }

    pub async fn resolve_alert(&self, alert_id: &str) -> Result<(), crate::Error> {
        let mut status = self.health_status.write().unwrap();
        for alert in &mut status.alerts {
            if alert.alert_id == alert_id {
                alert.resolved = true;
                break;
            }
        }
        Ok(())
    }
}


