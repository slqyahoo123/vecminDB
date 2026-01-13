//! 缓存系统公共定义
//!
//! 提供缓存系统中使用的公共定义，包括缓存项、淘汰策略、过期策略等。

use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// 缓存项定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheItem {
    /// 缓存键
    pub key: String,
    /// 缓存值
    pub value: Vec<u8>,
    /// 创建时间（序列化为RFC3339格式字符串）
    #[serde(with = "time_serde")]
    pub created_at: Instant,
    /// 最后访问时间（序列化为RFC3339格式字符串）
    #[serde(with = "time_serde")]
    pub accessed_at: Instant,
    /// 访问次数
    pub access_count: u64,
    /// 过期时间（序列化为RFC3339格式字符串）
    #[serde(with = "option_time_serde")]
    pub expires_at: Option<Instant>,
    /// 字节大小
    pub size_bytes: usize,
}

/// 用于Instant的序列化/反序列化
mod time_serde {
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use serde::{Deserialize, Deserializer, Serializer};

    /// 序列化Instant
    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = instant.elapsed();
        let secs_since_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(duration.as_secs());
        serializer.serialize_u64(secs_since_epoch)
    }

    /// 反序列化Instant
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs_since_epoch = u64::deserialize(deserializer)?;
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let diff_secs = now_secs.saturating_sub(secs_since_epoch);
        Ok(Instant::now() - Duration::from_secs(diff_secs))
    }
}

/// 用于Option<Instant>的序列化/反序列化
mod option_time_serde {
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use serde::{Deserialize, Deserializer, Serializer};

    /// 序列化Option<Instant>
    pub fn serialize<S>(
        opt_instant: &Option<Instant>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match opt_instant {
            Some(instant) => {
                let duration = instant.elapsed();
                let secs_since_epoch = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs()
                    .saturating_sub(duration.as_secs());
                serializer.serialize_some(&secs_since_epoch)
            }
            None => serializer.serialize_none(),
        }
    }

    /// 反序列化Option<Instant>
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Instant>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let opt_secs = Option::<u64>::deserialize(deserializer)?;
        match opt_secs {
            Some(secs_since_epoch) => {
                let now_secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let diff_secs = now_secs.saturating_sub(secs_since_epoch);
                Ok(Some(Instant::now() - Duration::from_secs(diff_secs)))
            }
            None => Ok(None),
        }
    }
}

impl CacheItem {
    /// 创建新的缓存项
    pub fn new(key: &str, value: Vec<u8>, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        let size_bytes = value.len();
        Self {
            key: key.to_string(),
            value,
            created_at: now,
            accessed_at: now,
            access_count: 0,
            expires_at: ttl.map(|t| now + t),
            size_bytes,
        }
    }

    /// 检查缓存项是否已过期
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|expires| Instant::now() >= expires)
            .unwrap_or(false)
    }

    /// 更新访问信息
    pub fn mark_accessed(&mut self) {
        self.accessed_at = Instant::now();
        self.access_count += 1;
    }
}

/// 缓存淘汰策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// 最近最少使用：淘汰最长时间未被访问的项
    LRU,
    /// 最近最常使用：淘汰最近被频繁访问的项
    MRU,
    /// 最不经常使用：淘汰访问次数最少的项
    LFU,
    /// 先进先出：淘汰最早加入缓存的项
    FIFO,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self::LRU
    }
}

/// 缓存过期策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpirationPolicy {
    /// 固定时间后过期
    TTL(Duration),
    /// 根据最后访问时间过期
    Sliding(Duration),
    /// 永不过期
    Never,
}

impl Default for ExpirationPolicy {
    fn default() -> Self {
        Self::TTL(Duration::from_secs(3600)) // 默认1小时
    }
}

impl ExpirationPolicy {
    /// 计算过期时间
    pub fn calculate_expiry(&self, last_access: Instant) -> Option<Instant> {
        match self {
            Self::TTL(ttl) => Some(Instant::now() + *ttl),
            Self::Sliding(ttl) => Some(last_access + *ttl),
            Self::Never => None,
        }
    }

    /// 检查是否已过期
    pub fn is_expired(&self, creation_time: Instant, last_access: Instant) -> bool {
        match self {
            Self::TTL(ttl) => Instant::now() > creation_time + *ttl,
            Self::Sliding(ttl) => Instant::now() > last_access + *ttl,
            Self::Never => false,
        }
    }
} 