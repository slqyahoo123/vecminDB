use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// 数据信息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInfo {
    /// 数据ID
    pub id: String,
    /// 数据名称
    pub name: String,
    /// 数据描述
    pub description: Option<String>,
    /// 数据类型
    pub data_type: String,
    /// 数据格式
    pub format: String,
    /// 数据大小（字节）
    pub size: u64,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 标签
    pub tags: Vec<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl Default for DataInfo {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: String::new(),
            name: String::new(),
            description: None,
            data_type: "unknown".to_string(),
            format: "json".to_string(),
            size: 0,
            created_at: now,
            updated_at: now,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// 存储项信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageItemInfo {
    /// 项目ID
    pub id: String,
    /// 项目类型
    pub item_type: String,
    /// 项目状态
    pub status: String,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 项目大小
    pub size_bytes: u64,
    /// 额外属性
    pub properties: HashMap<String, String>,
}

/// 存储统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// 总项目数量
    pub total_items: usize,
    /// 总大小（字节）
    pub total_size_bytes: u64,
    /// 按类型分组的统计
    pub by_type: HashMap<String, TypeStats>,
    /// 最后更新时间
    pub last_updated: DateTime<Utc>,
}

impl Default for StorageStats {
    fn default() -> Self {
        Self {
            total_items: 0,
            total_size_bytes: 0,
            by_type: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

/// 按类型统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeStats {
    /// 数量
    pub count: usize,
    /// 总大小
    pub total_size: u64,
    /// 平均大小
    pub avg_size: u64,
}

/// 存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfiguration {
    /// 最大项目大小
    pub max_item_size: u64,
    /// 是否启用压缩
    pub compression_enabled: bool,
    /// 是否启用加密
    pub encryption_enabled: bool,
    /// 备份间隔（秒）
    pub backup_interval_seconds: u64,
    /// 清理策略
    pub cleanup_policy: CleanupPolicy,
}

/// 清理策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupPolicy {
    /// 不清理
    None,
    /// 基于时间清理
    TimeBasedDays(u32),
    /// 基于大小清理
    SizeBasedMB(u64),
    /// 基于数量清理
    CountBased(usize),
}

impl Default for StorageConfiguration {
    fn default() -> Self {
        Self {
            max_item_size: 100 * 1024 * 1024, // 100MB
            compression_enabled: true,
            encryption_enabled: false,
            backup_interval_seconds: 3600, // 1小时
            cleanup_policy: CleanupPolicy::TimeBasedDays(30),
        }
    }
} 