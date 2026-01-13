use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 向量元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// 向量ID
    pub id: String,
    /// 标签
    pub tags: Vec<String>,
    /// 属性
    pub properties: HashMap<String, String>,
    /// 创建时间
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 最后更新时间
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl VectorMetadata {
    /// 创建新的元数据
    pub fn new(id: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            id,
            tags: Vec::new(),
            properties: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// 添加标签
    pub fn add_tag(&mut self, tag: &str) {
        if !self.tags.contains(&tag.to_string()) {
            self.tags.push(tag.to_string());
            self.updated_at = chrono::Utc::now();
        }
    }

    /// 设置属性
    pub fn set_property(&mut self, key: &str, value: &str) {
        self.properties.insert(key.to_string(), value.to_string());
        self.updated_at = chrono::Utc::now();
    }
}

/// 向量批量操作结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorBatchResult {
    /// 成功数量
    pub success_count: usize,
    /// 失败数量
    pub failed_count: usize,
    /// 错误详情
    pub errors: Vec<String>,
    /// 处理时间(毫秒)
    pub took_ms: u64,
} 