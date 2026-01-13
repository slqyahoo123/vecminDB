// 导入公共模块中的HNSWNode
use crate::vector::index::common::HNSWNode;

/// 搜索配置结构体
pub struct SearchConfig<T: Send + Sync = HNSWNode> {
    /// 搜索策略
    pub strategy: String,
    /// 限制结果数量
    pub limit: usize,
    /// 是否返回向量数据
    pub include_vectors: bool,
    /// 是否返回元数据
    pub include_metadata: bool,
    /// 过滤器函数
    pub filter: Option<Box<dyn Fn(&T) -> bool + Send + Sync>>,
}

impl<T: Send + Sync> Clone for SearchConfig<T> {
    fn clone(&self) -> Self {
        Self {
            strategy: self.strategy.clone(),
            limit: self.limit,
            include_vectors: self.include_vectors,
            include_metadata: self.include_metadata,
            // filter 函数无法 clone，设置为 None
            filter: None,
        }
    }
}

impl<T: Send + Sync> std::fmt::Debug for SearchConfig<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchConfig")
            .field("strategy", &self.strategy)
            .field("limit", &self.limit)
            .field("include_vectors", &self.include_vectors)
            .field("include_metadata", &self.include_metadata)
            .field("filter", &self.filter.as_ref().map(|_| "<function>"))
            .finish()
    }
}

impl<T: Send + Sync> SearchConfig<T> {
    /// 创建默认搜索配置
    pub fn new(limit: usize) -> Self {
        Self {
            strategy: "balanced".to_string(),
            limit,
            include_vectors: false,
            include_metadata: false,
            filter: None,
        }
    }
} 