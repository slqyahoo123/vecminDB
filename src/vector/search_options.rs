use crate::{Error, Result};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fmt;
use std::cmp::Ordering;
// use std::sync::Arc;
use crate::vector::types::Vector;
use crate::vector::search::VectorMetadata;
use crate::vector::core::operations::SimilarityMetric;

/// 搜索选项，用于控制向量搜索的行为
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchOptions {
    /// 相似性度量方式
    pub metric: SimilarityMetric,
    
    /// 最大返回结果数
    pub limit: usize,
    
    /// 是否包含原始向量数据在结果中
    pub include_vectors: bool,
    
    /// 是否包含元数据在结果中
    pub include_metadata: bool,
    
    /// 相似度阈值，低于此值的结果将被过滤
    pub score_threshold: Option<f32>,
    
    /// 元数据过滤条件
    pub filter: Option<FilterCondition>,
    
    /// 是否启用并行搜索
    pub parallel: bool,
    
    /// 高级检索参数
    pub params: HashMap<String, String>,
    
    /// 搜索指定ID的向量的近邻
    pub from_vector_id: Option<String>,
    
    /// 返回结果排序方式
    pub sort_by: SortOrder,
    
    /// 预过滤向量列表
    pub pre_filter_ids: Option<Vec<String>>,
    
    /// 重排序参数
    pub rerank: Option<RerankOptions>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            metric: SimilarityMetric::default(),
            limit: 10,
            include_vectors: false,
            include_metadata: true,
            score_threshold: None,
            filter: None,
            parallel: true,
            params: HashMap::new(),
            from_vector_id: None,
            sort_by: SortOrder::default(),
            pre_filter_ids: None,
            rerank: None,
        }
    }
}

impl SearchOptions {
    /// 创建新的搜索选项
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 设置相似度度量方式
    pub fn with_metric(mut self, metric: SimilarityMetric) -> Self {
        self.metric = metric;
        self
    }
    
    /// 设置结果数量限制
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
    
    /// 设置是否包含向量数据
    pub fn with_vectors(mut self, include: bool) -> Self {
        self.include_vectors = include;
        self
    }
    
    /// 设置是否包含元数据
    pub fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }
    
    /// 设置相似度阈值
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }
    
    /// 设置过滤条件
    pub fn with_filter(mut self, filter: FilterCondition) -> Self {
        self.filter = Some(filter);
        self
    }
    
    /// 设置是否并行搜索
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
    
    /// 添加高级参数
    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 设置从指定ID的向量搜索
    pub fn from_id(mut self, id: &str) -> Self {
        self.from_vector_id = Some(id.to_string());
        self
    }
    
    /// 设置结果排序方式
    pub fn with_sort(mut self, order: SortOrder) -> Self {
        self.sort_by = order;
        self
    }
    
    /// 设置预过滤ID列表
    pub fn with_pre_filter(mut self, ids: Vec<String>) -> Self {
        self.pre_filter_ids = Some(ids);
        self
    }
    
    /// 设置重排序选项
    pub fn with_rerank(mut self, rerank: RerankOptions) -> Self {
        self.rerank = Some(rerank);
        self
    }
    
    /// 验证搜索选项
    pub fn validate(&self) -> Result<()> {
        if self.limit == 0 {
            return Err(Error::invalid_argument("搜索结果限制不能为0"));
        }
        
        if let Some(threshold) = self.score_threshold {
            if threshold < 0.0 || threshold > 1.0 {
                return Err(Error::invalid_argument("相似度阈值必须在0到1之间"));
            }
        }
        
        Ok(())
    }
    
    /// 应用搜索选项到向量列表
    pub fn apply_to_results(&self, results: Vec<Vector>) -> Result<Vec<Vector>> {
        if results.is_empty() {
            return Ok(results);
        }
        
        let mut filtered = results;
        
        // 应用相似度阈值过滤
        if let Some(threshold) = self.score_threshold {
            filtered = filtered
                .into_iter()
                .filter(|v| {
                    if let Some(score) = v.metadata.as_ref().and_then(|m| m.properties.get("score")).and_then(|s| s.as_f64()) {
                        score as f32 >= threshold
                    } else {
                        true
                    }
                })
                .collect();
        }
        
        // 应用元数据过滤条件
        if let Some(filter) = &self.filter {
            filtered = filtered
                .into_iter()
                .filter(|v| {
                    if let Some(metadata) = &v.metadata {
                        filter.matches(metadata)
                    } else {
                        // 如果没有元数据，根据过滤条件决定是否保留
                        filter.matches_empty()
                    }
                })
                .collect();
        }
        
        // 应用预过滤ID列表
        if let Some(ids) = &self.pre_filter_ids {
            let id_set: std::collections::HashSet<_> = ids.iter().collect();
            filtered = filtered
                .into_iter()
                .filter(|v| id_set.contains(&v.id))
                .collect();
        }
        
        // 应用排序
        match &self.sort_by {
            SortOrder::ScoreAscending => {
                filtered.sort_by(|a, b| {
                    let a_score = a.metadata.as_ref().and_then(|m| m.properties.get("score")).and_then(|s| s.as_f64()).unwrap_or(0.0);
                    let b_score = b.metadata.as_ref().and_then(|m| m.properties.get("score")).and_then(|s| s.as_f64()).unwrap_or(0.0);
                    a_score.partial_cmp(&b_score).unwrap_or(Ordering::Equal)
                });
            },
            SortOrder::ScoreDescending => {
                filtered.sort_by(|a, b| {
                    let a_score = a.metadata.as_ref().and_then(|m| m.properties.get("score")).and_then(|s| s.as_f64()).unwrap_or(0.0);
                    let b_score = b.metadata.as_ref().and_then(|m| m.properties.get("score")).and_then(|s| s.as_f64()).unwrap_or(0.0);
                    b_score.partial_cmp(&a_score).unwrap_or(Ordering::Equal)
                });
            },
            SortOrder::IdAscending => {
                filtered.sort_by(|a, b| a.id.cmp(&b.id));
            },
            SortOrder::IdDescending => {
                filtered.sort_by(|a, b| b.id.cmp(&a.id));
            },
            SortOrder::Custom(field) => {
                filtered.sort_by(|a, b| {
                    let key = field.clone();
                    let a_val = a.metadata.as_ref().and_then(|m| m.properties.get(&key)).map(|v| v.to_string());
                    let b_val = b.metadata.as_ref().and_then(|m| m.properties.get(&key)).map(|v| v.to_string());
                    a_val.cmp(&b_val)
                });
            },
        }
        
        // 应用重排序
        if let Some(rerank) = &self.rerank {
            if let Some(model) = &rerank.model {
                // 使用重排序模型
                // 注意：这里通常需要一个外部模型的实现
                // 简化起见，这里只在注释中说明流程
                // 实际实现应该调用相应的重排序模型API
                
                // 1. 准备重排序输入
                // 2. 调用重排序模型
                // 3. 更新结果分数
                // 4. 重新排序
                
                // 由于重排序依赖于特定的模型实现，这里暂不提供具体实现
                log::info!("应用重排序模型: {}", model);
            }
        }
        
        // 限制结果数量
        if filtered.len() > self.limit {
            filtered.truncate(self.limit);
        }
        
        Ok(filtered)
    }
}

/// 结果排序顺序
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SortOrder {
    /// 按相似度升序（从低到高）
    ScoreAscending,
    
    /// 按相似度降序（从高到低，默认）
    ScoreDescending,
    
    /// 按ID升序
    IdAscending,
    
    /// 按ID降序
    IdDescending,
    
    /// 自定义字段排序
    Custom(String),
}

impl Default for SortOrder {
    fn default() -> Self {
        SortOrder::ScoreDescending
    }
}

/// 重排序选项
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RerankOptions {
    /// 重排序模型
    pub model: Option<String>,
    
    /// 重排序参数
    pub params: HashMap<String, String>,
}

/// 过滤条件操作符
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum FilterOperator {
    /// 等于
    Eq,
    
    /// 不等于
    Ne,
    
    /// 大于
    Gt,
    
    /// 大于等于
    Ge,
    
    /// 小于
    Lt,
    
    /// 小于等于
    Le,
    
    /// 包含（适用于字符串和数组）
    Contains,
    
    /// 不包含（适用于字符串和数组）
    NotContains,
    
    /// 开头是（适用于字符串）
    StartsWith,
    
    /// 结尾是（适用于字符串）
    EndsWith,
    
    /// 在范围内（包含两端）
    InRange,
    
    /// 在列表中
    In,
    
    /// 不在列表中
    NotIn,
    
    /// 存在属性
    Exists,
    
    /// 不存在属性
    NotExists,
    
    /// 与正则表达式匹配
    Regex,
}

impl fmt::Display for FilterOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterOperator::Eq => write!(f, "=="),
            FilterOperator::Ne => write!(f, "!="),
            FilterOperator::Gt => write!(f, ">"),
            FilterOperator::Ge => write!(f, ">="),
            FilterOperator::Lt => write!(f, "<"),
            FilterOperator::Le => write!(f, "<="),
            FilterOperator::Contains => write!(f, "contains"),
            FilterOperator::NotContains => write!(f, "not_contains"),
            FilterOperator::StartsWith => write!(f, "starts_with"),
            FilterOperator::EndsWith => write!(f, "ends_with"),
            FilterOperator::InRange => write!(f, "in_range"),
            FilterOperator::In => write!(f, "in"),
            FilterOperator::NotIn => write!(f, "not_in"),
            FilterOperator::Exists => write!(f, "exists"),
            FilterOperator::NotExists => write!(f, "not_exists"),
            FilterOperator::Regex => write!(f, "=~"),
        }
    }
}

/// 过滤条件组合逻辑
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum FilterLogic {
    /// 与操作（所有条件都必须满足）
    And,
    
    /// 或操作（满足任一条件即可）
    Or,
    
    /// 非操作（条件不满足时为真）
    Not,
}

impl fmt::Display for FilterLogic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterLogic::And => write!(f, "AND"),
            FilterLogic::Or => write!(f, "OR"),
            FilterLogic::Not => write!(f, "NOT"),
        }
    }
}

/// 过滤条件值
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    /// 字符串值
    String(String),
    
    /// 整数值
    Integer(i64),
    
    /// 浮点数值
    Float(f64),
    
    /// 布尔值
    Boolean(bool),
    
    /// 字符串数组
    StringArray(Vec<String>),
    
    /// 整数数组
    IntegerArray(Vec<i64>),
    
    /// 浮点数数组
    FloatArray(Vec<f64>),
    
    /// 过滤值对（用于范围过滤）
    Range(Box<(FilterValue, FilterValue)>),
    
    /// 空值
    Null,
}

impl FilterValue {
    /// 比较两个过滤值
    fn compare(&self, other: &FilterValue) -> Option<Ordering> {
        match (self, other) {
            (FilterValue::String(a), FilterValue::String(b)) => Some(a.cmp(b)),
            (FilterValue::Integer(a), FilterValue::Integer(b)) => Some(a.cmp(b)),
            (FilterValue::Float(a), FilterValue::Float(b)) => a.partial_cmp(b),
            (FilterValue::Boolean(a), FilterValue::Boolean(b)) => Some(a.cmp(b)),
            (FilterValue::Integer(a), FilterValue::Float(b)) => (*a as f64).partial_cmp(b),
            (FilterValue::Float(a), FilterValue::Integer(b)) => a.partial_cmp(&(*b as f64)),
            (FilterValue::Null, FilterValue::Null) => Some(Ordering::Equal),
            (FilterValue::Null, _) => Some(Ordering::Less),
            (_, FilterValue::Null) => Some(Ordering::Greater),
            _ => None, // 其他类型不可比较
        }
    }
    
    /// 从JSON值创建过滤值
    fn from_json(value: &serde_json::Value) -> Self {
        match value {
            serde_json::Value::String(s) => FilterValue::String(s.clone()),
            serde_json::Value::Number(n) => {
                if n.is_i64() {
                    FilterValue::Integer(n.as_i64().unwrap())
                } else {
                    FilterValue::Float(n.as_f64().unwrap())
                }
            },
            serde_json::Value::Bool(b) => FilterValue::Boolean(*b),
            serde_json::Value::Array(arr) => {
                if arr.is_empty() {
                    FilterValue::StringArray(Vec::new())
                } else if arr[0].is_string() {
                    FilterValue::StringArray(arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                } else if arr[0].is_i64() {
                    FilterValue::IntegerArray(arr.iter().filter_map(|v| v.as_i64()).collect())
                } else if arr[0].is_f64() {
                    FilterValue::FloatArray(arr.iter().filter_map(|v| v.as_f64()).collect())
                } else {
                    FilterValue::StringArray(Vec::new())
                }
            },
            serde_json::Value::Null => FilterValue::Null,
            _ => FilterValue::Null,
        }
    }
}

/// 原子过滤条件
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterCondition {
    /// 过滤条件类型
    #[serde(flatten)]
    pub condition: FilterConditionType,
}

impl FilterCondition {
    /// 创建属性过滤条件
    pub fn property(field: &str, op: FilterOperator, value: FilterValue) -> Self {
        Self {
            condition: FilterConditionType::Property {
                field: field.to_string(),
                operator: op,
                value,
            },
        }
    }
    
    /// 创建复合过滤条件
    pub fn composite(logic: FilterLogic, conditions: Vec<FilterCondition>) -> Self {
        Self {
            condition: FilterConditionType::Composite {
                logic,
                conditions,
            },
        }
    }
    
    /// 创建AND复合条件
    pub fn and(conditions: Vec<FilterCondition>) -> Self {
        Self::composite(FilterLogic::And, conditions)
    }
    
    /// 创建OR复合条件
    pub fn or(conditions: Vec<FilterCondition>) -> Self {
        Self::composite(FilterLogic::Or, conditions)
    }
    
    /// 创建NOT复合条件
    pub fn not(condition: FilterCondition) -> Self {
        Self::composite(FilterLogic::Not, vec![condition])
    }
    
    /// 检查条件是否匹配元数据
    pub fn matches(&self, metadata: &VectorMetadata) -> bool {
        match &self.condition {
            FilterConditionType::Property { field, operator, value } => {
                // 检查属性是否存在
                if !metadata.properties.contains_key(field) {
                    return match operator {
                        FilterOperator::NotExists => true,
                        FilterOperator::Ne => true,
                        FilterOperator::NotIn => true,
                        FilterOperator::NotContains => true,
                        _ => false,
                    };
                }
                
                let json_value = &metadata.properties[field];
                let property_value = FilterValue::from_json(json_value);
                
                match operator {
                    FilterOperator::Eq => self.equals(&property_value, value),
                    FilterOperator::Ne => !self.equals(&property_value, value),
                    FilterOperator::Gt => self.greater_than(&property_value, value),
                    FilterOperator::Ge => self.greater_than_or_equal(&property_value, value),
                    FilterOperator::Lt => self.less_than(&property_value, value),
                    FilterOperator::Le => self.less_than_or_equal(&property_value, value),
                    FilterOperator::Contains => self.contains(&property_value, value),
                    FilterOperator::NotContains => !self.contains(&property_value, value),
                    FilterOperator::StartsWith => self.starts_with(&property_value, value),
                    FilterOperator::EndsWith => self.ends_with(&property_value, value),
                    FilterOperator::InRange => self.in_range(&property_value, value),
                    FilterOperator::In => self.is_in(&property_value, value),
                    FilterOperator::NotIn => !self.is_in(&property_value, value),
                    FilterOperator::Exists => true,
                    FilterOperator::NotExists => false,
                    FilterOperator::Regex => self.matches_regex(&property_value, value),
                }
            },
            FilterConditionType::Composite { logic, conditions } => {
                match logic {
                    FilterLogic::And => conditions.iter().all(|c| c.matches(metadata)),
                    FilterLogic::Or => conditions.iter().any(|c| c.matches(metadata)),
                    FilterLogic::Not => {
                        if conditions.len() == 1 {
                            !conditions[0].matches(metadata)
                        } else {
                            // 如果有多个条件，则对所有条件取反并AND连接
                            conditions.iter().all(|c| !c.matches(metadata))
                        }
                    },
                }
            },
        }
    }
    
    /// 检查条件是否匹配空元数据
    pub fn matches_empty(&self) -> bool {
        match &self.condition {
            FilterConditionType::Property { operator, .. } => {
                match operator {
                    FilterOperator::NotExists => true,
                    FilterOperator::Ne => true,
                    FilterOperator::NotIn => true,
                    FilterOperator::NotContains => true,
                    _ => false,
                }
            },
            FilterConditionType::Composite { logic, conditions } => {
                match logic {
                    FilterLogic::And => conditions.iter().all(|c| c.matches_empty()),
                    FilterLogic::Or => conditions.iter().any(|c| c.matches_empty()),
                    FilterLogic::Not => {
                        if conditions.len() == 1 {
                            !conditions[0].matches_empty()
                        } else {
                            conditions.iter().all(|c| !c.matches_empty())
                        }
                    },
                }
            },
        }
    }
    
    // 各种比较操作实现
    fn equals(&self, a: &FilterValue, b: &FilterValue) -> bool {
        match (a, b) {
            (FilterValue::String(a_str), FilterValue::String(b_str)) => a_str == b_str,
            (FilterValue::Integer(a_int), FilterValue::Integer(b_int)) => a_int == b_int,
            (FilterValue::Float(a_float), FilterValue::Float(b_float)) => (a_float - b_float).abs() < 1e-6,
            (FilterValue::Boolean(a_bool), FilterValue::Boolean(b_bool)) => a_bool == b_bool,
            (FilterValue::Integer(a_int), FilterValue::Float(b_float)) => (*a_int as f64 - *b_float).abs() < 1e-6,
            (FilterValue::Float(a_float), FilterValue::Integer(b_int)) => (*a_float - *b_int as f64).abs() < 1e-6,
            _ => false,
        }
    }
    
    fn greater_than(&self, a: &FilterValue, b: &FilterValue) -> bool {
        a.compare(b).map_or(false, |ord| ord == Ordering::Greater)
    }
    
    fn greater_than_or_equal(&self, a: &FilterValue, b: &FilterValue) -> bool {
        a.compare(b).map_or(false, |ord| ord != Ordering::Less)
    }
    
    fn less_than(&self, a: &FilterValue, b: &FilterValue) -> bool {
        a.compare(b).map_or(false, |ord| ord == Ordering::Less)
    }
    
    fn less_than_or_equal(&self, a: &FilterValue, b: &FilterValue) -> bool {
        a.compare(b).map_or(false, |ord| ord != Ordering::Greater)
    }
    
    fn contains(&self, a: &FilterValue, b: &FilterValue) -> bool {
        match (a, b) {
            (FilterValue::String(a_str), FilterValue::String(b_str)) => a_str.contains(b_str),
            (FilterValue::StringArray(a_arr), FilterValue::String(b_str)) => a_arr.iter().any(|s| s == b_str),
            (FilterValue::IntegerArray(a_arr), FilterValue::Integer(b_int)) => a_arr.contains(b_int),
            (FilterValue::FloatArray(a_arr), FilterValue::Float(b_float)) => a_arr.iter().any(|&f| (f - *b_float).abs() < 1e-6),
            _ => false,
        }
    }
    
    fn starts_with(&self, a: &FilterValue, b: &FilterValue) -> bool {
        match (a, b) {
            (FilterValue::String(a_str), FilterValue::String(b_str)) => a_str.starts_with(b_str),
            _ => false,
        }
    }
    
    fn ends_with(&self, a: &FilterValue, b: &FilterValue) -> bool {
        match (a, b) {
            (FilterValue::String(a_str), FilterValue::String(b_str)) => a_str.ends_with(b_str),
            _ => false,
        }
    }
    
    fn in_range(&self, a: &FilterValue, b: &FilterValue) -> bool {
        if let FilterValue::Range(range) = b {
            let (min, max) = &**range;
            a.compare(min).map_or(false, |ord| ord != Ordering::Less) &&
            a.compare(max).map_or(false, |ord| ord != Ordering::Greater)
        } else {
            false
        }
    }
    
    fn is_in(&self, a: &FilterValue, b: &FilterValue) -> bool {
        match b {
            FilterValue::StringArray(arr) => {
                if let FilterValue::String(val) = a {
                    arr.contains(val)
                } else {
                    false
                }
            },
            FilterValue::IntegerArray(arr) => {
                if let FilterValue::Integer(val) = a {
                    arr.contains(val)
                } else if let FilterValue::Float(val) = a {
                    arr.iter().any(|&i| (i as f64 - *val).abs() < 1e-6)
                } else {
                    false
                }
            },
            FilterValue::FloatArray(arr) => {
                if let FilterValue::Float(val) = a {
                    arr.iter().any(|&f| (f - *val).abs() < 1e-6)
                } else if let FilterValue::Integer(val) = a {
                    arr.iter().any(|&f| (f - *val as f64).abs() < 1e-6)
                } else {
                    false
                }
            },
            _ => false,
        }
    }
    
    fn matches_regex(&self, a: &FilterValue, b: &FilterValue) -> bool {
        match (a, b) {
            (FilterValue::String(a_str), FilterValue::String(pattern)) => {
                // 使用正则表达式进行匹配
                // 由于标准库不包含正则表达式，这里只返回是否匹配的结果
                // 实际实现应该使用regex库进行匹配
                #[cfg(feature = "regex")]
                {
                    use regex::Regex;
                    if let Ok(re) = Regex::new(pattern) {
                        return re.is_match(a_str);
                    }
                }
                
                // 如果没有启用regex特性，则按字符串包含关系处理
                a_str.contains(pattern)
            },
            _ => false,
        }
    }
}

/// 过滤条件类型
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterConditionType {
    /// 属性过滤条件
    Property {
        /// 属性字段名
        field: String,
        
        /// 操作符
        operator: FilterOperator,
        
        /// 比较值
        value: FilterValue,
    },
    
    /// 复合过滤条件
    Composite {
        /// 逻辑操作符
        logic: FilterLogic,
        
        /// 子条件列表
        conditions: Vec<FilterCondition>,
    },
}

/// 用于按距离搜索的向量查询
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorSearchQuery {
    /// 查询向量
    pub vector: Vec<f32>,
    
    /// 查询向量元数据
    pub metadata: Option<VectorMetadata>,
    
    /// 搜索选项
    pub options: SearchOptions,
    
    /// 基于ID的多向量查询
    pub ids: Option<Vec<String>>,
}

impl VectorSearchQuery {
    /// 创建新的向量查询
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            metadata: None,
            options: SearchOptions::default(),
            ids: None,
        }
    }
    
    /// 添加查询向量元数据
    pub fn with_metadata(mut self, metadata: VectorMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }
    
    /// 设置搜索选项
    pub fn with_options(mut self, options: SearchOptions) -> Self {
        self.options = options;
        self
    }
    
    /// 添加多向量ID查询
    pub fn with_ids(mut self, ids: Vec<String>) -> Self {
        self.ids = Some(ids);
        self
    }
    
    /// 验证查询
    pub fn validate(&self) -> Result<()> {
        if self.vector.is_empty() && self.ids.is_none() {
            return Err(Error::invalid_argument("查询向量不能为空或必须提供ID列表"));
        }
        
        self.options.validate()?;
        
        Ok(())
    }
}

// 对外重新导出主要类型
pub use self::FilterOperator as VectorFilterOperator;
pub use self::FilterLogic as VectorFilterLogic;
pub use self::FilterValue as VectorFilterValue;
pub use self::FilterCondition as VectorFilterCondition;
pub use self::FilterConditionType as VectorFilterConditionType;
pub use self::SearchOptions as VectorSearchOptions;
pub use self::SortOrder as VectorSortOrder;
pub use self::RerankOptions as VectorRerankOptions;

/// 过滤选项（FilterCondition的别名）
pub type FilterOptions = FilterCondition;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_search_options_builder() {
        let options = SearchOptions::new()
            .with_limit(20)
            .with_vectors(true)
            .with_metadata(true)
            .with_threshold(0.5)
            .with_parallel(false);
        
        assert_eq!(options.limit, 20);
        assert_eq!(options.include_vectors, true);
        assert_eq!(options.include_metadata, true);
        assert_eq!(options.score_threshold, Some(0.5));
        assert_eq!(options.parallel, false);
    }
    
    #[test]
    fn test_filter_condition_property() {
        let filter = FilterCondition::property(
            "category",
            FilterOperator::Eq,
            FilterValue::String("electronics".to_string())
        );
        
        if let FilterConditionType::Property { field, operator, value } = &filter.condition {
            assert_eq!(field, "category");
            assert_eq!(*operator, FilterOperator::Eq);
            match value {
                FilterValue::String(s) => assert_eq!(s, "electronics"),
                _ => assert!(false, "Expected String value but got different type"),
            }
        } else {
            assert!(false, "Expected Property condition but got different type");
        }
    }
    
    #[test]
    fn test_filter_condition_composite() {
        let filter1 = FilterCondition::property(
            "price",
            FilterOperator::Lt,
            FilterValue::Float(100.0)
        );
        
        let filter2 = FilterCondition::property(
            "rating",
            FilterOperator::Ge,
            FilterValue::Float(4.5)
        );
        
        let combined = FilterCondition::and(vec![filter1, filter2]);
        
        match &combined.condition {
            FilterConditionType::Composite { logic, conditions } => {
                assert_eq!(*logic, FilterLogic::And);
                assert_eq!(conditions.len(), 2);
            },
            _ => assert!(false, "Expected Composite condition but got different type"),
        }
    }
    
    #[test]
    fn test_filter_matches() {
        let mut properties = HashMap::new();
        properties.insert("category".to_string(), serde_json::json!("electronics"));
        properties.insert("price".to_string(), serde_json::json!(85.5));
        properties.insert("in_stock".to_string(), serde_json::json!(true));
        properties.insert("tags".to_string(), serde_json::json!(["phone", "mobile", "android"]));
        
        let metadata = VectorMetadata { properties };
        
        // Test equality
        let filter1 = FilterCondition::property(
            "category",
            FilterOperator::Eq,
            FilterValue::String("electronics".to_string())
        );
        assert!(filter1.matches(&metadata));
        
        // Test numeric comparison
        let filter2 = FilterCondition::property(
            "price",
            FilterOperator::Lt,
            FilterValue::Float(100.0)
        );
        assert!(filter2.matches(&metadata));
        
        // Test array contains
        let filter3 = FilterCondition::property(
            "tags",
            FilterOperator::Contains,
            FilterValue::String("mobile".to_string())
        );
        assert!(filter3.matches(&metadata));
        
        // Test complex filter
        let filter4 = FilterCondition::and(vec![filter1, filter2, filter3]);
        assert!(filter4.matches(&metadata));
        
        // Test non-matching filter
        let filter5 = FilterCondition::property(
            "category",
            FilterOperator::Eq,
            FilterValue::String("clothing".to_string())
        );
        assert!(!filter5.matches(&metadata));
    }
} 