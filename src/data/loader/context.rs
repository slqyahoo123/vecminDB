// 数据加载上下文实现

use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::Utc;

/// 数据加载上下文
/// 用于在数据加载过程中传递元数据、配置和统计信息
#[derive(Debug, Clone)]
pub struct Context {
    /// 开始加载时间
    pub start_time: Option<std::time::Instant>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 统计信息
    pub stats: Option<LoadStats>,
    /// 配置信息
    pub config: HashMap<String, String>,
}

impl Context {
    /// 创建新的上下文
    pub fn new() -> Self {
        Self {
            start_time: Some(std::time::Instant::now()),
            metadata: HashMap::new(),
            stats: Some(LoadStats::new()),
            config: HashMap::new(),
        }
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// 添加多个元数据
    pub fn add_metadata_map(&mut self, metadata: HashMap<String, String>) -> &mut Self {
        self.metadata.extend(metadata);
        self
    }
    
    /// 添加配置
    pub fn add_config(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.config.insert(key.into(), value.into());
        self
    }
    
    /// 添加多个配置
    pub fn add_config_map(&mut self, config: HashMap<String, String>) -> &mut Self {
        self.config.extend(config);
        self
    }
    
    /// 获取开始时间
    pub fn get_start_time(&self) -> Option<std::time::Instant> {
        self.start_time
    }
    
    /// 获取元数据
    pub fn get_metadata(&self) -> Option<&HashMap<String, String>> {
        if self.metadata.is_empty() {
            None
        } else {
            Some(&self.metadata)
        }
    }
    
    /// 获取指定的元数据值
    pub fn get_metadata_value(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> Option<&LoadStats> {
        self.stats.as_ref()
    }
    
    /// 获取可变统计信息
    pub fn get_mut_stats(&mut self) -> Option<&mut LoadStats> {
        self.stats.as_mut()
    }
    
    /// 获取配置信息
    pub fn get_config(&self) -> Option<&HashMap<String, String>> {
        if self.config.is_empty() {
            None
        } else {
            Some(&self.config)
        }
    }
    
    /// 获取指定的配置值
    pub fn get_config_value(&self, key: &str) -> Option<&String> {
        self.config.get(key)
    }
    
    /// 重置开始时间
    pub fn reset_start_time(&mut self) -> &mut Self {
        self.start_time = Some(std::time::Instant::now());
        self
    }
    
    /// 合并另一个上下文
    pub fn merge(&mut self, other: &Context) -> &mut Self {
        // 合并元数据
        for (key, value) in &other.metadata {
            self.metadata.insert(key.clone(), value.clone());
        }
        
        // 合并配置
        for (key, value) in &other.config {
            self.config.insert(key.clone(), value.clone());
        }
        
        // 合并统计
        if let (Some(self_stats), Some(other_stats)) = (&mut self.stats, &other.stats) {
            self_stats.merge(other_stats);
        }
        
        self
    }
    
    /// 记录加载结束，更新统计信息
    pub fn record_load_complete(&mut self, records_count: usize) -> &mut Self {
        if let Some(start_time) = self.start_time {
            let duration = start_time.elapsed();
            
            if let Some(stats) = &mut self.stats {
                stats.increment("loader.records_loaded", records_count as i64);
                stats.add_time("loader.load_time", duration);
                
                // 计算加载速率（每秒记录数）
                let seconds = duration.as_secs_f64();
                if seconds > 0.0 {
                    let records_per_second = (records_count as f64 / seconds) as i64;
                    stats.set("loader.records_per_second", records_per_second);
                }
            }
            
            // 添加加载时间元数据
            self.add_metadata("load_time_ms", duration.as_millis().to_string());
            self.add_metadata("records_count", records_count.to_string());
        }
        
        self
    }
    
    /// 创建一个包含基本信息的新上下文
    pub fn with_basic_info(source_name: &str, format_name: &str) -> Self {
        let mut ctx = Self::new();
        ctx.add_metadata("source", source_name);
        ctx.add_metadata("format", format_name);
        ctx.add_metadata("created_at", Utc::now().to_rfc3339());
        ctx
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// 加载统计信息
/// 用于收集数据加载过程中的各种统计指标
#[derive(Debug, Clone)]
pub struct LoadStats {
    /// 统计计数器
    pub counters: HashMap<String, i64>,
    /// 统计计时器
    pub timers: HashMap<String, std::time::Duration>,
    /// 创建时间
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 最后更新时间
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl LoadStats {
    /// 创建新的统计对象
    pub fn new() -> Self {
        let now = chrono::Utc::now();
        Self {
            counters: HashMap::new(),
            timers: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
    
    /// 增加计数器
    pub fn increment(&mut self, key: &str, value: i64) -> &mut Self {
        *self.counters.entry(key.to_string()).or_insert(0) += value;
        self.updated_at = chrono::Utc::now();
        self
    }
    
    /// 设置计数器
    pub fn set(&mut self, key: &str, value: i64) -> &mut Self {
        self.counters.insert(key.to_string(), value);
        self.updated_at = chrono::Utc::now();
        self
    }
    
    /// 增加计时器
    pub fn add_time(&mut self, key: &str, duration: std::time::Duration) -> &mut Self {
        *self.timers.entry(key.to_string()).or_insert(std::time::Duration::new(0, 0)) += duration;
        self.updated_at = chrono::Utc::now();
        self
    }
    
    /// 开始计时
    pub fn start_timer(&self, key: &str) -> TimerHandle {
        TimerHandle {
            key: key.to_string(),
            start_time: Instant::now(),
        }
    }
    
    /// 获取计数器值
    pub fn get_counter(&self, key: &str) -> Option<i64> {
        self.counters.get(key).copied()
    }
    
    /// 获取计数器值或默认值
    pub fn get_counter_or(&self, key: &str, default: i64) -> i64 {
        self.counters.get(key).copied().unwrap_or(default)
    }
    
    /// 获取计时器值
    pub fn get_timer(&self, key: &str) -> Option<std::time::Duration> {
        self.timers.get(key).copied()
    }
    
    /// 获取计时器值或默认值
    pub fn get_timer_or(&self, key: &str, default: Duration) -> Duration {
        self.timers.get(key).copied().unwrap_or(default)
    }
    
    /// 合并另一个统计对象
    pub fn merge(&mut self, other: &LoadStats) -> &mut Self {
        // 合并计数器
        for (key, value) in &other.counters {
            *self.counters.entry(key.clone()).or_insert(0) += *value;
        }
        
        // 合并计时器
        for (key, value) in &other.timers {
            *self.timers.entry(key.clone()).or_insert(std::time::Duration::new(0, 0)) += *value;
        }
        
        // 更新时间
        self.updated_at = chrono::Utc::now();
        
        self
    }
    
    /// 重置统计
    pub fn reset(&mut self) -> &mut Self {
        self.counters.clear();
        self.timers.clear();
        let now = chrono::Utc::now();
        self.created_at = now;
        self.updated_at = now;
        self
    }
    
    /// 将统计数据转换为元数据HashMap
    pub fn to_metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        // 添加计数器
        for (key, value) in &self.counters {
            metadata.insert(format!("counter.{}", key), value.to_string());
        }
        
        // 添加计时器
        for (key, value) in &self.timers {
            metadata.insert(format!("timer.{}", key), value.as_millis().to_string());
        }
        
        // 添加时间戳
        metadata.insert("created_at".to_string(), self.created_at.to_rfc3339());
        metadata.insert("updated_at".to_string(), self.updated_at.to_rfc3339());
        
        metadata
    }
    
    /// 计算每秒处理记录数
    pub fn calculate_records_per_second(&self) -> Option<f64> {
        let records = self.get_counter("loader.records_loaded")?;
        let duration = self.get_timer("loader.load_time")?;
        
        let seconds = duration.as_secs_f64();
        if seconds > 0.0 {
            Some(records as f64 / seconds)
        } else {
            None
        }
    }
}

impl Default for LoadStats {
    fn default() -> Self {
        Self::new()
    }
}

/// 计时器句柄
/// 用于方便地测量代码块的执行时间
pub struct TimerHandle {
    key: String,
    start_time: Instant,
}

impl TimerHandle {
    /// 记录时间并更新统计信息
    pub fn stop(self, stats: &mut LoadStats) {
        let duration = self.start_time.elapsed();
        stats.add_time(&self.key, duration);
    }
    
    /// 获取当前经过的时间
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// 包含执行上下文和统计信息的高级上下文对象
pub struct ExecutionContext {
    /// 基础上下文
    pub context: Context,
    /// 开始时间
    pub start_time: Instant,
    /// 批次ID
    pub batch_id: String,
    /// 源信息
    pub source_info: String,
    /// 是否持久化统计信息
    pub persist_stats: bool,
}

impl ExecutionContext {
    /// 创建新的执行上下文
    pub fn new(source_info: &str) -> Self {
        Self {
            context: Context::new(),
            start_time: Instant::now(),
            batch_id: uuid::Uuid::new_v4().to_string(),
            source_info: source_info.to_string(),
            persist_stats: true,
        }
    }
    
    /// 记录执行完成
    pub fn record_completion(&mut self, records_count: usize) {
        let duration = self.start_time.elapsed();
        
        // 更新统计信息
        if let Some(stats) = &mut self.context.stats {
            stats.increment("loader.records_processed", records_count as i64);
            stats.add_time("loader.execution_time", duration);
        }
        
        // 添加完成元数据
        self.context.add_metadata("execution_time_ms", duration.as_millis().to_string());
        self.context.add_metadata("records_processed", records_count.to_string());
        self.context.add_metadata("batch_id", &self.batch_id);
        self.context.add_metadata("completed_at", Utc::now().to_rfc3339());
    }
    
    /// 添加自定义度量
    pub fn add_metric(&mut self, key: &str, value: i64) {
        if let Some(stats) = &mut self.context.stats {
            stats.set(key, value);
        }
    }
    
    /// 获取基于当前上下文的子上下文
    pub fn create_child_context(&self, operation: &str) -> Context {
        let mut child = Context::new();
        
        // 复制元数据
        if let Some(metadata) = self.context.get_metadata() {
            for (key, value) in metadata {
                if key.starts_with("global.") {
                    child.add_metadata(key, value);
                }
            }
        }
        
        // 添加父上下文信息
        child.add_metadata("parent_batch_id", &self.batch_id);
        child.add_metadata("operation", operation);
        
        child
    }
} 