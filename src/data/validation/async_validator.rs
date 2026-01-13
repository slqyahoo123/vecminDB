// 异步验证接口模块
// 提供统一的异步验证功能，支持并发验证和复杂验证逻辑

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use futures::{future::join_all};
// use serde::{Serialize, Deserialize};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, instrument};

use crate::error::Result;
use crate::data::{DataValue, DataBatch};
use super::{ValidationError, ValidationConfig, ValidationContext, ValidationLocation};
use crate::core::interfaces::ValidationResult;

/// 异步验证接口
#[async_trait]
pub trait AsyncValidator: Send + Sync {
    /// 获取验证器名称
    fn name(&self) -> &str;
    
    /// 异步验证单个值
    async fn validate_value(&self, value: &DataValue, context: &ValidationContext) -> Result<ValidationResult>;
    
    /// 异步验证数据批次
    async fn validate_batch(&self, batch: &DataBatch, context: &ValidationContext) -> Result<ValidationResult> {
        let mut results = Vec::new();
        
        // 并发验证批次中的每个记录
        let semaphore = Arc::new(Semaphore::new(context.config.max_concurrent_validations.unwrap_or(10)));
        
        for (index, record) in batch.records.iter().enumerate() {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let validator = self;
            let mut record_context = context.clone();
            record_context.record_index = Some(index);
            
            let validation_future = async move {
                let _permit = permit; // 保持permit直到任务完成
                let mut record_result = ValidationResult::success();
                
                for (field_name, field_value) in record {
                    let mut field_context = record_context.clone();
                    field_context.push_field(field_name.clone());
                    
                    // field_value 已经是 DataValue，直接使用
                    let data_value = field_value;
                    
                    match validator.validate_value(&data_value, &field_context).await {
                        Ok(field_result) => {
                            record_result.merge(field_result);
                        }
                        Err(e) => {
                            record_result.add_error(ValidationError::new(
                                "validation_error".to_string(),
                                format!("验证字段 {} 时发生错误: {}", field_name, e)
                            ).with_field(field_name.clone())
                             .with_location(ValidationLocation {
                                 line: Some(index),
                                 column: None,
                                 offset: None,
                                 context: None,
                             }));
                        }
                    }
                }
                
                record_result
            };
            
            results.push(validation_future);
        }
        
        // 等待所有验证完成
        let validation_results = join_all(results).await;
        
        // 合并所有结果
        let mut final_result = ValidationResult::success();
        for result in validation_results {
            final_result.merge(result);
        }
        
        Ok(final_result)
    }
    
    /// 异步验证数据映射
    async fn validate_map(&self, data: &HashMap<String, DataValue>, context: &ValidationContext) -> Result<ValidationResult> {
        let mut result = ValidationResult::success();
        
        for (key, value) in data {
            let mut field_context = context.clone();
            field_context.push_field(key.clone());
            
            match self.validate_value(value, &field_context).await {
                Ok(field_result) => {
                    result.merge(field_result);
                }
                Err(e) => {
                    result.add_error(ValidationError::new(
                        "validation_error".to_string(),
                        format!("验证字段 {} 时发生错误: {}", key, e)
                    ).with_field(key.clone()));
                }
            }
        }
        
        Ok(result)
    }
    
    /// 检查是否支持指定的数据类型
    fn supports_type(&self, data_type: &str) -> bool {
        true // 默认支持所有类型
    }
    
    /// 获取验证器配置
    fn config(&self) -> &ValidationConfig;
}

/// 复合异步验证器
/// 组合多个验证器，支持并行或串行验证
pub struct CompositeAsyncValidator {
    /// 验证器名称
    name: String,
    
    /// 子验证器列表
    validators: Vec<Box<dyn AsyncValidator>>,
    
    /// 验证配置
    config: ValidationConfig,
    
    /// 是否并行验证
    parallel: bool,
    
    /// 最大并发数
    max_concurrency: usize,
}

impl CompositeAsyncValidator {
    /// 创建新的复合验证器
    pub fn new(name: String, config: ValidationConfig) -> Self {
        Self {
            name,
            validators: Vec::new(),
            config,
            parallel: true,
            max_concurrency: 10,
        }
    }
    
    /// 添加子验证器
    pub fn add_validator(mut self, validator: Box<dyn AsyncValidator>) -> Self {
        self.validators.push(validator);
        self
    }
    
    /// 设置是否并行验证
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
    
    /// 设置最大并发数
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }
}

#[async_trait]
impl AsyncValidator for CompositeAsyncValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    #[instrument(skip(self, value, context))]
    async fn validate_value(&self, value: &DataValue, context: &ValidationContext) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut final_result = ValidationResult::success();
        
        if self.validators.is_empty() {
            debug!("复合验证器 {} 没有子验证器", self.name);
            return Ok(final_result);
        }
        
        if self.parallel {
            // 并行验证
            let semaphore = Arc::new(Semaphore::new(self.max_concurrency));
            let mut validation_futures = Vec::new();
            
            for validator in &self.validators {
                let permit = semaphore.clone().acquire_owned().await.unwrap();
                let validator_ref = validator.as_ref();
                let value_clone = value.clone();
                let context_clone = context.clone();
                
                let validation_future = async move {
                    let _permit = permit;
                    validator_ref.validate_value(&value_clone, &context_clone).await
                };
                
                validation_futures.push(validation_future);
            }
            
            // 等待所有验证完成
            let results = join_all(validation_futures).await;
            
            for (i, result) in results.into_iter().enumerate() {
                match result {
                    Ok(validation_result) => {
                        final_result.merge(validation_result);
                    }
                    Err(e) => {
                        error!("验证器 {} 执行失败: {}", self.validators[i].name(), e);
                        final_result.add_error(ValidationError::new(
                            "validator_error".to_string(),
                            format!("验证器 {} 执行失败: {}", self.validators[i].name(), e)
                        ));
                    }
                }
            }
        } else {
            // 串行验证
            for validator in &self.validators {
                match validator.validate_value(value, context).await {
                    Ok(validation_result) => {
                        final_result.merge(validation_result);
                        
                        // 如果是严格模式且已有错误，立即停止
                        if context.config.strict_mode && !final_result.is_valid {
                            break;
                        }
                    }
                    Err(e) => {
                        error!("验证器 {} 执行失败: {}", validator.name(), e);
                        final_result.add_error(ValidationError::new(
                            "validator_error".to_string(),
                            format!("验证器 {} 执行失败: {}", validator.name(), e)
                        ));
                        
                        if context.config.strict_mode {
                            break;
                        }
                    }
                }
            }
        }
        
        let elapsed = start_time.elapsed();
        debug!("复合验证器 {} 完成验证，耗时: {:?}", self.name, elapsed);
        
        Ok(final_result)
    }
    
    fn config(&self) -> &ValidationConfig {
        &self.config
    }
}

/// 缓存异步验证器
/// 为验证结果提供缓存功能，提高重复验证的性能
pub struct CachedAsyncValidator {
    /// 验证器名称
    name: String,
    
    /// 内部验证器
    inner: Box<dyn AsyncValidator>,
    
    /// 验证结果缓存
    cache: Arc<RwLock<HashMap<String, (ValidationResult, Instant)>>>,
    
    /// 缓存过期时间
    cache_ttl: Duration,
    
    /// 最大缓存条目数
    max_cache_size: usize,
}

impl CachedAsyncValidator {
    /// 创建新的缓存验证器
    pub fn new(name: String, inner: Box<dyn AsyncValidator>) -> Self {
        Self {
            name,
            inner,
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_ttl: Duration::from_secs(300), // 5分钟
            max_cache_size: 1000,
        }
    }
    
    /// 设置缓存TTL
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }
    
    /// 设置最大缓存大小
    pub fn with_max_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }
    
    /// 生成缓存键
    fn generate_cache_key(&self, value: &DataValue, context: &ValidationContext) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        format!("{:?}", value).hash(&mut hasher);
        context.current_path().hash(&mut hasher);
        format!("{}_{}", self.name, hasher.finish())
    }
    
    /// 清理过期缓存
    async fn cleanup_expired_cache(&self) {
        let mut cache = self.cache.write().await;
        let now = Instant::now();
        
        cache.retain(|_, (_, timestamp)| {
            now.duration_since(*timestamp) < self.cache_ttl
        });
        
        // 如果缓存仍然太大，移除最旧的条目
        if cache.len() > self.max_cache_size {
            let mut entries: Vec<_> = cache.iter().map(|(k, (_, ts))| (k.clone(), *ts)).collect();
            entries.sort_by_key(|(_, timestamp)| *timestamp);
            
            let to_remove_count = cache.len() - self.max_cache_size;
            let keys_to_remove: Vec<String> = entries.iter().take(to_remove_count).map(|(k, _)| k.clone()).collect();
            
            for key in keys_to_remove {
                cache.remove(&key);
            }
        }
    }
}

#[async_trait]
impl AsyncValidator for CachedAsyncValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    #[instrument(skip(self, value, context))]
    async fn validate_value(&self, value: &DataValue, context: &ValidationContext) -> Result<ValidationResult> {
        let cache_key = self.generate_cache_key(value, context);
        
        // 检查缓存
        {
            let cache = self.cache.read().await;
            if let Some((result, timestamp)) = cache.get(&cache_key) {
                let now = Instant::now();
                if now.duration_since(*timestamp) < self.cache_ttl {
                    debug!("缓存命中: {}", cache_key);
                    return Ok(result.clone());
                }
            }
        }
        
        // 缓存未命中，执行验证
        debug!("缓存未命中，执行验证: {}", cache_key);
        let result = self.inner.validate_value(value, context).await?;
        
        // 更新缓存
        {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, (result.clone(), Instant::now()));
        }
        
        // 异步清理过期缓存（使用独立的清理逻辑避免借用冲突）
        let cache_for_cleanup = self.cache.clone();
        let ttl = self.cache_ttl;
        let max_size = self.max_cache_size;
        tokio::spawn(async move {
            // 等待一段时间再清理，避免频繁清理
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            
            let mut cache_guard = cache_for_cleanup.write().await;
            let now = Instant::now();
            
            // 清理过期条目
            cache_guard.retain(|_, (_, timestamp)| {
                now.duration_since(*timestamp) < ttl
            });
            
            // 如果缓存仍然太大，移除最旧的条目
            if cache_guard.len() > max_size {
                let entries: Vec<(String, Instant)> = cache_guard.iter()
                    .map(|(k, (_, timestamp))| (k.clone(), *timestamp))
                    .collect();
                    
                let mut sorted_entries = entries;
                sorted_entries.sort_by_key(|(_, timestamp)| *timestamp);
                
                let to_remove = cache_guard.len() - max_size;
                for (key, _) in sorted_entries.iter().take(to_remove) {
                    cache_guard.remove(key);
                }
            }
        });
        
        Ok(result)
    }
    
    fn config(&self) -> &ValidationConfig {
        self.inner.config()
    }
}

 