use crate::data::{Dataset, DataRecord};
use crate::error::{Error, Result};
use crate::utils::hash::compute_hash;
use crate::utils::concurrency::ThreadPool;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use log::{info};
use serde_json;

// 数据分片配置
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// 每个分片的最大记录数
    pub max_records_per_shard: usize,
    /// 每个分片的最大大小（字节）
    pub max_shard_size_bytes: usize,
    /// 分片策略
    pub strategy: ShardStrategy,
    /// 是否按特征平衡分片
    pub balance_by_feature: Option<String>,
    /// 并行处理线程数，为0时使用CPU核心数
    pub parallel_threads: usize,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            max_records_per_shard: 10000,
            max_shard_size_bytes: 100 * 1024 * 1024, // 100MB
            strategy: ShardStrategy::Random,
            balance_by_feature: None,
            parallel_threads: 0,
        }
    }
}

/// 分片策略
#[derive(Debug, Clone, PartialEq)]
pub enum ShardStrategy {
    /// 随机分片
    Random,
    /// 按Hash分片
    Hash,
    /// 按范围分片（适用于连续特征）
    Range,
    /// 按特征分片
    ByFeature(String),
}

/// 分片键类型
pub type PartitionKey = String;

/// 数据分片
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// 分片ID
    pub id: String,
    /// 分片大小（字节）
    pub size: usize,
    /// 分片内记录数
    pub record_count: usize,
    /// 分片元数据
    pub metadata: HashMap<String, String>,
    /// 分片数据
    pub data: Vec<DataRecord>,
}

impl ShardInfo {
    /// 创建新的数据分片
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            size: 0,
            record_count: 0,
            metadata: HashMap::new(),
            data: Vec::new(),
        }
    }

    /// 添加记录到分片
    pub fn add_record(&mut self, record: DataRecord) -> Result<()> {
        // 估计记录大小
        let record_size = estimate_record_size(&record)?;
        
        self.data.push(record);
        self.size += record_size;
        self.record_count += 1;
        
        Ok(())
    }
    
    /// 获取分片统计信息
    pub fn stats(&self) -> ShardMetrics {
        ShardMetrics {
            id: self.id.clone(),
            size_bytes: self.size,
            record_count: self.record_count,
            features: self.extract_feature_stats(),
        }
    }
    
    /// 提取特征统计信息
    fn extract_feature_stats(&self) -> HashMap<String, FeatureStats> {
        // 简化实现，只统计数值型特征的基本统计量
        let mut feature_stats: HashMap<String, FeatureStats> = HashMap::new();
        
        // 遍历所有记录，累计统计信息
        for record in &self.data {
            for (name, value) in &record.fields {
                let stats = feature_stats.entry(name.clone()).or_insert_with(FeatureStats::new);
                // 将 data::record::Value 转换为 serde_json::Value
                let json_value = match value {
                    crate::data::record::Value::Data(data_value) => {
                        serde_json::to_value(data_value)
                            .unwrap_or(serde_json::Value::Null)
                    },
                    crate::data::record::Value::Record(_) => serde_json::Value::Null,
                    crate::data::record::Value::Reference(_) => serde_json::Value::Null,
                };
                stats.update(&json_value);
            }
        }
        
        feature_stats
    }
}

/// 分片统计信息
#[derive(Debug, Clone)]
pub struct ShardMetrics {
    /// 分片ID
    pub id: String,
    /// 分片大小（字节）
    pub size_bytes: usize,
    /// 记录数
    pub record_count: usize,
    /// 特征统计
    pub features: HashMap<String, FeatureStats>,
}

/// 特征统计信息
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// 值计数
    pub count: usize,
    /// 最小值
    pub min: f64,
    /// 最大值
    pub max: f64,
    /// 总和
    pub sum: f64,
    /// 均值
    pub mean: f64,
}

impl FeatureStats {
    /// 创建新的特征统计
    pub fn new() -> Self {
        Self {
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            mean: 0.0,
        }
    }
    
    /// 更新统计信息
    pub fn update(&mut self, value: &serde_json::Value) {
        if let Some(num) = value.as_f64() {
            self.count += 1;
            self.min = self.min.min(num);
            self.max = self.max.max(num);
            self.sum += num;
            self.mean = self.sum / self.count as f64;
        }
    }
}

/// 分片管理器
pub struct DataShardManager {
    /// 分片配置
    config: ShardConfig,
    /// 线程池
    thread_pool: ThreadPool,
}

impl DataShardManager {
    /// 创建新的分片管理器
    pub fn new(config: ShardConfig) -> Self {
        let threads = if config.parallel_threads == 0 {
            num_cpus::get()
        } else {
            config.parallel_threads
        };
        
        let pool_config = crate::utils::concurrency::ThreadPoolConfig {
            min_threads: 2,
            max_threads: threads.max(4),
            keep_alive: std::time::Duration::from_secs(60),
            queue_capacity: 1000,
            rejection_policy: crate::utils::concurrency::RejectionPolicy::Block,
        };
        
        Self {
            config,
            thread_pool: ThreadPool::new(pool_config),
        }
    }
    
    /// 处理大规模数据集
    pub fn process_large_dataset(&self, dataset: &Dataset) -> Result<Vec<ShardInfo>> {
        info!("开始处理大规模数据集，策略: {:?}", self.config.strategy);
        
        match self.config.strategy {
            ShardStrategy::Random => self.random_sharding(dataset),
            ShardStrategy::Hash => self.hash_based_sharding(dataset),
            ShardStrategy::Range => self.range_based_sharding(dataset),
            ShardStrategy::ByFeature(ref feature) => self.feature_based_sharding(dataset, feature),
        }
    }
    
    /// 随机分片策略
    fn random_sharding(&self, dataset: &Dataset) -> Result<Vec<ShardInfo>> {
        info!("使用随机分片策略处理数据集");
        
        // 估计记录总数
        let dataset_size = dataset.size;
        
        // 计算需要的分片数
        let shard_count = (dataset_size as f64 / self.config.max_records_per_shard as f64).ceil() as usize;
        let shard_count = shard_count.max(1); // 至少1个分片
        
        info!("数据集记录数: {}, 预计分片数: {}", dataset_size, shard_count);
        
        // 创建分片容器
        let shards: Arc<Mutex<Vec<ShardInfo>>> = Arc::new(Mutex::new(
            (0..shard_count).map(|i| ShardInfo::new(&format!("shard_{}", i))).collect()
        ));
        
        // 提取配置值，避免在并行闭包中捕获 self
        let max_records_per_shard = self.config.max_records_per_shard;
        let max_shard_size_bytes = self.config.max_shard_size_bytes;
        
        // 创建批次迭代器并并行处理
        let batch_iterator = dataset.create_batches(max_records_per_shard)?;
        batch_iterator.par_bridge().try_for_each(|batch_result| -> Result<()> {
            let batch = batch_result?;
            let local_shards = Arc::clone(&shards);
            
            // 在并行闭包内部创建 ThreadRng，每个线程都有自己的 RNG
            let mut rng = rand::thread_rng();
            
            for record_map in batch.records() {
                // 将 HashMap<String, DataValue> 转换为 Record
                let mut record = DataRecord::new();
                for (key, value) in record_map {
                    record.add_field(key, crate::data::record::Value::Data(value.clone()));
                }
                
                // 随机选择一个分片
                let shard_idx = rand::Rng::gen_range(&mut rng, 0..shard_count);
                
                // 添加记录到分片
                let mut shards_guard = local_shards.lock().map_err(|e| {
                    Error::LockError(format!("获取分片锁失败: {}", e))
                })?;
                
                // 如果当前分片已满，查找下一个有空间的分片
                let mut attempt = 0;
                let mut current_idx = shard_idx;
                
                while attempt < shard_count {
                    let shard = &mut shards_guard[current_idx];
                    
                    if shard.record_count < max_records_per_shard && 
                       shard.size < max_shard_size_bytes {
                        // 分片未满，添加记录
                        shard.add_record(record.clone())?;
                        break;
                    }
                    
                    // 尝试下一个分片
                    current_idx = (current_idx + 1) % shard_count;
                    attempt += 1;
                }
                
                // 如果所有分片都已满，创建新分片
                if attempt >= shard_count {
                    let new_shard_idx = shards_guard.len();
                    let mut new_shard = ShardInfo::new(&format!("shard_{}", new_shard_idx));
                    new_shard.add_record(record.clone())?;
                    shards_guard.push(new_shard);
                }
            }
            
            Ok(())
        })?;
        
        // 返回最终分片结果
        let result = shards.lock().map_err(|e| {
            Error::LockError(format!("获取分片锁失败: {}", e))
        })?.clone();
        
        info!("数据分片完成，生成了 {} 个分片", result.len());
        
        Ok(result)
    }
    
    /// 基于Hash的分片策略
    fn hash_based_sharding(&self, dataset: &Dataset) -> Result<Vec<ShardInfo>> {
        info!("使用Hash分片策略处理数据集");
        
        // 估计记录总数
        let dataset_size = dataset.size;
        
        // 计算需要的分片数
        let shard_count = (dataset_size as f64 / self.config.max_records_per_shard as f64).ceil() as usize;
        let shard_count = shard_count.max(1); // 至少1个分片
        
        info!("数据集记录数: {}, 预计分片数: {}", dataset_size, shard_count);
        
        // 创建分片容器
        let shards: Arc<Mutex<Vec<ShardInfo>>> = Arc::new(Mutex::new(
            (0..shard_count).map(|i| ShardInfo::new(&format!("shard_{}", i))).collect()
        ));
        
        // 提取配置值，避免在并行闭包中捕获 self
        let max_records_per_shard = self.config.max_records_per_shard;
        let max_shard_size_bytes = self.config.max_shard_size_bytes;
        
        // 创建批次迭代器并并行处理
        let batch_iterator = dataset.create_batches(max_records_per_shard)?;
        batch_iterator.par_bridge().try_for_each(|batch_result| -> Result<()> {
            let batch = batch_result?;
            let local_shards = Arc::clone(&shards);
            
            for record_map in batch.records() {
                // 将 HashMap<String, DataValue> 转换为 Record
                let mut record = DataRecord::new();
                for (key, value) in record_map {
                    record.add_field(key, crate::data::record::Value::Data(value.clone()));
                }
                
                // 计算记录哈希值：将记录序列化为JSON字符串，然后计算哈希
                let record_json = serde_json::to_string(&record)
                    .map_err(|e| Error::data(format!("序列化记录失败: {}", e)))?;
                let hash_bytes = record_json.as_bytes();
                let hash = compute_hash(hash_bytes);
                
                // 将哈希字符串转换为数字
                let hash_num = hash.chars()
                    .take(16)
                    .fold(0u64, |acc, c| {
                        acc.wrapping_mul(31).wrapping_add(c as u64)
                    });
                
                // 根据哈希值确定分片
                let shard_idx = (hash_num % shard_count as u64) as usize;
                
                // 添加记录到分片
                let mut shards_guard = local_shards.lock().map_err(|e| {
                    Error::LockError(format!("获取分片锁失败: {}", e))
                })?;
                
                let shard = &mut shards_guard[shard_idx];
                
                // 检查分片是否已满
                if shard.record_count >= max_records_per_shard || 
                   shard.size >= max_shard_size_bytes {
                    // 分片已满，创建新分片
                    let new_shard_idx = shards_guard.len();
                    let mut new_shard = ShardInfo::new(&format!("shard_{}", new_shard_idx));
                    new_shard.add_record(record.clone())?;
                    shards_guard.push(new_shard);
                } else {
                    // 分片未满，添加记录
                    shard.add_record(record.clone())?;
                }
            }
            
            Ok(())
        })?;
        
        // 返回最终分片结果
        let result = shards.lock().map_err(|e| {
            Error::LockError(format!("获取分片锁失败: {}", e))
        })?.clone();
        
        info!("数据分片完成，生成了 {} 个分片", result.len());
        
        Ok(result)
    }
    
    /// 基于范围的分片策略
    fn range_based_sharding(&self, dataset: &Dataset) -> Result<Vec<ShardInfo>> {
        info!("使用范围分片策略处理数据集");
        
        // 选择用于分片的特征
        let feature_name = match &self.config.balance_by_feature {
            Some(name) => name.clone(),
            None => {
                // 如果未指定特征，尝试找到第一个数值型特征
                let batch_iterator = dataset.create_batches(100)?;
                let mut found_feature = None;
                if let Some(batch_result) = batch_iterator.next() {
                    if let Ok(batch) = batch_result {
                        if let Some(record_map) = batch.records().first() {
                            for (name, value) in record_map {
                                // 检查 DataValue 是否为数字类型
                                if matches!(value, crate::data::value::DataValue::Integer(_) | 
                                                  crate::data::value::DataValue::Float(_) | 
                                                  crate::data::value::DataValue::Number(_)) {
                                    info!("自动选择特征 '{}' 进行范围分片", name);
                                    found_feature = Some(name.clone());
                                    break;
                                }
                            }
                        }
                    }
                }
                // 如果没有找到数值型特征，返回错误
                found_feature.ok_or_else(|| Error::invalid_input("未指定范围分片特征，且无法自动选择"))?
            }
        };
        
        // 首先扫描特征的范围
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;
        
        let batch_iterator = dataset.create_batches(self.config.max_records_per_shard)?;
        for batch_result in batch_iterator {
            let batch = batch_result?;
            for record_map in batch.records() {
                if let Some(value) = record_map.get(&feature_name) {
                    let num = match value {
                        crate::data::value::DataValue::Integer(i) => Some(*i as f64),
                        crate::data::value::DataValue::Float(f) => Some(*f),
                        crate::data::value::DataValue::Number(n) => Some(*n),
                        _ => None,
                    };
                    if let Some(num) = num {
                        min_value = min_value.min(num);
                        max_value = max_value.max(num);
                    }
                }
            }
        }
        
        // 如果没有找到有效值，返回错误
        if min_value == f64::INFINITY || max_value == f64::NEG_INFINITY {
            return Err(Error::invalid_input(format!("特征 '{}' 没有有效的数值", feature_name)));
        }
        
        // 估计记录总数
        let dataset_size = dataset.size;
        
        // 计算需要的分片数
        let shard_count = (dataset_size as f64 / self.config.max_records_per_shard as f64).ceil() as usize;
        let shard_count = shard_count.max(1); // 至少1个分片
        
        // 计算每个分片的范围
        let range_size = (max_value - min_value) / shard_count as f64;
        
        info!("特征 '{}' 范围: {} - {}, 分片数: {}, 每个分片范围: {}", 
              feature_name, min_value, max_value, shard_count, range_size);
        
        // 创建分片容器
        let shards: Arc<Mutex<Vec<ShardInfo>>> = Arc::new(Mutex::new(
            (0..shard_count).map(|i| {
                let range_start = min_value + i as f64 * range_size;
                let range_end = if i == shard_count - 1 {
                    max_value + 0.001 // 确保包含最大值
                } else {
                    min_value + (i + 1) as f64 * range_size
                };
                
                let mut shard = ShardInfo::new(&format!("shard_{}", i));
                shard.metadata.insert("range_start".to_string(), range_start.to_string());
                shard.metadata.insert("range_end".to_string(), range_end.to_string());
                shard.metadata.insert("feature".to_string(), feature_name.clone());
                
                shard
            }).collect()
        ));
        
        // 提取配置值，避免在并行闭包中捕获 self
        let max_records_per_shard = self.config.max_records_per_shard;
        let max_shard_size_bytes = self.config.max_shard_size_bytes;
        
        // 创建批次迭代器并并行处理
        let batch_iterator = dataset.create_batches(max_records_per_shard)?;
        batch_iterator.par_bridge().try_for_each(|batch_result| -> Result<()> {
            let batch = batch_result?;
            let local_shards = Arc::clone(&shards);
            
            for record_map in batch.records() {
                // 获取特征值
                let feature_value = match record_map.get(&feature_name) {
                    Some(value) => {
                        match value {
                            crate::data::value::DataValue::Integer(i) => Some(*i as f64),
                            crate::data::value::DataValue::Float(f) => Some(*f),
                            crate::data::value::DataValue::Number(n) => Some(*n),
                            _ => None,
                        }
                    },
                    None => None,
                };
                
                // 将 HashMap<String, DataValue> 转换为 Record
                let mut record = DataRecord::new();
                for (key, value) in record_map {
                    record.add_field(key, crate::data::record::Value::Data(value.clone()));
                }
                
                // 如果特征不存在或不是数值，分配到最后一个分片
                let shard_idx = if let Some(fv) = feature_value {
                    // 确定特征值所属分片
                    let idx = ((fv - min_value) / range_size).floor() as usize;
                    idx.min(shard_count - 1) // 确保索引不越界
                } else {
                    // 分配到最后一个分片
                    shard_count - 1
                };
                
                // 添加记录到分片
                let mut shards_guard = local_shards.lock().map_err(|e| {
                    Error::LockError(format!("获取分片锁失败: {}", e))
                })?;
                
                // 如果当前分片已满，创建溢出分片
                let shard = &mut shards_guard[shard_idx];
                if shard.record_count >= max_records_per_shard || 
                   shard.size >= max_shard_size_bytes {
                    
                    // 创建溢出分片
                    let overflow_idx = shards_guard.len();
                    let range_start = min_value + shard_idx as f64 * range_size;
                    let range_end = min_value + (shard_idx + 1) as f64 * range_size;
                    
                    let mut new_shard = ShardInfo::new(&format!("shard_{}_overflow_{}", shard_idx, overflow_idx - shard_count));
                    new_shard.metadata.insert("range_start".to_string(), range_start.to_string());
                    new_shard.metadata.insert("range_end".to_string(), range_end.to_string());
                    new_shard.metadata.insert("feature".to_string(), feature_name.clone());
                    new_shard.metadata.insert("overflow".to_string(), "true".to_string());
                    
                    new_shard.add_record(record.clone())?;
                    shards_guard.push(new_shard);
                } else {
                    // 分片未满，添加记录
                    shard.add_record(record.clone())?;
                }
            }
            
            Ok(())
        })?;
        
        // 返回最终分片结果
        let result = shards.lock().map_err(|e| {
            Error::LockError(format!("获取分片锁失败: {}", e))
        })?.clone();
        
        info!("数据分片完成，生成了 {} 个分片", result.len());
        
        Ok(result)
    }
    
    /// 基于特征的分片策略
    fn feature_based_sharding(&self, dataset: &Dataset, feature_name: &str) -> Result<Vec<ShardInfo>> {
        info!("使用特征分片策略处理数据集，特征: {}", feature_name);
        
        // 提取配置值，避免在并行闭包中捕获 self
        let max_records_per_shard = self.config.max_records_per_shard;
        let max_shard_size_bytes = self.config.max_shard_size_bytes;
        
        // 收集特征的唯一值
        let mut feature_values = std::collections::HashSet::new();
        
        let batch_iterator = dataset.create_batches(max_records_per_shard)?;
        for batch_result in batch_iterator {
            let batch = batch_result?;
            for record_map in batch.records() {
                if let Some(value) = record_map.get(feature_name) {
                    // 将值转换为字符串用于分组
                    let value_str = format!("{:?}", value);
                    feature_values.insert(value_str);
                }
            }
        }
        
        // 如果没有找到特征值，返回错误
        if feature_values.is_empty() {
            return Err(Error::invalid_input(format!("特征 '{}' 没有有效值", feature_name)));
        }
        
        // 将唯一值转换为有序列表
        let mut feature_values: Vec<String> = feature_values.into_iter().collect();
        feature_values.sort();
        
        info!("特征 '{}' 有 {} 个唯一值", feature_name, feature_values.len());
        
        // 创建分片映射（特征值 -> 分片索引）
        let mut value_to_shard = HashMap::new();
        
        // 创建分片容器
        let shards: Arc<Mutex<Vec<ShardInfo>>> = Arc::new(Mutex::new(
            feature_values.iter().enumerate().map(|(i, val)| {
                let mut shard = ShardInfo::new(&format!("shard_{}", i));
                shard.metadata.insert("feature".to_string(), feature_name.to_string());
                shard.metadata.insert("value".to_string(), val.clone());
                
                value_to_shard.insert(val.clone(), i);
                
                shard
            }).collect()
        ));
        
        // 额外创建一个"未知"分片用于处理缺失值
        {
            let mut shards_guard = shards.lock().map_err(|e| {
                Error::LockError(format!("获取分片锁失败: {}", e))
            })?;
            
            let unknown_idx = shards_guard.len();
            let mut unknown_shard = ShardInfo::new(&format!("shard_unknown"));
            unknown_shard.metadata.insert("feature".to_string(), feature_name.to_string());
            unknown_shard.metadata.insert("value".to_string(), "unknown".to_string());
            
            shards_guard.push(unknown_shard);
            
            // 记录未知分片索引
            value_to_shard.insert("unknown".to_string(), unknown_idx);
        }
        
        // 创建批次迭代器并并行处理
        let batch_iterator = dataset.create_batches(max_records_per_shard)?;
        batch_iterator.par_bridge().try_for_each(|batch_result| -> Result<()> {
            let batch = batch_result?;
            let local_shards = Arc::clone(&shards);
            
            for record_map in batch.records() {
                // 将 HashMap<String, DataValue> 转换为 Record
                let mut record = DataRecord::new();
                for (key, value) in record_map {
                    record.add_field(key, crate::data::record::Value::Data(value.clone()));
                }
                
                // 获取特征值
                let shard_idx = match record_map.get(feature_name) {
                    Some(value) => {
                        // 将值转换为字符串查找对应的分片
                        let value_str = format!("{:?}", value);
                        *value_to_shard.get(&value_str).unwrap_or_else(|| value_to_shard.get("unknown").unwrap())
                    },
                    None => {
                        // 特征不存在，使用"未知"分片
                        *value_to_shard.get("unknown").unwrap()
                    }
                };
                
                // 添加记录到分片
                let mut shards_guard = local_shards.lock().map_err(|e| {
                    Error::LockError(format!("获取分片锁失败: {}", e))
                })?;
                
                let shard = &mut shards_guard[shard_idx];
                
                // 检查分片是否已满
                if shard.record_count >= max_records_per_shard || 
                   shard.size >= max_shard_size_bytes {
                    
                    // 创建溢出分片
                    let overflow_idx = shards_guard.len();
                    let feature_value = record_map.get(feature_name)
                        .map(|v| format!("{:?}", v))
                        .unwrap_or_else(|| "unknown".to_string());
                    
                    let mut new_shard = ShardInfo::new(&format!("shard_{}_overflow_{}", shard_idx, overflow_idx - feature_values.len() - 1));
                    new_shard.metadata.insert("feature".to_string(), feature_name.to_string());
                    new_shard.metadata.insert("value".to_string(), feature_value);
                    new_shard.metadata.insert("overflow".to_string(), "true".to_string());
                    
                    new_shard.add_record(record.clone())?;
                    shards_guard.push(new_shard);
                } else {
                    // 分片未满，添加记录
                    shard.add_record(record.clone())?;
                }
            }
            
            Ok(())
        })?;
        
        // 返回最终分片结果
        let result = shards.lock().map_err(|e| {
            Error::LockError(format!("获取分片锁失败: {}", e))
        })?.clone();
        
        info!("数据分片完成，生成了 {} 个分片", result.len());
        
        Ok(result)
    }
    
    /// 平衡分片数据
    pub fn balance_shards(&self, shards: &mut Vec<ShardInfo>) -> Result<()> {
        info!("开始平衡分片数据");
        
        // 计算平均每个分片的记录数
        let total_records: usize = shards.iter().map(|s| s.record_count).sum();
        let avg_records = total_records / shards.len();
        
        info!("分片总数: {}, 总记录数: {}, 平均每个分片记录数: {}", 
              shards.len(), total_records, avg_records);
        
        // 按记录数排序，找出过多和过少的分片
        shards.sort_by_key(|s| s.record_count);
        
        // 从多的分片移动记录到少的分片
        let mut i = 0;
        let mut j = shards.len() - 1;
        
        while i < j {
            let under_loaded = &shards[i];
            let over_loaded = &shards[j];
            
            if under_loaded.record_count >= avg_records || over_loaded.record_count <= avg_records {
                break;
            }
            
            // 计算需要移动的记录数
            let deficit = avg_records - under_loaded.record_count;
            let excess = over_loaded.record_count - avg_records;
            let to_move = deficit.min(excess);
            
            if to_move == 0 {
                break;
            }
            
            // 执行移动
            // 注意：由于不可变借用问题，我们需要克隆数据
            let mut records_to_move = Vec::new();
            
            {
                let over_loaded = &shards[j];
                records_to_move = over_loaded.data[over_loaded.data.len() - to_move..].to_vec();
            }
            
            // 更新分片
            shards[j].data.truncate(shards[j].data.len() - to_move);
            shards[j].record_count -= to_move;
            shards[j].size = estimate_shard_size(&shards[j].data)?;
            
            for record in records_to_move {
                shards[i].add_record(record)?;
            }
            
            // 如果分片已经平衡，移动指针
            if shards[i].record_count >= avg_records {
                i += 1;
            }
            
            if shards[j].record_count <= avg_records {
                j -= 1;
            }
        }
        
        info!("分片平衡完成");
        
        Ok(())
    }
}

/// 估计记录大小（字节）
fn estimate_record_size(record: &DataRecord) -> Result<usize> {
    // 简化实现，只是基本估算
    let mut size = 0;
    
    // ID大小
    if let Some(id) = &record.id {
        size += id.len();
    }
    
    // 字段大小
    for (key, value) in &record.fields {
        size += key.len();
        match value {
            crate::data::record::Value::Data(data_value) => {
                size += format!("{:?}", data_value).len();
            },
            crate::data::record::Value::Record(_) => {
                size += 8; // 估算嵌套记录大小
            },
            crate::data::record::Value::Reference(ref_str) => {
                size += ref_str.len();
            },
        }
    }
    
    // 元数据大小
    for (key, value) in &record.metadata {
        size += key.len();
        size += value.len();
    }
    
    // 添加一些开销
    size += 100;
    
    Ok(size)
}

/// 估计分片大小
fn estimate_shard_size(records: &[DataRecord]) -> Result<usize> {
    let mut total_size = 0;
    
    for record in records {
        total_size += estimate_record_size(record)?;
    }
    
    Ok(total_size)
} 