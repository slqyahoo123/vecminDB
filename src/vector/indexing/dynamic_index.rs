use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{Error, Result};
use crate::vector::indexing::{Index, IndexConfig, IndexType, IndexStats, SearchParams};
use crate::compat::{TensorData, SearchResult};
use crate::storage::Storage;
use crate::data::pipeline::DataProcessPipeline;

/// 多粒度索引分层配置
#[derive(Clone, Debug)]
pub struct GranularIndexConfig {
    /// 索引粒度配置
    pub granularity_levels: Vec<GranularityLevel>,
    /// 索引类型（所有层使用相同类型）
    pub index_type: IndexType,
    /// 数据维度
    pub dimensions: usize,
    /// 最大向量数量
    pub max_elements: usize,
    /// 自动调优间隔（秒）
    pub auto_tune_interval_secs: u64,
    /// 流量统计窗口大小
    pub stats_window_size: usize,
}

/// 粒度级别定义
#[derive(Clone, Debug)]
pub struct GranularityLevel {
    /// 粒度等级ID
    pub level_id: usize,
    /// 粒度名称
    pub name: String,
    /// 量化位数
    pub quantization_bits: usize,
    /// 聚类数量
    pub n_clusters: usize,
    /// 采样率（1.0表示全量数据）
    pub sampling_ratio: f32,
    /// 额外参数
    pub extra_params: HashMap<String, String>,
}

/// 动态多粒度索引
pub struct DynamicGranularIndex {
    /// 索引配置
    config: GranularIndexConfig,
    /// 多层索引
    layers: Vec<Arc<RwLock<dyn Index + Send + Sync>>>,
    /// 层级使用统计
    layer_stats: Vec<RwLock<LayerStats>>,
    /// 上次自动调优时间
    last_auto_tune: RwLock<Instant>,
    /// 数据处理管道
    pipeline: Option<Arc<DataProcessPipeline>>,
    /// 全局索引统计
    global_stats: RwLock<IndexStats>,
}

/// 层级使用统计
struct LayerStats {
    /// 查询次数
    query_count: usize,
    /// 查询耗时（毫秒）
    query_time_ms: Vec<u64>,
    /// 命中次数
    hit_count: usize,
    /// 命中率历史
    hit_rates: Vec<f32>,
    /// 最近的查询质量评分
    recent_quality_scores: Vec<f32>,
}

impl DynamicGranularIndex {
    /// 创建新的动态多粒度索引
    pub fn new(config: GranularIndexConfig) -> Result<Self> {
        if config.granularity_levels.is_empty() {
            return Err(Error::param_error("必须至少配置一个粒度级别"));
        }
        
        // 创建各层索引
        let mut layers = Vec::with_capacity(config.granularity_levels.len());
        let mut layer_stats = Vec::with_capacity(config.granularity_levels.len());
        
        for level in &config.granularity_levels {
            // 根据粒度级别创建相应的索引配置
            let index_config = IndexConfig {
                index_type: config.index_type.clone(),
                dimensions: config.dimensions,
                max_elements: (config.max_elements as f32 * level.sampling_ratio) as usize,
                quantization_bits: level.quantization_bits,
                n_clusters: level.n_clusters,
                extra_params: level.extra_params.clone(),
            };
            
            // 创建索引实例
            let index = Index::create(index_config)?;
            layers.push(Arc::new(RwLock::new(index)));
            
            // 初始化统计数据
            layer_stats.push(RwLock::new(LayerStats {
                query_count: 0,
                query_time_ms: Vec::with_capacity(config.stats_window_size),
                hit_count: 0,
                hit_rates: Vec::with_capacity(config.stats_window_size),
                recent_quality_scores: Vec::with_capacity(config.stats_window_size),
            }));
        }
        
        Ok(Self {
            config,
            layers,
            layer_stats,
            last_auto_tune: RwLock::new(Instant::now()),
            pipeline: None,
            global_stats: RwLock::new(IndexStats::default()),
        })
    }
    
    /// 设置数据处理管道
    pub fn set_pipeline(&mut self, pipeline: Arc<DataProcessPipeline>) {
        self.pipeline = Some(pipeline);
    }
    
    /// 向所有层添加向量
    pub fn add_vector(&self, id: u64, vector: &[f32], metadata: Option<HashMap<String, String>>) -> Result<()> {
        // 向每个粒度层级添加向量（根据采样率决定是否添加）
        for (i, level) in self.config.granularity_levels.iter().enumerate() {
            // 根据采样率决定是否添加到当前层
            if level.sampling_ratio >= 1.0 || 
               rand::random::<f32>() <= level.sampling_ratio {
                // 获取当前层的索引
                let index = self.layers[i].clone();
                let mut index = index.write().map_err(|_| Error::lock_error("无法获取索引写锁"))?;
                
                // 添加向量到当前层
                index.add_vector(id, vector, metadata.clone())?;
            }
        }
        
        // 更新全局统计信息
        let mut stats = self.global_stats.write().map_err(|_| Error::lock_error("无法获取统计信息写锁"))?;
        stats.total_vectors += 1;
        
        Ok(())
    }
    
    /// 从所有层删除向量
    pub fn delete_vector(&self, id: u64) -> Result<()> {
        let mut found = false;
        
        // 从每个层级删除向量
        for layer in &self.layers {
            let mut index = layer.write().map_err(|_| Error::lock_error("无法获取索引写锁"))?;
            if index.delete_vector(id)? {
                found = true;
            }
        }
        
        // 更新全局统计信息
        if found {
            let mut stats = self.global_stats.write().map_err(|_| Error::lock_error("无法获取统计信息写锁"))?;
            stats.total_vectors = stats.total_vectors.saturating_sub(1);
        }
        
        // 如果向量不存在于任何层，返回错误
        if !found {
            return Err(Error::not_found("向量不存在"));
        }
        
        Ok(())
    }
    
    /// 批量添加向量
    pub fn add_vectors_batch(&self, ids: &[u64], vectors: &[Vec<f32>], metadata_batch: Option<Vec<HashMap<String, String>>>) -> Result<()> {
        if ids.len() != vectors.len() {
            return Err(Error::param_error("ID数量与向量数量不匹配"));
        }
        
        if let Some(ref metadata_batch) = metadata_batch {
            if metadata_batch.len() != ids.len() {
                return Err(Error::param_error("元数据数量与向量数量不匹配"));
            }
        }
        
        // 为每个层准备批量添加的数据
        let len = ids.len();
        let mut layer_ids = vec![Vec::with_capacity(len); self.layers.len()];
        let mut layer_vectors = vec![Vec::with_capacity(len); self.layers.len()];
        let mut layer_metadata = vec![Vec::with_capacity(len); self.layers.len()];
        
        // 根据采样率分配向量到不同层
        for i in 0..len {
            let id = ids[i];
            let vector = &vectors[i];
            let metadata = metadata_batch.as_ref().map(|batch| batch[i].clone());
            
            for (j, level) in self.config.granularity_levels.iter().enumerate() {
                if level.sampling_ratio >= 1.0 || 
                   rand::random::<f32>() <= level.sampling_ratio {
                    layer_ids[j].push(id);
                    layer_vectors[j].push(vector.clone());
                    if let Some(ref meta) = metadata {
                        layer_metadata[j].push(meta.clone());
                    }
                }
            }
        }
        
        // 向每个层批量添加向量
        for i in 0..self.layers.len() {
            if !layer_ids[i].is_empty() {
                let index = self.layers[i].clone();
                let mut index = index.write().map_err(|_| Error::lock_error("无法获取索引写锁"))?;
                
                let metadata_option = if layer_metadata[i].is_empty() {
                    None
                } else {
                    Some(layer_metadata[i].clone())
                };
                
                index.add_vectors_batch(&layer_ids[i], &layer_vectors[i], metadata_option)?;
            }
        }
        
        // 更新全局统计信息
        let mut stats = self.global_stats.write().map_err(|_| Error::lock_error("无法获取统计信息写锁"))?;
        stats.total_vectors += len;
        
        Ok(())
    }
    
    /// 批量删除向量
    pub fn delete_vectors_batch(&self, ids: &[u64]) -> Result<usize> {
        let mut total_deleted = 0;
        
        // 从每个层级批量删除向量
        for layer in &self.layers {
            let mut index = layer.write().map_err(|_| Error::lock_error("无法获取索引写锁"))?;
            let deleted = index.delete_vectors_batch(ids)?;
            if deleted > total_deleted {
                total_deleted = deleted;
            }
        }
        
        // 更新全局统计信息
        if total_deleted > 0 {
            let mut stats = self.global_stats.write().map_err(|_| Error::lock_error("无法获取统计信息写锁"))?;
            stats.total_vectors = stats.total_vectors.saturating_sub(total_deleted);
        }
        
        Ok(total_deleted)
    }
    
    /// 执行向量搜索，采用多粒度策略
    pub fn search(&self, query: &[f32], k: usize, params: &SearchParams) -> Result<Vec<SearchResult>> {
        // 检查是否需要进行自动调优
        self.check_auto_tune()?;
        
        // 决策使用哪个层级进行搜索
        let layer_index = self.select_best_layer_for_query(query, k, params)?;
        
        // 获取选定的索引层
        let start_time = Instant::now();
        let layer = self.layers[layer_index].clone();
        let index = layer.read().map_err(|_| Error::lock_error("无法获取索引读锁"))?;
        
        // 执行搜索
        let results = index.search(query, k, params)?;
        let elapsed = start_time.elapsed();
        
        // 更新层统计信息
        let mut layer_stat = self.layer_stats[layer_index].write().map_err(|_| Error::lock_error("无法获取统计信息写锁"))?;
        layer_stat.query_count += 1;
        layer_stat.query_time_ms.push(elapsed.as_millis() as u64);
        
        // 控制统计数据窗口大小
        if layer_stat.query_time_ms.len() > self.config.stats_window_size {
            layer_stat.query_time_ms.remove(0);
        }
        
        // 如果有结果，记录命中
        if !results.is_empty() {
            layer_stat.hit_count += 1;
        }
        
        // 计算并记录命中率
        let hit_rate = layer_stat.hit_count as f32 / layer_stat.query_count as f32;
        layer_stat.hit_rates.push(hit_rate);
        if layer_stat.hit_rates.len() > self.config.stats_window_size {
            layer_stat.hit_rates.remove(0);
        }
        
        // 评估搜索质量
        let quality_score = self.evaluate_search_quality(&results);
        layer_stat.recent_quality_scores.push(quality_score);
        if layer_stat.recent_quality_scores.len() > self.config.stats_window_size {
            layer_stat.recent_quality_scores.remove(0);
        }
        
        // 更新全局统计
        let mut global_stats = self.global_stats.write().map_err(|_| Error::lock_error("无法获取统计信息写锁"))?;
        global_stats.total_searches += 1;
        global_stats.avg_search_time_ms = (global_stats.avg_search_time_ms * (global_stats.total_searches - 1) as f32 + 
                                          elapsed.as_millis() as f32) / global_stats.total_searches as f32;
        
        Ok(results)
    }
    
    /// 选择最合适的层级用于查询
    fn select_best_layer_for_query(&self, query: &[f32], k: usize, params: &SearchParams) -> Result<usize> {
        // 简单策略：根据查询复杂度和所需结果数量选择
        
        // 对于小规模查询(k小)，倾向于使用高粒度(较粗粒度)索引以获得更快的速度
        // 对于大规模查询(k大)，倾向于使用低粒度(较细粒度)索引以获得更高的准确度
        
        // 计算查询复杂度因子 (0-1之间)
        let k_factor = (k as f32).min(100.0) / 100.0;  // 归一化k值，最大考虑到100
        
        // 复杂查询的定义（基于参数和向量特征）
        let is_complex_query = params.ef_search > 100 || 
                              self.is_high_dimensional_query(query) ||
                              params.exact_search;
        
        let complexity_factor = if is_complex_query { k_factor } else { k_factor * 0.5 };
        
        // 计算各层的性能评分
        let mut layer_scores = Vec::with_capacity(self.layers.len());
        
        for (i, _) in self.config.granularity_levels.iter().enumerate() {
            let stats = self.layer_stats[i].read().map_err(|_| Error::lock_error("无法获取统计信息读锁"))?;
            
            // 计算平均查询时间（毫秒）
            let avg_query_time = if stats.query_time_ms.is_empty() {
                50.0 // 默认值
            } else {
                stats.query_time_ms.iter().sum::<u64>() as f32 / stats.query_time_ms.len() as f32
            };
            
            // 计算平均命中率
            let avg_hit_rate = if stats.hit_rates.is_empty() {
                0.8 // 默认值
            } else {
                stats.hit_rates.iter().sum::<f32>() / stats.hit_rates.len() as f32
            };
            
            // 计算平均质量评分
            let avg_quality = if stats.recent_quality_scores.is_empty() {
                0.7 // 默认值
            } else {
                stats.recent_quality_scores.iter().sum::<f32>() / stats.recent_quality_scores.len() as f32
            };
            
            // 计算速度评分 (越快得分越高)
            let speed_score = 1.0 / (1.0 + avg_query_time / 100.0);
            
            // 计算质量评分
            let quality_score = avg_hit_rate * avg_quality;
            
            // 根据查询复杂度调整权重
            let speed_weight = 1.0 - complexity_factor;
            let quality_weight = complexity_factor;
            
            // 计算总分
            let total_score = speed_weight * speed_score + quality_weight * quality_score;
            
            layer_scores.push(total_score);
        }
        
        // 找出得分最高的层
        let mut best_layer = 0;
        let mut max_score = f32::NEG_INFINITY;
        
        for (i, score) in layer_scores.iter().enumerate() {
            if *score > max_score {
                max_score = *score;
                best_layer = i;
            }
        }
        
        Ok(best_layer)
    }
    
    /// 评估查询是否为高维复杂查询
    fn is_high_dimensional_query(&self, query: &[f32]) -> bool {
        // 简单启发式方法：计算向量的非零元素比例和方差
        let mut non_zero_count = 0;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        
        for &v in query {
            if v != 0.0 {
                non_zero_count += 1;
            }
            sum += v;
            sum_sq += v * v;
        }
        
        let n = query.len();
        let non_zero_ratio = non_zero_count as f32 / n as f32;
        
        // 计算方差
        let mean = sum / n as f32;
        let variance = (sum_sq / n as f32) - (mean * mean);
        
        // 高维复杂查询的特征：非零元素比例高且方差大
        non_zero_ratio > 0.7 && variance > 0.1
    }
    
    /// 评估搜索结果质量
    fn evaluate_search_quality(&self, results: &[SearchResult]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }
        
        // 质量评估指标：
        // 1. 距离分布 - 好的结果应该有明显的距离梯度
        // 2. 相似度水平 - 第一个结果应该有较高的相似度
        
        // 获取相似度值
        let similarities: Vec<f32> = results.iter()
            .map(|r| 1.0 - r.distance)  // 转换距离为相似度
            .collect();
        
        // 计算相似度的统计特征
        let max_sim = similarities[0];  // 最高相似度
        let min_sim = if similarities.len() > 1 { 
            *similarities.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap() 
        } else { 
            max_sim 
        };
        let sim_range = max_sim - min_sim;
        
        // 计算相似度梯度 (一阶导数的平均值)
        let mut gradient_sum = 0.0;
        for i in 1..similarities.len() {
            gradient_sum += (similarities[i-1] - similarities[i]).abs();
        }
        let avg_gradient = if similarities.len() > 1 {
            gradient_sum / (similarities.len() - 1) as f32
        } else {
            0.0
        };
        
        // 综合评分 (范围0-1)：最高相似度的权重 + 相似度梯度的权重
        let top_sim_score = max_sim.max(0.0).min(1.0);  // 限制在0-1范围
        let gradient_score = (avg_gradient / 0.1).min(1.0);  // 归一化梯度，最大认为是0.1
        
        // 加权组合
        0.7 * top_sim_score + 0.3 * gradient_score
    }
    
    /// 检查是否需要执行自动调优
    fn check_auto_tune(&self) -> Result<()> {
        let mut last_tune = self.last_auto_tune.write().map_err(|_| Error::lock_error("无法获取自动调优时间锁"))?;
        let now = Instant::now();
        let elapsed = now.duration_since(*last_tune);
        
        if elapsed.as_secs() >= self.config.auto_tune_interval_secs {
            // 更新自动调优时间
            *last_tune = now;
            
            // 执行自动调优
            drop(last_tune); // 释放锁
            self.auto_tune()?;
        }
        
        Ok(())
    }
    
    /// 执行自动调优
    fn auto_tune(&self) -> Result<()> {
        // 收集每个层的性能数据
        let mut layer_performance = Vec::with_capacity(self.layers.len());
        
        for (i, _) in self.config.granularity_levels.iter().enumerate() {
            let stats = self.layer_stats[i].read().map_err(|_| Error::lock_error("无法获取统计信息读锁"))?;
            
            let avg_query_time = if stats.query_time_ms.is_empty() {
                50.0
            } else {
                stats.query_time_ms.iter().sum::<u64>() as f32 / stats.query_time_ms.len() as f32
            };
            
            let utilization = stats.query_count as f32 / 
                              self.global_stats.read().map_err(|_| Error::lock_error("无法获取全局统计信息读锁"))?.total_searches.max(1) as f32;
            
            layer_performance.push((i, avg_query_time, utilization));
        }
        
        // 分析性能数据，确定需要优化的层
        // 此处简单实现：对最常用但性能不佳的层进行优化
        layer_performance.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)  // 按使用率降序
        });
        
        // 优化使用率最高的、且查询时间较长的层
        for &(layer_idx, query_time, utilization) in &layer_performance {
            if utilization > 0.2 && query_time > 50.0 {  // 使用率>20%且查询时间>50ms
                // 获取该层索引
                let layer = self.layers[layer_idx].clone();
                let mut index = layer.write().map_err(|_| Error::lock_error("无法获取索引写锁"))?;
                
                // 对该层执行优化
                log::info!("自动优化索引层 {} (利用率: {:.2}%, 平均查询时间: {:.2}ms)", 
                          layer_idx, utilization * 100.0, query_time);
                
                // 执行实际的优化操作
                index.optimize()?;
                
                // 只优化一个层级，避免过多资源消耗
                break;
            }
        }
        
        Ok(())
    }
    
    /// 获取当前状态
    pub fn get_stats(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut result = HashMap::new();
        
        // 全局统计信息
        let global_stats = self.global_stats.read().map_err(|_| Error::lock_error("无法获取全局统计信息"))?;
        result.insert("total_vectors".to_string(), serde_json::json!(global_stats.total_vectors));
        result.insert("total_searches".to_string(), serde_json::json!(global_stats.total_searches));
        result.insert("avg_search_time_ms".to_string(), serde_json::json!(global_stats.avg_search_time_ms));
        
        // 各层统计信息
        let mut layers_stats = Vec::with_capacity(self.layers.len());
        for (i, level) in self.config.granularity_levels.iter().enumerate() {
            let layer_stat = self.layer_stats[i].read().map_err(|_| Error::lock_error("无法获取层统计信息"))?;
            let layer_index = self.layers[i].read().map_err(|_| Error::lock_error("无法获取索引读锁"))?;
            let index_stats = layer_index.get_stats()?;
            
            let avg_query_time = if layer_stat.query_time_ms.is_empty() {
                0.0
            } else {
                layer_stat.query_time_ms.iter().sum::<u64>() as f32 / layer_stat.query_time_ms.len() as f32
            };
            
            let hit_rate = if layer_stat.query_count == 0 {
                0.0
            } else {
                layer_stat.hit_count as f32 / layer_stat.query_count as f32
            };
            
            let avg_quality = if layer_stat.recent_quality_scores.is_empty() {
                0.0
            } else {
                layer_stat.recent_quality_scores.iter().sum::<f32>() / layer_stat.recent_quality_scores.len() as f32
            };
            
            layers_stats.push(serde_json::json!({
                "level_id": level.level_id,
                "name": level.name,
                "quantization_bits": level.quantization_bits,
                "n_clusters": level.n_clusters,
                "sampling_ratio": level.sampling_ratio,
                "query_count": layer_stat.query_count,
                "avg_query_time_ms": avg_query_time,
                "hit_rate": hit_rate,
                "avg_quality_score": avg_quality,
                "index_stats": index_stats,
            }));
        }
        
        result.insert("layers".to_string(), serde_json::json!(layers_stats));
        
        Ok(result)
    }
    
    /// 保存索引到存储
    pub fn save(&self, storage: &dyn Storage, path: &str) -> Result<()> {
        // 创建保存目录
        storage.create_directory(path)?;
        
        // 保存配置信息
        let config_json = serde_json::to_string(&self.config).map_err(|e| Error::serialization(e.to_string()))?;
        storage.write_text(&format!("{}/config.json", path), &config_json)?;
        
        // 保存每个层级的索引
        for (i, level) in self.config.granularity_levels.iter().enumerate() {
            let layer_path = format!("{}/layer_{}", path, level.level_id);
            let layer = self.layers[i].read().map_err(|_| Error::lock_error("无法获取索引读锁"))?;
            layer.save(storage, &layer_path)?;
        }
        
        Ok(())
    }
    
    /// 从存储加载索引
    pub fn load(storage: &dyn Storage, path: &str) -> Result<Self> {
        // 加载配置信息
        let config_json = storage.read_text(&format!("{}/config.json", path))?;
        let config: GranularIndexConfig = serde_json::from_str(&config_json).map_err(|e| Error::serialization(e.to_string()))?;
        
        // 创建索引实例
        let mut index = Self::new(config.clone())?;
        
        // 加载每个层级的索引
        for (i, level) in config.granularity_levels.iter().enumerate() {
            let layer_path = format!("{}/layer_{}", path, level.level_id);
            if storage.exists(&layer_path)? {
                let index_config = IndexConfig {
                    index_type: config.index_type.clone(),
                    dimensions: config.dimensions,
                    max_elements: (config.max_elements as f32 * level.sampling_ratio) as usize,
                    quantization_bits: level.quantization_bits,
                    n_clusters: level.n_clusters,
                    extra_params: level.extra_params.clone(),
                };
                
                let mut layer_index = Index::create(index_config)?;
                layer_index.load(storage, &layer_path)?;
                
                index.layers[i] = Arc::new(RwLock::new(layer_index));
            }
        }
        
        Ok(index)
    }
} 