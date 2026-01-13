use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// 增量学习配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalLearningConfig {
    /// 记忆衰减因子 (0-1)，控制旧知识的保留率
    pub decay_factor: f64,
    /// 学习率，控制新知识的吸收速度
    pub learning_rate: f64,
    /// 最大词汇表大小
    pub max_vocabulary_size: usize,
    /// 特征重要性阈值，低于此值的特征可能被修剪
    pub importance_threshold: f64,
    /// 批量更新大小
    pub batch_size: usize,
    /// 是否使用滑动窗口
    pub use_sliding_window: bool,
    /// 滑动窗口大小 (样本数)
    pub window_size: usize,
    /// 是否使用正则化
    pub use_regularization: bool,
    /// 正则化强度
    pub regularization_strength: f64,
    /// 是否启用知识蒸馏
    pub enable_knowledge_distillation: bool,
}

impl Default for IncrementalLearningConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.95,
            learning_rate: 0.1,
            max_vocabulary_size: 50000,
            importance_threshold: 0.01,
            batch_size: 32,
            use_sliding_window: false,
            window_size: 1000,
            use_regularization: true,
            regularization_strength: 0.01,
            enable_knowledge_distillation: false,
        }
    }
}

/// 增量学习状态
/// 
/// 用于存储特征提取器的增量学习状态，支持在线学习和增量更新
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalLearningState {
    /// 已处理的样本数
    pub processed_samples: usize,
    /// 增量更新计数
    pub update_count: usize,
    /// 词汇表
    pub vocabulary: HashMap<String, usize>,
    /// 文档频率（每个词出现在多少文档中）
    pub document_frequencies: HashMap<String, usize>,
    /// 字段统计信息
    pub field_stats: HashMap<String, HashMap<String, f64>>,
    /// 特征重要性
    pub feature_importance: HashMap<String, f64>,
    /// 最后更新时间
    pub last_update_timestamp: u64,
    /// 状态是否有效
    pub is_valid: bool,
    /// 累积错误率
    pub cumulative_error_rate: f64,
    /// 其他元数据
    pub metadata: HashMap<String, String>,
    
    // 新增字段
    /// 特征向量记忆 - 保存历史数据的特征表示
    pub feature_memory: HashMap<String, Vec<f64>>,
    /// 记忆样本权重 - 控制历史样本的影响力
    pub memory_weights: HashMap<String, f64>,
    /// 新样本缓冲区 - 用于批量更新
    pub new_samples_buffer: Vec<String>,
    /// 滑动窗口样本ID
    pub window_sample_ids: Vec<String>,
    /// 遗忘样本计数
    pub forgotten_samples: usize,
    /// 配置信息
    pub config: IncrementalLearningConfig,
    /// 统计特征协方差矩阵 (简化版，仅存储选定特征对的协方差)
    pub feature_covariance: HashMap<(String, String), f64>,
    /// 均值跟踪 - 跟踪特征均值变化
    pub feature_means: HashMap<String, f64>,
    /// 方差跟踪 - 跟踪特征方差变化
    pub feature_variances: HashMap<String, f64>,
    /// 模型版本历史 - 用于跟踪模型变化并支持回滚
    pub version_history: Vec<ModelVersionInfo>,
    /// 最佳验证性能
    pub best_validation_performance: f64,
    /// 知识增长曲线 - 记录知识积累过程
    pub knowledge_growth: Vec<(u64, f64)>,
}

/// 模型版本信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersionInfo {
    /// 版本标识
    pub version_id: String,
    /// 时间戳
    pub timestamp: u64,
    /// 处理样本数
    pub samples_seen: usize,
    /// 性能指标
    pub performance_metrics: HashMap<String, f64>,
    /// 重要特征集合 (Top-N)
    pub important_features: Vec<String>,
    /// 描述
    pub description: String,
}

impl IncrementalLearningState {
    /// 创建新的增量学习状态
    pub fn new() -> Self {
        Self {
            processed_samples: 0,
            update_count: 0,
            vocabulary: HashMap::new(),
            document_frequencies: HashMap::new(),
            field_stats: HashMap::new(),
            feature_importance: HashMap::new(),
            last_update_timestamp: current_timestamp(),
            is_valid: true,
            cumulative_error_rate: 0.0,
            metadata: HashMap::new(),
            
            // 新增字段初始化
            feature_memory: HashMap::new(),
            memory_weights: HashMap::new(),
            new_samples_buffer: Vec::new(),
            window_sample_ids: Vec::new(),
            forgotten_samples: 0,
            config: IncrementalLearningConfig::default(),
            feature_covariance: HashMap::new(),
            feature_means: HashMap::new(),
            feature_variances: HashMap::new(),
            version_history: Vec::new(),
            best_validation_performance: 0.0,
            knowledge_growth: Vec::new(),
        }
    }
    
    /// 添加新文档
    pub fn add_document(&mut self, text: &str, document_id: Option<String>) {
        self.processed_samples += 1;
        
        // 更新词汇表和文档频率
        let mut seen_words = std::collections::HashSet::new();
        
        for word in text.split_whitespace() {
            let word = word.to_lowercase();
            
            // 更新词汇表
            let count = self.vocabulary.entry(word.clone()).or_insert(0);
            *count += 1;
            
            // 对于文档频率，每个词在每个文档中只统计一次
            seen_words.insert(word);
        }
        
        // 更新文档频率
        for word in seen_words {
            let count = self.document_frequencies.entry(word).or_insert(0);
            *count += 1;
        }
        
        // 将文档添加到批处理缓冲区
        if let Some(id) = document_id {
            self.new_samples_buffer.push(id.clone());
            
            // 如果使用滑动窗口，更新窗口
            if self.config.use_sliding_window {
                self.window_sample_ids.push(id);
                
                // 如果窗口超过大小，移除最旧的样本
                if self.window_sample_ids.len() > self.config.window_size {
                    let old_id = self.window_sample_ids.remove(0);
                    self.forgotten_samples += 1;
                    // 移除相关记忆
                    self.feature_memory.remove(&old_id);
                    self.memory_weights.remove(&old_id);
                }
            }
        }
        
        // 如果缓冲区达到批处理大小，执行批处理更新
        if self.new_samples_buffer.len() >= self.config.batch_size {
            self.batch_update();
        }
        
        self.update_count += 1;
        self.last_update_timestamp = current_timestamp();
    }
    
    /// 批量更新处理
    fn batch_update(&mut self) {
        if self.new_samples_buffer.is_empty() {
            return;
        }
        
        // 应用记忆衰减 - 降低旧样本的权重
        for weight in self.memory_weights.values_mut() {
            *weight *= self.config.decay_factor;
        }
        
        // 为新样本分配记忆权重
        let new_weight = self.config.learning_rate;
        for sample_id in &self.new_samples_buffer {
            self.memory_weights.insert(sample_id.clone(), new_weight);
        }
        
        // 清空缓冲区
        self.new_samples_buffer.clear();
        
        // 更新知识增长曲线
        let knowledge_score = self.estimate_knowledge_coverage();
        self.knowledge_growth.push((current_timestamp(), knowledge_score));
        
        // 如果词汇表超过最大大小，执行修剪
        if self.vocabulary.len() > self.config.max_vocabulary_size {
            self.prune_vocabulary();
        }
        
        // 创建新的模型版本
        self.create_version_snapshot("批量更新");
    }
    
    /// 估计知识覆盖度
    fn estimate_knowledge_coverage(&self) -> f64 {
        // 简化估计 - 基于词汇表大小和样本数
        let vocabulary_factor = (self.vocabulary.len() as f64).sqrt() / 100.0;
        let samples_factor = (self.processed_samples as f64).ln() / 10.0;
        
        (vocabulary_factor + samples_factor).min(1.0)
    }
    
    /// 创建模型版本快照
    pub fn create_version_snapshot(&mut self, description: &str) {
        let version_id = format!("v{}", self.update_count);
        let timestamp = current_timestamp();
        
        // 获取重要特征
        let mut important_features: Vec<(String, f64)> = self.feature_importance.iter()
            .filter(|(_, &importance)| importance > self.config.importance_threshold)
            .map(|(name, &importance)| (name.clone(), importance))
            .collect();
        
        important_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let top_features: Vec<String> = important_features.iter()
            .take(20) // 只保留前20个重要特征
            .map(|(name, _)| name.clone())
            .collect();
            
        // 创建性能指标
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 1.0 - self.cumulative_error_rate);
        metrics.insert("vocabulary_size".to_string(), self.vocabulary.len() as f64);
        metrics.insert("knowledge_coverage".to_string(), self.estimate_knowledge_coverage());
        
        let version_info = ModelVersionInfo {
            version_id,
            timestamp,
            samples_seen: self.processed_samples,
            performance_metrics: metrics,
            important_features: top_features,
            description: description.to_string(),
        };
        
        self.version_history.push(version_info);
        
        // 限制版本历史长度
        if self.version_history.len() > 10 {
            self.version_history.remove(0);
        }
    }
    
    /// 修剪词汇表 - 移除低频词和不重要特征
    pub fn prune_vocabulary(&mut self) {
        // 基于文档频率和特征重要性进行修剪
        let min_doc_freq = (self.processed_samples as f64 * 0.01) as usize; // 至少出现在1%的文档中
        
        // 标记要移除的词
        let words_to_remove: Vec<_> = self.vocabulary.keys()
            .filter(|word| {
                let doc_freq = self.document_frequencies.get(*word).cloned().unwrap_or(0);
                let importance = self.feature_importance.get(*word).cloned().unwrap_or(0.0);
                
                doc_freq < min_doc_freq || importance < self.config.importance_threshold
            })
            .cloned()
            .collect();
        
        // 移除低价值词
        for word in words_to_remove {
            self.vocabulary.remove(&word);
            self.document_frequencies.remove(&word);
            self.feature_importance.remove(&word);
        }
    }
    
    /// 更新字段统计信息 - 使用增量更新
    pub fn update_field_stats(&mut self, field: &str, stats: HashMap<String, f64>) {
        let field_stats = self.field_stats.entry(field.to_string()).or_insert_with(HashMap::new);
        
        for (key, value) in stats {
            let current = field_stats.entry(key.clone()).or_insert(0.0);
            // 使用配置的学习率进行增量更新
            *current = (*current * (1.0 - self.config.learning_rate)) + (value * self.config.learning_rate);
        }
    }
    
    /// 更新特征重要性 - 使用增量更新和正则化
    pub fn update_feature_importance(&mut self, feature: &str, importance: f64) {
        let current = self.feature_importance.entry(feature.to_string()).or_insert(0.0);
        
        // 基本的增量更新
        let raw_update = (*current * (1.0 - self.config.learning_rate)) + (importance * self.config.learning_rate);
        
        // 应用正则化 (L2)
        let regularized_update = if self.config.use_regularization {
            raw_update * (1.0 - self.config.regularization_strength)
        } else {
            raw_update
        };
        
        *current = regularized_update;
    }
    
    /// 计算时间衰减因子
    pub fn time_decay_factor(&self, last_used_timestamp: u64) -> f64 {
        let current = current_timestamp();
        let time_diff = current.saturating_sub(last_used_timestamp) as f64;
        
        // 随时间指数衰减
        (-time_diff / (30.0 * 86400.0)).exp() // 30天半衰期
    }
    
    /// 更新特征协方差
    pub fn update_feature_covariance(&mut self, feature1: &str, feature2: &str, value1: f64, value2: f64) {
        // 更新均值
        let mean1 = self.feature_means.entry(feature1.to_string()).or_insert(0.0);
        let mean2 = self.feature_means.entry(feature2.to_string()).or_insert(0.0);
        
        *mean1 = (*mean1 * (self.processed_samples as f64 - 1.0) + value1) / self.processed_samples as f64;
        *mean2 = (*mean2 * (self.processed_samples as f64 - 1.0) + value2) / self.processed_samples as f64;
        
        // 更新方差
        let var1 = self.feature_variances.entry(feature1.to_string()).or_insert(0.0);
        let var2 = self.feature_variances.entry(feature2.to_string()).or_insert(0.0);
        
        *var1 = (*var1 * (self.processed_samples as f64 - 1.0) + (value1 - *mean1).powi(2)) / self.processed_samples as f64;
        *var2 = (*var2 * (self.processed_samples as f64 - 1.0) + (value2 - *mean2).powi(2)) / self.processed_samples as f64;
        
        // 更新协方差
        let key = if feature1 < feature2 {
            (feature1.to_string(), feature2.to_string())
        } else {
            (feature2.to_string(), feature1.to_string())
        };
        
        let cov = self.feature_covariance.entry(key).or_insert(0.0);
        *cov = (*cov * (self.processed_samples as f64 - 1.0) + (value1 - *mean1) * (value2 - *mean2)) / self.processed_samples as f64;
    }
    
    /// 存储特征向量记忆
    pub fn store_feature_vector(&mut self, sample_id: &str, features: Vec<f64>) {
        self.feature_memory.insert(sample_id.to_string(), features);
        self.memory_weights.insert(sample_id.to_string(), 1.0);
    }
    
    /// 根据ID获取特征向量记忆
    pub fn get_feature_vector(&self, sample_id: &str) -> Option<&Vec<f64>> {
        self.feature_memory.get(sample_id)
    }
    
    /// 获取加权特征记忆 (用于模型预热)
    pub fn get_weighted_memory_features(&self) -> HashMap<String, Vec<f64>> {
        let mut weighted_features = HashMap::new();
        
        for (sample_id, features) in &self.feature_memory {
            if let Some(&weight) = self.memory_weights.get(sample_id) {
                if weight > 0.01 { // 忽略权重过小的记忆
                    let weighted: Vec<_> = features.iter().map(|&f| f * weight).collect();
                    weighted_features.insert(sample_id.clone(), weighted);
                }
            }
        }
        
        weighted_features
    }
    
    /// 合并两个增量学习状态
    pub fn merge(&mut self, other: &Self) {
        // 合并基本计数器
        self.processed_samples += other.processed_samples;
        self.update_count += other.update_count;
        
        // 合并词汇表和文档频率
        for (word, &count) in &other.vocabulary {
            let entry = self.vocabulary.entry(word.clone()).or_insert(0);
            *entry += count;
        }
        
        for (word, &count) in &other.document_frequencies {
            let entry = self.document_frequencies.entry(word.clone()).or_insert(0);
            *entry += count;
        }
        
        // 合并字段统计 - 简单取平均
        for (field, stats) in &other.field_stats {
            let self_stats = self.field_stats.entry(field.clone()).or_insert_with(HashMap::new);
            
            for (key, &value) in stats {
                let entry = self_stats.entry(key.clone()).or_insert(0.0);
                *entry = (*entry + value) / 2.0; // 简单平均
            }
        }
        
        // 合并特征重要性 - 加权平均
        let self_weight = self.processed_samples as f64 / (self.processed_samples + other.processed_samples) as f64;
        let other_weight = 1.0 - self_weight;
        
        for (feature, &importance) in &other.feature_importance {
            let entry = self.feature_importance.entry(feature.clone()).or_insert(0.0);
            *entry = (*entry * self_weight) + (importance * other_weight);
        }
        
        // 保留更新时间戳
        self.last_update_timestamp = current_timestamp();
        
        // 合并特征记忆 - 选择性合并
        for (id, features) in &other.feature_memory {
            if !self.feature_memory.contains_key(id) {
                self.feature_memory.insert(id.clone(), features.clone());
                
                if let Some(&weight) = other.memory_weights.get(id) {
                    self.memory_weights.insert(id.clone(), weight * 0.8); // 降低外部记忆权重
                }
            }
        }
        
        // 合并版本历史 - 只保留最近的N个版本
        let mut combined_versions = self.version_history.clone();
        combined_versions.extend(other.version_history.clone());
        
        combined_versions.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        if combined_versions.len() > 10 {
            combined_versions.truncate(10);
        }
        
        self.version_history = combined_versions;
        
        // 更新知识增长曲线
        let knowledge_score = self.estimate_knowledge_coverage();
        self.knowledge_growth.push((current_timestamp(), knowledge_score));
    }
    
    /// 知识蒸馏 - 压缩模型
    pub fn perform_knowledge_distillation(&mut self) -> bool {
        if !self.config.enable_knowledge_distillation {
            return false;
        }
        
        // 实现知识蒸馏
        // 1. 基于重要性保留最重要的特征
        let importance_threshold = self.config.importance_threshold * 2.0; // 提高蒸馏阈值
        
        // 标记要移除的低重要性特征
        let features_to_remove: Vec<_> = self.feature_importance.iter()
            .filter(|(_, &importance)| importance < importance_threshold)
            .map(|(feature, _)| feature.clone())
            .collect();
            
        // 移除低重要性特征
        for feature in features_to_remove {
            self.feature_importance.remove(&feature);
        }
        
        // 2. 整合冗余特征 (简化处理，实际应该基于协方差)
        let redundancy_threshold = 0.9; // 高协方差阈值
        let mut redundant_pairs = Vec::new();
        
        for ((f1, f2), &cov) in &self.feature_covariance {
            if cov > redundancy_threshold {
                redundant_pairs.push((f1.clone(), f2.clone()));
            }
        }
        
        // 处理冗余对
        for (f1, f2) in redundant_pairs {
            // 保留重要性更高的特征
            let importance1 = self.feature_importance.get(&f1).cloned().unwrap_or(0.0);
            let importance2 = self.feature_importance.get(&f2).cloned().unwrap_or(0.0);
            
            if importance1 >= importance2 {
                self.feature_importance.remove(&f2);
            } else {
                self.feature_importance.remove(&f1);
            }
        }
        
        // 3. 压缩特征记忆
        let memory_prune_threshold = 0.2;
        let memory_to_remove: Vec<_> = self.memory_weights.iter()
            .filter(|(_, &weight)| weight < memory_prune_threshold)
            .map(|(id, _)| id.clone())
            .collect();
            
        for id in memory_to_remove {
            self.memory_weights.remove(&id);
            self.feature_memory.remove(&id);
        }
        
        // 创建蒸馏版本
        self.create_version_snapshot("知识蒸馏优化");
        
        true
    }
    
    /// 回滚到指定版本
    pub fn rollback_to_version(&mut self, version_id: &str) -> bool {
        if let Some(pos) = self.version_history.iter().position(|v| v.version_id == version_id) {
            // 创建回滚记录
            self.create_version_snapshot(&format!("回滚前，目标版本: {}", version_id));
            
            // 简单回滚 - 实际应该恢复更多状态
            // 这里只回滚样本计数和时间戳
            let version = &self.version_history[pos];
            self.processed_samples = version.samples_seen;
            
            // 记录回滚事件
            self.metadata.insert("last_rollback".to_string(), version_id.to_string());
            self.metadata.insert("rollback_time".to_string(), current_timestamp().to_string());
            
            // 创建回滚后记录
            self.create_version_snapshot(&format!("回滚到版本: {}", version_id));
            
            return true;
        }
        
        false
    }
    
    /// 保持模型稳定性 - 避免过拟合和欠拟合
    pub fn maintain_stability(&mut self, validation_performance: f64) -> bool {
        let improved = validation_performance > self.best_validation_performance;
        
        if improved {
            // 如果性能提升，更新最佳性能并创建快照
            self.best_validation_performance = validation_performance;
            self.create_version_snapshot("性能提升，创建快照");
            return true;
        } else if self.best_validation_performance - validation_performance > 0.1 {
            // 如果性能显著下降，考虑回滚
            if let Some(best_version) = self.version_history.iter()
                .filter(|v| v.performance_metrics.get("accuracy").unwrap_or(&0.0) > &validation_performance)
                .max_by_key(|v| v.timestamp) {
                
                // 回滚到较好的版本
                return self.rollback_to_version(&best_version.version_id);
            }
        }
        
        false
    }
}

/// 获取当前时间戳
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
} 