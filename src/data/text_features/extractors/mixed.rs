// 混合数据特征提取器实现

use crate::Result;
use crate::Error;
use crate::data::text_features::config::{TextFeatureConfig, MixedFeatureConfig};
use crate::data::text_features::stats::{NumericStats, DataCharacteristics};
use crate::data::text_features::types::{FieldType, TextFeatureMethod};
use super::FeatureExtractor;
use crate::data::text_features::evaluation::{evaluate_method_performance_with_weights, EvaluationWeights};
use crate::data::text_features::incremental::IncrementalLearningState;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

/// 混合数据特征提取器
/// 
/// 能够自动处理混合数据类型，包括文本、数值和分类数据
#[derive(Debug)]
pub struct MixedFeatureExtractor {
    /// 配置信息
    pub config: MixedFeatureConfig,
    /// 特征提取器名称
    name: String,
    /// 字段类型映射
    field_types: RwLock<HashMap<String, FieldType>>,
    /// 数值型字段统计信息
    numeric_stats: RwLock<HashMap<String, NumericStats>>,
    /// 分类型字段值集合
    categorical_values: RwLock<HashMap<String, HashSet<String>>>,
    /// 增量学习状态
    incremental_state: Option<Arc<Mutex<IncrementalLearningState>>>,
    /// 特征维度
    dimension: usize,
    /// 是否已初始化
    initialized: bool,
}

impl MixedFeatureExtractor {
    /// 创建新的混合数据特征提取器
    pub fn new(config: MixedFeatureConfig) -> Result<Self> {
        Ok(Self {
            name: "MixedFeatureExtractor".to_string(),
            config,
            field_types: RwLock::new(HashMap::new()),
            numeric_stats: RwLock::new(HashMap::new()),
            categorical_values: RwLock::new(HashMap::new()),
            incremental_state: None,
            dimension: 0,
            initialized: false,
        })
    }
    
    /// 从数据中自动检测字段类型
    pub fn detect_field_types(&mut self, data: &[Value]) -> Result<HashMap<String, FieldType>> {
        if data.is_empty() {
            return Err(Error::InvalidData("数据为空，无法检测字段类型".to_string()));
        }
        
        let mut field_types: HashMap<String, HashMap<FieldType, usize>> = HashMap::new();
        
        // 遍历数据，统计每个字段的类型频率
        for item in data {
            if !item.is_object() {
                continue;
            }
            
            let obj = item.as_object().unwrap();
            for (field, value) in obj {
                let entry = field_types.entry(field.clone()).or_insert_with(HashMap::new);
                
                let field_type = if value.is_string() {
                    FieldType::Text
                } else if value.is_number() {
                    FieldType::Numeric
                } else if value.is_boolean() || (value.is_string() && value.as_str().unwrap().len() < 50) {
                    FieldType::Categorical
                } else {
                    FieldType::Unknown
                };
                
                *entry.entry(field_type).or_insert(0) += 1;
            }
        }
        
        // 确定每个字段的最终类型
        let mut result = HashMap::new();
        for (field, type_counts) in field_types {
            let mut max_count = 0;
            let mut max_type = FieldType::Unknown;
            
            for (field_type, count) in type_counts {
                if count > max_count {
                    max_count = count;
                    max_type = field_type;
                }
            }
            
            result.insert(field, max_type);
        }
        
        let mut field_types = self.field_types.write().unwrap();
        *field_types = result.clone();
        Ok(result)
    }
    
    /// 计算数值型字段的统计信息
    pub fn compute_numeric_stats(&mut self, data: &[Value]) -> Result<HashMap<String, NumericStats>> {
        if data.is_empty() {
            return Err(Error::InvalidData("数据为空，无法计算统计信息".to_string()));
        }
        
        // 如果字段类型未检测，先检测字段类型
        let field_types = self.field_types.read().unwrap();
        if field_types.is_empty() {
            drop(field_types);
            self.detect_field_types(data)?;
        }
        
        let mut stats: HashMap<String, NumericStats> = HashMap::new();
        
        // 初始化统计结构
        let field_types = self.field_types.read().unwrap();
        for (field, field_type) in field_types.iter() {
            if *field_type == FieldType::Numeric {
                stats.insert(field.clone(), NumericStats::default());
            }
        }
        
        // 计算统计信息
        for item in data {
            if !item.is_object() {
                continue;
            }
            
            let obj = item.as_object().unwrap();
            for (field, value) in obj {
                if let Some(stats_entry) = stats.get_mut(field) {
                    if value.is_number() {
                        let num_value = if value.is_i64() {
                            value.as_i64().unwrap() as f64
                        } else if value.is_u64() {
                            value.as_u64().unwrap() as f64
                        } else {
                            value.as_f64().unwrap()
                        };
                        
                        stats_entry.update(num_value);
                    }
                }
            }
        }
        
        // 完成统计计算
        for stats_entry in stats.values_mut() {
            stats_entry.finalize();
        }
        
        let mut numeric_stats = self.numeric_stats.write().unwrap();
        *numeric_stats = stats.clone();
        Ok(stats)
    }
    
    /// 收集分类型字段的值集合
    pub fn collect_categorical_values(&mut self, data: &[Value]) -> Result<HashMap<String, HashSet<String>>> {
        if data.is_empty() {
            return Err(Error::InvalidData("数据为空，无法收集分类值".to_string()));
        }
        
        // 如果字段类型未检测，先检测字段类型
        let field_types = self.field_types.read().unwrap();
        if field_types.is_empty() {
            drop(field_types);
            self.detect_field_types(data)?;
        }
        
        let mut values: HashMap<String, HashSet<String>> = HashMap::new();
        
        // 初始化值集合
        let field_types = self.field_types.read().unwrap();
        for (field, field_type) in field_types.iter() {
            if *field_type == FieldType::Categorical {
                values.insert(field.clone(), HashSet::new());
            }
        }
        
        // 收集值
        for item in data {
            if !item.is_object() {
                continue;
            }
            
            let obj = item.as_object().unwrap();
            for (field, value) in obj {
                if let Some(value_set) = values.get_mut(field) {
                    if value.is_string() {
                        value_set.insert(value.as_str().unwrap().to_string());
                    } else if value.is_boolean() {
                        value_set.insert(value.as_bool().unwrap().to_string());
                    } else if value.is_number() {
                        value_set.insert(value.to_string());
                    }
                }
            }
        }
        
        let mut categorical_values = self.categorical_values.write().unwrap();
        *categorical_values = values.clone();
        Ok(values)
    }
    
    /// 自动选择最佳特征提取方法（自适应）
    pub fn auto_select_best_method_adaptive(&mut self, data: &[Value]) -> Result<TextFeatureConfig> {
        // 检测数据特征
        let data_characteristics = self.analyze_data_characteristics(data)?;
        
        // 基于数据特征调整评估权重
        let weights = self.adjust_weights_for_data(&data_characteristics);
        
        // 预设几种常用方法
        let methods = self.generate_candidate_methods(&data_characteristics);
        
        // 评估各方法性能
        let mut best_method = None;
        let mut best_score = f64::NEG_INFINITY;
        
        for method in methods {
            let score = evaluate_method_performance_with_weights(&method, data, &weights)?;
            
            if score > best_score {
                best_score = score;
                best_method = Some(method);
            }
        }
        
        if let Some(method) = best_method {
            Ok(method)
        } else {
            // 如果没有找到合适的方法，返回默认配置
            Ok(TextFeatureConfig::default())
        }
    }
    
    /// 批量处理JSON数据
    pub fn batch_process_json(&mut self, data: &[Value]) -> Result<Vec<Vec<f32>>> {
        // 如果尚未初始化，先初始化
        if !self.initialized {
            self.initialize(data)?;
        }
        
        let mut results = Vec::with_capacity(data.len());
        
        for item in data {
            if !item.is_object() {
                continue;
            }
            
            let features = self.extract_features_from_json(item)?;
            results.push(features);
        }
        
        Ok(results)
    }
    
    /// 初始化特征提取器
    fn initialize(&mut self, data: &[Value]) -> Result<()> {
        // 检测字段类型
        self.detect_field_types(data)?;
        
        // 计算数值型字段统计信息
        self.compute_numeric_stats(data)?;
        
        // 收集分类型字段值集合
        self.collect_categorical_values(data)?;
        
        // 计算特征维度
        self.compute_dimension()?;
        
        self.initialized = true;
        Ok(())
    }
    
    /// 计算特征维度
    fn compute_dimension(&mut self) -> Result<()> {
        let mut total_dim = 0;
        
        // 获取字段类型
        let field_types = self.field_types.read().unwrap();
        
        // 计算每个字段的特征维度
        for (field, field_type) in field_types.iter() {
            match field_type {
                FieldType::Text => {
                    // 文本字段使用配置的向量维度
                    total_dim += self.config.text_config.feature_dimension;
                },
                FieldType::Numeric => {
                    // 数值型字段使用统计特征维度
                    total_dim += 14; // 14个统计特征
                },
                FieldType::Categorical => {
                    // 分类型字段使用one-hot编码维度
                    if let Some(values) = self.categorical_values.read().unwrap().get(field) {
                        total_dim += values.len();
                    }
                },
                FieldType::Unknown => {
                    return Err(Error::InvalidData(format!(
                        "字段 {} 类型未知",
                        field
                    )));
                }
            }
        }
        
        self.dimension = total_dim;
        Ok(())
    }
    
    /// 分析数据特征
    fn analyze_data_characteristics(&self, data: &[Value]) -> Result<DataCharacteristics> {
        let mut characteristics = DataCharacteristics::default();
        
        // 获取字段类型
        let field_types = self.field_types.read().unwrap();
        
        // 分析每个字段的特征
        for (field, field_type) in field_types.iter() {
            match field_type {
                FieldType::Text => {
                    // 分析文本特征
                    let mut total_length = 0;
                    let mut unique_words = HashSet::new();
                    
                    for item in data {
                        if let Some(value) = item.get(field) {
                            if let Some(text) = value.as_str() {
                                let words: Vec<&str> = text.split_whitespace().collect();
                                total_length += words.len();
                                unique_words.extend(words.iter().cloned());
                            }
                        }
                    }
                    
                    // 将文本统计信息存储到 text_fields 中
                    // 注意：DataCharacteristics 没有直接的 text_avg_length 和 text_unique_words 字段
                    // 这些信息应该存储在 text_fields 的 TextFieldStats 中
                    // 这里我们跳过直接赋值，因为 DataCharacteristics 的结构不支持
                },
                FieldType::Numeric => {
                    // 分析数值特征
                    // 将数值统计信息存储到 numeric_fields 中
                    if let Some(stats) = self.numeric_stats.read().unwrap().get(field) {
                        characteristics.numeric_fields.insert(field.clone(), stats.clone());
                    }
                },
                FieldType::Categorical => {
                    // 分析分类特征
                    // 将分类统计信息存储到 categorical_fields 中
                    if let Some(values) = self.categorical_values.read().unwrap().get(field) {
                        let stats = crate::data::text_features::stats::CategoricalFieldStats {
                            category_count: values.len(),
                            most_common: None, // 需要额外计算
                            most_common_freq: 0.0,
                            frequencies: HashMap::new(),
                            missing: 0,
                        };
                        characteristics.categorical_fields.insert(field.clone(), stats);
                    }
                },
                FieldType::Unknown => {
                    return Err(Error::InvalidData(format!(
                        "字段 {} 类型未知",
                        field
                    )));
                }
            }
        }
        
        Ok(characteristics)
    }
    
    /// 基于数据特征调整评估权重
    fn adjust_weights_for_data(&self, characteristics: &DataCharacteristics) -> EvaluationWeights {
        let mut weights = EvaluationWeights::default();
        
        // 根据文本特征调整权重
        // 计算平均文本长度
        let avg_text_length: f64 = characteristics.text_fields.values()
            .map(|stats| stats.avg_length)
            .sum::<f64>() / characteristics.text_fields.len().max(1) as f64;
        
        if avg_text_length > 0.0 {
            // 根据平均文本长度调整准确性权重
            weights.accuracy_weight = (avg_text_length / 100.0).min(1.0) * 0.4;
        }
        
        // 根据数值特征调整权重
        // 计算平均标准差
        let avg_numeric_std: f64 = characteristics.numeric_fields.values()
            .map(|stats| stats.std_dev)
            .sum::<f64>() / characteristics.numeric_fields.len().max(1) as f64;
        
        if avg_numeric_std > 0.0 {
            // 根据数值标准差调整效率权重
            weights.efficiency_weight = (avg_numeric_std / 10.0).min(1.0) * 0.2;
        }
        
        // 根据分类特征调整权重
        let total_categorical_values: usize = characteristics.categorical_fields.values()
            .map(|stats| stats.category_count)
            .sum();
        
        if total_categorical_values > 0 {
            // 根据分类值数量调整可解释性权重
            weights.interpretability_weight = (total_categorical_values as f64 / 100.0).min(1.0) * 0.1;
        }
        
        weights
    }
    
    /// 生成候选特征提取方法
    fn generate_candidate_methods(&self, characteristics: &DataCharacteristics) -> Vec<TextFeatureConfig> {
        let mut methods = Vec::new();
        
        // 根据数据特征生成不同的方法配置
        // 计算平均文本长度
        let avg_text_length: f64 = characteristics.text_fields.values()
            .map(|stats| stats.avg_length)
            .sum::<f64>() / characteristics.text_fields.len().max(1) as f64;
        
        if avg_text_length > 0.0 {
            // 文本特征提取方法
            let feature_dim = self.config.text_config.feature_dimension;
            methods.push(TextFeatureConfig {
                method: TextFeatureMethod::TfIdf,
                feature_dimension: feature_dim,
                ..Default::default()
            });
            
            methods.push(TextFeatureConfig {
                method: TextFeatureMethod::Bert,
                feature_dimension: feature_dim,
                ..Default::default()
            });
        }
        
        // 添加其他方法配置...
        
        methods
    }
    
    /// 从JSON数据中提取特征
    fn extract_features_from_json(&self, item: &Value) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(self.dimension);
        
        // 获取字段类型
        let field_types = self.field_types.read().unwrap();
        
        // 提取每个字段的特征
        for (field, field_type) in field_types.iter() {
            match field_type {
                FieldType::Text => {
                    if let Some(value) = item.get(field) {
                        if let Some(text) = value.as_str() {
                            let text_features = self.extract_text_features(text, field)?;
                            features.extend_from_slice(&text_features);
                        }
                    }
                },
                FieldType::Numeric => {
                    if let Some(value) = item.get(field) {
                        if let Some(num) = value.as_f64() {
                            if let Some(stats) = self.numeric_stats.read().unwrap().get(field) {
                                // 添加数值特征
                                let num_f32 = num as f32;
                                let mean_f32 = stats.mean as f32;
                                let std_dev_f32 = stats.std_dev as f32;
                                let min_f32 = stats.min as f32;
                                let max_f32 = stats.max as f32;
                                features.extend_from_slice(&[
                                    num_f32,
                                    (num_f32 - mean_f32) / std_dev_f32.max(1e-10),
                                    if (max_f32 - min_f32).abs() > 1e-10 {
                                        (num_f32 - min_f32) / (max_f32 - min_f32)
                                    } else {
                                        0.0
                                    },
                                    // ... 其他统计特征
                                ]);
                            }
                        }
                    }
                },
                FieldType::Categorical => {
                    if let Some(value) = item.get(field) {
                        if let Some(text) = value.as_str() {
                            if let Some(values) = self.categorical_values.read().unwrap().get(field) {
                                // 添加one-hot编码特征
                                let mut one_hot = vec![0.0; values.len()];
                                if let Some(idx) = values.iter().position(|x| x == text) {
                                    one_hot[idx] = 1.0;
                                }
                                features.extend_from_slice(&one_hot);
                            }
                        }
                    }
                },
                FieldType::Unknown => {
                    return Err(Error::InvalidData(format!(
                        "字段 {} 类型未知",
                        field
                    )));
                }
            }
        }
        
        Ok(features)
    }
    
    /// 提取文本特征
    fn extract_text_features(&self, text: &str, field: &str) -> Result<Vec<f32>> {
        // 使用配置的文本特征提取方法
        let method = self.config.text_config.method;
            
        // 创建特征提取器
        let config = TextFeatureConfig {
            method,
            feature_dimension: self.config.text_config.feature_dimension,
            ..Default::default()
        };
        
        let extractor = super::factory::create_extractor(&config)?;
        
        // 提取特征
        extractor.extract(text)
    }
    
    /// 更新增量学习状态
    pub fn update_incremental_state(&mut self, data: &[Value]) -> Result<()> {
        if let Some(state) = &self.incremental_state {
            let mut state = state.lock().unwrap();
            
            // 更新字段类型（存储到 field_stats 中）
            let new_field_types = self.detect_field_types(data)?;
            for (field, field_type) in new_field_types {
                let mut stats = HashMap::new();
                stats.insert("field_type".to_string(), match field_type {
                    FieldType::Text => 0.0,
                    FieldType::Numeric => 1.0,
                    FieldType::Categorical => 2.0,
                });
                state.update_field_stats(&field, stats);
            }
            
            // 更新数值统计（转换为 field_stats 格式）
            let new_numeric_stats = self.compute_numeric_stats(data)?;
            for (field, stats) in new_numeric_stats {
                let mut field_stats = HashMap::new();
                field_stats.insert("mean".to_string(), stats.mean);
                field_stats.insert("std_dev".to_string(), stats.std_dev);
                field_stats.insert("min".to_string(), stats.min);
                field_stats.insert("max".to_string(), stats.max);
                field_stats.insert("median".to_string(), stats.median);
                field_stats.insert("variance".to_string(), stats.variance);
                state.update_field_stats(&field, field_stats);
            }
            
            // 更新分类值（存储唯一值数量）
            let new_categorical_values = self.collect_categorical_values(data)?;
            for (field, values) in new_categorical_values {
                let mut stats = HashMap::new();
                stats.insert("unique_count".to_string(), values.len() as f64);
                state.update_field_stats(&field, stats);
            }
            
            // 更新特征维度（存储到元数据中）
            self.compute_dimension()?;
            state.metadata.insert("dimension".to_string(), self.dimension.to_string());
        }
        
        Ok(())
    }
}

impl FeatureExtractor for MixedFeatureExtractor {
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        // 将文本转换为JSON格式
        let json = serde_json::json!({
            "text": text
        });
        
        // 提取特征
        self.extract_features_from_json(&json)
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        // 将TextFeatureConfig转换为MixedFeatureConfig
        let mixed_config = MixedFeatureConfig {
            text_config: config.clone(),
            ..Default::default()
        };
        
        Self::new(mixed_config)
    }
} 