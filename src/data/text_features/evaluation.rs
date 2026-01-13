use crate::error::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use crate::data::text_features::types::TextFeatureMethod;
use crate::data::text_features::config::TextFeatureConfig;
use crate::data::text_features::extractors::{create_extractor, FeatureExtractor as TextFeatureExtractor};
use crate::data::text_features::stats::DataCharacteristics;
use std::time::{Instant, Duration};
use std::collections::HashSet;
use serde::{Serialize, Deserialize};

/// 评估权重配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationWeights {
    /// 准确性权重
    pub accuracy_weight: f64,
    /// 效率权重
    pub efficiency_weight: f64,
    /// 可解释性权重
    pub interpretability_weight: f64,
    /// 内存使用权重
    pub memory_usage_weight: f64,
    /// 训练时间权重
    pub training_time_weight: f64,
    /// 性能权重
    pub performance: f64,
    /// 推理时间权重
    pub inference_time: f64,
}

impl Default for EvaluationWeights {
    fn default() -> Self {
        Self {
            accuracy_weight: 0.4,
            efficiency_weight: 0.2,
            interpretability_weight: 0.1,
            memory_usage_weight: 0.1,
            training_time_weight: 0.1,
            performance: 0.5,
            inference_time: 0.2,
        }
    }
}

/// 特征提取方法评估
/// 
/// 本模块提供了评估不同特征提取方法性能的功能
/// 
/// 评估特征提取方法性能（带权重）
/// 
/// 基于准确性、效率和可解释性评估特征提取方法的性能
/// 
/// # 参数
/// * `config` - 特征提取配置
/// * `data` - 测试数据
/// * `weights` - 评估权重
/// 
/// # 返回
/// * `Result<f64>` - 加权评分
pub fn evaluate_method_performance_with_weights(
    config: &TextFeatureConfig,
    data: &[Value],
    weights: &EvaluationWeights,
) -> Result<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    
    // 创建特征提取器
    let extractor = create_extractor(config)?;
    
    // 1. 评估准确性 (使用一些合理的代理指标)
    let accuracy_score = evaluate_accuracy(&*extractor, data)?;
    
    // 2. 评估效率
    let efficiency_score = evaluate_efficiency(&*extractor, data)?;
    
    // 3. 评估可解释性
    let interpretability_score = evaluate_interpretability(&*extractor, config.method);
    
    // 计算加权得分
    let weighted_score = 
        accuracy_score * weights.accuracy_weight +
        efficiency_score * weights.efficiency_weight +
        interpretability_score * weights.interpretability_weight;
    
    Ok(weighted_score)
}

/// 调整权重
/// 
/// 根据数据特征动态调整评估权重
/// 
/// # 参数
/// * `weights` - 原始权重
/// * `data_characteristics` - 数据特征
/// 
/// # 返回
/// * `EvaluationWeights` - 调整后的权重
pub fn adjust_weights(
    weights: &EvaluationWeights,
    data_characteristics: &DataCharacteristics,
) -> EvaluationWeights {
    let mut adjusted = EvaluationWeights {
        accuracy_weight: weights.accuracy_weight,
        efficiency_weight: weights.efficiency_weight,
        interpretability_weight: weights.interpretability_weight,
        memory_usage_weight: weights.memory_usage_weight,
        training_time_weight: weights.training_time_weight,
        performance: weights.performance,
        inference_time: weights.inference_time,
    };
    
    // 1. 根据数据规模调整权重
    let scale_factor = match data_characteristics.sample_count {
        n if n > 100000 => 2.0,   // 大规模数据
        n if n > 10000 => 1.5,    // 中等规模
        n if n > 1000 => 1.2,     // 小规模
        _ => 1.0,                 // 极小规模
    };
    adjusted.efficiency_weight *= scale_factor;
    adjusted.memory_usage_weight *= scale_factor;
    
    // 2. 根据数据复杂度调整权重
    let complexity_score = calculate_data_complexity(data_characteristics);
    let complexity_factor = match complexity_score {
        s if s > 0.8 => 1.8,      // 高复杂度
        s if s > 0.6 => 1.5,      // 中高复杂度
        s if s > 0.4 => 1.2,      // 中等复杂度
        s if s > 0.2 => 0.9,      // 低复杂度
        _ => 0.7,                 // 极低复杂度
    };
    adjusted.accuracy_weight *= complexity_factor;
    
    // 3. 根据数据噪声水平调整权重
    let noise_level = estimate_noise_level(data_characteristics);
    let noise_factor = match noise_level {
        n if n > 0.7 => 0.7,      // 高噪声
        n if n > 0.5 => 0.8,      // 中高噪声
        n if n > 0.3 => 0.9,      // 中等噪声
        _ => 1.0,                 // 低噪声
    };
    adjusted.accuracy_weight *= noise_factor;
    
    // 4. 根据任务类型调整权重
    if data_characteristics.is_primarily_classification() {
        adjusted.interpretability_weight *= 1.3; // 分类任务解释性更重要
    } else if data_characteristics.is_primarily_regression() {
        adjusted.accuracy_weight *= 1.2;        // 回归任务精度更重要
    } else if data_characteristics.is_primarily_clustering() {
        adjusted.efficiency_weight *= 1.1;      // 聚类任务效率更重要
    }
    
    // 5. 根据特征多样性调整权重
    let feature_diversity = calculate_feature_diversity(data_characteristics);
    if feature_diversity > 0.8 {
        // 特征多样性高，需要更好的特征集成能力
        adjusted.training_time_weight *= 1.2;
    }
    
    // 6. 根据时间敏感性调整权重
    if data_characteristics.has_time_sensitive_fields() {
        adjusted.training_time_weight *= 1.3;
        adjusted.efficiency_weight *= 1.2;
    }
    
    // 7. 根据字段类型比例进行更细粒度的调整
    adjust_weights_by_field_types(&mut adjusted, data_characteristics);
    
    // 归一化权重
    normalize_weights(&mut adjusted);
    
    adjusted
}

/// 根据字段类型调整权重
fn adjust_weights_by_field_types(weights: &mut EvaluationWeights, data: &DataCharacteristics) {
    // 文本比例高时调整
    if data.text_field_ratio > 0.7 {
        weights.accuracy_weight *= 1.2;
        weights.interpretability_weight *= 1.1;
    } 
    // 数值比例高时调整
    else if data.numeric_field_ratio > 0.7 {
        weights.efficiency_weight *= 1.15;
        weights.memory_usage_weight *= 0.9;
    } 
    // 分类比例高时调整
    else if data.categorical_field_ratio > 0.7 {
        weights.interpretability_weight *= 1.25;
        weights.memory_usage_weight *= 0.85;
    }
    // 混合数据调整
    else if data.is_highly_mixed() {
        weights.training_time_weight *= 1.2;
        weights.memory_usage_weight *= 1.1;
    }
}

/// 归一化权重确保总和为1
fn normalize_weights(weights: &mut EvaluationWeights) {
    let sum = weights.accuracy_weight + 
              weights.efficiency_weight + 
              weights.interpretability_weight + 
              weights.memory_usage_weight + 
              weights.training_time_weight;
    
    weights.accuracy_weight /= sum;
    weights.efficiency_weight /= sum;
    weights.interpretability_weight /= sum;
    weights.memory_usage_weight /= sum;
    weights.training_time_weight /= sum;
}

/// 计算数据复杂度分数 (0-1)
fn calculate_data_complexity(data: &DataCharacteristics) -> f64 {
    let mut complexity_score = 0.0;
    let mut factors = 0;
    
    // 1. 考虑文本复杂度
    if !data.text_fields.is_empty() {
        let avg_length: f64 = data.text_fields.values()
            .map(|stats| stats.avg_length)
            .sum::<f64>() / data.text_fields.len() as f64;
            
        let text_complexity = (avg_length / 1000.0).min(1.0);
        complexity_score += text_complexity;
        factors += 1;
        
        // 检查多语言情况
        let languages: HashSet<_> = data.text_fields.values()
            .filter_map(|stats| stats.detected_language.clone())
            .collect();
        if languages.len() > 1 {
            complexity_score += 0.2;
        }
    }
    
    // 2. 考虑数值复杂度 - 基于方差和分布特性
    if !data.numeric_fields.is_empty() {
        let avg_variance: f64 = data.numeric_fields.values()
            .map(|stats| stats.variance)
            .sum::<f64>() / data.numeric_fields.len() as f64;
            
        // 归一化方差得分
        let variance_score = (avg_variance.log10() + 5.0) / 10.0;
        complexity_score += variance_score.clamp(0.0, 1.0);
        factors += 1;
        
        // 检查偏度和峰度
        let avg_skewness: f64 = data.numeric_fields.values()
            .map(|stats| stats.skewness.abs())
            .sum::<f64>() / data.numeric_fields.len() as f64;
        let skewness_score = (avg_skewness / 3.0).min(1.0);
        complexity_score += skewness_score;
        factors += 1;
    }
    
    // 3. 分类字段复杂度
    if !data.categorical_fields.is_empty() {
        let avg_categories: f64 = data.categorical_fields.values()
            .map(|stats| stats.category_count as f64)
            .sum::<f64>() / data.categorical_fields.len() as f64;
            
        let category_score = (avg_categories / 50.0).min(1.0);
        complexity_score += category_score;
        factors += 1;
    }
    
    // 4. 字段数量的影响
    let field_count_score = (data.field_count as f64 / 30.0).min(1.0);
    complexity_score += field_count_score;
    factors += 1;
    
    // 计算平均复杂度分数
    if factors > 0 {
        complexity_score /= factors as f64;
    }
    
    complexity_score.clamp(0.0, 1.0)
}

/// 估计数据噪声水平 (0-1)
fn estimate_noise_level(data: &DataCharacteristics) -> f64 {
    let mut noise_score = 0.0;
    let mut factors = 0;
    
    // 1. 数值字段噪声 - 基于异常值比例
    if !data.numeric_fields.is_empty() {
        let mut outlier_ratio_sum = 0.0;
        
        for stats in data.numeric_fields.values() {
            if stats.count < 10 { continue; }
            
            // 使用四分位范围检测异常值
            let iqr = stats.quartiles[2] - stats.quartiles[0];
            let lower_bound = stats.quartiles[0] - 1.5 * iqr;
            let upper_bound = stats.quartiles[2] + 1.5 * iqr;
            
            // 估计异常值比例 (这里只能粗略估计)
            let outlier_ratio = 0.1; // 实际应用中应该计算真实比例
            outlier_ratio_sum += outlier_ratio;
        }
        
        let avg_outlier_ratio = outlier_ratio_sum / data.numeric_fields.len() as f64;
        noise_score += avg_outlier_ratio * 5.0; // 放大噪声影响
        factors += 1;
    }
    
    // 2. 文本字段噪声 - 基于拼写错误和非标准语言使用的估计
    if !data.text_fields.is_empty() {
        // 拼写错误和非标准用法估计 - 这需要更高级的语言处理
        // 这里简化处理
        let text_noise_estimate = 0.3; // 固定估计值，实际应基于语言分析
        noise_score += text_noise_estimate;
        factors += 1;
    }
    
    // 3. 分类字段噪声 - 基于罕见类别的比例
    if !data.categorical_fields.is_empty() {
        let mut rare_category_ratio_sum = 0.0;
        
        for stats in data.categorical_fields.values() {
            // 计算罕见类别的比例
            let rare_categories = stats.frequencies.values()
                .filter(|&freq| *freq < 0.05) // 频率低于5%视为罕见
                .count();
                
            let rare_ratio = rare_categories as f64 / stats.category_count.max(1) as f64;
            rare_category_ratio_sum += rare_ratio;
        }
        
        let avg_rare_ratio = rare_category_ratio_sum / data.categorical_fields.len() as f64;
        noise_score += avg_rare_ratio;
        factors += 1;
    }
    
    // 4. 缺失值噪声
    let missing_ratio = data.get_missing_value_ratio();
    noise_score += missing_ratio * 2.0; // 缺失值是噪声的重要指标
    factors += 1;
    
    // 计算平均噪声分数
    if factors > 0 {
        noise_score /= factors as f64;
    }
    
    noise_score.clamp(0.0, 1.0)
}

/// 计算特征多样性分数 (0-1)
fn calculate_feature_diversity(data: &DataCharacteristics) -> f64 {
    // 字段类型多样性 - 基于熵的计算
    let type_counts = [
        data.text_fields.len(), 
        data.numeric_fields.len(), 
        data.categorical_fields.len()
    ];
    
    let total_fields = type_counts.iter().sum::<usize>() as f64;
    if total_fields == 0.0 {
        return 0.0;
    }
    
    // 计算类型分布的熵
    let mut entropy = 0.0;
    for &count in &type_counts {
        if count > 0 {
            let p = count as f64 / total_fields;
            entropy -= p * p.log2();
        }
    }
    
    // 归一化熵值 (最大熵为log2(3)，当三种类型均匀分布时)
    let max_entropy = 3f64.log2();
    let normalized_diversity = entropy / max_entropy;
    
    normalized_diversity
}

/// 比较方法性能
/// 
/// 比较多种特征提取方法的性能，返回最佳方法
/// 
/// # 参数
/// * `methods` - 待比较的方法列表
/// * `data` - 测试数据
/// * `weights` - 评估权重
/// 
/// # 返回
/// * `Result<(TextFeatureMethod, f64)>` - 最佳方法及其得分
pub fn compare_methods(
    methods: &[TextFeatureMethod],
    data: &[Value],
    weights: &EvaluationWeights,
) -> Result<(TextFeatureMethod, f64)> {
    if methods.is_empty() || data.is_empty() {
        return Err(Error::InvalidData("空方法列表或数据集".to_string()));
    }
    
    let mut best_method = methods[0];
    let mut best_score = f64::MIN;
    
    for &method in methods {
        let config = TextFeatureConfig {
            method: method.into(),
            ..Default::default()
        };
        
        let score = evaluate_method_performance_with_weights(&config, data, weights)?;
        
        if score > best_score {
            best_score = score;
            best_method = method;
        }
    }
    
    Ok((best_method, best_score))
}

/// 交叉验证评估
/// 
/// 使用K折交叉验证评估方法性能
/// 
/// # 参数
/// * `config` - 特征提取配置
/// * `data` - 测试数据
/// * `k` - 折数
/// * `weights` - 评估权重
/// 
/// # 返回
/// * `Result<f64>` - 平均评分
pub fn cross_validate(
    config: &TextFeatureConfig,
    data: &[Value],
    k: usize,
    weights: &EvaluationWeights,
) -> Result<f64> {
    if data.len() < k {
        return Err(Error::InvalidData(format!("数据集大小({})小于折数({})", data.len(), k)));
    }
    
    let fold_size = data.len() / k;
    let mut scores = Vec::with_capacity(k);
    
    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { data.len() } else { (i + 1) * fold_size };
        
        let test_data = &data[start..end];
        let mut train_data = Vec::with_capacity(data.len() - test_data.len());
        
        train_data.extend_from_slice(&data[0..start]);
        train_data.extend_from_slice(&data[end..]);
        
        let score = evaluate_method_performance_with_weights(config, test_data, weights)?;
        scores.push(score);
    }
    
    // 计算平均分
    let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
    Ok(avg_score)
}

// 内部辅助函数

/// 评估准确性
fn evaluate_accuracy(extractor: &(impl TextFeatureExtractor + ?Sized), data: &[Value]) -> Result<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    
    // 这里使用向量一致性作为准确性的代理指标
    // 实际应用中，应根据具体任务进行更精确的评估
    
    let mut consistency_score = 0.0;
    let sample_count = std::cmp::min(data.len(), 100); // 限制样本数以提高效率
    
    for i in 0..sample_count {
        let item = &data[i];
        // 提取特征
        let text: String = if let Some(s) = item.as_str() {
            s.to_string()
        } else {
            item.to_string()
        };
        
        let features1 = extractor.extract(&text)?;
        let features2 = extractor.extract(&text)?; // 重复提取以检查一致性
        
        // 计算特征向量的余弦相似度
        let similarity = cosine_similarity(&features1, &features2);
        consistency_score += similarity;
    }
    
    let avg_consistency = consistency_score / sample_count as f64;
    Ok(avg_consistency)
}

/// 评估效率
fn evaluate_efficiency(extractor: &(impl TextFeatureExtractor + ?Sized), data: &[Value]) -> Result<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    
    let sample_count = std::cmp::min(data.len(), 20); // 限制样本数以加快评估
    let mut times = Vec::with_capacity(sample_count);
    
    for i in 0..sample_count {
        let item = &data[i];
        let text: String = if let Some(s) = item.as_str() {
            s.to_string()
        } else {
            item.to_string()
        };
        
        // 测量特征提取时间
        let start = Instant::now();
        let _ = extractor.extract(&text)?;
        let duration = start.elapsed();
        
        times.push(duration);
    }
    
    // 计算平均处理时间
    let avg_time = times.iter().fold(Duration::new(0, 0), |acc, &time| acc + time) / times.len() as u32;
    let avg_time_ms = avg_time.as_millis() as f64;
    
    // 转换为得分 (0-1，处理时间越短得分越高)
    // 假设20ms是最佳时间, 2000ms是最差时间
    let max_time = 2000.0;
    let min_time = 20.0;
    let score = 1.0 - ((avg_time_ms - min_time) / (max_time - min_time)).min(1.0).max(0.0);
    
    Ok(score)
}

/// 评估可解释性
fn evaluate_interpretability(extractor: &(impl TextFeatureExtractor + ?Sized), method: TextFeatureMethod) -> f64 {
    // 不同方法的可解释性得分（基于先验知识）
    match method {
        TextFeatureMethod::BagOfWords => 0.9, // 简单直观
        TextFeatureMethod::TfIdf => 0.8, // 相对直观
        TextFeatureMethod::NGram => 0.7, // 较为直观
        TextFeatureMethod::FastText => 0.5, // 中等可解释性
        TextFeatureMethod::Word2Vec => 0.4, // 低可解释性
        TextFeatureMethod::Bert => 0.2, // 非常低的可解释性
        _ => 0.5, // 默认中等可解释性
    }
}

/// 计算余弦相似度
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f64 {
    if v1.is_empty() || v2.is_empty() || v1.len() != v2.len() {
        return 0.0;
    }
    
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    for i in 0..v1.len() {
        dot_product += (v1[i] * v2[i]) as f64;
        norm1 += (v1[i] * v1[i]) as f64;
        norm2 += (v2[i] * v2[i]) as f64;
    }
    
    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    
    dot_product / (norm1.sqrt() * norm2.sqrt())
}

/// 交叉验证结果
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// 平均评分
    pub avg_score: f64,
    /// 各折评分
    pub fold_scores: Vec<f64>,
    /// 准确性评分
    pub accuracy_score: f64,
    /// 效率评分
    pub efficiency_score: f64,
    /// 可解释性评分
    pub interpretability_score: f64,
    /// 评估时间 (毫秒)
    pub evaluation_time_ms: u64,
}

/// 完整评估
/// 
/// 对特征提取方法进行全面评估，包括准确性、效率、可解释性和交叉验证结果
/// 
/// # 参数
/// * `config` - 特征提取配置
/// * `data` - 测试数据
/// * `weights` - 评估权重
/// * `k` - 交叉验证折数
/// 
/// # 返回
/// * `Result<HashMap<String, f64>>` - 评估结果
pub fn comprehensive_evaluation(
    config: &TextFeatureConfig,
    data: &[Value],
    weights: &EvaluationWeights,
    k: Option<usize>,
) -> Result<HashMap<String, f64>> {
    if data.is_empty() {
        return Err(Error::InvalidData("空数据集".to_string()));
    }
    
    let start = Instant::now();
    let extractor = create_extractor(config)?;
    
    let mut results = HashMap::new();
    
    // 基础评估
    let accuracy_score = evaluate_accuracy(&*extractor, data)?;
    let efficiency_score = evaluate_efficiency(&*extractor, data)?;
    let interpretability_score = evaluate_interpretability(&*extractor, config.method);
    
    results.insert("accuracy".to_string(), accuracy_score);
    results.insert("efficiency".to_string(), efficiency_score);
    results.insert("interpretability".to_string(), interpretability_score);
    
    // 加权评分
    let weighted_score = 
        accuracy_score * weights.accuracy_weight +
        efficiency_score * weights.efficiency_weight +
        interpretability_score * weights.interpretability_weight;
    
    results.insert("weighted_score".to_string(), weighted_score);
    
    // 交叉验证 (如果需要)
    if let Some(folds) = k {
        if folds >= 2 {
            let cv_score = cross_validate(config, data, folds, weights)?;
            results.insert("cross_validation".to_string(), cv_score);
        }
    }
    
    // 评估时间
    let evaluation_time = start.elapsed().as_millis() as f64;
    results.insert("evaluation_time_ms".to_string(), evaluation_time);
    
    // 额外信息
    results.insert("feature_dimension".to_string(), extractor.dimension() as f64);
    
    Ok(results)
}

/// 特征提取正确性测试
/// 
/// 检查特征提取结果是否符合预期
/// 
/// # 参数
/// * `config` - 特征提取配置
/// * `test_cases` - 测试用例（输入文本和预期结果）
/// 
/// # 返回
/// * `Result<bool>` - 是否全部通过测试
pub fn test_extraction_correctness(
    config: &TextFeatureConfig,
    test_cases: &[(String, Vec<f32>)],
) -> Result<bool> {
    if test_cases.is_empty() {
        return Ok(true);
    }
    
    let extractor = create_extractor(config)?;
    
    for (text, expected) in test_cases {
        let features = extractor.extract(text)?;
        
        // 检查维度是否一致
        if features.len() != expected.len() {
            return Ok(false);
        }
        
        // 检查特征值是否接近
        for i in 0..features.len() {
            // 允许一定的误差
            if (features[i] - expected[i]).abs() > 0.001 {
                return Ok(false);
            }
        }
    }
    
    Ok(true)
} 