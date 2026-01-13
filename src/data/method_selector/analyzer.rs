// Data Analyzer for Method Selector
// 方法选择器的数据分析模块

use crate::data::method_selector::types::{DataCharacteristics, OtherTextFieldStats, OtherCategoricalFieldStats};
use crate::data::text_features::{TextFieldStats, CategoricalFieldStats, NumericStats};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// 数据分析器
pub struct DataAnalyzer;

impl DataAnalyzer {
    /// 创建新的数据分析器
    pub fn new() -> Self {
        DataAnalyzer
    }
    
    /// 分析数据特征
    pub fn analyze_data(&self, data: &[Value]) -> DataCharacteristics {
        if data.is_empty() {
            return DataCharacteristics::default();
        }
        
        let sample_data = if data.len() > 100 {
            &data[0..100]
        } else {
            data
        };
        
        let mut text_fields: HashMap<String, TextFieldStats> = HashMap::new();
        let mut numeric_fields: HashMap<String, NumericStats> = HashMap::new();
        let mut categorical_fields: HashMap<String, CategoricalFieldStats> = HashMap::new();
        
        let mut data_type = "mixed".to_string();
        
        // 分析第一个样本以确定字段类型
        if let Some(first_item) = sample_data.first() {
            if let Some(obj) = first_item.as_object() {
                for (key, value) in obj {
                    if let Some(text_value) = value.as_str() {
                        // 初始化文本字段统计
                        text_fields.insert(key.clone(), TextFieldStats {
                            avg_length: 0.0,
                            max_length: 0,
                            min_length: 0,
                            unique_tokens: 0,
                            language_confidence: 0.0,
                        });
                    } else if value.is_number() {
                        // 初始化数值字段统计
                        numeric_fields.insert(key.clone(), NumericStats {
                            min: f64::MAX,
                            max: f64::MIN,
                            mean: 0.0,
                            median: 0.0,
                            std_dev: 0.0,
                        });
                    } else if value.is_boolean() || value.is_array() || (value.is_object() && !value.is_null()) {
                        // 初始化分类字段统计
                        categorical_fields.insert(key.clone(), CategoricalFieldStats {
                            unique_values: 0,
                            most_common: Vec::new(),
                            entropy: 0.0,
                        });
                    }
                }
            }
        }
        
        // 分析所有样本
        for item in sample_data {
            if let Some(obj) = item.as_object() {
                for (key, value) in obj {
                    if let Some(text_stats) = text_fields.get_mut(key) {
                        if let Some(text_value) = value.as_str() {
                            // 更新文本统计
                            let length = text_value.len();
                            // 累积统计值用于后续计算平均值
                            text_stats.avg_length += (length as f32) / (sample_data.len() as f32);
                            
                            // 更新最大和最小长度
                            if length > text_stats.max_length {
                                text_stats.max_length = length;
                            }
                            if length < text_stats.min_length || text_stats.min_length == 0 {
                                text_stats.min_length = length;
                            }
                            
                            // 更新唯一token统计（简化实现）
                            let words: Vec<&str> = text_value.split_whitespace().collect();
                            text_stats.unique_tokens = text_stats.unique_tokens.max(words.len());
                        }
                    } else if let Some(numeric_stats) = numeric_fields.get_mut(key) {
                        if let Some(num_value) = value.as_f64() {
                            // 更新数值统计
                            if num_value < numeric_stats.min {
                                numeric_stats.min = num_value;
                            }
                            if num_value > numeric_stats.max {
                                numeric_stats.max = num_value;
                            }
                            // 简化实现：使用增量更新均值
                            let old_mean = numeric_stats.mean;
                            let count = 1.0;
                            numeric_stats.mean += (num_value - old_mean) / count;
                        }
                    } else if let Some(categorical_stats) = categorical_fields.get_mut(key) {
                        // 更新分类统计（简化版）
                        if !value.is_null() {
                            categorical_stats.unique_values += 1;
                        }
                    }
                }
            }
        }
        
        // 计算文本字段的最终统计值
        let mut total_vocabulary: HashSet<String> = HashSet::new();
        for (key, stats) in text_fields.iter_mut() {
            // 重新计算平均长度（如果需要）
            let mut total_length = 0.0;
            let mut sample_count = 0;
            
            for item in sample_data {
                if let Some(obj) = item.as_object() {
                    if let Some(value) = obj.get(key) {
                        if let Some(text) = value.as_str() {
                            total_length += text.len() as f32;
                            sample_count += 1;
                            
                            // 收集词汇
                            let words: Vec<&str> = text.split_whitespace().collect();
                            for word in words {
                                total_vocabulary.insert(word.to_lowercase());
                            }
                        }
                    }
                }
            }
            
            // 更新平均长度
            if sample_count > 0 {
                stats.avg_length = total_length / (sample_count as f32);
            }
        }
        
        // 计算数值字段标准差（简化实现）
        for stats in numeric_fields.values_mut() {
            // 标准差计算需要更多数据，这里简化处理
            // 实际实现中需要收集所有值来计算标准差
        }
        
        // 确定数据类型
        if !text_fields.is_empty() && numeric_fields.is_empty() && categorical_fields.is_empty() {
            data_type = "text".to_string();
        } else if text_fields.is_empty() && !numeric_fields.is_empty() && categorical_fields.is_empty() {
            data_type = "numeric".to_string();
        } else if text_fields.is_empty() && numeric_fields.is_empty() && !categorical_fields.is_empty() {
            data_type = "categorical".to_string();
        }
        
        // 计算平均文本长度
        let avg_text_length = if !text_fields.is_empty() {
            text_fields.values().map(|s| s.avg_length as f64).sum::<f64>() / text_fields.len() as f64
        } else {
            0.0
        };
        
        // 计算词汇量大小
        let vocabulary_size = total_vocabulary.len();
        
        // 在移动字段前先计算field_count和sample_count
        let field_count = text_fields.len() + numeric_fields.len() + categorical_fields.len();
        let sample_count = sample_data.len();
        
        // 序列模式关键词
        let sequential_keywords = vec![
            "next", "previous", "after", "before", "first", "last", "sequence",
            "step", "stage", "phase", "order", "series", "following", "preceding"
        ];

        // 时间相关关键词
        let time_keywords = vec![
            "time", "date", "year", "month", "day", "hour", "minute", "second",
            "morning", "afternoon", "evening", "night", "today", "tomorrow", "yesterday",
            "schedule", "period", "duration", "interval", "frequency"
        ];

        // 复杂语义关键词
        let semantic_keywords = vec![
            "meaning", "context", "relationship", "similar", "different", "compare",
            "analogy", "metaphor", "concept", "idea", "thought", "understand",
            "interpret", "analyze", "nuance", "subtlety", "ambiguity"
        ];

        // 初始化计数器
        let mut sequential_pattern_count = 0;
        let mut time_related_count = 0;
        let mut complex_semantics_count = 0;
        let mut total_text_samples = 0;

        // 记录词频，用于计算文本多样性
        let mut all_words: HashSet<String> = HashSet::new();
        let mut word_frequencies: HashMap<String, usize> = HashMap::new();

        // 遍历所有文本样本检测模式
        for (field_name, field_stats) in &text_fields {
            // 简化实现：从样本数据中收集词汇
            for item in sample_data {
                if let Some(obj) = item.as_object() {
                    if let Some(value) = obj.get(field_name) {
                        if let Some(text) = value.as_str() {
                            let words: Vec<&str> = text.split_whitespace().collect();
                            for word in words {
                                let word_lower = word.to_lowercase();
                                *word_frequencies.entry(word_lower.clone()).or_insert(0) += 1;
                                all_words.insert(word_lower);
                            }
                        }
                    }
                }
            }
            
            // 对每个样本进行内容分析
            for item in sample_data {
                if let Some(obj) = item.as_object() {
                    if let Some(value) = obj.get(field_name) {
                        if let Some(text) = value.as_str() {
                            total_text_samples += 1;
                            
                            // 检查序列模式
                            for keyword in &sequential_keywords {
                                if text.to_lowercase().contains(keyword) {
                                    sequential_pattern_count += 1;
                                    break;
                                }
                            }
                            
                            // 检查时间相关文本
                            for keyword in &time_keywords {
                                if text.to_lowercase().contains(keyword) {
                                    time_related_count += 1;
                                    break;
                                }
                            }
                            
                            // 检查复杂语义
                            for keyword in &semantic_keywords {
                                if text.to_lowercase().contains(keyword) {
                                    complex_semantics_count += 1;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 计算各种比例
        let sequential_ratio = if total_text_samples > 0 {
            sequential_pattern_count as f64 / total_text_samples as f64
        } else {
            0.0
        };

        let time_related_ratio = if total_text_samples > 0 {
            time_related_count as f64 / total_text_samples as f64
        } else {
            0.0
        };

        let complex_semantics_ratio = if total_text_samples > 0 {
            complex_semantics_count as f64 / total_text_samples as f64
        } else {
            0.0
        };

        // 计算文本多样性得分 (基于词频分布)
        let total_word_count: usize = word_frequencies.values().sum();
        let text_diversity_score = if total_word_count > 0 && !word_frequencies.is_empty() {
            // 计算标准化熵作为多样性得分
            let entropy: f64 = word_frequencies.values()
                .map(|&count| {
                    let prob = count as f64 / total_word_count as f64;
                    if prob > 0.0 {
                        -prob * prob.log2()
                    } else {
                        0.0
                    }
                })
                .sum();
            
            // 标准化为0-1范围
            let max_entropy = (word_frequencies.len() as f64).log2();
            if max_entropy > 0.0 {
                entropy / max_entropy
            } else {
                0.0
            }
        } else {
            0.0
        };

        // 计算字段比例
        let text_field_ratio = if field_count > 0 {
            text_fields.len() as f64 / field_count as f64
        } else {
            0.0
        };

        let numeric_field_ratio = if field_count > 0 {
            numeric_fields.len() as f64 / field_count as f64
        } else {
            0.0
        };

        let categorical_field_ratio = if field_count > 0 {
            categorical_fields.len() as f64 / field_count as f64
        } else {
            0.0
        };

        // 判断是否存在混合数据类型
        let has_mixed_data_types = text_fields.len() > 0 && (numeric_fields.len() > 0 || categorical_fields.len() > 0);

        // Calculate to DataCharacteristics
        let data_characteristics = DataCharacteristics {
            data_type,
            avg_text_length,
            vocabulary_size,
            numeric_feature_count: numeric_fields.len(),
            categorical_feature_count: categorical_fields.len(),
            has_structured_data: !numeric_fields.is_empty() || !categorical_fields.is_empty(),
            has_unstructured_data: !text_fields.is_empty(),
            language: None,
            domain: Some("general".to_string()),
            missing_ratio: 0.0,
            quality_score: self.calculate_quality_score(
                &text_fields,
                &numeric_fields,
                &categorical_fields,
                0.0,
                text_diversity_score,
                avg_text_length,
                vocabulary_size,
                sample_count,
                field_count,
                has_mixed_data_types
            ),
            text_fields: text_fields.into_iter().map(|(k, v)| {
                (k, OtherTextFieldStats {
                    avg_length: v.avg_length as f64,
                    avg_word_count: 0.0, // 简化实现
                    special_char_ratio: 0.0, // 简化实现
                    missing_ratio: 0.0, // 简化实现
                    vocabulary: None, // 简化实现
                    sample_count: 0, // 简化实现
                    sparsity_score: 0.0, // 简化实现
                })
            }).collect(),
            numeric_fields,
            categorical_fields: categorical_fields.into_iter().map(|(k, v)| {
                (k, OtherCategoricalFieldStats {
                    cardinality: v.unique_values,
                    top_frequency: if !v.most_common.is_empty() {
                        v.most_common[0].1 as f64 / sample_data.len() as f64
                    } else {
                        0.0
                    },
                    missing_ratio: 0.0, // 简化实现
                })
            }).collect(),
            field_count,
            sample_count,
            contains_sequential_patterns: sequential_ratio > 0.2,
            contains_time_related_text: time_related_ratio > 0.2,
            contains_complex_semantics: complex_semantics_ratio > 0.15 || text_diversity_score > 0.7,
            contains_ambiguous_meanings: complex_semantics_ratio > 0.1 && avg_text_length > 50.0,
            text_field_ratio,
            numeric_field_ratio,
            categorical_field_ratio,
            has_mixed_data_types,
            text_diversity_score,
        };
        
        data_characteristics
    }
    
    /// 提取样本数据
    pub fn extract_sample(&self, data: &[Value], sample_size: usize) -> Vec<Value> {
        if data.len() <= sample_size {
            return data.to_vec();
        }
        
        let step = data.len() / sample_size;
        let mut sample = Vec::with_capacity(sample_size);
        
        for i in 0..sample_size {
            let index = i * step;
            if index < data.len() {
                sample.push(data[index].clone());
            } else {
                break;
            }
        }
        
        sample
    }
    
    /// 确定数据类型
    fn determine_data_type(
        text_fields: &HashMap<String, TextFieldStats>,
        numeric_fields: &HashMap<String, NumericStats>,
        categorical_fields: &HashMap<String, CategoricalFieldStats>,
    ) -> String {
        if !text_fields.is_empty() && numeric_fields.is_empty() && categorical_fields.is_empty() {
            "text".to_string()
        } else if text_fields.is_empty() && !numeric_fields.is_empty() && categorical_fields.is_empty() {
            "numeric".to_string()
        } else if text_fields.is_empty() && numeric_fields.is_empty() && !categorical_fields.is_empty() {
            "categorical".to_string()
        } else {
            "mixed".to_string()
        }
    }
    
    /// 计算平均文本长度
    fn calculate_avg_text_length(text_fields: &HashMap<String, TextFieldStats>) -> f64 {
        if text_fields.is_empty() {
            return 0.0;
        }
        
        let mut total_length = 0.0f64;
        let mut field_count = 0;
        
        for stats in text_fields.values() {
            total_length += stats.avg_length as f64;
            field_count += 1;
        }
        
        if field_count > 0 {
            total_length / field_count as f64
        } else {
            0.0
        }
    }
    
    /// 计算数据稀疏度
    fn calculate_sparsity(
        text_fields: &HashMap<String, TextFieldStats>,
        numeric_fields: &HashMap<String, NumericStats>,
    ) -> f64 {
        let mut total_sparsity = 0.0;
        let mut field_count = 0;
        
        // 文本字段稀疏度（简化实现）
        for _stats in text_fields.values() {
            // 简化实现：不计算稀疏度
            field_count += 1;
        }
        
        // 数值字段稀疏度（简化实现）
        for _stats in numeric_fields.values() {
            // 简化实现：不计算稀疏度
            field_count += 1;
        }
        
        if field_count > 0 {
            total_sparsity / field_count as f64
        } else {
            0.0
        }
    }

    fn calculate_quality_score(
        &self,
        text_fields: &HashMap<String, TextFieldStats>,
        numeric_fields: &HashMap<String, NumericStats>,
        categorical_fields: &HashMap<String, CategoricalFieldStats>,
        missing_value_ratio: f64,
        text_diversity_score: f64,
        avg_text_length: f64,
        vocabulary_size: usize,
        sample_count: usize,
        field_count: usize,
        has_mixed_data_types: bool
    ) -> f64 {
        if sample_count == 0 || field_count == 0 {
            return 0.0;
        }
        
        // 1. 数据完整性评分 (0-1)
        // 缺失值越少，评分越高
        let completeness_score = 1.0 - missing_value_ratio.min(1.0);
        
        // 2. 样本量评分 (0-1)
        // 样本数越多，评分越高，但有上限
        let sample_size_score = match sample_count {
            0 => 0.0,
            1..=9 => 0.2,
            10..=49 => 0.4,
            50..=99 => 0.6,
            100..=499 => 0.8,
            500..=999 => 0.9,
            _ => 1.0,
        };
        
        // 3. 文本质量评分 (0-1)
        let text_quality_score = if !text_fields.is_empty() {
            // 文本多样性和平均长度对文本质量的影响
            let diversity_factor = text_diversity_score;
            
            // 平均文本长度评分
            let length_factor = match avg_text_length as usize {
                0 => 0.0,
                1..=5 => 0.3,   // 太短
                6..=20 => 0.7,  // 适中
                21..=100 => 1.0, // 理想
                101..=500 => 0.8, // 较长
                _ => 0.6,       // 非常长
            };
            
            // 词汇量评分
            let vocabulary_factor = match vocabulary_size {
                0 => 0.0,
                1..=100 => 0.4,
                101..=1000 => 0.7,
                1001..=10000 => 1.0,
                10001..=50000 => 0.9,
                _ => 0.8,       // 非常大的词汇量可能包含噪声
            };
            
            // 词汇多样性评分（基于独特词汇占比）
            let vocab_diversity = if vocabulary_size > 0 {
                // 简化实现：使用 unique_tokens 估算
                let total_tokens: usize = text_fields.values()
                    .map(|stats| stats.unique_tokens)
                    .sum();
                    
                if total_tokens > 0 {
                    vocabulary_size as f64 / total_tokens as f64
                } else {
                    0.5
                }
            } else {
                0.0
            };
            
            let vocab_diversity_score = match vocab_diversity {
                d if d <= 0.05 => 0.3,  // 非常低的多样性
                d if d <= 0.1 => 0.5,   // 低多样性
                d if d <= 0.2 => 0.7,   // 中等多样性
                d if d <= 0.5 => 1.0,   // 理想多样性
                d if d <= 0.8 => 0.8,   // 高多样性（可能过于松散）
                _ => 0.6                // 极高多样性（可能是噪声或无关文本）
            };
            
            // 语法质量评分（基于文本特征，简化实现）
            let grammar_quality = text_fields.values()
                .filter_map(|stats| {
                    // 简化实现：基于平均长度估算
                    if stats.avg_length > 0.0 {
                        // 假设理想平均长度在 50-100 之间
                        let ideal_length = 75.0;
                        Some(((stats.avg_length - ideal_length).abs() / ideal_length).min(1.0))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
                
            let grammar_score = if !grammar_quality.is_empty() {
                1.0 - grammar_quality.iter().map(|&x| x as f64).sum::<f64>() / grammar_quality.len() as f64
            } else {
                0.5  // 默认中等分数
            };
            
            // 计算最终文本质量分数，加入词汇多样性和语法质量
            (diversity_factor * 0.3 + 
             length_factor * 0.2 + 
             vocabulary_factor * 0.2 +
             vocab_diversity_score * 0.15 +
             grammar_score * 0.15).min(1.0)
        } else {
            0.5 // 无文本字段时默认为中等分数
        };
        
        // 4. 数值字段质量评分 (0-1)
        let numeric_quality_score = if !numeric_fields.is_empty() {
            let mut std_dev_sum = 0.0;
            let mut outlier_ratio_sum = 0.0;
            let mut distribution_score_sum = 0.0;
            let mut range_score_sum = 0.0;
            
            for stats in numeric_fields.values() {
                // 标准差评分：适当的标准差表示数据有变异性
                if stats.std_dev.is_normal() {
                    // 归一化标准差 (期望一定水平的变异性)
                    let normalized_std = if stats.std_dev > 0.0 {
                        let ratio = stats.std_dev / (stats.max - stats.min).max(0.001);
                        if ratio < 0.01 { 0.3 }      // 太小的变异性
                        else if ratio > 0.5 { 0.7 }  // 太大的变异性
                        else { 1.0 }                 // 适当的变异性
                    } else {
                        0.1 // 无变异性
                    };
                    std_dev_sum += normalized_std;
                }
                
                // 异常值比例：异常值太多会降低质量（简化实现）
                // 基于 min/max/mean/std_dev 估算异常值
                let range = stats.max - stats.min;
                if range > 0.0 && stats.std_dev > 0.0 {
                    // 使用3-sigma规则估算异常值比例
                    let lower_bound = stats.mean - 3.0 * stats.std_dev;
                    let upper_bound = stats.mean + 3.0 * stats.std_dev;
                    
                    // 简化实现：假设超出3-sigma范围的值比例为异常值
                    let outlier_ratio: f64 = if stats.min < lower_bound || stats.max > upper_bound {
                        0.1 // 简化估算
                    } else {
                        0.0
                    };
                    outlier_ratio_sum += 1.0 - outlier_ratio.min(1.0);
                }
                
                // 分布评分：评估数据分布是否近似正态分布（简化实现）
                // 使用 mean 和 median 的差异估算偏度
                let sample_size_estimate = 30; // 简化假设
                if sample_size_estimate >= 30 {
                    let skewness_estimate = if stats.std_dev > 0.0 {
                        (stats.mean - stats.median) / stats.std_dev
                    } else {
                        0.0
                    };
                    let kurtosis_estimate = 0.0; // 简化实现
                    
                    // 偏度接近0且峰度接近3表示近似正态分布
                    let skewness_score = 1.0 - (skewness_estimate.abs() / 2.0).min(1.0);
                    let kurtosis_score = 0.5; // 简化实现
                    
                    distribution_score_sum += skewness_score * 0.5 + kurtosis_score * 0.5;
                } else {
                    // 样本数不足，给予中等分数
                    distribution_score_sum += 0.5;
                }
                
                // 数值范围评分：评估值域是否适当
                let range = stats.max - stats.min;
                if range > 0.0 {
                    // 针对不同数量级的数据，评估范围是否适当
                    let magnitude = (stats.max.abs().max(stats.min.abs()) + 1.0).log10();
                    let range_ratio = range / 10.0f64.powf(magnitude);
                    
                    let range_score = match range_ratio {
                        r if r < 0.001 => 0.3,  // 范围太小
                        r if r < 0.01 => 0.5,   // 范围较小
                        r if r < 0.1 => 0.8,    // 范围适中
                        r if r < 10.0 => 1.0,   // 理想范围
                        r if r < 100.0 => 0.7,  // 范围较大
                        _ => 0.5                // 范围过大
                    };
                    range_score_sum += range_score;
                } else {
                    // 没有范围（单一值）
                    range_score_sum += 0.2;
                }
            }
            
            // 计算平均分数
            let num_fields = numeric_fields.len() as f64;
            if num_fields > 0.0 {
                let avg_std_score = std_dev_sum / num_fields;
                let avg_outlier_score = outlier_ratio_sum / num_fields;
                let avg_distribution_score = distribution_score_sum / num_fields;
                let avg_range_score = range_score_sum / num_fields;
                
                (avg_std_score * 0.3 + 
                 avg_outlier_score * 0.3 + 
                 avg_distribution_score * 0.2 + 
                 avg_range_score * 0.2).min(1.0)
            } else {
                0.5
            }
        } else {
            0.5 // 无数值字段时默认为中等分数
        };
        
        // 5. 分类字段质量评分 (0-1)
        let categorical_quality_score = if !categorical_fields.is_empty() {
            let mut balance_score_sum = 0.0;
            let mut cardinality_score_sum = 0.0;
            let mut consistency_score_sum = 0.0;
            
            for stats in categorical_fields.values() {
                // 类别平衡性：各类别分布均匀为佳（简化实现）
                // 使用 most_common 和 unique_values
                if stats.unique_values > 0 && !stats.most_common.is_empty() {
                    let category_count = stats.unique_values;
                    let total_count: usize = stats.most_common.iter().map(|(_, count)| count).sum();
                    let ideal_freq = total_count as f64 / category_count as f64;
                    
                    let mut deviation_sum = 0.0;
                    for (_, freq) in &stats.most_common {
                        let deviation = (*freq as f64 - ideal_freq).abs() / ideal_freq.max(1.0);
                        deviation_sum += deviation;
                    }
                    
                    // 类别均衡度评分 (1 - 归一化偏差)
                    let balance_score = 1.0 - (deviation_sum / category_count as f64).min(1.0);
                    balance_score_sum += balance_score;
                    
                    // 类别数量评分：太多或太少类别都不理想
                    let cardinality_score = match category_count {
                        0 => 0.0,
                        1 => 0.1,           // 单一类别无信息量
                        2..=10 => 1.0,      // 理想的类别数量
                        11..=50 => 0.8,     // 较多类别
                        51..=100 => 0.6,    // 类别数量大
                        101..=500 => 0.4,   // 类别数量过大
                        _ => 0.2,           // 类别数量极大
                    };
                    cardinality_score_sum += cardinality_score;
                    
                    // 类别命名一致性评分
                    // 评估类别名称格式是否一致，例如大小写、前缀等
                    let categories: Vec<&String> = stats.most_common.iter().map(|(name, _)| name).collect();
                    let mut format_consistency = 1.0;
                    
                    if categories.len() >= 2 {
                        // 检查首字母大小写一致性
                        let first_caps = categories.iter()
                            .filter(|c| !c.is_empty() && c.chars().next().unwrap().is_uppercase())
                            .count();
                            
                        if first_caps > 0 && first_caps < categories.len() {
                            // 大小写混用
                            format_consistency *= 0.8;
                        }
                        
                        // 检查类别长度一致性
                        let avg_len = categories.iter()
                            .map(|c| c.len())
                            .sum::<usize>() as f64 / categories.len() as f64;
                            
                        let len_variance = categories.iter()
                            .map(|c| (c.len() as f64 - avg_len).powi(2))
                            .sum::<f64>() / categories.len() as f64;
                            
                        // 长度变异性评分
                        let len_consistency = 1.0 - (len_variance / avg_len).min(1.0);
                        format_consistency *= len_consistency;
                    }
                    
                    consistency_score_sum += format_consistency;
                }
            }
            
            // 计算平均分数
            let cat_fields = categorical_fields.len() as f64;
            if cat_fields > 0.0 {
                let avg_balance_score = balance_score_sum / cat_fields;
                let avg_cardinality_score = cardinality_score_sum / cat_fields;
                let avg_consistency_score = consistency_score_sum / cat_fields;
                
                (avg_balance_score * 0.4 + 
                 avg_cardinality_score * 0.3 + 
                 avg_consistency_score * 0.3).min(1.0)
            } else {
                0.5
            }
        } else {
            0.5 // 无分类字段时默认为中等分数
        };
        
        // 6. 数据类型多样性评分
        // 混合数据类型可能更有价值
        let diversity_score = if has_mixed_data_types {
            // 当多种数据类型存在且均衡分布时，评分高
            let type_count = ((!text_fields.is_empty()) as u8) +
                            ((!numeric_fields.is_empty()) as u8) +
                            ((!categorical_fields.is_empty()) as u8);
                            
            // 计算各类型字段的比例
            let text_ratio = if field_count > 0 {
                text_fields.len() as f64 / field_count as f64
            } else {
                0.0
            };
            
            let numeric_ratio = if field_count > 0 {
                numeric_fields.len() as f64 / field_count as f64
            } else {
                0.0
            };
            
            let categorical_ratio = if field_count > 0 {
                categorical_fields.len() as f64 / field_count as f64
            } else {
                0.0
            };
            
            // 计算分布均衡度（使用熵）
            let entropy = -1.0 * (
                if text_ratio > 0.0 { text_ratio * text_ratio.log2() } else { 0.0 } +
                if numeric_ratio > 0.0 { numeric_ratio * numeric_ratio.log2() } else { 0.0 } +
                if categorical_ratio > 0.0 { categorical_ratio * categorical_ratio.log2() } else { 0.0 }
            );
            
            // 归一化熵（最大熵为log2(3)≈1.585）
            let normalized_entropy = entropy / 1.585;
            
            // 计算最终多样性评分
            match type_count {
                0 => 0.0,
                1 => 0.5,  // 单一类型
                2 => 0.5 + 0.3 * normalized_entropy,  // 两种类型的均衡度
                3 => 0.7 + 0.3 * normalized_entropy,  // 三种类型的均衡度
                _ => 0.5,  // 不可能的情况
            }
        } else {
            0.5 // 单一数据类型
        };
        
        // 7. 数据字段数量评分
        let field_count_score = match field_count {
            0 => 0.0,
            1 => 0.3,           // 单一字段信息有限
            2..=5 => 0.7,       // 少量字段
            6..=15 => 1.0,      // 理想的字段数量
            16..=30 => 0.8,     // 较多字段
            31..=50 => 0.6,     // 字段数量大
            51..=100 => 0.4,    // 字段数量过大
            _ => 0.3,           // 字段数量极大
        };
        
        // 8. 数据一致性评分
        // 基于字段命名风格、缺失值分布等因素评估数据一致性
        let consistency_score = {
            // 评估字段命名一致性
            let field_names = text_fields.keys()
                .chain(numeric_fields.keys())
                .chain(categorical_fields.keys())
                .collect::<Vec<_>>();
                
            // 检查命名风格（驼峰式、下划线式等）
            let mut camel_case = 0;
            let mut snake_case = 0;
            let mut kebab_case = 0;
            
            for &name in &field_names {
                if name.contains('_') {
                    snake_case += 1;
                } else if name.contains('-') {
                    kebab_case += 1;
                } else if name.chars().any(|c| c.is_uppercase()) {
                    camel_case += 1;
                }
            }
            
            let total_fields = field_names.len();
            let naming_consistency = if total_fields > 1 {
                let max_style = camel_case.max(snake_case).max(kebab_case);
                max_style as f64 / total_fields as f64
            } else {
                1.0  // 单个字段认为是一致的
            };
            
            // 缺失值分布一致性（检查是否相同样本在多个字段中都有缺失）
            // 在实际数据中实现，这里使用命名一致性代替
            naming_consistency
        };
        
        // 计算最终质量分数
        // 权重根据重要性分配
        let final_score =
            completeness_score * 0.20 +     // 数据完整性
            sample_size_score * 0.10 +      // 样本量
            text_quality_score * 0.15 +     // 文本质量
            numeric_quality_score * 0.15 +  // 数值字段质量
            categorical_quality_score * 0.15 + // 分类字段质量
            diversity_score * 0.05 +        // 数据类型多样性
            field_count_score * 0.10 +      // 字段数量
            consistency_score * 0.10;       // 数据一致性
        
        // 确保评分在0-1范围内
        final_score.max(0.0).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_analyze_data() {
        let data = vec![
            serde_json::json!({
                "text": "This is a sample text",
                "number": 42,
                "category": "A"
            }),
            serde_json::json!({
                "text": "Another example with different length",
                "number": 73,
                "category": "B"
            }),
        ];
        
        let analyzer = DataAnalyzer::new();
        let characteristics = analyzer.analyze_data(&data);
        
        assert_eq!(characteristics.data_type, "mixed_text");  // 更新期望值匹配新的数据类型格式
        assert!(characteristics.avg_text_length > 0.0);
        assert!(characteristics.vocabulary_size > 0);
        assert_eq!(characteristics.numeric_feature_count, 1);
        assert_eq!(characteristics.categorical_feature_count, 1);
    }
    
    #[test]
    fn test_extract_sample() {
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(serde_json::json!({ "value": i }));
        }
        
        let analyzer = DataAnalyzer::new();
        let sample = analyzer.extract_sample(&data, 10);
        
        assert_eq!(sample.len(), 10);
    }
} 