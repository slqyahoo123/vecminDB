use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// 数值字段统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericStats {
    /// 最小值
    pub min: f64,
    /// 最大值
    pub max: f64,
    /// 平均值
    pub mean: f64,
    /// 中位数
    pub median: f64,
    /// 标准差
    pub std_dev: f64,
    /// 方差
    pub variance: f64,
    /// 样本数
    pub count: usize,
    /// 四分位数
    pub quartiles: [f64; 3],
    /// 偏度
    pub skewness: f64,
    /// 峰度
    pub kurtosis: f64,
    /// 缺失值数量
    pub missing: usize,
}

impl Default for NumericStats {
    fn default() -> Self {
        Self {
            min: f64::MAX,
            max: f64::MIN,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            variance: 0.0,
            count: 0,
            quartiles: [0.0, 0.0, 0.0],
            skewness: 0.0,
            kurtosis: 0.0,
            missing: 0,
        }
    }
}

impl NumericStats {
    /// 创建新的数值型统计信息
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 从数值数组计算统计信息
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        
        let mut stats = Self::default();
        
        // 计算基本统计量
        stats.count = values.len();
        stats.min = values.iter().cloned().fold(f64::MAX, f64::min);
        stats.max = values.iter().cloned().fold(f64::MIN, f64::max);
        
        // 计算平均值
        let sum: f64 = values.iter().sum();
        stats.mean = sum / stats.count as f64;
        
        // 计算方差和标准差
        let sum_squared_diff: f64 = values.iter()
            .map(|&x| (x - stats.mean).powi(2))
            .sum();
        stats.variance = sum_squared_diff / stats.count as f64;
        stats.std_dev = stats.variance.sqrt();
        
        // 计算中位数和四分位数需要排序后的数组
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // 计算中位数
        let mid = stats.count / 2;
        stats.median = if stats.count % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        };
        
        // 计算四分位数
        let q1_pos = stats.count / 4;
        let q2_pos = stats.count / 2;
        let q3_pos = (3 * stats.count) / 4;
        
        stats.quartiles = [
            sorted[q1_pos],
            sorted[q2_pos],
            sorted[q3_pos],
        ];
        
        // 计算偏度 (skewness)
        let sum_cubed_diff: f64 = values.iter()
            .map(|&x| (x - stats.mean).powi(3))
            .sum();
        stats.skewness = sum_cubed_diff / (stats.count as f64 * stats.std_dev.powi(3));
        
        // 计算峰度 (kurtosis)
        let sum_fourth_diff: f64 = values.iter()
            .map(|&x| (x - stats.mean).powi(4))
            .sum();
        stats.kurtosis = sum_fourth_diff / (stats.count as f64 * stats.variance.powi(2)) - 3.0;
        
        stats
    }
    
    /// 更新统计信息（在线计算）
    pub fn update(&mut self, value: f64) {
        // 更新最小值和最大值
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        
        // 增量更新平均值和方差
        let old_count = self.count as f64;
        let new_count = old_count + 1.0;
        let old_mean = self.mean;
        
        // 更新平均值
        self.mean = old_mean + (value - old_mean) / new_count;
        
        // 更新方差（使用Welford算法）
        if self.count > 0 {
            let delta = value - old_mean;
            let delta2 = value - self.mean;
            self.variance = (self.variance * old_count + delta * delta2) / new_count;
        }
        
        // 更新样本数
        self.count += 1;
    }
    
    /// 完成统计计算
    pub fn finalize(&mut self) {
        // 计算标准差
        self.std_dev = self.variance.sqrt();
        
        // 对于四分位数、偏度和峰度，需要收集所有数据点才能精确计算
        // 在这个简化的实现中，我们只使用已经计算的统计信息来估计
        // 真正的实现应该存储原始数据或使用更复杂的算法
        
        self.median = self.mean; // 简化，实际上不准确
        self.quartiles = [
            self.mean - 0.675 * self.std_dev, // 估计Q1
            self.mean,                       // 估计Q2 (中位数)
            self.mean + 0.675 * self.std_dev, // 估计Q3
        ];
        
        // 偏度和峰度需要完整数据，这里简化处理
        self.skewness = 0.0;
        self.kurtosis = 0.0;
    }
}

/// 文本字段统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFieldStats {
    /// 最小长度
    pub min_length: usize,
    /// 最大长度
    pub max_length: usize,
    /// 平均长度
    pub avg_length: f64,
    /// 样本数
    pub count: usize,
    /// 唯一值数量
    pub unique_values: usize,
    /// 空值数量
    pub empty_values: usize,
    /// 最常见的词
    pub most_common_words: HashMap<String, usize>,
    /// 最常见的字符
    pub most_common_chars: HashMap<char, usize>,
    /// 语言检测结果
    pub detected_language: Option<String>,
    /// 平均词数
    pub avg_word_count: f64,
    /// 平均句子数
    pub avg_sentence_count: f64,
}

impl Default for TextFieldStats {
    fn default() -> Self {
        Self {
            min_length: usize::MAX,
            max_length: 0,
            avg_length: 0.0,
            count: 0,
            unique_values: 0,
            empty_values: 0,
            most_common_words: HashMap::new(),
            most_common_chars: HashMap::new(),
            detected_language: None,
            avg_word_count: 0.0,
            avg_sentence_count: 0.0,
        }
    }
}

impl TextFieldStats {
    /// 创建新的文本字段统计信息
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 从文本数组计算统计信息
    pub fn from_texts(texts: &[String]) -> Self {
        let mut stats = Self::default();
        
        if texts.is_empty() {
            return stats;
        }
        
        // 基本统计信息
        stats.count = texts.len();
        
        // 计算长度相关统计
        let mut total_length = 0;
        let mut total_words = 0;
        let mut total_sentences = 0;
        
        // 词频和字符频率统计
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut char_counts: HashMap<char, usize> = HashMap::new();
        let mut unique_values = std::collections::HashSet::new();
        
        for text in texts {
            // 更新长度统计
            let length = text.len();
            stats.min_length = stats.min_length.min(length);
            stats.max_length = stats.max_length.max(length);
            total_length += length;
            
            // 检查空值
            if text.trim().is_empty() {
                stats.empty_values += 1;
            }
            
            // 计入唯一值
            unique_values.insert(text.clone());
            
            // 词频统计 (简单的以空格分割)
            let words: Vec<&str> = text.split_whitespace().collect();
            total_words += words.len();
            
            for word in words {
                *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
            }
            
            // 字符频率统计
            for ch in text.chars() {
                *char_counts.entry(ch).or_insert(0) += 1;
            }
            
            // 句子统计 (简单地以句号、问号和感叹号作为分隔符)
            let sentences: Vec<&str> = text.split(&['.', '?', '!'][..]).filter(|s| !s.trim().is_empty()).collect();
            total_sentences += sentences.len();
        }
        
        // 计算平均长度
        stats.avg_length = total_length as f64 / stats.count as f64;
        
        // 计算平均词数和句子数
        stats.avg_word_count = total_words as f64 / stats.count as f64;
        stats.avg_sentence_count = total_sentences as f64 / stats.count as f64;
        
        // 设置唯一值数量
        stats.unique_values = unique_values.len();
        
        // 提取最常见的词和字符
        stats.most_common_words = word_counts.into_iter()
            .filter(|(_, count)| *count > 1) // 只保留出现多次的词
            .collect();
        
        stats.most_common_chars = char_counts.into_iter()
            .filter(|(ch, _)| !ch.is_whitespace()) // 排除空白字符
            .collect();
        
        // 简单的语言检测 (实际应使用专门的语言检测库)
        // 这里仅作示例，未实现真正的语言检测逻辑
        stats.detected_language = Some("unknown".to_string());
        
        stats
    }
    
    /// 更新统计信息
    pub fn update(&mut self, text: &str) {
        // 更新长度统计
        let length = text.len();
        self.min_length = self.min_length.min(length);
        self.max_length = self.max_length.max(length);
        
        // 更新平均长度 (增量计算)
        let old_count = self.count as f64;
        let new_count = old_count + 1.0;
        let old_avg = self.avg_length;
        self.avg_length = old_avg + (length as f64 - old_avg) / new_count;
        
        // 更新样本数
        self.count += 1;
        
        // 检查空值
        if text.trim().is_empty() {
            self.empty_values += 1;
        }
        
        // 更新词频统计
        let words: Vec<&str> = text.split_whitespace().collect();
        
        // 更新平均词数 (增量计算)
        let old_word_avg = self.avg_word_count;
        self.avg_word_count = old_word_avg + (words.len() as f64 - old_word_avg) / new_count;
        
        for word in words {
            *self.most_common_words.entry(word.to_lowercase()).or_insert(0) += 1;
        }
        
        // 更新字符频率统计
        for ch in text.chars() {
            if !ch.is_whitespace() {
                *self.most_common_chars.entry(ch).or_insert(0) += 1;
            }
        }
        
        // 更新句子统计
        let sentences: Vec<&str> = text.split(&['.', '?', '!'][..]).filter(|s| !s.trim().is_empty()).collect();
        
        // 更新平均句子数 (增量计算)
        let old_sent_avg = self.avg_sentence_count;
        self.avg_sentence_count = old_sent_avg + (sentences.len() as f64 - old_sent_avg) / new_count;
    }
}

/// 类别字段统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalFieldStats {
    /// 类别数量
    pub category_count: usize,
    /// 最常见类别
    pub most_common: Option<String>,
    /// 最常见类别频率
    pub most_common_freq: f64,
    /// 类别频率分布
    pub frequencies: HashMap<String, f64>,
    /// 缺失值数量
    pub missing: usize,
}

/// 数据特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    /// 文本字段统计信息
    pub text_fields: HashMap<String, TextFieldStats>,
    /// 数值字段统计信息
    pub numeric_fields: HashMap<String, NumericStats>,
    /// 类别字段统计信息
    pub categorical_fields: HashMap<String, CategoricalFieldStats>,
    /// 样本数量
    pub sample_count: usize,
    /// 字段数量
    pub field_count: usize,
    /// 文本字段比例
    pub text_field_ratio: f64,
    /// 数值字段比例
    pub numeric_field_ratio: f64,
    /// 类别字段比例
    pub categorical_field_ratio: f64,
}

impl Default for DataCharacteristics {
    fn default() -> Self {
        Self {
            text_fields: HashMap::new(),
            numeric_fields: HashMap::new(),
            categorical_fields: HashMap::new(),
            sample_count: 0,
            field_count: 0,
            text_field_ratio: 0.0,
            numeric_field_ratio: 0.0,
            categorical_field_ratio: 0.0,
        }
    }
}

impl DataCharacteristics {
    /// 计算字段类型比例
    pub fn calculate_field_ratios(&mut self) {
        let total_fields = self.field_count as f64;
        if total_fields > 0.0 {
            self.text_field_ratio = self.text_fields.len() as f64 / total_fields;
            self.numeric_field_ratio = self.numeric_fields.len() as f64 / total_fields;
            self.categorical_field_ratio = self.categorical_fields.len() as f64 / total_fields;
        }
    }
    
    /// 检测数据集是否主要用于分类任务
    pub fn is_primarily_classification(&self) -> bool {
        // 如果有明显的类别字段，且其分布符合分类特征，则视为分类任务
        if let Some((_, stats)) = self.categorical_fields.iter()
            .max_by_key(|(_, stats)| stats.category_count) {
            
            // 分类特征通常类别数量中等且分布不太均匀
            let balanced_distribution = stats.frequencies.values()
                .filter(|&freq| *freq > 0.05 && *freq < 0.9) // 中等频率
                .count();
                
            // 如果有一个合适的分类字段，且数据集不是太大
            return stats.category_count >= 2 && 
                   stats.category_count <= 100 && 
                   balanced_distribution >= 2 &&
                   self.sample_count < 1_000_000;
        }
        false
    }
    
    /// 检测数据集是否主要用于回归任务
    pub fn is_primarily_regression(&self) -> bool {
        // 如果有明显的连续型数值目标字段，则视为回归任务
        if let Some((_, stats)) = self.numeric_fields.iter()
            .max_by(|(_, a), (_, b)| 
                (a.std_dev / a.mean.abs().max(1e-10))
                    .partial_cmp(&(b.std_dev / b.mean.abs().max(1e-10)))
                    .unwrap_or(std::cmp::Ordering::Equal)
            ) {
            
            // 回归目标通常有较大范围和连续分布
            return stats.count > 100 && 
                   stats.unique_values_ratio() > 0.5 && // 连续性指标
                   stats.has_normal_distribution();     // 正态分布检测
        }
        false
    }
    
    /// 检测数据集是否主要用于聚类任务
    pub fn is_primarily_clustering(&self) -> bool {
        // 聚类任务通常有多个数值特征，没有明显的目标变量
        // 且数据分布有明显的分群潜力
        let numeric_field_count = self.numeric_fields.len();
        let has_sufficient_numerics = numeric_field_count >= 3;
        
        // 检测数据是否有聚类潜力 (简化版)
        let has_clustering_potential = self.numeric_fields.values()
            .any(|stats| stats.variance > 0.0 && stats.has_multimodal_distribution());
            
        has_sufficient_numerics && has_clustering_potential && 
        !self.is_primarily_classification() && !self.is_primarily_regression()
    }
    
    /// 检测数据是否高度混合
    pub fn is_highly_mixed(&self) -> bool {
        self.text_field_ratio > 0.3 && 
        self.numeric_field_ratio > 0.3 && 
        self.categorical_field_ratio > 0.3
    }
    
    /// 获取缺失值比例
    pub fn get_missing_value_ratio(&self) -> f64 {
        let mut total_missing = 0;
        let mut total_fields = 0;
        
        // 统计数值字段缺失值
        for stats in self.numeric_fields.values() {
            total_missing += stats.missing;
            total_fields += stats.count;
        }
        
        // 统计文本字段缺失值
        for stats in self.text_fields.values() {
            total_missing += stats.empty_values;
            total_fields += stats.count;
        }
        
        // 统计分类字段缺失值
        for stats in self.categorical_fields.values() {
            total_missing += stats.missing;
            total_fields += stats.frequency_total();
        }
        
        if total_fields > 0 {
            total_missing as f64 / total_fields as f64
        } else {
            0.0
        }
    }
    
    /// 检测数据集中是否包含时间敏感字段
    pub fn has_time_sensitive_fields(&self) -> bool {
        // 检查字段名称中的时间相关关键词
        let time_keywords = [
            "time", "date", "year", "month", "day", "hour", "minute",
            "timestamp", "created", "updated", "modified", 
            "period", "duration", "interval"
        ];
        
        let has_time_in_fields = self.text_fields.keys()
            .chain(self.numeric_fields.keys())
            .chain(self.categorical_fields.keys())
            .any(|field| {
                let field_lower = field.to_lowercase();
                time_keywords.iter().any(|&keyword| field_lower.contains(keyword))
            });
            
        // 检查数值字段是否符合时间序列特征
        let has_time_series_pattern = self.numeric_fields.values()
            .any(|stats| stats.has_temporal_pattern());
            
        has_time_in_fields || has_time_series_pattern
    }
    
    /// 检测特征之间的相关性
    pub fn detect_correlations(&self) -> HashMap<(String, String), f64> {
        let mut correlations = HashMap::new();
        let numeric_fields: Vec<_> = self.numeric_fields.keys().cloned().collect();
        
        // 暂时只计算数值字段间的相关性
        for i in 0..numeric_fields.len() {
            for j in i+1..numeric_fields.len() {
                let field1 = &numeric_fields[i];
                let field2 = &numeric_fields[j];
                
                // 注意：计算真实的相关系数需要原始数据，而DataCharacteristics只包含统计量
                // 这里返回0.0作为占位值，表示相关性未知
                // 如果需要真实的相关系数，应该使用包含原始数据的其他方法
                let correlation = 0.0; // 占位值：需要原始数据才能计算真实相关系数
                correlations.insert((field1.clone(), field2.clone()), correlation);
            }
        }
        
        correlations
    }
    
    /// 获取数据集的复杂度评分
    pub fn get_complexity_score(&self) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;
        
        // 考虑字段数量
        score += (self.field_count as f64 / 20.0).min(1.0);
        factors += 1;
        
        // 考虑样本数量
        score += (self.sample_count as f64 / 10000.0).min(1.0);
        factors += 1;
        
        // 考虑数据类型多样性
        let type_entropy = -(self.text_field_ratio * self.text_field_ratio.ln() +
                           self.numeric_field_ratio * self.numeric_field_ratio.ln() +
                           self.categorical_field_ratio * self.categorical_field_ratio.ln()) / 3f64.ln();
        score += type_entropy;
        factors += 1;
        
        // 考虑数值复杂度
        if !self.numeric_fields.is_empty() {
            let avg_variance = self.numeric_fields.values()
                .map(|s| s.variance)
                .sum::<f64>() / self.numeric_fields.len() as f64;
            score += (avg_variance.ln() + 10.0) / 10.0;
            factors += 1;
        }
        
        // 返回归一化得分
        if factors > 0 {
            (score / factors as f64).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
    
    /// 生成数据集摘要
    pub fn generate_summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        
        summary.insert("sample_count".to_string(), self.sample_count.to_string());
        summary.insert("field_count".to_string(), self.field_count.to_string());
        summary.insert("text_field_ratio".to_string(), format!("{:.2}", self.text_field_ratio));
        summary.insert("numeric_field_ratio".to_string(), format!("{:.2}", self.numeric_field_ratio));
        summary.insert("categorical_field_ratio".to_string(), format!("{:.2}", self.categorical_field_ratio));
        summary.insert("missing_value_ratio".to_string(), format!("{:.2}", self.get_missing_value_ratio()));
        summary.insert("complexity_score".to_string(), format!("{:.2}", self.get_complexity_score()));
        
        // 检测可能的任务类型
        let task_type = if self.is_primarily_classification() {
            "classification"
        } else if self.is_primarily_regression() {
            "regression"
        } else if self.is_primarily_clustering() {
            "clustering"
        } else {
            "mixed/unknown"
        };
        summary.insert("suggested_task_type".to_string(), task_type.to_string());
        
        summary
    }
}

/// 扩展NumericStats结构以支持更高级的分析
impl NumericStats {
    /// 计算唯一值比例
    pub fn unique_values_ratio(&self) -> f64 {
        // 注意：实际实现需要跟踪唯一值的数量
        // 这里使用启发式估计
        let estimated_unique_values = (self.max - self.min) / 
                                      self.std_dev.max(1e-10) *
                                      0.1; // 启发式因子
        
        (estimated_unique_values / self.count as f64).min(1.0)
    }
    
    /// 检测数值是否呈现正态分布
    pub fn has_normal_distribution(&self) -> bool {
        // 使用偏度和峰度检测正态性
        // 正态分布的偏度为0，峰度为3
        let skewness_normal = self.skewness.abs() < 0.5;
        let kurtosis_normal = (self.kurtosis - 3.0).abs() < 1.0;
        
        skewness_normal && kurtosis_normal
    }
    
    /// 检测是否存在多峰分布 (可能表明有聚类结构)
    pub fn has_multimodal_distribution(&self) -> bool {
        // 简化检测 - 使用峰度估计
        // 多峰分布通常峰度低于正态分布
        self.kurtosis < 2.5
    }
    
    /// 检测是否存在时间序列模式
    pub fn has_temporal_pattern(&self) -> bool {
        // 这是一个启发式检测，实际需要时间序列分析
        // 时间序列通常有特定范围和规律性
        
        // 检查范围是否符合时间特征 (如年份、时间戳等)
        let range_suggests_time = 
            (self.min >= 1970.0 && self.max <= 2100.0) || // 可能是年份
            (self.min >= 0.0 && self.max <= 24.0) ||      // 可能是小时
            (self.min >= 0.0 && self.max <= 60.0) ||      // 可能是分钟/秒
            (self.min >= 946684800.0 && self.max <= 2524608000.0); // Unix时间戳范围
            
        // 检查方差和分布是否符合时间序列特征
        let distribution_suggests_time = self.variance > 0.0 && 
                                         self.skewness.abs() < 1.0; // 时间序列通常分布较均匀
        
        range_suggests_time && distribution_suggests_time
    }
}

/// 扩展CategoricalFieldStats以获取总频率
impl CategoricalFieldStats {
    /// 获取频率总和 (用于计算缺失值比例)
    pub fn frequency_total(&self) -> usize {
        self.frequencies.values()
            .map(|&freq| (freq * 100.0).round() as usize)
            .sum()
    }
} 