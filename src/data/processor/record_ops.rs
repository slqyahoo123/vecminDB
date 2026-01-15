// Record Operations Module
// 记录操作模块

use std::collections::HashMap;
use std::error::Error as StdError;
use std::hash::{Hash, Hasher};
use crate::data::processor::config::ProcessorConfig as ImportedProcessorConfig;
use crate::data::record::{Record, Value as RecordFieldValue};
use crate::data::value::DataValue;

/// 记录处理器
pub struct RecordProcessor {
}

impl RecordProcessor {
    /// 创建新的记录处理器
    pub fn new() -> Self {
        Self {}
    }

    /// 处理单条记录
    pub fn process_single_record_with_config(&self, record: &Record, config: &ImportedProcessorConfig) -> std::result::Result<Record, Box<dyn StdError>> {
        let mut processed_record = record.clone();
        
        // 应用数据清理
        if config.trim_whitespace || config.clean_special_chars {
            self.clean_record_data(&mut processed_record)?;
        }
        
        // 应用标准化
        if config.normalize {
            self.normalize_record_data(&mut processed_record)?;
        }
        
        // 处理缺失值
        if config.handle_missing_values || config.fill_missing {
            self.handle_missing_values(&mut processed_record)?;
        }
        
        // 应用转换器
        self.process_record(&mut processed_record)?;
        
        Ok(processed_record)
    }

    /// 清理记录数据
    pub fn clean_record_data(&self, record: &mut Record) -> std::result::Result<(), Box<dyn StdError>> {
        // 复用 DataCleaner 中的实现，保持行为一致
        let cleaner = crate::data::processor::data_ops::DataCleaner::new();
        cleaner.clean_record_data(record)
    }
    
    /// 标准化记录数据
    pub fn normalize_record_data(&self, record: &mut Record) -> std::result::Result<(), Box<dyn StdError>> {
        // 简单数值标准化：对 Number/Float/Integer 做线性缩放
        for (_field_name, field_val) in record.fields.iter_mut() {
            if let RecordFieldValue::Data(dv) = field_val {
                match dv {
                    DataValue::Number(f) | DataValue::Float(f) => {
                        if *f != 0.0 {
                            *f = (*f - 0.5) / 0.5;
                        }
                    }
                    DataValue::Integer(i) => {
                        let normalized = (*i as f64 - 50.0) / 50.0;
                        *dv = DataValue::Number(normalized);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
    
    /// 处理缺失值
    pub fn handle_missing_values(&self, record: &mut Record) -> std::result::Result<(), Box<dyn StdError>> {
        let mut fields_to_update = Vec::new();
        
        for (field_name, field_val) in &record.fields {
            if let RecordFieldValue::Data(dv) = field_val {
                match dv {
                    DataValue::Null => {
                        let default_value = self.get_default_value_for_field(field_name);
                        fields_to_update.push((field_name.clone(), default_value));
                    }
                    DataValue::String(s) | DataValue::Text(s) if s.is_empty() => {
                        fields_to_update.push((field_name.clone(), DataValue::String("未知".to_string())));
                    }
                    _ => {}
                }
            }
        }
        
        // 更新缺失值
        for (field_name, new_value) in fields_to_update {
            record.fields.insert(field_name, RecordFieldValue::Data(new_value));
        }
        
        Ok(())
    }
    
    /// 获取字段的默认值
    pub fn get_default_value_for_field(&self, field_name: &str) -> DataValue {
        // 根据字段名推断合适的默认值
        if field_name.contains("age") || field_name.contains("count") || field_name.contains("num") {
            DataValue::Integer(0)
        } else if field_name.contains("rate") || field_name.contains("score") || field_name.contains("price") {
            DataValue::Number(0.0)
        } else if field_name.contains("flag") || field_name.contains("is_") || field_name.contains("has_") {
            DataValue::Boolean(false)
        } else {
            DataValue::String("未知".to_string())
        }
    }

    /// 基本的记录处理
    /// 
    /// 这是一个可选的扩展点，子类可以重写此方法来实现自定义的记录处理逻辑
    /// 比如字段变换、数据类型转换等
    fn process_record(&self, _record: &mut Record) -> std::result::Result<(), Box<dyn StdError>> {
        // 默认实现：不做任何处理，直接返回成功
        // 子类可以重写此方法来实现具体的处理逻辑
        Ok(())
    }

    /// 应用字段映射
    pub fn apply_field_mapping(&self, record: &mut Record, mapping: &HashMap<String, String>) -> std::result::Result<(), Box<dyn StdError>> {
        let mut new_data = HashMap::new();
        
        for (old_name, value) in &record.fields {
            let new_name = mapping.get(old_name).unwrap_or(old_name);
            new_data.insert(new_name.clone(), value.clone());
        }
        
        record.fields = new_data;
        Ok(())
    }

    /// 过滤字段
    pub fn filter_fields(&self, record: &mut Record, allowed_fields: &[String]) -> std::result::Result<(), Box<dyn StdError>> {
        record.fields.retain(|field_name, _| allowed_fields.contains(field_name));
        Ok(())
    }

    /// 添加计算字段
    pub fn add_computed_fields(&self, record: &mut Record, computations: &HashMap<String, String>) -> std::result::Result<(), Box<dyn StdError>> {
        for (field_name, expression) in computations {
            let computed_value = self.evaluate_expression(expression, record)?;
            record.fields.insert(field_name.clone(), RecordFieldValue::Data(computed_value));
        }
        Ok(())
    }

    /// 表达式求值
    /// 
    /// 支持以下表达式：
    /// - len(field_name): 获取字符串或数组的长度
    /// - upper(field_name): 将字符串转换为大写
    /// - lower(field_name): 将字符串转换为小写
    /// - field_name: 直接返回字段值
    fn evaluate_expression(&self, expression: &str, record: &Record) -> std::result::Result<DataValue, Box<dyn StdError>> {
        // 表达式解析器实现
        // 支持基本操作：len, upper, lower, 以及直接字段访问
        
        if expression.starts_with("len(") && expression.ends_with(")") {
            let field_name = &expression[4..expression.len()-1];
            if let Some(field_val) = record.fields.get(field_name) {
                if let RecordFieldValue::Data(dv) = field_val {
                    match dv {
                        DataValue::String(s) | DataValue::Text(s) => Ok(DataValue::Integer(s.len() as i64)),
                        DataValue::Array(arr) => Ok(DataValue::Integer(arr.len() as i64)),
                        _ => Ok(DataValue::Integer(0)),
                    }
                } else {
                    Ok(DataValue::Integer(0))
                }
            } else {
                Ok(DataValue::Integer(0))
            }
        } else if expression.starts_with("upper(") && expression.ends_with(")") {
            let field_name = &expression[6..expression.len()-1];
            if let Some(RecordFieldValue::Data(DataValue::String(s))) = record.fields.get(field_name) {
                Ok(DataValue::String(s.to_uppercase()))
            } else {
                Ok(DataValue::String(String::new()))
            }
        } else if expression.starts_with("lower(") && expression.ends_with(")") {
            let field_name = &expression[6..expression.len()-1];
            if let Some(RecordFieldValue::Data(DataValue::String(s))) = record.fields.get(field_name) {
                Ok(DataValue::String(s.to_lowercase()))
            } else {
                Ok(DataValue::String(String::new()))
            }
        } else {
            // 默认返回字段值或空字符串
            if let Some(field_val) = record.fields.get(expression) {
                if let RecordFieldValue::Data(dv) = field_val {
                    Ok(dv.clone())
                } else {
                    Ok(DataValue::Null)
                }
            } else {
                Ok(DataValue::Null)
            }
        }
    }

    /// 验证记录
    pub fn validate_record(&self, record: &Record, rules: &[ValidationRule]) -> Vec<String> {
        let mut errors = Vec::new();
        
        for rule in rules {
            if let Err(error) = self.apply_validation_rule(record, rule) {
                errors.push(error);
            }
        }
        
        errors
    }

    /// 应用验证规则
    fn apply_validation_rule(&self, record: &Record, rule: & ValidationRule) -> std::result::Result<(), String> {
        match rule {
            ValidationRule::Required(field_name) => {
                if !record.fields.contains_key(field_name) {
                    return Err(format!("必填字段 {} 缺失", field_name));
                }
                if let Some(RecordFieldValue::Data(DataValue::Null)) = record.fields.get(field_name) {
                    return Err(format!("必填字段 {} 为空", field_name));
                }
                Ok(())
            },
            ValidationRule::Type(field_name, expected_type) => {
                if let Some(field_val) = record.fields.get(field_name) {
                    if let RecordFieldValue::Data(dv) = field_val {
                        if !self.matches_type(dv, expected_type) {
                            return Err(format!(
                                "字段 {} 类型不匹配: 期望 {:?}, 实际 {:?}",
                                field_name, expected_type, dv
                            ));
                        }
                    } else {
                        return Err(format!(
                            "字段 {} 类型不匹配: 期望 {:?}, 实际 {:?}",
                            field_name, expected_type, field_val
                        ));
                    }
                }
                Ok(())
            },
            ValidationRule::Range(field_name, min, max) => {
                if let Some(field_val) = record.fields.get(field_name) {
                    let numeric_value = if let RecordFieldValue::Data(dv) = field_val {
                        match dv {
                            DataValue::Integer(i) => Some(*i as f64),
                            DataValue::Number(f) | DataValue::Float(f) => Some(*f),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    
                    if let Some(val) = numeric_value {
                        if val < *min || val > *max {
                            return Err(format!(
                                "字段 {} 超出范围: {} 不在 [{}, {}] 范围内",
                                field_name, val, min, max
                            ));
                        }
                    }
                }
                Ok(())
            },
            ValidationRule::Pattern(field_name, pattern) => {
                if let Some(RecordFieldValue::Data(DataValue::String(s))) = record.fields.get(field_name) {
                    // 这里应该使用正则表达式库
                    // 为了简化，我们只做简单的包含检查
                    if !s.contains(pattern) {
                        return Err(format!(
                            "字段 {} 不匹配模式: {} 不包含 {}",
                            field_name, s, pattern
                        ));
                    }
                }
                Ok(())
            },
        }
    }

    /// 检查值类型
    fn matches_type(&self, value: &DataValue, expected_type: &RecordValueType) -> bool {
        match (value, expected_type) {
            (DataValue::String(_) | DataValue::Text(_), RecordValueType::String) => true,
            (DataValue::Integer(_), RecordValueType::Integer) => true,
            (DataValue::Float(_) | DataValue::Number(_), RecordValueType::Float) => true,
            (DataValue::Boolean(_), RecordValueType::Boolean) => true,
            (DataValue::Array(_), RecordValueType::Array) => true,
            (DataValue::Object(_), RecordValueType::Object) => true,
            (DataValue::Null, _) => true, // 空值与任何类型兼容
            _ => false,
        }
    }
}

/// 特征提取器
pub struct FeatureExtractor {
}

impl FeatureExtractor {
    /// 创建新的特征提取器
    pub fn new() -> Self {
        Self {}
    }

    /// 从记录中提取特征
    pub fn extract_features_from_records(&self, records: &[Record], _config: &ImportedProcessorConfig) -> std::result::Result<crate::compat::tensor::TensorData, Box<dyn StdError>> {
        if records.is_empty() {
            return Ok(crate::compat::tensor::TensorData {
                shape: vec![0],
                data: crate::compat::tensor::TensorValues::F32(Vec::new()),
                dtype: crate::compat::tensor::DataType::Float32,
                metadata: std::collections::HashMap::new(),
            });
        }
        
        // 收集所有数值特征
        let mut feature_matrix = Vec::new();
        let mut feature_dimension = 0;
        
        for record in records {
            let mut row_features = Vec::new();
            
            for (_field_name, field_val) in &record.fields {
                if let RecordFieldValue::Data(dv) = field_val {
                    match dv {
                        DataValue::Number(f) | DataValue::Float(f) => row_features.push(*f as f32),
                        DataValue::Integer(i) => row_features.push(*i as f32),
                        DataValue::Boolean(b) => row_features.push(if *b { 1.0 } else { 0.0 }),
                        DataValue::String(s) | DataValue::Text(s) => {
                            let hash_feature = self.string_to_feature(s);
                            row_features.push(hash_feature);
                        },
                        _ => row_features.push(0.0),
                    }
                } else {
                    row_features.push(0.0);
                }
            }
            
            if feature_dimension == 0 {
                feature_dimension = row_features.len();
            } else if row_features.len() != feature_dimension {
                // 补齐或截断特征以保持一致的维度
                row_features.resize(feature_dimension, 0.0);
            }
            
            feature_matrix.extend(row_features);
        }
        
        Ok(crate::compat::tensor::TensorData {
            shape: vec![records.len(), feature_dimension],
            data: crate::compat::tensor::TensorValues::F32(feature_matrix),
            dtype: crate::compat::tensor::DataType::Float32,
            metadata: std::collections::HashMap::new(),
        })
    }
    
    /// 内联提取（用于自检/就绪检查）
    pub fn extract_inline(&self, records: &[Record]) -> std::result::Result<crate::compat::tensor::TensorData, Box<dyn StdError>> {
        self.extract_features_from_records(records, &ImportedProcessorConfig::default())
    }
    
    /// 从记录中提取标签
    pub fn extract_labels_from_records(&self, records: &[Record], config: &ImportedProcessorConfig) -> std::result::Result<Option<crate::compat::tensor::TensorData>, Box<dyn StdError>> {
        // 如果配置中指定了标签字段，则提取标签
        if let Some(ref label_field) = config.label_column {
            let mut labels = Vec::new();
            
            for record in records {
                let label = if let Some(field_val) = record.fields.get(label_field) {
                    if let RecordFieldValue::Data(dv) = field_val {
                        match dv {
                            DataValue::Integer(i) => *i as f32,
                            DataValue::Number(f) | DataValue::Float(f) => *f as f32,
                            DataValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
                            DataValue::String(s) | DataValue::Text(s) => self.string_to_label(s),
                            _ => 0.0,
                        }
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                labels.push(label);
            }
            
            Ok(Some(crate::compat::tensor::TensorData {
                shape: vec![records.len()],
                data: crate::compat::tensor::TensorValues::F32(labels),
                dtype: crate::compat::tensor::DataType::Float32,
                metadata: std::collections::HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// 字符串转换为特征值
    pub fn string_to_feature(&self, s: &str) -> f32 {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        let hash = hasher.finish();
        
        // 将哈希值归一化到[-1, 1]范围
        (hash as f32 / u64::MAX as f32) * 2.0 - 1.0
    }
    
    /// 字符串转换为标签值
    pub fn string_to_label(&self, s: &str) -> f32 {
        // 简单的字符串到标签的映射
        match s.to_lowercase().as_str() {
            "yes" | "true" | "positive" | "1" => 1.0,
            "no" | "false" | "negative" | "0" => 0.0,
            _ => {
                // 对于其他字符串，使用哈希值
                self.string_to_feature(s).abs()
            }
        }
    }

    /// 提取文本特征
    pub fn extract_text_features(&self, text: &str) -> HashMap<String, f32> {
        let mut features = HashMap::new();
        
        // 基本统计特征
        features.insert("length".to_string(), text.len() as f32);
        features.insert("word_count".to_string(), text.split_whitespace().count() as f32);
        features.insert("char_count".to_string(), text.chars().count() as f32);
        
        // 字符频率统计
        let mut char_counts = HashMap::new();
        for ch in text.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }
        
        features.insert("unique_chars".to_string(), char_counts.len() as f32);
        features.insert("avg_word_length".to_string(), 
            if text.split_whitespace().count() > 0 {
                text.chars().count() as f32 / text.split_whitespace().count() as f32
            } else {
                0.0
            }
        );
        
        // 特殊字符统计
        features.insert("digit_count".to_string(), text.chars().filter(|c| c.is_ascii_digit()).count() as f32);
        features.insert("upper_count".to_string(), text.chars().filter(|c| c.is_uppercase()).count() as f32);
        features.insert("lower_count".to_string(), text.chars().filter(|c| c.is_lowercase()).count() as f32);
        features.insert("punct_count".to_string(), text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32);
        
        features
    }

    /// 提取数值特征
    pub fn extract_numeric_features(&self, values: &[f64]) -> HashMap<String, f32> {
        let mut features = HashMap::new();
        
        if values.is_empty() {
            return features;
        }
        
        let count = values.len() as f64;
        let sum: f64 = values.iter().sum();
        let mean = sum / count;
        
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / count;
        let std_dev = variance.sqrt();
        
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        let median = if sorted_values.len() % 2 == 0 {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };
        
        features.insert("count".to_string(), count as f32);
        features.insert("mean".to_string(), mean as f32);
        features.insert("median".to_string(), median as f32);
        features.insert("std_dev".to_string(), std_dev as f32);
        features.insert("variance".to_string(), variance as f32);
        features.insert("min".to_string(), min as f32);
        features.insert("max".to_string(), max as f32);
        features.insert("range".to_string(), (max - min) as f32);
        
        // 四分位数
        let q1_idx = sorted_values.len() / 4;
        let q3_idx = (sorted_values.len() * 3) / 4;
        let q1 = sorted_values[q1_idx];
        let q3 = sorted_values[q3_idx];
        let iqr = q3 - q1;
        
        features.insert("q1".to_string(), q1 as f32);
        features.insert("q3".to_string(), q3 as f32);
        features.insert("iqr".to_string(), iqr as f32);
        
        features
    }
}

/// 验证规则枚举
#[derive(Debug, Clone)]
pub enum ValidationRule {
    /// 必填字段
    Required(String),
    /// 类型检查
    Type(String, RecordValueType),
    /// 数值范围检查
    Range(String, f64, f64),
    /// 模式匹配
    Pattern(String, String),
}

/// 记录值类型枚举
#[derive(Debug, Clone, PartialEq)]
pub enum RecordValueType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
    Null,
}

/// 记录转换器
pub struct RecordTransformer {
}

impl RecordTransformer {
    /// 创建新的记录转换器
    pub fn new() -> Self {
        Self {}
    }

    /// 批量转换记录
    pub fn transform_records(&self, records: &mut Vec<Record>, transformations: &[Transformation]) -> std::result::Result<(), Box<dyn StdError>> {
        for record in records.iter_mut() {
            for transformation in transformations {
                self.apply_transformation(record, transformation)?;
            }
        }
        Ok(())
    }

    /// 应用单个转换
    fn apply_transformation(&self, record: &mut Record, transformation: &Transformation) -> std::result::Result<(), Box<dyn StdError>> {
        match transformation {
            Transformation::Rename(old_name, new_name) => {
                if let Some(value) = record.fields.remove(old_name) {
                    record.fields.insert(new_name.clone(), value);
                }
            },
            Transformation::Convert(field_name, target_type) => {
                if let Some(field_val) = record.fields.get_mut(field_name) {
                    if let RecordFieldValue::Data(dv) = field_val {
                        let new_dv = self.convert_value(dv, target_type)?;
                        *dv = new_dv;
                    }
                }
            },
            Transformation::Scale(field_name, factor) => {
                if let Some(field_val) = record.fields.get_mut(field_name) {
                    if let RecordFieldValue::Data(dv) = field_val {
                        match dv {
                            DataValue::Integer(i) => {
                                let scaled = (*i as f64 * factor) as i64;
                                *i = scaled;
                            }
                            DataValue::Number(f) | DataValue::Float(f) => {
                                *f = *f * factor;
                            }
                            _ => return Err(format!("无法缩放非数值字段: {}", field_name).into()),
                        }
                    }
                }
            },
        }
        Ok(())
    }

    /// 转换值类型
    fn convert_value(&self, value: &DataValue, target_type: &RecordValueType) -> std::result::Result<DataValue, Box<dyn StdError>> {
        match (value, target_type) {
            (DataValue::String(s) | DataValue::Text(s), RecordValueType::Integer) => {
                match s.parse::<i64>() {
                    Ok(i) => Ok(DataValue::Integer(i)),
                    Err(_) => Err(format!("无法将字符串 '{}' 转换为整数", s).into()),
                }
            },
            (DataValue::String(s) | DataValue::Text(s), RecordValueType::Float) => {
                match s.parse::<f64>() {
                    Ok(f) => Ok(DataValue::Number(f)),
                    Err(_) => Err(format!("无法将字符串 '{}' 转换为浮点数", s).into()),
                }
            },
            (DataValue::Integer(i), RecordValueType::Float) => {
                Ok(DataValue::Number(*i as f64))
            },
            (DataValue::Number(f) | DataValue::Float(f), RecordValueType::Integer) => {
                Ok(DataValue::Integer(*f as i64))
            },
            (DataValue::Integer(i), RecordValueType::String) => {
                Ok(DataValue::String(i.to_string()))
            },
            (DataValue::Number(f) | DataValue::Float(f), RecordValueType::String) => {
                Ok(DataValue::String(f.to_string()))
            },
            (DataValue::Boolean(b), RecordValueType::String) => {
                Ok(DataValue::String(b.to_string()))
            },
            _ => Ok(value.clone()),
        }
    }
}

/// 转换操作枚举
#[derive(Debug, Clone)]
pub enum Transformation {
    /// 重命名字段
    Rename(String, String),
    /// 转换字段类型
    Convert(String, RecordValueType),
    /// 缩放数值字段
    Scale(String, f64),
} 