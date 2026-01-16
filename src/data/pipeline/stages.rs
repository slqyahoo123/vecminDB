use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;

use serde_json::Value;
use crate::error::{Error, Result};
use crate::data::{DataSchema, DataBatch, DataValue};
use crate::data::schema::schema::FieldType;
use crate::data::record::DataRecord;
use log::{debug, info, warn};

use super::core::{DataStage, DataTransformer, PipelineStage, StageExecutionContext, StageConfig};

/// 类型转换阶段
pub struct TypeConverterStage {
    /// 阶段名称
    name: String,
    
    /// 需要转换的字段及目标类型
    field_conversions: HashMap<String, FieldType>,
    
    /// 是否跳过错误
    skip_errors: bool,
}

impl TypeConverterStage {
    /// 创建新的类型转换阶段
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            field_conversions: HashMap::new(),
            skip_errors: false,
        }
    }
    
    /// 添加字段转换
    pub fn add_conversion<S: Into<String>>(&mut self, field_name: S, target_type: FieldType) -> &mut Self {
        self.field_conversions.insert(field_name.into(), target_type);
        self
    }
    
    /// 设置是否跳过错误
    pub fn with_skip_errors(&mut self, skip: bool) -> &mut Self {
        self.skip_errors = skip;
        self
    }
    
    /// 尝试将值转换为目标类型
    fn convert_value(&self, value: &DataValue, target_type: &FieldType) -> Result<DataValue> {
        match (value, target_type) {
            // 已经是目标类型
            (DataValue::Null, _) => Ok(DataValue::Null),
            (DataValue::Integer(_) | DataValue::Float(_), FieldType::Numeric) => Ok(value.clone()),
            (DataValue::Boolean(_), FieldType::Boolean) => Ok(value.clone()),
            (DataValue::String(_) | DataValue::Text(_), FieldType::Text) => Ok(value.clone()),
            (DataValue::DateTime(_), FieldType::DateTime) => Ok(value.clone()),
            (DataValue::Array(_), FieldType::Array(_)) => Ok(value.clone()),
            (DataValue::Object(_), FieldType::Object(_)) => Ok(value.clone()),
            
            // 需要转换
            (DataValue::String(s) | DataValue::Text(s), FieldType::Numeric) => {
                match s.parse::<f64>() {
                    Ok(f) => Ok(DataValue::Float(f)),
                    Err(_) => {
                        if self.skip_errors {
                            Ok(DataValue::Null)
                        } else {
                            Err(Error::invalid_input(format!("无法将字符串 '{}' 转换为数字", s)))
                        }
                    }
                }
            },
            (DataValue::String(s) | DataValue::Text(s), FieldType::Boolean) => {
                match s.to_lowercase().as_str() {
                    "true" | "yes" | "1" | "y" => Ok(DataValue::Boolean(true)),
                    "false" | "no" | "0" | "n" => Ok(DataValue::Boolean(false)),
                    _ => {
                        if self.skip_errors {
                            Ok(DataValue::Null)
                        } else {
                            Err(Error::invalid_input(format!("无法将字符串 '{}' 转换为布尔值", s)))
                        }
                    }
                }
            },
            (DataValue::Integer(i), FieldType::Numeric) => Ok(DataValue::Float(*i as f64)),
            (DataValue::Integer(i), FieldType::Text) => Ok(DataValue::String(i.to_string())),
            (DataValue::Integer(i), FieldType::Boolean) => Ok(DataValue::Boolean(*i != 0)),
            (DataValue::Float(f), FieldType::Text) => Ok(DataValue::String(f.to_string())),
            (DataValue::Boolean(b), FieldType::Text) => Ok(DataValue::String(b.to_string())),
            
            // 其他情况暂时不支持
            _ => {
                if self.skip_errors {
                    Ok(DataValue::Null)
                } else {
                    Err(Error::invalid_input(format!(
                        "不支持从 {:?} 转换为 {:?}", 
                        value, 
                        target_type
                    )))
                }
            }
        }
    }
}

impl DataStage for TypeConverterStage {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn process(&self, batch: &mut DataBatch) -> Result<()> {
        let schema = batch.schema()
            .ok_or_else(|| Error::validation("批次缺少schema".to_string()))?;
        
        // 对每条记录进行处理
        for record in &mut batch.records {
            for (field_name, target_type) in &self.field_conversions {
                // HashMap 使用字段名访问，而不是索引
                if let Some(current_value) = record.get(field_name) {
                    match self.convert_value(current_value, target_type) {
                        Ok(new_value) => {
                            record.insert(field_name.clone(), new_value);
                        },
                        Err(e) => {
                            if self.skip_errors {
                                warn!("类型转换错误，已忽略: {}", e);
                                record.insert(field_name.clone(), DataValue::Null);
                            } else {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn output_schema(&self, input_schema: &DataSchema) -> Result<DataSchema> {
        let mut new_fields = input_schema.fields().clone();
        
        // 更新字段类型
        for field in &mut new_fields {
            if let Some(target_type) = self.field_conversions.get(field.name()) {
                field.set_field_type(target_type.clone());
            }
        }
        
        let mut new_schema = input_schema.clone();
        new_schema.fields = new_fields;
        
        Ok(new_schema)
    }
}

/// 缺失值处理策略
#[derive(Debug, Clone, PartialEq)]
pub enum MissingValueStrategy {
    /// 删除包含缺失值的记录
    DropRecord,
    
    /// 用固定值替换缺失值
    FillWithConstant(DataValue),
    
    /// 用平均值替换缺失值(仅适用于数值型)
    FillWithMean,
    
    /// 用中位数替换缺失值(仅适用于数值型)
    FillWithMedian,
    
    /// 用众数替换缺失值
    FillWithMode,
}

/// 缺失值处理阶段
pub struct MissingValueHandlerStage {
    /// 阶段名称
    name: String,
    
    /// 需要处理的字段及策略
    field_strategies: HashMap<String, MissingValueStrategy>,
    
    /// 自定义替换值(用于某些策略)
    replacement_values: HashMap<String, DataValue>,
}

impl MissingValueHandlerStage {
    /// 创建新的缺失值处理阶段
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            field_strategies: HashMap::new(),
            replacement_values: HashMap::new(),
        }
    }
    
    /// 添加字段处理策略
    pub fn add_strategy<S: Into<String>>(&mut self, field_name: S, strategy: MissingValueStrategy) -> &mut Self {
        self.field_strategies.insert(field_name.into(), strategy);
        self
    }
    
    /// 计算数值字段的平均值
    fn calculate_mean(&self, batch: &DataBatch, field_name: &str) -> Option<DataValue> {
        let mut sum_int = 0i64;
        let mut sum_float = 0.0f64;
        let mut count = 0;
        let mut is_float = false;
        
        for record in batch.records() {
            if let Some(value) = record.get(field_name) {
                match value {
                    DataValue::Integer(i) => {
                        sum_int += i;
                        count += 1;
                    },
                    DataValue::Float(f) => {
                        sum_float += f;
                        count += 1;
                        is_float = true;
                    },
                    _ => {}
                }
            }
        }
        
        if count == 0 {
            return None;
        }
        
        if is_float {
            Some(DataValue::Float(sum_float / count as f64))
        } else {
            Some(DataValue::Integer(sum_int / count as i64))
        }
    }
    
    /// 计算数值字段的中位数
    fn calculate_median(&self, batch: &DataBatch, field_name: &str) -> Option<DataValue> {
        let mut int_values = Vec::new();
        let mut float_values = Vec::new();
        let mut is_float = false;
        
        for record in batch.records() {
            if let Some(value) = record.get(field_name) {
                match value {
                    DataValue::Integer(i) => int_values.push(*i),
                    DataValue::Float(f) => {
                        float_values.push(*f);
                        is_float = true;
                    },
                    _ => {}
                }
            }
        }
        
        if is_float {
            if float_values.is_empty() {
                return None;
            }
            
            float_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = float_values.len() / 2;
            
            let median = if float_values.len() % 2 == 0 {
                (float_values[mid - 1] + float_values[mid]) / 2.0
            } else {
                float_values[mid]
            };
            
            Some(DataValue::Float(median))
        } else {
            if int_values.is_empty() {
                return None;
            }
            
            int_values.sort();
            let mid = int_values.len() / 2;
            
            let median = if int_values.len() % 2 == 0 {
                (int_values[mid - 1] + int_values[mid]) / 2
            } else {
                int_values[mid]
            };
            
            Some(DataValue::Integer(median))
        }
    }
    
    /// 计算字段的众数
    fn calculate_mode(&self, batch: &DataBatch, field_name: &str) -> Option<DataValue> {
        // 使用序列化后的字符串作为键，因为 DataValue 没有实现 Hash
        let mut value_counts: HashMap<String, (DataValue, usize)> = HashMap::new();
        
        for record in batch.records() {
            if let Some(value) = record.get(field_name) {
                if !value.is_null() {
                    let key = serde_json::to_string(value)
                        .unwrap_or_else(|_| format!("{:?}", value));
                    let entry = value_counts.entry(key).or_insert_with(|| (value.clone(), 0));
                    entry.1 += 1;
                }
            }
        }
        
        if value_counts.is_empty() {
            return None;
        }
        
        let mut max_count = 0;
        let mut mode = None;
        
        for (_, (value, count)) in value_counts {
            if count > max_count {
                max_count = count;
                mode = Some(value);
            }
        }
        
        mode
    }
}

impl DataStage for MissingValueHandlerStage {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn process(&self, batch: &mut DataBatch) -> Result<()> {
        let schema = batch.schema()
            .ok_or_else(|| Error::validation("批次缺少schema".to_string()))?;
        
        // 计算替换值
        let mut cached_values = HashMap::new();
        
        for (field_name, strategy) in &self.field_strategies {
            match strategy {
                MissingValueStrategy::FillWithMean => {
                    if let Some(mean) = self.calculate_mean(batch, field_name) {
                        cached_values.insert(field_name.clone(), mean);
                    }
                },
                MissingValueStrategy::FillWithMedian => {
                    if let Some(median) = self.calculate_median(batch, field_name) {
                        cached_values.insert(field_name.clone(), median);
                    }
                },
                MissingValueStrategy::FillWithMode => {
                    if let Some(mode) = self.calculate_mode(batch, field_name) {
                        cached_values.insert(field_name.clone(), mode);
                    }
                },
                _ => {}
            }
        }
        
        // 处理缺失值
        let mut records_to_keep = Vec::with_capacity(batch.len());
        
        for (record_idx, record) in batch.records().iter().enumerate() {
            let mut should_drop = false;
            let mut new_record = record.clone();
            
            for (field_name, strategy) in &self.field_strategies {
                if let Some(value) = new_record.get(field_name) {
                    if value.is_null() {
                        match strategy {
                            MissingValueStrategy::DropRecord => {
                                should_drop = true;
                                break;
                            },
                            MissingValueStrategy::FillWithConstant(fill_value) => {
                                new_record.insert(field_name.clone(), fill_value.clone());
                            },
                            MissingValueStrategy::FillWithMean | 
                            MissingValueStrategy::FillWithMedian | 
                            MissingValueStrategy::FillWithMode => {
                                if let Some(fill_value) = cached_values.get(field_name) {
                                    new_record.insert(field_name.clone(), fill_value.clone());
                                }
                            }
                        }
                    }
                } else {
                    // 字段不存在，根据策略处理
                    match strategy {
                        MissingValueStrategy::DropRecord => {
                            should_drop = true;
                            break;
                        },
                        MissingValueStrategy::FillWithConstant(fill_value) => {
                            new_record.insert(field_name.clone(), fill_value.clone());
                        },
                        MissingValueStrategy::FillWithMean | 
                        MissingValueStrategy::FillWithMedian | 
                        MissingValueStrategy::FillWithMode => {
                            if let Some(fill_value) = cached_values.get(field_name) {
                                new_record.insert(field_name.clone(), fill_value.clone());
                            }
                        }
                    }
                }
            }
            
            if !should_drop {
                records_to_keep.push(new_record);
            }
        }
        
        // 更新批次中的记录
        batch.records = records_to_keep;
        
        Ok(())
    }
    
    fn output_schema(&self, input_schema: &DataSchema) -> Result<DataSchema> {
        // 缺失值处理不会改变模式,除非某些策略需要更新字段可空性
        Ok(input_schema.clone())
    }
}

/// 数据标准化阶段
pub struct NormalizerStage {
    /// 阶段名称
    name: String,
    
    /// 需要标准化的字段
    fields: HashSet<String>,
    
    /// 标准化方法
    method: NormalizationMethod,
}

/// 标准化方法
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizationMethod {
    /// 最小-最大缩放 (Min-Max Scaling)
    MinMax {
        min: f64,
        max: f64,
    },
    
    /// Z-score标准化
    ZScore,
    
    /// 绝对最大缩放
    MaxAbs,
}

impl NormalizerStage {
    /// 创建新的数据标准化阶段
    pub fn new<S: Into<String>>(name: S, method: NormalizationMethod) -> Self {
        Self {
            name: name.into(),
            fields: HashSet::new(),
            method,
        }
    }
    
    /// 添加需要标准化的字段
    pub fn add_field<S: Into<String>>(&mut self, field_name: S) -> &mut Self {
        self.fields.insert(field_name.into());
        self
    }
    
    /// 添加多个需要标准化的字段
    pub fn add_fields<S: Into<String>, I: IntoIterator<Item = S>>(&mut self, field_names: I) -> &mut Self {
        for name in field_names {
            self.fields.insert(name.into());
        }
        self
    }
    
    /// 计算数值统计信息
    fn calculate_stats(&self, batch: &DataBatch, field_name: &str) -> Result<NumericalStats> {
        let mut stats = NumericalStats {
            min: f64::MAX,
            max: f64::MIN,
            sum: 0.0,
            count: 0,
            mean: 0.0,
            variance: 0.0,
        };
        
        // 第一次遍历计算基本统计量
        for record in batch.records() {
            if let Some(value) = record.get(field_name) {
                match value {
                    DataValue::Integer(i) => {
                        let value = *i as f64;
                        stats.min = stats.min.min(value);
                        stats.max = stats.max.max(value);
                        stats.sum += value;
                        stats.count += 1;
                    },
                    DataValue::Float(f) => {
                        stats.min = stats.min.min(*f);
                        stats.max = stats.max.max(*f);
                        stats.sum += f;
                        stats.count += 1;
                    },
                    _ => {}
                }
            }
        }
        
        if stats.count == 0 {
            return Err(Error::invalid_data("无有效数值数据进行标准化"));
        }
        
        stats.mean = stats.sum / stats.count as f64;
        
        // 第二次遍历计算方差
        let mut sum_squared_diff = 0.0;
        
        for record in batch.records() {
            if let Some(value) = record.get(field_name) {
                match value {
                    DataValue::Integer(i) => {
                        let value = *i as f64;
                        let diff = value - stats.mean;
                        sum_squared_diff += diff * diff;
                    },
                    DataValue::Float(f) => {
                        let diff = f - stats.mean;
                        sum_squared_diff += diff * diff;
                    },
                    _ => {}
                }
            }
        }
        
        stats.variance = sum_squared_diff / stats.count as f64;
        
        Ok(stats)
    }
    
    /// 标准化数值
    fn normalize_value(&self, value: f64, stats: &NumericalStats) -> f64 {
        match self.method {
            NormalizationMethod::MinMax { min, max } => {
                if stats.max == stats.min {
                    return 0.5 * (max - min) + min; // 避免除以零
                }
                (value - stats.min) / (stats.max - stats.min) * (max - min) + min
            },
            NormalizationMethod::ZScore => {
                if stats.variance == 0.0 {
                    return 0.0; // 避免除以零
                }
                (value - stats.mean) / stats.variance.sqrt()
            },
            NormalizationMethod::MaxAbs => {
                if stats.max == 0.0 && stats.min == 0.0 {
                    return 0.0; // 避免除以零
                }
                let abs_max = stats.max.abs().max(stats.min.abs());
                value / abs_max
            }
        }
    }
}

/// 数值统计信息
struct NumericalStats {
    min: f64,
    max: f64,
    sum: f64,
    count: usize,
    mean: f64,
    variance: f64,
}

impl DataStage for NormalizerStage {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn process(&self, batch: &mut DataBatch) -> Result<()> {
        let schema = batch.schema()
            .ok_or_else(|| Error::validation("批次缺少schema".to_string()))?;
        
        // 计算每个字段的统计信息
        let mut field_stats = HashMap::new();
        
        for field_name in &self.fields {
            match self.calculate_stats(batch, field_name) {
                Ok(stats) => {
                    field_stats.insert(field_name.clone(), stats);
                },
                Err(e) => {
                    warn!("计算字段 '{}' 的统计信息失败: {}", field_name, e);
                    continue;
                }
            }
        }
        
        // 标准化数据
        for record in &mut batch.records {
            for field_name in &self.fields {
                if let Some(stats) = field_stats.get(field_name) {
                    if let Some(value) = record.get(field_name) {
                        match value {
                            DataValue::Integer(i) => {
                                let value = *i as f64;
                                let normalized = self.normalize_value(value, stats);
                                record.insert(field_name.clone(), DataValue::Float(normalized));
                            },
                            DataValue::Float(f) => {
                                let normalized = self.normalize_value(*f, stats);
                                record.insert(field_name.clone(), DataValue::Float(normalized));
                            },
                            _ => {} // 跳过非数值类型
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn output_schema(&self, input_schema: &DataSchema) -> Result<DataSchema> {
        let mut new_fields = input_schema.fields().clone();
        
        // 更新标准化字段的类型为浮点型（Numeric）
        for field in &mut new_fields {
            if self.fields.contains(field.name()) {
                if field.field_type() == &FieldType::Numeric {
                    // 已经是 Numeric 类型，保持不变
                } else {
                    // 如果是数值类型，转换为 Numeric
                    field.set_field_type(FieldType::Numeric);
                }
            }
        }
        
        let mut new_schema = input_schema.clone();
        new_schema.fields = new_fields;
        
        Ok(new_schema)
    }
}

/// 数据过滤阶段
pub struct FilterStage {
    /// 阶段名称
    name: String,
    
    /// 过滤条件
    condition: Arc<FilterCondition>,
    
    /// 阶段配置
    config: StageConfig,
}

/// 过滤条件
#[derive(Debug, Clone)]
pub enum FilterCondition {
    /// 相等条件
    Equal(String, Value),
    /// 不等条件
    NotEqual(String, Value),
    /// 大于条件
    GreaterThan(String, Value),
    /// 小于条件
    LessThan(String, Value),
    /// 包含条件
    Contains(String, String),
    /// 正则表达式匹配条件
    Regex(String, String),
    /// 复合条件 - 与
    And(Vec<FilterCondition>),
    /// 复合条件 - 或
    Or(Vec<FilterCondition>),
}

impl FilterCondition {
    /// 从字符串解析过滤条件
    pub fn from_str(s: &str) -> Result<Self> {
        // 简单实现，仅支持基本条件
        let parts: Vec<&str> = s.splitn(3, ' ').collect();
        if parts.len() < 3 {
            return Err(crate::Error::invalid_input(
                "过滤条件格式无效，应为 '字段 操作符 值'".to_string()
            ));
        }
        
        let field_name = parts[0].to_string();
        let operator = parts[1];
        let value_str = parts[2];
        
        // 根据操作符创建不同的条件
        match operator {
            "=" | "==" => {
                // 尝试解析值为JSON
                let value = serde_json::from_str(value_str)
                    .unwrap_or_else(|_| Value::String(value_str.to_string()));
                Ok(FilterCondition::Equal(field_name, value))
            },
            "!=" | "<>" => {
                let value = serde_json::from_str(value_str)
                    .unwrap_or_else(|_| Value::String(value_str.to_string()));
                Ok(FilterCondition::NotEqual(field_name, value))
            },
            ">" => {
                let value = serde_json::from_str(value_str)
                    .unwrap_or_else(|_| Value::String(value_str.to_string()));
                Ok(FilterCondition::GreaterThan(field_name, value))
            },
            "<" => {
                let value = serde_json::from_str(value_str)
                    .unwrap_or_else(|_| Value::String(value_str.to_string()));
                Ok(FilterCondition::LessThan(field_name, value))
            },
            "contains" => {
                Ok(FilterCondition::Contains(field_name, value_str.to_string()))
            },
            "regex" => {
                Ok(FilterCondition::Regex(field_name, value_str.to_string()))
            },
            _ => Err(crate::Error::invalid_input(format!(
                "不支持的操作符: {}", operator
            ))),
        }
    }
    
    /// 评估条件是否满足
    pub fn evaluate(&self, record: &DataRecord, schema: &DataSchema) -> bool {
        match self {
            FilterCondition::Equal(field_name, value) => {
                match record.get_field(field_name) {
                    Some(field_value) => {
                        // 转换为相同类型进行比较
                        let field_json = serde_json::to_value(field_value).unwrap_or(Value::Null);
                        field_json == *value
                    },
                    None => false,
                }
            },
            FilterCondition::NotEqual(field_name, value) => {
                match record.get_field(field_name) {
                    Some(field_value) => {
                        let field_json = serde_json::to_value(field_value).unwrap_or(Value::Null);
                        field_json != *value
                    },
                    None => true,
                }
            },
            FilterCondition::GreaterThan(field_name, value) => {
                match record.get_field(field_name) {
                    Some(field_value) => {
                        // serde_json::Value 不能直接比较，需要实现比较逻辑
                        Self::compare_values(field_value, value) == Some(std::cmp::Ordering::Greater)
                    },
                    None => false,
                }
            },
            FilterCondition::LessThan(field_name, value) => {
                match record.get_field(field_name) {
                    Some(field_value) => {
                        // serde_json::Value 不能直接比较，需要实现比较逻辑
                        Self::compare_values(field_value, value) == Some(std::cmp::Ordering::Less)
                    },
                    None => false,
                }
            },
            FilterCondition::Contains(field_name, substring) => {
                match record.get_field(field_name) {
                    Some(field_value) => {
                        let field_str = format!("{}", field_value);
                        field_str.contains(substring)
                    },
                    None => false,
                }
            },
            FilterCondition::Regex(field_name, pattern) => {
                match record.get_field(field_name) {
                    Some(field_value) => {
                        let field_str = format!("{}", field_value);
                        match regex::Regex::new(pattern) {
                            Ok(re) => re.is_match(&field_str),
                            Err(_) => false,
                        }
                    },
                    None => false,
                }
            },
            FilterCondition::And(conditions) => {
                conditions.iter().all(|c| c.evaluate(record, schema))
            },
            FilterCondition::Or(conditions) => {
                conditions.iter().any(|c| c.evaluate(record, schema))
            },
        }
    }
    
    /// 比较两个值（用于大于/小于比较）
    fn compare_values(field_value: &crate::data::record::Value, json_value: &serde_json::Value) -> Option<std::cmp::Ordering> {
        // 将 field_value 转换为 serde_json::Value 进行比较
        let field_json = serde_json::to_value(field_value).ok()?;
        
        // 尝试提取数值进行比较
        match (&field_json, json_value) {
            (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
                let a_f64 = a.as_f64()?;
                let b_f64 = b.as_f64()?;
                a_f64.partial_cmp(&b_f64)
            },
            (serde_json::Value::String(a), serde_json::Value::String(b)) => {
                Some(a.cmp(b))
            },
            _ => None,
        }
    }
}

// 为FilterCondition实现FromStr特性
impl FromStr for FilterCondition {
    type Err = crate::Error;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        FilterCondition::from_str(s)
    }
}

impl FilterStage {
    /// 创建新的过滤阶段
    pub fn new(name: &str, condition: FilterCondition, config: StageConfig) -> Self {
        debug!("创建过滤阶段: {}", name);
        Self {
            name: name.to_string(),
            condition: Arc::new(condition),
            config,
        }
    }
    
    /// 克隆当前过滤条件以便共享
    pub fn clone_condition(&self) -> Arc<FilterCondition> {
        Arc::clone(&self.condition)
    }
}

impl PipelineStage for FilterStage {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn config(&self) -> &StageConfig {
        &self.config
    }
    
    fn execute(&self, batch: DataBatch, context: &mut StageExecutionContext) -> Result<DataBatch> {
        info!("执行过滤阶段: {}", self.name);
        
        let schema = batch.schema()
            .ok_or_else(|| Error::validation("批次缺少schema".to_string()))?;
        let records = batch.records();
        
        // 将 HashMap<String, DataValue> 转换为 Record，然后过滤
        let filtered_records: Vec<HashMap<String, DataValue>> = records
            .iter()
            .filter_map(|record_map| {
                // 转换为 Record
                let mut record = DataRecord::new();
                for (field_name, data_value) in record_map {
                    record.add_field(field_name, crate::data::record::Value::Data(data_value.clone()));
                }
                
                // 评估条件
                if self.condition.evaluate(&record, schema) {
                    Some(record_map.clone())
                } else {
                    None
                }
            })
            .collect();
        
        info!("过滤后记录数: {} -> {}", records.len(), filtered_records.len());
        
        // 更新上下文统计信息
        context.add_statistic(
            &format!("{}_input_records", self.name), 
            records.len() as i64
        );
        context.add_statistic(
            &format!("{}_output_records", self.name), 
            filtered_records.len() as i64
        );
        
        // 创建新的数据批次
        let result = DataBatch::from_records(filtered_records, Some(schema.clone()))?;
        Ok(result)
    }
}

/// 数据转换阶段
pub struct TransformStage {
    /// 阶段名称
    name: String,
    
    /// 字段转换器
    transformers: HashMap<String, Arc<FieldTransformer>>,
    
    /// 阶段配置
    config: StageConfig,
}

/// 字段转换器
#[derive(Debug, Clone)]
pub enum FieldTransformer {
    /// 替换字符串
    Replace(String, String),
    /// 大写转换
    ToUpper,
    /// 小写转换
    ToLower,
    /// 数值乘法
    Multiply(f64),
    /// 数值加法
    Add(f64),
    /// 自定义脚本
    Script(String),
}

impl TransformStage {
    /// 创建新的转换阶段
    pub fn new(name: &str, config: StageConfig) -> Self {
        debug!("创建转换阶段: {}", name);
        Self {
            name: name.to_string(),
            transformers: HashMap::new(),
            config,
        }
    }
    
    /// 添加字段转换器
    pub fn add_transformer(&mut self, field_name: &str, transformer: FieldTransformer) -> &mut Self {
        self.transformers.insert(field_name.to_string(), Arc::new(transformer));
        self
    }
}

impl PipelineStage for TransformStage {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn config(&self) -> &StageConfig {
        &self.config
    }
    
    fn execute(&self, batch: DataBatch, context: &mut StageExecutionContext) -> Result<DataBatch> {
        info!("执行转换阶段: {}", self.name);
        
        let schema = batch.schema()
            .ok_or_else(|| Error::validation("批次缺少schema".to_string()))?;
        let records = batch.records();
        let mut transformed_records = Vec::with_capacity(records.len());
        
        // 转换每条记录
        for record in records {
            let mut transformed_record = record.clone();
            
            // 应用每个字段的转换
            for (field_name, transformer) in &self.transformers {
                // 使用字段名直接访问 HashMap
                if let Some(value) = transformed_record.get_mut(field_name) {
                    // 根据转换器类型应用不同的转换
                    match &**transformer {
                        FieldTransformer::Replace(from, to) => {
                            // 字符串替换
                            let string_value = format!("{}", value);
                            let new_value = string_value.replace(from, to);
                            *value = DataValue::String(new_value);
                        },
                        FieldTransformer::ToUpper => {
                            // 转换为大写
                            let string_value = format!("{}", value);
                            let upper_value = string_value.to_uppercase();
                            *value = DataValue::String(upper_value);
                        },
                        FieldTransformer::ToLower => {
                            // 转换为小写
                            let string_value = format!("{}", value);
                            let lower_value = string_value.to_lowercase();
                            *value = DataValue::String(lower_value);
                        },
                        FieldTransformer::Multiply(factor) => {
                            // 数值乘法
                            if let Some(num) = value.as_number() {
                                let new_value = num * factor;
                                *value = DataValue::Float(new_value);
                            }
                        },
                        FieldTransformer::Add(addend) => {
                            // 数值加法
                            if let Some(num) = value.as_number() {
                                let new_value = num + addend;
                                *value = DataValue::Float(new_value);
                            }
                        },
                        FieldTransformer::Script(script) => {
                            // 生产级脚本执行：使用 rhai 作为安全的嵌入式脚本引擎
                            debug!("执行转换脚本: {}", script);
                            #[cfg(feature = "rhai")]
                            {
                                use rhai::{Engine, Scope, Dynamic};
                                let engine = Engine::new();
                                let mut scope = Scope::new();

                                // 将当前值注入脚本环境（若无则为 Null）
                                let current_val = if let Some(v) = value.as_number() {
                                    Dynamic::from_float(v)
                                } else if let Some(i) = value.as_integer() {
                                    Dynamic::from_int(i)
                                } else if let Some(s) = value.as_str() {
                                    Dynamic::from(s.to_string())
                                } else {
                                    Dynamic::UNIT
                                };
                                scope.push_dynamic("value", current_val);

                                // 执行脚本，期望返回新值
                                match engine.eval_with_scope::<Dynamic>(&mut scope, script) {
                                    Ok(result) => {
                                        // 将结果动态映射回 DataValue
                                        if result.is_float() {
                                            let f = result.as_float().unwrap();
                                            *value = DataValue::Float(f);
                                        } else if result.is_int() {
                                            let i = result.as_int().unwrap();
                                            *value = DataValue::Integer(i);
                                        } else if result.is_string() {
                                            let s = result.into_string().unwrap_or_default();
                                            *value = DataValue::String(s);
                                        } else {
                                            warn!("脚本返回了不支持的类型，保持原值");
                                        }
                                    }
                                    Err(e) => {
                                        warn!("脚本执行失败: {}，保持原值", e);
                                    }
                                }
                            }
                            #[cfg(not(feature = "rhai"))]
                            {
                                warn!("脚本功能需要启用 'rhai' feature，跳过脚本转换");
                            }
                        },
                    }
                }
            }
            
            transformed_records.push(transformed_record);
        }
        
        // 更新上下文统计信息
        context.add_statistic(
            &format!("{}_transformed_fields", self.name), 
            self.transformers.len() as i64
        );
        
        // 创建新的数据批次
        let result = DataBatch::from_records(transformed_records, Some(schema.clone()))?;
        Ok(result)
    }
} 