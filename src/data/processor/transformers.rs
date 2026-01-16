// Data Transformers
// 数据转换器实现

use std::fmt::Debug;
use std::collections::HashMap;
use chrono::{NaiveDate, NaiveDateTime};
use crate::Error;
use crate::Result;

/// 数据转换器特质
pub trait DataTransformer: Debug + Clone {
    fn transform(&self, value: &str) -> Result<f32>;
    fn inverse_transform(&self, value: f32) -> Result<String>;
}

/// 数值转换器
#[derive(Debug, Clone)]
pub struct NumericTransformer {
    pub scale: f32,
    pub offset: f32,
}

impl NumericTransformer {
    /// 创建新的数值转换器
    pub fn new(scale: f32, offset: f32) -> Self {
        Self { scale, offset }
    }
}

impl DataTransformer for NumericTransformer {
    fn transform(&self, value: &str) -> Result<f32> {
        let num = value.parse::<f32>()
            .map_err(|e| Error::invalid_data(format!("Failed to parse numeric value '{}': {}", value, e)))?;
        Ok(num * self.scale + self.offset)
    }

    fn inverse_transform(&self, value: f32) -> Result<String> {
        let original = (value - self.offset) / self.scale;
        Ok(original.to_string())
    }
}

/// 类别编码器
pub struct CategoryEncoder {
    categories: HashMap<String, usize>,
    inverse_categories: HashMap<usize, String>,
    next_id: usize,
}

impl CategoryEncoder {
    pub fn new() -> Self {
        Self {
            categories: HashMap::new(),
            inverse_categories: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn encode(&mut self, value: &str) -> usize {
        if let Some(&id) = self.categories.get(value) {
            id
        } else {
            let id = self.next_id;
            self.categories.insert(value.to_string(), id);
            self.inverse_categories.insert(id, value.to_string());
            self.next_id += 1;
            id
        }
    }

    pub fn decode(&self, id: usize) -> Option<&str> {
        self.inverse_categories.get(&id).map(|s| s.as_str())
    }
}

/// 类别转换器
#[derive(Debug, Clone)]
pub struct CategoricalTransformer {
    categories: HashMap<String, usize>,
    inverse_categories: HashMap<usize, String>,
}

impl CategoricalTransformer {
    pub fn new(categories: Vec<String>) -> Self {
        let mut cat_map = HashMap::new();
        let mut inverse_map = HashMap::new();
        for (i, cat) in categories.into_iter().enumerate() {
            cat_map.insert(cat.clone(), i);
            inverse_map.insert(i, cat);
        }
        Self {
            categories: cat_map,
            inverse_categories: inverse_map,
        }
    }
}

impl DataTransformer for CategoricalTransformer {
    fn transform(&self, value: &str) -> Result<f32> {
        self.categories.get(value)
            .map(|&i| i as f32)
            .ok_or_else(|| Error::invalid_data(format!("Unknown category: {}", value)))
    }

    fn inverse_transform(&self, value: f32) -> Result<String> {
        let index = value.round() as usize;
        self.inverse_categories.get(&index)
            .cloned()
            .ok_or_else(|| Error::invalid_data(format!("Invalid category index: {}", value)))
    }
}

/// 日期时间转换器
#[derive(Debug, Clone)]
pub struct DateTimeTransformer {
    pub format: String,
    pub reference_date: NaiveDateTime,
}

impl DateTimeTransformer {
    /// 创建新的日期时间转换器
    pub fn new(format: &str) -> Self {
        Self {
            format: format.to_string(),
            reference_date: NaiveDate::from_ymd_opt(1970, 1, 1)
                .and_then(|d| d.and_hms_opt(0, 0, 0))
                .unwrap_or_else(|| {
                    // 使用 DateTime::from_timestamp 然后转换为 NaiveDateTime
                    chrono::DateTime::from_timestamp(0, 0)
                        .map(|dt| dt.naive_utc())
                        .unwrap_or_else(|| NaiveDateTime::MIN)
                }),
        }
    }
}

impl DataTransformer for DateTimeTransformer {
    fn transform(&self, value: &str) -> Result<f32> {
        let date = NaiveDateTime::parse_from_str(value, &self.format)
            .map_err(|e| Error::invalid_data(format!("Failed to parse date '{}': {}", value, e)))?;
        let duration = date.signed_duration_since(self.reference_date);
        Ok(duration.num_seconds() as f32)
    }

    fn inverse_transform(&self, value: f32) -> Result<String> {
        let seconds = value as i64;
        let date = self.reference_date + chrono::Duration::seconds(seconds);
        Ok(date.format(&self.format).to_string())
    }
}

/// 转换器类型枚举
#[derive(Debug, Clone)]
pub enum TransformerType {
    Numeric(NumericTransformer),
    DateTime(DateTimeTransformer),
    Categorical(CategoricalTransformer),
}

impl TransformerType {
    pub fn transform(&self, value: &str) -> Result<f32> {
        match self {
            TransformerType::Numeric(t) => t.transform(value),
            TransformerType::DateTime(t) => t.transform(value),
            TransformerType::Categorical(t) => t.transform(value),
        }
    }

    pub fn inverse_transform(&self, value: f32) -> Result<String> {
        match self {
            TransformerType::Numeric(t) => t.inverse_transform(value),
            TransformerType::DateTime(t) => t.inverse_transform(value),
            TransformerType::Categorical(t) => t.inverse_transform(value),
        }
    }
} 