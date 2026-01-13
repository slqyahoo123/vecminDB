// Feature Extractor Models
// 特征提取器的数据模型定义

use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use crate::data::schema::FieldType;
use crate::data::text_features::FeatureType;

/// 特征重要性
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// 特征名称
    pub name: String,
    /// 重要性分数
    pub importance: f32,
    /// 特征类型
    pub feature_type: FeatureType,
}

/// 数据特征分析
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataCharacteristics {
    pub field_types: HashMap<String, FieldType>,
    pub stats: DataStats,
    pub detected_languages: HashMap<String, String>,
    pub categorical_values: HashMap<String, HashSet<String>>,
}

/// 数据统计信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataStats {
    pub text_stats: HashMap<String, TextFieldStats>,
    pub numeric_stats: HashMap<String, NumericFieldStats>,
    pub categorical_stats: HashMap<String, CategoricalFieldStats>,
}

/// 文本字段统计
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextFieldStats {
    pub avg_length: f32,
    pub max_length: usize,
    pub min_length: usize,
    pub unique_tokens: usize,
    pub language_confidence: f32,
}

/// 数值字段统计
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericFieldStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
}

/// 分类字段统计
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CategoricalFieldStats {
    pub unique_values: usize,
    pub most_common: Vec<(String, usize)>,
    pub entropy: f32,
}

/// 数值统计
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
}

/// 评估权重
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationWeights {
    pub accuracy: f32,
    pub speed: f32,
    pub memory: f32,
} 