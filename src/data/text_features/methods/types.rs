use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// 特征类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureType {
    /// 数值型特征
    Numeric,
    /// 类别型特征
    Categorical,
    /// 文本特征
    Text,
    /// 图像特征
    Image,
    /// 音频特征
    Audio,
    /// 视频特征
    Video,
    /// 时间序列特征
    TimeSeries,
    /// 图结构特征
    Graph,
    /// 混合特征
    Mixed,
    /// 未知特征
    Unknown,
}

/// 特征归一化方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// 最小最大归一化
    MinMax,
    /// Z-Score标准化
    ZScore,
    /// 绝对最大缩放
    MaxAbs,
    /// 鲁棒缩放
    Robust,
    /// L1正则化
    L1,
    /// L2正则化
    L2,
    /// 不归一化
    None,
}

/// 特征信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInfo {
    /// 特征名称
    pub name: String,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 特征维度
    pub dimension: usize,
    /// 归一化方法
    pub normalization: NormalizationMethod,
    /// 是否稀疏
    pub is_sparse: bool,
    /// 额外元数据
    pub metadata: HashMap<String, String>,
}

impl FeatureInfo {
    /// 创建新的特征信息
    pub fn new(name: &str, feature_type: FeatureType, dimension: usize) -> Self {
        FeatureInfo {
            name: name.to_string(),
            feature_type,
            dimension,
            normalization: NormalizationMethod::None,
            is_sparse: false,
            metadata: HashMap::new(),
        }
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
    
    /// 设置归一化方法
    pub fn set_normalization(&mut self, method: NormalizationMethod) {
        self.normalization = method;
    }
    
    /// 设置是否稀疏
    pub fn set_sparse(&mut self, sparse: bool) {
        self.is_sparse = sparse;
    }
} 