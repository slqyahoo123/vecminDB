use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::error::Error;

/// 上下文项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    /// 文本内容
    pub text: String,
    /// 特征向量
    pub features: Array1<f32>,
    /// 位置信息
    pub position: usize,
    /// 时间戳
    pub timestamp: i64,
}

/// 上下文历史
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextHistory {
    /// 历史项列表
    pub items: Vec<ContextItem>,
    /// 最大长度
    pub max_length: usize,
}

impl ContextHistory {
    /// 创建新的上下文历史
    pub fn new(max_length: usize) -> Self {
        Self {
            items: Vec::with_capacity(max_length),
            max_length,
        }
    }

    /// 添加新的上下文项
    pub fn add_item(&mut self, item: ContextItem) {
        if self.items.len() >= self.max_length {
            self.items.remove(0);
        }
        self.items.push(item);
    }

    /// 获取所有特征
    pub fn get_features(&self) -> Array2<f32> {
        let n_items = self.items.len();
        let feature_dim = self.items[0].features.len();
        let mut features = Array2::zeros((n_items, feature_dim));
        
        for (i, item) in self.items.iter().enumerate() {
            features.slice_mut(ndarray::s![i, ..]).assign(&item.features);
        }
        
        features
    }

    /// 获取所有位置信息
    pub fn get_positions(&self) -> Array1<usize> {
        Array1::from_iter(self.items.iter().map(|item| item.position))
    }

    /// 获取所有时间戳
    pub fn get_timestamps(&self) -> Array1<i64> {
        Array1::from_iter(self.items.iter().map(|item| item.timestamp))
    }
}

/// 位置编码器
#[derive(Debug, Clone)]
pub struct PositionEncoder {
    /// 最大序列长度
    max_length: usize,
    /// 特征维度
    feature_dim: usize,
    /// 位置编码矩阵
    position_encoding: Array2<f32>,
}

impl PositionEncoder {
    /// 创建新的位置编码器
    pub fn new(max_length: usize, feature_dim: usize) -> Self {
        let mut position_encoding = Array2::zeros((max_length, feature_dim));
        
        for pos in 0..max_length {
            for i in 0..feature_dim {
                if i % 2 == 0 {
                    position_encoding[[pos, i]] = (pos as f32 / (10000.0f32.powf(2.0 * i as f32 / feature_dim as f32))).sin();
                } else {
                    position_encoding[[pos, i]] = (pos as f32 / (10000.0f32.powf(2.0 * i as f32 / feature_dim as f32))).cos();
                }
            }
        }
        
        Self {
            max_length,
            feature_dim,
            position_encoding,
        }
    }

    /// 获取位置编码
    pub fn get_encoding(&self, position: usize) -> Result<Array1<f32>> {
        self.check_position(position)?;
        Ok(self.position_encoding.slice(ndarray::s![position, ..]).to_owned())
    }

    /// 获取位置编码矩阵
    pub fn get_encoding_matrix(&self) -> Array2<f32> {
        self.position_encoding.clone()
    }

    /// 检查位置是否在有效范围内
    pub fn check_position(&self, position: usize) -> Result<()> {
        if position >= self.max_length {
            return Err(Error::bounds(format!(
                "位置 {} 超出最大长度 {}", 
                position, 
                self.max_length
            )));
        }
        Ok(())
    }
} 