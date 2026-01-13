use ndarray::{Array1, Array2, Array3};
use crate::data::text_features::extractors::config::BatchNormParams;

/// 归一化类型
#[derive(Debug, Clone, Copy)]
pub enum NormalizationType {
    /// 层归一化
    LayerNorm,
    /// 批归一化
    BatchNorm,
    /// 实例归一化
    InstanceNorm,
}

/// 归一化实现
pub struct Normalization {
    /// 归一化类型
    norm_type: NormalizationType,
    /// 参数
    params: BatchNormParams,
}

impl Normalization {
    /// 创建新的归一化器
    pub fn new(norm_type: NormalizationType, params: BatchNormParams) -> Self {
        Self {
            norm_type,
            params,
        }
    }

    /// 应用归一化
    pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
        match self.norm_type {
            NormalizationType::LayerNorm => self.layer_norm(x),
            NormalizationType::BatchNorm => self.batch_norm(x),
            NormalizationType::InstanceNorm => self.instance_norm(x),
        }
    }

    /// 层归一化
    fn layer_norm(&self, x: &Array2<f32>) -> Array2<f32> {
        let eps = self.params.eps;
        let mut result = x.clone();
        
        for mut row in result.outer_iter_mut() {
            let mean = row.mean().unwrap();
            let var = row.var().unwrap();
            let std = (var + eps).sqrt();
            
            row -= mean;
            row /= std;
        }
        
        result
    }

    /// 批归一化
    fn batch_norm(&self, x: &Array2<f32>) -> Array2<f32> {
        let eps = self.params.eps;
        let momentum = self.params.momentum;
        let mut result = x.clone();
        
        for mut col in result.outer_iter_mut() {
            let mean = col.mean().unwrap();
            let var = col.var().unwrap();
            let std = (var + eps).sqrt();
            
            col -= mean;
            col /= std;
        }
        
        result
    }

    /// 实例归一化
    fn instance_norm(&self, x: &Array2<f32>) -> Array2<f32> {
        let eps = self.params.eps;
        let mut result = x.clone();
        
        for mut row in result.outer_iter_mut() {
            let mean = row.mean().unwrap();
            let var = row.var().unwrap();
            let std = (var + eps).sqrt();
            
            row -= mean;
            row /= std;
        }
        
        result
    }
}

impl Default for Normalization {
    fn default() -> Self {
        Self::new(NormalizationType::LayerNorm, BatchNormParams::default())
    }
} 