use ndarray::{Array1, Array2, Array3};
use crate::data::text_features::extractors::{
    types::{ContextItem, ContextHistory},
    config::ContextAwareConfig,
    processors::ContextProcessor,
    utils::{activation::Activation, normalization::Normalization},
};

/// RNN处理器
pub struct RNNProcessor {
    /// 配置
    config: ContextAwareConfig,
    /// 权重矩阵
    weights: Array2<f32>,
    /// 偏置向量
    bias: Array1<f32>,
    /// 激活函数
    activation: Activation,
    /// 归一化层
    normalization: Normalization,
}

impl RNNProcessor {
    /// 创建新的RNN处理器
    pub fn new(config: ContextAwareConfig) -> Self {
        let input_dim = config.input_dim;
        let hidden_dim = config.hidden_dim;
        
        let weights = Array2::zeros((input_dim, hidden_dim));
        let bias = Array1::zeros(hidden_dim);
        let activation = Activation::default();
        let normalization = Normalization::default();
        
        Self {
            config,
            weights,
            bias,
            activation,
            normalization,
        }
    }

    /// 前向传播
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut h = Array2::zeros((x.nrows(), self.config.hidden_dim));
        
        for t in 0..x.nrows() {
            let xt = x.slice(ndarray::s![t, ..]);
            let mut ht = xt.dot(&self.weights) + &self.bias;
            ht = self.activation.apply(&ht);
            h.slice_mut(ndarray::s![t, ..]).assign(&ht);
        }
        
        self.normalization.apply(&h)
    }
}

impl ContextProcessor for RNNProcessor {
    fn process(&self, context: &ContextHistory) -> Array1<f32> {
        let features = context.get_features();
        let output = self.forward(&features);
        output.slice(ndarray::s![output.nrows()-1, ..]).to_owned()
    }
    
    fn update_config(&mut self, config: &ContextAwareConfig) {
        self.config = config.clone();
    }
    
    fn get_config(&self) -> &ContextAwareConfig {
        &self.config
    }
} 