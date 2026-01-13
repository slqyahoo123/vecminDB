use ndarray::{Array1, Array2, Array3};
use crate::data::text_features::extractors::{
    types::{ContextItem, ContextHistory},
    config::ContextAwareConfig,
    processors::ContextProcessor,
    utils::{activation::Activation, normalization::Normalization},
};

/// LSTM处理器
pub struct LSTMProcessor {
    /// 配置
    config: ContextAwareConfig,
    /// 输入权重矩阵
    input_weights: Array2<f32>,
    /// 隐藏状态权重矩阵
    hidden_weights: Array2<f32>,
    /// 偏置向量
    bias: Array1<f32>,
    /// 激活函数
    activation: Activation,
    /// 归一化层
    normalization: Normalization,
}

impl LSTMProcessor {
    /// 创建新的LSTM处理器
    pub fn new(config: ContextAwareConfig) -> Self {
        let input_dim = config.input_dim;
        let hidden_dim = config.hidden_dim;
        
        let input_weights = Array2::zeros((input_dim, 4 * hidden_dim));
        let hidden_weights = Array2::zeros((hidden_dim, 4 * hidden_dim));
        let bias = Array1::zeros(4 * hidden_dim);
        let activation = Activation::default();
        let normalization = Normalization::default();
        
        Self {
            config,
            input_weights,
            hidden_weights,
            bias,
            activation,
            normalization,
        }
    }

    /// 前向传播
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut h = Array2::zeros((x.nrows(), self.config.hidden_dim));
        let mut c = Array2::zeros((x.nrows(), self.config.hidden_dim));
        
        for t in 0..x.nrows() {
            let xt = x.slice(ndarray::s![t, ..]);
            let ht_prev = if t > 0 { h.slice(ndarray::s![t-1, ..]) } else { Array1::zeros(self.config.hidden_dim) };
            let ct_prev = if t > 0 { c.slice(ndarray::s![t-1, ..]) } else { Array1::zeros(self.config.hidden_dim) };
            
            let gates = xt.dot(&self.input_weights) + ht_prev.dot(&self.hidden_weights) + &self.bias;
            let gates = gates.into_shape((self.config.hidden_dim, 4)).unwrap();
            
            let i = self.activation.apply(&gates.slice(ndarray::s![.., 0]));
            let f = self.activation.apply(&gates.slice(ndarray::s![.., 1]));
            let o = self.activation.apply(&gates.slice(ndarray::s![.., 2]));
            let g = self.activation.apply(&gates.slice(ndarray::s![.., 3]));
            
            let ct = f * &ct_prev + i * g;
            let ht = o * self.activation.apply(&ct);
            
            h.slice_mut(ndarray::s![t, ..]).assign(&ht);
            c.slice_mut(ndarray::s![t, ..]).assign(&ct);
        }
        
        self.normalization.apply(&h)
    }
}

impl ContextProcessor for LSTMProcessor {
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