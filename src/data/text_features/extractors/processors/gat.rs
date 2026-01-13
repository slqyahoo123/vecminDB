use ndarray::{Array1, Array2, Array3};
use crate::data::text_features::extractors::{
    types::{ContextItem, ContextHistory},
    config::ContextAwareConfig,
    processors::ContextProcessor,
    utils::{activation::Activation, normalization::Normalization},
};

/// GAT处理器
pub struct GATProcessor {
    /// 配置
    config: ContextAwareConfig,
    /// 输入权重矩阵
    input_weights: Array2<f32>,
    /// 注意力权重矩阵
    attention_weights: Array2<f32>,
    /// 输出权重矩阵
    output_weights: Array2<f32>,
    /// 偏置向量
    bias: Array1<f32>,
    /// 激活函数
    activation: Activation,
    /// 归一化层
    normalization: Normalization,
}

impl GATProcessor {
    /// 创建新的GAT处理器
    pub fn new(config: ContextAwareConfig) -> Self {
        let input_dim = config.input_dim;
        let hidden_dim = config.hidden_dim;
        
        let input_weights = Array2::zeros((input_dim, hidden_dim));
        let attention_weights = Array2::zeros((2 * hidden_dim, 1));
        let output_weights = Array2::zeros((hidden_dim, hidden_dim));
        let bias = Array1::zeros(hidden_dim);
        let activation = Activation::default();
        let normalization = Normalization::default();
        
        Self {
            config,
            input_weights,
            attention_weights,
            output_weights,
            bias,
            activation,
            normalization,
        }
    }

    /// 计算注意力分数
    fn compute_attention(&self, x: &Array2<f32>) -> Array2<f32> {
        let n_nodes = x.nrows();
        let mut scores = Array2::zeros((n_nodes, n_nodes));
        
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                let hi = x.slice(ndarray::s![i, ..]);
                let hj = x.slice(ndarray::s![j, ..]);
                let concat = Array1::concatenate(ndarray::Axis(0), &[hi.view(), hj.view()]).unwrap();
                scores[[i, j]] = concat.dot(&self.attention_weights)[[0]];
            }
        }
        
        scores
    }

    /// 前向传播
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // 输入变换
        let mut h = x.dot(&self.input_weights);
        
        // 计算注意力分数
        let attention_scores = self.compute_attention(&h);
        let attention_probs = self.activation.apply(&attention_scores);
        
        // 注意力聚合
        let mut output = Array2::zeros((x.nrows(), self.config.hidden_dim));
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                output.slice_mut(ndarray::s![i, ..]) += 
                    &(attention_probs[[i, j]] * h.slice(ndarray::s![j, ..]));
            }
        }
        
        // 输出变换
        output = output.dot(&self.output_weights) + &self.bias;
        output = self.activation.apply(&output);
        
        // 归一化
        self.normalization.apply(&output)
    }
}

impl ContextProcessor for GATProcessor {
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