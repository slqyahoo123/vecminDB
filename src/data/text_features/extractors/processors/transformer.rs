use ndarray::{Array1, Array2, Array3};
use crate::data::text_features::extractors::{
    types::{ContextItem, ContextHistory, PositionEncoder},
    config::ContextAwareConfig,
    processors::ContextProcessor,
    utils::{activation::Activation, normalization::Normalization},
};

/// Transformer处理器
pub struct TransformerProcessor {
    /// 配置
    config: ContextAwareConfig,
    /// 查询权重矩阵
    query_weights: Array2<f32>,
    /// 键权重矩阵
    key_weights: Array2<f32>,
    /// 值权重矩阵
    value_weights: Array2<f32>,
    /// 输出权重矩阵
    output_weights: Array2<f32>,
    /// 前馈网络权重矩阵1
    ffn_weights1: Array2<f32>,
    /// 前馈网络权重矩阵2
    ffn_weights2: Array2<f32>,
    /// 偏置向量
    bias: Array1<f32>,
    /// 激活函数
    activation: Activation,
    /// 归一化层1
    norm1: Normalization,
    /// 归一化层2
    norm2: Normalization,
    /// 位置编码器
    position_encoder: PositionEncoder,
}

impl TransformerProcessor {
    /// 创建新的Transformer处理器
    pub fn new(config: ContextAwareConfig) -> Self {
        let input_dim = config.input_dim;
        let hidden_dim = config.hidden_dim;
        
        let query_weights = Array2::zeros((input_dim, hidden_dim));
        let key_weights = Array2::zeros((input_dim, hidden_dim));
        let value_weights = Array2::zeros((input_dim, hidden_dim));
        let output_weights = Array2::zeros((hidden_dim, hidden_dim));
        let ffn_weights1 = Array2::zeros((hidden_dim, 4 * hidden_dim));
        let ffn_weights2 = Array2::zeros((4 * hidden_dim, hidden_dim));
        let bias = Array1::zeros(hidden_dim);
        let activation = Activation::default();
        let norm1 = Normalization::default();
        let norm2 = Normalization::default();
        let position_encoder = PositionEncoder::new(config.context_window, hidden_dim);
        
        Self {
            config,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            ffn_weights1,
            ffn_weights2,
            bias,
            activation,
            norm1,
            norm2,
            position_encoder,
        }
    }

    /// 多头注意力
    fn multi_head_attention(&self, x: &Array2<f32>) -> Array2<f32> {
        let n_heads = 8;
        let head_dim = self.config.hidden_dim / n_heads;
        
        let q = x.dot(&self.query_weights);
        let k = x.dot(&self.key_weights);
        let v = x.dot(&self.value_weights);
        
        let q = q.into_shape((x.nrows(), n_heads, head_dim)).unwrap();
        let k = k.into_shape((x.nrows(), n_heads, head_dim)).unwrap();
        let v = v.into_shape((x.nrows(), n_heads, head_dim)).unwrap();
        
        let mut scores = Array3::zeros((x.nrows(), n_heads, x.nrows()));
        for i in 0..x.nrows() {
            for j in 0..x.nrows() {
                for h in 0..n_heads {
                    scores[[i, h, j]] = q.slice(ndarray::s![i, h, ..])
                        .dot(&k.slice(ndarray::s![j, h, ..]));
                }
            }
        }
        
        scores = scores / (head_dim as f32).sqrt();
        let attn = self.activation.apply(&scores);
        
        let mut output = Array3::zeros((x.nrows(), n_heads, head_dim));
        for i in 0..x.nrows() {
            for h in 0..n_heads {
                for j in 0..x.nrows() {
                    output.slice_mut(ndarray::s![i, h, ..]) += 
                        &(attn[[i, h, j]] * v.slice(ndarray::s![j, h, ..]));
                }
            }
        }
        
        output.into_shape((x.nrows(), self.config.hidden_dim)).unwrap()
    }

    /// 前馈网络
    fn feed_forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut h = x.dot(&self.ffn_weights1);
        h = self.activation.apply(&h);
        h.dot(&self.ffn_weights2)
    }

    /// 前向传播
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut h = x.clone();
        
        // 添加位置编码
        if self.config.use_position_encoding {
            let pos_encoding = self.position_encoder.get_encoding_matrix();
            h += &pos_encoding.slice(ndarray::s![..x.nrows(), ..]);
        }
        
        // 多头注意力
        let attn_output = self.multi_head_attention(&h);
        h = self.norm1.apply(&(h + attn_output));
        
        // 前馈网络
        let ffn_output = self.feed_forward(&h);
        h = self.norm2.apply(&(h + ffn_output));
        
        h
    }
}

impl ContextProcessor for TransformerProcessor {
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