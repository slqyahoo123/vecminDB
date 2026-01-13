use ndarray::{Array1, Array2, Array3};
use crate::data::text_features::extractors::{
    types::{ContextItem, ContextHistory},
    config::ContextAwareConfig,
    processors::{ContextProcessor, RNNProcessor, LSTMProcessor, GRUProcessor, TransformerProcessor},
    utils::{activation::Activation, normalization::Normalization},
};

/// 层次化处理器
pub struct HierarchicalProcessor {
    /// 配置
    config: ContextAwareConfig,
    /// 局部处理器
    local_processor: Box<dyn ContextProcessor>,
    /// 全局处理器
    global_processor: Box<dyn ContextProcessor>,
    /// 融合权重矩阵
    fusion_weights: Array2<f32>,
    /// 融合偏置向量
    fusion_bias: Array1<f32>,
    /// 激活函数
    activation: Activation,
    /// 归一化层
    normalization: Normalization,
}

impl HierarchicalProcessor {
    /// 创建新的层次化处理器
    pub fn new(config: ContextAwareConfig) -> Self {
        let hidden_dim = config.hidden_dim;
        
        // 创建局部处理器
        let local_processor: Box<dyn ContextProcessor> = match config.processor_type.as_str() {
            "rnn" => Box::new(RNNProcessor::new(config.clone())),
            "lstm" => Box::new(LSTMProcessor::new(config.clone())),
            "gru" => Box::new(GRUProcessor::new(config.clone())),
            "transformer" => Box::new(TransformerProcessor::new(config.clone())),
            _ => Box::new(TransformerProcessor::new(config.clone())),
        };
        
        // 创建全局处理器
        let global_processor: Box<dyn ContextProcessor> = match config.processor_type.as_str() {
            "rnn" => Box::new(RNNProcessor::new(config.clone())),
            "lstm" => Box::new(LSTMProcessor::new(config.clone())),
            "gru" => Box::new(GRUProcessor::new(config.clone())),
            "transformer" => Box::new(TransformerProcessor::new(config.clone())),
            _ => Box::new(TransformerProcessor::new(config.clone())),
        };
        
        let fusion_weights = Array2::zeros((2 * hidden_dim, hidden_dim));
        let fusion_bias = Array1::zeros(hidden_dim);
        let activation = Activation::default();
        let normalization = Normalization::default();
        
        Self {
            config,
            local_processor,
            global_processor,
            fusion_weights,
            fusion_bias,
            activation,
            normalization,
        }
    }

    /// 前向传播
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // 局部处理
        let local_features = self.local_processor.process(&ContextHistory::new(x.nrows()));
        
        // 全局处理
        let global_features = self.global_processor.process(&ContextHistory::new(x.nrows()));
        
        // 特征融合
        let mut concat = Array1::concatenate(
            ndarray::Axis(0),
            &[local_features.view(), global_features.view()],
        ).unwrap();
        
        concat = concat.dot(&self.fusion_weights) + &self.fusion_bias;
        concat = self.activation.apply(&concat);
        
        // 归一化
        self.normalization.apply(&concat.into_shape((1, self.config.hidden_dim)).unwrap())
    }
}

impl ContextProcessor for HierarchicalProcessor {
    fn process(&self, context: &ContextHistory) -> Array1<f32> {
        let features = context.get_features();
        let output = self.forward(&features);
        output.slice(ndarray::s![0, ..]).to_owned()
    }
    
    fn update_config(&mut self, config: &ContextAwareConfig) {
        self.config = config.clone();
        self.local_processor.update_config(config);
        self.global_processor.update_config(config);
    }
    
    fn get_config(&self) -> &ContextAwareConfig {
        &self.config
    }
} 