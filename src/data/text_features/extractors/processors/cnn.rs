use ndarray::{Array1, Array2, Array3};
use crate::data::text_features::extractors::{
    types::{ContextItem, ContextHistory},
    config::{ContextAwareConfig, ConvLayerConfig},
    processors::ContextProcessor,
    utils::{activation::Activation, normalization::Normalization},
};

/// CNN处理器
pub struct CNNProcessor {
    /// 配置
    config: ContextAwareConfig,
    /// 卷积层配置
    conv_config: ConvLayerConfig,
    /// 卷积权重
    conv_weights: Array3<f32>,
    /// 卷积偏置
    conv_bias: Array1<f32>,
    /// 全连接层权重
    fc_weights: Array2<f32>,
    /// 全连接层偏置
    fc_bias: Array1<f32>,
    /// 激活函数
    activation: Activation,
    /// 归一化层
    normalization: Normalization,
}

impl CNNProcessor {
    /// 创建新的CNN处理器
    pub fn new(config: ContextAwareConfig) -> Self {
        let input_dim = config.input_dim;
        let hidden_dim = config.hidden_dim;
        
        let conv_config = ConvLayerConfig {
            in_channels: 1,
            out_channels: hidden_dim,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            use_batch_norm: true,
            activation: "relu".to_string(),
        };
        
        let conv_weights = Array3::zeros((
            conv_config.out_channels,
            conv_config.in_channels,
            conv_config.kernel_size,
        ));
        let conv_bias = Array1::zeros(conv_config.out_channels);
        let fc_weights = Array2::zeros((hidden_dim, hidden_dim));
        let fc_bias = Array1::zeros(hidden_dim);
        let activation = Activation::default();
        let normalization = Normalization::default();
        
        Self {
            config,
            conv_config,
            conv_weights,
            conv_bias,
            fc_weights,
            fc_bias,
            activation,
            normalization,
        }
    }

    /// 卷积操作
    fn conv(&self, x: &Array2<f32>) -> Array2<f32> {
        let batch_size = x.nrows();
        let seq_len = x.ncols();
        let mut output = Array2::zeros((batch_size, seq_len));
        
        for b in 0..batch_size {
            for c in 0..self.conv_config.out_channels {
                for i in 0..seq_len {
                    let mut sum = 0.0;
                    for k in 0..self.conv_config.kernel_size {
                        let pos = i + k - self.conv_config.padding;
                        if pos >= 0 && pos < seq_len {
                            sum += x[[b, pos]] * self.conv_weights[[c, 0, k]];
                        }
                    }
                    output[[b, i]] += sum + self.conv_bias[c];
                }
            }
        }
        
        output
    }

    /// 前向传播
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // 卷积层
        let mut h = self.conv(x);
        h = self.activation.apply(&h);
        
        // 归一化
        h = self.normalization.apply(&h);
        
        // 全连接层
        h = h.dot(&self.fc_weights) + &self.fc_bias;
        h = self.activation.apply(&h);
        
        h
    }
}

impl ContextProcessor for CNNProcessor {
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