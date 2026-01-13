use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 上下文感知特征提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAwareConfig {
    /// 处理器类型
    pub processor_type: String,
    /// 输入维度
    pub input_dim: usize,
    /// 隐藏层维度
    pub hidden_dim: usize,
    /// 输出维度
    pub output_dim: usize,
    /// 上下文窗口大小
    pub context_window: usize,
    /// 是否使用位置编码
    pub use_position_encoding: bool,
    /// 是否使用层归一化
    pub use_layer_norm: bool,
    /// 是否使用批归一化
    pub use_batch_norm: bool,
    /// 激活函数类型
    pub activation: String,
    /// 处理器特定配置
    pub processor_config: HashMap<String, serde_json::Value>,
}

/// 卷积层配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvLayerConfig {
    /// 输入通道数
    pub in_channels: usize,
    /// 输出通道数
    pub out_channels: usize,
    /// 卷积核大小
    pub kernel_size: usize,
    /// 步长
    pub stride: usize,
    /// 填充
    pub padding: usize,
    /// 是否使用批归一化
    pub use_batch_norm: bool,
    /// 激活函数类型
    pub activation: String,
}

/// 全连接层配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FCLayerConfig {
    /// 输入维度
    pub input_dim: usize,
    /// 输出维度
    pub output_dim: usize,
    /// 是否使用批归一化
    pub use_batch_norm: bool,
    /// 激活函数类型
    pub activation: String,
}

/// 批归一化参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormParams {
    /// 特征数量
    pub num_features: usize,
    /// epsilon值
    pub eps: f32,
    /// momentum值
    pub momentum: f32,
    /// 是否使用仿射变换
    pub affine: bool,
    /// 是否跟踪运行统计
    pub track_running_stats: bool,
}

impl Default for ContextAwareConfig {
    fn default() -> Self {
        Self {
            processor_type: "transformer".to_string(),
            input_dim: 768,
            hidden_dim: 768,
            output_dim: 768,
            context_window: 512,
            use_position_encoding: true,
            use_layer_norm: true,
            use_batch_norm: false,
            activation: "gelu".to_string(),
            processor_config: HashMap::new(),
        }
    }
}

impl Default for BatchNormParams {
    fn default() -> Self {
        Self {
            num_features: 768,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            track_running_stats: true,
        }
    }
} 