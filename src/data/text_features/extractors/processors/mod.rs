mod rnn;
mod lstm;
mod gru;
mod transformer;
mod cnn;
mod gat;
mod custom;
mod hierarchical;

pub use rnn::RNNProcessor;
pub use lstm::LSTMProcessor;
pub use gru::GRUProcessor;
pub use transformer::TransformerProcessor;
pub use cnn::CNNProcessor;
pub use gat::GATProcessor;
pub use custom::CustomAttentionProcessor;
pub use hierarchical::HierarchicalProcessor;

use ndarray::{Array1, Array2, Array3};
use crate::data::text_features::extractors::types::{ContextItem, ContextHistory};
use crate::data::text_features::extractors::config::ContextAwareConfig;

/// 上下文处理器特征
pub trait ContextProcessor: Send + Sync {
    /// 处理上下文
    fn process(&self, context: &ContextHistory) -> Array1<f32>;
    
    /// 更新配置
    fn update_config(&mut self, config: &ContextAwareConfig);
    
    /// 获取配置
    fn get_config(&self) -> &ContextAwareConfig;
} 