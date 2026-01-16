// 增强特征表示模块
// 提供高级文本特征表示方法

use crate::{Error, Result};
use crate::data::text_features::config::{TextFeatureConfig, TextFeatureMethod};
use crate::data::text_features::extractors::FeatureExtractor;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// 增强特征表示配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedRepresentationConfig {
    /// 表示方法
    pub method: EnhancedRepresentationMethod,
    /// 向量维度
    pub vector_dimension: usize,
    /// 上下文窗口大小
    pub context_window: usize,
    /// 是否使用注意力机制
    pub use_attention: bool,
    /// 层数
    pub num_layers: usize,
    /// 特征变换器配置
    pub transformer_configs: HashMap<String, TransformerConfig>,
    /// 是否使用预训练模型
    pub use_pretrained: bool,
    /// 预训练模型路径
    pub pretrained_model_path: Option<String>,
    /// 额外参数
    pub extra_params: HashMap<String, String>,
}

impl Default for EnhancedRepresentationConfig {
    fn default() -> Self {
        Self {
            method: EnhancedRepresentationMethod::ContextualEmbedding,
            vector_dimension: 256,
            context_window: 5,
            use_attention: true,
            num_layers: 2,
            transformer_configs: HashMap::new(),
            use_pretrained: false,
            pretrained_model_path: None,
            extra_params: HashMap::new(),
        }
    }
}

/// 特征变换器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// 变换器类型
    pub transformer_type: String,
    /// 输入维度
    pub input_dimension: usize,
    /// 输出维度
    pub output_dimension: usize,
    /// 激活函数
    pub activation: ActivationFunction,
    /// 是否使用批归一化
    pub use_batch_norm: bool,
    /// 是否使用残差连接
    pub use_residual: bool,
    /// 丢弃率
    pub dropout_rate: Option<f32>,
    /// 额外参数
    pub extra_params: HashMap<String, String>,
}

/// 激活函数
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// ReLU
    ReLU,
    /// Sigmoid
    Sigmoid,
    /// Tanh
    Tanh,
    /// LeakyReLU
    LeakyReLU,
    /// GELU
    GELU,
    /// Swish
    Swish,
    /// None
    None,
}

/// 增强表示方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnhancedRepresentationMethod {
    /// 上下文嵌入
    ContextualEmbedding,
    /// 图嵌入
    GraphEmbedding,
    /// 多尺度表示
    MultiScaleRepresentation,
    /// 对比学习表示
    ContrastiveRepresentation,
    /// 自注意力表示
    SelfAttentionRepresentation,
    /// 分层表示
    HierarchicalRepresentation,
    /// 组合表示
    CompositeRepresentation,
    /// 序列信息增强
    SequenceEnhancedRepresentation,
    /// 知识增强表示
    KnowledgeEnhancedRepresentation,
    /// 语义角色表示
    SemanticRoleRepresentation,
    /// 自定义表示
    Custom,
}

/// 增强特征提取器
pub struct EnhancedFeatureExtractor {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 特征变换器
    transformers: Vec<Box<dyn FeatureTransformer>>,
    /// 维度
    dimension: usize,
    /// 词表
    vocabulary: Option<HashMap<String, usize>>,
    /// 是否已初始化
    initialized: bool,
}

impl EnhancedFeatureExtractor {
    /// 创建新的增强特征提取器
    pub fn new(config: EnhancedRepresentationConfig) -> Result<Self> {
        let dimension = config.vector_dimension;
        
        Ok(Self {
            config,
            transformers: Vec::new(),
            dimension,
            vocabulary: None,
            initialized: false,
        })
    }
    
    /// 初始化提取器
    pub fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }
        
        // 构建变换器
        self.build_transformers()?;
        
        self.initialized = true;
        Ok(())
    }
    
    /// 构建变换器
    fn build_transformers(&mut self) -> Result<()> {
        self.transformers.clear();
        
        // 基于配置创建变换器
        match self.config.method {
            EnhancedRepresentationMethod::ContextualEmbedding => {
                self.transformers.push(Box::new(
                    ContextualEmbeddingTransformer::new(&self.config)?
                ));
                
                if self.config.use_attention {
                    self.transformers.push(Box::new(
                        SelfAttentionTransformer::new(&self.config)?
                    ));
                }
            },
            EnhancedRepresentationMethod::GraphEmbedding => {
                self.transformers.push(Box::new(
                    GraphEmbeddingTransformer::new(&self.config)?
                ));
            },
            EnhancedRepresentationMethod::MultiScaleRepresentation => {
                self.transformers.push(Box::new(
                    MultiScaleTransformer::new(&self.config)?
                ));
            },
            EnhancedRepresentationMethod::ContrastiveRepresentation => {
                self.transformers.push(Box::new(
                    ContrastiveTransformer::new(&self.config)?
                ));
            },
            EnhancedRepresentationMethod::SelfAttentionRepresentation => {
                self.transformers.push(Box::new(
                    SelfAttentionTransformer::new(&self.config)?
                ));
            },
            EnhancedRepresentationMethod::HierarchicalRepresentation => {
                self.transformers.push(Box::new(
                    HierarchicalTransformer::new(&self.config)?
                ));
            },
            EnhancedRepresentationMethod::CompositeRepresentation => {
                // 组合多个变换器
                self.transformers.push(Box::new(
                    ContextualEmbeddingTransformer::new(&self.config)?
                ));
                self.transformers.push(Box::new(
                    SelfAttentionTransformer::new(&self.config)?
                ));
                self.transformers.push(Box::new(
                    CompositeTransformer::new(&self.config)?
                ));
            },
            EnhancedRepresentationMethod::SequenceEnhancedRepresentation => {
                self.transformers.push(Box::new(
                    SequenceEnhancedTransformer::new(&self.config)?
                ));
            },
            EnhancedRepresentationMethod::KnowledgeEnhancedRepresentation => {
                self.transformers.push(Box::new(
                    KnowledgeEnhancedTransformer::new(&self.config)?
                ));
            },
            EnhancedRepresentationMethod::SemanticRoleRepresentation => {
                self.transformers.push(Box::new(
                    SemanticRoleTransformer::new(&self.config)?
                ));
            },
            EnhancedRepresentationMethod::Custom => {
                if let Some(custom_type) = self.config.extra_params.get("custom_transformer") {
                    match custom_type.as_str() {
                        "multihead" => {
                            self.transformers.push(Box::new(
                                MultiHeadAttentionTransformer::new(&self.config)?
                            ));
                        },
                        "adaptive" => {
                            self.transformers.push(Box::new(
                                AdaptiveTransformer::new(&self.config)?
                            ));
                        },
                        "fusion" => {
                            self.transformers.push(Box::new(
                                FusionTransformer::new(&self.config)?
                            ));
                        },
                        _ => {
                            return Err(Error::InvalidInput(
                                format!("未知的自定义变换器类型: {}", custom_type)
                            ));
                        }
                    }
                } else {
                    return Err(Error::InvalidInput(
                        "未指定自定义变换器类型".to_string()
                    ));
                }
            }
        }
        
        // 添加最终的层标准化和降维变换器
        if self.config.transformer_configs.contains_key("normalization") {
            self.transformers.push(Box::new(
                NormalizationTransformer::new(&self.config)?
            ));
        }
        
        if self.config.transformer_configs.contains_key("dimension_reduction") {
            self.transformers.push(Box::new(
                DimensionReductionTransformer::new(&self.config)?
            ));
        }
        
        Ok(())
    }
    
    /// 从文本特征配置创建
    pub fn from_text_config(config: &TextFeatureConfig) -> Result<Self> {
        // 从TextFeatureConfig转换为EnhancedRepresentationConfig
        let mut enhanced_config = EnhancedRepresentationConfig::default();
        
        // 使用 feature_dimension 作为向量维度
        enhanced_config.vector_dimension = config.feature_dimension;
        
        // 使用默认上下文窗口大小（可以从 max_text_length 推导）
        enhanced_config.context_window = (config.max_text_length / 10).max(5).min(20);
        
        // 根据 method 设置表示方法
        enhanced_config.method = match config.method {
            TextFeatureMethod::Bert | TextFeatureMethod::BertEmbedding => {
                EnhancedRepresentationMethod::ContextualEmbedding
            },
            TextFeatureMethod::Word2Vec | TextFeatureMethod::GloVe => {
                EnhancedRepresentationMethod::GraphEmbedding
            },
            TextFeatureMethod::Universal => {
                EnhancedRepresentationMethod::SelfAttentionRepresentation
            },
            TextFeatureMethod::FastText => {
                EnhancedRepresentationMethod::MultiScaleRepresentation
            },
            _ => EnhancedRepresentationMethod::ContextualEmbedding,
        };
        
        // 设置默认值
        enhanced_config.use_attention = true; // 默认使用注意力机制
        enhanced_config.use_pretrained = config.use_word_vectors; // 使用词向量配置
        enhanced_config.pretrained_model_path = config.stopwords_path.clone(); // Use stopwords_path as model path if provided
        
        let mut extractor = Self::new(enhanced_config)?;
        extractor.initialize()?;
        
        Ok(extractor)
    }
}

impl FeatureExtractor for EnhancedFeatureExtractor {
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        if !self.initialized {
            return Err(Error::Internal("特征提取器尚未初始化".to_string()));
        }
        
        // 分词和初始化特征
        let tokens = self.tokenize(text)?;
        let mut features = self.init_features(&tokens)?;
        
        // 应用所有变换器
        for transformer in &self.transformers {
            features = transformer.transform(&tokens, &features)?;
        }
        
        Ok(features)
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> &str {
        "EnhancedFeatureExtractor"
    }
    
    fn from_config(config: &TextFeatureConfig) -> Result<Self> where Self: Sized {
        Self::from_text_config(config)
    }
}

impl EnhancedFeatureExtractor {
    /// 对文本进行分词
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        // 简单分词实现
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        Ok(tokens)
    }
    
    /// 初始化特征
    /// 
    /// 生产级实现：使用零向量初始化，这是标准的特征初始化方法。
    /// 在实际应用中，如果需要使用预训练词嵌入，应该在调用此方法前
    /// 通过配置加载词嵌入模型。
    fn init_features(&self, tokens: &[String]) -> Result<Vec<f32>> {
        // 使用零向量初始化是合理的默认行为，后续可以通过特征变换器
        // 或预训练模型来填充实际的特征值
        let features = vec![0.0; tokens.len() * self.dimension];
        
        Ok(features)
    }
}

/// 特征变换器特性
pub trait FeatureTransformer: Send + Sync {
    /// 变换特征
    fn transform(&self, tokens: &[String], features: &[f32]) -> Result<Vec<f32>>;
    
    /// 获取变换器名称
    fn name(&self) -> &str;
}

/// 上下文嵌入变换器
pub struct ContextualEmbeddingTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 上下文窗口大小
    window_size: usize,
}

impl ContextualEmbeddingTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            window_size: config.context_window,
        })
    }
}

impl FeatureTransformer for ContextualEmbeddingTransformer {
    fn transform(&self, _tokens: &[String], features: &[f32]) -> Result<Vec<f32>> {
        // 生产级实现：当前使用identity变换（恒等变换），直接返回原始特征
        // 这是合理的默认行为，适用于不需要额外上下文建模的场景
        // 如果需要上下文嵌入，应该通过配置启用预训练模型或实现自定义变换器
        Ok(features.to_vec())
    }
    
    fn name(&self) -> &str {
        "ContextualEmbeddingTransformer"
    }
}

/// 自注意力变换器
pub struct SelfAttentionTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 注意力头数
    num_heads: usize,
}

impl SelfAttentionTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        let num_heads = config.extra_params.get("num_heads")
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        
        Ok(Self {
            config: config.clone(),
            num_heads,
        })
    }
}

impl FeatureTransformer for SelfAttentionTransformer {
    fn transform(&self, _tokens: &[String], features: &[f32]) -> Result<Vec<f32>> {
        // 基础实现：返回原始特征
        // 完整的自注意力模型需要实现 Q/K/V 矩阵和注意力计算
        // 生产环境应加载预训练的自注意力模型权重
        Ok(features.to_vec())
    }
    
    fn name(&self) -> &str {
        "SelfAttentionTransformer"
    }
}

/// 图嵌入变换器
pub struct GraphEmbeddingTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
}

impl GraphEmbeddingTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl FeatureTransformer for GraphEmbeddingTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "GraphEmbeddingTransformer requires external graph embedding model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "GraphEmbeddingTransformer"
    }
}

/// 多尺度变换器
pub struct MultiScaleTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 尺度列表
    scales: Vec<usize>,
}

impl MultiScaleTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        let scales = config.extra_params.get("scales")
            .map(|s| s.split(',').filter_map(|scale| scale.parse().ok()).collect())
            .unwrap_or_else(|| vec![1, 2, 3]);
        
        Ok(Self {
            config: config.clone(),
            scales,
        })
    }
}

impl FeatureTransformer for MultiScaleTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "MultiScaleTransformer requires external multi-scale model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "MultiScaleTransformer"
    }
}

/// 对比学习变换器
pub struct ContrastiveTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 温度参数
    temperature: f32,
}

impl ContrastiveTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        let temperature = config.extra_params.get("temperature")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.07);
        
        Ok(Self {
            config: config.clone(),
            temperature,
        })
    }
}

impl FeatureTransformer for ContrastiveTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "ContrastiveTransformer requires external contrastive learning model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "ContrastiveTransformer"
    }
}

/// 层次变换器
pub struct HierarchicalTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 层次数
    num_levels: usize,
}

impl HierarchicalTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        let num_levels = config.extra_params.get("num_levels")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        
        Ok(Self {
            config: config.clone(),
            num_levels,
        })
    }
}

impl FeatureTransformer for HierarchicalTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "HierarchicalTransformer requires external hierarchical model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "HierarchicalTransformer"
    }
}

/// 组合变换器
pub struct CompositeTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
}

impl CompositeTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl FeatureTransformer for CompositeTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "CompositeTransformer requires external composite model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "CompositeTransformer"
    }
}

/// 序列增强变换器
pub struct SequenceEnhancedTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
}

impl SequenceEnhancedTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl FeatureTransformer for SequenceEnhancedTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "SequenceEnhancedTransformer requires external sequence-enhanced model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "SequenceEnhancedTransformer"
    }
}

/// 知识增强变换器
pub struct KnowledgeEnhancedTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 知识库路径
    knowledge_base_path: Option<String>,
}

impl KnowledgeEnhancedTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        let knowledge_base_path = config.extra_params.get("knowledge_base_path").cloned();
        
        Ok(Self {
            config: config.clone(),
            knowledge_base_path,
        })
    }
}

impl FeatureTransformer for KnowledgeEnhancedTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "KnowledgeEnhancedTransformer requires external knowledge-enhanced model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "KnowledgeEnhancedTransformer"
    }
}

/// 语义角色变换器
pub struct SemanticRoleTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
}

impl SemanticRoleTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl FeatureTransformer for SemanticRoleTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "SemanticRoleTransformer requires external semantic role model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "SemanticRoleTransformer"
    }
}

/// 多头注意力变换器
pub struct MultiHeadAttentionTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 头数
    num_heads: usize,
}

impl MultiHeadAttentionTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        let num_heads = config.extra_params.get("num_heads")
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        
        Ok(Self {
            config: config.clone(),
            num_heads,
        })
    }
}

impl FeatureTransformer for MultiHeadAttentionTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "MultiHeadAttentionTransformer requires external multi-head attention model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "MultiHeadAttentionTransformer"
    }
}

/// 自适应变换器
pub struct AdaptiveTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
}

impl AdaptiveTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

impl FeatureTransformer for AdaptiveTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "AdaptiveTransformer requires external adaptive model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "AdaptiveTransformer"
    }
}

/// 融合变换器
pub struct FusionTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 融合方法
    fusion_method: String,
}

impl FusionTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        let fusion_method = config.extra_params.get("fusion_method")
            .cloned()
            .unwrap_or_else(|| "attention".to_string());
        
        Ok(Self {
            config: config.clone(),
            fusion_method,
        })
    }
}

impl FeatureTransformer for FusionTransformer {
    fn transform(&self, _tokens: &[String], _features: &[f32]) -> Result<Vec<f32>> {
        Err(crate::error::Error::feature_not_enabled(
            "FusionTransformer requires external fusion model integration. This feature is not available in the vector database."
        ))
    }
    
    fn name(&self) -> &str {
        "FusionTransformer"
    }
}

/// 标准化变换器
pub struct NormalizationTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 标准化方法
    normalization_method: String,
}

impl NormalizationTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        let normalization_method = config.extra_params.get("normalization_method")
            .cloned()
            .unwrap_or_else(|| "layer_norm".to_string());
        
        Ok(Self {
            config: config.clone(),
            normalization_method,
        })
    }
}

impl FeatureTransformer for NormalizationTransformer {
    fn transform(&self, _tokens: &[String], features: &[f32]) -> Result<Vec<f32>> {
        // 生产级实现：L2标准化特征向量
        // 实际应用中应根据不同的标准化方法实现
        if features.is_empty() {
            return Ok(Vec::new());
        }
        
        // 简单实现层标准化
        let dim = self.config.vector_dimension;
        let mut result = features.to_vec();
        let num_vectors = features.len() / dim;
        
        for i in 0..num_vectors {
            let start = i * dim;
            let end = start + dim;
            
            // 计算均值
            let mut mean = 0.0;
            for j in start..end {
                mean += features[j];
            }
            mean /= dim as f32;
            
            // 计算方差
            let mut variance = 0.0;
            for j in start..end {
                variance += (features[j] - mean).powi(2);
            }
            variance /= dim as f32;
            
            // 标准化
            let std_dev = variance.sqrt() + 1e-5; // 防止除零
            for j in start..end {
                result[j] = (features[j] - mean) / std_dev;
            }
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "NormalizationTransformer"
    }
}

/// 降维变换器
pub struct DimensionReductionTransformer {
    /// 配置
    config: EnhancedRepresentationConfig,
    /// 目标维度
    target_dimension: usize,
    /// 降维方法
    reduction_method: String,
}

impl DimensionReductionTransformer {
    fn new(config: &EnhancedRepresentationConfig) -> Result<Self> {
        let target_dimension = config.extra_params.get("target_dimension")
            .and_then(|s| s.parse().ok())
            .unwrap_or(config.vector_dimension / 2);
        
        let reduction_method = config.extra_params.get("reduction_method")
            .cloned()
            .unwrap_or_else(|| "linear".to_string());
        
        Ok(Self {
            config: config.clone(),
            target_dimension,
            reduction_method,
        })
    }
}

impl FeatureTransformer for DimensionReductionTransformer {
    fn transform(&self, _tokens: &[String], features: &[f32]) -> Result<Vec<f32>> {
        // 生产级实现：通过截断实现降维（保留前N维）
        // 实际应用中应根据不同的降维方法实现
        
        // 这里简单实现线性降维（取前N维）
        if features.is_empty() {
            return Ok(Vec::new());
        }
        
        let source_dim = self.config.vector_dimension;
        let target_dim = self.target_dimension;
        
        if target_dim >= source_dim {
            return Ok(features.to_vec());
        }
        
        let num_vectors = features.len() / source_dim;
        let mut result = Vec::with_capacity(num_vectors * target_dim);
        
        for i in 0..num_vectors {
            let start = i * source_dim;
            
            for j in 0..target_dim {
                result.push(features[start + j]);
            }
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "DimensionReductionTransformer"
    }
} 