// 统一特征提取器类型定义
// 用于整合各个模块中定义的不同特征提取器

use serde::{Serialize, Deserialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use crate::error::Result;

/// 特征提取器类型
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExtractorType {
    /// 文本特征提取器
    Text(TextExtractorType),
    /// 图像特征提取器
    Image,
    /// 音频特征提取器
    Audio,
    /// 视频特征提取器
    Video,
    /// 数值特征提取器
    Numeric(NumericExtractorType),
    /// 类别特征提取器
    Categorical(CategoricalExtractorType),
    /// 多模态特征提取器
    MultiModal(MultiModalExtractorType),
    /// 通用特征提取器
    Generic(GenericExtractorType),
    /// 自定义特征提取器
    Custom(String),
    /// 通用文本提取器
    TextTfIdf,
    /// 词袋模型
    TextBagOfWords,
    /// BERT模型
    TextBERT,
    /// Word2Vec模型
    TextWord2Vec,
    /// FastText模型
    TextFastText,
    /// GloVe模型
    TextGloVe,
    /// LSTM模型
    TextLSTM,
    /// 通用Transformer模型
    TextTransformer,
    /// 卷积神经网络
    ImageCNN,
    /// ResNet模型
    ImageResNet,
    /// VGG模型
    ImageVGG,
    /// Inception模型
    ImageInception,
    /// SIFT特征
    ImageSIFT,
    /// HOG特征
    ImageHOG,
    /// 通用多模态提取器
    MultiModalCLIP,
    /// ViLT模型
    MultiModalViLT,
    /// 组合提取器
    Composite,
}

impl PartialEq for ExtractorType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ExtractorType::Text(a), ExtractorType::Text(b)) => a == b,
            (ExtractorType::Image, ExtractorType::Image) => true,
            (ExtractorType::Audio, ExtractorType::Audio) => true,
            (ExtractorType::Video, ExtractorType::Video) => true,
            (ExtractorType::Numeric(a), ExtractorType::Numeric(b)) => a == b,
            (ExtractorType::Categorical(a), ExtractorType::Categorical(b)) => a == b,
            (ExtractorType::MultiModal(a), ExtractorType::MultiModal(b)) => a == b,
            (ExtractorType::Generic(a), ExtractorType::Generic(b)) => a == b,
            (ExtractorType::Custom(a), ExtractorType::Custom(b)) => a == b,
            (ExtractorType::TextTfIdf, ExtractorType::TextTfIdf) => true,
            (ExtractorType::TextBagOfWords, ExtractorType::TextBagOfWords) => true,
            (ExtractorType::TextBERT, ExtractorType::TextBERT) => true,
            (ExtractorType::TextWord2Vec, ExtractorType::TextWord2Vec) => true,
            (ExtractorType::TextFastText, ExtractorType::TextFastText) => true,
            (ExtractorType::TextGloVe, ExtractorType::TextGloVe) => true,
            (ExtractorType::TextLSTM, ExtractorType::TextLSTM) => true,
            (ExtractorType::TextTransformer, ExtractorType::TextTransformer) => true,
            (ExtractorType::ImageCNN, ExtractorType::ImageCNN) => true,
            (ExtractorType::ImageResNet, ExtractorType::ImageResNet) => true,
            (ExtractorType::ImageVGG, ExtractorType::ImageVGG) => true,
            (ExtractorType::ImageInception, ExtractorType::ImageInception) => true,
            (ExtractorType::ImageSIFT, ExtractorType::ImageSIFT) => true,
            (ExtractorType::ImageHOG, ExtractorType::ImageHOG) => true,
            (ExtractorType::MultiModalCLIP, ExtractorType::MultiModalCLIP) => true,
            (ExtractorType::MultiModalViLT, ExtractorType::MultiModalViLT) => true,
            (ExtractorType::Composite, ExtractorType::Composite) => true,
            _ => false,
        }
    }
}

impl Eq for ExtractorType {}

impl Hash for ExtractorType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            ExtractorType::Text(t) => {
                0.hash(state);
                t.hash(state);
            }
            ExtractorType::Image => 1.hash(state),
            ExtractorType::Audio => 2.hash(state),
            ExtractorType::Video => 3.hash(state),
            ExtractorType::Numeric(t) => {
                4.hash(state);
                t.hash(state);
            }
            ExtractorType::Categorical(t) => {
                5.hash(state);
                t.hash(state);
            }
            ExtractorType::MultiModal(t) => {
                6.hash(state);
                t.hash(state);
            }
            ExtractorType::Generic(t) => {
                7.hash(state);
                t.hash(state);
            }
            ExtractorType::Custom(s) => {
                8.hash(state);
                s.hash(state);
            }
            ExtractorType::TextTfIdf => 9.hash(state),
            ExtractorType::TextBagOfWords => 10.hash(state),
            ExtractorType::TextBERT => 11.hash(state),
            ExtractorType::TextWord2Vec => 12.hash(state),
            ExtractorType::TextFastText => 13.hash(state),
            ExtractorType::TextGloVe => 14.hash(state),
            ExtractorType::TextLSTM => 15.hash(state),
            ExtractorType::TextTransformer => 16.hash(state),
            ExtractorType::ImageCNN => 17.hash(state),
            ExtractorType::ImageResNet => 18.hash(state),
            ExtractorType::ImageVGG => 19.hash(state),
            ExtractorType::ImageInception => 20.hash(state),
            ExtractorType::ImageSIFT => 21.hash(state),
            ExtractorType::ImageHOG => 22.hash(state),
            ExtractorType::MultiModalCLIP => 23.hash(state),
            ExtractorType::MultiModalViLT => 24.hash(state),
            ExtractorType::Composite => 25.hash(state),
        }
    }
}

/// 文本特征提取器类型
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TextExtractorType {
    /// TF-IDF（词频-逆文档频率）
    TfIdf,
    /// 词袋模型
    BagOfWords,
    /// Word2Vec
    Word2Vec,
    /// BERT
    BERT,
    /// FastText
    FastText,
    /// 自定义文本特征提取器
    Custom(String),
}

impl fmt::Display for TextExtractorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TextExtractorType::TfIdf => write!(f, "TF-IDF"),
            TextExtractorType::BagOfWords => write!(f, "Bag of Words"),
            TextExtractorType::Word2Vec => write!(f, "Word2Vec"),
            TextExtractorType::BERT => write!(f, "BERT"),
            TextExtractorType::FastText => write!(f, "FastText"),
            TextExtractorType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// 数值特征提取器类型
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumericExtractorType {
    /// 标准化（均值为0，标准差为1）
    Standardize,
    /// 归一化（缩放到[0,1]区间）
    Normalize,
    /// 对数变换
    LogTransform,
    /// 幂变换
    PowerTransform,
    /// 离散化
    Discretize,
    /// 自定义数值特征提取器
    Custom(String),
}

impl fmt::Display for NumericExtractorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericExtractorType::Standardize => write!(f, "Standardize"),
            NumericExtractorType::Normalize => write!(f, "Normalize"),
            NumericExtractorType::LogTransform => write!(f, "Log Transform"),
            NumericExtractorType::PowerTransform => write!(f, "Power Transform"),
            NumericExtractorType::Discretize => write!(f, "Discretize"),
            NumericExtractorType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// 分类特征提取器类型
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CategoricalExtractorType {
    /// 独热编码
    OneHot,
    /// 标签编码
    LabelEncoding,
    /// 频率编码
    FrequencyEncoding,
    /// 目标编码
    TargetEncoding,
    /// 哈希编码
    HashEncoding,
    /// 自定义分类特征提取器
    Custom(String),
}

impl fmt::Display for CategoricalExtractorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CategoricalExtractorType::OneHot => write!(f, "One-Hot Encoding"),
            CategoricalExtractorType::LabelEncoding => write!(f, "Label Encoding"),
            CategoricalExtractorType::FrequencyEncoding => write!(f, "Frequency Encoding"),
            CategoricalExtractorType::TargetEncoding => write!(f, "Target Encoding"),
            CategoricalExtractorType::HashEncoding => write!(f, "Hash Encoding"),
            CategoricalExtractorType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// 多模态特征提取器类型
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MultiModalExtractorType {
    /// 特征融合
    Fusion,
    /// 多模态BERT
    MultiModalBERT,
    /// 多模态Transformer
    MultiModalTransformer,
    /// 自定义多模态特征提取器
    Custom(String),
}

impl fmt::Display for MultiModalExtractorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiModalExtractorType::Fusion => write!(f, "Feature Fusion"),
            MultiModalExtractorType::MultiModalBERT => write!(f, "Multi-Modal BERT"),
            MultiModalExtractorType::MultiModalTransformer => write!(f, "Multi-Modal Transformer"),
            MultiModalExtractorType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// 通用特征提取器类型
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GenericExtractorType {
    /// 身份特征提取器（不改变输入）
    Identity,
    /// 特征选择
    FeatureSelection,
    /// 主成分分析
    PCA,
    /// 自编码器
    Autoencoder,
    /// 自定义通用特征提取器
    Custom(String),
}

impl fmt::Display for GenericExtractorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenericExtractorType::Identity => write!(f, "Identity"),
            GenericExtractorType::FeatureSelection => write!(f, "Feature Selection"),
            GenericExtractorType::PCA => write!(f, "PCA"),
            GenericExtractorType::Autoencoder => write!(f, "Autoencoder"),
            GenericExtractorType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

impl fmt::Display for ExtractorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtractorType::Text(t) => write!(f, "Text::{}", t),
            ExtractorType::Image => write!(f, "Image"),
            ExtractorType::Audio => write!(f, "Audio"),
            ExtractorType::Video => write!(f, "Video"),
            ExtractorType::Numeric(t) => write!(f, "Numeric::{}", t),
            ExtractorType::Categorical(t) => write!(f, "Categorical::{}", t),
            ExtractorType::MultiModal(t) => write!(f, "MultiModal::{}", t),
            ExtractorType::Generic(t) => write!(f, "Generic::{}", t),
            ExtractorType::Custom(s) => write!(f, "Custom::{}", s),
            ExtractorType::TextTfIdf => write!(f, "TextTfIdf"),
            ExtractorType::TextBagOfWords => write!(f, "TextBagOfWords"),
            ExtractorType::TextBERT => write!(f, "TextBERT"),
            ExtractorType::TextWord2Vec => write!(f, "TextWord2Vec"),
            ExtractorType::TextFastText => write!(f, "TextFastText"),
            ExtractorType::TextGloVe => write!(f, "TextGloVe"),
            ExtractorType::TextLSTM => write!(f, "TextLSTM"),
            ExtractorType::TextTransformer => write!(f, "TextTransformer"),
            ExtractorType::ImageCNN => write!(f, "ImageCNN"),
            ExtractorType::ImageResNet => write!(f, "ImageResNet"),
            ExtractorType::ImageVGG => write!(f, "ImageVGG"),
            ExtractorType::ImageInception => write!(f, "ImageInception"),
            ExtractorType::ImageSIFT => write!(f, "ImageSIFT"),
            ExtractorType::ImageHOG => write!(f, "ImageHOG"),
            ExtractorType::MultiModalCLIP => write!(f, "MultiModalCLIP"),
            ExtractorType::MultiModalViLT => write!(f, "MultiModalViLT"),
            ExtractorType::Composite => write!(f, "Composite"),
        }
    }
}

impl Default for ExtractorType {
    fn default() -> Self {
        ExtractorType::Text(TextExtractorType::TfIdf)
    }
}

/// 特征类型
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureType {
    /// 文本特征
    Text,
    /// 图像特征
    Image,
    /// 音频特征
    Audio,
    /// 视频特征
    Video,
    /// 数值特征
    Numeric,
    /// 类别特征
    Categorical,
    /// 时间特征
    Temporal,
    /// 多模态特征
    Multimodal,
    /// 自定义特征
    Custom(u32),
    /// 混合特征
    Mixed,
    /// 通用特征，格式未知
    Generic,
}

impl fmt::Display for FeatureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeatureType::Text => write!(f, "文本特征"),
            FeatureType::Image => write!(f, "图像特征"),
            FeatureType::Audio => write!(f, "音频特征"),
            FeatureType::Video => write!(f, "视频特征"),
            FeatureType::Numeric => write!(f, "数值特征"),
            FeatureType::Categorical => write!(f, "类别特征"),
            FeatureType::Temporal => write!(f, "时间特征"),
            FeatureType::Multimodal => write!(f, "多模态特征"),
            FeatureType::Custom(id) => write!(f, "自定义特征({})", id),
            FeatureType::Mixed => write!(f, "混合特征"),
            FeatureType::Generic => write!(f, "通用特征"),
        }
    }
}

impl Default for FeatureType {
    fn default() -> Self {
        FeatureType::Text
    }
}

/// 特征提取结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionResult {
    /// 提取的特征向量
    pub features: Vec<f32>,
    /// 特征维度
    pub dimension: usize,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 提取时间（毫秒）
    pub extraction_time_ms: u64,
    /// 额外元数据
    pub metadata: std::collections::HashMap<String, String>,
}

impl FeatureExtractionResult {
    /// 创建新的特征提取结果
    pub fn new(features: Vec<f32>, feature_type: FeatureType) -> Self {
        Self {
            dimension: features.len(),
            features,
            feature_type,
            extraction_time_ms: 0,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 设置提取时间
    pub fn with_extraction_time(mut self, time_ms: u64) -> Self {
        self.extraction_time_ms = time_ms;
        self
    }
}

impl ExtractorType {
    /// 获取提取器类型的默认特征类型
    pub fn default_feature_type(&self) -> FeatureType {
        match self {
            ExtractorType::Text(t) => FeatureType::Text,
            ExtractorType::Image => FeatureType::Image,
            ExtractorType::Audio => FeatureType::Audio,
            ExtractorType::Video => FeatureType::Video,
            ExtractorType::Numeric(t) => FeatureType::Numeric,
            ExtractorType::Categorical(t) => FeatureType::Categorical,
            ExtractorType::MultiModal(t) => FeatureType::Multimodal,
            ExtractorType::Generic(t) => FeatureType::Generic,
            ExtractorType::Custom(_) => FeatureType::Generic,
            ExtractorType::TextTfIdf => FeatureType::Text,
            ExtractorType::TextBagOfWords => FeatureType::Text,
            ExtractorType::TextBERT => FeatureType::Text,
            ExtractorType::TextWord2Vec => FeatureType::Text,
            ExtractorType::TextFastText => FeatureType::Text,
            ExtractorType::TextGloVe => FeatureType::Text,
            ExtractorType::TextLSTM => FeatureType::Text,
            ExtractorType::TextTransformer => FeatureType::Text,
            ExtractorType::ImageCNN => FeatureType::Image,
            ExtractorType::ImageResNet => FeatureType::Image,
            ExtractorType::ImageVGG => FeatureType::Image,
            ExtractorType::ImageInception => FeatureType::Image,
            ExtractorType::ImageSIFT => FeatureType::Image,
            ExtractorType::ImageHOG => FeatureType::Image,
            ExtractorType::MultiModalCLIP => FeatureType::Multimodal,
            ExtractorType::MultiModalViLT => FeatureType::Multimodal,
            ExtractorType::Composite => FeatureType::Generic,
        }
    }
    
    /// 从字符串解析提取器类型
    pub fn from_str(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        let category = parts[0];
        let method = parts.get(1).map(|s| *s);
        
        match (category, method) {
            ("text", None) => Some(ExtractorType::Text(TextExtractorType::TfIdf)),
            ("text", Some("tfidf")) => Some(ExtractorType::Text(TextExtractorType::TfIdf)),
            ("text", Some("bow")) => Some(ExtractorType::Text(TextExtractorType::BagOfWords)),
            ("text", Some("bert")) => Some(ExtractorType::Text(TextExtractorType::BERT)),
            ("text", Some("word2vec")) => Some(ExtractorType::Text(TextExtractorType::Word2Vec)),
            ("text", Some("fasttext")) => Some(ExtractorType::Text(TextExtractorType::FastText)),
            ("text", Some("glove")) => Some(ExtractorType::Text(TextExtractorType::Custom("glove".to_string()))),
            ("text", Some("lstm")) => Some(ExtractorType::Text(TextExtractorType::Custom("lstm".to_string()))),
            ("text", Some("transformer")) => Some(ExtractorType::Text(TextExtractorType::Custom("transformer".to_string()))),
            
            ("image", None) => Some(ExtractorType::Image),
            ("image", Some("cnn")) => Some(ExtractorType::ImageCNN),
            ("image", Some("resnet")) => Some(ExtractorType::ImageResNet),
            ("image", Some("vgg")) => Some(ExtractorType::ImageVGG),
            ("image", Some("inception")) => Some(ExtractorType::ImageInception),
            ("image", Some("sift")) => Some(ExtractorType::ImageSIFT),
            ("image", Some("hog")) => Some(ExtractorType::ImageHOG),
            
            ("multimodal", None) => Some(ExtractorType::MultiModal(MultiModalExtractorType::Fusion)),
            ("multimodal", Some("clip")) => Some(ExtractorType::MultiModal(MultiModalExtractorType::MultiModalBERT)),
            ("multimodal", Some("vilt")) => Some(ExtractorType::MultiModal(MultiModalExtractorType::MultiModalTransformer)),
            
            ("numeric", None) => Some(ExtractorType::Numeric(NumericExtractorType::Standardize)),
            ("numeric", Some("pca")) => Some(ExtractorType::Numeric(NumericExtractorType::Custom("pca".to_string()))),
            ("numeric", Some("tsne")) => Some(ExtractorType::Numeric(NumericExtractorType::Custom("tsne".to_string()))),
            ("numeric", Some("autoencoder")) => Some(ExtractorType::Numeric(NumericExtractorType::Custom("autoencoder".to_string()))),
            
            ("categorical", None) => Some(ExtractorType::Categorical(CategoricalExtractorType::OneHot)),
            ("categorical", Some("onehot")) => Some(ExtractorType::Categorical(CategoricalExtractorType::OneHot)),
            ("categorical", Some("label")) => Some(ExtractorType::Categorical(CategoricalExtractorType::LabelEncoding)),
            ("categorical", Some("target")) => Some(ExtractorType::Categorical(CategoricalExtractorType::TargetEncoding)),
            
            ("custom", Some(name)) => Some(ExtractorType::Custom(name.to_string())),
            ("composite", None) => Some(ExtractorType::Composite),
            
            _ => None,
        }
    }
}

/// 特征结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Feature {
    /// 特征名称
    pub name: String,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 特征数据
    pub data: Vec<f32>,
    /// 特征元数据
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl Feature {
    /// 创建新特征
    pub fn new(name: impl Into<String>, feature_type: FeatureType, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            feature_type,
            data,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// 增加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// 获取特征维度
    pub fn dimension(&self) -> usize {
        self.data.len()
    }
}

/// 特征向量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// 特征列表
    pub features: Vec<Feature>,
    /// 向量元数据
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl FeatureVector {
    /// 创建新特征向量
    pub fn new() -> Self {
        Self {
            features: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// 添加特征
    pub fn add_feature(&mut self, feature: Feature) {
        self.features.push(feature);
    }
    
    /// 查找特征
    pub fn find_feature(&self, name: &str) -> Option<&Feature> {
        self.features.iter().find(|f| f.name == name)
    }
    
    /// 获取特征数量
    pub fn len(&self) -> usize {
        self.features.len()
    }
    
    /// 判断是否为空
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self::new()
    }
}

/// 特征组
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureGroup {
    /// 组名称
    pub name: String,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 特征向量
    pub vectors: Vec<FeatureVector>,
    /// 组元数据
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl FeatureGroup {
    /// 创建新特征组
    pub fn new(name: impl Into<String>, feature_type: FeatureType) -> Self {
        Self {
            name: name.into(),
            feature_type,
            vectors: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// 添加特征向量
    pub fn add_vector(&mut self, vector: FeatureVector) {
        self.vectors.push(vector);
    }
    
    /// 获取向量数量
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    
    /// 判断是否为空
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}

/// 特征提取器接口
pub trait FeatureExtractor: Send + Sync {
    /// 提取器名称
    fn name(&self) -> &str;
    
    /// 特征类型
    fn feature_type(&self) -> FeatureType;
    
    /// 提取特征
    fn extract(&self, data: &[u8]) -> Result<Feature>;
    
    /// 批量提取特征
    fn extract_batch(&self, data: &[Vec<u8>]) -> Result<Vec<Feature>> {
        data.iter()
            .map(|item| self.extract(item))
            .collect()
    }
}

/// 特征转换器接口
pub trait FeatureTransformer: Send + Sync {
    /// 转换器名称
    fn name(&self) -> &str;
    
    /// 输入特征类型
    fn input_type(&self) -> FeatureType;
    
    /// 输出特征类型
    fn output_type(&self) -> FeatureType;
    
    /// 转换特征
    fn transform(&self, feature: &Feature) -> Result<Feature>;
    
    /// 批量转换特征
    fn transform_batch(&self, features: &[Feature]) -> Result<Vec<Feature>> {
        features.iter()
            .map(|feature| self.transform(feature))
            .collect()
    }
} 