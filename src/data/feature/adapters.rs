use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use serde_json;
use crate::error::{Error, Result};
use crate::data::feature::types::{ExtractorType, FeatureType, FeatureExtractionResult, MultiModalExtractorType};
use crate::data::feature::interface::{FeatureExtractor, ExtractorConfig, FeatureExtractionResult as InterfaceFeatureExtractionResult, FeatureVector};
use crate::data::multimodal::extractors::interface::{FeatureExtractor as MultimodalFeatureExtractor, ExtractorType as MultimodalInterfaceExtractorType};
use crate::compat::tensor::TensorData;
use base64::{Engine as _, engine::general_purpose};



/// 多模态特征提取器适配器
/// 将多模态特征提取器适配为通用特征提取器接口
#[derive(Debug)]
pub struct MultimodalFeatureExtractorAdapter {
    /// 内部多模态特征提取器
    inner: Arc<dyn MultimodalFeatureExtractor>,
    /// 适配器配置
    config: ExtractorConfig,
    /// 适配器名称
    name: String,
    /// 输出维度
    output_dimension: usize,
}

impl MultimodalFeatureExtractorAdapter {
    /// 创建新的多模态特征提取器适配器
    pub fn new(
        extractor: Arc<dyn MultimodalFeatureExtractor>,
        config: ExtractorConfig,
        name: String,
    ) -> Self {
        let output_dimension = extractor.get_output_dim();
        Self {
            inner: extractor,
            config,
            name,
            output_dimension,
        }
    }

    /// 从配置创建适配器
    pub fn from_config(config: ExtractorConfig) -> Result<Self> {
        // 根据配置创建对应的多模态特征提取器
        let extractor_type = config.get_option("extractor_type")
            .unwrap_or(&"text".to_string())
            .clone();
        
        let dimension = config.target_dimension.unwrap_or(512);
        
        // 创建模拟的多模态特征提取器
        let extractor = Arc::new(MockMultimodalExtractor::new(extractor_type.clone(), dimension));
        
        Ok(Self::new(
            extractor,
            config,
            format!("multimodal_{}", extractor_type),
        ))
    }

    /// 转换字节数据为多模态格式
    fn convert_bytes_to_multimodal_data(&self, data: &[u8], metadata: Option<&HashMap<String, String>>) -> Result<Vec<f32>> {
        // 将字节数据转换为特征向量
        let features = self.inner.extract_features(data, metadata)?;
        Ok(features)
    }
}

#[async_trait]
impl FeatureExtractor for MultimodalFeatureExtractorAdapter {
    fn extractor_type(&self) -> ExtractorType {
        self.config.extractor_type.clone()
    }

    fn feature_type(&self) -> FeatureType {
        self.config.feature_type.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &ExtractorConfig {
        &self.config
    }

    fn get_config(&self) -> ExtractorConfig {
        self.config.clone()
    }

    async fn extract(&self, input: &str) -> Result<InterfaceFeatureExtractionResult> {
        let start_time = Instant::now();
        
        // 将字符串转换为字节
        let data = input.as_bytes();
        
        // 提取特征
        let features = self.convert_bytes_to_multimodal_data(data, None)?;
        
        let extraction_time = start_time.elapsed();
        
        // 创建结果
        let result = InterfaceFeatureExtractionResult::new(
            features,
            self.config.feature_type.clone(),
            self.config.extractor_type.clone(),
        )
        .with_extraction_time(extraction_time)
        .with_metadata("adapter_type", "multimodal")
        .with_metadata("extractor_name", &self.name);

        Ok(result)
    }

    async fn extract_batch(&self, inputs: &[String]) -> Result<Vec<InterfaceFeatureExtractionResult>> {
        let mut results = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let result = self.extract(input).await?;
            results.push(result);
        }
        
        Ok(results)
    }
}

/// 文本特征提取器适配器
/// 将文本特征提取器适配为通用接口
#[derive(Debug)]
pub struct TextFeatureExtractorAdapter {
    /// 内部文本特征提取器
    inner: Arc<dyn TextFeatureExtractor>,
    /// 适配器配置
    config: ExtractorConfig,
    /// 适配器名称
    name: String,
}

impl TextFeatureExtractorAdapter {
    /// 创建新的文本特征提取器适配器
    pub fn new(
        extractor: Arc<dyn TextFeatureExtractor>,
        config: ExtractorConfig,
        name: String,
    ) -> Self {
        Self {
            inner: extractor,
            config,
            name,
        }
    }

    /// 从配置创建适配器
    pub fn from_config(config: ExtractorConfig) -> Result<Self> {
        let extractor_type = config.get_option("text_extractor_type")
            .unwrap_or(&"tfidf".to_string())
            .clone();
        
        let dimension = config.target_dimension.unwrap_or(300);
        
        // 创建对应的文本特征提取器
        let extractor: Arc<dyn TextFeatureExtractor> = match extractor_type.as_str() {
            "tfidf" => Arc::new(TfIdfExtractor::new(dimension)),
            "word2vec" => Arc::new(Word2VecExtractor::new(dimension)),
            "bert" => Arc::new(BertExtractor::new(dimension)),
            _ => Arc::new(TfIdfExtractor::new(dimension)),
        };
        
        Ok(Self::new(
            extractor,
            config,
            format!("text_{}", extractor_type),
        ))
    }
}

#[async_trait]
impl FeatureExtractor for TextFeatureExtractorAdapter {
    fn extractor_type(&self) -> ExtractorType {
        self.config.extractor_type.clone()
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Text
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &ExtractorConfig {
        &self.config
    }

    fn get_config(&self) -> ExtractorConfig {
        self.config.clone()
    }

    async fn extract(&self, input: &str) -> Result<InterfaceFeatureExtractionResult> {
        let start_time = Instant::now();
        
        // 使用内部提取器提取特征
        let features = self.inner.extract_text_features(input)?;
        
        let extraction_time = start_time.elapsed();
        
        // 创建结果
        let result = InterfaceFeatureExtractionResult::new(
            features,
            FeatureType::Text,
            self.config.extractor_type.clone(),
        )
        .with_extraction_time(extraction_time)
        .with_metadata("adapter_type", "text")
        .with_metadata("extractor_name", &self.name);

        Ok(result)
    }
}

/// 音频特征提取器适配器
/// 将音频特征提取器适配为通用接口
#[derive(Debug)]
pub struct AudioFeatureExtractorAdapter {
    /// 内部音频特征提取器
    inner: Arc<dyn AudioFeatureExtractor>,
    /// 适配器配置
    config: ExtractorConfig,
    /// 适配器名称
    name: String,
}

impl AudioFeatureExtractorAdapter {
    /// 创建新的音频特征提取器适配器
    pub fn new(
        extractor: Arc<dyn AudioFeatureExtractor>,
        config: ExtractorConfig,
        name: String,
    ) -> Self {
        Self {
            inner: extractor,
            config,
            name,
        }
    }

    /// 从配置创建适配器
    pub fn from_config(config: ExtractorConfig) -> Result<Self> {
        let extractor_type = config.get_option("audio_extractor_type")
            .unwrap_or(&"mfcc".to_string())
            .clone();
        
        let dimension = config.target_dimension.unwrap_or(128);
        
        // 创建对应的音频特征提取器
        let extractor: Arc<dyn AudioFeatureExtractor> = match extractor_type.as_str() {
            "mfcc" => Arc::new(MfccExtractor::new(dimension)),
            "spectral" => Arc::new(SpectralExtractor::new(dimension)),
            "chroma" => Arc::new(ChromaExtractor::new(dimension)),
            _ => Arc::new(MfccExtractor::new(dimension)),
        };
        
        Ok(Self::new(
            extractor,
            config,
            format!("audio_{}", extractor_type),
        ))
    }
}

#[async_trait]
impl FeatureExtractor for AudioFeatureExtractorAdapter {
    fn extractor_type(&self) -> ExtractorType {
        self.config.extractor_type.clone()
    }

    fn feature_type(&self) -> FeatureType {
        FeatureType::Audio
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &ExtractorConfig {
        &self.config
    }

    fn get_config(&self) -> ExtractorConfig {
        self.config.clone()
    }

    async fn extract(&self, input: &str) -> Result<InterfaceFeatureExtractionResult> {
        let start_time = Instant::now();
        
        // 将输入解析为音频数据路径或base64编码的音频数据
        let audio_data = self.parse_audio_input(input)?;
        
        // 使用内部提取器提取特征
        let features = self.inner.extract_audio_features(&audio_data)?;
        
        let extraction_time = start_time.elapsed();
        
        // 创建结果
        let result = InterfaceFeatureExtractionResult::new(
            features,
            FeatureType::Audio,
            self.config.extractor_type.clone(),
        )
        .with_extraction_time(extraction_time)
        .with_metadata("adapter_type", "audio")
        .with_metadata("extractor_name", &self.name);

        Ok(result)
    }
}

impl AudioFeatureExtractorAdapter {
    /// 解析音频输入
    fn parse_audio_input(&self, input: &str) -> Result<Vec<u8>> {
        // 尝试解析为base64编码的音频数据
        if let Ok(data) = general_purpose::STANDARD.decode(input) {
            return Ok(data);
        }
        
        // 尝试作为文件路径读取
        if std::path::Path::new(input).exists() {
            return std::fs::read(input)
                .map_err(|e| Error::io_error(format!("读取音频文件失败: {}", e)));
        }
        
        // 默认返回空数据
        Ok(Vec::new())
    }
}

/// 通用特征提取器适配器工厂
pub struct FeatureExtractorAdapterFactory {
    /// 已注册的适配器创建函数
    creators: HashMap<String, Box<dyn Fn(&ExtractorConfig) -> Result<Box<dyn FeatureExtractor>> + Send + Sync>>,
}

impl std::fmt::Debug for FeatureExtractorAdapterFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeatureExtractorAdapterFactory")
            .field("registered_adapters", &self.creators.keys().collect::<Vec<_>>())
            .field("count", &self.creators.len())
            .finish()
    }
}

impl FeatureExtractorAdapterFactory {
    /// 创建新的工厂
    pub fn new() -> Self {
        let mut factory = Self {
            creators: HashMap::new(),
        };
        
        // 注册默认适配器
        factory.register_default_adapters();
        factory
    }

    /// 注册默认适配器
    fn register_default_adapters(&mut self) {
        // 注册多模态适配器
        self.register_adapter("multimodal", Box::new(|config: &ExtractorConfig| {
            let adapter = MultimodalFeatureExtractorAdapter::from_config(config.clone())?;
            Ok(Box::new(adapter) as Box<dyn FeatureExtractor>)
        }));

        // 注册文本适配器
        self.register_adapter("text", Box::new(|config: &ExtractorConfig| {
            let adapter = TextFeatureExtractorAdapter::from_config(config.clone())?;
            Ok(Box::new(adapter) as Box<dyn FeatureExtractor>)
        }));

        // 注册音频适配器
        self.register_adapter("audio", Box::new(|config: &ExtractorConfig| {
            let adapter = AudioFeatureExtractorAdapter::from_config(config.clone())?;
            Ok(Box::new(adapter) as Box<dyn FeatureExtractor>)
        }));
    }

    /// 注册适配器创建函数
    pub fn register_adapter<F>(&mut self, name: &str, creator: F)
    where
        F: Fn(&ExtractorConfig) -> Result<Box<dyn FeatureExtractor>> + Send + Sync + 'static,
    {
        self.creators.insert(name.to_string(), Box::new(creator));
    }

    /// 创建适配器
    pub fn create_adapter(&self, adapter_type: &str, config: &ExtractorConfig) -> Result<Box<dyn FeatureExtractor>> {
        if let Some(creator) = self.creators.get(adapter_type) {
            creator(config)
        } else {
            Err(Error::not_found(format!("未找到适配器类型: {}", adapter_type)))
        }
    }

    /// 获取所有已注册的适配器类型
    pub fn get_registered_types(&self) -> Vec<String> {
        self.creators.keys().cloned().collect()
    }
}

impl Default for FeatureExtractorAdapterFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// 文本特征提取器工厂适配器
/// 用于兼容旧的工厂接口
#[derive(Debug)]
pub struct TextFeatureExtractorFactoryAdapter {
    inner_factory: FeatureExtractorAdapterFactory,
}

impl TextFeatureExtractorFactoryAdapter {
    pub fn new() -> Self {
        Self {
            inner_factory: FeatureExtractorAdapterFactory::new(),
        }
    }

    /// 创建文本特征提取器
    pub fn create_text_extractor(&self, config: &crate::data::feature::FeatureExtractorConfig) -> Result<Box<dyn crate::data::feature::FeatureExtractor>> {
        // 转换配置格式
        let adapter_config = ExtractorConfig::new(ExtractorType::TextTfIdf, FeatureType::Text)
            .with_dimension(config.dimension)
            .with_option("text_extractor_type", "tfidf");

        // 创建适配器
        let adapter = TextFeatureExtractorAdapter::from_config(adapter_config)?;
        
        // 包装为旧接口
        Ok(Box::new(LegacyFeatureExtractorWrapper::new(Box::new(adapter))))
    }
}

/// 音频特征提取器工厂适配器
/// 用于兼容旧的工厂接口
#[derive(Debug)]
pub struct AudioFeatureExtractorFactoryAdapter {
    inner_factory: FeatureExtractorAdapterFactory,
}

impl AudioFeatureExtractorFactoryAdapter {
    pub fn new() -> Self {
        Self {
            inner_factory: FeatureExtractorAdapterFactory::new(),
        }
    }

    /// 创建音频特征提取器
    pub fn create_audio_extractor(&self, config: &crate::data::feature::FeatureExtractorConfig) -> Result<Box<dyn crate::data::feature::FeatureExtractor>> {
        // 转换配置格式
        let adapter_config = ExtractorConfig::new(ExtractorType::Audio, FeatureType::Audio)
            .with_dimension(config.dimension)
            .with_option("audio_extractor_type", "mfcc");

        // 创建适配器
        let adapter = AudioFeatureExtractorAdapter::from_config(adapter_config)?;
        
        // 包装为旧接口
        Ok(Box::new(LegacyFeatureExtractorWrapper::new(Box::new(adapter))))
    }
}

/// 旧接口包装器
/// 将新的异步特征提取器接口包装为旧的同步接口
#[derive(Debug)]
pub struct LegacyFeatureExtractorWrapper {
    inner: Box<dyn FeatureExtractor>,
    runtime: tokio::runtime::Runtime,
}

impl LegacyFeatureExtractorWrapper {
    pub fn new(inner: Box<dyn FeatureExtractor>) -> Self {
        let runtime = tokio::runtime::Runtime::new()
            .expect("Failed to create tokio runtime");
        
        Self {
            inner,
            runtime,
        }
    }
}

impl crate::data::feature::FeatureExtractor for LegacyFeatureExtractorWrapper {
    fn extract(&self, data: &[u8], metadata: Option<&HashMap<String, String>>) -> Result<crate::data::feature::FeatureVector> {
        // 将字节数据转换为字符串
        let input = String::from_utf8_lossy(data);
        
        // 使用运行时执行异步操作
        let result = self.runtime.block_on(async {
            self.inner.extract(&input).await
        })?;
        
        // 转换结果格式
        let legacy_vector = crate::data::feature::FeatureVector::new(
            result.feature_vector,
            &result.feature_type.to_string(),
        );
        
        Ok(legacy_vector)
    }

    fn dimension(&self) -> usize {
        self.inner.config().target_dimension.unwrap_or(512)
    }

    fn extractor_type(&self) -> &str {
        self.inner.name()
    }

    fn get_config(&self) -> crate::data::feature::FeatureExtractorConfig {
        let config = self.inner.get_config();
        crate::data::feature::FeatureExtractorConfig {
            extractor_type: config.extractor_type.to_string(),
            dimension: config.target_dimension.unwrap_or(512),
            params: HashMap::new(),
        }
    }
}

// ============================================================================
// 内部特征提取器trait定义
// ============================================================================

/// 文本特征提取器trait
pub trait TextFeatureExtractor: Send + Sync + std::fmt::Debug {
    /// 提取文本特征
    fn extract_text_features(&self, text: &str) -> Result<Vec<f32>>;
    
    /// 获取输出维度
    fn get_dimension(&self) -> usize;
    
    /// 获取提取器名称
    fn get_name(&self) -> &str;
}

/// 音频特征提取器trait
pub trait AudioFeatureExtractor: Send + Sync + std::fmt::Debug {
    /// 提取音频特征
    fn extract_audio_features(&self, audio_data: &[u8]) -> Result<Vec<f32>>;
    
    /// 获取输出维度
    fn get_dimension(&self) -> usize;
    
    /// 获取提取器名称
    fn get_name(&self) -> &str;
}

// ============================================================================
// 具体特征提取器实现
// ============================================================================

/// TF-IDF文本特征提取器
#[derive(Debug)]
pub struct TfIdfExtractor {
    dimension: usize,
    name: String,
    vocabulary: HashMap<String, usize>,
    idf_scores: HashMap<String, f32>,
}

impl TfIdfExtractor {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            name: "tfidf".to_string(),
            vocabulary: HashMap::new(),
            idf_scores: HashMap::new(),
        }
    }

    /// 构建词汇表
    fn build_vocabulary(&mut self, texts: &[&str]) {
        let mut word_counts = HashMap::new();
        let total_docs = texts.len() as f32;

        // 统计词频
        for text in texts {
            let words: std::collections::HashSet<String> = text
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();
            
            for word in words {
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        // 选择最频繁的词作为词汇表
        let mut word_freq: Vec<_> = word_counts.into_iter().collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));
        
        for (i, (word, count)) in word_freq.into_iter().take(self.dimension).enumerate() {
            self.vocabulary.insert(word.clone(), i);
            // 计算IDF分数
            let idf = (total_docs / count as f32).ln();
            self.idf_scores.insert(word, idf);
        }
    }

    /// 计算TF分数
    fn calculate_tf(&self, text: &str) -> HashMap<String, f32> {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        
        let total_words = words.len() as f32;
        let mut tf_scores = HashMap::new();
        
        for word in words {
            if self.vocabulary.contains_key(&word) {
                *tf_scores.entry(word).or_insert(0.0) += 1.0 / total_words;
            }
        }
        
        tf_scores
    }
}

impl TextFeatureExtractor for TfIdfExtractor {
    fn extract_text_features(&self, text: &str) -> Result<Vec<f32>> {
        let mut features = vec![0.0; self.dimension];
        
        // 计算TF分数
        let tf_scores = self.calculate_tf(text);
        
        // 计算TF-IDF分数
        for (word, tf) in tf_scores {
            if let (Some(&index), Some(&idf)) = (self.vocabulary.get(&word), self.idf_scores.get(&word)) {
                features[index] = tf * idf;
            }
        }
        
        Ok(features)
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Word2Vec文本特征提取器
#[derive(Debug)]
pub struct Word2VecExtractor {
    dimension: usize,
    name: String,
    word_vectors: HashMap<String, Vec<f32>>,
}

impl Word2VecExtractor {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            name: "word2vec".to_string(),
            word_vectors: HashMap::new(),
        }
    }

    /// 初始化随机词向量（实际应用中应该加载预训练模型）
    fn initialize_random_vectors(&mut self, vocabulary: &[String]) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for word in vocabulary {
            let vector: Vec<f32> = (0..self.dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            self.word_vectors.insert(word.clone(), vector);
        }
    }
}

impl TextFeatureExtractor for Word2VecExtractor {
    fn extract_text_features(&self, text: &str) -> Result<Vec<f32>> {
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        
        if words.is_empty() {
            return Ok(vec![0.0; self.dimension]);
        }
        
        let mut features = vec![0.0; self.dimension];
        let mut count = 0;
        
        // 计算词向量的平均值
        for word in words {
            if let Some(vector) = self.word_vectors.get(&word) {
                for (i, &value) in vector.iter().enumerate() {
                    features[i] += value;
                }
                count += 1;
            }
        }
        
        // 归一化
        if count > 0 {
            for feature in &mut features {
                *feature /= count as f32;
            }
        }
        
        Ok(features)
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// BERT文本特征提取器
#[derive(Debug)]
pub struct BertExtractor {
    dimension: usize,
    name: String,
}

impl BertExtractor {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            name: "bert".to_string(),
        }
    }

    /// 模拟BERT特征提取（实际应用中应该调用BERT模型）
    fn simulate_bert_extraction(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // 使用文本哈希生成确定性的"特征"
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        
        // 生成伪随机但确定性的特征向量
        let mut features = Vec::with_capacity(self.dimension);
        let mut seed = hash;
        
        for _ in 0..self.dimension {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((seed / 65536) % 32768) as f32 / 32768.0 - 0.5;
            features.push(value);
        }
        
        // 归一化
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for feature in &mut features {
                *feature /= norm;
            }
        }
        
        features
    }
}

impl TextFeatureExtractor for BertExtractor {
    fn extract_text_features(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.simulate_bert_extraction(text))
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// MFCC音频特征提取器
#[derive(Debug)]
pub struct MfccExtractor {
    dimension: usize,
    name: String,
}

impl MfccExtractor {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            name: "mfcc".to_string(),
        }
    }

    /// 模拟MFCC特征提取
    fn simulate_mfcc_extraction(&self, audio_data: &[u8]) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        // 使用音频数据哈希生成确定性的特征
        let mut hasher = DefaultHasher::new();
        audio_data.hash(&mut hasher);
        let hash = hasher.finish();
        
        let mut features = Vec::with_capacity(self.dimension);
        let mut seed = hash;
        
        for _ in 0..self.dimension {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((seed / 65536) % 32768) as f32 / 32768.0;
            features.push(value);
        }
        
        features
    }
}

impl AudioFeatureExtractor for MfccExtractor {
    fn extract_audio_features(&self, audio_data: &[u8]) -> Result<Vec<f32>> {
        Ok(self.simulate_mfcc_extraction(audio_data))
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// 频谱音频特征提取器
#[derive(Debug)]
pub struct SpectralExtractor {
    dimension: usize,
    name: String,
}

impl SpectralExtractor {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            name: "spectral".to_string(),
        }
    }
}

impl AudioFeatureExtractor for SpectralExtractor {
    fn extract_audio_features(&self, audio_data: &[u8]) -> Result<Vec<f32>> {
        // 模拟频谱特征提取
        let mut features = vec![0.0; self.dimension];
        
        // 简单的频谱分析模拟
        for (i, &byte) in audio_data.iter().take(self.dimension).enumerate() {
            features[i] = byte as f32 / 255.0;
        }
        
        Ok(features)
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// 色度音频特征提取器
#[derive(Debug)]
pub struct ChromaExtractor {
    dimension: usize,
    name: String,
}

impl ChromaExtractor {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            name: "chroma".to_string(),
        }
    }
}

impl AudioFeatureExtractor for ChromaExtractor {
    fn extract_audio_features(&self, audio_data: &[u8]) -> Result<Vec<f32>> {
        // 模拟色度特征提取
        let mut features = vec![0.0; self.dimension];
        
        // 简单的色度分析模拟
        for i in 0..self.dimension {
            let index = (i * audio_data.len()) / self.dimension;
            if index < audio_data.len() {
                features[i] = (audio_data[index] as f32 / 255.0).sin();
            }
        }
        
        Ok(features)
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// 模拟多模态特征提取器
#[derive(Debug)]
pub struct MockMultimodalExtractor {
    extractor_type: String,
    dimension: usize,
}

impl MockMultimodalExtractor {
    pub fn new(extractor_type: String, dimension: usize) -> Self {
        Self {
            extractor_type,
            dimension,
        }
    }
}

impl MultimodalFeatureExtractor for MockMultimodalExtractor {
    fn extract_features(&self, data: &[u8], _metadata: Option<&HashMap<String, String>>) -> Result<Vec<f32>> {
        // 模拟多模态特征提取
        let mut features = vec![0.0; self.dimension];
        
        for (i, &byte) in data.iter().take(self.dimension).enumerate() {
            features[i] = (byte as f32 / 255.0) * 2.0 - 1.0; // 归一化到[-1, 1]
        }
        
        Ok(features)
    }

    fn batch_extract(&self, data_batch: &[Vec<u8>], metadata_batch: Option<&[HashMap<String, String>]>) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(data_batch.len());
        
        for (i, data) in data_batch.iter().enumerate() {
            let metadata = metadata_batch.and_then(|batch| batch.get(i));
            let features = self.extract_features(data, metadata)?;
            results.push(features);
        }
        
        Ok(results)
    }

    fn get_output_dim(&self) -> usize {
        self.dimension
    }

    fn get_extractor_type(&self) -> String {
        self.extractor_type.clone()
    }
}

// ============================================================================
// 公共API函数
// ============================================================================

/// 创建文本特征提取器适配器
pub fn create_text_extractor_adapter(extractor: Arc<dyn TextFeatureExtractor>) -> Result<Box<dyn FeatureExtractor>> {
    let config = ExtractorConfig::new(ExtractorType::TextTfIdf, FeatureType::Text)
        .with_dimension(extractor.get_dimension());
    
    let adapter = TextFeatureExtractorAdapter::new(
        extractor,
        config,
        "text_adapter".to_string(),
    );
    
    Ok(Box::new(adapter))
}

/// 创建多模态特征提取器适配器
pub fn create_multimodal_extractor_adapter(
    extractor: Arc<dyn MultimodalFeatureExtractor>,
    modality_type: String
) -> Result<Box<dyn FeatureExtractor>> {
    let extractor_type = match modality_type.as_str() {
        "text" => ExtractorType::TextTfIdf,
        "image" => ExtractorType::ImageCNN,
        "audio" => ExtractorType::Audio,
        _ => ExtractorType::Generic(crate::data::feature::types::GenericExtractorType::Identity),
    };
    
    let feature_type = match modality_type.as_str() {
        "text" => FeatureType::Text,
        "image" => FeatureType::Image,
        "audio" => FeatureType::Audio,
        _ => FeatureType::Generic,
    };
    
    let config = ExtractorConfig::new(extractor_type, feature_type)
        .with_dimension(extractor.get_output_dim())
        .with_option("modality_type", &modality_type);
    
    let adapter = MultimodalFeatureExtractorAdapter::new(
        extractor,
        config,
        format!("multimodal_{}", modality_type),
    );
    
    Ok(Box::new(adapter))
}

/// 从文本特征提取器创建适配器
pub fn from_text_extractor(text_extractor: Arc<dyn TextFeatureExtractor>, dimension: usize) -> Box<dyn FeatureExtractor> {
    let config = ExtractorConfig::new(ExtractorType::TextTfIdf, FeatureType::Text)
        .with_dimension(dimension);
    
    let adapter = TextFeatureExtractorAdapter::new(
        text_extractor,
        config,
        "text_extractor_adapter".to_string(),
    );
    
    Box::new(adapter)
}

/// 从文本配置创建适配器
pub fn from_text_config(config: &serde_json::Value) -> Result<Box<dyn FeatureExtractor>> {
    let dimension = config.get("dimension")
        .and_then(|v| v.as_u64())
        .unwrap_or(300) as usize;
    
    let extractor_type = config.get("extractor_type")
        .and_then(|v| v.as_str())
        .unwrap_or("tfidf");
    
    let extractor_config = ExtractorConfig::new(ExtractorType::TextTfIdf, FeatureType::Text)
        .with_dimension(dimension)
        .with_option("text_extractor_type", extractor_type);
    
    TextFeatureExtractorAdapter::from_config(extractor_config)
        .map(|adapter| Box::new(adapter) as Box<dyn FeatureExtractor>)
}

/// 创建音频特征提取器适配器
pub fn create_audio_extractor_adapter(extractor: Arc<dyn AudioFeatureExtractor>) -> Result<Box<dyn FeatureExtractor>> {
    let config = ExtractorConfig::new(ExtractorType::Audio, FeatureType::Audio)
        .with_dimension(extractor.get_dimension());
    
    let adapter = AudioFeatureExtractorAdapter::new(
        extractor,
        config,
        "audio_adapter".to_string(),
    );
    
    Ok(Box::new(adapter))
}

/// 从音频配置创建适配器
pub fn from_audio_config(config: &serde_json::Value, dimension: usize) -> Result<Box<dyn FeatureExtractor>> {
    let extractor_type = config.get("extractor_type")
        .and_then(|v| v.as_str())
        .unwrap_or("mfcc");
    
    let extractor_config = ExtractorConfig::new(ExtractorType::Audio, FeatureType::Audio)
        .with_dimension(dimension)
        .with_option("audio_extractor_type", extractor_type);
    
    AudioFeatureExtractorAdapter::from_config(extractor_config)
        .map(|adapter| Box::new(adapter) as Box<dyn FeatureExtractor>)
}

/// 转换特征类型
pub fn convert_feature_type(model_feature_type: String) -> String {
    match model_feature_type.to_lowercase().as_str() {
        "text" => "text".to_string(),
        "image" => "image".to_string(),
        "audio" => "audio".to_string(),
        "video" => "video".to_string(),
        "numeric" => "numeric".to_string(),
        "categorical" => "categorical".to_string(),
        "multimodal" => "multimodal".to_string(),
        _ => "generic".to_string(),
    }
}

/// 高级特征向量转换器
/// 用于在不同特征向量格式之间进行转换
pub struct FeatureVectorConverter {
    /// 转换器配置
    config: HashMap<String, String>,
}

impl FeatureVectorConverter {
    /// 创建新的特征向量转换器
    pub fn new() -> Self {
        Self {
            config: HashMap::new(),
        }
    }

    /// 将InterfaceFeatureVector转换为FeatureVector
    pub fn convert_to_feature_vector(
        &self,
        interface_vector: FeatureVector,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<crate::data::feature::types::FeatureVector> {
        // 实现转换逻辑
        // 这里需要实际的转换逻辑，将interface的FeatureVector转换为types的FeatureVector
        // 暂时创建一个空的特征向量
        let mut result = crate::data::feature::types::FeatureVector::new();
        
        if let Some(meta) = metadata {
            for (key, value) in meta {
                result.metadata.insert(key, value);
            }
        }
        
        Ok(result)
    }

    /// 将FeatureExtractionResult转换为TensorData
    pub fn convert_to_tensor_data(
        &self,
        extraction_result: &FeatureExtractionResult,
    ) -> Result<TensorData> {
        use crate::compat::tensor::{TensorValues, DataType};
        
        let features = &extraction_result.features;
        let dimension = extraction_result.dimension;
        
        // 创建适当形状的张量数据
        let shape = vec![1, dimension]; // 批次大小为1，特征维度为dimension
        
        Ok(TensorData {
            data: TensorValues::F32(features.clone()),
            shape,
            dtype: DataType::Float32,
            metadata: extraction_result.metadata.clone(),
            version: 1,
        })
    }

    /// 批量转换特征提取结果为张量数据
    pub fn batch_convert_to_tensor_data(
        &self,
        extraction_results: &[FeatureExtractionResult],
    ) -> Result<TensorData> {
        if extraction_results.is_empty() {
            return Err(Error::invalid_input("Empty extraction results"));
        }

        let batch_size = extraction_results.len();
        let dimension = extraction_results[0].dimension;
        
        // 验证所有结果具有相同的维度
        for result in extraction_results {
            if result.dimension != dimension {
                return Err(Error::invalid_input(
                    format!("Inconsistent feature dimensions: expected {}, got {}", 
                            dimension, result.dimension)
                ));
            }
        }

        // 合并所有特征
        let mut data = Vec::with_capacity(batch_size * dimension);
        for result in extraction_results {
            data.extend_from_slice(&result.features);
        }

        let shape = vec![batch_size, dimension];
        
        use crate::compat::tensor::{TensorValues, DataType};
        
        // 合并所有元数据
        let mut metadata = HashMap::new();
        for result in extraction_results {
            metadata.extend(result.metadata.clone());
        }
        
        Ok(TensorData {
            data: TensorValues::F32(data),
            shape,
            dtype: DataType::Float32,
            metadata,
            version: 1,
        })
    }
}

/// 多模态特征提取器类型映射器
/// 用于在不同的多模态提取器类型之间进行映射
pub struct MultimodalExtractorTypeMapper {
    /// 类型映射表
    type_mappings: HashMap<String, MultiModalExtractorType>,
}

impl MultimodalExtractorTypeMapper {
    /// 创建新的类型映射器
    pub fn new() -> Self {
        let mut type_mappings = HashMap::new();
        
        // 建立默认的类型映射
        type_mappings.insert("fusion".to_string(), MultiModalExtractorType::Fusion);
        type_mappings.insert("multimodal_bert".to_string(), MultiModalExtractorType::MultiModalBERT);
        type_mappings.insert("multimodal_transformer".to_string(), MultiModalExtractorType::MultiModalTransformer);
        
        Self {
            type_mappings,
        }
    }

    /// 映射字符串类型到多模态提取器类型
    pub fn map_extractor_type(&self, type_str: &str) -> MultiModalExtractorType {
        self.type_mappings.get(type_str)
            .cloned()
            .unwrap_or_else(|| MultiModalExtractorType::Custom(type_str.to_string()))
    }

    /// 添加自定义类型映射
    pub fn add_mapping(&mut self, type_str: String, extractor_type: MultiModalExtractorType) {
        self.type_mappings.insert(type_str, extractor_type);
    }

    /// 映射多模态接口类型到内部类型
    pub fn map_interface_to_internal(interface_type: MultimodalInterfaceExtractorType) -> MultiModalExtractorType {
        match interface_type {
            MultimodalInterfaceExtractorType::Fusion => MultiModalExtractorType::Fusion,
            MultimodalInterfaceExtractorType::Text => MultiModalExtractorType::Custom("text".to_string()),
            MultimodalInterfaceExtractorType::Image => MultiModalExtractorType::Custom("image".to_string()),
            MultimodalInterfaceExtractorType::Audio => MultiModalExtractorType::Custom("audio".to_string()),
            MultimodalInterfaceExtractorType::Video => MultiModalExtractorType::Custom("video".to_string()),
        }
    }
}

/// 特征提取性能监控器
/// 用于监控特征提取过程的性能指标
pub struct FeatureExtractionMonitor {
    /// 提取时间记录
    extraction_times: Vec<Duration>,
    /// 错误计数
    error_count: usize,
    /// 成功计数
    success_count: usize,
}

impl FeatureExtractionMonitor {
    /// 创建新的性能监控器
    pub fn new() -> Self {
        Self {
            extraction_times: Vec::new(),
            error_count: 0,
            success_count: 0,
        }
    }

    /// 记录成功的特征提取
    pub fn record_success(&mut self, extraction_time: Duration) {
        self.extraction_times.push(extraction_time);
        self.success_count += 1;
    }

    /// 记录失败的特征提取
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// 获取平均提取时间
    pub fn average_extraction_time(&self) -> Option<Duration> {
        if self.extraction_times.is_empty() {
            None
        } else {
            let total: Duration = self.extraction_times.iter().sum();
            Some(total / self.extraction_times.len() as u32)
        }
    }

    /// 获取成功率
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count + self.error_count;
        if total == 0 {
            0.0
        } else {
            self.success_count as f64 / total as f64
        }
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("success_count".to_string(), self.success_count.to_string());
        stats.insert("error_count".to_string(), self.error_count.to_string());
        stats.insert("success_rate".to_string(), format!("{:.2}%", self.success_rate() * 100.0));
        
        if let Some(avg_time) = self.average_extraction_time() {
            stats.insert("average_extraction_time_ms".to_string(), avg_time.as_millis().to_string());
        }
        
        stats
    }
} 