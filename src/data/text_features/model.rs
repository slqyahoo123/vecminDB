// src/data/text_features/model.rs
//
// Transformer 核心模型模块

use std::collections::HashMap;
use log::info;
use super::error::TransformerError;
use super::config::TransformerConfig;
use super::encoder::Encoder;
use super::encoder::EncoderConfig;
use super::tokenizer::Tokenizer;
use super::features::FeatureExtractor;
use super::features::FeatureExtractionConfig;
use super::similarity::SimilarityCalculator;
use super::similarity::SimilarityConfig;
use super::language::LanguageProcessor;
use super::language::LanguageProcessingConfig;

/// Transformer模型
#[derive(Debug)]
pub struct TransformerModel {
    /// 模型配置
    config: TransformerConfig,
    /// 编码器
    encoder: Encoder,
    /// 分词器
    tokenizer: Tokenizer,
    /// 特征提取器
    feature_extractor: FeatureExtractor,
    /// 相似度计算器
    similarity_calculator: SimilarityCalculator,
    /// 语言处理器
    language_processor: LanguageProcessor,
    /// 模型状态
    state: ModelState,
    /// 模型元数据
    metadata: HashMap<String, String>,
}

/// 模型状态
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelState {
    /// 是否已初始化
    pub is_initialized: bool,
    /// 是否正在训练
    pub is_training: bool,
    /// 训练轮数
    pub training_epochs: usize,
    /// 当前损失
    pub current_loss: f32,
    /// 最佳损失
    pub best_loss: f32,
    /// 训练步数
    pub training_steps: usize,
    /// 验证准确率
    pub validation_accuracy: f32,
    /// 模型版本
    pub version: String,
    /// 最后更新时间
    pub last_updated: std::time::SystemTime,
}

impl Default for ModelState {
    fn default() -> Self {
        Self {
            is_initialized: false,
            is_training: false,
            training_epochs: 0,
            current_loss: f32::INFINITY,
            best_loss: f32::INFINITY,
            training_steps: 0,
            validation_accuracy: 0.0,
            version: "1.0.0".to_string(),
            last_updated: std::time::SystemTime::now(),
        }
    }
}

impl TransformerModel {
    /// 创建新的Transformer模型
    pub fn new(config: TransformerConfig) -> Result<Self, TransformerError> {
        // 初始化分词器
        let mut tokenizer = Tokenizer::new(config.clone());
        tokenizer.initialize()?;
        
        // 初始化编码器
        let encoder_config = EncoderConfig::default();
        let encoder = Encoder::new(encoder_config, tokenizer.clone(), &config);
        
        // 初始化特征提取器
        let feature_config = FeatureExtractionConfig::default();
        let feature_extractor = FeatureExtractor::new(feature_config);
        
        // 初始化相似度计算器
        let similarity_config = SimilarityConfig::default();
        let similarity_calculator = SimilarityCalculator::new(similarity_config);
        
        // 初始化语言处理器
        let language_config = LanguageProcessingConfig::default();
        let language_processor = LanguageProcessor::new(language_config);
        
        let mut model = Self {
            config,
            encoder,
            tokenizer,
            feature_extractor,
            similarity_calculator,
            language_processor,
            state: ModelState::default(),
            metadata: HashMap::new(),
        };
        
        // 初始化模型
        model.initialize()?;
        
        Ok(model)
    }
    
    /// 初始化模型
    fn initialize(&mut self) -> Result<(), TransformerError> {
        info!("初始化Transformer模型...");
        
        // 验证配置
        self.validate_config()?;
        
        // 初始化分词器
        self.tokenizer.initialize()?;
        
        // 设置编码器
        self.feature_extractor.set_encoder(self.encoder.clone());
        
        // 更新状态
        self.state.is_initialized = true;
        self.state.last_updated = std::time::SystemTime::now();
        
        info!("Transformer模型初始化完成");
        Ok(())
    }

    /// 允许外部注入编码器（用于训练/推理快速接线）
    pub fn set_encoder(&mut self, encoder: Encoder) {
        self.encoder = encoder.clone();
        self.feature_extractor.set_encoder(encoder);
    }
    
    /// 验证配置
    fn validate_config(&self) -> Result<(), TransformerError> {
        if self.config.hidden_size == 0 {
            return Err(TransformerError::config_error("隐藏层大小不能为0"));
        }
        
        if self.config.num_heads == 0 {
            return Err(TransformerError::config_error("注意力头数不能为0"));
        }
        
        if self.config.num_layers == 0 {
            return Err(TransformerError::config_error("层数不能为0"));
        }
        
        if self.config.max_seq_length == 0 {
            return Err(TransformerError::config_error("最大序列长度不能为0"));
        }
        
        if self.config.vocab_size == 0 {
            return Err(TransformerError::config_error("词汇表大小不能为0"));
        }
        
        // 检查隐藏层大小是否能被注意力头数整除
        if self.config.hidden_size % self.config.num_heads != 0 {
            return Err(TransformerError::config_error(
                "隐藏层大小必须能被注意力头数整除"
            ));
        }
        
        Ok(())
    }
    
    /// 处理文本
    pub fn process_text(&self, text: &str) -> Result<ProcessedText, TransformerError> {
        if !self.state.is_initialized {
            return Err(TransformerError::config_error("模型未初始化"));
        }
        
        // 语言处理
        let processed_text = self.language_processor.process_text(text)?;
        
        // 特征提取
        let features = self.feature_extractor.extract_features(&processed_text.processed_text)?;
        
        // 编码
        let encoded = self.encoder.encode(&processed_text.processed_text)?;
        
        Ok(ProcessedText {
            original_text: text.to_string(),
            processed_text: processed_text.processed_text,
            features,
            encoded: encoded,
            metadata: processed_text.metadata,
        })
    }
    
    /// 计算文本相似度
    pub fn compute_similarity(&mut self, text1: &str, text2: &str) -> Result<f32, TransformerError> {
        let processed1 = self.process_text(text1)?;
        let processed2 = self.process_text(text2)?;
        
        self.similarity_calculator.calculate_similarity(&processed1.features, &processed2.features)
    }
    
    /// 批量处理文本
    pub fn process_batch(&self, texts: &[String]) -> Result<Vec<ProcessedText>, TransformerError> {
        let mut results = Vec::with_capacity(texts.len());
        
        for text in texts {
            let processed = self.process_text(text)?;
            results.push(processed);
        }
        
        Ok(results)
    }
    
    // Training methods removed: vector database does not need training functionality
    
    /// 预测
    fn predict(&self, encoded: &[f32]) -> Result<Vec<f32>, TransformerError> {
        // 简化的预测实现
        // 在实际应用中，这里应该使用训练好的模型参数
        Ok(encoded.to_vec())
    }
    
    // Training-related methods (calculate_loss, update_parameters) removed: vector database does not need training functionality
    
    /// 保存模型
    pub fn save_model(&self, path: &str) -> Result<(), TransformerError> {
        use std::fs;
        use serde_json;
        
        let model_data = ModelData {
            config: self.config.clone(),
            state: self.state.clone(),
            metadata: self.metadata.clone(),
        };
        
        let json = serde_json::to_string_pretty(&model_data)
            .map_err(|e| TransformerError::serialization_error(e.to_string()))?;
        
        fs::write(path, json)
            .map_err(|e| TransformerError::IoError(e))?;
        
        info!("模型已保存到: {}", path);
        Ok(())
    }
    
    /// 加载模型
    pub fn load_model(path: &str) -> Result<Self, TransformerError> {
        use std::fs;
        use serde_json;
        
        let json = fs::read_to_string(path)
            .map_err(|e| TransformerError::IoError(e))?;
        
        let model_data: ModelData = serde_json::from_str(&json)
            .map_err(|e| TransformerError::serialization_error(e.to_string()))?;
        
        let mut model = Self::new(model_data.config)?;
        model.state = model_data.state;
        model.metadata = model_data.metadata;
        
        info!("模型已从 {} 加载", path);
        Ok(model)
    }
    
    /// 获取模型状态
    pub fn get_state(&self) -> &ModelState {
        &self.state
    }
    
    /// 获取配置
    pub fn get_config(&self) -> &TransformerConfig {
        &self.config
    }
    
    /// 获取分词器
    pub fn get_tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    
    /// 获取编码器
    pub fn get_encoder(&self) -> &Encoder {
        &self.encoder
    }
    
    /// 设置元数据
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// 重置模型
    pub fn reset(&mut self) -> Result<(), TransformerError> {
        self.state = ModelState::default();
        self.metadata.clear();
        self.initialize()?;
        
        info!("模型已重置");
        Ok(())
    }
}

/// 处理后的文本
#[derive(Debug, Clone)]
pub struct ProcessedText {
    /// 原始文本
    pub original_text: String,
    /// 处理后的文本
    pub processed_text: String,
    /// 提取的特征
    pub features: super::features::FeatureVector,
    /// 编码后的向量
    pub encoded: Vec<f32>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

// Training types (TrainingExample, TrainingResult) moved to compat.rs for backward compatibility
// Vector database does not need training functionality

/// 模型数据（用于序列化）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ModelData {
    /// 配置
    config: TransformerConfig,
    /// 状态
    state: ModelState,
    /// 元数据
    metadata: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transformer_model_creation() {
        let config = TransformerConfig::default();
        let model = TransformerModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_text_processing() {
        let config = TransformerConfig::default();
        let model = TransformerModel::new(config).unwrap();
        
        let result = model.process_text("Hello world");
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert_eq!(processed.original_text, "Hello world");
        assert!(!processed.processed_text.is_empty());
    }

    #[test]
    fn test_similarity_computation() {
        let config = TransformerConfig::default();
        let mut model = TransformerModel::new(config).unwrap();
        
        let similarity = model.compute_similarity("Hello world", "Hello universe");
        assert!(similarity.is_ok());
        
        let sim_value = similarity.unwrap();
        assert!(sim_value >= 0.0 && sim_value <= 1.0);
    }

    #[test]
    fn test_model_save_load() {
        let config = TransformerConfig::default();
        let model = TransformerModel::new(config).unwrap();
        
        let save_path = "test_model.json";
        let save_result = model.save_model(save_path);
        assert!(save_result.is_ok());
        
        let load_result = TransformerModel::load_model(save_path);
        assert!(load_result.is_ok());
        
        // 清理测试文件
        let _ = std::fs::remove_file(save_path);
    }
} 