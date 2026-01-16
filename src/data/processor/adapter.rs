// 处理器适配器 - 使旧系统processor能够适配到新的pipeline系统
// Processor Adapter - Adapts old processor system to new pipeline system

use crate::data::shared::processor_core::ProcessorStats;
use crate::data::shared::{SharedProcessorContext as ProcessorContext, SharedProcessorResult as ProcessorResult};
use crate::data::DataValue;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tempfile::tempdir;
use std::io::Write;
use std::fs;
use async_trait::async_trait;
use log::{debug, warn, info};
use super::DataProcessor;
use super::core::ProcessorType;

// 定义一个简单的Processor特性
pub trait Processor {
    fn process(&mut self, context: ProcessorContext) -> Result<ProcessorResult>;
    fn get_stats(&self) -> &ProcessorStats;
    fn reset_stats(&mut self) -> ();
}

/// 处理器适配器接口
/// 用于支持不同类型处理器的统一访问
#[async_trait]
pub trait ProcessorAdapter: Send + Sync {
    /// 获取适配器名称
    fn name(&self) -> &str;
    
    /// 获取支持的处理器类型
    fn supported_types(&self) -> Vec<ProcessorType>;
    
    /// 处理数据文件
    async fn process_file(&self, source_path: &str, output_path: &str, processor_type: &str, options: &HashMap<String, String>) -> Result<()>;
    
    /// 是否支持指定的处理器类型
    fn supports(&self, processor_type: &str) -> bool {
        self.supported_types().iter().any(|pt| match pt {
            ProcessorType::Custom(name) => name == processor_type,
            _ => pt.to_string() == processor_type,
        })
    }
}

/// 处理器类型映射
pub struct ProcessorTypeMapping {
    /// 类型映射关系
    pub mappings: HashMap<String, String>,
}

impl ProcessorTypeMapping {
    /// 创建新的类型映射
    pub fn new() -> Self {
        let mut mappings = HashMap::new();
        // 标准映射
        mappings.insert("normalize".to_string(), "normalize".to_string());
        mappings.insert("tokenize".to_string(), "tokenize".to_string());
        mappings.insert("encode".to_string(), "encode".to_string());
        mappings.insert("transform".to_string(), "transform".to_string());
        mappings.insert("filter".to_string(), "filter".to_string());
        mappings.insert("augment".to_string(), "augment".to_string());
        
        Self { mappings }
    }
    
    /// 添加映射
    pub fn add_mapping(&mut self, from: &str, to: &str) -> &mut Self {
        self.mappings.insert(from.to_string(), to.to_string());
        self
    }
    
    /// 获取映射
    pub fn get_mapping(&self, processor_type: &str) -> Option<&String> {
        self.mappings.get(processor_type)
    }
    
    /// 解析处理器类型
    pub fn resolve(&self, processor_type: &str) -> String {
        match self.get_mapping(processor_type) {
            Some(mapped) => mapped.clone(),
            None => processor_type.to_string(),
        }
    }
}

impl Default for ProcessorTypeMapping {
    fn default() -> Self {
        Self::new()
    }
}

/// 旧系统处理器的适配器
/// Adapter for legacy processor system
pub struct LegacyProcessorAdapter {
    /// 处理器名称
    name: String,
    /// 处理器类型
    processor_type: String,
    /// 处理器配置
    config: HashMap<String, String>,
    /// 处理器统计
    stats: ProcessorStats,
    /// 运行时，用于执行异步处理器函数
    runtime: Arc<Runtime>,
    /// 处理器实例
    processor: Arc<DataProcessor>,
    /// 类型映射
    type_mapping: ProcessorTypeMapping,
}

impl LegacyProcessorAdapter {
    /// 创建新的适配器
    pub fn new(name: &str, processor_type: &str, config: HashMap<String, String>) -> Self {
        let runtime = Arc::new(Runtime::new().expect("Failed to create Tokio runtime"));
        
        Self {
            name: name.to_string(),
            processor_type: processor_type.to_string(),
            config,
            stats: ProcessorStats::new(name),
            runtime,
            processor: Arc::new(DataProcessor::new().expect("Failed to create DataProcessor")),
            type_mapping: ProcessorTypeMapping::new(),
        }
    }
    
    /// 将ProcessorContext转换为临时文件
    fn context_to_temp_files(&self, context: &ProcessorContext) -> Result<(PathBuf, PathBuf)> {
        // 创建临时目录
        let temp_dir = tempdir().map_err(|e| 
            Error::IoError(format!("创建临时目录失败: {}", e)))?;
        
        // 创建输入目录
        let input_dir = temp_dir.path().join("input");
        fs::create_dir_all(&input_dir).map_err(|e| 
            Error::IoError(format!("创建输入目录失败: {}", e)))?;
        
        // 创建输出目录
        let output_dir = temp_dir.path().join("output");
        fs::create_dir_all(&output_dir).map_err(|e| 
            Error::IoError(format!("创建输出目录失败: {}", e)))?;
        
        // 将数据写入文件
        let data_file = input_dir.join("data.json");
        let data_json = serde_json::to_string(&context.data).map_err(|e| 
            Error::serialization(format!("数据序列化失败: {}", e)))?;
        
        let mut file = std::fs::File::create(&data_file).map_err(|e| 
            Error::IoError(format!("创建数据文件失败: {}", e)))?;
        file.write_all(data_json.as_bytes()).map_err(|e| 
            Error::IoError(format!("写入数据文件失败: {}", e)))?;
        
        // 将元数据写入文件
        let metadata_file = input_dir.join("metadata.json");
        let metadata_json = serde_json::to_string(&context.metadata).map_err(|e| 
            Error::serialization(format!("元数据序列化失败: {}", e)))?;
        
        let mut file = std::fs::File::create(&metadata_file).map_err(|e| 
            Error::IoError(format!("创建元数据文件失败: {}", e)))?;
        file.write_all(metadata_json.as_bytes()).map_err(|e| 
            Error::IoError(format!("写入元数据文件失败: {}", e)))?;
        
        debug!("上下文转换为临时文件: 输入={:?}, 输出={:?}", input_dir, output_dir);
        Ok((input_dir, output_dir))
    }
    
    /// 从临时文件读取处理结果
    fn read_result_from_temp_files(&self, output_dir: &Path) -> Result<ProcessorResult> {
        // 读取处理后的数据
        let data_file = output_dir.join("data.json");
        let data_json = fs::read_to_string(&data_file).map_err(|e| 
            Error::IoError(format!("读取数据文件失败: {}", e)))?;
        
        let data: HashMap<String, DataValue> = serde_json::from_str(&data_json).map_err(|e| 
            Error::serialization(format!("数据反序列化失败: {}", e)))?;
        
        // 读取处理后的元数据
        let metadata_file = output_dir.join("metadata.json");
        let metadata_json = fs::read_to_string(&metadata_file).map_err(|e| 
            Error::IoError(format!("读取元数据文件失败: {}", e)))?;
        
        let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json).map_err(|e| 
            Error::serialization(format!("元数据反序列化失败: {}", e)))?;
        
        debug!("从临时文件读取结果: 数据键数量={}, 元数据键数量={}", data.len(), metadata.len());
        Ok(ProcessorResult { data, metadata })
    }
}

impl Processor for LegacyProcessorAdapter {
    fn process(&mut self, context: ProcessorContext) -> Result<ProcessorResult> {
        let start = Instant::now();
        info!("开始处理: 类型={}, 名称={}", self.processor_type, self.name);
        
        // 根据processor_type调用不同的旧系统处理函数
        let result = match self.processor_type.as_str() {
            "normalize" => {
                // 将上下文转换为临时文件
                let (input_dir, output_dir) = self.context_to_temp_files(&context)?;
                
                // 在运行时执行normalize函数
                self.runtime.block_on(async {
                    crate::data::processor::processors::normalize(
                        input_dir.to_str().unwrap(),
                        output_dir.to_str().unwrap(),
                        &self.config
                    ).await
                })?;
                
                // 从输出目录读取结果
                self.read_result_from_temp_files(&output_dir)
            },
            "tokenize" => {
                // 类似normalize的处理模式
                let (input_dir, output_dir) = self.context_to_temp_files(&context)?;
                
                self.runtime.block_on(async {
                    crate::data::processor::processors::tokenize(
                        input_dir.to_str().unwrap(),
                        output_dir.to_str().unwrap(),
                        &self.config
                    ).await
                })?;
                
                self.read_result_from_temp_files(&output_dir)
            },
            "encode" => {
                let (input_dir, output_dir) = self.context_to_temp_files(&context)?;
                
                self.runtime.block_on(async {
                    crate::data::processor::processors::encode(
                        input_dir.to_str().unwrap(),
                        output_dir.to_str().unwrap(),
                        &self.config
                    ).await
                })?;
                
                self.read_result_from_temp_files(&output_dir)
            },
            "transform" => {
                let (input_dir, output_dir) = self.context_to_temp_files(&context)?;
                
                self.runtime.block_on(async {
                    crate::data::processor::processors::transform(
                        input_dir.to_str().unwrap(),
                        output_dir.to_str().unwrap(),
                        &self.config
                    ).await
                })?;
                
                self.read_result_from_temp_files(&output_dir)
            },
            "filter" => {
                let (input_dir, output_dir) = self.context_to_temp_files(&context)?;
                
                self.runtime.block_on(async {
                    crate::data::processor::processors::filter(
                        input_dir.to_str().unwrap(),
                        output_dir.to_str().unwrap(),
                        &self.config
                    ).await
                })?;
                
                self.read_result_from_temp_files(&output_dir)
            },
            "augment" => {
                let (input_dir, output_dir) = self.context_to_temp_files(&context)?;
                
                self.runtime.block_on(async {
                    crate::data::processor::processors::augment(
                        input_dir.to_str().unwrap(),
                        output_dir.to_str().unwrap(),
                        &self.config
                    ).await
                })?;
                
                self.read_result_from_temp_files(&output_dir)
            },
            "text_preprocessing" => {
                debug!("执行文本预处理: 配置={:?}", self.config);
                // 处理文本预处理请求
                let text = match context.data.get("text") {
                    Some(DataValue::String(t)) => t.clone(),
                    _ => return Err(Error::InvalidInput("文本预处理需要提供'text'字段".into())),
                };
                
                // 创建预处理器配置
                let mut preprocessing_config = crate::data::text_features::preprocessing::PreprocessingConfig::default();
                
                if let Some(lang) = self.config.get("language") {
                    preprocessing_config.language = lang.clone();
                }
                
                if let Some(value) = self.config.get("remove_stopwords") {
                    preprocessing_config.remove_stopwords = value.to_lowercase() == "true";
                }
                
                if let Some(value) = self.config.get("use_ngrams") {
                    preprocessing_config.use_ngrams = value.to_lowercase() == "true";
                }
                
                if let Some(value) = self.config.get("min_token_length") {
                    if let Ok(len) = value.parse::<usize>() {
                        preprocessing_config.min_token_length = len;
                    }
                }
                
                // 创建预处理器并处理文本
                let preprocessor = crate::data::text_features::preprocessing::preprocessor::create_default_preprocessor();
                let processed_text = preprocessor.process(&text)?;
                
                // 创建结果
                let mut data = HashMap::new();
                data.insert("processed_text".to_string(), DataValue::String(processed_text));
                
                let mut metadata = HashMap::new();
                metadata.insert("processor".to_string(), "text_preprocessing".to_string());
                metadata.insert("original_length".to_string(), text.len().to_string());
                metadata.insert("processed_length".to_string(), data.get("processed_text")
                    .and_then(|v| if let DataValue::String(s) = v { Some(s.len()) } else { None })
                    .unwrap_or(0).to_string());
                
                Ok(ProcessorResult { data, metadata })
            },
            "text_tokenization" => {
                debug!("执行文本分词: 配置={:?}", self.config);
                // 处理文本分词请求
                let text = match context.data.get("text") {
                    Some(DataValue::String(t)) => t.clone(),
                    _ => return Err(Error::InvalidInput("文本分词需要提供'text'字段".into())),
                };
                
                // 确定分词策略
                let tokenizer_type = self.config.get("tokenizer_type")
                    .map(|s| s.as_str())
                    .unwrap_or("whitespace");
                
                // 创建分词器并处理文本
                let tokenizer = crate::data::text_features::preprocessing::create_tokenizer(tokenizer_type);
                let tokens = tokenizer.tokenize(&text)?;
                
                // 创建结果
                let mut data = HashMap::new();
                data.insert("tokens".to_string(), DataValue::StringArray(tokens.clone()));
                
                let mut metadata = HashMap::new();
                metadata.insert("processor".to_string(), "text_tokenization".to_string());
                metadata.insert("tokenizer_type".to_string(), tokenizer_type.to_string());
                metadata.insert("token_count".to_string(), tokens.len().to_string());
                
                Ok(ProcessorResult { data, metadata })
            },
            _ => {
                warn!("未知的处理器类型: {}", self.processor_type);
                return Err(Error::InvalidProcessorType(
                    format!("未知的处理器类型: {}", self.processor_type)
                ));
            }
        };
        
        // 更新统计信息
        let elapsed = start.elapsed();
        self.stats.processed_count += 1;
        self.stats.processing_time += elapsed;
        
        info!("处理完成: 类型={}, 名称={}, 耗时={:?}", 
              self.processor_type, self.name, elapsed);
        
        result
    }
    
    fn get_stats(&self) -> &ProcessorStats {
        &self.stats
    }
    
    fn reset_stats(&mut self) {
        info!("重置处理器统计: 类型={}, 名称={}", self.processor_type, self.name);
        self.stats = ProcessorStats::new(&self.name);
    }
}

#[async_trait]
impl ProcessorAdapter for LegacyProcessorAdapter {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn supported_types(&self) -> Vec<ProcessorType> {
        vec![
            ProcessorType::File,
            ProcessorType::Custom("normalize".to_string()),
            ProcessorType::Custom("tokenize".to_string()),
            ProcessorType::Custom("encode".to_string()),
            ProcessorType::Custom("transform".to_string()),
            ProcessorType::Custom("filter".to_string()),
            ProcessorType::Custom("augment".to_string()),
        ]
    }
    
    async fn process_file(&self, source_path: &str, output_path: &str, processor_type: &str, options: &HashMap<String, String>) -> Result<()> {
        debug!("使用旧版处理器适配器处理文件: {}", source_path);
        
        // 解析处理器类型
        let resolved_type = self.type_mapping.resolve(processor_type);
        info!("处理器类型解析: {} -> {}", processor_type, resolved_type);
        
        // 检查是否支持该类型
        if !self.supports(&resolved_type) {
            return Err(Error::invalid_argument(format!("不支持的处理器类型: {}", resolved_type)));
        }
        
        // 检查源路径
        let source = Path::new(source_path);
        if !source.exists() {
            return Err(Error::not_found(format!("源文件不存在: {}", source_path)));
        }
        
        // 检查输出目录，不存在则创建
        let output = Path::new(output_path);
        if !output.exists() {
            fs::create_dir_all(output).await
                .map_err(|e| Error::io_error(format!("无法创建输出目录: {}", e)))?;
        }
        
        // 根据处理器类型调用相应方法
        match resolved_type.as_str() {
            "normalize" => {
                debug!("调用normalize处理器");
                self.processor.normalize(source_path, output_path, options).await?;
            },
            "tokenize" => {
                debug!("调用tokenize处理器");
                self.processor.tokenize(source_path, output_path, options).await?;
            },
            "encode" => {
                debug!("调用encode处理器");
                self.processor.encode(source_path, output_path, options).await?;
            },
            "transform" => {
                debug!("调用transform处理器");
                self.processor.transform(source_path, output_path, options).await?;
            },
            "filter" => {
                debug!("调用filter处理器");
                self.processor.filter(source_path, output_path, options).await?;
            },
            "augment" => {
                debug!("调用augment处理器");
                self.processor.augment(source_path, output_path, options).await?;
            },
            _ => {
                return Err(Error::invalid_argument(format!("未实现的处理器类型: {}", resolved_type)));
            }
        }
        
        info!("处理完成: {} -> {}", source_path, output_path);
        Ok(())
    }
} 