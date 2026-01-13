// 高级文本处理管道模块
// 提供组合式文本处理流程，支持多级处理、条件处理和并行处理

use crate::{Error, Result};
use crate::data::text_features::preprocessing::TextPreprocessor;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use serde::{Serialize, Deserialize};
// regex::Regex is used in RegexCondition below
use crate::data::text_features::preprocessing::cleaner::{TextCleaner, HtmlCleaner, UrlCleaner, StandardTextCleaner};
use crate::data::text_features::preprocessing::normalizer::{TextNormalizer};
use crate::data::text_features::normalizers::{TextCaseNormalizer, AccentNormalizer};
use crate::data::text_features::preprocessing::tokenizer::{Tokenizer, WhitespaceTokenizer};
use crate::data::text_features::preprocessing::filter::{TextFilter, StopwordFilter, LengthFilter};
// logging macros can be wired later; not used in current pipeline implementation

// 导入处理器实现
// keep focused imports; wildcards below were unused in this module

/// 处理阶段类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingStageType {
    /// 清洗 - 去除不需要的内容
    Cleaning,
    /// 规范化 - 统一格式
    Normalization,
    /// 分词 - 文本切分
    Tokenization,
    /// 过滤 - 内容筛选
    Filtering,
    /// 转换 - 内容变换
    Transformation,
    /// 增强 - 内容扩充
    Augmentation,
    /// 分析 - 提取元数据
    Analysis,
    /// 自定义处理
    Custom,
}

/// 处理器接口
pub trait TextProcessor: Send + Sync {
    /// 处理文本
    fn process(&self, text: &str) -> Result<String>;
    
    /// 处理文本并返回附加信息
    fn process_with_metadata(&self, text: &str) -> Result<(String, HashMap<String, String>)> {
        let processed = self.process(text)?;
        Ok((processed, HashMap::new()))
    }
    
    /// 获取处理器名称
    fn name(&self) -> &str;
    
    /// 获取处理器类型
    fn processor_type(&self) -> ProcessingStageType;
    
    /// 克隆处理器
    fn box_clone(&self) -> Box<dyn TextProcessor>;
}

impl Clone for Box<dyn TextProcessor> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// 适配器：将 TextCleaner 转换为 TextProcessor
struct TextCleanerAdapter {
    cleaner: Box<dyn TextCleaner>,
    name: String,
}

impl TextProcessor for TextCleanerAdapter {
    fn process(&self, text: &str) -> Result<String> {
        self.cleaner.clean(text)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Cleaning
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        // 由于 TextCleaner 可能不支持克隆，我们创建一个新的适配器
        // 这里假设 cleaner 可以通过其他方式重新创建
        Box::new(TextCleanerAdapter {
            cleaner: Box::new(StandardTextCleaner::new()),
            name: self.name.clone(),
        })
    }
}

/// 适配器：将 TextNormalizer 转换为 TextProcessor
struct TextNormalizerAdapter {
    normalizer: Box<dyn TextNormalizer>,
    name: String,
}

impl TextProcessor for TextNormalizerAdapter {
    fn process(&self, text: &str) -> Result<String> {
        self.normalizer.normalize(text)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Normalization
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(TextNormalizerAdapter {
            normalizer: Box::new(crate::data::text_features::preprocessing::normalizer::StandardTextNormalizer::new()),
            name: self.name.clone(),
        })
    }
}

/// 适配器：将 TextFilter 转换为 TextProcessor
struct TextFilterAdapter {
    filter: Box<dyn TextFilter>,
    name: String,
}

impl TextProcessor for TextFilterAdapter {
    fn process(&self, text: &str) -> Result<String> {
        let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let filtered = self.filter.filter(&tokens)?;
        Ok(filtered.join(" "))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Filtering
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(TextFilterAdapter {
            filter: Box::new(StopwordFilter::new("en".to_string())),
            name: self.name.clone(),
        })
    }
}

/// 适配器：将 Tokenizer 转换为 TextProcessor
struct TextTokenizerAdapter {
    tokenizer: Box<dyn Tokenizer>,
    name: String,
}

impl TextProcessor for TextTokenizerAdapter {
    fn process(&self, text: &str) -> Result<String> {
        let tokens = self.tokenizer.tokenize(text)?;
        Ok(tokens.join(" "))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Tokenization
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(TextTokenizerAdapter {
            tokenizer: Box::new(WhitespaceTokenizer::new()),
            name: self.name.clone(),
        })
    }
}

/// 条件评估器 - 用于条件处理
pub trait ConditionEvaluator: Send + Sync {
    /// 评估条件
    fn evaluate(&self, text: &str, metadata: &HashMap<String, String>) -> Result<bool>;
    
    /// 获取评估器名称
    fn name(&self) -> &str;
    
    /// 克隆评估器
    fn box_clone(&self) -> Box<dyn ConditionEvaluator>;
}

impl Clone for Box<dyn ConditionEvaluator> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// 长度条件评估器
pub struct LengthCondition {
    /// 最小长度
    min_length: Option<usize>,
    /// 最大长度
    max_length: Option<usize>,
    /// 名称
    name: String,
}

impl LengthCondition {
    /// 创建新的长度条件评估器
    pub fn new(min_length: Option<usize>, max_length: Option<usize>, name: Option<String>) -> Self {
        Self {
            min_length,
            max_length,
            name: name.unwrap_or_else(|| "LengthCondition".to_string()),
        }
    }
}

impl ConditionEvaluator for LengthCondition {
    fn evaluate(&self, text: &str, _metadata: &HashMap<String, String>) -> Result<bool> {
        let length = text.len();
        
        let min_check = if let Some(min) = self.min_length {
            length >= min
        } else {
            true
        };
        
        let max_check = if let Some(max) = self.max_length {
            length <= max
        } else {
            true
        };
        
        Ok(min_check && max_check)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn box_clone(&self) -> Box<dyn ConditionEvaluator> {
        Box::new(Self {
            min_length: self.min_length,
            max_length: self.max_length,
            name: self.name.clone(),
        })
    }
}

/// 正则表达式条件评估器
pub struct RegexCondition {
    /// 正则表达式
    regex: regex::Regex,
    /// 名称
    name: String,
    /// 是否匹配(true)或不匹配(false)时满足条件
    match_for_true: bool,
}

impl RegexCondition {
    /// 创建新的正则表达式条件评估器
    pub fn new(pattern: &str, match_for_true: bool, name: Option<String>) -> Result<Self> {
        let regex = regex::Regex::new(pattern)
            .map_err(|e| Error::InvalidData(format!("无效的正则表达式: {}", e)))?;
        
        Ok(Self {
            regex,
            name: name.unwrap_or_else(|| "RegexCondition".to_string()),
            match_for_true,
        })
    }
}

impl ConditionEvaluator for RegexCondition {
    fn evaluate(&self, text: &str, _metadata: &HashMap<String, String>) -> Result<bool> {
        let matches = self.regex.is_match(text);
        Ok(if self.match_for_true { matches } else { !matches })
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn box_clone(&self) -> Box<dyn ConditionEvaluator> {
        let pattern = self.regex.as_str();
        Box::new(Self::new(pattern, self.match_for_true, Some(self.name.clone())).unwrap())
    }
}

/// 元数据条件评估器
pub struct MetadataCondition {
    /// 键
    key: String,
    /// 值模式
    value_pattern: regex::Regex,
    /// 名称
    name: String,
    /// 是否匹配(true)或不匹配(false)时满足条件
    match_for_true: bool,
}

impl MetadataCondition {
    /// 创建新的元数据条件评估器
    pub fn new(key: &str, value_pattern: &str, match_for_true: bool, name: Option<String>) -> Result<Self> {
        let regex = regex::Regex::new(value_pattern)
            .map_err(|e| Error::InvalidData(format!("无效的正则表达式: {}", e)))?;
        
        Ok(Self {
            key: key.to_string(),
            value_pattern: regex,
            name: name.unwrap_or_else(|| "MetadataCondition".to_string()),
            match_for_true,
        })
    }
}

impl ConditionEvaluator for MetadataCondition {
    fn evaluate(&self, _text: &str, metadata: &HashMap<String, String>) -> Result<bool> {
        if let Some(value) = metadata.get(&self.key) {
            let matches = self.value_pattern.is_match(value);
            Ok(if self.match_for_true { matches } else { !matches })
        } else {
            // 如果键不存在，则不满足条件
            Ok(false)
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn box_clone(&self) -> Box<dyn ConditionEvaluator> {
        let pattern = self.value_pattern.as_str();
        Box::new(Self::new(&self.key, pattern, self.match_for_true, Some(self.name.clone())).unwrap())
    }
}

/// 组合条件评估器
pub struct CombinedCondition {
    /// 条件列表
    conditions: Vec<Box<dyn ConditionEvaluator>>,
    /// 组合方式
    combination_type: CombinationType,
    /// 名称
    name: String,
}

/// 条件组合方式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinationType {
    /// 所有条件都满足(AND)
    All,
    /// 任一条件满足(OR)
    Any,
    /// 条件都不满足(NOR)
    None,
}

impl CombinedCondition {
    /// 创建新的组合条件评估器
    pub fn new(conditions: Vec<Box<dyn ConditionEvaluator>>, combination_type: CombinationType, name: Option<String>) -> Self {
        Self {
            conditions,
            combination_type,
            name: name.unwrap_or_else(|| "CombinedCondition".to_string()),
        }
    }
}

impl ConditionEvaluator for CombinedCondition {
    fn evaluate(&self, text: &str, metadata: &HashMap<String, String>) -> Result<bool> {
        match self.combination_type {
            CombinationType::All => {
                for condition in &self.conditions {
                    if !condition.evaluate(text, metadata)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
            CombinationType::Any => {
                for condition in &self.conditions {
                    if condition.evaluate(text, metadata)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
            CombinationType::None => {
                for condition in &self.conditions {
                    if condition.evaluate(text, metadata)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            },
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn box_clone(&self) -> Box<dyn ConditionEvaluator> {
        let cloned_conditions = self.conditions.iter()
            .map(|c| c.box_clone())
            .collect();
        Box::new(Self::new(cloned_conditions, self.combination_type, Some(self.name.clone())))
    }
}

/// 处理阶段配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStageConfig {
    /// 阶段类型
    pub stage_type: ProcessingStageType,
    /// 阶段名称
    pub name: String,
    /// 是否启用
    pub enabled: bool,
    /// 处理器配置
    pub processor_config: HashMap<String, String>,
    /// 条件处理
    pub condition: Option<String>,
}

/// 处理阶段结果
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// 处理后的文本
    pub text: String,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 日志
    pub logs: Vec<String>,
    /// 是否成功处理
    pub success: bool,
    /// 错误信息
    pub error: Option<String>,
}

impl ProcessingResult {
    /// 创建新的成功结果
    pub fn success(text: String, metadata: HashMap<String, String>) -> Self {
        Self {
            text,
            metadata,
            logs: Vec::new(),
            success: true,
            error: None,
        }
    }
    
    /// 创建新的失败结果
    pub fn failure(original_text: &str, error: String) -> Self {
        let mut result = Self {
            text: original_text.to_string(),
            metadata: HashMap::new(),
            logs: Vec::new(),
            success: false,
            error: Some(error),
        };
        result.log("处理失败");
        result
    }
    
    /// 添加日志
    pub fn log(&mut self, message: &str) {
        self.logs.push(message.to_string());
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

/// 处理管道
pub struct TextProcessingPipeline {
    /// 处理阶段
    stages: Vec<ProcessingStage>,
    /// 名称
    name: String,
    /// 是否在错误时继续处理
    continue_on_error: bool,
    /// 是否收集详细日志
    verbose_logging: bool,
}

impl TextProcessingPipeline {
    /// 创建新的处理管道
    pub fn new(name: &str, continue_on_error: bool, verbose_logging: bool) -> Self {
        Self {
            stages: Vec::new(),
            name: name.to_string(),
            continue_on_error,
            verbose_logging,
        }
    }
    
    /// 添加处理阶段
    pub fn add_stage(&mut self, stage: ProcessingStage) {
        self.stages.push(stage);
    }
    
    /// 从配置构建处理管道
    pub fn from_configs(configs: Vec<ProcessingStageConfig>, name: &str, continue_on_error: bool, verbose_logging: bool) -> Result<Self> {
        let mut pipeline = Self::new(name, continue_on_error, verbose_logging);
        
        for config in configs {
            if !config.enabled {
                continue;
            }
            
            let processor = create_processor_from_config(&config)?;
            
            let condition = if let Some(condition_str) = config.condition {
                Some(parse_condition_expression(&condition_str)?)
            } else {
                None
            };
            
            let stage = ProcessingStage {
                processor,
                condition,
                name: config.name,
                enabled: true,
            };
            
            pipeline.add_stage(stage);
        }
        
        Ok(pipeline)
    }
    
    /// 处理文本
    pub fn process(&self, text: &str) -> Result<String> {
        let result = self.process_with_details(text)?;
        Ok(result.text)
    }
    
    /// 处理文本并返回详细结果
    pub fn process_with_details(&self, text: &str) -> Result<ProcessingResult> {
        let mut current_text = text.to_string();
        let mut metadata = HashMap::new();
        let mut logs = Vec::new();
        
        for (i, stage) in self.stages.iter().enumerate() {
            if !stage.enabled {
                if self.verbose_logging {
                    logs.push(format!("阶段 {} ({}) 已禁用，跳过", i, stage.name));
                }
                continue;
            }
            
            // 检查条件
            if let Some(ref condition) = stage.condition {
                match condition.evaluate(&current_text, &metadata) {
                    Ok(true) => {
                        if self.verbose_logging {
                            logs.push(format!("阶段 {} ({}) 条件满足，继续执行", i, stage.name));
                        }
                    },
                    Ok(false) => {
                        if self.verbose_logging {
                            logs.push(format!("阶段 {} ({}) 条件不满足，跳过", i, stage.name));
                        }
                        continue;
                    },
                    Err(e) => {
                        let error_msg = format!("阶段 {} ({}) 条件评估失败: {}", i, stage.name, e);
                        logs.push(error_msg.clone());
                        
                        if !self.continue_on_error {
                            return Ok(ProcessingResult {
                                text: current_text,
                                metadata,
                                logs,
                                success: false,
                                error: Some(error_msg),
                            });
                        }
                        
                        continue;
                    }
                }
            }
            
            // 执行处理
            match stage.processor.process_with_metadata(&current_text) {
                Ok((processed, stage_metadata)) => {
                    if self.verbose_logging {
                        logs.push(format!("阶段 {} ({}) 处理成功", i, stage.name));
                    }
                    
                    current_text = processed;
                    
                    // 合并元数据
                    for (key, value) in stage_metadata {
                        metadata.insert(key, value);
                    }
                },
                Err(e) => {
                    let error_msg = format!("阶段 {} ({}) 处理失败: {}", i, stage.name, e);
                    logs.push(error_msg.clone());
                    
                    if !self.continue_on_error {
                        return Ok(ProcessingResult {
                            text: current_text,
                            metadata,
                            logs,
                            success: false,
                            error: Some(error_msg),
                        });
                    }
                }
            }
        }
        
        Ok(ProcessingResult {
            text: current_text,
            metadata,
            logs,
            success: true,
            error: None,
        })
    }
    
    /// 获取处理管道名称
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// 获取处理阶段数量
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
    
    /// 获取处理阶段引用
    pub fn get_stage(&self, index: usize) -> Option<&ProcessingStage> {
        self.stages.get(index)
    }
    
    /// 获取可变处理阶段引用
    pub fn get_stage_mut(&mut self, index: usize) -> Option<&mut ProcessingStage> {
        self.stages.get_mut(index)
    }
    
    /// 启用阶段
    pub fn enable_stage(&mut self, index: usize) -> Result<()> {
        if let Some(stage) = self.stages.get_mut(index) {
            stage.enabled = true;
            Ok(())
        } else {
            Err(Error::InvalidData(format!("阶段索引 {} 不存在", index)))
        }
    }
    
    /// 禁用阶段
    pub fn disable_stage(&mut self, index: usize) -> Result<()> {
        if let Some(stage) = self.stages.get_mut(index) {
            stage.enabled = false;
            Ok(())
        } else {
            Err(Error::InvalidData(format!("阶段索引 {} 不存在", index)))
        }
    }
    
    /// 获取所有阶段
    pub fn stages(&self) -> &[ProcessingStage] {
        &self.stages
    }
    
    /// 设置是否在错误时继续处理
    pub fn set_continue_on_error(&mut self, continue_on_error: bool) {
        self.continue_on_error = continue_on_error;
    }
    
    /// 设置是否收集详细日志
    pub fn set_verbose_logging(&mut self, verbose_logging: bool) {
        self.verbose_logging = verbose_logging;
    }
    
    /// 克隆处理管道
    pub fn clone(&self) -> Self {
        Self {
            stages: self.stages.iter().map(|s| s.clone()).collect(),
            name: self.name.clone(),
            continue_on_error: self.continue_on_error,
            verbose_logging: self.verbose_logging,
        }
    }
}

/// 处理阶段
pub struct ProcessingStage {
    /// 处理器
    processor: Box<dyn TextProcessor>,
    /// 条件
    condition: Option<Box<dyn ConditionEvaluator>>,
    /// 名称
    name: String,
    /// 是否启用
    enabled: bool,
}

impl ProcessingStage {
    /// 创建新的处理阶段
    pub fn new(
        processor: Box<dyn TextProcessor>,
        condition: Option<Box<dyn ConditionEvaluator>>,
        name: Option<String>,
    ) -> Self {
        Self {
            processor: processor.clone(),
            condition,
            name: name.unwrap_or_else(|| processor.name().to_string()),
            enabled: true,
        }
    }
    
    /// 获取处理器
    pub fn processor(&self) -> &dyn TextProcessor {
        self.processor.as_ref()
    }
    
    /// 获取条件
    pub fn condition(&self) -> Option<&dyn ConditionEvaluator> {
        self.condition.as_ref().map(|c| c.as_ref())
    }
    
    /// 获取名称
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// 是否启用
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// 设置是否启用
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Clone for ProcessingStage {
    fn clone(&self) -> Self {
        Self {
            processor: self.processor.box_clone(),
            condition: self.condition.clone(),
            name: self.name.clone(),
            enabled: self.enabled,
        }
    }
}

/// 处理器适配器 - 将预处理器适配为文本处理器
pub struct PreprocessorAdapter {
    /// 预处理器
    preprocessor: Arc<dyn TextPreprocessor>,
    /// 名称
    name: String,
    /// 处理器类型
    processor_type: ProcessingStageType,
}

impl PreprocessorAdapter {
    /// 创建新的处理器适配器
    pub fn new(preprocessor: Arc<dyn TextPreprocessor>, name: Option<String>, processor_type: ProcessingStageType) -> Self {
        Self {
            preprocessor: preprocessor.clone(),
            name: name.unwrap_or_else(|| preprocessor.name().to_string()),
            processor_type,
        }
    }
}

impl TextProcessor for PreprocessorAdapter {
    fn process(&self, text: &str) -> Result<String> {
        self.preprocessor.preprocess(text)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        self.processor_type
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(Self {
            preprocessor: self.preprocessor.clone(),
            name: self.name.clone(),
            processor_type: self.processor_type,
        })
    }
}

/// 创建处理器工厂
pub struct TextProcessorFactory;

impl TextProcessorFactory {
    /// 创建标准处理管道
    pub fn create_standard_pipeline() -> Result<TextProcessingPipeline> {
        let configs = vec![
            ProcessingStageConfig {
                stage_type: ProcessingStageType::Cleaning,
                name: "HTML清理".to_string(),
                enabled: true,
                processor_config: {
                    let mut config = HashMap::new();
                    config.insert("remove_html".to_string(), "true".to_string());
                    config
                },
                condition: None,
            },
            ProcessingStageConfig {
                stage_type: ProcessingStageType::Normalization,
                name: "文本规范化".to_string(),
                enabled: true,
                processor_config: {
                    let mut config = HashMap::new();
                    config.insert("lowercase".to_string(), "true".to_string());
                    config.insert("normalize_whitespace".to_string(), "true".to_string());
                    config
                },
                condition: None,
            },
            ProcessingStageConfig {
                stage_type: ProcessingStageType::Tokenization,
                name: "分词".to_string(),
                enabled: true,
                processor_config: HashMap::new(),
                condition: None,
            },
        ];
        
        TextProcessingPipeline::from_configs(configs, "标准处理管道", false, false)
    }
}

/// 根据配置创建处理器
fn create_processor_from_config(config: &ProcessingStageConfig) -> Result<Box<dyn TextProcessor>> {
    // 根据处理阶段类型创建对应的处理器
    match config.stage_type {
        ProcessingStageType::Cleaning => {
            // 创建文本清洗处理器
            if let Some(pattern) = config.processor_config.get("pattern") {
                // 正则表达式清洗
                let replacement = config.processor_config.get("replacement").unwrap_or(&String::new()).to_string();
                let case_sensitive = config.processor_config.get("case_sensitive")
                    .map(|v| v.to_lowercase() == "true")
                    .unwrap_or(true);
                    
                /* 暂时注释未实现的处理器
                let regex_cleaner = RegexCleaner::new(
                    pattern, 
                    &replacement, 
                    case_sensitive, 
                    Some(config.name.clone())
                )?;
                
                Ok(Box::new(regex_cleaner))
                */
                Err(Error::data("RegexCleaner未实现".to_string()))
            } else if config.processor_config.contains_key("remove_html") {
                // HTML标签清洗
                let html_cleaner = HtmlCleaner::new();
                Ok(Box::new(TextCleanerAdapter {
                    cleaner: Box::new(html_cleaner),
                    name: "HtmlCleaner".to_string(),
                }))
            } else if config.processor_config.contains_key("remove_urls") {
                // URL清洗
                let url_cleaner = UrlCleaner::new();
                Ok(Box::new(TextCleanerAdapter {
                    cleaner: Box::new(url_cleaner),
                    name: "UrlCleaner".to_string(),
                }))
            } else {
                Err(Error::invalid_argument("清洗处理器配置不完整"))
            }
        },
        
        ProcessingStageType::Normalization => {
            // 创建文本规范化处理器
            if config.processor_config.contains_key("lowercase") {
                // 转小写
                let lowercase = TextCaseNormalizer::new(false, Some(config.name.clone()));
                Ok(Box::new(TextNormalizerAdapter {
                    normalizer: Box::new(lowercase),
                    name: config.name.clone(),
                }))
            } else if config.processor_config.contains_key("uppercase") {
                // 转大写
                let uppercase = TextCaseNormalizer::new(true, Some(config.name.clone()));
                Ok(Box::new(TextNormalizerAdapter {
                    normalizer: Box::new(uppercase),
                    name: config.name.clone(),
                }))
            } else if config.processor_config.contains_key("remove_accents") {
                // 去除重音符号
                let accent_remover = AccentNormalizer::new(Some(config.name.clone()));
                Ok(Box::new(TextNormalizerAdapter {
                    normalizer: Box::new(accent_remover),
                    name: config.name.clone(),
                }))
            } else {
                Err(Error::invalid_argument("规范化处理器配置不完整"))
            }
        },
        
        ProcessingStageType::Tokenization => {
            // 创建分词处理器
            if let Some(method) = config.processor_config.get("method") {
                match method.as_str() {
                    "whitespace" => {
                        let tokenizer = WhitespaceTokenizer::new();
                        Ok(Box::new(TextTokenizerAdapter {
                            tokenizer: Box::new(tokenizer),
                            name: config.name.clone(),
                        }))
                    },
                    "regex" => {
                        if let Some(pattern) = config.processor_config.get("pattern") {
                            /* 暂时注释未实现的分词处理器
                            let tokenizer = RegexTokenizer::new(pattern, Some(config.name.clone()))?;
                            Ok(Box::new(tokenizer))
                            */
                            Err(Error::data("RegexTokenizer未实现".to_string()))
                        } else {
                            Err(Error::invalid_argument("正则分词器需要指定pattern参数"))
                        }
                    },
                    _ => Err(Error::invalid_argument(format!("不支持的分词方法: {}", method)))
                }
            } else {
                Err(Error::invalid_argument("分词处理器配置不完整"))
            }
        },
        
        ProcessingStageType::Filtering => {
            // 创建过滤处理器
            if config.processor_config.contains_key("stopwords") {
                // 停用词过滤
                let stopwords: HashSet<String> = config.processor_config.get("stopwords")
                    .map(|s| s.split(',').map(|w| w.trim().to_string()).collect())
                    .unwrap_or_else(HashSet::new);
                    
                let filter = StopwordFilter::with_custom_stopwords(stopwords);
                Ok(Box::new(TextFilterAdapter {
                    filter: Box::new(filter),
                    name: config.name.clone(),
                }))
            } else if let Some(min_length) = config.processor_config.get("min_length") {
                // 长度过滤
                let min = min_length.parse::<usize>().unwrap_or(0);
                let max = config.processor_config.get("max_length")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(usize::MAX);
                    
                let filter = LengthFilter::new(min, Some(max));
                Ok(Box::new(TextFilterAdapter {
                    filter: Box::new(filter),
                    name: config.name.clone(),
                }))
            } else {
                Err(Error::invalid_argument("过滤处理器配置不完整"))
            }
        },
        
        ProcessingStageType::Transformation => {
            // 创建转换处理器
            if config.processor_config.contains_key("lemmatize") {
                // 词形还原
                /* 暂时注释未实现的转换处理器
                let lemmatizer = LemmatizerTransformer::new(Some(config.name.clone()));
                Ok(Box::new(lemmatizer))
                */
                Err(Error::data("LemmatizerTransformer未实现".to_string()))
            } else if config.processor_config.contains_key("stem") {
                // 词干提取
                /* 暂时注释未实现的转换处理器
                let stemmer = StemmerTransformer::new(Some(config.name.clone()));
                Ok(Box::new(stemmer))
                */
                Err(Error::data("StemmerTransformer未实现".to_string()))
            } else {
                Err(Error::invalid_argument("转换处理器配置不完整"))
            }
        },
        
        ProcessingStageType::Augmentation => {
            // 创建增强处理器
            if let Some(method) = config.processor_config.get("method") {
                match method.as_str() {
                    "synonym" => {
                        /* 暂时注释未实现的增强处理器
                        let augmentor = SynonymAugmentor::new(Some(config.name.clone()));
                        Ok(Box::new(augmentor))
                        */
                        Err(Error::data("SynonymAugmentor未实现".to_string()))
                    },
                    "backtranslation" => {
                        /* 暂时注释未实现的增强处理器
                        let augmentor = BackTranslationAugmentor::new(Some(config.name.clone()));
                        Ok(Box::new(augmentor))
                        */
                        Err(Error::data("BackTranslationAugmentor未实现".to_string()))
                    },
                    _ => Err(Error::invalid_argument(format!("不支持的增强方法: {}", method)))
                }
            } else {
                Err(Error::invalid_argument("增强处理器配置不完整"))
            }
        },
        
        ProcessingStageType::Analysis => {
            // 创建分析处理器
            if config.processor_config.contains_key("sentiment") {
                // 情感分析
                /* 暂时注释未实现的分析处理器
                let analyzer = SentimentAnalyzer::new(Some(config.name.clone()));
                Ok(Box::new(analyzer))
                */
                Err(Error::data("SentimentAnalyzer未实现".to_string()))
            } else if config.processor_config.contains_key("language") {
                // 语言检测
                /* 暂时注释未实现的分析处理器
                let detector = LanguageDetector::new(Some(config.name.clone()));
                Ok(Box::new(detector))
                */
                Err(Error::data("LanguageDetector未实现".to_string()))
            } else {
                Err(Error::invalid_argument("分析处理器配置不完整"))
            }
        },
        
        ProcessingStageType::Custom => {
            // 处理自定义处理器
            if let Some(processor_type) = config.processor_config.get("type") {
                match processor_type.as_str() {
                    "custom_regex" => {
                        if let Some(pattern) = config.processor_config.get("pattern") {
                            let replacement = config.processor_config.get("replacement").unwrap_or(&String::new()).to_string();
                            /* 暂时注释未实现的自定义处理器
                            let custom_processor = CustomRegexProcessor::new(
                                pattern, 
                                &replacement, 
                                Some(config.name.clone())
                            )?;
                            Ok(Box::new(custom_processor))
                            */
                            Err(Error::data("CustomRegexProcessor未实现".to_string()))
                        } else {
                            Err(Error::invalid_argument("自定义正则处理器需要指定pattern参数"))
                        }
                    },
                    _ => Err(Error::invalid_argument(format!("不支持的自定义处理器类型: {}", processor_type)))
                }
            } else {
                Err(Error::invalid_argument("自定义处理器配置不完整"))
            }
        },
    }
}

/// 解析条件表达式
fn parse_condition_expression(expression: &str) -> Result<Box<dyn ConditionEvaluator>> {
    let expression = expression.trim();
    
    // 处理空表达式
    if expression.is_empty() {
        return Err(Error::invalid_argument("条件表达式不能为空"));
    }
    
    // 处理AND组合条件: "condition1 AND condition2"
    if expression.contains(" AND ") {
        let parts: Vec<&str> = expression.split(" AND ").collect();
        let mut conditions = Vec::new();
        
        for part in parts {
            let condition = parse_condition_expression(part)?;
            conditions.push(condition);
        }
        
        return Ok(Box::new(CombinedCondition::new(
            conditions,
            CombinationType::All,
            Some("AND组合条件".to_string())
        )));
    }
    
    // 处理OR组合条件: "condition1 OR condition2"
    if expression.contains(" OR ") {
        let parts: Vec<&str> = expression.split(" OR ").collect();
        let mut conditions = Vec::new();
        
        for part in parts {
            let condition = parse_condition_expression(part)?;
            conditions.push(condition);
        }
        
        return Ok(Box::new(CombinedCondition::new(
            conditions,
            CombinationType::Any,
            Some("OR组合条件".to_string())
        )));
    }
    
    // 处理NOT条件: "NOT condition"
    if expression.starts_with("NOT ") {
        let inner_expression = &expression[4..];
        let condition = parse_condition_expression(inner_expression)?;
        
        let mut conditions = Vec::new();
        conditions.push(condition);
        
        return Ok(Box::new(CombinedCondition::new(
            conditions,
            CombinationType::None,
            Some("NOT条件".to_string())
        )));
    }
    
    // 处理常见内置条件
    
    // 长度条件: "LENGTH > 10" 或 "LENGTH < 5" 或 "LENGTH BETWEEN 5 AND 10"
    if expression.starts_with("LENGTH ") {
        if let Some(pos) = expression.find('>') {
            let value_str = expression[pos+1..].trim();
            if let Ok(min_length) = value_str.parse::<usize>() {
                return Ok(Box::new(LengthCondition::new(
                    Some(min_length),
                    None,
                    Some(format!("LENGTH > {}", min_length))
                )));
            }
        } else if let Some(pos) = expression.find('<') {
            let value_str = expression[pos+1..].trim();
            if let Ok(max_length) = value_str.parse::<usize>() {
                return Ok(Box::new(LengthCondition::new(
                    None,
                    Some(max_length),
                    Some(format!("LENGTH < {}", max_length))
                )));
            }
        } else if expression.contains("BETWEEN") {
            let parts: Vec<&str> = expression.split("BETWEEN").collect();
            if parts.len() == 2 {
                let range_str = parts[1].trim();
                let range_parts: Vec<&str> = range_str.split("AND").collect();
                
                if range_parts.len() == 2 {
                    let min_str = range_parts[0].trim();
                    let max_str = range_parts[1].trim();
                    
                    if let (Ok(min), Ok(max)) = (min_str.parse::<usize>(), max_str.parse::<usize>()) {
                        return Ok(Box::new(LengthCondition::new(
                            Some(min),
                            Some(max),
                            Some(format!("LENGTH BETWEEN {} AND {}", min, max))
                        )));
                    }
                }
            }
        }
    }
    
    // 正则匹配条件: "MATCHES 正则表达式"
    if expression.starts_with("MATCHES ") {
        let pattern = expression[8..].trim();
        if !pattern.is_empty() {
            return Ok(Box::new(RegexCondition::new(
                pattern,
                true,
                Some(format!("MATCHES {}", pattern))
            )?));
        }
    }
    
    // 正则不匹配条件: "NOT_MATCHES 正则表达式"
    if expression.starts_with("NOT_MATCHES ") {
        let pattern = expression[12..].trim();
        if !pattern.is_empty() {
            return Ok(Box::new(RegexCondition::new(
                pattern,
                false,
                Some(format!("NOT_MATCHES {}", pattern))
            )?));
        }
    }
    
    // 元数据条件: "METADATA.key MATCHES pattern"
    if expression.starts_with("METADATA.") {
        let parts: Vec<&str> = expression.splitn(3, ' ').collect();
        if parts.len() == 3 {
            let key = parts[0].replace("METADATA.", "");
            let op = parts[1];
            let value = parts[2].trim();
            
            if op == "MATCHES" {
                return Ok(Box::new(MetadataCondition::new(
                    &key,
                    value,
                    true,
                    Some(format!("METADATA.{} MATCHES {}", key, value))
                )?));
            } else if op == "NOT_MATCHES" {
                return Ok(Box::new(MetadataCondition::new(
                    &key,
                    value,
                    false,
                    Some(format!("METADATA.{} NOT_MATCHES {}", key, value))
                )?));
            }
        }
    }
    
    // 如果无法解析，返回错误
    Err(Error::invalid_argument(format!("无法解析条件表达式: {}", expression)))
} 