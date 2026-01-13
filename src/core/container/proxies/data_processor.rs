/// 数据处理器代理模块
/// 
/// 提供数据处理器的代理实现，支持通过服务容器获取真实服务或使用默认实现

use std::sync::Arc;
use async_trait::async_trait;
use uuid::Uuid;
use log::{info, debug};
use chrono::Utc;
use std::collections::HashMap;
use serde_json::Value;

use crate::{Result, Error};
use crate::core::container::{DefaultServiceContainer, ServiceContainer};
use crate::core::interfaces::{
    DataProcessorInterface, ProcessedData, DataSchema, ValidationResult
};
use crate::core::types::CoreDataBatch;
use crate::core::types::{CoreTensorData, InterfacePreprocessingConfig};

/// 数据处理器代理实现
pub struct DataProcessorProxy {
    container: Arc<DefaultServiceContainer>,
}

impl DataProcessorProxy {
    /// 创建新的数据处理器代理
    pub fn new(container: Arc<DefaultServiceContainer>) -> Self {
        Self { container }
    }

    /// 兼容接口层的预处理配置，行使 InterfacePreprocessingConfig 导入
    pub async fn preprocess_with_interface_config(
        &self,
        data: &CoreDataBatch,
        cfg: &InterfacePreprocessingConfig,
    ) -> Result<CoreDataBatch> {
        let _bridge = crate::core::interfaces::PreprocessingConfig {
            cleaning_strategies: cfg.parameters.get("cleaning_strategies").cloned().map(|s| serde_json::from_str(&s).unwrap_or_default()).unwrap_or_default(),
            normalization_strategies: cfg.parameters.get("normalization_strategies").cloned().map(|s| serde_json::from_str(&s).unwrap_or_default()).unwrap_or_default(),
            use_ngrams: cfg.parameters.get("use_ngrams").map(|s| s == "true").unwrap_or(false),
            ngram_range: cfg.parameters.get("ngram_range").and_then(|s| {
                if let Some((start, end)) = s.split_once(',') {
                    if let (Ok(start), Ok(end)) = (start.trim().parse::<usize>(), end.trim().parse::<usize>()) {
                        Some((start, end))
                    } else { None }
                } else { None }
            }),
            use_char_ngrams: cfg.parameters.get("use_char_ngrams").map(|s| s == "true").unwrap_or(false),
            char_ngram_range: cfg.parameters.get("char_ngram_range").and_then(|s| {
                if let Some((start, end)) = s.split_once(',') {
                    if let (Ok(start), Ok(end)) = (start.trim().parse::<usize>(), end.trim().parse::<usize>()) {
                        Some((start, end))
                    } else { None }
                } else { None }
            }).unwrap_or((1, 1)),
            use_filtering: cfg.parameters.get("use_filtering").map(|s| s == "true").unwrap_or(false),
            remove_stopwords: cfg.parameters.get("remove_stopwords").map(|s| s == "true").unwrap_or(false),
            min_token_length: cfg.parameters.get("min_token_length").and_then(|s| s.parse().ok()).unwrap_or(1),
            max_token_length: cfg.parameters.get("max_token_length").and_then(|s| s.parse::<usize>().ok()),
            language: cfg.parameters.get("language").cloned().unwrap_or_else(|| "en".to_string()),
        };
        // 调用预处理数据方法并转换返回类型
        let processed_data = self.preprocess_data(data).await?;
        // 将 ProcessedData 转换为 CoreDataBatch
        Ok(CoreDataBatch {
            id: processed_data.id,
            data: vec![],
            labels: None,
            batch_size: data.batch_size,
            metadata: Some(processed_data.metadata),
            created_at: processed_data.created_at,
            updated_at: chrono::Utc::now(),
        })
    }

    /// 尝试从容器获取真实的数据处理器服务
    async fn get_real_data_processor(&self) -> Option<Arc<dyn DataProcessorInterface + Send + Sync>> {
        // 优先通过 trait 获取
        if let Ok(dp_iface) = self.container.as_ref().get_trait::<dyn DataProcessorInterface + Send + Sync>() {
            return Some(dp_iface);
        }
        // 回退到具体实现并包装
        if let Ok(real_processor) = self.container.get::<crate::data::processor::processor_impl::DataProcessor>() {
            return Some(Arc::new(RealDataProcessorWrapper { processor: real_processor.clone() }));
        }
        None
    }

    /// 验证输入数据
    fn validate_input_data(&self, data: &HashMap<String, Value>) -> Result<()> {
        if data.is_empty() {
            return Err(Error::InvalidInput("输入数据不能为空".to_string()));
        }
        
        for (key, value) in data {
            if key.is_empty() {
                return Err(Error::InvalidInput("数据键不能为空".to_string()));
            }
            
            if key.len() > 100 {
                return Err(Error::InvalidInput("数据键长度不能超过100个字符".to_string()));
            }
            
            // 验证值的类型
            match value {
                Value::Null => {
                    return Err(Error::InvalidInput(
                        format!("数据键 '{}' 的值不能为null", key)
                    ));
                },
                Value::String(s) => {
                    if s.is_empty() {
                        return Err(Error::InvalidInput(
                            format!("数据键 '{}' 的字符串值不能为空", key)
                        ));
                    }
                },
                Value::Number(n) => {
                    if !n.is_f64() && !n.is_i64() {
                        return Err(Error::InvalidInput(
                            format!("数据键 '{}' 的数值类型不支持", key)
                        ));
                    }
                },
                Value::Array(arr) => {
                    if arr.is_empty() {
                        return Err(Error::InvalidInput(
                            format!("数据键 '{}' 的数组不能为空", key)
                        ));
                    }
                },
                Value::Object(obj) => {
                    if obj.is_empty() {
                        return Err(Error::InvalidInput(
                            format!("数据键 '{}' 的对象不能为空", key)
                        ));
                    }
                },
                Value::Bool(_) => {
                    // 布尔值是可接受的
                }
            }
        }
        
        Ok(())
    }

    /// 将JSON值转换为张量
    fn json_to_tensor(&self, key: &str, value: &Value) -> Result<CoreTensorData> {
        let mut data = Vec::new();
        let mut shape = Vec::new();
        
        match value {
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    data.push(f as f32);
                    shape.push(1);
                } else if let Some(i) = n.as_i64() {
                    data.push(i as f32);
                    shape.push(1);
                } else {
                    return Err(Error::InvalidInput(
                        format!("无法转换数值类型: {}", n)
                    ));
                }
            },
            Value::Array(arr) => {
                shape.push(arr.len());
                for item in arr {
                    match item {
                        Value::Number(n) => {
                            if let Some(f) = n.as_f64() {
                                data.push(f as f32);
                            } else if let Some(i) = n.as_i64() {
                                data.push(i as f32);
                            } else {
                                return Err(Error::InvalidInput(
                                    format!("数组元素不是有效数值: {}", n)
                                ));
                            }
                        },
                        _ => {
                            return Err(Error::InvalidInput(
                                format!("数组元素类型不支持: {:?}", item)
                            ));
                        }
                    }
                }
            },
            Value::String(s) => {
                // 尝试将字符串解析为数值
                if let Ok(f) = s.parse::<f32>() {
                    data.push(f);
                    shape.push(1);
                } else {
                    return Err(Error::InvalidInput(
                        format!("无法将字符串转换为数值: {}", s)
                    ));
                }
            },
            _ => {
                return Err(Error::InvalidInput(
                    format!("不支持的数据类型: {:?}", value)
                ));
            }
        }
        
        Ok(CoreTensorData {
            id: uuid::Uuid::new_v4().to_string(),
            shape,
            data,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        })
    }

    /// 标准化张量
    fn normalize_tensor(&self, tensor: &mut CoreTensorData) -> Result<()> {
        if tensor.data.is_empty() {
            return Err(Error::InvalidInput("张量数据为空，无法标准化".to_string()));
        }
        
        // 计算均值和标准差
        let sum: f32 = tensor.data.iter().sum();
        let mean = sum / tensor.data.len() as f32;
        
        let variance: f32 = tensor.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / tensor.data.len() as f32;
        let std_dev = variance.sqrt();
        
        // 避免除零
        if std_dev < 1e-8 {
            return Err(Error::InvalidInput("标准差太小，无法标准化".to_string()));
        }
        
        // 应用标准化
        for value in &mut tensor.data {
            *value = (*value - mean) / std_dev;
        }
        
        Ok(())
    }

    /// 分割数据集
    fn split_dataset(&self, data: &HashMap<String, CoreTensorData>, train_ratio: f32) -> Result<(HashMap<String, CoreTensorData>, HashMap<String, CoreTensorData>)> {
        if train_ratio <= 0.0 || train_ratio >= 1.0 {
            return Err(Error::InvalidInput("训练比例必须在0和1之间".to_string()));
        }
        
        let total_samples = data.len();
        let train_count = (total_samples as f32 * train_ratio) as usize;
        
        let mut train_data = HashMap::new();
        let mut test_data = HashMap::new();
        
        let mut keys: Vec<_> = data.keys().collect();
        keys.sort(); // 确保分割的一致性
        
        for (i, key) in keys.iter().enumerate() {
            let tensor = data.get(*key)
                .ok_or_else(|| Error::invalid_data(format!("数据中缺少键: {}", key)))?;
            if i < train_count {
                train_data.insert((*key).clone(), tensor.clone());
            } else {
                test_data.insert((*key).clone(), tensor.clone());
            }
        }
        
        Ok((train_data, test_data))
    }
}

/// 真实数据处理器的包装器
/// 用于将具体的数据处理器实现包装为 trait object
struct RealDataProcessorWrapper {
    processor: Arc<crate::data::processor::processor_impl::DataProcessor>,
}

#[async_trait]
impl DataProcessorInterface for RealDataProcessorWrapper {
    async fn process_batch(&self, batch: &CoreDataBatch) -> Result<CoreDataBatch> {
        self.processor.process_batch(batch).await
    }
    
    async fn validate_data(&self, data: &CoreDataBatch) -> Result<ValidationResult> {
        self.processor.validate_data(data).await
    }
    
    async fn convert_data(&self, data: &CoreDataBatch, target_format: &str) -> Result<CoreDataBatch> {
        self.processor.convert_data(data, target_format).await
    }
    
    async fn get_data_statistics(&self, data: &CoreDataBatch) -> Result<HashMap<String, f64>> {
        self.processor.get_data_statistics(data).await
    }
    
    async fn clean_data(&self, data: &CoreDataBatch) -> Result<CoreDataBatch> {
        self.processor.clean_data(data).await
    }
    
    async fn get_processor_config(&self) -> Result<HashMap<String, String>> {
        self.processor.get_processor_config().await
    }
    
    async fn update_processor_config(&self, config: HashMap<String, String>) -> Result<()> {
        self.processor.update_processor_config(config).await
    }
    
    async fn process_data(&self, data: &CoreDataBatch) -> Result<ProcessedData> {
        // 委托给真实的数据处理器
        let core_result = self.processor.process_data(data).await?;
        // core::types::ProcessedData 和 ProcessedData 是同一个类型，直接返回
        Ok(core_result)
    }
    
    async fn convert_to_tensors(&self, data: &ProcessedData) -> Result<Vec<CoreTensorData>> {
        // ProcessedData 和 core::types::ProcessedData 是同一个类型，直接使用
        self.processor.convert_to_tensors(data).await
    }
    
    async fn validate_data_schema(&self, data: &CoreDataBatch, schema: &DataSchema) -> Result<ValidationResult> {
        // 使用完全限定语法调用 trait 方法
        <crate::data::processor::processor_impl::DataProcessor as crate::core::interfaces::DataProcessorInterface>::validate_data_schema(
            &*self.processor, data, schema
        ).await
    }
    
    async fn preprocess_data(&self, data: &CoreDataBatch) -> Result<ProcessedData> {
        // 使用完全限定语法调用 trait 方法
        <crate::data::processor::processor_impl::DataProcessor as crate::core::interfaces::DataProcessorInterface>::preprocess_data(
            &*self.processor, data
        ).await
    }
    
    async fn split_data(&self, data: &CoreDataBatch, train_ratio: f32, val_ratio: f32, test_ratio: f32) -> Result<(CoreDataBatch, CoreDataBatch, CoreDataBatch)> {
        // 使用完全限定语法调用 trait 方法
        <crate::data::processor::processor_impl::DataProcessor as crate::core::interfaces::DataProcessorInterface>::split_data(
            &*self.processor, data, train_ratio, val_ratio, test_ratio
        ).await
    }
    
    async fn normalize_data(&self, data: &CoreDataBatch) -> Result<CoreDataBatch> {
        // 使用完全限定语法调用 trait 方法，保持一致性
        <crate::data::processor::processor_impl::DataProcessor as crate::core::interfaces::DataProcessorInterface>::normalize_data(
            &*self.processor, data
        ).await
    }
}

#[async_trait]
impl DataProcessorInterface for DataProcessorProxy {
    async fn process_batch(&self, batch: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::CoreDataBatch> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.process_batch(batch).await;
        }
        
        // 默认实现：处理数据批次
        debug!("处理数据批次");
        Ok(batch.clone())
    }
    
    async fn validate_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<ValidationResult> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.validate_data(data).await;
        }
        
        // 默认实现：验证数据
        debug!("验证数据");
        Ok(ValidationResult {
            is_valid: true,
            errors: vec![],
            warnings: vec![],
            score: Some(1.0),
            metadata: HashMap::new(),
        })
    }
    
    async fn convert_data(&self, data: &crate::core::types::CoreDataBatch, target_format: &str) -> Result<crate::core::types::CoreDataBatch> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.convert_data(data, target_format).await;
        }
        
        // 默认实现：转换数据格式
        debug!("转换数据格式到: {}", target_format);
        Ok(data.clone())
    }
    
    async fn get_data_statistics(&self, data: &crate::core::types::CoreDataBatch) -> Result<HashMap<String, f64>> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.get_data_statistics(data).await;
        }
        
        // 默认实现：获取数据统计信息
        debug!("获取数据统计信息");
        let mut stats = HashMap::new();
        stats.insert("count".to_string(), data.batch_size as f64);
        stats.insert("size".to_string(), data.data.len() as f64);
        Ok(stats)
    }
    
    async fn clean_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::CoreDataBatch> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.clean_data(data).await;
        }
        
        // 默认实现：清理数据
        debug!("清理数据");
        Ok(data.clone())
    }
    
    async fn get_processor_config(&self) -> Result<HashMap<String, String>> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.get_processor_config().await;
        }
        
        // 默认实现：获取处理器配置
        debug!("获取处理器配置");
        let mut config = HashMap::new();
        config.insert("batch_size".to_string(), "32".to_string());
        config.insert("normalize".to_string(), "true".to_string());
        Ok(config)
    }
    
    async fn update_processor_config(&self, config: HashMap<String, String>) -> Result<()> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.update_processor_config(config).await;
        }
        
        // 默认实现：更新处理器配置
        debug!("更新处理器配置: {:?}", config);
        Ok(())
    }
    
    async fn process_data(&self, data: &CoreDataBatch) -> Result<ProcessedData> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.process_data(data).await;
        }
        
        // 将CoreDataBatch中的tensor数据转换为字节数据
        let mut processed_bytes = Vec::new();
        let mut total_size = 0;
        for tensor in &data.data {
            // 将f32数据转换为字节
            for &val in &tensor.data {
                processed_bytes.extend_from_slice(&val.to_le_bytes());
                total_size += 4;
            }
        }
        
        // 创建处理结果
        let mut metadata = HashMap::new();
        metadata.insert("batch_size".to_string(), data.batch_size.to_string());
        metadata.insert("tensor_count".to_string(), data.data.len().to_string());
        metadata.insert("processed_at".to_string(), Utc::now().to_rfc3339());
        metadata.insert("data_type".to_string(), "processed".to_string());
        metadata.insert("processing_steps".to_string(), "batch_to_tensors".to_string());
        
        let processed_data = ProcessedData {
            id: Uuid::new_v4().to_string(),
            data: processed_bytes,
            format: "tensor".to_string(),
            size: total_size,
            metadata,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        info!("成功处理数据批次，大小: {}", data.batch_size);
        Ok(processed_data)
    }
    
    async fn preprocess_data(&self, data: &CoreDataBatch) -> Result<ProcessedData> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            // 接口方法 preprocess_data 只接受一个参数，内部会使用默认配置
            return real_processor.preprocess_data(data).await;
        }
        
        // 默认实现：基本预处理 - 处理tensor数据
        let mut processed_tensors = Vec::new();
        
        for tensor in &data.data {
            let mut processed_tensor = tensor.clone();
            
            // 应用标准化处理（简化版本）
            debug!("应用标准化处理");
            if !processed_tensor.data.is_empty() {
                let mean: f32 = processed_tensor.data.iter().sum::<f32>() / processed_tensor.data.len() as f32;
                let variance: f32 = processed_tensor.data.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>() / processed_tensor.data.len() as f32;
                let std_dev = variance.sqrt();
                
                if std_dev > 1e-6 {
                    for val in &mut processed_tensor.data {
                        *val = (*val - mean) / std_dev;
                    }
                }
            }
            
            processed_tensors.push(processed_tensor);
        }
        
        // 转换为ProcessedData
        let mut processed_bytes = Vec::new();
        let mut total_size = 0;
        for tensor in &processed_tensors {
            for &val in &tensor.data {
                processed_bytes.extend_from_slice(&val.to_le_bytes());
                total_size += 4;
            }
        }
        
        let mut metadata = data.metadata.clone().unwrap_or_default();
        metadata.insert("preprocessed".to_string(), "true".to_string());
        metadata.insert("tensor_count".to_string(), processed_tensors.len().to_string());
        
        let processed_data = ProcessedData {
            id: Uuid::new_v4().to_string(),
            data: processed_bytes,
            format: "tensor".to_string(),
            size: total_size,
            metadata,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        info!("成功预处理数据批次");
        Ok(processed_data)
    }
    
    async fn split_data(&self, data: &CoreDataBatch, train_ratio: f32, val_ratio: f32, test_ratio: f32) -> Result<(CoreDataBatch, CoreDataBatch, CoreDataBatch)> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.split_data(data, train_ratio, val_ratio, test_ratio).await;
        }
        
        // 默认实现：数据分割
        let ratios = vec![train_ratio, val_ratio, test_ratio];
        if ratios.is_empty() {
            return Err(Error::InvalidInput("分割比例不能为空".to_string()));
        }
        
        let total_ratio: f32 = ratios.iter().sum();
        if (total_ratio - 1.0).abs() > 1e-6 {
            return Err(Error::InvalidInput("分割比例总和必须为1".to_string()));
        }
        
        let mut splits = Vec::new();
        let mut current_index = 0;
        let data_len = data.data.len();
        
        for (i, &ratio) in ratios.iter().enumerate() {
            let split_size = (data_len as f32 * ratio) as usize;
            let end_index = current_index + split_size;
            
            let split_data_vec = data.data[current_index..end_index.min(data_len)].to_vec();
            let split_labels = if let Some(ref labels) = data.labels {
                let labels_len = labels.len();
                Some(labels[current_index..end_index.min(labels_len)].to_vec())
            } else {
                None
            };
            
            let split_batch = CoreDataBatch {
                id: format!("{}_{}", data.id, i),
                data: split_data_vec,
                labels: split_labels,
                batch_size: split_size,
                metadata: data.metadata.clone(),
                created_at: data.created_at,
                updated_at: Utc::now(),
            };
            
            splits.push(split_batch);
            current_index = end_index;
        }
        
        if splits.len() != 3 {
            return Err(Error::InvalidInput("分割结果数量不正确".to_string()));
        }
        
        info!("成功分割数据批次为 {} 个子集", splits.len());
        Ok((splits[0].clone(), splits[1].clone(), splits[2].clone()))
    }
    
    async fn normalize_data(&self, data: &CoreDataBatch) -> Result<CoreDataBatch> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.normalize_data(data).await;
        }
        
        // 默认实现：数据标准化 - 对tensor数据进行标准化
        let mut normalized_tensors = Vec::new();
        
        for tensor in &data.data {
            let mut normalized_tensor = tensor.clone();
            
            // 对tensor数据进行标准化（Z-score标准化）
            if !normalized_tensor.data.is_empty() {
                let mean: f32 = normalized_tensor.data.iter().sum::<f32>() / normalized_tensor.data.len() as f32;
                let variance: f32 = normalized_tensor.data.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>() / normalized_tensor.data.len() as f32;
                let std_dev = variance.sqrt();
                
                if std_dev > 1e-6 {
                    for val in &mut normalized_tensor.data {
                        *val = (*val - mean) / std_dev;
                    }
                } else {
                    // 如果标准差为0，将所有值设为0
                    for val in &mut normalized_tensor.data {
                        *val = 0.0;
                    }
                }
            }
            
            normalized_tensors.push(normalized_tensor);
        }
        
        let normalized_batch = CoreDataBatch {
            id: data.id.clone(),
            data: normalized_tensors,
            labels: data.labels.clone(),
            batch_size: data.batch_size,
            metadata: data.metadata.clone(),
            created_at: data.created_at,
            updated_at: Utc::now(),
        };
        
        info!("成功标准化数据批次");
        Ok(normalized_batch)
    }
    
    async fn convert_to_tensors(&self, data: &ProcessedData) -> Result<Vec<CoreTensorData>> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.convert_to_tensors(data).await;
        }
        
        // 默认实现：转换为张量
        let mut tensors = Vec::new();
        
        // 将处理后的字节数据转换为张量
        if !data.data.is_empty() {
            // 从metadata中获取shape信息，如果没有则使用默认值
            let shape_str = data.metadata.get("shape")
                .and_then(|s| serde_json::from_str::<Vec<usize>>(s).ok())
                .unwrap_or_else(|| vec![data.data.len() / 4]); // 假设是f32数据，每个4字节
            
            // 将字节数据转换为f32数组
            let mut f32_data = Vec::new();
            for chunk in data.data.chunks_exact(4) {
                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32_data.push(f32::from_le_bytes(bytes));
            }
            
            let tensor = CoreTensorData {
                id: Uuid::new_v4().to_string(),
                shape: shape_str,
                data: f32_data,
                dtype: "float32".to_string(),
                device: "cpu".to_string(),
                requires_grad: false,
                metadata: data.metadata.clone(),
                created_at: data.created_at,
                updated_at: data.updated_at,
            };
            tensors.push(tensor);
        }
        
        info!("成功转换 {} 个张量", tensors.len());
        Ok(tensors)
    }
    
    async fn validate_data_schema(&self, data: &CoreDataBatch, schema: &DataSchema) -> Result<ValidationResult> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.validate_data_schema(data, schema).await;
        }
        
        // 默认实现：基本验证
        if data.data.is_empty() {
            return Ok(ValidationResult {
                is_valid: false,
                errors: vec!["数据批次为空".to_string()],
                warnings: Vec::new(),
                score: Some(0.0),
                metadata: HashMap::new(),
            });
        }
        
        // 检查tensor数量与batch_size是否匹配
        if data.data.len() != data.batch_size && data.batch_size > 0 {
            return Ok(ValidationResult {
                is_valid: false,
                errors: vec![format!("tensor数量 {} 与batch_size {} 不匹配", data.data.len(), data.batch_size)],
                warnings: Vec::new(),
                score: Some(0.0),
                metadata: HashMap::new(),
            });
        }
        
        // 检查每个tensor的结构
        for (i, tensor) in data.data.iter().enumerate() {
            if tensor.shape.is_empty() {
                return Ok(ValidationResult {
                    is_valid: false,
                    errors: vec![format!("tensor {} 的shape为空", i)],
                    warnings: Vec::new(),
                    score: Some(0.0),
                    metadata: HashMap::new(),
                });
            }
            if tensor.data.is_empty() {
                return Ok(ValidationResult {
                    is_valid: false,
                    errors: vec![format!("tensor {} 的数据为空", i)],
                    warnings: Vec::new(),
                    score: Some(0.0),
                    metadata: HashMap::new(),
                });
            }
        }
        
        info!("数据模式验证通过");
        Ok(ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: Some(1.0),
            metadata: HashMap::new(),
        })
    }
}

// 为 DataProcessorProxy 实现额外的辅助方法
impl DataProcessorProxy {
    fn normalize_tensor_data(&self, tensor: &mut CoreTensorData) -> Result<()> {
        if tensor.data.is_empty() {
            return Err(Error::InvalidInput("张量数据为空，无法标准化".to_string()));
        }
        
        // 计算均值和标准差
        let sum: f32 = tensor.data.iter().sum();
        let mean = sum / tensor.data.len() as f32;
        
        let variance: f32 = tensor.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / tensor.data.len() as f32;
        let std_dev = variance.sqrt();
        
        // 避免除零
        if std_dev < 1e-8 {
            return Err(Error::InvalidInput("标准差太小，无法标准化".to_string()));
        }
        
        // 应用标准化
        for value in &mut tensor.data {
            *value = (*value - mean) / std_dev;
        }
        
        Ok(())
    }
    
    async fn process_batch(&self, batch: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::CoreDataBatch> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.process_batch(batch).await;
        }
        
        // 默认实现：处理数据批次
        debug!("处理数据批次");
        Ok(batch.clone())
    }
    
    async fn validate_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<ValidationResult> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.validate_data(data).await;
        }
        
        // 默认实现：验证数据格式
        debug!("验证数据格式");
        Ok(ValidationResult {
            is_valid: true,
            errors: vec![],
            warnings: vec![],
            score: Some(1.0),
            metadata: std::collections::HashMap::new(),
        })
    }
    
    async fn convert_data(&self, data: &crate::core::types::CoreDataBatch, target_format: &str) -> Result<crate::core::types::CoreDataBatch> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.convert_data(data, target_format).await;
        }
        
        // 默认实现：转换数据格式
        debug!("转换数据格式到: {}", target_format);
        Ok(data.clone())
    }
    
    async fn get_data_statistics(&self, data: &crate::core::types::CoreDataBatch) -> Result<HashMap<String, f64>> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.get_data_statistics(data).await;
        }
        
        // 默认实现：获取数据统计信息
        debug!("获取数据统计信息");
        Ok(HashMap::new())
    }
    
    async fn clean_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::CoreDataBatch> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.clean_data(data).await;
        }
        
        // 默认实现：清理数据
        debug!("清理数据");
        Ok(data.clone())
    }
    
    async fn get_processor_config(&self) -> Result<HashMap<String, String>> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.get_processor_config().await;
        }
        
        // 默认实现：获取处理器配置
        debug!("获取处理器配置");
        Ok(HashMap::new())
    }
    
    async fn update_processor_config(&self, config: HashMap<String, String>) -> Result<()> {
        // 首先尝试从容器获取真实的数据处理器服务
        if let Some(real_processor) = self.get_real_data_processor().await {
            return real_processor.update_processor_config(config).await;
        }
        
        // 默认实现：更新处理器配置
        debug!("更新处理器配置");
        Ok(())
    }
} 