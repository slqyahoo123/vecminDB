use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use crate::error::Result;
use crate::data::pipeline;
use crate::data::pipeline::pipeline::Pipeline;
use crate::data::types::DataFormat;
use crate::data::dataset::Dataset;
use crate::data::loader::DataLoader;
use crate::data::processor::DataProcessor;
use crate::data::batch::DataBatch;

/// 数据管道配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataPipelineConfig {
    pub data_dir: String,
    pub cache_dir: String,
    pub max_cache_size: usize,
    pub batch_size: usize,
    pub shuffle_buffer: usize,
    pub num_workers: usize,
}

impl Default for DataPipelineConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".to_string(),
            cache_dir: "cache".to_string(),
            max_cache_size: 1024 * 1024 * 1024, // 1GB
            batch_size: 32,
            shuffle_buffer: 1000,
            num_workers: 4,
        }
    }
}

/// 数据管道
pub struct DataPipeline {
    loader: Arc<dyn DataLoader>,
    processor: Arc<DataProcessor>,
    datasets: Arc<RwLock<HashMap<String, Dataset>>>,
    config: DataPipelineConfig,
    stages: Vec<Box<dyn pipeline::DataStage>>,
    metrics: HashMap<String, f64>,
}

impl DataPipeline {
    /// 创建新的数据管道
    pub fn new(config: DataPipelineConfig) -> Result<Self> {
        // 创建基础组件
        let loader = Arc::new(crate::data::loader::memory::MemoryDataLoader::new());
        let processor = Arc::new(DataProcessor::new_default());
        
        Ok(Self {
            loader,
            processor,
            datasets: Arc::new(RwLock::new(HashMap::new())),
            config,
            stages: Vec::new(),
            metrics: HashMap::new(),
        })
    }

    /// 处理数据批次
    pub fn process(&self, data: DataBatch) -> Result<DataBatch> {
        // 应用所有处理阶段
        let mut processed_data = data;
        for stage in &self.stages {
            stage.process(&mut processed_data)?;
        }
        Ok(processed_data)
    }

    /// 加载数据集
    pub async fn load_dataset(&self, path: &str, format: DataFormat) -> Result<String> {
        let dataset_id = uuid::Uuid::new_v4().to_string();
        
        // 创建数据集
        let format_description = format!("Loaded from {} in {:?} format", path, format);
        let dataset = Dataset {
            id: dataset_id.clone(),
            name: format!("Dataset from {}", path),
            description: Some(format_description.clone()),
            format: format.clone(),
            size: 0, // 待计算
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: crate::data::types::DatasetMetadata {
                id: dataset_id.clone(),
                name: format!("Dataset from {}", path),
                description: Some(format_description),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                version: "1.0".to_string(),
                owner: "system".to_string(),
                schema: None,
                properties: std::collections::HashMap::new(),
                tags: vec!["loaded".to_string(), format!("{:?}", format)],
                records_count: 0, // 待计算
                size_bytes: 0, // 待计算
            },
            path: path.to_string(),
            processed: false,
            loader: self.loader.clone(),
            batch_size: self.config.batch_size,
            schema: None,
            batches: Vec::new(), // 初始为空，后续处理时填充
        };
        
        // 存储数据集
        {
            let mut datasets = self.datasets.write().unwrap();
            datasets.insert(dataset_id.clone(), dataset);
        }
        
        Ok(dataset_id)
    }

    /// 处理数据集
    pub async fn process_dataset(
        &self,
        dataset_id: &str,
        processor_type: &str,
        _options: &HashMap<String, String>,
    ) -> Result<String> {
        let dataset = {
            let datasets = self.datasets.read().unwrap();
            datasets.get(dataset_id)
                .ok_or_else(|| crate::error::Error::NotFound(format!("Dataset {}", dataset_id)))?
                .clone()
        };
        
        // 创建处理后的数据集ID
        let processed_id = uuid::Uuid::new_v4().to_string();
        
        // 执行处理
        log::info!("Processing dataset {} with {} processor", dataset_id, processor_type);
        
        // 创建处理后的数据集
        let mut processed_dataset = dataset.clone();
        processed_dataset.id = processed_id.clone();
        processed_dataset.name = format!("{} (processed)", dataset.name);
        processed_dataset.processed = true;
        processed_dataset.updated_at = chrono::Utc::now();
        
        // 存储处理后的数据集
        {
            let mut datasets = self.datasets.write().unwrap();
            datasets.insert(processed_id.clone(), processed_dataset);
        }
        
        Ok(processed_id)
    }

    /// 分割数据集
    pub async fn split_dataset(
        &self,
        dataset_id: &str,
        ratios: &[f32],
        shuffle: bool,
    ) -> Result<crate::data::types::DataSplit> {
        let dataset = {
            let datasets = self.datasets.read().unwrap();
            datasets.get(dataset_id)
                .ok_or_else(|| crate::error::Error::NotFound(format!("Dataset {}", dataset_id)))?
                .clone()
        };
        
        // 验证比例
        let total_ratio: f32 = ratios.iter().sum();
        if (total_ratio - 1.0).abs() > 1e-6 {
            return Err(crate::error::Error::invalid_argument(
                format!("Ratios must sum to 1.0, got {}", total_ratio)
            ));
        }
        
        // 创建分割结果
        let train_id = format!("{}_train", dataset_id);
        let validation_id = if ratios.len() > 1 {
            Some(format!("{}_val", dataset_id))
        } else {
            None
        };
        let test_id = if ratios.len() > 2 {
            Some(format!("{}_test", dataset_id))
        } else {
            None
        };
        
        // 创建分割后的数据集
        let train_dataset = Dataset {
            id: train_id.clone(),
            name: format!("{} (train)", dataset.name),
            description: Some("Training split".to_string()),
            format: dataset.format.clone(),
            size: (dataset.size as f32 * ratios[0]) as usize,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: dataset.metadata.clone(),
            path: format!("{}_train", dataset.path),
            processed: dataset.processed,
            loader: dataset.loader.clone(),
            batch_size: dataset.batch_size,
            schema: dataset.schema.clone(),
            batches: Vec::new(), // 分割后重新生成批次
        };
        
        // 存储训练集
        {
            let mut datasets = self.datasets.write().unwrap();
            datasets.insert(train_id.clone(), train_dataset);
        }
        
        // 如果有验证集，创建并存储
        if let Some(val_id) = &validation_id {
            let val_dataset = Dataset {
                id: val_id.clone(),
                name: format!("{} (validation)", dataset.name),
                description: Some("Validation split".to_string()),
                format: dataset.format.clone(),
                size: (dataset.size as f32 * ratios.get(1).unwrap_or(&0.0)) as usize,
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                metadata: dataset.metadata.clone(),
                path: format!("{}_val", dataset.path),
                processed: dataset.processed,
                loader: dataset.loader.clone(),
                batch_size: dataset.batch_size,
                schema: dataset.schema.clone(),
                batches: Vec::new(), // 分割后重新生成批次
            };
            
            let mut datasets = self.datasets.write().unwrap();
            datasets.insert(val_id.clone(), val_dataset);
        }
        
        Ok(crate::data::types::DataSplit {
            train: train_id,
            validation: validation_id,
            test: test_id,
            ratios: ratios.to_vec(),
            shuffle,
        })
    }

    /// 获取数据批次
    pub async fn get_batch(&self, dataset_id: &str, batch_size: usize) -> Result<DataBatch> {
        let dataset = {
            let datasets = self.datasets.read().unwrap();
            datasets.get(dataset_id)
                .ok_or_else(|| crate::error::Error::NotFound(format!("Dataset {}", dataset_id)))?
                .clone()
        };
        
        // 使用数据加载器获取批次
        let source = crate::data::loader::DataSource::File(dataset.path.clone());
        // 将data::types::DataFormat转换为loader::types::DataFormat
        let loader_format = match &dataset.format {
            crate::data::types::DataFormat::CSV => crate::data::loader::types::DataFormat::csv(),
            crate::data::types::DataFormat::JSON => crate::data::loader::types::DataFormat::json(),
            crate::data::types::DataFormat::Parquet => crate::data::loader::types::DataFormat::parquet(),
            crate::data::types::DataFormat::Avro => crate::data::loader::types::DataFormat::avro(),
            crate::data::types::DataFormat::CustomText(fmt) => crate::data::loader::types::DataFormat::CustomText(fmt.clone()),
            _ => crate::data::loader::types::DataFormat::json(), // 默认为JSON
        };
        dataset.loader.load_batch(&source, &loader_format, batch_size, 0).await
    }

    /// 获取数据迭代器
    pub async fn get_iterator(&self, dataset_id: &str, batch_size: usize) -> Result<crate::data::batch::DataIterator> {
        let dataset = {
            let datasets = self.datasets.read().unwrap();
            datasets.get(dataset_id)
                .ok_or_else(|| crate::error::Error::NotFound(format!("Dataset {}", dataset_id)))?
                .clone()
        };
        
        Ok(crate::data::batch::DataIterator::new(
            dataset.loader.clone(),
            dataset.path,
            batch_size,
        ))
    }

    /// 获取数据集
    pub async fn get_dataset(&self, dataset_id: &str) -> Result<Option<Dataset>> {
        let datasets = self.datasets.read().unwrap();
        Ok(datasets.get(dataset_id).cloned())
    }

    /// 列出所有数据集
    pub async fn list_datasets(&self) -> Result<Vec<Dataset>> {
        let datasets = self.datasets.read().unwrap();
        Ok(datasets.values().cloned().collect())
    }

    /// 删除数据集
    pub async fn delete_dataset(&self, dataset_id: &str) -> Result<()> {
        let mut datasets = self.datasets.write().unwrap();
        if datasets.remove(dataset_id).is_some() {
            log::info!("Deleted dataset: {}", dataset_id);
            Ok(())
        } else {
            Err(crate::error::Error::NotFound(format!("Dataset {}", dataset_id)))
        }
    }
}

/// 数据处理系统
pub struct DataProcessingSystem {
    pipelines: RwLock<HashMap<String, Box<dyn pipeline::Pipeline>>>,
}

impl DataProcessingSystem {
    /// 创建新的数据处理系统
    pub fn new() -> Self {
        Self {
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// 注册管道
    pub fn register_pipeline(&self, id: &str, pipeline: Box<dyn pipeline::Pipeline>) -> Result<()> {
        let mut pipelines = self.pipelines.write().unwrap();
        pipelines.insert(id.to_string(), pipeline);
        Ok(())
    }

    /// 获取管道
    pub fn get_pipeline(&self, id: &str) -> Result<Box<dyn pipeline::Pipeline>> {
        let pipelines = self.pipelines.read().unwrap();
        match pipelines.get(id) {
            Some(pipeline) => {
                // 这里需要实现克隆，或者返回引用
                Err(crate::error::Error::NotImplemented("Pipeline cloning not implemented".to_string()))
            }
            None => Err(crate::error::Error::NotFound(format!("Pipeline {}", id)))
        }
    }

    /// 列出所有管道
    pub fn list_pipelines(&self) -> Result<Vec<(String, String, String)>> {
        let pipelines = self.pipelines.read().unwrap();
        let mut result = Vec::new();
        
        for (id, _pipeline) in pipelines.iter() {
            result.push((
                id.clone(),
                "Unknown".to_string(), // 类型信息需要从pipeline获取
                "No description".to_string(), // 描述信息需要从pipeline获取
            ));
        }
        
        Ok(result)
    }

    /// 处理数据
    pub fn process_data(&self, pipeline_id: &str, input: &[u8], params: &HashMap<String, String>) -> Result<Vec<u8>> {
        let pipelines = self.pipelines.read().unwrap();
        
        match pipelines.get(pipeline_id) {
            Some(pipeline) => {
                log::info!("Processing data with pipeline: {}", pipeline_id);
                log::debug!("Input size: {} bytes", input.len());
                log::debug!("Parameters: {:?}", params);
                
                // 将输入字节转换为DataBatch
                // 尝试从参数中获取数据类型，默认为Binary
                let data_type = params.get("data_type")
                    .and_then(|s| {
                        match s.as_str() {
                            "string" | "text" => Some(crate::data::processor::DataType::String),
                            "integer" | "int" => Some(crate::data::processor::DataType::Integer),
                            "float" | "number" => Some(crate::data::processor::DataType::Float),
                            "binary" | "raw" => Some(crate::data::processor::DataType::Binary),
                            _ => None,
                        }
                    })
                    .unwrap_or(crate::data::processor::DataType::Binary);
                
                let input_batch = crate::data::batch::DataBatch::from_bytes(input, data_type)?;
                
                // 将DataBatch序列化为JSON并放入PipelineContext
                let batch_json = serde_json::to_value(&input_batch)
                    .map_err(|e| crate::error::Error::Serialization(format!("无法序列化DataBatch: {}", e)))?;
                
                let mut context = pipeline::pipeline::PipelineContext::new();
                context.data.insert("batch".to_string(), batch_json);
                
                // 执行pipeline
                let result = pipeline.execute(context);
                
                // 从结果中提取处理后的数据
                let output = match result {
                    pipeline::pipeline::PipelineResult::Success { context, .. } |
                    pipeline::pipeline::PipelineResult::RecordSet { context, .. } |
                    pipeline::pipeline::PipelineResult::Value { context, .. } => {
                        // 尝试从context中获取处理后的batch数据
                        if let Some(processed_batch_json) = context.data.get("batch") {
                            // 反序列化处理后的DataBatch
                            let processed_batch: crate::data::batch::DataBatch = serde_json::from_value(processed_batch_json.clone())
                                .map_err(|e| crate::error::Error::Deserialization(format!("无法反序列化处理后的DataBatch: {}", e)))?;
                            processed_batch.to_bytes()?
                        } else {
                            // 如果没有batch数据，返回原始输入
                            input.to_vec()
                        }
                    },
                    pipeline::pipeline::PipelineResult::Error(msg) => {
                        return Err(crate::error::Error::Processing(format!("Pipeline处理失败: {}", msg)));
                    },
                    pipeline::pipeline::PipelineResult::ErrorWithContext { message, .. } => {
                        return Err(crate::error::Error::Processing(format!("Pipeline处理失败: {}", message)));
                    },
                    _ => {
                        // 其他结果类型，返回原始输入
                        input.to_vec()
                    }
                };
                
                log::debug!("Processed output size: {} bytes", output.len());
                Ok(output)
            }
            None => Err(crate::error::Error::NotFound(format!("Pipeline {}", pipeline_id)))
        }
    }
} 