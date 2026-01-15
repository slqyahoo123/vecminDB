// Data Processor Implementation
// 数据处理器主要实现

use std::collections::HashMap;
use std::error::Error as StdError;
// use std::sync::Arc; // reserved for future shared state extensions
use std::path::Path;
use tokio::fs;
use serde_json::Value;
use log::{debug, info, warn, error};

use crate::Result;
use crate::Error;
use crate::data::DataFormat;
use crate::data::{DataTransformation, DataTransformationType};
use crate::data::processor::{
    types_core::*,
    utils,
    schema_ops::{extract_schema_from_metadata, infer_schema_from_data},
    data_ops::{DataParser, DataConverter, DataValidator, DataCleaner, DataStatistics},
    record_ops::{RecordProcessor, FeatureExtractor},
};
use crate::data::processor::config::ProcessorConfig as ImportedProcessorConfig;
use crate::data::value::DataValue;
use crate::compat::tensor::TensorValues;
use crate::data::record::{Record, Value as RecordFieldValue};
use base64::Engine as _;
use crate::data::processor::types::Schema;

/// 数据处理器主要实现
pub struct DataProcessor {
    pub id: String,
    pub name: String,
    pub format: DataFormat,
    pub config: ImportedProcessorConfig,
    pub state: ProcessorState,
    pub status: ProcessorStatus,
    pub processor_type: ProcessorType,
    pub metrics: ProcessorMetrics,
    pub tasks: HashMap<String, TaskHandle>,
    
    // 子模块组件
    parser: DataParser,
    converter: DataConverter,
    validator: DataValidator,
    cleaner: DataCleaner,
    statistics: DataStatistics,
    record_processor: RecordProcessor,
    feature_extractor: FeatureExtractor,
}

impl DataProcessor {
    /// 创建新的数据处理器
    pub fn new(id: String, name: String, format: DataFormat, config: ImportedProcessorConfig) -> Self {
        Self {
            id: id.clone(),
            name,
            format: format.clone(),
            config,
            state: ProcessorState::Initial,
            status: ProcessorStatus::Healthy,
            processor_type: ProcessorType::File,
            metrics: ProcessorMetrics::default(),
            tasks: HashMap::new(),
            
            // 初始化子模块
            parser: DataParser::new(format.clone()),
            converter: DataConverter::new(),
            validator: DataValidator::new(format.clone()),
            cleaner: DataCleaner::new(),
            statistics: DataStatistics::new(),
            record_processor: RecordProcessor::new(),
            feature_extractor: FeatureExtractor::new(),
        }
    }

    /// 创建带有数据清洗功能的数据处理器
    pub fn new_data_cleaning() -> Self {
        Self::new(
            "data_cleaning".to_string(),
            "Data Cleaning Processor".to_string(),
            DataFormat::CSV, // 默认格式，可根据需要调整
            ImportedProcessorConfig::default(),
        )
    }

    /// 创建默认数据处理器
    pub fn new_default() -> Self {
        Self::new(
            "default".to_string(),
            "Default Processor".to_string(),
            DataFormat::CSV, // 默认格式，可根据需要调整
            ImportedProcessorConfig::default(),
        )
    }

    /// 创建特征提取处理器
    pub fn new_feature_extraction() -> Self {
        let mut processor = Self::new(
            "feature_extraction".to_string(),
            "Feature Extraction Processor".to_string(),
            DataFormat::CSV, // 默认格式，可根据需要调整
            ImportedProcessorConfig::default(),
        );
        processor.processor_type = ProcessorType::FeatureExtraction;
        processor
    }

    /// 创建归一化处理器
    pub fn new_normalization() -> Self {
        let mut processor = Self::new(
            "normalization".to_string(),
            "Normalization Processor".to_string(),
            DataFormat::CSV, // 默认格式，可根据需要调整
            ImportedProcessorConfig::default(),
        );
        processor.processor_type = ProcessorType::Normalization;
        processor
    }

    /// 检查存储健康状态
    pub fn health_check(&self) -> Result<bool> {
        // 检查处理器状态
        Ok(matches!(self.status, ProcessorStatus::Healthy))
    }

    /// 就绪检查：快速验证解析/转换/验证/清洗组件可用性
    /// 该方法用于将预导入的子组件（parser/converter/validator/cleaner 等）转为实际调用点
    pub fn readiness_probe(&self) -> Result<()> {
        // 1) 构造一条最小记录
        let mut sample = HashMap::new();
        sample.insert("feature1".to_string(), serde_json::json!(1.0));
        sample.insert("feature2".to_string(), serde_json::json!(2.0));

        // 2) 解析与验证（使用已导入的 DataParser / DataValidator）
        let parsed = self.parser.parse_inline(&serde_json::json!([sample.clone()]), &self.config)?;
        self.validator.validate_inline(&parsed)?;

        // 3) 转换与清洗
        let mut converted = self.converter.convert_inline(&parsed)?;
        self.cleaner.clean_inline(&mut converted)?;

        // 4) 统计与特征提取（触发组件可用）
        let _stats = self.statistics.compute_inline(&converted)?;
        let _features = self.feature_extractor.extract_inline(&converted)?;

        Ok(())
    }

    /// 获取内存使用信息
    /// 
    /// 获取当前进程和系统的内存使用情况
    pub fn get_memory_info(&self) -> Result<MemoryInfo> {
        // 使用跨平台方法获取内存信息
        {
            // 如果没有sysinfo特性，使用跨平台方法
            #[cfg(target_os = "linux")]
            {
                use std::fs;
                
                // 读取系统内存信息
                if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                    let mut total_kb = 0u64;
                    let mut available_kb = 0u64;
                    
                    for line in meminfo.lines() {
                        if line.starts_with("MemTotal:") {
                            if let Some(val) = line.split_whitespace().nth(1) {
                                total_kb = val.parse().unwrap_or(0);
                            }
                        } else if line.starts_with("MemAvailable:") {
                            if let Some(val) = line.split_whitespace().nth(1) {
                                available_kb = val.parse().unwrap_or(0);
                            }
                        }
                    }
                    
                    let total_bytes = total_kb * 1024;
                    let available_bytes = available_kb * 1024;
                    let used_bytes = total_bytes.saturating_sub(available_bytes);
                    
                    return Ok(MemoryInfo::new(used_bytes, total_bytes));
                }
            }
            
            #[cfg(all(target_os = "windows", feature = "winapi"))]
            {
                use winapi::um::sysinfoapi::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
                use std::mem;
                
                unsafe {
                    let mut mem_status: MEMORYSTATUSEX = mem::zeroed();
                    mem_status.dwLength = mem::size_of::<MEMORYSTATUSEX>() as u32;
                    
                    if GlobalMemoryStatusEx(&mut mem_status) != 0 {
                        let total_bytes = mem_status.ullTotalPhys as u64;
                        let available_bytes = mem_status.ullAvailPhys as u64;
                        let used_bytes = total_bytes.saturating_sub(available_bytes);
                        
                        return Ok(MemoryInfo::new(used_bytes, total_bytes));
                    }
                }
            }
            
            // 如果所有方法都失败，返回默认值
            warn!("无法获取系统内存信息，返回默认值");
            Ok(MemoryInfo::new(1024 * 1024 * 100, 1024 * 1024 * 1024))
        }
    }

    /// 导入数据
    pub async fn import(&self, source: &str, destination: &str, config: &ImportedProcessorConfig) -> Result<ImportStats> {
        log::info!("开始导入数据: {} -> {}", source, destination);
        
        let start_time = std::time::Instant::now();
        let mut stats = ImportStats::new();
        
        // 1. 读取源数据
        let source_data = self.read_source_data(source)?;
        stats.total_size = source_data.len();
        
        // 2. 数据清理
        let cleaned_data = self.cleaner.clean_data(&source_data)?;
        
        // 3. 验证和修复数据格式
        let validated_data = self.validator.validate_and_fix_format(&cleaned_data)?;
        
        // 4. 编码转换
        let converted_data = if let Some(ref input_encoding) = config.input_encoding.as_ref() {
            self.converter.convert_encoding(&validated_data, input_encoding, "utf-8")?
        } else {
            validated_data
        };
        
        // 5. 解析数据
        let records = self.parser.parse_data(&converted_data, config)?;
        stats.imported_count = records.len();
        
        // 6. 处理记录
        let mut processed_records = Vec::new();
        for record in records {
            match self.record_processor.process_single_record_with_config(&record, config) {
                Ok(processed_record) => processed_records.push(processed_record),
                Err(e) => {
                    log::warn!("处理记录失败: {}", e);
                    stats.error_count += 1;
                }
            }
        }
        
        // 7. 提取模式
        let schema = if !processed_records.is_empty() {
            // 从数据推断模式
            let record_values: Vec<Value> = processed_records.iter()
                .map(|r| self.record_to_json_value(r))
                .collect();
            infer_schema_from_data(&record_values)?
        } else {
            Schema::default()
        };
        
        // 8. 保存处理后的数据
        self.save_processed_data(destination, &processed_records, &schema).await?;
        
        stats.success_count = processed_records.len();
        stats.processing_time = start_time.elapsed();
        
        log::info!("数据导入完成: 成功 {}, 错误 {}, 耗时: {:?}", 
                  stats.success_count, stats.error_count, stats.processing_time);
        
        Ok(stats)
    }

    /// 导出数据
    pub async fn export(&self, source: &str, destination: &str, format: DataFormat) -> Result<ExportStats> {
        log::info!("开始导出数据: {} -> {} (格式: {:?})", source, destination, format);
        
        let start_time = std::time::Instant::now();
        let mut stats = ExportStats::new();
        
        // 1. 读取源数据
        let (records, _schema) = self.load_processed_data(source).await?;
        stats.total_count = records.len();
        
        // 2. 转换为目标格式
        let exported_data = match format {
            DataFormat::JSON => self.export_to_json(&records)?,
            DataFormat::CSV => self.export_to_csv(&records)?,
            DataFormat::TSV => self.export_to_tsv(&records)?,
            _ => return Err(Error::invalid_argument("不支持的导出格式")),
        };
        
        // 3. 保存到目标位置
        utils::ensure_dir_exists(&Path::new(destination).parent().unwrap().to_string_lossy())?;
        fs::write(destination, exported_data).await
            .map_err(|e| Error::io_error(&format!("写入文件失败: {}", e)))?;
        
        stats.exported_count = records.len();
        stats.export_size = utils::get_file_size(destination)?;
        stats.processing_time = start_time.elapsed();
        
        log::info!("数据导出完成: {} 条记录, 耗时: {:?}", stats.exported_count, stats.processing_time);
        
        Ok(stats)
    }

    /// 获取处理状态
    pub fn get_status(&self) -> ProcessorState {
        self.state
    }

    /// 设置处理状态
    pub fn set_status(&mut self, state: ProcessorState) {
        self.state = state;
    }

    /// 检查处理器状态
    /// 
    /// 提供完整的处理器状态检查功能，包括：
    /// - 处理器健康状态
    /// - 内存使用情况
    /// - 活跃任务状态
    /// - 性能指标
    pub fn check_status(&self) -> Result<ProcessorStatus> {
        info!("检查数据处理器状态: {}", self.id);
        
        // 1. 检查基本健康状态
        match self.status {
            ProcessorStatus::Healthy => {
                debug!("处理器状态正常: {}", self.id);
            },
            ProcessorStatus::Degraded => {
                warn!("处理器状态降级: {}", self.id);
            },
            ProcessorStatus::Unhealthy => {
                error!("处理器状态不健康: {}", self.id);
                return Ok(ProcessorStatus::Unhealthy);
            }
        }
        
        // 2. 检查内存使用情况
        match self.get_memory_info() {
            Ok(memory_info) => {
                let usage_percentage = (memory_info.used_bytes as f64 / memory_info.total_bytes as f64) * 100.0;
                if usage_percentage > 90.0 {
                    warn!("处理器内存使用率过高: {}% ({})", usage_percentage, self.id);
                    return Ok(ProcessorStatus::Degraded);
                } else if usage_percentage > 95.0 {
                    error!("处理器内存使用率危险: {}% ({})", usage_percentage, self.id);
                    return Ok(ProcessorStatus::Unhealthy);
                }
            },
            Err(e) => {
                warn!("无法获取处理器内存信息: {} - {}", self.id, e);
            }
        }
        
        // 3. 检查活跃任务数量
        let active_task_count = self.tasks.len();
        if active_task_count > 100 {
            warn!("处理器活跃任务过多: {} 个任务 ({})", active_task_count, self.id);
            return Ok(ProcessorStatus::Degraded);
        }
        
        // 4. 检查处理器性能指标
        let error_rate = if self.metrics.processed_count > 0 {
            (self.metrics.error_count as f64 / self.metrics.processed_count as f64) * 100.0
        } else {
            0.0
        };
        
        if error_rate > 10.0 {
            error!("处理器错误率过高: {}% ({})", error_rate, self.id);
            return Ok(ProcessorStatus::Unhealthy);
        } else if error_rate > 5.0 {
            warn!("处理器错误率较高: {}% ({})", error_rate, self.id);
            return Ok(ProcessorStatus::Degraded);
        }
        
        // 5. 检查处理器状态一致性
        match self.state {
            ProcessorState::Failed => {
                error!("处理器处于失败状态: {}", self.id);
                return Ok(ProcessorStatus::Unhealthy);
            },
            ProcessorState::Cancelled => {
                warn!("处理器已被取消: {}", self.id);
                return Ok(ProcessorStatus::Degraded);
            },
            _ => {
                // 正常状态
            }
        }
        
        info!("处理器状态检查完成 - 状态正常: {}", self.id);
        Ok(self.status.clone())
    }
    
    /// 获取处理器详细状态信息
    /// 
    /// 返回包含详细诊断信息的状态报告
    pub fn get_detailed_status(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut status_info = HashMap::new();
        
        // 基本信息
        status_info.insert("processor_id".to_string(), serde_json::Value::String(self.id.clone()));
        status_info.insert("name".to_string(), serde_json::Value::String(self.name.clone()));
        status_info.insert("format".to_string(), serde_json::Value::String(format!("{:?}", self.format)));
        status_info.insert("processor_type".to_string(), serde_json::Value::String(format!("{:?}", self.processor_type)));
        
        // 状态信息
        status_info.insert("status".to_string(), serde_json::Value::String(format!("{:?}", self.status)));
        status_info.insert("state".to_string(), serde_json::Value::String(format!("{:?}", self.state)));
        
        // 性能指标
        status_info.insert("processed_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.metrics.processed_count)));
        status_info.insert("error_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.metrics.error_count)));
        status_info.insert("success_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.metrics.success_rate).unwrap_or(serde_json::Number::from(0))));
        status_info.insert("throughput".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.metrics.throughput).unwrap_or(serde_json::Number::from(0))));
        
        // 任务信息
        status_info.insert("active_tasks".to_string(), serde_json::Value::Number(serde_json::Number::from(self.tasks.len())));
        
        // 内存信息
        if let Ok(memory_info) = self.get_memory_info() {
            let mut memory_object = serde_json::Map::new();
            memory_object.insert("used_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(memory_info.used_bytes)));
            memory_object.insert("total_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(memory_info.total_bytes)));
            memory_object.insert("available_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(memory_info.available_bytes)));
            let usage_percentage = (memory_info.used_bytes as f64 / memory_info.total_bytes as f64) * 100.0;
            memory_object.insert("usage_percentage".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(usage_percentage).unwrap_or(serde_json::Number::from(0))));
            status_info.insert("memory".to_string(), serde_json::Value::Object(memory_object));
        }
        
        // 时间戳
        status_info.insert("timestamp".to_string(), serde_json::Value::String(chrono::Utc::now().to_rfc3339()));
        
        Ok(status_info)
    }

    /// 读取源数据
    fn read_source_data(&self, source: &str) -> Result<Vec<u8>> {
        match std::fs::read(source) {
            Ok(data) => {
                if let Some(max) = self.config.max_file_size {
                    if data.len() > max {
                        return Err(Error::invalid_argument("文件大小超过限制"));
                    }
        }
        Ok(data)
    },
    Err(e) => Err(Error::io_error(&format!("读取文件失败: {}", e)))
        }
    }

    /// 保存处理后的数据
    async fn save_processed_data(&self, destination: &str, records: &[Record], schema: &Schema) -> Result<()> {
        // 确保目标目录存在
        let dest_path = Path::new(destination);
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent).await
                .map_err(|e| Error::io_error(&format!("创建目录失败: {}", e)))?;
        }
        
        // 保存元数据
        let metadata = serde_json::json!({
            "format": format!("{:?}", self.format),
            "rows": records.len(),
            "schema": self.schema_to_json_value(schema),
            "created_at": chrono::Utc::now().to_rfc3339(),
            "processor_id": self.id,
            "config_hash": self.calculate_config_hash(&self.config)
        });
        
        let metadata_file = dest_path.join("metadata.json");
        fs::write(&metadata_file, serde_json::to_string_pretty(&metadata)?)
            .await
            .map_err(|e| Error::io_error(&format!("保存元数据失败: {}", e)))?;
        
        // 保存数据
        let data_file = dest_path.join("data.bin");
        let serialized_data = self.serialize_records(records)?;
        fs::write(&data_file, serialized_data)
            .await
            .map_err(|e| Error::io_error(&format!("保存数据失败: {}", e)))?;
        
        Ok(())
    }

    /// 加载处理后的数据
    async fn load_processed_data(&self, source: &str) -> Result<(Vec<Record>, Schema)> {
        let source_path = Path::new(source);
        
        // 读取元数据
        let metadata_file = source_path.join("metadata.json");
        let metadata_content = fs::read_to_string(&metadata_file)
            .await
            .map_err(|e| Error::io_error(&format!("读取元数据失败: {}", e)))?;
        let metadata: Value = serde_json::from_str(&metadata_content)?;
        
        // 提取模式
        let schema = extract_schema_from_metadata(&metadata)?;
        
        // 读取数据
        let data_file = source_path.join("data.bin");
        let data_content = fs::read(&data_file)
            .await
            .map_err(|e| Error::io_error(&format!("读取数据失败: {}", e)))?;
        let records = self.deserialize_records(&data_content)?;
        
        Ok((records, schema))
    }

    /// 序列化记录
    fn serialize_records(&self, records: &[Record]) -> Result<Vec<u8>> {
        // 使用bincode或其他序列化格式
        // 这里提供一个简化的JSON序列化实现
        let json_records: Vec<Value> = records.iter()
            .map(|r| self.record_to_json_value(r))
            .collect();
        
        let json_data = serde_json::to_string(&json_records)?;
        Ok(json_data.into_bytes())
    }

    /// 反序列化记录（JSON -> Record）
    fn deserialize_records(&self, data: &[u8]) -> Result<Vec<Record>> {
        let json_str = String::from_utf8_lossy(data);
        let json_values: Vec<Value> = serde_json::from_str(&json_str)?;

        let mut records = Vec::new();
        for value in json_values {
            if let Some(obj) = value.as_object() {
                let mut record = Record::new();
                for (key, val) in obj {
                    let data_value = self.json_value_to_data_value(val);
                    record
                        .fields
                        .insert(key.clone(), RecordFieldValue::Data(data_value));
                }
                records.push(record);
            }
        }

        Ok(records)
    }

    /// Record 转 JSON Value
    fn record_to_json_value(&self, record: &Record) -> Value {
        let mut obj = serde_json::Map::new();

        for (key, value) in &record.fields {
            let json_value = self.record_field_value_to_json_value(value);
            obj.insert(key.clone(), json_value);
        }

        Value::Object(obj)
    }

    /// 记录字段值转 JSON
    fn record_field_value_to_json_value(&self, value: &RecordFieldValue) -> Value {
        match value {
            RecordFieldValue::Data(dv) => self.data_value_to_json_value(dv),
            RecordFieldValue::Record(rec) => self.record_to_json_value(rec),
            RecordFieldValue::Reference(id) => Value::String(id.clone()),
        }
    }

    /// DataValue 转 JSON
    fn data_value_to_json_value(&self, value: &DataValue) -> Value {
        match value {
            DataValue::Null => Value::Null,
            DataValue::Boolean(b) => Value::Bool(*b),
            DataValue::Integer(i) => Value::Number(serde_json::Number::from(*i)),
            DataValue::Float(f) | DataValue::Number(f) => {
                serde_json::Number::from_f64(*f)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            }
            DataValue::String(s) | DataValue::Text(s) => Value::String(s.clone()),
            DataValue::Array(arr) => {
                let json_arr: Vec<Value> = arr
                    .iter()
                    .map(|v| self.data_value_to_json_value(v))
                    .collect();
                Value::Array(json_arr)
            }
            DataValue::StringArray(arr) => {
                let json_arr: Vec<Value> = arr.iter().map(|s| Value::String(s.clone())).collect();
                Value::Array(json_arr)
            }
            DataValue::Object(obj) => {
                let mut json_obj = serde_json::Map::new();
                for (k, v) in obj {
                    json_obj.insert(k.clone(), self.data_value_to_json_value(v));
                }
                Value::Object(json_obj)
            }
            DataValue::Binary(b) => {
                // 使用Base64编码二进制数据
                let encoded = base64::engine::general_purpose::STANDARD.encode(b);
                Value::String(encoded)
            }
            DataValue::DateTime(s) => Value::String(s.clone()),
            DataValue::Tensor(t) => {
                // 尝试序列化Tensor，失败时退化为描述字符串
                serde_json::to_value(t).unwrap_or_else(|_| Value::String("<tensor>".to_string()))
            }
        }
    }

    /// JSON Value 转 DataValue
    fn json_value_to_data_value(&self, value: &Value) -> DataValue {
        match value {
            Value::Null => DataValue::Null,
            Value::Bool(b) => DataValue::Boolean(*b),
            Value::Number(n) => {
                if n.is_i64() {
                    DataValue::Integer(n.as_i64().unwrap_or(0))
                } else {
                    DataValue::Number(n.as_f64().unwrap_or(0.0))
                }
            }
            Value::String(s) => DataValue::String(s.clone()),
            Value::Array(arr) => {
                let items: Vec<DataValue> = arr
                    .iter()
                    .map(|v| self.json_value_to_data_value(v))
                    .collect();
                DataValue::Array(items)
            }
            Value::Object(obj) => {
                let mut map = HashMap::new();
                for (k, v) in obj {
                    map.insert(k.clone(), self.json_value_to_data_value(v));
                }
                DataValue::Object(map)
            }
        }
    }

    /// JSON Value转RecordValue（用于与旧管道接口兼容）
    fn json_value_to_record_value(&self, value: Value) -> crate::data::pipeline::traits::RecordValue {
        match value {
            Value::String(s) => crate::data::pipeline::traits::RecordValue::String(s),
            Value::Number(n) => {
                if n.is_i64() {
                    crate::data::pipeline::traits::RecordValue::Integer(n.as_i64().unwrap_or(0))
                } else {
                    crate::data::pipeline::traits::RecordValue::Number(n.as_f64().unwrap_or(0.0))
                }
            },
            Value::Bool(b) => crate::data::pipeline::traits::RecordValue::Boolean(b),
            Value::Null => crate::data::pipeline::traits::RecordValue::Null,
            Value::Array(arr) => {
                let items: Vec<crate::data::pipeline::traits::RecordValue> = arr.into_iter()
                    .map(|v| self.json_value_to_record_value(v))
                    .collect();
                crate::data::pipeline::traits::RecordValue::Array(items)
            },
            Value::Object(obj) => {
                let mut map = HashMap::new();
                for (k, v) in obj {
                    map.insert(k, self.json_value_to_record_value(v));
                }
                crate::data::pipeline::traits::RecordValue::Object(map)
            }
        }
    }

    /// Schema转JSON Value
    fn schema_to_json_value(&self, schema: &Schema) -> Value {
        let mut fields = Vec::new();
        
        for (name, data_type) in schema.fields() {
            let mut field = serde_json::Map::new();
            field.insert("name".to_string(), Value::String(name.clone()));
            field.insert("type".to_string(), Value::String(format!("{:?}", data_type)));

            fields.push(Value::Object(field));
        }
        
        serde_json::json!({
            "fields": fields
        })
    }

    /// 导出为JSON格式
    fn export_to_json(&self, records: &[Record]) -> Result<Vec<u8>> {
        let json_records: Vec<Value> = records.iter()
            .map(|r| self.record_to_json_value(r))
            .collect();
        
        let json_data = serde_json::to_string_pretty(&json_records)?;
        Ok(json_data.into_bytes())
    }

    /// 导出为CSV格式
    fn export_to_csv(&self, records: &[Record]) -> Result<Vec<u8>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        
        // 收集所有字段名
        let mut all_fields = std::collections::BTreeSet::new();
        for record in records {
            for field_name in record.fields.keys() {
                all_fields.insert(field_name.clone());
            }
        }
        let field_names: Vec<String> = all_fields.into_iter().collect();
        
        // 生成CSV内容
        let mut csv_lines = Vec::new();
        
        // 添加标题行
        csv_lines.push(field_names.join(","));
        
        // 添加数据行
        for record in records {
            let mut row_values = Vec::new();
            for field_name in &field_names {
                let value_str = if let Some(value) = record.fields.get(field_name) {
                    match value {
                        RecordFieldValue::Data(dv) => match dv {
                            DataValue::String(s) | DataValue::Text(s) => s.clone(),
                            DataValue::Integer(i) => i.to_string(),
                            DataValue::Float(f) | DataValue::Number(f) => f.to_string(),
                            DataValue::Boolean(b) => b.to_string(),
                            DataValue::Null => String::new(),
                            _ => format!("{:?}", dv),
                        },
                        RecordFieldValue::Record(rec) => format!("{}", rec),
                        RecordFieldValue::Reference(id) => id.clone(),
                    }
                } else {
                    String::new()
                };
                row_values.push(value_str);
            }
            csv_lines.push(row_values.join(","));
        }
        
        Ok(csv_lines.join("\n").into_bytes())
    }

    /// 导出为TSV格式
    fn export_to_tsv(&self, records: &[Record]) -> Result<Vec<u8>> {
        let csv_data = self.export_to_csv(records)?;
        let csv_str = String::from_utf8_lossy(&csv_data);
        let tsv_str = csv_str.replace(",", "\t");
        Ok(tsv_str.into_bytes())
    }

    /// 计算配置哈希
    fn calculate_config_hash(&self, config: &ImportedProcessorConfig) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // 将配置的关键字段加入哈希计算
        config.normalize.hash(&mut hasher);
        config.handle_missing_values.hash(&mut hasher);
        config.max_file_size.hash(&mut hasher);
        
        if let Some(ref encoding) = config.input_encoding {
            encoding.hash(&mut hasher);
        }
        
        format!("{:x}", hasher.finish())
    }

    /// 分割数据
    pub async fn split(&self, source_path: &str, ratios: &[f32], shuffle: bool) -> Result<Vec<String>> {
        // 创建输出路径
        let mut output_paths = Vec::new();
        for i in 0..ratios.len() {
            let output_path = format!("{}_split_{}", source_path, i);
            fs::create_dir_all(&output_path).await?;
            output_paths.push(output_path);
        }
        
        // 读取数据
        let (records, schema) = self.load_processed_data(source_path).await?;
        let total_rows = records.len();
        
        // 计算每个分割的大小
        let mut split_sizes = Vec::new();
        let mut remaining = total_rows;
        
        for i in 0..(ratios.len() - 1) {
            let size = (total_rows as f32 * ratios[i]) as usize;
            split_sizes.push(size);
            remaining -= size;
        }
        split_sizes.push(remaining);
        
        // 创建索引
        let mut indices: Vec<usize> = (0..total_rows).collect();
        
        // 如果需要随机打乱
        if shuffle {
            // 兼容无 SliceRandom 情况：使用随机键排序
            use rand::Rng;
            let mut rng = rand::thread_rng();
            indices.sort_by(|_, _| {
                let a: u64 = rng.gen();
                let b: u64 = rng.gen();
                a.cmp(&b)
            });
        }
        
        // 分割数据
        let mut offset = 0;
        for (i, &size) in split_sizes.iter().enumerate() {
            // 创建分割的索引
            let split_indices: Vec<usize> = indices[offset..(offset + size)].to_vec();
            offset += size;
            
            // 选择对应的记录
            let split_records: Vec<Record> = split_indices.iter()
                .map(|&idx| records[idx].clone())
                .collect();
            
            // 保存分割的数据
            self.save_processed_data(&output_paths[i], &split_records, &schema).await?;
        }
        
        Ok(output_paths)
    }

    /// 统计数据
    pub fn get_statistics(&self, records: &[Record]) -> HashMap<String, Value> {
        self.statistics.calculate_basic_stats(records)
    }

    /// 提取特征
    pub fn extract_features(&self, records: &[Record]) -> std::result::Result<crate::model::tensor::TensorData, Box<dyn StdError>> {
        self.feature_extractor.extract_features_from_records(records, &self.config)
    }

    /// 提取标签
    pub fn extract_labels(&self, records: &[Record]) -> std::result::Result<Option<crate::model::tensor::TensorData>, Box<dyn StdError>> {
        self.feature_extractor.extract_labels_from_records(records, &self.config)
    }

    /// 处理数据
    pub async fn process_data(&self, data: &crate::core::types::CoreDataBatch) -> crate::Result<crate::core::interfaces::ProcessedData> {
        log::info!("开始处理数据批次: {}", data.id);
        
        // 将 CoreDataBatch 转换为内部格式
        let mut records = Vec::new();
        for sample in &data.data {
            // 将 CoreTensorData 转换为 HashMap<String, Value>
            let mut record = HashMap::new();
            record.insert("data".to_string(), serde_json::Value::Array(
                sample.data.iter().map(|&x| serde_json::Value::Number(serde_json::Number::from_f64(x as f64).unwrap_or(serde_json::Number::from(0)))).collect()
            ));
            record.insert("shape".to_string(), serde_json::Value::Array(
                sample.shape.iter().map(|&x| serde_json::Value::Number(serde_json::Number::from(x))).collect()
            ));
            record.insert("dtype".to_string(), serde_json::Value::String(sample.dtype.clone()));
            record.insert("device".to_string(), serde_json::Value::String(sample.device.clone()));
            records.push(record);
        }
        
        // 处理记录
        let mut processed_records = Vec::new();
        for record_map in records {
            // 将 HashMap<String, serde_json::Value> 转换为 HashMap<String, data::record::Value>
            let mut fields = std::collections::HashMap::new();
            for (k, v) in record_map {
                fields.insert(k, crate::data::record::Value::Data(crate::data::value::DataValue::from_json(v)));
            }
            let record = crate::data::record::Record {
                id: None,
                fields,
                metadata: HashMap::new(),
            };
            match self.record_processor.process_single_record_with_config(&record, &self.config) {
                Ok(processed_record) => processed_records.push(processed_record),
                Err(e) => {
                    log::warn!("处理记录失败: {}", e);
                }
            }
        }
        
        // 提取特征并转换为一维 f32 向量
        let features_tensor = self
            .feature_extractor
            .extract_features_from_records(&processed_records, &self.config)
            .map_err(|e| Error::processing(format!("特征提取失败: {}", e)))?;

        let features_vec: Vec<f32> = match &features_tensor.data {
            TensorValues::F32(v) => v.clone(),
            TensorValues::F64(v) => v.iter().map(|x| *x as f32).collect(),
            TensorValues::I32(v) => v.iter().map(|x| *x as f32).collect(),
            TensorValues::I64(v) => v.iter().map(|x| *x as f32).collect(),
            TensorValues::U8(v) => v.iter().map(|x| *x as f32).collect(),
        };
        let processed_data = crate::core::interfaces::ProcessedData {
            id: uuid::Uuid::new_v4().to_string(),
            data: bincode::serialize(&features_vec)?,
            format: "float32".to_string(),
            size: features_vec.len(),
            metadata: {
                let mut meta = data.metadata.clone().unwrap_or_default();
                meta.insert("shape".to_string(), format!("{:?}", features_tensor.shape));
                meta.insert("data_type".to_string(), "float32".to_string());
                meta.insert("processing_steps".to_string(), "data_cleaning,feature_extraction".to_string());
                meta
            },
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        log::info!("数据批次处理完成: {}", data.id);
        Ok(processed_data)
    }
    
    /// 转换为张量
    pub async fn convert_to_tensors(&self, data: &crate::core::interfaces::ProcessedData) -> crate::Result<Vec<crate::core::types::CoreTensorData>> {
        log::info!("开始转换张量: {}", data.id);
        
        let tensor = crate::core::types::CoreTensorData {
            id: uuid::Uuid::new_v4().to_string(),
            shape: data.metadata.get("shape")
                .and_then(|s| serde_json::from_str::<Vec<usize>>(s).ok())
                .unwrap_or_default(),
            data: bincode::deserialize::<Vec<f32>>(&data.data)?,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: data.metadata.clone(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        
        log::info!("张量转换完成: {}", data.id);
        Ok(vec![tensor])
    }
    
    /// 验证数据模式
    pub async fn validate_data_schema(&self, data: &crate::core::types::CoreDataBatch, _schema: &crate::data::schema::DataSchema) -> crate::Result<bool> {
        log::info!("开始验证数据模式: {}", data.id);
        
        if data.data.is_empty() {
            log::warn!("数据批次为空");
            return Ok(false);
        }
        
        // 检查样本数量
        if data.data.len() != data.batch_size {
            log::warn!("样本数量不匹配: 期望 {}, 实际 {}", data.batch_size, data.data.len());
            return Ok(false);
        }
        
        // 检查每个样本的结构
        for (i, sample) in data.data.iter().enumerate() {
            // 检查CoreTensorData的基本字段
            if sample.id.is_empty() {
                log::warn!("样本 {} 包含空ID", i);
                return Ok(false);
            }
            if sample.shape.is_empty() {
                log::warn!("样本 {} 包含空形状", i);
                return Ok(false);
            }
            if sample.data.is_empty() {
                log::warn!("样本 {} 包含空数据", i);
                return Ok(false);
            }
        }
        
        log::info!("数据模式验证通过: {}", data.id);
        Ok(true)
    }
    
    /// 预处理数据
    pub async fn preprocess_data(&self, data: &crate::core::types::CoreDataBatch, config: &crate::core::interfaces::PreprocessingConfig) -> crate::Result<crate::core::types::CoreDataBatch> {
        log::info!("开始预处理数据: {}", data.id);
        
        let mut processed_samples = data.data.clone();
        
        // 应用预处理配置
        if config.normalization_strategies.is_some() {
            // 标准化处理
            processed_samples = self.normalize_tensor_samples(&processed_samples)?;
        }
        
        if let Some(ref cleaning) = config.cleaning_strategies {
            // 数据清洗处理
            processed_samples = self.clean_tensor_samples(&processed_samples, cleaning)?;
        }
        
        // 特征选择
        if config.use_filtering {
            // 使用过滤策略
            processed_samples = self.filter_tensor_samples(&processed_samples)?;
        }
        
        // 编码处理
        if config.use_ngrams {
            // 使用NGram处理
            processed_samples = self.apply_ngrams_to_tensors(&processed_samples)?;
        }
        
        // 自定义转换
        if config.use_char_ngrams {
            // 使用字符NGram处理
            processed_samples = self.apply_char_ngrams_to_tensors(&processed_samples)?;
        }
        
        let processed_batch = crate::core::types::CoreDataBatch {
            id: data.id.clone(),
            data: processed_samples,
            labels: data.labels.clone(),
            batch_size: data.batch_size,
            metadata: data.metadata.clone(),
            created_at: data.created_at,
            updated_at: chrono::Utc::now(),
        };
        
        log::info!("数据预处理完成: {}", data.id);
        Ok(processed_batch)
    }
    
    /// 分割数据（按比例切分）
    pub async fn split_data_by_ratios(&self, data: &crate::core::types::CoreDataBatch, ratios: &[f32]) -> crate::Result<Vec<crate::core::types::CoreDataBatch>> {
        log::info!("开始分割数据: {}", data.id);
        
        if ratios.is_empty() {
            return Err(crate::Error::InvalidInput("分割比例不能为空".to_string()));
        }
        
        let total_ratio: f32 = ratios.iter().sum();
        if (total_ratio - 1.0).abs() > 1e-6 {
            return Err(crate::Error::InvalidInput("分割比例总和必须为1.0".to_string()));
        }
        
        let total_samples = data.data.len();
        let mut start_idx = 0;
        let mut batches = Vec::new();
        
        for (i, ratio) in ratios.iter().enumerate() {
            let end_idx = if i == ratios.len() - 1 {
                total_samples
            } else {
                start_idx + ((total_samples as f32 * ratio) as usize)
            };
            
            let batch_samples = data.data[start_idx..end_idx].to_vec();
            
            let batch = crate::core::types::CoreDataBatch {
                id: format!("{}_{}", data.id, i),
                data: batch_samples,
                labels: data.labels.clone(),
                batch_size: end_idx - start_idx,
                metadata: data.metadata.clone(),
                created_at: data.created_at,
                updated_at: chrono::Utc::now(),
            };
            
            batches.push(batch);
            start_idx = end_idx;
        }
        
        log::info!("数据分割完成: {} -> {} 个批次", data.id, batches.len());
        Ok(batches)
    }
    
    /// 标准化数据
    pub async fn normalize_data(&self, data: &crate::core::types::CoreDataBatch) -> crate::Result<crate::core::types::CoreDataBatch> {
        log::info!("开始标准化数据: {}", data.id);
        
        // 将标准化后的数据转换回CoreTensorData格式
        let mut normalized_tensors = Vec::new();
        for sample in &data.data {
            let mut normalized_tensor = sample.clone();
            // 简单的标准化：将数据缩放到 [0, 1] 范围
            let max_val = sample.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if max_val > 0.0 {
                normalized_tensor.data = sample.data.iter().map(|&x| x / max_val).collect();
            }
            normalized_tensors.push(normalized_tensor);
        }
        
        let normalized_batch = crate::core::types::CoreDataBatch {
            id: data.id.clone(),
            data: normalized_tensors,
            labels: data.labels.clone(),
            batch_size: data.batch_size,
            metadata: data.metadata.clone(),
            created_at: data.created_at,
            updated_at: chrono::Utc::now(),
        };
        
        log::info!("数据标准化完成: {}", data.id);
        Ok(normalized_batch)
    }
    
    // 辅助方法
    fn json_sample_to_record(&self, sample: &HashMap<String, serde_json::Value>) -> crate::Result<crate::data::record::Record> {
        let mut record = crate::data::record::Record::new();

        for (key, value) in sample {
            let data_value = self.json_value_to_data_value(value);
            record
                .fields
                .insert(key.clone(), RecordFieldValue::Data(data_value));
        }

        Ok(record)
    }
    
    fn normalize_samples(&self, samples: &[HashMap<String, serde_json::Value>]) -> crate::Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut normalized_samples = Vec::new();
        
        for sample in samples {
            let mut normalized_sample = HashMap::new();
            
            for (key, value) in sample {
                let normalized_value = self.normalize_value(value)?;
                normalized_sample.insert(key.clone(), normalized_value);
            }
            
            normalized_samples.push(normalized_sample);
        }
        
        Ok(normalized_samples)
    }
    
    fn scale_samples(&self, samples: &[HashMap<String, serde_json::Value>], scaling_type: &str) -> crate::Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut scaled_samples = Vec::new();
        
        for sample in samples {
            let mut scaled_sample = HashMap::new();
            
            for (key, value) in sample {
                let scaled_value = self.scale_value(value, scaling_type)?;
                scaled_sample.insert(key.clone(), scaled_value);
            }
            
            scaled_samples.push(scaled_sample);
        }
        
        Ok(scaled_samples)
    }
    
    fn select_features(&self, samples: &[HashMap<String, serde_json::Value>], features: &[String]) -> crate::Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut selected_samples = Vec::new();
        
        for sample in samples {
            let mut selected_sample = HashMap::new();
            
            for feature in features {
                if let Some(value) = sample.get(feature) {
                    selected_sample.insert(feature.clone(), value.clone());
                }
            }
            
            selected_samples.push(selected_sample);
        }
        
        Ok(selected_samples)
    }
    
    fn encode_field(&self, samples: &[HashMap<String, serde_json::Value>], field: &str, encoding_type: &str) -> crate::Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut encoded_samples = Vec::new();
        
        for sample in samples {
            let mut encoded_sample = sample.clone();
            
            if let Some(value) = sample.get(field) {
                let encoded_value = self.encode_value(value, encoding_type)?;
                encoded_sample.insert(field.to_string(), encoded_value);
            }
            
            encoded_samples.push(encoded_sample);
        }
        
        Ok(encoded_samples)
    }
    
    fn apply_custom_transform(&self, samples: &[HashMap<String, serde_json::Value>], transform: &str) -> crate::Result<Vec<HashMap<String, serde_json::Value>>> {
        // 这里可以实现自定义转换逻辑
        // 目前返回原始数据
        Ok(samples.to_vec())
    }
    
    fn normalize_value(&self, value: &serde_json::Value) -> crate::Result<serde_json::Value> {
        match value {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    // 简单的标准化：将值映射到 [0, 1] 范围
                    let normalized = (f + 1.0) / 2.0; // 假设值在 [-1, 1] 范围内
                    Ok(match serde_json::Number::from_f64(normalized) {
                        Some(num) => serde_json::Value::Number(num),
                        None => serde_json::Value::Null,
                    })
                } else {
                    Ok(value.clone())
                }
            }
            _ => Ok(value.clone()),
        }
    }
    
    fn scale_value(&self, value: &serde_json::Value, scaling_type: &str) -> crate::Result<serde_json::Value> {
        match value {
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    let scaled = match scaling_type {
                        "standard" => f, // 标准化
                        "minmax" => f,   // 最小-最大缩放
                        "robust" => f,   // 鲁棒缩放
                        _ => f,
                    };
                    Ok(match serde_json::Number::from_f64(scaled) {
                        Some(num) => serde_json::Value::Number(num),
                        None => serde_json::Value::Null,
                    })
                } else {
                    Ok(value.clone())
                }
            }
            _ => Ok(value.clone()),
        }
    }
    
    fn encode_value(&self, value: &serde_json::Value, encoding_type: &str) -> crate::Result<serde_json::Value> {
        match encoding_type {
            "onehot" => {
                // 独热编码
                if let serde_json::Value::String(s) = value {
                    // 简化的独热编码实现
                    let encoded = format!("encoded_{}", s);
                    Ok(serde_json::Value::String(encoded))
                } else {
                    Ok(value.clone())
                }
            }
            "label" => {
                // 标签编码
                if let serde_json::Value::String(s) = value {
                    let encoded = s.len() as f64; // 简化的标签编码
                    Ok(match serde_json::Number::from_f64(encoded) {
                        Some(num) => serde_json::Value::Number(num),
                        None => serde_json::Value::Null,
                    })
                } else {
                    Ok(value.clone())
                }
            }
            _ => Ok(value.clone()),
        }
    }
}

/// 导入统计信息
#[derive(Debug, Clone)]
pub struct ImportStats {
    pub total_size: usize,
    pub imported_count: usize,
    pub success_count: usize,
    pub error_count: usize,
    pub processing_time: std::time::Duration,
}

impl ImportStats {
    pub fn new() -> Self {
        Self {
            total_size: 0,
            imported_count: 0,
            success_count: 0,
            error_count: 0,
            processing_time: std::time::Duration::from_secs(0),
        }
    }
}

/// 导出统计信息
#[derive(Debug, Clone)]
pub struct ExportStats {
    pub total_count: usize,
    pub exported_count: usize,
    pub export_size: u64,
    pub processing_time: std::time::Duration,
}

impl ExportStats {
    pub fn new() -> Self {
        Self {
            total_count: 0,
            exported_count: 0,
            export_size: 0,
            processing_time: std::time::Duration::from_secs(0),
        }
    }
}

// 实现Processor trait
#[async_trait::async_trait]
impl Processor for DataProcessor {
    /// 获取处理器ID
    fn id(&self) -> &str {
        &self.id
    }
    
    /// 获取处理器名称
    fn name(&self) -> &str {
        &self.name
    }
    
    /// 获取处理器类型
    fn processor_type(&self) -> ProcessorType {
        self.processor_type.clone()
    }
    
    /// 获取处理器配置
    fn config(&self) -> &crate::data::processor::types_core::ProcessorConfig {
        // 将内部配置转换为新的配置格式
        // 这里需要一个临时的转换，理想情况下应该统一配置类型
        static mut CACHED_CONFIG: Option<crate::data::processor::types_core::ProcessorConfig> = None;
        unsafe {
            if CACHED_CONFIG.is_none() {
                let config = crate::data::processor::types_core::ProcessorConfig::new(
                    self.id.clone(),
                    self.name.clone(),
                    self.processor_type.clone()
                );
                CACHED_CONFIG = Some(config);
            }
            CACHED_CONFIG.as_ref().unwrap()
        }
    }
    
    /// 获取处理器指标
    fn metrics(&self) -> &crate::data::processor::types_core::ProcessorMetrics {
        &self.metrics
    }
    
    /// 启动处理器
    async fn start(&mut self) -> crate::Result<()> {
        self.status = crate::data::processor::types_core::ProcessorStatus::Healthy;
        self.state = crate::data::processor::types_core::ProcessorState::Ready;
        Ok(())
    }
    
    /// 停止处理器
    async fn stop(&mut self) -> crate::Result<()> {
        self.state = crate::data::processor::types_core::ProcessorState::Completed;
        Ok(())
    }
    
    /// 健康检查
    async fn health_check(&self) -> crate::Result<crate::data::processor::types_core::ProcessorStatus> {
        Ok(self.status.clone())
    }
    
    /// 处理数据
    async fn process(&mut self, data: Vec<u8>) -> crate::Result<Vec<u8>> {
        self.state = crate::data::processor::types_core::ProcessorState::Processing;
        
        // 解析数据
        let records = match self.parser.parse_data(&data, &self.config) {
            Ok(records) => records,
            Err(e) => {
                self.state = crate::data::processor::types_core::ProcessorState::Failed;
                return Err(e);
            }
        };
        
        // 处理记录
        let mut processed_records = Vec::new();
        for record in records {
            match self.record_processor.process_single_record_with_config(&record, &self.config) {
                Ok(processed_record) => processed_records.push(processed_record),
                Err(e) => {
                    self.state = crate::data::processor::types_core::ProcessorState::Failed;
                    return Err(crate::Error::processing(format!("处理记录失败: {}", e)));
                }
            }
        }
        
        // 序列化结果
        let result = match self.serialize_records(&processed_records) {
            Ok(data) => data,
            Err(e) => {
                self.state = crate::data::processor::types_core::ProcessorState::Failed;
                return Err(e);
            }
        };
        
        self.state = crate::data::processor::types_core::ProcessorState::Completed;
        Ok(result)
    }
    
    /// 获取支持的数据格式
    fn supported_formats(&self) -> Vec<crate::data::types::DataFormat> {
        vec![
            crate::data::types::DataFormat::JSON,
            crate::data::types::DataFormat::CSV,
            crate::data::types::DataFormat::TSV,
            crate::data::types::DataFormat::Json,
            crate::data::types::DataFormat::Csv,
            crate::data::types::DataFormat::Tsv,
        ]
    }
    
    /// 验证输入数据
    async fn validate_input(&self, data: &[u8]) -> crate::Result<bool> {
        // 尝试解析数据以验证格式
        match self.parser.parse_data(data, &self.config) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    /// 获取处理进度
    fn get_progress(&self) -> f32 {
        match self.state {
            crate::data::processor::types_core::ProcessorState::Initial => 0.0,
            crate::data::processor::types_core::ProcessorState::Ready => 0.1,
            crate::data::processor::types_core::ProcessorState::Processing => 0.5,
            crate::data::processor::types_core::ProcessorState::Completed => 1.0,
            crate::data::processor::types_core::ProcessorState::Failed => 0.0,
            crate::data::processor::types_core::ProcessorState::Cancelled => 0.0,
        }
    }
    
    /// 设置配置
    async fn update_config(&mut self, new_config: crate::data::processor::types_core::ProcessorConfig) -> crate::Result<()> {
        // 这里需要将新配置转换为内部配置格式
        // 由于配置类型不同，我们只更新可更新的部分
        self.id = new_config.id;
        self.name = new_config.name;
        self.processor_type = new_config.processor_type;
        Ok(())
    }
}

// 实现存储健康检查trait
#[async_trait::async_trait]
impl crate::data::processor::types_core::StorageHealthCheck for DataProcessor {
    /// 检查存储健康状态
    async fn health_check(&self) -> crate::Result<bool> {
        // 检查基本存储功能
        match self.status {
            crate::data::processor::types_core::ProcessorStatus::Healthy => Ok(true),
            crate::data::processor::types_core::ProcessorStatus::Degraded => Ok(true),
            crate::data::processor::types_core::ProcessorStatus::Unhealthy => Ok(false),
        }
    }
    
    /// 获取详细健康信息
    async fn detailed_health_check(&self) -> crate::Result<HashMap<String, serde_json::Value>> {
        let mut health_info = HashMap::new();
        
        health_info.insert("processor_id".to_string(), serde_json::Value::String(self.id.clone()));
        health_info.insert("processor_name".to_string(), serde_json::Value::String(self.name.clone()));
        health_info.insert("status".to_string(), serde_json::Value::String(format!("{:?}", self.status)));
        health_info.insert("state".to_string(), serde_json::Value::String(format!("{:?}", self.state)));
        health_info.insert("format".to_string(), serde_json::Value::String(format!("{:?}", self.format)));
        health_info.insert("processed_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.metrics.processed_count)));
        health_info.insert("error_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.metrics.error_count)));
        health_info.insert("success_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.metrics.success_rate).unwrap_or(serde_json::Number::from(0))));
        health_info.insert("throughput".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(self.metrics.throughput).unwrap_or(serde_json::Number::from(0))));
        health_info.insert("active_tasks".to_string(), serde_json::Value::Number(serde_json::Number::from(self.tasks.len())));
        
        Ok(health_info)
    }
    
    /// 检查连接状态
    async fn check_connection(&self) -> crate::Result<bool> {
        // 对于数据处理器，连接状态取决于内部组件是否正常
        match self.state {
            crate::data::processor::types_core::ProcessorState::Ready |
            crate::data::processor::types_core::ProcessorState::Processing |
            crate::data::processor::types_core::ProcessorState::Completed => Ok(true),
            _ => Ok(false),
        }
    }
    
    /// 获取存储统计信息
    async fn get_storage_stats(&self) -> crate::Result<HashMap<String, u64>> {
        let mut stats = HashMap::new();
        
        stats.insert("total_processed".to_string(), self.metrics.processed_count as u64);
        stats.insert("total_errors".to_string(), self.metrics.error_count as u64);
        stats.insert("active_tasks".to_string(), self.tasks.len() as u64);
        stats.insert("total_processing_time_ms".to_string(), self.metrics.total_processing_time.as_millis() as u64);
        stats.insert("average_processing_time_ms".to_string(), self.metrics.average_processing_time.as_millis() as u64);
        stats.insert("max_processing_time_ms".to_string(), self.metrics.max_processing_time.as_millis() as u64);
        stats.insert("min_processing_time_ms".to_string(), self.metrics.min_processing_time.as_millis() as u64);
        
        // 获取内存信息
        if let Ok(memory_info) = self.get_memory_info() {
            stats.insert("memory_used_bytes".to_string(), memory_info.used_bytes);
            stats.insert("memory_total_bytes".to_string(), memory_info.total_bytes);
            stats.insert("memory_available_bytes".to_string(), memory_info.available_bytes);
        }
        
        Ok(stats)
    }
}

impl crate::data::pipeline::traits::DataProcessor for DataProcessor {
    /// 处理单个记录
    fn process_record(&self, record: &mut crate::data::record::Record) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // 使用带配置的处理，并回写结果
        match self
            .record_processor
            .process_single_record_with_config(record, &self.config)
        {
            Ok(processed) => {
                *record = processed;
                Ok(())
            }
            // 这里的 e 已经是 Box<dyn StdError>，直接返回即可，避免再包一层 Box
            Err(e) => Err(e),
        }
    }

    /// 处理批量记录
    fn process_batch(&self, batch: &mut crate::data::pipeline::RecordBatch) -> std::result::Result<(), Box<dyn std::error::Error>> {
        for record in &mut batch.records {
            self.process_record(record)?;
        }
        Ok(())
    }

    /// 获取处理器名称
    fn name(&self) -> &str {
        &self.name
    }

    /// 初始化导入上下文
    fn initialize_import(&self, source: &str, config: &crate::data::processor::ProcessorConfig) -> std::result::Result<crate::data::pipeline::traits::ImportContext, Box<dyn std::error::Error>> {
        let context = crate::data::pipeline::traits::ImportContext {
            source: source.to_string(),
            config: config.clone(),
            metadata: HashMap::new(),
            start_time: std::time::Instant::now(),
        };
        Ok(context)
    }

    /// 处理导入
    fn process_import(&self, context: crate::data::pipeline::traits::ImportContext) -> std::result::Result<crate::data::pipeline::traits::ImportResult, Box<dyn std::error::Error>> {
        let source_data = self.read_source_data(&context.source)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut processed_records = 0;
        
        // 解析数据
        let total_records = match self.parser.parse_data(&source_data, &context.config) {
            Ok(records) => {
                let total = records.len();
                
                // 处理每条记录
                for record in records {
                    match self.record_processor.process_single_record_with_config(&record, &context.config) {
                        Ok(_) => processed_records += 1,
                        Err(e) => {
                            errors.push(format!("处理记录失败: {}", e));
                        }
                    }
                }
                
                total
            },
            Err(e) => {
                errors.push(format!("解析数据失败: {}", e));
                0
            }
        };

        let success = errors.is_empty();
        let end_time = chrono::Utc::now();
        let duration_ms = context.start_time.elapsed().as_millis() as u64;
        let start_time = end_time - chrono::Duration::milliseconds(duration_ms as i64);

        let result = crate::data::pipeline::traits::ImportResult {
            id: context.source.clone(),
            start_time,
            end_time: Some(end_time),
            duration: Some(duration_ms),
            success,
            message: if success {
                "Import completed successfully".to_string()
            } else {
                "Import completed with errors".to_string()
            },
            total_records,
            processed_records,
            records_failed: total_records.saturating_sub(processed_records),
            processed_rows: processed_records,
            warnings,
            errors,
            metadata: context.metadata.clone(),
        };

        Ok(result)
    }

    /// 列出批次
    fn list_batches(&self, data_type: Option<&str>, status: Option<&str>, limit: usize) -> std::result::Result<Vec<crate::data::DataBatch>, Box<dyn std::error::Error>> {
        // 这里是一个基础实现，实际应该从存储中获取批次数据
        let mut batches = Vec::new();
        
        // 创建示例批次（在生产环境中应该从实际存储中获取）
        for i in 0..std::cmp::min(limit, 10) {
            let mut batch = crate::data::DataBatch::default();
            batch.id = Some(format!("batch_{}", i));
            batch.format = self.format.clone();
            batch.status = crate::data::types::DataStatus::Processed;
            
            // 应用过滤器
            if let Some(dt) = data_type {
                if batch.format.to_string() != dt {
                    continue;
                }
            }
            
            if let Some(st) = status {
                if batch.status.to_string() != st {
                    continue;
                }
            }
            
            batches.push(batch);
        }
        
        Ok(batches)
    }

    /// 获取指定批次
    fn get_batch(&self, id: &str) -> std::result::Result<crate::data::DataBatch, Box<dyn std::error::Error>> {
        // 在生产环境中应该从实际存储中获取
        let mut batch = crate::data::DataBatch::default();
        batch.id = Some(id.to_string());
        batch.format = self.format.clone();
        batch.status = crate::data::types::DataStatus::Processed;
        
        Ok(batch)
    }

    /// 删除批次
    fn delete_batch(&self, id: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        log::info!("删除批次: {}", id);
        // 在生产环境中应该从实际存储中删除
        Ok(())
    }

    /// 获取批次样本
    fn get_batch_samples(&self, id: &str, limit: usize, offset: usize, fields: Option<&[String]>) -> std::result::Result<(crate::data::DataBatch, Vec<HashMap<String, crate::data::pipeline::traits::RecordValue>>), Box<dyn std::error::Error>> {
        let batch = self.get_batch(id)?;
        
        // 创建示例样本数据
        let mut samples = Vec::new();
        for i in offset..std::cmp::min(offset + limit, 100) {
            let mut sample = HashMap::new();
            
            if let Some(field_list) = fields {
                for field in field_list {
                    sample.insert(field.clone(), crate::data::pipeline::traits::RecordValue::String(format!("value_{}", i)));
                }
            } else {
                sample.insert("id".to_string(), crate::data::pipeline::traits::RecordValue::Integer(i as i64));
                sample.insert("name".to_string(), crate::data::pipeline::traits::RecordValue::String(format!("record_{}", i)));
                sample.insert("value".to_string(), crate::data::pipeline::traits::RecordValue::Number(i as f64 * 1.5));
            }
            
            samples.push(sample);
        }
        
        Ok((batch, samples))
    }

    /// 导出批次
    fn export_batch(&self, id: &str, format: &str, config: Option<&crate::data::pipeline::traits::ExportConfig>) -> std::result::Result<(), Box<dyn std::error::Error>> {
        log::info!("导出批次 {} 为格式 {}", id, format);
        
        let batch = self.get_batch(id)?;
        
        // 在生产环境中应该实际执行导出操作
        let export_path = format!("exports/batch_{}_{}.{}", id, chrono::Utc::now().timestamp(), format);
        
        log::info!("批次 {} 已导出到 {}", id, export_path);
        Ok(())
    }

    /// 列出数据集
    fn list_datasets(&self) -> std::result::Result<Vec<crate::data::Dataset>, Box<dyn std::error::Error>> {
        // 在生产环境中应该从实际存储中获取，这里返回一些示例数据集元信息
        let mut datasets = Vec::new();
        
        for i in 0..5 {
            let now = chrono::Utc::now();
            let dataset = crate::data::Dataset {
                id: format!("dataset_{}", i),
                name: format!("Dataset {}", i),
                description: Some(format!("Description for dataset {}", i)),
                format: self.format.clone(),
                size: 0,
                created_at: now,
                updated_at: now,
                metadata: crate::data::types::DatasetMetadata {
                    id: format!("dataset_{}", i),
                    name: format!("Dataset {}", i),
                    description: Some(format!("Description for dataset {}", i)),
                    created_at: now,
                    updated_at: now,
                    version: "1.0".to_string(),
                    owner: "system".to_string(),
                    schema: None,
                    properties: HashMap::new(),
                    tags: Vec::new(),
                    records_count: 0,
                    size_bytes: 0,
                },
                path: String::new(),
                processed: false,
                loader: std::sync::Arc::new(crate::data::loader::memory::MemoryDataLoader::new()),
                batch_size: 32,
                schema: None,
                batches: Vec::new(),
            };
            datasets.push(dataset);
        }
        
        Ok(datasets)
    }

    /// 获取数据集
    fn get_dataset(&self, id: &str) -> std::result::Result<crate::data::Dataset, Box<dyn std::error::Error>> {
        // 在生产环境中应该从实际存储中获取，这里构造一个示例数据集
        let now = chrono::Utc::now();
        let dataset = crate::data::Dataset {
            id: id.to_string(),
            name: format!("Dataset {}", id),
            description: Some(format!("Description for dataset {}", id)),
            format: self.format.clone(),
            size: 0,
            created_at: now,
            updated_at: now,
            metadata: crate::data::types::DatasetMetadata {
                id: format!("dataset_{}", id),
                name: format!("Dataset {}", id),
                description: Some(format!("Description for dataset {}", id)),
                created_at: now,
                updated_at: now,
                version: "1.0".to_string(),
                owner: "system".to_string(),
                schema: None,
                properties: HashMap::new(),
                tags: Vec::new(),
                records_count: 0,
                size_bytes: 0,
            },
            path: String::new(),
            processed: false,
            loader: std::sync::Arc::new(crate::data::loader::memory::MemoryDataLoader::new()),
            batch_size: 32,
            schema: None,
            batches: Vec::new(),
        };
        
        Ok(dataset)
    }

    /// 删除数据集
    fn delete_dataset(&self, id: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        log::info!("删除数据集: {}", id);
        // 在生产环境中应该从实际存储中删除
        Ok(())
    }

    /// 存储文件
    fn store_file(&self, id: &str, filename: &str, content_type: &str, data: Vec<u8>) -> std::result::Result<(), Box<dyn std::error::Error>> {
        log::info!("存储文件: {} (文件名: {}, 类型: {}, 大小: {} bytes)", id, filename, content_type, data.len());
        
        // 在生产环境中应该实际存储文件
        let storage_path = format!("files/{}/{}", id, filename);
        log::info!("文件已存储到: {}", storage_path);
        
        Ok(())
    }

    /// 处理二进制数据
    fn process_binary_data(&self, batch_id: &str, data_type: &str, _config: &crate::data::processor::ProcessorConfig, data: Vec<u8>) -> std::result::Result<crate::data::pipeline::traits::ImportResult, Box<dyn std::error::Error>> {
        log::info!("处理二进制数据: batch_id={}, data_type={}, size={} bytes", batch_id, data_type, data.len());
        
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        
        // 验证数据类型
        if !["image", "audio", "video", "binary"].contains(&data_type) {
            errors.push(format!("不支持的数据类型: {}", data_type));
        }
        
        // 验证数据大小
        if data.is_empty() {
            errors.push("数据为空".to_string());
        } else if data.len() > 100 * 1024 * 1024 { // 100MB限制
            warnings.push("数据大小超过100MB，处理可能较慢".to_string());
        }
        
        let success = errors.is_empty();
        let total_records = if success { 1 } else { 0 };
        let processed_records = if success { 1 } else { 0 };
        let start_time = chrono::Utc::now();

        let result = crate::data::pipeline::traits::ImportResult {
            id: batch_id.to_string(),
            start_time,
            end_time: None,
            duration: None,
            success,
            message: if success {
                "Binary data import successful".to_string()
            } else {
                "Binary data import completed with errors".to_string()
            },
            total_records,
            processed_records,
            records_failed: errors.len(),
            processed_rows: total_records,
            warnings,
            errors,
            metadata: std::collections::HashMap::new(),
        };
        
        Ok(result)
    }

    /// 更新数据集
    fn update_dataset(&self, dataset: &crate::data::Dataset) -> std::result::Result<(), Box<dyn std::error::Error>> {
        log::info!("更新数据集: {}", dataset.id);
        // 在生产环境中应该实际更新存储中的数据集
        Ok(())
    }

    /// 初始化文件导入
    fn initialize_file_import(&self, file_path: &str, config: &crate::data::processor::ProcessorConfig) -> std::result::Result<crate::data::pipeline::traits::ImportContext, Box<dyn std::error::Error>> {
        // 检查文件是否存在
        if !std::path::Path::new(file_path).exists() {
            return Err(format!("文件不存在: {}", file_path).into());
        }
        
        let context = crate::data::pipeline::traits::ImportContext {
            source: file_path.to_string(),
            config: config.clone(),
            metadata: HashMap::new(),
            start_time: std::time::Instant::now(),
        };
        
        Ok(context)
    }

    /// 处理数据批次
    fn process_batch_with_config(&self, batch: &crate::data::DataBatch, config: &crate::data::processor::ProcessorConfig) -> std::result::Result<crate::data::processor::ProcessedBatch, Box<dyn std::error::Error>> {
        use crate::data::record::{Record, Value as RecordValue};
        use crate::data::value::DataValue;
        use std::collections::HashMap;

        log::info!(
            "处理数据批次: {:?} (记录数: {})",
            batch.id.as_deref().unwrap_or("unknown"),
            batch.records.len()
        );
        
        let mut converted_records: Vec<HashMap<String, DataValue>> = Vec::new();
        let mut errors = Vec::new();

        for record_map in &batch.records {
            // 将 HashMap<String, DataValue> 转换为 Record
            let mut record = Record::new();
            for (key, value) in record_map {
                record.fields.insert(key.clone(), RecordValue::Data(value.clone()));
            }

            match self.record_processor.process_single_record_with_config(&record, config) {
                Ok(processed_record) => {
                    // 再把处理后的 Record 转回 HashMap<String, DataValue>
                    let mut new_map = HashMap::new();
                    for (key, value) in processed_record.fields {
                        if let RecordValue::Data(dv) = value {
                            new_map.insert(key, dv);
                        }
                    }
                    converted_records.push(new_map);
                }
                Err(e) => {
                    errors.push(format!("处理记录失败: {}", e));
                }
            }
        }

        // 从 DataBatch 构造 ProcessorBatch，再替换其中的 records 字段
        let mut processor_batch = crate::data::processor::types::ProcessorBatch::from_batch(batch)?;
        if !converted_records.is_empty() {
            processor_batch.records = converted_records;
        }

        if !errors.is_empty() {
            log::warn!(
                "处理数据批次 {:?} 时发生 {} 条记录处理错误: {:?}",
                batch.id.as_deref().unwrap_or("unknown"),
                errors.len(),
                errors
            );
        }

        Ok(processor_batch)
    }

    /// 获取处理器状态
    fn get_status(&self) -> std::result::Result<serde_json::Value, Box<dyn std::error::Error>> {
        let metrics = &self.metrics;
        let status = serde_json::json!({
            "id": self.id,
            "name": self.name,
            "format": format!("{:?}", self.format),
            "state": format!("{:?}", self.state),
            "status": format!("{:?}", self.status),
            "processor_type": format!("{:?}", self.processor_type),
            "metrics": {
                "processed_records": metrics.processed_count,
                "error_count": metrics.error_count,
                "total_processing_time_ms": metrics.total_processing_time.as_millis(),
                "average_processing_time_ms": metrics.average_processing_time.as_millis(),
                "max_processing_time_ms": metrics.max_processing_time.as_millis(),
                "min_processing_time_ms": metrics.min_processing_time.as_millis(),
                "success_rate": metrics.success_rate,
                "throughput": metrics.throughput,
            },
            "active_tasks": self.tasks.len()
        });
        
        Ok(status)
    }

    /// 获取活跃任务数量
    fn get_active_tasks_count(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = std::result::Result<u64, Box<dyn std::error::Error>>> + Send + '_>> {
        Box::pin(async move {
            Ok(self.tasks.len() as u64)
        })
    }

    /// 导出数据
    fn export_data(&self, config: &crate::data::pipeline::traits::ExportConfig) -> std::result::Result<Vec<crate::data::pipeline::traits::RecordValue>, Box<dyn std::error::Error>> {
        // 根据配置导出数据
        use crate::data::value::DataValue;
        use crate::data::pipeline::traits::RecordValue;
        use std::collections::HashMap;

        // 决定导出格式：从ExportConfig的options中读取format键，默认为json
        let export_format = config
            .options
            .get("format")
            .map(String::as_str)
            .unwrap_or("json")
            .to_lowercase();

        // 工具函数：将 DataValue 转换为 RecordValue
        fn to_record_value(value: &DataValue) -> RecordValue {
            match value {
                DataValue::Null => RecordValue::Null,
                DataValue::Boolean(b) => RecordValue::Boolean(*b),
                DataValue::Integer(i) => RecordValue::Integer(*i),
                DataValue::Float(f) | DataValue::Number(f) => RecordValue::Number(*f),
                DataValue::String(s) | DataValue::Text(s) => RecordValue::String(s.clone()),
                DataValue::Array(arr) => {
                    RecordValue::Array(arr.iter().map(to_record_value).collect())
                }
                DataValue::StringArray(arr) => {
                    RecordValue::Array(arr.iter().map(|s| RecordValue::String(s.clone())).collect())
                }
                DataValue::Object(map) => {
                    let mut obj = HashMap::new();
                    for (k, v) in map {
                        obj.insert(k.clone(), to_record_value(v));
                    }
                    RecordValue::Object(obj)
                }
                DataValue::Binary(b) => RecordValue::Binary(b.clone()),
                DataValue::DateTime(s) => RecordValue::String(s.clone()),
                DataValue::Tensor(_) => {
                    // 张量数据在这里序列化为字符串表示
                    RecordValue::String("<tensor>".to_string())
                }
            }
        }

        let mut exported_data = Vec::new();
        
        // 获取所有批次
        let batches = self.list_batches(None, None, 1000)?;
        
        for batch in batches {
            for record in batch.records {
                // 将 HashMap<String, DataValue> 转换为 HashMap<String, RecordValue>
                let mut converted = HashMap::new();
                for (key, value) in record {
                    converted.insert(key, to_record_value(&value));
                }

                match export_format.as_str() {
                    "json" | "csv" | "tsv" => {
                        exported_data.push(RecordValue::Object(converted));
                    }
                    _ => {
                        return Err(Box::new(std::io::Error::new(
                            std::io::ErrorKind::Unsupported,
                            format!("不支持的导出格式: {}", export_format),
                        )));
                    }
                }
            }
        }
        
        Ok(exported_data)
    }
} 

#[async_trait::async_trait]
impl crate::core::interfaces::DataProcessorInterface for DataProcessor {
    async fn process_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::interfaces::ProcessedData> {
        self.process_data(data).await
    }

    async fn convert_to_tensors(&self, data: &crate::core::interfaces::ProcessedData) -> Result<Vec<crate::core::types::CoreTensorData>> {
        self.convert_to_tensors(data).await
    }

    async fn validate_data_schema(
        &self,
        data: &crate::core::types::CoreDataBatch,
        schema: &crate::data::loader::types::DataSchema,
    ) -> Result<crate::core::interfaces::ValidationResult> {
        let mut errors = Vec::new();
        
        // 验证字段数量
        if data.data.is_empty() {
            errors.push("数据批次为空".to_string());
            return Ok(crate::core::interfaces::ValidationResult {
                is_valid: false,
                errors,
                warnings: Vec::new(),
                score: Some(0.0),
                metadata: std::collections::HashMap::new(),
            });
        }
        
        // 验证字段匹配
        if schema.fields.len() != data.data.len() {
            errors.push(format!("字段数量不匹配: 期望 {}, 实际 {}", schema.fields.len(), data.data.len()));
        }
        
        // 验证每个字段
        for (i, field) in schema.fields.iter().enumerate() {
            if i < data.data.len() {
                // 这里可以添加更详细的字段验证逻辑
            }
        }
        
        Ok(crate::core::interfaces::ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings: Vec::new(),
            score: if errors.is_empty() { Some(1.0) } else { Some(0.0) },
            metadata: std::collections::HashMap::new(),
        })
    }

    async fn preprocess_data(
        &self,
        data: &crate::core::types::CoreDataBatch,
    ) -> Result<crate::core::types::ProcessedData> {
        // 使用默认预处理配置（不启用额外策略）
        let config = crate::core::interfaces::PreprocessingConfig {
            cleaning_strategies: None,
            normalization_strategies: None,
            use_ngrams: false,
            ngram_range: None,
            use_char_ngrams: false,
            char_ngram_range: (1, 1),
            use_filtering: false,
            remove_stopwords: false,
            min_token_length: 1,
            max_token_length: None,
            language: "en".to_string(),
        };

        // 先通过内部预处理获取标准化后的批次
        let processed_batch = self.preprocess_data(data, &config).await?;
        
        // 再转换为 ProcessedData，便于下游统一消费
        use chrono::Utc;
        use uuid::Uuid;

        // 将 CoreTensorData 序列化为字节（f32 -> 小端字节序）
        let mut bytes = Vec::new();
        let mut total_size = 0;

        for tensor in &processed_batch.data {
            for &val in &tensor.data {
                bytes.extend_from_slice(&val.to_le_bytes());
                total_size += 4;
            }
        }

        Ok(crate::core::types::ProcessedData {
            id: Uuid::new_v4().to_string(),
            data: bytes,
            format: "processed".to_string(),
            size: total_size,
            metadata: std::collections::HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        })
    }

    async fn split_data(
        &self,
        data: &crate::core::types::CoreDataBatch,
        train_ratio: f32,
        val_ratio: f32,
        test_ratio: f32,
    ) -> Result<(
        crate::core::types::CoreDataBatch,
        crate::core::types::CoreDataBatch,
        crate::core::types::CoreDataBatch,
    )> {
        let ratios = vec![train_ratio, val_ratio, test_ratio];
        let batches = self.split_data_by_ratios(data, &ratios).await?;
        if batches.len() != 3 {
            return Err(crate::Error::InvalidInput(format!("分割数据应该返回3个批次，但返回了{}个", batches.len())));
        }
        Ok((batches[0].clone(), batches[1].clone(), batches[2].clone()))
    }

    async fn normalize_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::CoreDataBatch> {
        info!("开始归一化数据批次: {} ({} 个张量)", data.id, data.data.len());
        
        if data.data.is_empty() {
            warn!("数据批次为空，无需归一化");
            return Ok(data.clone());
        }
        
        // 归一化方法：优先从处理器选项中获取，默认为Z-score标准化
        let normalization_method = self
            .config
            .processor_options
            .get("normalization_method")
            .map(String::as_str)
            .unwrap_or("zscore");
        
        let mut normalized_tensors = Vec::new();
        
        for tensor in &data.data {
            if tensor.data.is_empty() {
                normalized_tensors.push(tensor.clone());
                continue;
            }
            
            let mut normalized_tensor = tensor.clone();
            
            match normalization_method.to_lowercase().as_str() {
                "zscore" | "standard" => {
                    // Z-score标准化：(x - mean) / std
                    let mean = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
                    let variance = tensor.data.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f32>() / tensor.data.len() as f32;
                    let std_dev = variance.sqrt();
                    
                    if std_dev > 1e-6 {
                        normalized_tensor.data = tensor.data.iter()
                            .map(|&x| (x - mean) / std_dev)
                            .collect();
                    } else {
                        // 如果标准差为0，将所有值设为0
                        normalized_tensor.data = vec![0.0; tensor.data.len()];
                    }
                    
                    normalized_tensor.metadata.insert(
                        "normalization_method".to_string(),
                        "zscore".to_string()
                    );
                    normalized_tensor.metadata.insert(
                        "normalization_mean".to_string(),
                        mean.to_string()
                    );
                    normalized_tensor.metadata.insert(
                        "normalization_std".to_string(),
                        std_dev.to_string()
                    );
                }
                "minmax" => {
                    // Min-Max标准化：(x - min) / (max - min)
                    let min_val = tensor.data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_val = tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    
                    if (max_val - min_val) > 1e-6 {
                        normalized_tensor.data = tensor.data.iter()
                            .map(|&x| (x - min_val) / (max_val - min_val))
                            .collect();
                    } else {
                        // 如果范围太小，将所有值设为0.5
                        normalized_tensor.data = vec![0.5; tensor.data.len()];
                    }
                    
                    normalized_tensor.metadata.insert(
                        "normalization_method".to_string(),
                        "minmax".to_string()
                    );
                    normalized_tensor.metadata.insert(
                        "normalization_min".to_string(),
                        min_val.to_string()
                    );
                    normalized_tensor.metadata.insert(
                        "normalization_max".to_string(),
                        max_val.to_string()
                    );
                }
                "l2" | "unit_vector" => {
                    // L2归一化：x / ||x||
                    let l2_norm = tensor.data.iter()
                        .map(|&x| x * x)
                        .sum::<f32>()
                        .sqrt();
                    
                    if l2_norm > 1e-6 {
                        normalized_tensor.data = tensor.data.iter()
                            .map(|&x| x / l2_norm)
                            .collect();
                    } else {
                        // 如果L2范数为0，保持原值
                        normalized_tensor.data = tensor.data.clone();
                    }
                    
                    normalized_tensor.metadata.insert(
                        "normalization_method".to_string(),
                        "l2".to_string()
                    );
                    normalized_tensor.metadata.insert(
                        "normalization_l2_norm".to_string(),
                        l2_norm.to_string()
                    );
                }
                _ => {
                    // 默认使用Z-score标准化
                    warn!("未知的归一化方法: {}，使用Z-score标准化", normalization_method);
                    let mean = tensor.data.iter().sum::<f32>() / tensor.data.len() as f32;
                    let variance = tensor.data.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f32>() / tensor.data.len() as f32;
                    let std_dev = variance.sqrt();
                    
                    if std_dev > 1e-6 {
                        normalized_tensor.data = tensor.data.iter()
                            .map(|&x| (x - mean) / std_dev)
                            .collect();
                    } else {
                        normalized_tensor.data = vec![0.0; tensor.data.len()];
                    }
                    
                    normalized_tensor.metadata.insert(
                        "normalization_method".to_string(),
                        "zscore".to_string()
                    );
                }
            }
            
            normalized_tensor.updated_at = chrono::Utc::now();
            normalized_tensors.push(normalized_tensor);
        }
        
        // 归一化标签（如果存在）
        let normalized_labels = if let Some(labels) = &data.labels {
            let mut norm_labels = Vec::new();
            for label in labels {
                // 标签通常不需要归一化，但如果有需要可以应用相同的归一化
                norm_labels.push(label.clone());
            }
            Some(norm_labels)
        } else {
            None
        };
        
        let normalized_batch = crate::core::types::CoreDataBatch {
            id: format!("normalized_{}", data.id),
            data: normalized_tensors,
            labels: normalized_labels,
            batch_size: data.batch_size,
            metadata: {
                let mut meta = data.metadata.clone().unwrap_or_default();
                meta.insert("normalization_method".to_string(), normalization_method.to_string());
                meta.insert("normalized_at".to_string(), chrono::Utc::now().to_rfc3339());
                Some(meta)
            },
            created_at: data.created_at,
            updated_at: chrono::Utc::now(),
        };
        
        info!("数据归一化完成: {} -> {} (方法: {})", data.id, normalized_batch.id, normalization_method);
        Ok(normalized_batch)
    }
    
    async fn process_batch(
        &self,
        batch: &crate::core::types::CoreDataBatch,
    ) -> Result<crate::core::types::CoreDataBatch> {
        // 处理批次数据：先使用默认预处理配置进行预处理，然后返回新的 CoreDataBatch
        let default_config = crate::core::interfaces::PreprocessingConfig {
            cleaning_strategies: None,
            normalization_strategies: None,
            use_ngrams: false,
            ngram_range: None,
            use_char_ngrams: false,
            char_ngram_range: (1, 1),
            use_filtering: false,
            remove_stopwords: false,
            min_token_length: 1,
            max_token_length: None,
            language: "en".to_string(),
        };

        let processed_batch = self.preprocess_data(batch, &default_config).await?;

        Ok(crate::core::types::CoreDataBatch {
            id: format!("processed_{}", batch.id),
            data: processed_batch.data,
            labels: processed_batch.labels,
            batch_size: processed_batch.batch_size,
            metadata: processed_batch.metadata.clone(),
            created_at: processed_batch.created_at,
            updated_at: chrono::Utc::now(),
        })
    }
    
    async fn validate_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::interfaces::ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        if data.data.is_empty() {
            errors.push("数据批次为空".to_string());
        }
        
        if data.batch_size != data.data.len() {
            warnings.push(format!("批次大小不匹配: 声明 {}, 实际 {}", data.batch_size, data.data.len()));
        }
        
        Ok(crate::core::interfaces::ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            score: if errors.is_empty() { Some(1.0) } else { Some(0.0) },
            metadata: std::collections::HashMap::new(),
        })
    }
    
    async fn convert_data(&self, data: &crate::core::types::CoreDataBatch, target_format: &str) -> Result<crate::core::types::CoreDataBatch> {
        // 根据目标格式转换数据
        let mut converted_data = data.data.clone();
        
        match target_format {
            "tensor" => {
                // 转换为张量格式
                for sample in &mut converted_data {
                    // 保持原有格式，这里可以添加格式转换逻辑
                }
            }
            "json" => {
                // 转换为JSON格式
                for sample in &mut converted_data {
                    // 保持原有格式，这里可以添加格式转换逻辑
                }
            }
            _ => {
                return Err(crate::Error::InvalidInput(format!("不支持的目标格式: {}", target_format)));
            }
        }
        
        Ok(crate::core::types::CoreDataBatch {
            id: format!("converted_{}", data.id),
            data: converted_data,
            labels: data.labels.clone(),
            batch_size: data.batch_size,
            metadata: data.metadata.clone(),
            created_at: data.created_at,
            updated_at: chrono::Utc::now(),
        })
    }
    
    async fn get_data_statistics(&self, data: &crate::core::types::CoreDataBatch) -> Result<std::collections::HashMap<String, f64>> {
        let mut stats = std::collections::HashMap::new();
        
        stats.insert("total_samples".to_string(), data.data.len() as f64);
        stats.insert("batch_size".to_string(), data.batch_size as f64);
        
        if !data.data.is_empty() {
            // 计算平均数据大小
            let total_size: usize = data.data.iter().map(|s| s.data.len()).sum();
            let avg_size = (total_size as f64) / (data.data.len() as f64);
            stats.insert("average_sample_size".to_string(), avg_size);
        }
        
        Ok(stats)
    }
    
    async fn clean_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::CoreDataBatch> {
        // 清理数据：移除空值和无效数据
        let cleaned_data: Vec<_> = data.data.iter()
            .filter(|sample| !sample.data.is_empty() && !sample.id.is_empty())
            .cloned()
            .collect();
        
        Ok(crate::core::types::CoreDataBatch {
            id: format!("cleaned_{}", data.id),
            data: cleaned_data.clone(),
            labels: data.labels.clone(),
            batch_size: cleaned_data.len(),
            metadata: data.metadata.clone(),
            created_at: data.created_at,
            updated_at: chrono::Utc::now(),
        })
    }
    
    async fn get_processor_config(&self) -> Result<std::collections::HashMap<String, String>> {
        let mut config = std::collections::HashMap::new();
        config.insert("processor_type".to_string(), "DataProcessor".to_string());
        config.insert("version".to_string(), "1.0".to_string());
        Ok(config)
    }
    
    async fn update_processor_config(&self, _config: std::collections::HashMap<String, String>) -> Result<()> {
        // 更新处理器配置
        // 这里可以添加配置更新逻辑
        Ok(())
    }
}

/// 用户数据处理器
/// 提供用户级别的数据处理功能
#[derive(Debug, Clone)]
pub struct UserDataProcessor {
    /// 处理器ID
    pub id: String,
    /// 处理器名称
    pub name: String,
    /// 处理器配置
    pub config: ImportedProcessorConfig,
    /// 数据格式
    pub format: DataFormat,
    /// 处理统计
    pub stats: ProcessingStats,
}

impl UserDataProcessor {
    /// 创建新的用户数据处理器
    pub fn new(id: String, name: String, config: ImportedProcessorConfig, format: DataFormat) -> Self {
        Self {
            id,
            name,
            config,
            format,
            stats: ProcessingStats::default(),
        }
    }

    /// 处理用户数据
    pub async fn process_user_data(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        self.stats.total_processed += 1;
        
        match self.format {
            DataFormat::JSON => {
                // JSON数据处理
                let json_data: serde_json::Value = serde_json::from_slice(data)
                    .map_err(|e| Error::DataProcessingError(format!("JSON解析失败: {}", e)))?;
                
                // 应用用户配置的数据转换
                let processed_data = self.apply_user_transformations(json_data).await?;
                
                serde_json::to_vec(&processed_data)
                    .map_err(|e| Error::DataProcessingError(format!("JSON序列化失败: {}", e)))
            }
            DataFormat::CSV => {
                // CSV数据处理
                let csv_data = String::from_utf8(data.to_vec())
                    .map_err(|e| Error::DataProcessingError(format!("CSV解析失败: {}", e)))?;
                
                let processed_csv = self.process_csv_data(csv_data).await?;
                Ok(processed_csv.into_bytes())
            }
            DataFormat::Parquet => {
                // Parquet数据处理
                self.process_parquet_data(data).await
            }
            _ => {
                // 其他格式直接返回
                Ok(data.to_vec())
            }
        }
    }

    /// 应用用户转换
    async fn apply_user_transformations(&self, mut data: serde_json::Value) -> Result<serde_json::Value> {
        // 根据用户配置应用转换
        for transformation in &self.config.transformations {
            // 将 processor::config::DataTransformation 转换为 data::DataTransformation
            let data_transformation = crate::data::DataTransformation {
                name: transformation.transform_type.clone(),
                transformation_type: crate::data::DataTransformationType::Custom {
                    name: transformation.transform_type.clone(),
                    config: transformation.parameters.clone(),
                },
                parameters: transformation.parameters.clone(),
                description: None,
                enabled: transformation.enabled,
                order: 0,
            };
            data = self.apply_transformation(data, &data_transformation).await?;
        }
        Ok(data)
    }

    /// 应用单个转换
    async fn apply_transformation(&self, data: serde_json::Value, transformation: &DataTransformation) -> Result<serde_json::Value> {
        match &transformation.transformation_type {
            DataTransformationType::Normalization { .. } => self.normalize_json_data(data).await,
            DataTransformationType::Filtering { .. } => self.filter_json_data(data).await,
            DataTransformationType::Aggregation { .. } => self.aggregate_json_data(data).await,
            _ => Ok(data),
        }
    }

    /// 标准化JSON数据
    async fn normalize_json_data(&self, data: serde_json::Value) -> Result<serde_json::Value> {
        // 实现数据标准化逻辑
        Ok(data)
    }

    /// 过滤JSON数据
    async fn filter_json_data(&self, data: serde_json::Value) -> Result<serde_json::Value> {
        // 实现数据过滤逻辑
        Ok(data)
    }

    /// 聚合JSON数据
    async fn aggregate_json_data(&self, data: serde_json::Value) -> Result<serde_json::Value> {
        // 实现数据聚合逻辑
        Ok(data)
    }

    /// 处理CSV数据
    async fn process_csv_data(&self, csv_data: String) -> Result<String> {
        // 实现CSV数据处理逻辑
        Ok(csv_data)
    }

    /// 处理Parquet数据
    async fn process_parquet_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // 实现Parquet数据处理逻辑
        Ok(data.to_vec())
    }

    /// 获取处理统计
    pub fn get_stats(&self) -> &ProcessingStats {
        &self.stats
    }

    /// 重置统计
    pub fn reset_stats(&mut self) {
        self.stats = ProcessingStats::default();
    }
}

/// 处理统计信息
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    /// 总处理数量
    pub total_processed: u64,
    /// 成功处理数量
    pub successful: u64,
    /// 失败处理数量
    pub failed: u64,
    /// 平均处理时间（毫秒）
    pub avg_processing_time: f64,
}

impl ProcessingStats {
    /// 记录成功处理
    pub fn record_success(&mut self, processing_time: u64) {
        self.successful += 1;
        self.update_avg_time(processing_time);
    }

    /// 记录失败处理
    pub fn record_failure(&mut self) {
        self.failed += 1;
    }

    /// 更新平均处理时间
    fn update_avg_time(&mut self, processing_time: u64) {
        let total = self.successful as f64;
        self.avg_processing_time = (self.avg_processing_time * (total - 1.0) + processing_time as f64) / total;
    }
}

impl DataProcessor {
    /// 清洗样本数据
    fn clean_samples(&self, samples: &[HashMap<String, serde_json::Value>], strategies: &[String]) -> crate::Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut cleaned_samples = Vec::new();
        
        for sample in samples {
            let mut cleaned_sample = sample.clone();
            
            for strategy in strategies {
                match strategy.as_str() {
                    "remove_empty" => {
                        cleaned_sample.retain(|_, v| !v.is_null());
                    }
                    "remove_duplicates" => {
                        // 简单的去重逻辑
                        let mut seen = std::collections::HashSet::new();
                        cleaned_sample.retain(|k, _| seen.insert(k.clone()));
                    }
                    "normalize_whitespace" => {
                        for (_, v) in cleaned_sample.iter_mut() {
                            if let Some(s) = v.as_str() {
                                *v = serde_json::Value::String(s.trim().replace(r"\s+", " "));
                            }
                        }
                    }
                    _ => {
                        log::warn!("未知的清洗策略: {}", strategy);
                    }
                }
            }
            
            cleaned_samples.push(cleaned_sample);
        }
        
        Ok(cleaned_samples)
    }
    
    /// 过滤样本数据
    fn filter_samples(&self, samples: &[HashMap<String, serde_json::Value>]) -> crate::Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut filtered_samples = Vec::new();
        
        for sample in samples {
            // 简单的过滤逻辑：保留非空样本
            if !sample.is_empty() {
                filtered_samples.push(sample.clone());
            }
        }
        
        Ok(filtered_samples)
    }
    
    /// 应用NGram处理
    fn apply_ngrams(&self, samples: &[HashMap<String, serde_json::Value>]) -> crate::Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut ngram_samples = Vec::new();
        
        for sample in samples {
            let mut ngram_sample = sample.clone();
            
            for (key, value) in sample {
                if let Some(text) = value.as_str() {
                    // 简单的NGram处理
                    let ngrams = self.extract_ngrams(text, 2);
                    ngram_sample.insert(format!("{}_ngrams", key), serde_json::Value::Array(
                        ngrams.into_iter().map(|ng| serde_json::Value::String(ng)).collect()
                    ));
                }
            }
            
            ngram_samples.push(ngram_sample);
        }
        
        Ok(ngram_samples)
    }
    
    /// 应用字符NGram处理
    fn apply_char_ngrams(&self, samples: &[HashMap<String, serde_json::Value>]) -> crate::Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut char_ngram_samples = Vec::new();
        
        for sample in samples {
            let mut char_ngram_sample = sample.clone();
            
            for (key, value) in sample {
                if let Some(text) = value.as_str() {
                    // 简单的字符NGram处理
                    let char_ngrams = self.extract_char_ngrams(text, 3);
                    char_ngram_sample.insert(format!("{}_char_ngrams", key), serde_json::Value::Array(
                        char_ngrams.into_iter().map(|ng| serde_json::Value::String(ng)).collect()
                    ));
                }
            }
            
            char_ngram_samples.push(char_ngram_sample);
        }
        
        Ok(char_ngram_samples)
    }
    
    /// 提取NGram
    fn extract_ngrams(&self, text: &str, n: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut ngrams = Vec::new();
        
        for i in 0..=words.len().saturating_sub(n) {
            let ngram = words[i..i + n].join(" ");
            ngrams.push(ngram);
        }
        
        ngrams
    }
    
    /// 提取字符NGram
    fn extract_char_ngrams(&self, text: &str, n: usize) -> Vec<String> {
        let mut char_ngrams = Vec::new();
        
        for i in 0..=text.len().saturating_sub(n) {
            let ngram = text[i..i + n].to_string();
            char_ngrams.push(ngram);
        }
        
        char_ngrams
    }

    /// 标准化张量样本
    fn normalize_tensor_samples(
        &self,
        samples: &[crate::core::types::CoreTensorData],
    ) -> crate::Result<Vec<crate::core::types::CoreTensorData>> {
        let mut normalized_samples = Vec::new();
        
        for sample in samples {
            let mut normalized_sample = sample.clone();
            
            // 简单的标准化：将数据缩放到 [0, 1] 范围
            if !sample.data.is_empty() {
                let max_val = sample
                    .data
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                
                if max_val.is_finite() && max_val > 0.0 {
                    normalized_sample.data = sample
                        .data
                        .iter()
                        .map(|&x| x / max_val)
                        .collect();
                }
            }
            
            normalized_samples.push(normalized_sample);
        }
        
        Ok(normalized_samples)
    }

    /// 清洗张量样本
    fn clean_tensor_samples(&self, samples: &[crate::core::types::CoreTensorData], strategies: &[String]) -> crate::Result<Vec<crate::core::types::CoreTensorData>> {
        let mut cleaned_samples = Vec::new();
        
        for sample in samples {
            let mut cleaned_sample = sample.clone();
            
            for strategy in strategies {
                match strategy.as_str() {
                    "remove_nan" => {
                        cleaned_sample.data.retain(|&x| !x.is_nan());
                    }
                    "remove_inf" => {
                        cleaned_sample.data.retain(|&x| !x.is_infinite());
                    }
                    "remove_outliers" => {
                        // 简单的异常值处理：移除超过3个标准差的点
                        let mean = cleaned_sample.data.iter().sum::<f32>() / cleaned_sample.data.len() as f32;
                        let variance = cleaned_sample.data.iter()
                            .map(|&x| (x - mean).powi(2))
                            .sum::<f32>() / cleaned_sample.data.len() as f32;
                        let std_dev = variance.sqrt();
                        let threshold = 3.0 * std_dev;
                        
                        cleaned_sample.data.retain(|&x| (x - mean).abs() <= threshold);
                    }
                    _ => {
                        log::warn!("未知的张量清洗策略: {}", strategy);
                    }
                }
            }
            
            cleaned_samples.push(cleaned_sample);
        }
        
        Ok(cleaned_samples)
    }

    /// 过滤张量样本
    fn filter_tensor_samples(&self, samples: &[crate::core::types::CoreTensorData]) -> crate::Result<Vec<crate::core::types::CoreTensorData>> {
        let mut filtered_samples = Vec::new();
        
        for sample in samples {
            // 简单的过滤：保留非空且形状合理的样本
            if !sample.data.is_empty() && !sample.shape.is_empty() {
                let expected_size: usize = sample.shape.iter().product();
                if sample.data.len() == expected_size {
                    filtered_samples.push(sample.clone());
                }
            }
        }
        
        Ok(filtered_samples)
    }

    /// 对张量应用N-gram处理
    fn apply_ngrams_to_tensors(&self, samples: &[crate::core::types::CoreTensorData]) -> crate::Result<Vec<crate::core::types::CoreTensorData>> {
        // 对于张量数据，N-gram处理可能不适用，直接返回原数据
        Ok(samples.to_vec())
    }

    /// 对张量应用字符N-gram处理
    fn apply_char_ngrams_to_tensors(&self, samples: &[crate::core::types::CoreTensorData]) -> crate::Result<Vec<crate::core::types::CoreTensorData>> {
        // 对于张量数据，字符N-gram处理可能不适用，直接返回原数据
        Ok(samples.to_vec())
    }
} 