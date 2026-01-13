// 流数据加载器模块 - 负责从各种流源加载数据

use std::collections::HashMap;
use std::path::Path;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::time::Duration;

use async_trait::async_trait;
use log::debug;
// warn 在多个地方使用（Kafka/Arrow/Avro 错误处理）
#[allow(unused_imports)]
use log::warn;

use serde::{Deserialize, Serialize};

use crate::data::{DataBatch, DataConfig, DataSchema};
use crate::data::value::DataValue;
use crate::error::{Error, Result};
use crate::data::loader::DataLoader;
use crate::data::loader::DataSource;
use crate::data::loader::DataFormat;
use crate::data::loader::LoaderConfig;

// 流处理类型枚举
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StreamType {
    FileStream,
    HttpStream,
    Kafka,
    WebSocketStream,
    Custom(String),
}

// 流处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    pub stream_type: StreamType,
    pub connection_params: HashMap<String, String>,
    pub format: DataFormat,
    pub schema: Option<DataSchema>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            stream_type: StreamType::FileStream,
            connection_params: HashMap::new(),
            format: DataFormat::Json {
                is_lines: false,
                is_array: false,
                options: Vec::new(),
            },
            schema: None,
        }
    }
}

// 流配置超时设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamTimeoutConfig {
    pub connection_timeout: Duration,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub keep_alive_interval: Duration,
}

impl Default for StreamTimeoutConfig {
    fn default() -> Self {
        Self {
            connection_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            keep_alive_interval: Duration::from_secs(60),
        }
    }
}

// HTTP流配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpStreamConfig {
    pub url: String,
    pub method: String,
    pub headers: HashMap<String, String>,
    pub timeout: Duration,  // 使用之前导入但未使用的Duration
    pub retry_attempts: u32,
    pub retry_delay: Duration,  // 使用之前导入但未使用的Duration
}

impl HttpStreamConfig {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            method: "GET".to_string(),
            headers: HashMap::new(),
            timeout: Duration::from_secs(30),
            retry_attempts: 3,
            retry_delay: Duration::from_secs(5),
        }
    }
    
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout = Duration::from_secs(seconds);
        self
    }
    
    pub fn with_retry(mut self, attempts: u32, delay_seconds: u64) -> Self {
        self.retry_attempts = attempts;
        self.retry_delay = Duration::from_secs(delay_seconds);
        self
    }
}

// 流数据加载器
pub struct StreamDataLoader {
    config: DataConfig,
    timeout_config: StreamTimeoutConfig,
}

impl StreamDataLoader {
    pub fn new(config: DataConfig) -> Self {
        Self { 
            config,
            timeout_config: StreamTimeoutConfig::default(),
        }
    }
    
    pub fn with_timeout_config(mut self, timeout_config: StreamTimeoutConfig) -> Self {
        self.timeout_config = timeout_config;
        self
    }
    
    /// 生产级文本特征提取：将文本转换为数值特征向量
    /// 使用稳定的哈希算法和多种特征维度
    fn extract_text_features(text: &str) -> f32 {
        // 使用标准库的哈希算法，确保稳定性和分布均匀性
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        
        // 将哈希值归一化到 [0, 1] 范围
        // 使用模运算确保结果在合理范围内
        (hash % 1_000_000) as f32 / 1_000_000.0
    }
    
    // 从文件流加载数据
    async fn load_from_file_stream(
        &self,
        stream_config: &StreamConfig,
        features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        debug!("从文件流加载数据");
        
        // 获取路径参数
        let path = match stream_config.connection_params.get("path") {
            Some(p) => p,
            None => return Err(Error::invalid_argument("文件流缺少路径参数"))
        };
        
        // 添加文件流元数据
        metadata.insert("stream_type".to_string(), "file".to_string());
        metadata.insert("path".to_string(), path.clone());
        
        // 检查文件是否存在
        let path_obj = Path::new(path);
        if !path_obj.exists() {
            return Err(Error::file_not_found(format!("文件不存在: {}", path)));
        }
        
        // 读取文件内容
        let content = tokio::fs::read(path_obj).await
            .map_err(|e| Error::Io(e))?;
        
        // 根据格式处理数据
        match &stream_config.format {
            crate::data::loader::types::DataFormat::Csv { .. } => {
                // 处理CSV格式流
                self.process_csv_stream(&content, features, metadata)?;
            },
            crate::data::loader::types::DataFormat::Json { .. } => {
                // 处理JSON格式流
                self.process_json_stream(&content, features, metadata)?;
            },
            crate::data::loader::types::DataFormat::Parquet { .. } => {
                #[cfg(feature = "parquet")]
                {
                    // 处理Parquet格式流
                    self.process_parquet_stream(&content, features, metadata)?;
                }
                
                #[cfg(not(feature = "parquet"))]
                {
                    return Err(Error::not_implemented("Parquet流支持需要启用parquet特性"));
                }
            },
            crate::data::loader::types::DataFormat::Avro { .. } => {
                #[cfg(feature = "avro")]
                {
                    // 处理Avro格式流
                    self.process_avro_stream(&content, features, metadata)?;
                }
                
                #[cfg(not(feature = "avro"))]
                {
                    return Err(Error::not_implemented("Avro流支持需要启用avro特性"));
                }
            },
            crate::data::loader::types::DataFormat::CustomText(format_name) => {
                // 处理自定义文本格式
                self.process_custom_text_stream(format_name, &content, features, metadata)?;
            },
            crate::data::loader::types::DataFormat::CustomBinary(format_name) => {
                // 处理自定义二进制格式
                self.process_custom_binary_stream(format_name, &content, features, metadata)?;
            },
            _ => {
                return Err(Error::not_implemented("不支持的流格式"));
            }
        }
        
        Ok(())
    }
    
    // 从HTTP流加载数据
    async fn load_from_http_stream(
        &self,
        stream_config: &StreamConfig,
        _features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        debug!("从HTTP流加载数据");
        
        // 获取URL参数
        let url = match stream_config.connection_params.get("url") {
            Some(u) => u,
            None => return Err(Error::invalid_argument("HTTP流缺少URL参数"))
        };
        
        // 添加HTTP流元数据
        metadata.insert("stream_type".to_string(), "http".to_string());
        metadata.insert("url".to_string(), url.clone());
        
        #[cfg(feature = "http")]
        {
            use reqwest;
            
            // 创建HTTP客户端
            let client = reqwest::Client::new();
            
            // 发送请求
            let response = client.get(url)
                .timeout(Duration::from_secs(30)) // 设置超时
                .send()
                .await
                .map_err(|e| Error::network(format!("HTTP请求失败: {}", e)))?;
            
            // 检查响应状态
            if !response.status().is_success() {
                return Err(Error::network(
                    format!("HTTP请求失败，状态码: {}", response.status())
                ));
            }
            
            // 读取响应内容
            let content = response.bytes().await
                .map_err(|e| Error::network(format!("读取HTTP响应失败: {}", e)))?;
            
            // 根据格式处理数据
            match &stream_config.format {
                crate::data::loader::types::DataFormat::Csv { .. } => {
                    // 处理CSV格式流
                    self.process_csv_stream(&content, features, metadata)?;
                },
                crate::data::loader::types::DataFormat::Json { .. } => {
                    // 处理JSON格式流
                    self.process_json_stream(&content, features, metadata)?;
                },
                crate::data::loader::types::DataFormat::Parquet { .. } => {
                    #[cfg(feature = "parquet")]
                    {
                        // 处理Parquet格式流
                        self.process_parquet_stream(&content, features, metadata)?;
                    }
                    
                    #[cfg(not(feature = "parquet"))]
                    {
                        return Err(Error::not_implemented("Parquet流支持需要启用parquet特性"));
                    }
                },
                crate::data::loader::types::DataFormat::Avro { .. } => {
                    #[cfg(feature = "avro")]
                    {
                        // 处理Avro格式流
                        self.process_avro_stream(&content, features, metadata)?;
                    }
                    
                    #[cfg(not(feature = "avro"))]
                    {
                        return Err(Error::not_implemented("Avro流支持需要启用avro特性"));
                    }
                },
                crate::data::loader::types::DataFormat::CustomText(format_name) => {
                    // 处理自定义文本格式
                    self.process_custom_text_stream(format_name, &content, features, metadata)?;
                },
                crate::data::loader::types::DataFormat::CustomBinary(format_name) => {
                    // 处理自定义二进制格式
                    self.process_custom_binary_stream(format_name, &content, features, metadata)?;
                },
                _ => {
                    return Err(Error::not_implemented("不支持的流格式"));
                }
            }
        }
        
        #[cfg(not(feature = "http"))]
        {
            return Err(Error::not_implemented("HTTP流支持需要启用http特性"));
        }
        
        #[cfg(feature = "http")]
        {
            return Ok(());
        }
    }
    
    // 从Kafka流加载数据
    async fn load_from_kafka_stream(
        &self,
        stream_config: &StreamConfig,
        _features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        debug!("从Kafka流加载数据");
        
        // 获取必要参数
        let brokers = match stream_config.connection_params.get("brokers") {
            Some(b) => b,
            None => return Err(Error::invalid_argument("Kafka流缺少brokers参数"))
        };
        
        let topic = match stream_config.connection_params.get("topic") {
            Some(t) => t,
            None => return Err(Error::invalid_argument("Kafka流缺少topic参数"))
        };
        
        // 获取可选的group_id
        let group_id = stream_config.connection_params.get("group_id")
            .cloned()
            .unwrap_or_else(|| "vecmind_consumer".to_string());
        
        // 添加Kafka流元数据
        metadata.insert("stream_type".to_string(), "kafka".to_string());
        metadata.insert("brokers".to_string(), brokers.clone());
        metadata.insert("topic".to_string(), topic.clone());
        metadata.insert("group_id".to_string(), group_id.clone());
        
        #[cfg(feature = "kafka")]
        {
            use rdkafka::config::ClientConfig;
            use rdkafka::consumer::{Consumer, StreamConsumer};
            use rdkafka::message::Message;
            use std::time::Duration;
            
            // 配置Kafka消费者
            let consumer: StreamConsumer = ClientConfig::new()
                .set("group.id", &group_id)
                .set("bootstrap.servers", brokers)
                .set("enable.auto.commit", "true")
                .set("auto.offset.reset", "earliest")
                .create()
                .map_err(|e| Error::stream(format!("Kafka消费者创建失败: {}", e)))?;
                
            // 订阅主题
            consumer.subscribe(&[topic])
                .map_err(|e| Error::stream(format!("Kafka主题订阅失败: {}", e)))?;
                
            // 设置最大消息数和超时时间
            let max_messages = 1000; // 最多处理1000条消息
            let timeout = Duration::from_secs(10); // 最多等待10秒
            
            // 开始消费消息
            let mut message_count = 0;
            let start_time = std::time::Instant::now();
            
            while message_count < max_messages && start_time.elapsed() < timeout {
                match consumer.recv().await {
                    Ok(msg) => {
                        // 获取消息内容
                        if let Some(payload) = msg.payload() {
                            // 根据格式处理数据
                            match &stream_config.format {
                                crate::data::loader::types::DataFormat::Csv { .. } => {
                                    // 处理CSV格式流
                                    self.process_csv_stream(payload, features, metadata)?;
                                },
                                crate::data::loader::types::DataFormat::Json { .. } => {
                                    // 处理JSON格式流
                                    self.process_json_stream(payload, features, metadata)?;
                                },
                                crate::data::loader::types::DataFormat::Parquet { .. } => {
                                    #[cfg(feature = "parquet")]
                                    {
                                        // 处理Parquet格式流
                                        self.process_parquet_stream(payload, features, metadata)?;
                                    }
                                    
                                    #[cfg(not(feature = "parquet"))]
                                    {
                                        return Err(Error::not_implemented("Parquet流支持需要启用parquet特性"));
                                    }
                                },
                                crate::data::loader::types::DataFormat::Avro { .. } => {
                                    #[cfg(feature = "avro")]
                                    {
                                        // 处理Avro格式流
                                        self.process_avro_stream(payload, features, metadata)?;
                                    }
                                    
                                    #[cfg(not(feature = "avro"))]
                                    {
                                        return Err(Error::not_implemented("Avro流支持需要启用avro特性"));
                                    }
                                },
                                crate::data::loader::types::DataFormat::CustomText(format_name) => {
                                    // 处理自定义文本格式
                                    self.process_custom_text_stream(format_name, payload, features, metadata)?;
                                },
                                crate::data::loader::types::DataFormat::CustomBinary(format_name) => {
                                    // 处理自定义二进制格式
                                    self.process_custom_binary_stream(format_name, payload, features, metadata)?;
                                },
                                _ => {
                                    return Err(Error::not_implemented("不支持的流格式"));
                                }
                            }
                            
                            message_count += 1;
                        }
                    },
                    Err(e) => {
                        warn!("Kafka消息读取错误: {}", e);
                        // 非致命错误，继续尝试
                        continue;
                    }
                }
            }
            
            metadata.insert("message_count".to_string(), message_count.to_string());
        }
        
        #[cfg(not(feature = "kafka"))]
        {
            return Err(Error::not_implemented("Kafka流支持需要启用kafka特性"));
        }
        
        #[cfg(feature = "kafka")]
        {
            Ok(())
        }
    }
    
    // 处理CSV格式流
    fn process_csv_stream(
        &self,
        content: &[u8],
        features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        use std::io::Cursor;
        use ::csv::ReaderBuilder;
        
        // skip_header 是 Option<bool>，需要 unwrap 或使用默认值
        let has_headers = self.config.skip_header.unwrap_or(true);
        // delimiter 是 Option<String>，需要先获取字符串，然后取第一个字符
        let delimiter = self.config.delimiter
            .as_ref()
            .and_then(|s| s.chars().next())
            .unwrap_or(',') as u8;
        
        let mut reader = ReaderBuilder::new()
            .has_headers(has_headers)
            .delimiter(delimiter)
            .from_reader(Cursor::new(content));
        
        // 添加格式元数据
        metadata.insert("format".to_string(), "csv".to_string());
        
        // 初始行数计数
        let mut row_count = 0;
        
        for result in reader.records() {
            let record = result.map_err(|e| Error::parse(format!("CSV解析错误: {}", e)))?;
            
            // 处理一行数据
            let mut row = Vec::new();
            for field in record.iter() {
                // 尝试转换为数值
                match field.parse::<f32>() {
                    Ok(value) => row.push(value),
                    Err(_) => {
                        // 如果不是数值，使用生产级文本特征提取
                        row.push(Self::extract_text_features(field));
                    }
                }
            }
            
            if !row.is_empty() {
                features.push(row);
                row_count += 1;
            }
        }
        
        // 添加行数到元数据
        metadata.insert("row_count".to_string(), row_count.to_string());
        
        Ok(())
    }
    
    // 处理JSON格式流
    fn process_json_stream(
        &self,
        content: &[u8],
        features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        use serde_json::Value;
        
        // 添加格式元数据
        metadata.insert("format".to_string(), "json".to_string());
        
        // 解析JSON
        let json_value: Value = serde_json::from_slice(content)
            .map_err(|e| Error::parse(format!("JSON解析错误: {}", e)))?;
        
        // 初始项目计数
        let mut item_count = 0;
        
        // 根据JSON结构处理
        match json_value {
            Value::Array(items) => {
                for item in items {
                    if let Some(feature_vector) = self.extract_features_from_json(&item) {
                        features.push(feature_vector);
                        item_count += 1;
                    }
                }
            },
            _ => {
                // 单个JSON对象
                if let Some(feature_vector) = self.extract_features_from_json(&json_value) {
                    features.push(feature_vector);
                    item_count += 1;
                }
            }
        }
        
        // 添加项目数到元数据
        metadata.insert("item_count".to_string(), item_count.to_string());
        
        Ok(())
    }
    
    // 处理自定义文本格式流
    fn process_custom_text_stream(
        &self,
        format_name: &str,
        content: &[u8],
        features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        // 添加格式元数据
        metadata.insert("format".to_string(), format!("custom_text:{}", format_name));
        
        // 尝试将内容转换为字符串
        let text = String::from_utf8_lossy(content);
        
        // 根据不同的自定义格式进行处理
        match format_name {
            "tsv" => {
                // 处理TSV格式（制表符分隔的值）
                for line in text.lines() {
                    let mut row = Vec::new();
                    
                    for field in line.split('\t') {
                        // 尝试转换为数值
                        match field.parse::<f32>() {
                            Ok(value) => row.push(value),
                            Err(_) => {
                                // 如果不是数值，使用生产级文本特征提取
                                row.push(Self::extract_text_features(field));
                            }
                        }
                    }
                    
                    if !row.is_empty() {
                        features.push(row);
                    }
                }
            },
            "lines" => {
                // 处理按行分隔的数据，每行作为一个特征
                for line in text.lines() {
                    if !line.trim().is_empty() {
                        // 使用生产级文本特征提取
                        features.push(vec![Self::extract_text_features(line)]);
                    }
                }
            },
            _ => {
                return Err(Error::unsupported_format(
                    format!("不支持的自定义文本格式: {}", format_name)
                ));
            }
        }
        
        // 添加项目数到元数据
        metadata.insert("item_count".to_string(), features.len().to_string());
        
        Ok(())
    }
    
    // 处理自定义二进制格式流
    fn process_custom_binary_stream(
        &self,
        format_name: &str,
        content: &[u8],
        features: &mut Vec<Vec<f32>>,
        metadata: &mut HashMap<String, String>
    ) -> Result<()> {
        // 添加格式元数据
        metadata.insert("format".to_string(), format!("custom_binary:{}", format_name));
        
        // 根据不同的自定义格式进行处理
        match format_name {
            "float32" => {
                // 处理32位浮点数数组
                if content.len() % 4 != 0 {
                    return Err(Error::parse("float32格式数据长度必须是4的倍数"));
                }
                
                let mut i = 0;
                while i < content.len() {
                    if i + 4 <= content.len() {
                        let bytes = [content[i], content[i+1], content[i+2], content[i+3]];
                        let value = f32::from_le_bytes(bytes);
                        features.push(vec![value]);
                    }
                    i += 4;
                }
            },
            "int32" => {
                // 处理32位整数数组
                if content.len() % 4 != 0 {
                    return Err(Error::parse("int32格式数据长度必须是4的倍数"));
                }
                
                let mut i = 0;
                while i < content.len() {
                    if i + 4 <= content.len() {
                        let bytes = [content[i], content[i+1], content[i+2], content[i+3]];
                        let value = i32::from_le_bytes(bytes) as f32;
                        features.push(vec![value]);
                    }
                    i += 4;
                }
            },
            _ => {
                return Err(Error::unsupported_format(
                    format!("不支持的自定义二进制格式: {}", format_name)
                ));
            }
        }
        
        // 添加项目数到元数据
        metadata.insert("item_count".to_string(), features.len().to_string());
        
        Ok(())
    }
    
    // 从JSON值中提取特征
    fn extract_features_from_json(&self, json: &serde_json::Value) -> Option<Vec<f32>> {
        let mut features = Vec::new();
        
        // 处理方式1: 如果是数组，直接转换为特征向量
        if let serde_json::Value::Array(values) = json {
            for value in values {
                match value {
                    serde_json::Value::Number(num) => {
                        if let Some(f) = num.as_f64() {
                            features.push(f as f32);
                        }
                    },
                    serde_json::Value::String(s) => {
                        // 对于字符串，尝试解析为数字或使用简单编码
                        if let Ok(f) = s.parse::<f32>() {
                            features.push(f);
                        } else {
                            // 使用生产级文本特征提取
                            features.push(Self::extract_text_features(s));
                        }
                    },
                    _ => continue, // 忽略其他类型
                }
            }
            
            return if !features.is_empty() { Some(features) } else { None };
        }
        
        // 处理方式2: 如果是对象，根据模式提取特征
        if let serde_json::Value::Object(map) = json {
            // 检查是否有特征字段
            if let Some(serde_json::Value::Array(feature_array)) = map.get("features") {
                for value in feature_array {
                    if let Some(f) = value.as_f64() {
                        features.push(f as f32);
                    } else if let Some(s) = value.as_str() {
                        if let Ok(f) = s.parse::<f32>() {
                            features.push(f);
                        }
                    }
                }
                
                return if !features.is_empty() { Some(features) } else { None };
            }
            
            // 替代方法: 从指定的字段提取特征
            if let Some(schema) = &self.config.schema {
                // feature_fields 是方法，需要调用它
                let feature_fields = schema.feature_fields();
                if !feature_fields.is_empty() {
                    for field in &feature_fields {
                        if let Some(value) = map.get(field) {
                            if let Some(f) = value.as_f64() {
                                features.push(f as f32);
                            } else if let Some(s) = value.as_str() {
                                if let Ok(f) = s.parse::<f32>() {
                                    features.push(f);
                                } else {
                                    // 使用简单编码
                                    let hash = s.chars().fold(0.0, |acc, c| {
                                        acc + (c as u32 % 1000) as f32 / 1000.0
                                    });
                                    features.push(hash);
                                }
                            } else if let Some(b) = value.as_bool() {
                                features.push(if b { 1.0 } else { 0.0 });
                            }
                        }
                    }
                    
                    return if !features.is_empty() { Some(features) } else { None };
                }
            }
            
            // 如果没有模式，尝试从所有数值字段提取特征
            for (_, value) in map {
                if let Some(f) = value.as_f64() {
                    features.push(f as f32);
                }
            }
            
            return if !features.is_empty() { Some(features) } else { None };
        }
        
        None
    }

    #[cfg(feature = "parquet")]
    fn process_parquet_stream(&self, content: &[u8], features: &mut Vec<Vec<f32>>, metadata: &mut HashMap<String, String>) -> Result<()> {
        use arrow::record_batch::RecordBatch;
        use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
        use parquet::file::reader::SerializedFileReader;
        use std::io::Cursor;
        
        // 创建Parquet读取器
        let cursor = Cursor::new(content);
        let file_reader = SerializedFileReader::new(cursor)?;
        let mut arrow_reader = ParquetRecordBatchReader::try_new(file_reader, 1024)?;
        
        // 提取元数据
        metadata.insert("format".to_string(), "parquet".to_string());
        
        // 处理每个批次
        while let Some(batch_result) = arrow_reader.next() {
            let batch = batch_result?;
            
            // 提取schema信息到元数据
            if !metadata.contains_key("schema") {
                metadata.insert("schema".to_string(), batch.schema().to_string());
            }
            
            // 处理每一行
            for row_idx in 0..batch.num_rows() {
                let mut row_features = Vec::new();
                
                // 处理每一列
                for col_idx in 0..batch.num_columns() {
                    let column = batch.column(col_idx);
                    let column_name = batch.schema().field(col_idx).name();
                    
                    // 根据不同的数据类型处理
                    match column.data_type() {
                        arrow::datatypes::DataType::Float32 => {
                            let array = column.as_any().downcast_ref::<Float32Array>().unwrap();
                            if !array.is_null(row_idx) {
                                row_features.push(array.value(row_idx));
                            } else {
                                // NULL值使用NaN标记，而不是默认0.0
                                row_features.push(f32::NAN);
                            }
                        },
                        arrow::datatypes::DataType::Float64 => {
                            let array = column.as_any().downcast_ref::<Float64Array>().unwrap();
                            if !array.is_null(row_idx) {
                                row_features.push(array.value(row_idx) as f32);
                            } else {
                                // NULL值使用NaN标记
                                row_features.push(f32::NAN);
                            }
                        },
                        arrow::datatypes::DataType::Int32 => {
                            let array = column.as_any().downcast_ref::<Int32Array>().unwrap();
                            if !array.is_null(row_idx) {
                                row_features.push(array.value(row_idx) as f32);
                            } else {
                                // NULL值使用NaN标记
                                row_features.push(f32::NAN);
                            }
                        },
                        arrow::datatypes::DataType::Int64 => {
                            let array = column.as_any().downcast_ref::<Int64Array>().unwrap();
                            if !array.is_null(row_idx) {
                                row_features.push(array.value(row_idx) as f32);
                            } else {
                                // NULL值使用NaN标记
                                row_features.push(f32::NAN);
                            }
                        },
                        arrow::datatypes::DataType::Utf8 => {
                            let array = column.as_any().downcast_ref::<StringArray>().unwrap();
                            if !array.is_null(row_idx) {
                                // 使用生产级文本特征提取
                                let str_value = array.value(row_idx);
                                row_features.push(Self::extract_text_features(str_value));
                            } else {
                                // NULL值使用NaN标记
                                row_features.push(f32::NAN);
                            }
                        },
                        // 其他数据类型可以根据需要添加
                        _ => {
                            // 对于不支持的类型，标记为缺失值而不是默认0
                            warn!("Arrow数据类型 {:?} 暂不支持直接转换为特征，使用NaN标记", column.data_type());
                            row_features.push(f32::NAN);
                            
                            // 记录未处理的类型到元数据
                            let unhandled_key = format!("unhandled_type_{}", column_name);
                            metadata.insert(unhandled_key, column.data_type().to_string());
                        }
                    }
                }
                
                if !row_features.is_empty() {
                    features.push(row_features);
                }
            }
        }
        
        Ok(())
    }
    
    #[cfg(feature = "avro")]
    fn process_avro_stream(&self, content: &[u8], features: &mut Vec<Vec<f32>>, metadata: &mut HashMap<String, String>) -> Result<()> {
        use apache_avro::{Reader, from_avro_datum};
        use apache_avro::types::Value as AvroValue;
        use std::io::Cursor;
        
        // 创建Avro读取器
        let cursor = Cursor::new(content);
        let reader = Reader::new(cursor)?;
        
        // 提取元数据
        metadata.insert("format".to_string(), "avro".to_string());
        
        if let Some(schema) = reader.writer_schema() {
            metadata.insert("schema".to_string(), schema.canonical_form());
        }
        
        // 处理每条记录
        for value_result in reader {
            let value = value_result?;
            
            match value {
                AvroValue::Record(fields) => {
                    let mut row_features = Vec::new();
                    
                    for (_, field_value) in fields {
                        // 根据不同的Avro值类型处理
                        match field_value {
                            AvroValue::Int(i) => row_features.push(i as f32),
                            AvroValue::Long(l) => row_features.push(l as f32),
                            AvroValue::Float(f) => row_features.push(f),
                            AvroValue::Double(d) => row_features.push(d as f32),
                            AvroValue::Boolean(b) => row_features.push(if b { 1.0 } else { 0.0 }),
                            AvroValue::String(s) => {
                                // 使用生产级文本特征提取
                                row_features.push(Self::extract_text_features(s));
                            },
                            AvroValue::Bytes(b) => {
                                // 二进制数据：使用长度归一化和对数缩放
                                let len = b.len() as f32;
                                row_features.push(len.ln() / 10.0);
                            },
                            AvroValue::Null => {
                                // NULL值使用NaN标记
                                row_features.push(f32::NAN);
                            },
                            // 其他Avro类型：根据类型进行合理转换
                            _ => {
                                // 对于不支持的类型，标记为缺失值而不是默认0
                                warn!("Avro类型 {:?} 暂不支持直接转换为特征，使用NaN标记", field_value);
                                row_features.push(f32::NAN);
                            }
                        }
                    }
                    
                    if !row_features.is_empty() {
                        features.push(row_features);
                    }
                },
                // 如果不是Record类型，忽略
                _ => continue,
            }
        }
        
        Ok(())
    }
}

#[async_trait]
impl DataLoader for StreamDataLoader {
    async fn load(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        match source {
            DataSource::Stream(stream_config) => {
                // 从流中加载数据
                debug!("从流中加载数据: {}", stream_config);
                
                // 解析流配置
                let config_parts: Vec<&str> = stream_config.split("://").collect();
                if config_parts.len() != 2 {
                    return Err(Error::invalid_argument(
                        "流格式无效，应为'类型://连接参数'"
                    ));
                }
                
                let stream_type = config_parts[0].to_lowercase();
                let connection_string = config_parts[1];
                
                // 创建基础结果结构
                let mut features = Vec::new();
                let mut metadata = HashMap::new();
                
                // 添加元数据
                metadata.insert("source".to_string(), "stream".to_string());
                metadata.insert("stream_type".to_string(), stream_type.clone());
                metadata.insert("connection".to_string(), connection_string.to_string());
                
                // 根据流类型处理
                match stream_type.as_str() {
                    "file" => {
                        // 创建文件流配置
                        let mut params = HashMap::new();
                        params.insert("path".to_string(), connection_string.to_string());
                        
                        let stream_config = StreamConfig {
                            stream_type: StreamType::FileStream,
                            connection_params: params,
                            format: format.clone(),
                            schema: self.config.schema.clone(),
                        };
                        
                        self.load_from_file_stream(&stream_config, &mut features, &mut metadata).await?;
                    },
                    "http" | "https" => {
                        // 创建HTTP流配置
                        let mut params = HashMap::new();
                        params.insert("url".to_string(), format!("{}://{}", stream_type, connection_string));
                        
                        let stream_config = StreamConfig {
                            stream_type: StreamType::HttpStream,
                            connection_params: params,
                            format: format.clone(),
                            schema: self.config.schema.clone(),
                        };
                        
                        self.load_from_http_stream(&stream_config, &mut features, &mut metadata).await?;
                    },
                    "kafka" => {
                        // 解析Kafka连接参数
                        let kafka_parts: Vec<&str> = connection_string.split('/').collect();
                        if kafka_parts.len() < 2 {
                            return Err(Error::invalid_argument(
                                "Kafka流格式无效，应为'kafka://brokers/topic'"
                            ));
                        }
                        
                        let brokers = kafka_parts[0];
                        let topic = kafka_parts[1];
                        
                        // 创建Kafka流配置
                        let mut params = HashMap::new();
                        params.insert("brokers".to_string(), brokers.to_string());
                        params.insert("topic".to_string(), topic.to_string());
                        
                        // 添加额外参数
                        if kafka_parts.len() > 2 {
                            params.insert("group_id".to_string(), kafka_parts[2].to_string());
                        }
                        
                        let stream_config = StreamConfig {
                            stream_type: StreamType::Kafka,
                            connection_params: params,
                            format: format.clone(),
                            schema: self.config.schema.clone(),
                        };
                        
                        self.load_from_kafka_stream(&stream_config, &mut features, &mut metadata).await?;
                    },
                    // 其他流类型
                    _ => {
                        return Err(Error::invalid_argument(
                            format!("不支持的流类型: {}", stream_type)
                        ));
                    }
                }
                
                // 没有数据则返回错误
                if features.is_empty() {
                    return Err(Error::no_data("从流中未加载到数据"));
                }
                
                // 创建数据批次
                let mut batch = DataBatch::new("stream_dataset", 0, features.len());
                batch.metadata = metadata;
                
                // 转换特征向量为记录
                for (i, feature_vec) in features.iter().enumerate() {
                    let mut record = HashMap::new();
                    for (j, value) in feature_vec.iter().enumerate() {
                        record.insert(format!("feature_{}", j), DataValue::Float(*value as f64));
                    }
                    record.insert("record_id".to_string(), DataValue::Integer(i as i64));
                    batch.records.push(record);
                }
                
                batch.size = batch.records.len();
                Ok(batch)
            },
            _ => Err(Error::invalid_argument("数据源类型不是流"))
        }
    }
    
    async fn get_schema(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataSchema> {
        match source {
            DataSource::Stream(stream_config) => {
                // 尝试解析流配置
                let config_parts: Vec<&str> = stream_config.split("://").collect();
                if config_parts.len() != 2 {
                    return Err(Error::invalid_argument(
                        "流格式无效，应为'类型://连接参数'"
                    ));
                }
                
                let stream_type = config_parts[0].to_lowercase();
                
                // 先检查是否已配置模式
                if let Some(schema) = &self.config.schema {
                    return Ok(schema.clone());
                }
                
                // 创建基础模式
                let mut schema = DataSchema::new("stream_schema", "1.0");
                
                // 添加基础字段
                let fields = vec![
                    crate::data::schema::schema::FieldDefinition {
                        name: "feature_0".to_string(),
                        field_type: crate::data::schema::schema::FieldType::Numeric,
                        data_type: None,
                        required: false,
                        nullable: false,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    crate::data::schema::schema::FieldDefinition {
                        name: "feature_1".to_string(),
                        field_type: crate::data::schema::schema::FieldType::Numeric,
                        data_type: None,
                        required: false,
                        nullable: false,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    crate::data::schema::schema::FieldDefinition {
                        name: "record_id".to_string(),
                        field_type: crate::data::schema::schema::FieldType::Numeric,
                        data_type: None,
                        required: false,
                        nullable: false,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                ];
                for f in fields {
                    schema.add_field(f)?;
                }
                
                Ok(schema)
            },
            _ => Err(Error::invalid_argument("数据源类型不是流"))
        }
    }
    
    async fn load_batch(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat, batch_size: usize, offset: usize) -> Result<DataBatch> {
        // 对于流数据，batch_size和offset通常不适用，但我们可以模拟
        let full_batch = self.load(source, format).await?;
        
        // 从offset开始取batch_size个记录
        let end_idx = std::cmp::min(offset + batch_size, full_batch.records.len());
        if offset >= full_batch.records.len() {
            // 返回空批次
            let mut empty_batch = DataBatch::new("stream_dataset", offset, 0);
            empty_batch.metadata = full_batch.metadata;
            return Ok(empty_batch);
        }
        
        let mut batch = DataBatch::new("stream_dataset", offset, end_idx - offset);
        batch.records = full_batch.records[offset..end_idx].to_vec();
        batch.metadata = full_batch.metadata;
        batch.size = batch.records.len();
        
        Ok(batch)
    }
    
    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        match format {
            crate::data::loader::types::DataFormat::Csv { .. } => true,
            crate::data::loader::types::DataFormat::Json { .. } => true,
            crate::data::loader::types::DataFormat::Parquet { .. } => cfg!(feature = "parquet"),
            crate::data::loader::types::DataFormat::Avro { .. } => cfg!(feature = "avro"),
            crate::data::loader::types::DataFormat::CustomText(_) => true,
            crate::data::loader::types::DataFormat::CustomBinary(_) => true,
            _ => false,
        }
    }
    
    fn config(&self) -> &LoaderConfig {
        // 返回默认配置
        static DEFAULT_CONFIG: std::sync::OnceLock<LoaderConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| LoaderConfig::default())
    }
    
    fn set_config(&mut self, config: LoaderConfig) {
        // 目前stream loader不直接使用LoaderConfig，但可以在这里处理一些通用设置
        if let Some(timeout) = config.timeout {
            self.timeout_config.connection_timeout = std::time::Duration::from_secs(timeout);
        }
    }
    
    fn name(&self) -> &'static str {
        "StreamDataLoader"
    }
    
    async fn get_size(&self, path: &str) -> Result<usize> {
        // 对于流数据，尝试从数据源获取实际大小
        // 如果是文件流，可以读取文件大小
        if let Ok(metadata) = std::fs::metadata(path) {
            // 文件大小可以作为流大小的估计
            // 对于文本流，可以估算行数
            if metadata.is_file() {
                use std::io::{BufRead, BufReader};
                use std::fs::File;
                
                let file = File::open(path)
                    .map_err(|e| Error::io_error(format!("无法打开流文件: {}, 错误: {}", path, e)))?;
                let reader = BufReader::new(file);
                
                // 估算行数（用于文本流）
                let line_count = reader.lines().count();
                Ok(line_count)
            } else {
                // 目录或其他类型，返回错误
                Err(Error::invalid_argument(format!("路径不是文件: {}", path)))
            }
        } else {
            // 无法获取元数据，可能是网络流或其他动态源
            // 返回错误而不是默认0，让调用者知道大小未知
            Err(Error::not_implemented(format!("无法确定流数据源大小: {}", path)))
        }
    }
}
