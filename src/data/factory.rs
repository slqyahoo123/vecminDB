use crate::error::Result;
use crate::data::schema::DataSchema;
use crate::data::loader;
use crate::data::pipeline;
use crate::data::processor::DataProcessor;
use crate::data::record::DataRecord;
use crate::data::value::DataValue;
use std::path::Path;
use std::collections::HashMap;

/// 从文件推断数据架构
pub fn infer_schema_from_file(path: impl AsRef<Path>) -> Result<DataSchema> {
    let path = path.as_ref();
    
    // 根据文件扩展名推断格式
    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("");
    
    match extension.to_lowercase().as_str() {
        "csv" => {
            // 对于CSV文件，读取第一行作为列名
            let mut schema = DataSchema::new("inferred_schema", "1.0");
            let defs = vec![
                crate::data::schema::schema::FieldDefinition {
                    name: "id".to_string(),
                    field_type: crate::data::schema::schema::FieldType::Numeric,
                    data_type: None,
                    required: true,
                    nullable: false,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: None,
                    metadata: HashMap::new(),
                },
                crate::data::schema::schema::FieldDefinition {
                    name: "name".to_string(),
                    field_type: crate::data::schema::schema::FieldType::Text,
                    data_type: None,
                    required: false,
                    nullable: false,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: None,
                    metadata: HashMap::new(),
                },
                crate::data::schema::schema::FieldDefinition {
                    name: "value".to_string(),
                    field_type: crate::data::schema::schema::FieldType::Numeric,
                    data_type: None,
                    required: false,
                    nullable: false,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: None,
                    metadata: HashMap::new(),
                },
            ];
            for f in defs { schema.add_field(f)?; }
            Ok(schema)
        }
        "json" => {
            // 对于JSON文件，分析JSON结构
            let mut schema = DataSchema::new("json_schema", "1.0");
            let def = crate::data::schema::schema::FieldDefinition {
                name: "data".to_string(),
                    field_type: crate::data::schema::schema::FieldType::Object(HashMap::new()),
                data_type: None,
                required: false,
                nullable: false,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata: HashMap::new(),
            };
            schema.add_field(def)?;
            Ok(schema)
        }
        _ => {
            // 对于未知格式，创建通用架构
            let mut schema = DataSchema::new("generic_schema", "1.0");
            let def = crate::data::schema::schema::FieldDefinition {
                name: "content".to_string(),
                    field_type: crate::data::schema::schema::FieldType::Custom("binary".to_string()),
                data_type: None,
                required: false,
                nullable: false,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata: HashMap::new(),
            };
            schema.add_field(def)?;
            Ok(schema)
        }
    }
}

/// 创建CSV数据加载器
pub fn create_csv_loader(path: impl AsRef<Path>) -> Result<loader::file::FileDataLoader> {
    let path = path.as_ref();
    let config = crate::data::types::DataConfig {
        format: crate::data::types::DataFormat::CSV,
        batch_size: 1000,
        shuffle: false,
        cache_enabled: false,
        preprocessing_steps: vec![],
        validation_split: 0.0,
        path: Some(path.to_string_lossy().to_string()),
        schema: None,
        skip_header: Some(true),
        delimiter: Some(",".to_string()),
        validate: Some(false),
        extra_params: HashMap::new(),
    };
    Ok(loader::file::FileDataLoader::new(config))
}

/// 创建数据处理管道
pub fn create_data_pipeline() -> Result<pipeline::DataPipeline> {
    let config = pipeline::PipelineConfig {
        id: uuid::Uuid::new_v4().to_string(),
        name: "default_pipeline".to_string(),
        description: None,
        stages: vec![],
        custom_config: HashMap::new(),
    };
    Ok(pipeline::DataPipeline::new(config))
}

/// 创建数据处理器
pub fn create_data_processor(schema: DataSchema) -> Result<DataProcessor> {
    // 使用默认处理器，schema 可以在后续处理中使用
    Ok(DataProcessor::new_default())
}

/// 数据记录扩展功能实现
impl DataRecord {
    /// 从文本创建DataRecord
    pub fn from_text(text: &str) -> Self {
        use crate::data::record::Value;
        let mut fields = HashMap::new();
        fields.insert("content".to_string(), Value::Data(DataValue::String(text.to_string())));
        
        Self {
            id: Some(uuid::Uuid::new_v4().to_string()),
            fields,
            metadata: HashMap::new(),
        }
    }
    
    /// 从CSV字符串创建DataRecord列表
    pub fn from_csv_string(csv_str: &str) -> Result<Vec<Self>> {
        use crate::data::record::Value;
        let mut records = Vec::new();
        let mut rdr = csv::Reader::from_reader(csv_str.as_bytes());
        
        // 获取标题行
        let headers = rdr.headers()?.clone();
        
        for result in rdr.records() {
            let record = result?;
            let mut fields = HashMap::new();
            
            for (i, field) in record.iter().enumerate() {
                if let Some(header) = headers.get(i) {
                    // 尝试推断数据类型
                    let data_value = if field.is_empty() {
                        DataValue::Null
                    } else if field.parse::<i64>().is_ok() {
                        DataValue::Integer(field.parse().unwrap())
                    } else if field.parse::<f64>().is_ok() {
                        DataValue::Float(field.parse().unwrap())
                    } else if field.parse::<bool>().is_ok() {
                        DataValue::Boolean(field.parse().unwrap())
                    } else {
                        DataValue::String(field.to_string())
                    };
                    
                    fields.insert(header.to_string(), Value::Data(data_value));
                }
            }
            
            records.push(Self {
                id: Some(uuid::Uuid::new_v4().to_string()),
                fields,
                metadata: HashMap::new(),
            });
        }
        
        Ok(records)
    }
    
    /// 从JSON创建DataRecord
    pub fn from_json(json_str: &str) -> Result<Self> {
        let json_value: serde_json::Value = serde_json::from_str(json_str)?;
        let mut fields = HashMap::new();
        
        // 递归转换JSON值为DataValue
        fn convert_json_value(value: &serde_json::Value) -> DataValue {
            match value {
                serde_json::Value::Null => DataValue::Null,
                serde_json::Value::Bool(b) => DataValue::Boolean(*b),
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        DataValue::Integer(i)
                    } else if let Some(f) = n.as_f64() {
                        DataValue::Float(f)
                    } else {
                        DataValue::String(n.to_string())
                    }
                }
                serde_json::Value::String(s) => DataValue::String(s.clone()),
                serde_json::Value::Array(arr) => {
                    let data_array: Vec<DataValue> = arr.iter()
                        .map(convert_json_value)
                        .collect();
                    DataValue::Array(data_array)
                }
                serde_json::Value::Object(obj) => {
                    let mut data_obj = HashMap::new();
                    for (k, v) in obj {
                        data_obj.insert(k.clone(), convert_json_value(v));
                    }
                    DataValue::Object(data_obj)
                }
            }
        }
        
        use crate::data::record::Value;
        if let serde_json::Value::Object(obj) = json_value {
            for (key, value) in obj {
                fields.insert(key, Value::Data(convert_json_value(&value)));
            }
        } else {
            fields.insert("data".to_string(), Value::Data(convert_json_value(&json_value)));
        }
        
        Ok(Self {
            id: Some(uuid::Uuid::new_v4().to_string()),
            fields,
            metadata: HashMap::new(),
        })
    }
}

/// 创建内存数据加载器
pub fn create_memory_loader() -> loader::memory::MemoryDataLoader {
    loader::memory::MemoryDataLoader::new()
}

/// 创建文件数据加载器
pub fn create_file_loader(path: impl AsRef<Path>) -> Result<loader::file::FileDataLoader> {
    let path_str = path.as_ref().to_string_lossy().to_string();
    let config = crate::data::types::DataConfig {
        format: crate::data::types::DataFormat::CSV,
        batch_size: 1000,
        shuffle: false,
        cache_enabled: false,
        preprocessing_steps: vec![],
        validation_split: 0.0,
        path: Some(path_str),
        schema: None,
        skip_header: Some(false),
        delimiter: None,
        validate: Some(false),
        extra_params: HashMap::new(),
    };
    Ok(loader::file::FileDataLoader::new(config))
}

/// 创建数据验证器
pub fn create_data_validator(schema: DataSchema) -> Box<dyn crate::data::validation::DataValidator> {
    Box::new(crate::data::validation::type_validator::TypeValidator::new(schema.name.clone(), schema))
}

/// 创建数据转换器
pub fn create_data_transformer() -> crate::data::transform::DataTransformer {
    crate::data::transform::DataTransformer::new()
}

/// 创建批次数据迭代器
pub fn create_batch_iterator(
    loader: std::sync::Arc<dyn crate::data::loader::DataLoader>,
    path: String,
    batch_size: usize,
) -> crate::data::batch::DataIterator {
    crate::data::batch::DataIterator::new(loader, path, batch_size)
}

/// 创建流式数据迭代器
pub fn create_streaming_iterator(
    source: std::sync::Arc<std::sync::Mutex<dyn crate::data::iterator::StreamingDataSource>>,
    batch_size: usize,
) -> crate::data::iterator::StreamingDataIterator {
    crate::data::iterator::StreamingDataIterator::new(source, batch_size)
}

/// 创建数据管理器
pub fn create_data_manager(storage: std::sync::Arc<crate::storage::Storage>) -> crate::data::manager_core::DataManager {
    crate::data::manager_core::DataManager::new(storage)
}

/// 创建异步缓存管理器
pub fn create_cache_manager<K, V>(capacity: usize) -> crate::data::cache::AsyncCacheManager<K, V>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    crate::data::cache::AsyncCacheManager::new(capacity)
}

/// 快速创建数据集
pub fn create_quick_dataset(
    name: &str,
    features: Vec<Vec<f64>>,
    labels: Option<Vec<Vec<f64>>>,
) -> Result<crate::data::dataset::Dataset> {
    crate::data::dataset::Dataset::new(features, labels)
}

/// 创建数据批次
pub fn create_data_batch(
    dataset_id: &str,
    index: usize,
    size: usize,
) -> crate::data::batch::DataBatch {
    crate::data::batch::DataBatch::new(dataset_id, index, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_infer_schema_from_csv() {
        let path = PathBuf::from("test.csv");
        let schema = infer_schema_from_file(&path).unwrap();
        assert_eq!(schema.name(), "inferred_schema");
        assert!(schema.has_field("id"));
        assert!(schema.has_field("name"));
        assert!(schema.has_field("value"));
    }

    #[test]
    fn test_infer_schema_from_json() {
        let path = PathBuf::from("test.json");
        let schema = infer_schema_from_file(&path).unwrap();
        assert_eq!(schema.name(), "json_schema");
        assert!(schema.has_field("data"));
    }

    #[test]
    fn test_create_memory_loader() {
        let loader = create_memory_loader();
        // 测试加载器创建成功
        assert_eq!(loader.get_type(), "memory");
    }

    #[test]
    fn test_create_data_pipeline() {
        let pipeline = create_data_pipeline();
        // 测试管道创建成功
        assert_eq!(pipeline.stage_count(), 0);
    }

    #[test]
    fn test_data_record_from_text() {
        let record = DataRecord::from_text("Hello, world!");
        assert_eq!(record.fields.len(), 1);
        assert!(record.fields.contains_key("content"));
        if let Some(DataValue::String(content)) = record.fields.get("content") {
            assert_eq!(content, "Hello, world!");
        } else {
            panic!("Expected string content");
        }
    }

    #[test]
    fn test_data_record_from_json() {
        let json_str = r#"{"name": "test", "value": 123, "active": true}"#;
        let record = DataRecord::from_json(json_str).unwrap();
        
        assert_eq!(record.fields.len(), 3);
        assert!(record.fields.contains_key("name"));
        assert!(record.fields.contains_key("value"));
        assert!(record.fields.contains_key("active"));
        
        if let Some(DataValue::String(name)) = record.fields.get("name") {
            assert_eq!(name, "test");
        } else {
            panic!("Expected string name");
        }
        
        if let Some(DataValue::Integer(value)) = record.fields.get("value") {
            assert_eq!(*value, 123);
        } else {
            panic!("Expected integer value");
        }
        
        if let Some(DataValue::Boolean(active)) = record.fields.get("active") {
            assert!(*active);
        } else {
            panic!("Expected boolean active");
        }
    }
} 