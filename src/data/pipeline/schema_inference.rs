use std::path::{Path, PathBuf};
use log::{debug, info};
use std::collections::HashMap;

use crate::data::schema::DataSchema;
use crate::data::pipeline::{PipelineStage, PipelineContext, Result};
use crate::Error;
use crate::data::loader::file::{infer_schema, LoaderFileType};
use crate::data::loader::FileType;

/// 模式推断阶段
pub struct SchemaInferenceStage {
    /// 是否推断模式
    infer_schema: bool,
    /// 用户提供的模式
    user_schema: Option<DataSchema>,
}

impl SchemaInferenceStage {
    /// 创建新的模式推断阶段
    pub fn new() -> Self {
        SchemaInferenceStage {
            infer_schema: true,
            user_schema: None,
        }
    }
    
    /// 设置用户提供的模式
    pub fn with_schema(mut self, schema: DataSchema) -> Self {
        self.user_schema = Some(schema);
        self
    }
    
    /// 设置是否需要推断模式
    pub fn with_infer_schema(mut self, infer: bool) -> Self {
        self.infer_schema = infer;
        self
    }
    
    /// 从文件推断模式
    fn infer_schema_from_file(&self, path: &Path, file_type: FileType) -> Result<DataSchema> {
        debug!("从文件推断模式: {}", path.display());
        
        // 将 FileType 转换为 LoaderFileType
        let loader_file_type = match file_type {
            FileType::Csv => LoaderFileType::Csv,
            FileType::Json => LoaderFileType::Json,
            FileType::Parquet => LoaderFileType::Parquet,
            FileType::Excel => LoaderFileType::Excel,
            FileType::Avro | FileType::Unknown | FileType::Sqlite | FileType::Xml | FileType::Text | FileType::Binary | FileType::Other => LoaderFileType::Other, // 其他类型映射为 Other
        };
        
        // 调用通用的模式推断函数
        let schema = infer_schema(path, loader_file_type)?;
        
        debug!("推断模式成功，字段数: {}", schema.fields().len());
        
        Ok(schema)
    }
}

impl PipelineStage for SchemaInferenceStage {
    fn name(&self) -> &str {
        "模式推断阶段"
    }
    
    fn can_process(&self, context: &PipelineContext) -> bool {
        context.params.contains_key("source_path") &&
        (context.params.contains_key("file_type") || context.params.contains_key("file_path"))
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        info!("执行模式推断阶段");
        
        // 检查是否需要推断模式
        let infer_schema = match ctx.get_param::<bool>("infer_schema") {
            Ok(value) => value,
            Err(_) => self.infer_schema, // 使用默认值
        };
        
        // 如果不需要推断且有用户提供的模式，直接使用
        if !infer_schema {
            if let Some(schema) = &self.user_schema {
                debug!("使用用户提供的模式，字段数: {}", schema.fields().len());
                ctx.add_data("schema", schema.clone())?;
                return Ok(());
            } else if let Ok(schema) = ctx.get_param::<DataSchema>("schema") {
                debug!("使用上下文中的模式，字段数: {}", schema.fields().len());
                return Ok(());
            } else {
                return Err(Error::invalid_parameter(
                    "未指定推断模式，但也未提供用户模式"
                ));
            }
        }
        
        // 获取源文件路径
        let source_path = ctx.get_param::<String>("source_path")
            .map_err(|_| Error::invalid_parameter("source_path"))?;
        
        // 获取文件类型（从字符串获取，因为 FileType 无法反序列化）
        let file_type = if let Ok(ft_str) = ctx.get_param::<String>("file_type") {
            // 从字符串解析 FileType
            match ft_str.as_str() {
                "Csv" => FileType::Csv,
                "Json" => FileType::Json,
                "Parquet" => FileType::Parquet,
                "Avro" => FileType::Avro,
                "Excel" => FileType::Excel,
                "Unknown" => FileType::Unknown,
                _ => FileType::Unknown,
            }
        } else {
            // 如果没有提前检测到文件类型，尝试从上下文中的文件路径检测
            let file_path = ctx.get_param::<String>("file_path")
                .map_err(|_| Error::invalid_parameter("file_type 或 file_path"))?;
            
            let path = PathBuf::from(&file_path);
            FileType::from_path(&path)
        };
        
        debug!("从文件推断模式，类型: {:?}", file_type);
        
        // 转换为路径对象
        let path = PathBuf::from(&source_path);
        
        // 从文件推断模式
        let schema = self.infer_schema_from_file(&path, file_type)?;
        
        // 将结果存入上下文
        ctx.add_data("schema", schema.clone())?;
        
        info!("模式推断完成，字段数: {}", schema.fields().len());
        
        Ok(())
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("requires_file_path".to_string(), "true".to_string());
        metadata.insert("requires_file_type".to_string(), "true".to_string());
        metadata
    }
} 