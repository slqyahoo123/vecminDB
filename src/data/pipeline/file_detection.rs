use std::path::{Path, PathBuf};
use std::fs;
use log::{debug, info, warn};
use std::collections::HashMap;

use crate::data::pipeline::{PipelineStage, PipelineContext, Result};
use crate::Error;
use crate::data::loader::file::detect_file_type;
use crate::data::loader::FileType;

/// 文件检测阶段
pub struct FileDetectionStage {
    /// 源文件路径
    source_path: Option<String>,
}

impl FileDetectionStage {
    /// 创建新的文件检测阶段
    pub fn new() -> Self {
        FileDetectionStage {
            source_path: None,
        }
    }
    
    /// 检查文件是否存在
    fn check_file_exists(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(Error::not_found(
                format!("文件不存在: {}", path.display())
            ));
        }
        
        if !path.is_file() {
            return Err(Error::invalid_input(
                format!("路径不是文件: {}", path.display())
            ));
        }
        
        Ok(())
    }
    
    /// 获取文件大小
    fn get_file_size(&self, path: &Path) -> Result<u64> {
        let metadata = fs::metadata(path)
            .map_err(|e| Error::io_error(format!("获取文件元数据失败: {}", e)))?;
            
        Ok(metadata.len())
    }
    
    /// 检测文件类型
    fn detect_file_type(&self, path: &Path) -> Result<FileType> {
        // detect_file_type 返回 String，需要转换为 FileType
        let file_type_str = detect_file_type(path)?;
        debug!("检测到文件类型字符串: {}", file_type_str);
        
        // 将字符串转换为 FileType 枚举
        let file_type = match file_type_str.to_lowercase().as_str() {
            "csv" => FileType::Csv,
            "json" => FileType::Json,
            "parquet" => FileType::Parquet,
            "avro" => FileType::Avro,
            "excel" => FileType::Excel,
            "sqlite" => FileType::Sqlite,
            "xml" => FileType::Xml,
            "text" => FileType::Text,
            "binary" => FileType::Binary,
            "other" => FileType::Other,
            _ => FileType::Unknown,
        };
        
        if file_type == FileType::Unknown {
            warn!("无法确定文件类型，将使用扩展名或其他提示");
        }
        
        Ok(file_type)
    }
}

impl PipelineStage for FileDetectionStage {
    fn name(&self) -> &str {
        "文件检测阶段"
    }
    
    fn can_process(&self, context: &PipelineContext) -> bool {
        context.params.contains_key("source_path")
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        info!("执行文件检测阶段");
        
        // 从上下文获取源文件路径
        let source_path = ctx.get_param::<String>("source_path")
            .map_err(|_| Error::invalid_input("source_path"))?;
        
        debug!("检测文件: {}", source_path);
        
        // 转换为路径对象
        let path = PathBuf::from(&source_path);
        
        // 检查文件是否存在
        self.check_file_exists(&path)?;
        
        // 获取文件大小
        let file_size = self.get_file_size(&path)?;
        debug!("文件大小: {} 字节", file_size);
        
        // 检测文件类型
        let file_type = self.detect_file_type(&path)?;
        
        // 将结果存入上下文
        ctx.add_data("file_exists", true)?;
        ctx.add_data("file_size", file_size)?;
        // FileType 无法序列化，使用字符串表示
        ctx.add_data("file_type", format!("{:?}", file_type))?;
        ctx.add_data("file_path", source_path.clone())?;
        
        info!("文件检测完成: 类型={:?}, 大小={} 字节", file_type, file_size);
        
        Ok(())
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("requires_file_path".to_string(), "true".to_string());
        metadata
    }
} 