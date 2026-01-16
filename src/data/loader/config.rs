use std::collections::HashMap;
use std::path::Path;
use crate::data::DataSchema;
use crate::error::{Error, Result};
use crate::data::loader::types::FileType;

/// 导入配置结构体
#[derive(Debug, Clone)]
pub struct ImportConfig {
    /// 源文件路径
    pub source_path: String,
    /// 文件格式（自动检测则为None）
    pub format: Option<String>,
    /// 目标存储位置
    pub target_location: String,
    /// 批处理大小
    pub batch_size: Option<usize>,
    /// 是否验证
    pub validate: Option<bool>,
    /// 是否覆盖已有数据
    pub overwrite: Option<bool>,
    /// 是否推断模式
    pub infer_schema: Option<bool>,
    /// 自定义模式
    pub schema: Option<DataSchema>,
    /// 处理器选项
    pub processor_options: HashMap<String, String>,
    /// 文件处理器选项
    pub file_options: HashMap<String, String>,
    /// 处理器配置
    pub processor_config: Option<crate::data::processor::config::ProcessorConfig>,
    /// API兼容性字段 - 目标数据集ID
    pub target_dataset_id: Option<String>,
    /// API兼容性字段 - 是否有标题行
    pub has_header: Option<bool>,
    /// API兼容性字段 - 分隔符
    pub delimiter: Option<String>,
    /// API兼容性字段 - 编码格式
    pub encoding: Option<String>,
    /// API兼容性字段 - 分块大小
    pub chunk_size: Option<usize>,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            source_path: String::new(),
            format: None,
            target_location: String::new(),
            batch_size: Some(1000),
            validate: Some(true),
            overwrite: Some(false),
            infer_schema: Some(true),
            schema: None,
            processor_options: HashMap::new(),
            file_options: HashMap::new(),
            processor_config: None,
            target_dataset_id: None,
            has_header: None,
            delimiter: None,
            encoding: None,
            chunk_size: None,
        }
    }
}

impl ImportConfig {
    /// 创建新的导入配置
    pub fn new(source_path: &str) -> Self {
        Self {
            source_path: source_path.to_string(),
            ..Default::default()
        }
    }
    
    /// 设置源路径
    pub fn with_source_path<S: Into<String>>(mut self, path: S) -> Self {
        self.source_path = path.into();
        self
    }
    
    /// 设置目标位置
    pub fn with_target_location<S: Into<String>>(mut self, location: S) -> Self {
        self.target_location = location.into();
        self
    }
    
    /// 设置文件格式
    pub fn with_format<S: Into<String>>(mut self, format: S) -> Self {
        self.format = Some(format.into());
        self
    }
    
    /// 设置批处理大小
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }
    
    /// 设置验证标志
    pub fn with_validate(mut self, validate: bool) -> Self {
        self.validate = Some(validate);
        self
    }
    
    /// 设置覆盖标志
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = Some(overwrite);
        self
    }
    
    /// 设置自定义模式
    pub fn with_schema(mut self, schema: DataSchema) -> Self {
        self.schema = Some(schema);
        self
    }
    
    /// 添加处理器选项
    pub fn with_processor_option<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.processor_options.insert(key.into(), value.into());
        self
    }
    
    /// 添加文件处理器选项
    pub fn with_file_option<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.file_options.insert(key.into(), value.into());
        self
    }
    
    /// 从文件路径创建配置
    pub fn from_file_path<P: AsRef<Path>>(path: P) -> Self {
        let source_path = path.as_ref().to_string_lossy().to_string();
        
        // 检测文件类型
        let format = match FileType::from_path(path.as_ref()).to_string() {
            s if !s.is_empty() => Some(s),
            _ => None,
        };
        
        Self {
            source_path,
            format,
            ..Default::default()
        }
    }
    
    /// 验证配置
    pub fn validate(&self) -> Result<()> {
        // 检查源路径
        if self.source_path.is_empty() {
            return Err(Error::invalid_input("源路径不能为空"));
        }
        
        // 检查源路径是否存在
        let path = Path::new(&self.source_path);
        if !path.exists() {
            return Err(Error::not_found(&format!("源文件不存在: {:?}", path)));
        }
        
        // 检查批处理大小是否为正数
        if let Some(batch_size) = self.batch_size {
            if batch_size == 0 {
                return Err(Error::invalid_input("批处理大小必须大于0"));
            }
        }
        
        Ok(())
    }

    /// 获取批处理大小
    pub fn get_batch_size(&self) -> usize {
        self.batch_size.unwrap_or(1000)
    }

    /// 是否验证
    pub fn should_validate(&self) -> bool {
        self.validate.unwrap_or(true)
    }

    /// 是否覆盖已有数据
    pub fn should_overwrite(&self) -> bool {
        self.overwrite.unwrap_or(false)
    }

    /// 将处理器选项合并到当前配置中
    pub fn merge_processor_options(&mut self, options: &HashMap<String, String>) -> &mut Self {
        for (key, value) in options {
            self.processor_options.insert(key.clone(), value.clone());
        }
        self
    }

    /// 将文件处理器选项合并到当前配置中
    pub fn merge_file_options(&mut self, options: &HashMap<String, String>) -> &mut Self {
        for (key, value) in options {
            self.file_options.insert(key.clone(), value.clone());
        }
        self
    }

    /// 设置处理器配置
    pub fn with_processor_config(mut self, config: crate::data::processor::config::ProcessorConfig) -> Self {
        self.processor_config = Some(config);
        self
    }
    
    /// 获取处理器配置（带默认值）
    pub fn get_processor_config(&self) -> crate::data::processor::config::ProcessorConfig {
        self.processor_config.clone().unwrap_or_default()
    }
}

/// 批量导入配置
#[derive(Debug, Clone)]
pub struct BatchImportConfig {
    /// 基础导入配置
    pub base_config: ImportConfig,
    /// 最大并发处理数
    pub max_concurrent: usize,
    /// 超时时间（秒）
    pub timeout_seconds: u64,
    /// 目录匹配模式
    pub pattern: Option<String>,
    /// 是否递归处理子目录
    pub recursive: bool,
}

impl BatchImportConfig {
    /// 创建新的批量导入配置
    pub fn new(base_config: ImportConfig) -> Self {
        Self {
            base_config,
            max_concurrent: 5,
            timeout_seconds: 300,
            pattern: None,
            recursive: false,
        }
    }

    /// 设置最大并发数
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// 设置匹配模式
    pub fn with_pattern<S: Into<String>>(mut self, pattern: S) -> Self {
        self.pattern = Some(pattern.into());
        self
    }

    /// 设置是否递归处理子目录
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// 验证配置
    pub fn validate(&self) -> Result<()> {
        // 验证基础配置
        if self.base_config.source_path.is_empty() {
            return Err(Error::invalid_input("源路径不能为空"));
        }
        
        // 验证最大并发数
        if self.max_concurrent == 0 {
            return Err(Error::invalid_input("最大并发数必须大于0"));
        }
        
        // 验证超时时间
        if self.timeout_seconds == 0 {
            return Err(Error::invalid_input("超时时间必须大于0"));
        }
        
        Ok(())
    }
} 