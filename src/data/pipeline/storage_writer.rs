use std::collections::HashMap;
use std::path::Path;
use log::{info, debug, warn};

use crate::data::pipeline::pipeline::{PipelineStage, PipelineContext};
use crate::data::schema::Schema;
use crate::data::batch::DataBatch;
use crate::storage::{StorageEngineImpl, config::StorageConfig};

// UnifiedStorageConfig stub - 使用StorageConfig替代
type UnifiedStorageConfig = StorageConfig;
use crate::Error;

/// 存储写入阶段
#[derive(Clone)]
pub struct StorageWriteStage {
    /// 阶段名称
    name: String,
    /// 目标存储位置
    target_location: Option<String>,
    /// 存储配置
    storage_config: Option<UnifiedStorageConfig>,
    /// 是否覆盖已有数据
    overwrite: bool,
    /// 数据模式
    schema: Option<Schema>,
}

impl StorageWriteStage {
    /// 创建新的存储写入阶段
    pub fn new(name: &str) -> Self {
        StorageWriteStage {
            name: name.to_string(),
            target_location: None,
            storage_config: None,
            overwrite: false,
            schema: None,
        }
    }
    
    /// 设置目标位置
    pub fn with_target(mut self, target: &str) -> Self {
        self.target_location = Some(target.to_string());
        self
    }
    
    /// 设置存储配置
    pub fn with_config(mut self, config: UnifiedStorageConfig) -> Self {
        self.storage_config = Some(config);
        self
    }
    
    /// 设置是否覆盖
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }
    
    /// 设置数据模式
    pub fn with_schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }
    
    /// 从上下文准备阶段数据
    fn prepare_from_context(&mut self, ctx: &PipelineContext) -> Result<(), Error> {
        // 获取目标位置
        if self.target_location.is_none() {
            if let Ok(target) = ctx.get_string("target_location") {
                self.target_location = Some(target);
            } else {
                return Err(Error::new("缺少存储目标位置"));
            }
        }
        
        // 获取存储配置
        if self.storage_config.is_none() {
            if let Ok(config) = ctx.get_data::<UnifiedStorageConfig>("storage_config") {
                self.storage_config = Some(config);
            }
        }
        
        // 获取覆盖选项
        if let Ok(overwrite) = ctx.get_data::<bool>("overwrite") {
            self.overwrite = overwrite;
        }
        
        // 获取数据模式
        if self.schema.is_none() {
            if let Ok(s) = ctx.get_data::<Schema>("schema") {
                self.schema = Some(s);
            }
        }
        
        Ok(())
    }
}

impl PipelineStage for StorageWriteStage {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> Option<&str> {
        Some("将数据写入存储系统")
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<(), Error> {
        info!("执行存储写入阶段: {}", self.name);
        
        // 克隆阶段以进行修改
        let mut stage = self.clone();
        
        // 从上下文准备参数
        stage.prepare_from_context(ctx)?;
        
        // 检查目标位置
        let target = stage.target_location.as_ref()
            .ok_or_else(|| Error::new("未指定存储目标位置"))?;
        
        // 获取数据
        let batch = ctx.get_data::<DataBatch>("processed_data")
            .map_err(|_| Error::new("上下文中缺少处理后的数据"))?;
        
        // 检查目标位置是否已存在数据
        let target_path = Path::new(target);
        if target_path.exists() && !stage.overwrite {
            warn!("目标位置 {} 已存在数据且未设置覆盖选项", target);
            return Err(Error::new(&format!("目标位置 {} 已存在数据", target)));
        }
        
        debug!("写入数据到存储位置: {}", target);
        
        // 创建存储引擎实例
        // 将 UnifiedStorageConfig 转换为 StorageConfig
        let unified_config = stage.storage_config.clone().unwrap_or_default();
        let mut config = StorageConfig::default();
        config.path = unified_config.data_path.clone();
        config.create_if_missing = true;
        config.max_background_jobs = 4;
        config.write_buffer_size = unified_config.write_buffer_mb * 1024 * 1024;
        config.max_open_files = unified_config.max_open_files as i32;
        // 如果 unified_config 有 compression 字段（不为 None），则启用压缩
        config.use_compression = unified_config.compression.is_some();
        config.max_file_size = 64 * 1024 * 1024; // 64MB
        config.max_connections = 100;
        config.connection_timeout = unified_config.transaction_timeout_secs * 1000;
        let storage_engine = StorageEngineImpl::new(config)?;
        
        // 生成唯一数据ID
        let data_id = uuid::Uuid::new_v4().to_string();
        
        // 写入数据
        debug!("使用数据ID {} 保存数据", data_id);
        
        // 序列化数据并保存
        let serialized_data = bincode::serialize(&batch)?;
        let key = format!("data:{}:content", data_id);
        
        // 使用存储引擎的写入方法
        // 由于 StorageEngineImpl 使用异步锁，我们需要在运行时上下文中执行
        let db = storage_engine.get_db_clone();
        
        // 使用 tokio runtime 来执行异步操作
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| Error::system(&format!("无法创建运行时: {}", e)))?;
        
        rt.block_on(async {
            let mut db_guard = db.write().await;
            
            // 保存数据
            let data_key = format!("data:{}:content", data_id);
            let data_bytes = bincode::serialize(&batch)
                .map_err(|e| Error::serialization(&format!("序列化数据失败: {}", e)))?;
            db_guard.insert(data_key.as_bytes(), data_bytes.as_slice())
                .map_err(|e| Error::StorageError(format!("写入数据失败: {}", e)))?;
            
            // 如果有模式，保存模式信息
            if let Some(schema) = &stage.schema {
                let schema_key = format!("data:{}:schema", data_id);
                let schema_data = bincode::serialize(schema)
                    .map_err(|e| Error::serialization(&format!("序列化模式失败: {}", e)))?;
                db_guard.insert(schema_key.as_bytes(), schema_data.as_slice())
                    .map_err(|e| Error::StorageError(format!("写入模式失败: {}", e)))?;
            }
            
            // 保存目标路径与数据ID的映射
            let path_key = format!("path:{}", target);
            db_guard.insert(path_key.as_bytes(), data_id.as_bytes())
                .map_err(|e| Error::StorageError(format!("写入路径映射失败: {}", e)))?;
            
            Ok::<(), Error>(())
        })?;
        
        info!("成功写入数据到 {}, 数据ID: {}", target, data_id);
        
        // 更新上下文
        ctx.add_data("data_id", data_id.clone())?;
        ctx.add_data("storage_target", target.clone())?;
        
        // 获取写入记录数
        let count = batch.len();
        ctx.add_data("written_count", count)?;
        
        Ok(())
    }
    
    fn can_process(&self, ctx: &PipelineContext) -> bool {
        // 检查上下文中是否有处理后的数据
        ctx.get_data::<DataBatch>("processed_data").is_ok()
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), self.name.clone());
        
        if let Some(target) = &self.target_location {
            metadata.insert("target".to_string(), target.clone());
        }
        
        metadata.insert("overwrite".to_string(), self.overwrite.to_string());
        
        metadata
    }
}

// 辅助函数：获取数据记录数
fn get_record_count(data: &Box<dyn std::any::Any>) -> Option<usize> {
    // 尝试从不同类型的数据结构中获取记录数
    if let Some(batch) = data.downcast_ref::<crate::data::DataBatch>() {
        return Some(batch.len());
    } else if let Some(vec_data) = data.downcast_ref::<Vec<HashMap<String, serde_json::Value>>>() {
        return Some(vec_data.len());
    } else if let Some(map_data) = data.downcast_ref::<HashMap<String, Vec<serde_json::Value>>>() {
        return Some(map_data.values().map(|v| v.len()).sum());
    }
    
    None
} 