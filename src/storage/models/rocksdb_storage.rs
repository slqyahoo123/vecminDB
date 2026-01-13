use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use chrono::Utc;
use rocksdb::{DB, Options, ColumnFamilyDescriptor, WriteBatch};
use serde_json;
use log::debug;

use crate::{Result, Error};
use crate::compat::{Model, ModelArchitecture, ModelParameters};
use crate::storage::models::implementation::{
    ModelStorage, StoredModel,
    ModelInfo, ModelMetrics, StorageFormat, StorageOptions, CompressionType, PersistenceOptions, WriteMode
};
// removed unused config and hashing imports

/// 将 ManagedTensorData 转换为 TensorData
fn convert_managed_to_tensor_data(tensor: &ManagedTensorData) -> Result<TensorData> {
    let data = tensor.to_vec_f32()?;
    
    Ok(TensorData {
        shape: tensor.shape().to_vec(),
        data: crate::model::tensor::TensorValues::F32(data),
        dtype: crate::model::tensor::DataType::Float32,
        metadata: std::collections::HashMap::new(),
        version: 1,
    })
}

/// RocksDB模型存储实现
pub struct RocksDBModelStorage {
    /// RocksDB实例
    db: Arc<RwLock<DB>>,
    /// 存储选项
    options: StorageOptions,
    /// 列族句柄缓存（RocksDB ColumnFamily 不是线程安全的，需要在使用时从 DB 获取）
    _cf_names: Vec<String>,
}

// 列族常量
const CF_MODELS: &str = "models";
const CF_PARAMETERS: &str = "parameters";
const CF_ARCHITECTURES: &str = "architectures";
const CF_MODEL_INFO: &str = "model_info";
const CF_MODEL_METRICS: &str = "model_metrics";

impl RocksDBModelStorage {
    /// 创建新的RocksDB模型存储
    pub fn new(options: StorageOptions) -> Result<Self> {
        // 创建存储目录
        let path = PathBuf::from(&options.path);
        if !path.exists() && options.create_if_missing {
            std::fs::create_dir_all(&path)?;
        }

        // 创建RocksDB选项
        let mut db_opts = Options::default();
        db_opts.create_if_missing(options.create_if_missing);
        db_opts.create_missing_column_families(true);

        // 配置RocksDB选项
        if let Some(cache_size) = options.cache_size_mb {
            db_opts.set_db_write_buffer_size(cache_size as usize * 1024 * 1024);
        }
        if options.use_wal {
            db_opts.set_manual_wal_flush(false);
        } else {
            db_opts.set_manual_wal_flush(true);
        }
        if let Some(threads) = options.max_threads {
            db_opts.increase_parallelism(threads as i32);
        }
        if let Some(write_buffer) = options.write_cache_size_kb {
            db_opts.set_write_buffer_size(write_buffer as usize * 1024);
        }

        // 设置压缩选项
        match &options.compression_type {
            Some(CompressionType::Gzip) => {
                db_opts.set_compression_type(rocksdb::DBCompressionType::Zlib);
            },
            Some(CompressionType::LZ4) => {
                db_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            },
            Some(CompressionType::Zstd) => {
                db_opts.set_compression_type(rocksdb::DBCompressionType::Zstd);
            },
            Some(CompressionType::Snappy) => {
                db_opts.set_compression_type(rocksdb::DBCompressionType::Snappy);
            },
            _ => {
                db_opts.set_compression_type(rocksdb::DBCompressionType::None);
            }
        }

        // 定义列族
        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_MODELS, db_opts.clone()),
            ColumnFamilyDescriptor::new(CF_PARAMETERS, db_opts.clone()),
            ColumnFamilyDescriptor::new(CF_ARCHITECTURES, db_opts.clone()),
            ColumnFamilyDescriptor::new(CF_MODEL_INFO, db_opts.clone()),
            ColumnFamilyDescriptor::new(CF_MODEL_METRICS, db_opts.clone()),
        ];

        // 打开数据库
        let db = DB::open_cf_descriptors(&db_opts, &path, cf_descriptors)?;

        // 保存列族名称（RocksDB ColumnFamily 不是线程安全的，需要在使用时从 DB 获取）
        let _cf_names = vec![
            CF_MODELS.to_string(),
            CF_PARAMETERS.to_string(),
            CF_ARCHITECTURES.to_string(),
            CF_MODEL_INFO.to_string(),
            CF_MODEL_METRICS.to_string(),
        ];

        Ok(Self {
            db: Arc::new(RwLock::new(db)),
            options,
            _cf_names,
        })
    }

    /// 获取列族句柄（从 DB 直接获取，因为 ColumnFamily 不是线程安全的）
    fn get_cf_handle(&self, cf_name: &str) -> Result<Arc<rocksdb::BoundColumnFamily>> {
        let db = self.db.read().map_err(|e| Error::lock(e.to_string()))?;
        db.cf_handle(cf_name)
            .ok_or_else(|| Error::invalid_state(format!("找不到列族: {}", cf_name)))
    }

    /// 构建模型键
    fn make_model_key(model_id: &str) -> String {
        format!("model:{}", model_id)
    }

    /// 构建参数键
    fn make_params_key(model_id: &str, version: Option<&str>) -> String {
        match version {
            Some(v) => format!("params:{}:{}", model_id, v),
            None => format!("params:{}", model_id),
        }
    }

    /// 构建架构键
    fn make_arch_key(model_id: &str, version: Option<&str>) -> String {
        match version {
            Some(v) => format!("arch:{}:{}", model_id, v),
            None => format!("arch:{}", model_id),
        }
    }

    /// 构建信息键
    fn make_info_key(model_id: &str) -> String {
        format!("info:{}", model_id)
    }

    /// 构建指标键
    fn make_metrics_key(model_id: &str) -> String {
        format!("metrics:{}", model_id)
    }
}

impl ModelStorage for RocksDBModelStorage {
    /// 保存模型
    fn save_model(&self, model: &Model) -> Result<()> {
        // 创建存储模型
        let stored_model = StoredModel {
            id: model.id.clone(),
            name: model.name.clone(),
            description: model.description.clone(),
            created_by: "system".to_string(), // Model 没有 created_by 字段
            created_at: model.created_at.timestamp() as u64,
            updated_at: model.updated_at.timestamp() as u64,
            tags: Vec::new(), // Model 没有 tags 字段
            metadata: model.metadata.clone(),
        };

        // 序列化模型
        let model_data = bincode::serialize(&stored_model)?;
        let key = Self::make_model_key(&model.id);

        // 保存模型
        let mut db = self.db.write().map_err(|e| Error::lock(e.to_string()))?;
        let cf_models = db.cf_handle(CF_MODELS)
            .ok_or_else(|| Error::invalid_state(format!("找不到列族: {}", CF_MODELS)))?;
        db.put_cf(&cf_models, key, model_data)?;

        // 保存参数和架构（新结构：smart_parameters + architecture 必填）
        {
            // 使用 from_parameters 方法转换
            let params = crate::model::parameters::ModelParameters::new(
                model.id.clone(),
                model.smart_parameters.get_parameters().iter()
                    .filter_map(|(k, v)| convert_managed_to_tensor_data(v).ok().map(|t| (k.clone(), t)))
                    .collect()
            );
            self.save_model_params(&model.id, &params)?;
        }

        {
            self.save_model_architecture(&model.id, &model.architecture)?;
        }

        Ok(())
    }

    /// 获取模型
    fn get_model(&self, model_id: &str) -> Result<Option<Model>> {
        let key = Self::make_model_key(model_id);
        let cf_models = self.get_cf_handle(CF_MODELS)?;

        // 获取模型数据
        let db = self.db.read().map_err(|e| Error::lock(e.to_string()))?;
        let model_data = match db.get_cf(&cf_models, key)? {
            Some(data) => data,
            None => return Ok(None),
        };

        // 反序列化模型
        let stored_model: StoredModel = bincode::deserialize(&model_data)?;

        // 获取参数和架构
        let parameters = self.get_model_params(model_id)?;
        let architecture = self.get_model_architecture(model_id)?;

        // 构建模型对象
        let created_at = chrono::DateTime::from_timestamp(stored_model.created_at as i64, 0)
            .unwrap_or_else(|| Utc::now());
        let updated_at = chrono::DateTime::from_timestamp(stored_model.updated_at as i64, 0)
            .unwrap_or_else(|| Utc::now());

        let model = Model {
            id: stored_model.id,
            name: stored_model.name,
            description: stored_model.description,
            version: "1.0.0".to_string(),
            model_type: "generic".to_string(),
            smart_parameters: match parameters {
                Some(p) => crate::model::memory_management::SmartModelParameters::from_parameters(p),
                None => crate::model::memory_management::SmartModelParameters::default(),
            },
            architecture: architecture.unwrap_or_default(),
            status: ModelStatus::Created,
            metrics: None,
            created_at,
            updated_at,
            parent_id: None,
            metadata: stored_model.metadata,
            input_shape: architecture.as_ref().map(|a| a.input_shape.clone()).unwrap_or_default(),
            output_shape: architecture.as_ref().map(|a| a.output_shape.clone()).unwrap_or_default(),
            import_source: None,
            memory_monitor: Arc::new(Mutex::new(crate::model::memory_monitor::ModelMemoryMonitor::new())),
            // 兼容旧字段
            // tags 在新结构中并未直接存在，如有需要可存入 metadata
            // Model 没有 tags 字段，已移除
        };

        Ok(Some(model))
    }

    /// 删除模型
    fn delete_model(&self, model_id: &str) -> Result<bool> {
        let key = Self::make_model_key(model_id);
        let cf_models = self.get_cf_handle(CF_MODELS)?;
        let db = self.db.write().map_err(|e| Error::lock(e.to_string()))?;

        // 检查模型是否存在
        if db.get_cf(&cf_models, &key)?.is_none() {
            return Ok(false);
        }

        // 创建批量删除
        let mut batch = WriteBatch::default();
        
        // 删除模型
        batch.delete_cf(&cf_models, key);
        
        // 删除参数
        let cf_params = self.get_cf_handle(CF_PARAMETERS)?;
        let params_key = Self::make_params_key(model_id, None);
        batch.delete_cf(&cf_params, params_key);
        
        // 删除架构
        let cf_arch = self.get_cf_handle(CF_ARCHITECTURES)?;
        let arch_key = Self::make_arch_key(model_id, None);
        batch.delete_cf(&cf_arch, arch_key);
        
        // 删除信息
        let cf_info = self.get_cf_handle(CF_MODEL_INFO)?;
        let info_key = Self::make_info_key(model_id);
        batch.delete_cf(&cf_info, info_key);
        
        // 删除指标
        let cf_metrics = self.get_cf_handle(CF_MODEL_METRICS)?;
        let metrics_key = Self::make_metrics_key(model_id);
        batch.delete_cf(&cf_metrics, metrics_key);
        
        // 执行批量删除
        db.write(batch)?;
        
        Ok(true)
    }

    /// 保存模型参数
    fn save_model_params(&self, model_id: &str, params: &ModelParameters) -> Result<()> {
        // 序列化参数
        let params_data = bincode::serialize(params)?;
        let key = Self::make_params_key(model_id, Some(&params.version));
        
        // 保存参数
        let cf_params = self.get_cf_handle(CF_PARAMETERS)?;
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put_cf(&cf_params, key, &params_data)?;
        
        // 保存当前版本的别名
        let current_key = Self::make_params_key(model_id, None);
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put_cf(&cf_params, current_key, &params_data)?;
        
        Ok(())
    }

    /// 获取模型参数
    fn get_model_params(&self, model_id: &str) -> Result<Option<ModelParameters>> {
        let key = Self::make_params_key(model_id, None);
        let cf_params = self.get_cf_handle(CF_PARAMETERS)?;
        
        // 获取参数数据
        let db = self.db.read().map_err(|e| Error::lock(e.to_string()))?;
        let params_data = match db.get_cf(&cf_params, key)? {
            Some(data) => data,
            None => return Ok(None),
        };
        
        // 反序列化参数
        let params: ModelParameters = bincode::deserialize(&params_data)?;
        
        Ok(Some(params))
    }

    /// 保存模型架构
    fn save_model_architecture(&self, model_id: &str, architecture: &ModelArchitecture) -> Result<()> {
        // 序列化架构
        let arch_data = bincode::serialize(architecture)?;
        let key = Self::make_arch_key(model_id, None);
        
        // 保存架构
        let cf_arch = self.get_cf_handle(CF_ARCHITECTURES)?;
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put_cf(&cf_arch, key, arch_data)?;
        
        Ok(())
    }

    /// 获取模型架构
    fn get_model_architecture(&self, model_id: &str) -> Result<Option<ModelArchitecture>> {
        let key = Self::make_arch_key(model_id, None);
        let cf_arch = self.get_cf_handle(CF_ARCHITECTURES)?;
        
        // 获取架构数据
        let db = self.db.read().map_err(|e| Error::lock(e.to_string()))?;
        let arch_data = match db.get_cf(&cf_arch, key)? {
            Some(data) => data,
            None => return Ok(None),
        };
        
        // 反序列化架构
        let architecture: ModelArchitecture = bincode::deserialize(&arch_data)?;
        
        Ok(Some(architecture))
    }

    /// 列出所有模型
    fn list_models(&self) -> Result<Vec<StoredModel>> {
        let cf_models = self.get_cf_handle(CF_MODELS)?;
        let db = self.db.read().map_err(|e| Error::lock(e.to_string()))?;
        
        let mut models = Vec::new();
        let iter = db.iterator_cf(&cf_models, rocksdb::IteratorMode::Start);
        
        for item in iter {
            let (_, value) = item?;
            let model: StoredModel = bincode::deserialize(&value)?;
            models.push(model);
        }
        
        Ok(models)
    }

    /// 根据标签查找模型
    fn find_models_by_tag(&self, tag: &str) -> Result<Vec<StoredModel>> {
        let models = self.list_models()?;
        let filtered_models = models.into_iter()
            .filter(|model| model.tags.contains(&tag.to_string()))
            .collect();
        
        Ok(filtered_models)
    }

    /// 根据创建者查找模型
    fn find_models_by_creator(&self, creator_id: &str) -> Result<Vec<StoredModel>> {
        let models = self.list_models()?;
        let filtered_models = models.into_iter()
            .filter(|model| model.created_by == creator_id)
            .collect();
        
        Ok(filtered_models)
    }

    /// 获取模型信息
    fn get_model_info(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        let key = Self::make_info_key(model_id);
        let cf_info = self.get_cf_handle(CF_MODEL_INFO)?;
        
        // 获取信息数据
        let db = self.db.read().map_err(|e| Error::lock(e.to_string()))?;
        let info_data = match db.get_cf(&cf_info, key)? {
            Some(data) => data,
            None => return Ok(None),
        };
        
        // 反序列化信息
        let info: ModelInfo = bincode::deserialize(&info_data)?;
        
        Ok(Some(info))
    }

    /// 保存模型信息
    fn save_model_info(&self, info: &ModelInfo) -> Result<()> {
        // 序列化信息
        let info_data = bincode::serialize(info)?;
        let key = Self::make_info_key(&info.id);
        
        // 保存信息
        let cf_info = self.get_cf_handle(CF_MODEL_INFO)?;
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put_cf(&cf_info, key, info_data)?;
        
        Ok(())
    }

    /// 获取模型指标
    fn get_model_metrics(&self, model_id: &str) -> Result<Option<ModelMetrics>> {
        let key = format!("model:{}:metrics", model_id);
        let db = self.db.read().map_err(|e| Error::storage(format!("获取数据库读锁失败: {}", e)))?;
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取模型指标失败: {}", e)))? {
            
            let metrics = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化模型指标失败: {}", e)))?;
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }

    /// 更新模型指标
    fn update_model_metrics(&self, model_id: &str, metrics: &ModelMetrics) -> Result<()> {
        let key = format!("model:{}:metrics", model_id);
        let value = bincode::serialize(metrics)
            .map_err(|e| Error::serialization(format!("序列化模型指标失败: {}", e)))?;
            
        let db = self.db.read().map_err(|e| Error::storage(format!("获取数据库读锁失败: {}", e)))?;
        db.put(key.as_bytes(), &value)
            .map_err(|e| Error::storage(format!("保存模型指标失败: {}", e)))?;
            
        debug!("已保存模型 {} 的指标", model_id);
        Ok(())
    }

    /// 保存模型指标
    fn save_model_metrics(&self, model_id: &str, metrics: &ModelMetrics) -> Result<()> {
        self.update_model_metrics(model_id, metrics)
    }

    /// 导出模型
    fn export_model(&self, model_id: &str, format: StorageFormat, path: &Path) -> Result<()> {
        // 获取模型、参数和架构
        let model = self.get_model(model_id)?
            .ok_or_else(|| Error::not_found(format!("模型不存在: {}", model_id)))?;
        
        // 导出模型到指定路径
        match format {
            StorageFormat::Native => {
                // 创建目录
                if !path.exists() {
                    std::fs::create_dir_all(path)?;
                }
                
                // 导出模型元数据
                let model_path = path.join(format!("{}.json", model_id));
                let model_json = serde_json::to_string_pretty(&model)?;
                std::fs::write(model_path, model_json)?;
                
                // 导出参数（从 smart_parameters 导出）
                {
                    // 获取版本信息（通过元数据或默认值）
                    let version = model.smart_parameters.metadata.get("version")
                        .cloned()
                        .unwrap_or_else(|| "1.0.0".to_string());
                    
                    // 转换参数
                    let params_map: HashMap<String, TensorData> = model.smart_parameters.get_parameters().iter()
                        .filter_map(|(k, v)| convert_managed_to_tensor_data(v).ok().map(|t| (k.clone(), t)))
                        .collect();
                    
                    let params = crate::model::parameters::ModelParameters {
                        id: uuid::Uuid::new_v4().to_string(),
                        model_id: model.id.clone(),
                        version,
                        created_at: model.created_at.timestamp() as u64,
                        updated_at: model.updated_at.timestamp() as u64,
                        params: params_map,
                        gradients: model.smart_parameters.get_gradients().map(|g| {
                            g.iter()
                                .filter_map(|(k, v)| convert_managed_to_tensor_data(v).ok().map(|t| (k.clone(), t)))
                                .collect()
                        }),
                        optimizer_state: model.smart_parameters.get_optimizer_state().map(|o| {
                            o.iter()
                                .filter_map(|(k, v)| convert_managed_to_tensor_data(v).ok().map(|t| (k.clone(), t)))
                                .collect()
                        }),
                        metadata: model.smart_parameters.metadata.clone(),
                    };
                    let params_path = path.join(format!("{}_params.bin", model_id));
                    let params_data = bincode::serialize(&params)?;
                    std::fs::write(params_path, params_data)?;
                }
                
                // 导出架构
                {
                    let arch_path = path.join(format!("{}_arch.json", model_id));
                    let arch_json = serde_json::to_string_pretty(&model.architecture)?;
                    std::fs::write(arch_path, arch_json)?;
                }
            },
            _ => return Err(Error::not_implemented(format!("不支持的导出格式: {:?}", format))),
        }
        
        Ok(())
    }

    /// 导入模型
    fn import_model(&self, path: &Path, format: StorageFormat) -> Result<String> {
        match format {
            StorageFormat::Native => {
                // 检查路径
                if !path.exists() {
                    return Err(Error::not_found(format!("导入路径不存在: {:?}", path)));
                }
                
                // 检查是文件还是目录
                if path.is_file() {
                    // 单文件导入（元数据JSON）
                    let model_data = std::fs::read_to_string(path)?;
                    let model: Model = serde_json::from_str(&model_data)?;
                    
                    // 保存模型
                    self.save_model(&model)?;
                    
                    return Ok(model.id);
                } else {
                    // 目录导入，查找元数据文件
                    let entries = std::fs::read_dir(path)?;
                    let mut model_path = None;
                    
                    for entry in entries {
                        let entry = entry?;
                        let file_path = entry.path();
                        
                        if file_path.is_file() && file_path.extension().map_or(false, |ext| ext == "json") {
                            let file_name = file_path.file_name().unwrap().to_string_lossy();
                            if !file_name.ends_with("_arch.json") && !file_name.ends_with("_params.json") {
                                model_path = Some(file_path);
                                break;
                            }
                        }
                    }
                    
                    if let Some(model_path) = model_path {
                        // 读取模型元数据
                        let model_data = std::fs::read_to_string(model_path)?;
                        let mut model: Model = serde_json::from_str(&model_data)?;
                        
                        // 读取参数和架构
                        let params_path = path.join(format!("{}_params.bin", model.id));
                        if params_path.exists() {
                        let params_data = std::fs::read(params_path)?;
                        let params: ModelParameters = bincode::deserialize(&params_data)?;
                        // 使用 from_parameters 方法将持久化参数恢复到 smart_parameters
                        model.smart_parameters = crate::model::memory_management::SmartModelParameters::from_parameters(params);
                        }
                        
                        let arch_path = path.join(format!("{}_arch.json", model.id));
                        if arch_path.exists() {
                            let arch_data = std::fs::read_to_string(arch_path)?;
                            let arch: ModelArchitecture = serde_json::from_str(&arch_data)?;
                            // Model.architecture 的类型是 ModelArchitecture（非 Option）
                            model.architecture = arch;
                        }
                        
                        // 保存模型
                        self.save_model(&model)?;
                        
                        return Ok(model.id);
                    } else {
                        return Err(Error::not_found("在导入目录中找不到模型元数据文件"));
                    }
                }
            },
            _ => return Err(Error::not_implemented(format!("不支持的导入格式: {:?}", format))),
        }
    }
}

impl Default for PersistenceOptions {
    fn default() -> Self {
        Self {
            sync_writes: false,
            write_mode: WriteMode::Batch,
            crash_recovery: true,
            checkpoint_interval_seconds: Some(300), // 5分钟
        }
    }
} 