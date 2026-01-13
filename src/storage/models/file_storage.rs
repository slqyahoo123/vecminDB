use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
// serde derives are not used directly in this module
use chrono::{Utc};

use crate::{Result, Error};
use crate::compat::{Model, ModelArchitecture, ModelParameters};
use crate::storage::models::implementation::{
    ModelStorage, StoredModel,
    ModelInfo, ModelMetrics, StorageFormat, StorageOptions
};

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

/// 文件模型存储实现
pub struct FileModelStorage {
    /// 基础路径
    base_path: PathBuf,
    /// 存储选项
    options: StorageOptions,
    /// 模型缓存
    model_cache: Arc<Mutex<HashMap<String, StoredModel>>>,
}

impl FileModelStorage {
    /// 创建新的文件模型存储
    pub fn new(options: StorageOptions) -> Result<Self> {
        // 创建存储目录
        let base_path = PathBuf::from(&options.path);
        if !base_path.exists() && options.create_if_missing {
            fs::create_dir_all(&base_path)?;
        }

        // 创建子目录
        let models_dir = base_path.join("models");
        let params_dir = base_path.join("parameters");
        let arch_dir = base_path.join("architectures");
        let info_dir = base_path.join("info");
        let metrics_dir = base_path.join("metrics");

        // 创建所有必要的目录
        if !models_dir.exists() {
            fs::create_dir_all(&models_dir)?;
        }
        if !params_dir.exists() {
            fs::create_dir_all(&params_dir)?;
        }
        if !arch_dir.exists() {
            fs::create_dir_all(&arch_dir)?;
        }
        if !info_dir.exists() {
            fs::create_dir_all(&info_dir)?;
        }
        if !metrics_dir.exists() {
            fs::create_dir_all(&metrics_dir)?;
        }

        // 加载现有模型到缓存
        let mut model_cache = HashMap::new();
        if models_dir.exists() {
            for entry in fs::read_dir(&models_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    let file_name = entry.file_name();
                    let file_name = file_name.to_string_lossy();
                    if file_name.ends_with(".bin") {
                        let model_id = file_name.trim_end_matches(".bin");
                        if let Ok(model) = Self::load_model_from_file(&models_dir.join(file_name.as_ref())) {
                            model_cache.insert(model_id.to_string(), model);
                        }
                    }
                }
            }
        }

        Ok(Self {
            base_path,
            options,
            model_cache: Arc::new(Mutex::new(model_cache)),
        })
    }

    /// 从文件加载模型
    fn load_model_from_file(path: &Path) -> Result<StoredModel> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        let model = bincode::deserialize(&data)?;
        Ok(model)
    }

    /// 保存模型到文件
    fn save_model_to_file(&self, model: &StoredModel) -> Result<()> {
        let models_dir = self.base_path.join("models");
        let file_path = models_dir.join(format!("{}.bin", model.id));
        let data = bincode::serialize(model)?;
        let mut file = File::create(file_path)?;
        file.write_all(&data)?;
        
        // 更新缓存
        let mut cache = self.model_cache.lock().map_err(|e| Error::lock(e.to_string()))?;
        cache.insert(model.id.clone(), model.clone());
        
        Ok(())
    }

    /// 获取参数目录
    fn params_dir(&self) -> PathBuf {
        self.base_path.join("parameters")
    }

    /// 获取架构目录
    fn arch_dir(&self) -> PathBuf {
        self.base_path.join("architectures")
    }

    /// 获取信息目录
    fn info_dir(&self) -> PathBuf {
        self.base_path.join("info")
    }

    /// 获取指标目录
    fn metrics_dir(&self) -> PathBuf {
        self.base_path.join("metrics")
    }
}

impl ModelStorage for FileModelStorage {
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

        // 保存模型到文件
        self.save_model_to_file(&stored_model)?;

        // 保存参数和架构（从新结构导出）
        {
            // 获取参数数据并转换为 TensorData
            let parameters = model.smart_parameters.get_parameters();
            let mut params_map = std::collections::HashMap::new();
            for (k, v) in parameters.iter() {
                // 将 ManagedTensorData 转换为 TensorData
                // 使用 convert_managed_to_tensor_data 辅助函数
                if let Ok(tensor_data) = convert_managed_to_tensor_data(v) {
                    params_map.insert(k.clone(), tensor_data);
                }
            }
            
            // 获取版本信息（通过元数据或默认值）
            let version = model.smart_parameters.metadata.get("version")
                .cloned()
                .unwrap_or_else(|| "1.0.0".to_string());
            
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
            self.save_model_params(&model.id, &params)?;
        }

        {
            self.save_model_architecture(&model.id, &model.architecture)?;
        }

        Ok(())
    }

    /// 获取模型
    fn get_model(&self, model_id: &str) -> Result<Option<Model>> {
        // 检查缓存
        let cache = self.model_cache.lock().map_err(|e| Error::lock(e.to_string()))?;
        let stored_model = match cache.get(model_id) {
            Some(model) => model.clone(),
            None => {
                // 检查文件
                let models_dir = self.base_path.join("models");
                let file_path = models_dir.join(format!("{}.bin", model_id));
                if !file_path.exists() {
                    return Ok(None);
                }
                
                match Self::load_model_from_file(&file_path) {
                    Ok(model) => model,
                    Err(_) => return Ok(None),
                }
            }
        };
        
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
                Some(p) => {
                    // 使用 from_parameters 方法转换
                    crate::model::memory_management::SmartModelParameters::from_parameters(p)
                },
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
        };

        Ok(Some(model))
    }

    /// 删除模型
    fn delete_model(&self, model_id: &str) -> Result<bool> {
        // 检查模型是否存在
        let models_dir = self.base_path.join("models");
        let model_path = models_dir.join(format!("{}.bin", model_id));
        
        if !model_path.exists() {
            return Ok(false);
        }
        
        // 删除模型文件
        fs::remove_file(model_path)?;
        
        // 删除参数文件
        let params_dir = self.params_dir();
        let params_path = params_dir.join(format!("{}.bin", model_id));
        if params_path.exists() {
            fs::remove_file(params_path)?;
        }
        
        // 删除架构文件
        let arch_dir = self.arch_dir();
        let arch_path = arch_dir.join(format!("{}.bin", model_id));
        if arch_path.exists() {
            fs::remove_file(arch_path)?;
        }
        
        // 删除信息文件
        let info_dir = self.info_dir();
        let info_path = info_dir.join(format!("{}.bin", model_id));
        if info_path.exists() {
            fs::remove_file(info_path)?;
        }
        
        // 删除指标文件
        let metrics_dir = self.metrics_dir();
        let metrics_path = metrics_dir.join(format!("{}.bin", model_id));
        if metrics_path.exists() {
            fs::remove_file(metrics_path)?;
        }
        
        // 更新缓存
        let mut cache = self.model_cache.lock().map_err(|e| Error::lock(e.to_string()))?;
        cache.remove(model_id);
        
        Ok(true)
    }

    /// 保存模型参数
    fn save_model_params(&self, model_id: &str, params: &ModelParameters) -> Result<()> {
        let params_dir = self.params_dir();
        let file_path = params_dir.join(format!("{}.bin", model_id));
        
        // 序列化参数
        let data = bincode::serialize(params)?;
        
        // 写入文件
        let mut file = File::create(file_path)?;
        file.write_all(&data)?;
        
        Ok(())
    }

    /// 获取模型参数
    fn get_model_params(&self, model_id: &str) -> Result<Option<ModelParameters>> {
        let params_dir = self.params_dir();
        let file_path = params_dir.join(format!("{}.bin", model_id));
        
        if !file_path.exists() {
            return Ok(None);
        }
        
        // 读取文件
        let mut file = File::open(file_path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        
        // 反序列化参数
        let params: ModelParameters = bincode::deserialize(&data)?;
        
        Ok(Some(params))
    }

    /// 保存模型架构
    fn save_model_architecture(&self, model_id: &str, architecture: &ModelArchitecture) -> Result<()> {
        let arch_dir = self.arch_dir();
        let file_path = arch_dir.join(format!("{}.bin", model_id));
        
        // 序列化架构
        let data = bincode::serialize(architecture)?;
        
        // 写入文件
        let mut file = File::create(file_path)?;
        file.write_all(&data)?;
        
        Ok(())
    }

    /// 获取模型架构
    fn get_model_architecture(&self, model_id: &str) -> Result<Option<ModelArchitecture>> {
        let arch_dir = self.arch_dir();
        let file_path = arch_dir.join(format!("{}.bin", model_id));
        
        if !file_path.exists() {
            return Ok(None);
        }
        
        // 读取文件
        let mut file = File::open(file_path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        
        // 反序列化架构
        let architecture: ModelArchitecture = bincode::deserialize(&data)?;
        
        Ok(Some(architecture))
    }

    /// 列出所有模型
    fn list_models(&self) -> Result<Vec<StoredModel>> {
        let cache = self.model_cache.lock().map_err(|e| Error::lock(e.to_string()))?;
        let models: Vec<StoredModel> = cache.values().cloned().collect();
        
        // 如果缓存为空，从文件加载
        if models.is_empty() {
            let models_dir = self.base_path.join("models");
            
            if !models_dir.exists() {
                return Ok(Vec::new());
            }
            
            let mut models = Vec::new();
            for entry in fs::read_dir(models_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() {
                    let file_name = entry.file_name();
                    let file_name = file_name.to_string_lossy();
                    
                    if file_name.ends_with(".bin") {
                        if let Ok(model) = Self::load_model_from_file(&entry.path()) {
                            models.push(model);
                        }
                    }
                }
            }
            
            return Ok(models);
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
        let info_dir = self.info_dir();
        let file_path = info_dir.join(format!("{}.bin", model_id));
        
        if !file_path.exists() {
            return Ok(None);
        }
        
        // 读取文件
        let mut file = File::open(file_path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        
        // 反序列化信息
        let info: ModelInfo = bincode::deserialize(&data)?;
        
        Ok(Some(info))
    }

    /// 保存模型信息
    fn save_model_info(&self, info: &ModelInfo) -> Result<()> {
        let info_dir = self.info_dir();
        let file_path = info_dir.join(format!("{}.bin", info.id));
        
        // 序列化信息
        let data = bincode::serialize(info)?;
        
        // 写入文件
        let mut file = File::create(file_path)?;
        file.write_all(&data)?;
        
        Ok(())
    }

    /// 获取模型指标
    fn get_model_metrics(&self, model_id: &str) -> Result<Option<ModelMetrics>> {
        let metrics_dir = self.metrics_dir();
        let file_path = metrics_dir.join(format!("{}.bin", model_id));
        
        if !file_path.exists() {
            return Ok(None);
        }
        
        // 读取文件
        let mut file = File::open(file_path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        
        // 反序列化指标
        let metrics: ModelMetrics = bincode::deserialize(&data)?;
        
        Ok(Some(metrics))
    }

    /// 更新模型指标
    fn update_model_metrics(&self, model_id: &str, metrics: &ModelMetrics) -> Result<()> {
        let metrics_dir = self.metrics_dir();
        let file_path = metrics_dir.join(format!("{}.bin", model_id));
        
        // 序列化指标
        let data = bincode::serialize(metrics)?;
        
        // 写入文件
        let mut file = File::create(file_path)?;
        file.write_all(&data)?;
        
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
                    fs::create_dir_all(path)?;
                }
                
                // 导出模型元数据
                let model_path = path.join(format!("{}.json", model_id));
                let model_json = serde_json::to_string_pretty(&model)?;
                fs::write(model_path, model_json)?;
                
                // 导出参数（从 smart_parameters 导出）
                {
                    // 使用 from_parameters 方法转换
                    let params = crate::model::parameters::ModelParameters::new(
                        model.id.clone(),
                        model.smart_parameters.get_parameters().iter()
                            .filter_map(|(k, v)| convert_managed_to_tensor_data(v).ok().map(|t| (k.clone(), t)))
                            .collect()
                    );
                    let params_path = path.join(format!("{}_params.bin", model_id));
                    let params_data = bincode::serialize(&params)?;
                    fs::write(params_path, params_data)?;
                }
                
                // 导出架构
                {
                    let arch_path = path.join(format!("{}_arch.json", model_id));
                    let arch_json = serde_json::to_string_pretty(&model.architecture)?;
                    fs::write(arch_path, arch_json)?;
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
                
                if path.is_file() {
                    // 单文件导入（元数据JSON）
                    let model_data = fs::read_to_string(path)?;
                    let model: Model = serde_json::from_str(&model_data)?;
                    
                    // 保存模型
                    self.save_model(&model)?;
                    
                    return Ok(model.id);
                } else {
                    // 目录导入，查找元数据文件
                    let entries = fs::read_dir(path)?;
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
                        let model_data = fs::read_to_string(model_path)?;
                        let mut model: Model = serde_json::from_str(&model_data)?;
                        
                        // 读取参数和架构（转换为新结构）
                        let params_path = path.join(format!("{}_params.bin", model.id));
                        if params_path.exists() {
                            let params_data = fs::read(params_path)?;
                            let params: ModelParameters = bincode::deserialize(&params_data)?;
                            // 使用 from_parameters 方法转换
                            model.smart_parameters = crate::model::memory_management::SmartModelParameters::from_parameters(params);
                        }
                        
                        let arch_path = path.join(format!("{}_arch.json", model.id));
                        if arch_path.exists() {
                            let arch_data = fs::read_to_string(arch_path)?;
                            let arch: ModelArchitecture = serde_json::from_str(&arch_data)?;
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