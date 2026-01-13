use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, TimeZone};
use crate::data::DataFormat;
use crate::error::Result;
use crate::compat::{Model, ModelArchitecture, ModelParameters, TensorData, SmartModelParameters, ModelStatus, ModelMemoryMonitor};
use std::collections::HashMap;
use std::sync::Arc;
use std::path::Path;
use std::sync::Mutex;
use crate::data::schema::DataSchema as Schema;
// removed unused data/storage imports to align with current implementation scope

/// 数据信息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInfo {
    pub id: String,
    pub name: String,
    pub format: DataFormat,
    pub size: u64,
    pub created_at: DateTime<Utc>,
    /// 数据源信息
    pub source: Option<String>,
    /// 数据标签
    pub tags: Vec<String>,
    /// 数据schema
    pub schema: Option<Schema>,
    /// 行数
    pub row_count: Option<u64>,
    /// 列数
    pub column_count: Option<u32>,
    /// 最后更新时间
    pub updated_at: Option<DateTime<Utc>>,
    /// 自定义元数据
    pub metadata: HashMap<String, String>,
}

/// 数据项信息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemInfo {
    /// 条目ID
    pub id: String,
    /// 条目类型
    pub item_type: String,
    /// 条目名称
    pub name: String,
    /// 条目大小（字节）
    pub size: u64,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 修改时间
    pub modified_at: DateTime<Utc>,
    /// 访问时间
    pub accessed_at: DateTime<Utc>,
    /// 引用路径
    pub path: String,
    /// 条目状态
    pub status: ItemStatus,
    /// 关联对象
    pub associations: Vec<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

/// 条目状态枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ItemStatus {
    /// 可用
    Available,
    /// 处理中
    Processing,
    /// 已归档
    Archived,
    /// 已删除
    Deleted,
    /// 已锁定
    Locked,
    /// 已损坏
    Corrupted,
}

/// 元数据信息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataInfo {
    /// 元数据键
    pub key: String,
    /// 元数据值
    pub value: String,
    /// 元数据命名空间
    pub namespace: String,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: Option<DateTime<Utc>>,
    /// 数据类型
    pub data_type: MetadataType,
    /// 描述
    pub description: Option<String>,
    /// 是否系统元数据
    pub is_system: bool,
    /// 是否可见
    pub is_visible: bool,
    /// 权限
    pub permissions: MetadataPermissions,
    /// 关联ID
    pub associated_id: Option<String>,
}

/// 元数据类型枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataType {
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Json,
    Binary,
}

/// 元数据权限结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataPermissions {
    /// 是否可读
    pub readable: bool,
    /// 是否可写
    pub writable: bool,
    /// 是否可删除
    pub deletable: bool,
}

/// 模型信息结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// 模型ID
    pub id: String,
    /// 模型名称
    pub name: String,
    /// 模型版本
    pub version: String,
    /// 模型描述
    pub description: Option<String>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 模型类型
    pub model_type: String,
    /// 框架信息
    pub framework: String,
    /// 框架版本
    pub framework_version: String,
    /// 模型大小（字节）
    pub size: u64,
    /// 输入格式
    pub input_format: String,
    /// 输出格式
    pub output_format: String,
    /// 预处理步骤
    pub preprocessing: Vec<String>,
    /// 后处理步骤
    pub postprocessing: Vec<String>,
    /// 标签
    pub tags: Vec<String>,
    /// 许可证
    pub license: Option<String>,
    /// 作者
    pub author: Option<String>,
    /// 指标
    pub metrics: Option<ModelMetrics>,
    /// 依赖项
    pub dependencies: Vec<ModelDependency>,
    /// 存储格式
    pub storage_format: StorageFormat,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 校验和（用于数据完整性校验）
    pub checksum: Option<String>,
}

/// 模型依赖项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDependency {
    /// 依赖名称
    pub name: String,
    /// 依赖版本
    pub version: String,
    /// 是否必需
    pub required: bool,
}

/// 模型指标结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// 准确率
    pub accuracy: Option<f64>,
    /// 精确率
    pub precision: Option<f64>,
    /// 召回率
    pub recall: Option<f64>,
    /// F1分数
    pub f1_score: Option<f64>,
    /// 损失值
    pub loss: Option<f64>,
    /// 验证损失值
    pub val_loss: Option<f64>,
    /// 平均绝对误差
    pub mae: Option<f64>,
    /// 均方误差
    pub mse: Option<f64>,
    /// 训练时间（秒）
    pub training_time: Option<f64>,
    /// 推理延迟（毫秒）
    pub inference_latency: Option<f64>,
    /// 自定义指标
    pub custom_metrics: HashMap<String, f64>,
    /// 混淆矩阵
    pub confusion_matrix: Option<Vec<Vec<i64>>>,
    /// ROC曲线数据
    pub roc_curve: Option<Vec<(f64, f64)>>,
    /// 精确率-召回率曲线数据
    pub pr_curve: Option<Vec<(f64, f64)>>,
    /// 最后评估时间
    pub last_evaluation: Option<DateTime<Utc>>,
    /// 评估数据集ID
    pub evaluation_dataset_id: Option<String>,
    /// 训练历史记录
    pub training_history: Option<HashMap<String, Vec<f64>>>,
    
    // 新增字段 - 训练相关指标
    /// 其他指标（训练指标）
    pub metrics: Option<HashMap<String, f64>>,
    /// 验证指标
    pub val_metrics: Option<HashMap<String, f64>>,
    /// 当前训练轮数
    pub epoch: Option<u64>,
    /// 总训练轮数
    pub total_epochs: Option<u64>,
    /// 学习率
    pub learning_rate: Option<f64>,
    /// 训练经过时间
    pub time_elapsed: Option<f64>,
    /// 当前批次
    pub batch: Option<u64>,
    /// 总批次数
    pub total_batches: Option<u64>,
    /// 时间戳
    pub timestamp: Option<u64>,
    /// 前向传播时间（毫秒）
    pub forward_time: Option<u64>,
    /// 反向传播时间（毫秒）
    pub backward_time: Option<u64>,
    /// 参数更新时间（毫秒）
    pub update_time: Option<u64>,
    /// 当前训练步骤
    pub step: Option<u64>,
    /// 损失历史记录
    pub loss_history: Option<Vec<f64>>,
    /// 准确率历史记录
    pub accuracy_history: Option<Vec<f64>>,
}

/// 存储格式枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageFormat {
    /// 原生格式
    Native,
    /// ONNX格式
    ONNX,
    /// TensorFlow SavedModel
    TensorFlowSavedModel,
    /// PyTorch
    PyTorch,
    /// TensorRT
    TensorRT,
    /// 自定义格式
    Custom(String),
}

/// 存储选项结构
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageOptions {
    /// 存储路径
    pub path: String,
    /// 缓存大小（MB）
    pub cache_size_mb: Option<u32>,
    /// 压缩级别 (0-9)
    pub compression_level: Option<u8>,
    /// 压缩类型
    pub compression_type: Option<CompressionType>,
    /// 加密类型
    pub encryption_type: Option<EncryptionType>,
    /// 加密密钥
    pub encryption_key: Option<String>,
    /// 分片大小（KB）
    pub shard_size_kb: Option<u32>,
    /// 备份频率（小时）
    pub backup_frequency_hours: Option<u32>,
    /// 最大备份数量
    pub max_backups: Option<u8>,
    /// 创建目录（如果不存在）
    pub create_if_missing: bool,
    /// 启用WAL
    pub use_wal: bool,
    /// 启用多线程
    pub enable_multithreading: bool,
    /// 最大线程数
    pub max_threads: Option<u8>,
    /// 预分配空间（MB）
    pub preallocation_size_mb: Option<u32>,
    /// 读缓存大小（KB）
    pub read_cache_size_kb: Option<u32>,
    /// 写缓存大小（KB）
    pub write_cache_size_kb: Option<u32>,
    /// 持久化选项
    pub persistence_options: PersistenceOptions,
}

/// 压缩类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionType {
    /// 无压缩
    None,
    /// Gzip压缩
    Gzip,
    /// LZ4压缩
    LZ4,
    /// Zstd压缩
    Zstd,
    /// Snappy压缩
    Snappy,
}

/// 加密类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionType {
    /// 无加密
    None,
    /// AES-256
    AES256,
    /// ChaCha20
    ChaCha20,
}

/// 持久化选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceOptions {
    /// 启用同步写入
    pub sync_writes: bool,
    /// 写入模式
    pub write_mode: WriteMode,
    /// 崩溃恢复
    pub crash_recovery: bool,
    /// 检查点间隔（秒）
    pub checkpoint_interval_seconds: Option<u32>,
}

/// 写入模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WriteMode {
    /// 顺序写入
    Sequential,
    /// 随机写入
    Random,
    /// 批量写入
    Batch,
}

/// 存储模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredModel {
    /// 模型ID
    pub id: String,
    /// 模型名称
    pub name: String,
    /// 模型描述
    pub description: Option<String>,
    /// 创建者ID
    pub created_by: String,
    /// 创建时间
    pub created_at: u64,
    /// 更新时间
    pub updated_at: u64,
    /// 模型标签
    pub tags: Vec<String>,
    /// 模型元数据
    pub metadata: HashMap<String, String>,
}

/// 存储模型参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredModelParams {
    /// 模型ID
    pub model_id: String,
    /// 参数版本
    pub version: String,
    /// 创建时间
    pub created_at: u64,
    /// 参数格式
    pub format: String,
    /// 参数数据
    pub data: TensorData,
}

/// 存储模型架构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredModelArchitecture {
    /// 模型ID
    pub model_id: String,
    /// 架构版本
    pub version: String,
    /// 创建时间
    pub created_at: u64,
    /// 架构类型
    pub architecture_type: String,
    /// 架构配置
    pub config: String,
}

/// 模型存储接口
pub trait ModelStorage: Send + Sync {
    /// 保存模型
    fn save_model(&self, model: &Model) -> Result<()>;
    
    /// 获取模型
    fn get_model(&self, model_id: &str) -> Result<Option<Model>>;
    
    /// 删除模型
    fn delete_model(&self, model_id: &str) -> Result<bool>;
    
    /// 保存模型参数
    fn save_model_params(&self, model_id: &str, params: &ModelParameters) -> Result<()>;
    
    /// 获取模型参数
    fn get_model_params(&self, model_id: &str) -> Result<Option<ModelParameters>>;
    
    /// 保存模型架构
    fn save_model_architecture(&self, model_id: &str, architecture: &ModelArchitecture) -> Result<()>;
    
    /// 获取模型架构
    fn get_model_architecture(&self, model_id: &str) -> Result<Option<ModelArchitecture>>;
    
    /// 列出所有模型
    fn list_models(&self) -> Result<Vec<StoredModel>>;
    
    /// 根据标签查找模型
    fn find_models_by_tag(&self, tag: &str) -> Result<Vec<StoredModel>>;
    
    /// 根据创建者查找模型
    fn find_models_by_creator(&self, creator_id: &str) -> Result<Vec<StoredModel>>;

    /// 获取模型信息
    fn get_model_info(&self, model_id: &str) -> Result<Option<ModelInfo>>;
    
    /// 保存模型信息
    fn save_model_info(&self, info: &ModelInfo) -> Result<()>;
    
    /// 获取模型指标
    fn get_model_metrics(&self, model_id: &str) -> Result<Option<ModelMetrics>>;
    
    /// 更新模型指标
    fn update_model_metrics(&self, model_id: &str, metrics: &ModelMetrics) -> Result<()>;
    
    /// 保存模型指标
    fn save_model_metrics(&self, model_id: &str, metrics: &ModelMetrics) -> Result<()>;
    
    /// 导出模型
    fn export_model(&self, model_id: &str, format: StorageFormat, path: &Path) -> Result<()>;
    
    /// 导入模型
    fn import_model(&self, path: &Path, format: StorageFormat) -> Result<String>;
}

/// 将模型转换为存储模型
pub fn model_to_stored_model(model: &Model) -> StoredModel {
    // 优先使用 Model 顶层字段，其余超出字段统一落到 metadata（必要时使用前缀区分）
    let mut metadata = model.metadata.clone();

    // 基本标识信息
    metadata.insert("model_type".to_string(), model.model_type.clone());
    metadata.insert("version".to_string(), model.version.clone());
    if let Some(pid) = &model.parent_id { metadata.insert("parent_id".to_string(), pid.clone()); }

    // 形状信息
    if !model.architecture.input_shape.is_empty() {
        if let Ok(s) = serde_json::to_string(&model.architecture.input_shape) { metadata.insert("input_shape".to_string(), s); }
    }
    if !model.architecture.output_shape.is_empty() {
        if let Ok(s) = serde_json::to_string(&model.architecture.output_shape) { metadata.insert("output_shape".to_string(), s); }
    }

    // 导入源（若存在）
    if let Some(src) = &model.import_source {
        metadata.insert("import_type".to_string(), format!("{:?}", src.import_type));
        metadata.insert("import_path".to_string(), src.source_path.clone());
        metadata.insert("imported_at".to_string(), src.imported_at.timestamp().to_string());
        metadata.insert("original_format".to_string(), src.original_format.clone());
    }

    // 合并架构 metadata（加前缀避免冲突）
    for (k, v) in &model.architecture.metadata {
        metadata.insert(format!("arch.{}", k), v.clone());
    }

    // 从元数据或架构元数据提取 tags
    let tags = metadata.get("tags")
        .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
        .unwrap_or_else(|| model.architecture.metadata.get("tags")
            .map(|s| s.split(',').map(|t| t.trim().to_string()).collect())
            .unwrap_or_default());

    // 创建者
    let created_by = metadata.get("created_by").cloned().unwrap_or_else(|| "System".to_string());

    StoredModel {
        id: model.id.clone(),
        name: model.name.clone(),
        description: model.description.clone(),
        created_by,
        created_at: model.created_at.timestamp() as u64,
        updated_at: model.updated_at.timestamp() as u64,
        tags,
        metadata,
    }
}

/// 将存储模型转换为模型
pub fn stored_model_to_model(
    stored_model: &StoredModel,
    architecture: Option<ModelArchitecture>,
    parameters: Option<ModelParameters>,
) -> Model {
    // 基于存储数据和入参组装架构
    let mut arch = architecture.unwrap_or_else(|| ModelArchitecture::default());
    // 将存储的以 arch. 前缀的键回灌到架构元数据
    for (k, v) in &stored_model.metadata {
        if let Some(stripped) = k.strip_prefix("arch.") {
            arch.metadata.insert(stripped.to_string(), v.clone());
        }
    }
    // 保留基本描述信息在架构元数据中，便于追踪
    arch.metadata.insert("name".to_string(), stored_model.name.clone());
    if let Some(desc) = &stored_model.description { arch.metadata.insert("description".to_string(), desc.clone()); }
    arch.metadata.insert("created_by".to_string(), stored_model.created_by.clone());
    arch.metadata.insert("tags".to_string(), stored_model.tags.join(","));

    // 还原基础字段
    let model_type = stored_model.metadata.get("model_type").cloned().unwrap_or_else(|| "generic".to_string());
    let version = stored_model.metadata.get("version").cloned().unwrap_or_else(|| "1.0.0".to_string());
    let parent_id = stored_model.metadata.get("parent_id").cloned();

    // 形状：优先从 metadata 解析，否则回退到架构
    let input_shape = stored_model.metadata.get("input_shape")
        .and_then(|s| serde_json::from_str::<Vec<usize>>(s).ok())
        .unwrap_or_else(|| arch.input_shape.clone());
    let output_shape = stored_model.metadata.get("output_shape")
        .and_then(|s| serde_json::from_str::<Vec<usize>>(s).ok())
        .unwrap_or_else(|| arch.output_shape.clone());

    // 过滤出非 arch.* 的元数据作为模型元数据
    let mut model_metadata: HashMap<String, String> = stored_model.metadata.iter()
        .filter_map(|(k,v)| if !k.starts_with("arch.") { Some((k.clone(), v.clone())) } else { None })
        .collect();

    // 构造 SmartModelParameters（若给入参数则保留版本到元数据，不尝试跨类型深度转换）
    let mut smart_parameters = SmartModelParameters::default();
    if let Some(params) = parameters {
        smart_parameters.metadata.insert("version".to_string(), params.version);
    }

    // 时间
    let created_at = Utc.timestamp_opt(stored_model.created_at as i64, 0).single().unwrap_or_else(Utc::now);
    let updated_at = Utc.timestamp_opt(stored_model.updated_at as i64, 0).single().unwrap_or_else(Utc::now);

    Model {
        id: stored_model.id.clone(),
        name: stored_model.name.clone(),
        description: stored_model.description.clone(),
        version,
        model_type,
        smart_parameters,
        architecture: arch,
        status: ModelStatus::Ready,
        metrics: None,
        created_at,
        updated_at,
        parent_id,
        metadata: model_metadata,
        input_shape,
        output_shape,
        import_source: None,
        memory_monitor: Arc::new(Mutex::new(ModelMemoryMonitor::default())),
    }
}

/// 将模型信息转换为JSON
pub fn model_info_to_json(info: &ModelInfo) -> Result<serde_json::Value> {
    Ok(serde_json::to_value(info)?)
}

/// 从JSON转换为模型信息
pub fn json_to_model_info(json: &serde_json::Value) -> Result<ModelInfo> {
    Ok(serde_json::from_value(json.clone())?)
}

/// 创建默认存储选项
pub fn default_storage_options() -> StorageOptions {
    StorageOptions {
        path: "./data".to_string(),
        cache_size_mb: Some(512),
        compression_level: Some(6),
        compression_type: Some(CompressionType::LZ4),
        encryption_type: None,
        encryption_key: None,
        shard_size_kb: Some(64 * 1024), // 64MB
        backup_frequency_hours: Some(24),
        max_backups: Some(7),
        create_if_missing: true,
        use_wal: true,
        enable_multithreading: true,
        max_threads: Some(4),
        preallocation_size_mb: Some(100),
        read_cache_size_kb: Some(256 * 1024), // 256MB
        write_cache_size_kb: Some(64 * 1024), // 64MB
        persistence_options: PersistenceOptions {
            sync_writes: false,
            write_mode: WriteMode::Batch,
            crash_recovery: true,
            checkpoint_interval_seconds: Some(300), // 5分钟
        },
    }
}
