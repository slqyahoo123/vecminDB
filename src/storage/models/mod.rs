// 导入implementation模块
pub mod implementation;
// 添加工厂模块
pub mod factory;
// 导出存储实现模块
pub mod file_storage;
pub mod rocksdb_storage;

// 重新导出模型组件
pub use implementation::{
    DataInfo,
    StoredModel,
    StoredModelParams,
    StoredModelArchitecture,
    ModelStorage,
    // 新增导出类型
    ItemInfo,
    ItemStatus,
    MetadataInfo,
    MetadataType,
    MetadataPermissions,
    ModelInfo,
    ModelMetrics,
    ModelDependency,
    StorageFormat,
    StorageOptions,
    CompressionType,
    EncryptionType,
    PersistenceOptions,
    WriteMode,
    // 工具函数
    model_to_stored_model,
    stored_model_to_model,
    model_info_to_json,
    json_to_model_info,
    default_storage_options,
};

// 导出工厂组件
pub use factory::{ModelStorageFactory, ModelStorageUtil, ModelStorageType};
