// 存储相关类型定义

use std::time::SystemTime;
use serde::{Serialize, Deserialize};
use super::model::SerializableModel;

/// 存储格式枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageFormat {
    Json,
    Binary,
    Protobuf,
}

/// 模型导出数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExportData {
    pub model: SerializableModel,
    pub parameters: Option<crate::model::ModelParameters>,
    pub architecture: Option<crate::model::ModelArchitecture>,
    pub format: StorageFormat,
    pub exported_at: SystemTime,
}

/// 模型备份数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBackupData {
    pub model: SerializableModel,
    pub parameters: Option<crate::model::ModelParameters>,
    pub architecture: Option<crate::model::ModelArchitecture>,
    pub backup_type: BackupType,
    pub created_at: SystemTime,
}

// 类型别名
pub type BackupType = crate::model::manager::enums::BackupType; 