// 导入implementation模块
mod implementation;

// 重新导出权限组件
pub use implementation::{
    Permission as ImplPermission,
    ResourceType as ImplResourceType,
    PermissionEntry as ImplPermissionEntry,
    PermissionManager as ImplPermissionManager,
};

// 权限管理模块
//
// 提供数据库对象的访问控制功能

use std::sync::RwLock;
use serde::{Serialize, Deserialize};
use crate::Result;
use crate::error::Error;
use log::debug;

/// 资源类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// 数据资源
    Data,
    
    /// 模型资源
    Model,
    
    /// 算法资源
    Algorithm,
    
    /// 系统资源
    System,
}

impl ResourceType {
    /// 将资源类型转换为字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceType::Data => "data",
            ResourceType::Model => "model",
            ResourceType::Algorithm => "algorithm",
            ResourceType::System => "system",
        }
    }
    
    /// 从字符串解析资源类型
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "data" => Some(ResourceType::Data),
            "model" => Some(ResourceType::Model),
            "algorithm" => Some(ResourceType::Algorithm),
            "system" => Some(ResourceType::System),
            _ => None,
        }
    }
}

/// 权限类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// 读取权限
    Read,
    
    /// 写入权限
    Write,
    
    /// 执行权限
    Execute,
    
    /// 管理权限
    Manage,
}

impl Permission {
    /// 将权限转换为字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            Permission::Read => "read",
            Permission::Write => "write",
            Permission::Execute => "execute",
            Permission::Manage => "manage",
        }
    }
    
    /// 从字符串解析权限
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "read" => Some(Permission::Read),
            "write" => Some(Permission::Write),
            "execute" => Some(Permission::Execute),
            "manage" => Some(Permission::Manage),
            _ => None,
        }
    }
}

/// 权限条目结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionEntry {
    /// 用户ID
    pub user_id: String,
    
    /// 资源类型
    pub resource_type: ResourceType,
    
    /// 资源ID
    pub resource_id: Option<String>,
    
    /// 权限
    pub permission: Permission,
    
    /// 创建时间
    #[serde(default = "get_current_timestamp")]
    pub created_at: u64,
    
    /// 过期时间
    pub expires_at: Option<u64>,
}

/// 获取当前时间戳
fn get_current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl PermissionEntry {
    /// 创建新的权限条目
    pub fn new(
        user_id: String,
        resource_type: ResourceType,
        resource_id: Option<String>,
        permission: Permission,
        expires_at: Option<u64>,
    ) -> Self {
        Self {
            user_id,
            resource_type,
            resource_id,
            permission,
            created_at: get_current_timestamp(),
            expires_at,
        }
    }
    
    /// 检查权限是否过期
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            expires_at <= get_current_timestamp()
        } else {
            false
        }
    }
}

/// 权限管理器
pub struct PermissionManager {
    /// 权限条目
    entries: RwLock<Vec<PermissionEntry>>,
}

impl PermissionManager {
    /// 创建新的权限管理器
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }
    
    /// 添加权限条目
    pub fn add_permission(&self, entry: PermissionEntry) -> Result<()> {
        let mut entries = self.entries.write().map_err(|e| Error::lock(e.to_string()))?;
        entries.push(entry);
        Ok(())
    }
    
    /// 移除权限条目
    pub fn remove_permission(
        &self,
        user_id: &str,
        resource_type: ResourceType,
        resource_id: Option<&str>,
        permission: Permission,
    ) -> Result<bool> {
        let mut entries = self.entries.write().map_err(|e| Error::lock(e.to_string()))?;
        
        let original_len = entries.len();
        
        entries.retain(|entry| {
            !(entry.user_id == user_id
                && entry.resource_type == resource_type
                && entry.resource_id.as_deref() == resource_id
                && entry.permission == permission)
        });
        
        Ok(entries.len() < original_len)
    }
    
    /// 检查用户是否具有指定资源的权限
    pub fn check_permission(
        &self,
        user_id: &str,
        resource_type: ResourceType,
        resource_id: Option<&str>,
        permission: Permission,
    ) -> Result<bool> {
        let entries = self.entries.read().map_err(|e| Error::lock(e.to_string()))?;
        
        // 清理过期权限
        self.cleanup_expired_permissions()?;
        
        // 检查是否有全局权限
        let has_global_permission = entries.iter().any(|entry| {
            !entry.is_expired()
                && entry.user_id == user_id
                && entry.resource_type == resource_type
                && entry.resource_id.is_none()
                && entry.permission == permission
        });
        
        if has_global_permission {
            return Ok(true);
        }
        
        // 检查是否有特定资源的权限
        let has_specific_permission = entries.iter().any(|entry| {
            !entry.is_expired()
                && entry.user_id == user_id
                && entry.resource_type == resource_type
                && entry.resource_id.as_deref() == resource_id
                && entry.permission == permission
        });
        
        Ok(has_specific_permission)
    }
    
    /// 获取用户的所有权限
    pub fn get_user_permissions(&self, user_id: &str) -> Result<Vec<PermissionEntry>> {
        let entries = self.entries.read().map_err(|e| Error::lock(e.to_string()))?;
        
        Ok(entries
            .iter()
            .filter(|entry| entry.user_id == user_id && !entry.is_expired())
            .cloned()
            .collect())
    }
    
    /// 清理过期的权限
    pub fn cleanup_expired_permissions(&self) -> Result<usize> {
        let mut entries = self.entries.write().map_err(|e| Error::lock(e.to_string()))?;
        
        let original_len = entries.len();
        
        entries.retain(|entry| !entry.is_expired());
        
        let removed_count = original_len - entries.len();
        
        if removed_count > 0 {
            debug!("已清理 {} 个过期权限", removed_count);
        }
        
        Ok(removed_count)
    }
    
    /// 导出所有权限条目
    pub fn export_permissions(&self) -> Result<Vec<PermissionEntry>> {
        let entries = self.entries.read().map_err(|e| Error::lock(e.to_string()))?;
        Ok(entries.clone())
    }
    
    /// 导入权限条目
    pub fn import_permissions(&self, permissions: Vec<PermissionEntry>) -> Result<()> {
        let mut entries = self.entries.write().map_err(|e| Error::lock(e.to_string()))?;
        *entries = permissions;
        Ok(())
    }
}

impl Default for PermissionManager {
    fn default() -> Self {
        Self::new()
    }
}
