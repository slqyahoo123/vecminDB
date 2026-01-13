use serde::{Serialize, Deserialize};
use std::collections::HashSet;
use std::sync::RwLock;
use crate::error::{Error, Result};
// use log::{debug, info, warn, error};
// 日志宏未使用，已注释

/// 权限枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// 读取权限
    Read,
    /// 写入权限
    Write,
    /// 更新权限
    Update,
    /// 删除权限
    Delete,
    /// 管理权限
    Manage,
    /// 执行权限
    Execute,
    /// 所有权限
    All,
}

/// 资源类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// 模型资源
    Model,
    /// 数据资源
    Data,
    /// 索引资源
    Index,
    /// 算法资源
    Algorithm,
    /// 任务资源
    Task,
    /// 用户资源
    User,
    /// 系统资源
    System,
    /// 所有资源
    All,
}

/// 权限条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionEntry {
    /// 用户ID
    pub user_id: String,
    /// 资源类型
    pub resource_type: ResourceType,
    /// 资源ID
    pub resource_id: Option<String>,
    /// 权限集合
    pub permissions: HashSet<Permission>,
    /// 创建时间
    pub created_at: u64,
    /// 过期时间
    pub expires_at: Option<u64>,
}

impl PermissionEntry {
    /// 创建新的权限条目
    pub fn new(
        user_id: String,
        resource_type: ResourceType,
        resource_id: Option<String>,
        permissions: HashSet<Permission>,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        Self {
            user_id,
            resource_type,
            resource_id,
            permissions,
            created_at: now,
            expires_at: None,
        }
    }
    
    /// 设置过期时间
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
    
    /// 检查是否过期
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
                
            expires_at < now
        } else {
            false
        }
    }
    
    /// 检查是否具有特定权限
    pub fn has_permission(&self, permission: Permission) -> bool {
        if self.is_expired() {
            return false;
        }
        
        self.permissions.contains(&permission) || self.permissions.contains(&Permission::All)
    }
    
    /// 检查是否具有多个权限中的任一权限
    pub fn has_any_permission(&self, permissions: &[Permission]) -> bool {
        if self.is_expired() {
            return false;
        }
        
        if self.permissions.contains(&Permission::All) {
            return true;
        }
        
        permissions.iter().any(|p| self.permissions.contains(p))
    }
    
    /// 检查是否具有指定的所有权限
    pub fn has_all_permissions(&self, permissions: &[Permission]) -> bool {
        if self.is_expired() {
            return false;
        }
        
        if self.permissions.contains(&Permission::All) {
            return true;
        }
        
        permissions.iter().all(|p| self.permissions.contains(p))
    }
}

/// 权限管理器
pub struct PermissionManager {
    permissions: RwLock<Vec<PermissionEntry>>,
}

impl PermissionManager {
    /// 创建新的权限管理器
    pub fn new() -> Self {
        Self {
            permissions: RwLock::new(Vec::new()),
        }
    }
    
    /// 添加权限条目
    pub fn add_permission(&self, entry: PermissionEntry) -> Result<()> {
        let mut permissions = self.permissions.write().map_err(|e| Error::Internal(format!("获取权限写锁失败: {}", e)))?;
        
        // 移除过期的相同条目
        permissions.retain(|e| 
            !(e.user_id == entry.user_id && 
              e.resource_type == entry.resource_type && 
              e.resource_id == entry.resource_id) || 
            !e.is_expired()
        );
        
        permissions.push(entry);
        Ok(())
    }
    
    /// 移除权限条目
    pub fn remove_permission(
        &self,
        user_id: &str,
        resource_type: ResourceType,
        resource_id: Option<&str>,
    ) -> Result<bool> {
        let mut permissions = self.permissions.write().map_err(|e| Error::Internal(format!("获取权限写锁失败: {}", e)))?;
        
        let initial_len = permissions.len();
        
        permissions.retain(|e| 
            !(e.user_id == user_id && 
              e.resource_type == resource_type && 
              match (&e.resource_id, resource_id) {
                  (Some(e_id), Some(r_id)) => e_id != r_id,
                  (None, None) => false,
                  _ => true,
              })
        );
        
        Ok(permissions.len() < initial_len)
    }
    
    /// 检查用户是否对资源有指定权限
    pub fn check_permission(
        &self,
        user_id: &str,
        resource_type: ResourceType,
        resource_id: Option<&str>,
        permission: Permission,
    ) -> Result<bool> {
        let permissions = self.permissions.read().map_err(|e| Error::Internal(format!("获取权限读锁失败: {}", e)))?;
        
        // 清理过期权限在实际使用中应该由定时任务处理
        // 这里简化处理逻辑
        
        // 检查具体资源权限
        let specific_permission = match resource_id {
            Some(id) => permissions.iter().any(|e| 
                !e.is_expired() && 
                e.user_id == user_id && 
                e.resource_type == resource_type && 
                e.resource_id.as_deref() == Some(id) && 
                e.has_permission(permission)
            ),
            None => false,
        };
        
        if specific_permission {
            return Ok(true);
        }
        
        // 检查资源类型权限
        let type_permission = permissions.iter().any(|e| 
            !e.is_expired() && 
            e.user_id == user_id && 
            e.resource_type == resource_type && 
            e.resource_id.is_none() && 
            e.has_permission(permission)
        );
        
        if type_permission {
            return Ok(true);
        }
        
        // 检查全局权限
        let global_permission = permissions.iter().any(|e| 
            !e.is_expired() && 
            e.user_id == user_id && 
            e.resource_type == ResourceType::All && 
            e.has_permission(permission)
        );
        
        Ok(global_permission)
    }
    
    /// 获取用户对资源的所有权限
    pub fn get_permissions(
        &self,
        user_id: &str,
        resource_type: ResourceType,
        resource_id: Option<&str>,
    ) -> Result<HashSet<Permission>> {
        let permissions = self.permissions.read().map_err(|e| Error::Internal(format!("获取权限读锁失败: {}", e)))?;
        
        let mut result = HashSet::new();
        
        // 收集具体资源权限
        if let Some(id) = resource_id {
            for entry in permissions.iter() {
                if !entry.is_expired() && 
                   entry.user_id == user_id && 
                   entry.resource_type == resource_type && 
                   entry.resource_id.as_deref() == Some(id) {
                    // 如果有All权限，直接添加所有权限
                    if entry.permissions.contains(&Permission::All) {
                        result.insert(Permission::Read);
                        result.insert(Permission::Write);
                        result.insert(Permission::Update);
                        result.insert(Permission::Delete);
                        result.insert(Permission::Manage);
                        result.insert(Permission::Execute);
                        return Ok(result);
                    }
                    
                    // 否则添加具体权限
                    result.extend(entry.permissions.iter().cloned());
                }
            }
        }
        
        // 收集资源类型权限
        for entry in permissions.iter() {
            if !entry.is_expired() && 
               entry.user_id == user_id && 
               entry.resource_type == resource_type && 
               entry.resource_id.is_none() {
                // 如果有All权限，直接添加所有权限
                if entry.permissions.contains(&Permission::All) {
                    result.insert(Permission::Read);
                    result.insert(Permission::Write);
                    result.insert(Permission::Update);
                    result.insert(Permission::Delete);
                    result.insert(Permission::Manage);
                    result.insert(Permission::Execute);
                    return Ok(result);
                }
                
                // 否则添加具体权限
                result.extend(entry.permissions.iter().cloned());
            }
        }
        
        // 收集全局权限
        for entry in permissions.iter() {
            if !entry.is_expired() && 
               entry.user_id == user_id && 
               entry.resource_type == ResourceType::All {
                // 如果有All权限，直接添加所有权限
                if entry.permissions.contains(&Permission::All) {
                    result.insert(Permission::Read);
                    result.insert(Permission::Write);
                    result.insert(Permission::Update);
                    result.insert(Permission::Delete);
                    result.insert(Permission::Manage);
                    result.insert(Permission::Execute);
                    return Ok(result);
                }
                
                // 否则添加具体权限
                result.extend(entry.permissions.iter().cloned());
            }
        }
        
        Ok(result)
    }
    
    /// 列出用户的所有权限条目
    pub fn list_user_permissions(&self, user_id: &str) -> Result<Vec<PermissionEntry>> {
        let permissions = self.permissions.read().map_err(|e| Error::Internal(format!("获取权限读锁失败: {}", e)))?;
        
        Ok(permissions
            .iter()
            .filter(|e| !e.is_expired() && e.user_id == user_id)
            .cloned()
            .collect())
    }
    
    /// 列出资源的所有权限条目
    pub fn list_resource_permissions(
        &self,
        resource_type: ResourceType,
        resource_id: &str,
    ) -> Result<Vec<PermissionEntry>> {
        let permissions = self.permissions.read().map_err(|e| Error::Internal(format!("获取权限读锁失败: {}", e)))?;
        
        Ok(permissions
            .iter()
            .filter(|e| 
                !e.is_expired() && 
                e.resource_type == resource_type && 
                e.resource_id.as_deref() == Some(resource_id)
            )
            .cloned()
            .collect())
    }
    
    /// 清理过期的权限条目
    pub fn clean_expired_permissions(&self) -> Result<usize> {
        let mut permissions = self.permissions.write().map_err(|e| Error::Internal(format!("获取权限写锁失败: {}", e)))?;
        
        let initial_len = permissions.len();
        
        permissions.retain(|e| !e.is_expired());
        
        Ok(initial_len - permissions.len())
    }
}



