// 权限管理操作模块
// 实现权限管理相关方法

use crate::{Error, Result};
use super::core::Storage;
use super::data::DATA_RAW_PREFIX;

impl Storage {
    /// 创建角色
    pub fn create_role(&mut self, name: String, permissions: std::collections::HashSet<crate::storage::permissions::Permission>) -> Result<String> {
        // 转换权限类型：从 storage::permissions::Permission 到 module::permissions::Permission
        let module_permissions: std::collections::HashSet<crate::storage::module::permissions::Permission> = permissions.into_iter()
            .map(|p| match p {
                crate::storage::permissions::Permission::Read => crate::storage::module::permissions::Permission::Read,
                crate::storage::permissions::Permission::Write => crate::storage::module::permissions::Permission::Write,
                crate::storage::permissions::Permission::Execute => crate::storage::module::permissions::Permission::Admin,
                crate::storage::permissions::Permission::Manage => crate::storage::module::permissions::Permission::Admin,
            })
            .collect();
        self.permission_manager.borrow_mut().create_role(name, module_permissions, None)
    }
    
    /// 将角色分配给用户
    pub fn assign_role_to_user(&mut self, user_id: &str, role_id: &str) -> Result<()> {
        self.permission_manager.borrow_mut().assign_role_to_user(user_id, role_id)
    }
    
    /// 创建会话
    pub fn create_session(&mut self, user_id: String, ttl_seconds: i64) -> Result<String> {
        self.permission_manager.borrow_mut().create_session(user_id, ttl_seconds)
    }
    
    /// 添加数据（带会话权限检查）
    pub fn put_data_with_session(&self, session_id: &str, data_id: &str, data: &[u8]) -> Result<()> {
        // 检查权限
        if !self.permission_manager.borrow().check_permission(
            session_id,
            data_id,
            crate::storage::module::permissions::Permission::Write,
        )? {
            return Err(Error::Permission("Permission denied".to_string()));
        }
        
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        self.put(key.as_bytes(), data)?;
        Ok(())
    }
    
    /// 获取数据（带会话权限检查）
    pub fn get_data_with_session(&self, session_id: &str, data_id: &str) -> Result<Option<Vec<u8>>> {
        // 检查权限
        if !self.permission_manager.borrow().check_permission(
            session_id,
            data_id,
            crate::storage::module::permissions::Permission::Read,
        )? {
            return Err(Error::Permission("Permission denied".to_string()));
        }
        
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        self.get(key.as_bytes())
    }
    
    /// 删除数据（带会话权限检查）
    pub fn delete_data_with_session(&self, session_id: &str, data_id: &str) -> Result<()> {
        // 检查权限
        if !self.permission_manager.borrow().check_permission(
            session_id,
            data_id,
            crate::storage::module::permissions::Permission::Write,
        )? {
            return Err(Error::Permission("Permission denied".to_string()));
        }
        
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        self.delete(key.as_bytes())?;
        Ok(())
    }
    
    /// 添加数据（带权限检查）
    /// 
    /// 此方法会检查资源ACL：
    /// - 如果资源有ACL，则要求使用put_data_with_session方法并提供session_id
    /// - 如果资源没有ACL，则允许操作（系统级操作）
    pub fn put_data_with_permission(
        &self, 
        data_id: &str, 
        data: &[u8], 
        permission: crate::storage::module::permissions::Permission,
    ) -> Result<()> {
        // 检查资源是否有ACL
        let has_acl = match self.permission_manager.borrow().resource_acl_exists(data_id) {
            Ok(exists) => exists,
            Err(_) => false, // 如果检查失败，假设没有ACL以保持向后兼容
        };
        
        // 如果资源有ACL，必须提供调用者信息进行权限验证
        if has_acl {
            log::warn!(
                "资源 {} 有ACL但put_data_with_permission方法未提供调用者标识，拒绝操作。请使用put_data_with_session方法",
                data_id
            );
            return Err(Error::Permission(format!(
                "资源 {} 需要权限验证，请使用put_data_with_session方法并提供session_id进行权限验证",
                data_id
            )));
        }
        
        // 如果资源没有ACL，允许操作（系统级操作，向后兼容）
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        self.put(key.as_bytes(), data)?;
        Ok(())
    }
    
    /// 获取数据（带权限检查）
    /// 
    /// 此方法会检查资源ACL：
    /// - 如果资源有ACL，则要求使用get_data_with_session方法并提供session_id
    /// - 如果资源没有ACL，则允许操作（系统级操作）
    pub fn get_data_with_permission(
        &self, 
        data_id: &str, 
        permission: crate::storage::module::permissions::Permission,
    ) -> Result<Option<Vec<u8>>> {
        // 检查资源是否有ACL
        let has_acl = match self.permission_manager.borrow().resource_acl_exists(data_id) {
            Ok(exists) => exists,
            Err(_) => false, // 如果检查失败，假设没有ACL以保持向后兼容
        };
        
        // 如果资源有ACL，必须提供调用者信息进行权限验证
        if has_acl {
            log::warn!(
                "资源 {} 有ACL但get_data_with_permission方法未提供调用者标识，拒绝操作。请使用get_data_with_session方法",
                data_id
            );
            return Err(Error::Permission(format!(
                "资源 {} 需要权限验证，请使用get_data_with_session方法并提供session_id进行权限验证",
                data_id
            )));
        }
        
        // 如果资源没有ACL，允许操作（系统级操作，向后兼容）
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        self.get(key.as_bytes())
    }
    
    /// 创建资源访问控制列表
    pub fn create_resource_acl(&mut self, resource_id: &str, resource_type: crate::storage::module::permissions::ResourceType, owner: &str) -> Result<()> {
        self.permission_manager.borrow_mut().create_resource_acl(resource_id.to_string(), resource_type, owner.to_string())
    }
}

