// 权限管理模块 - 为存储引擎提供权限控制功能

use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use rocksdb::{DB, Options, ColumnFamilyDescriptor};
use crate::error::{Error, Result};
use std::path::Path;

// 列族名称常量
const CF_ROLES: &str = "roles";
const CF_ACLS: &str = "acls";
const CF_USER_ROLES: &str = "user_roles";
const CF_SESSIONS: &str = "sessions";
const CF_GROUPS: &str = "groups";
const CF_USER_GROUPS: &str = "user_groups";
const CF_GROUP_ROLES: &str = "group_roles";
const CF_TEMP_PERMISSIONS: &str = "temp_permissions";
const CF_RESOURCE_ACLS: &str = "resource_acls";

/// 权限类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    Read,   // 读取权限
    Write,  // 写入权限
    Delete, // 删除权限
    Admin,  // 管理权限
    Deny,   // 拒绝权限
}

/// 资源类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Data,     // 数据资源
    Model,    // 模型资源
    Algorithm, // 算法资源
}

/// 角色定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: String,
    pub name: String,
    pub permissions: HashSet<Permission>,
    pub parent: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Role {
    /// 获取角色的有效权限（包括继承的权限）
    pub fn get_effective_permissions(&self, roles: &HashMap<String, Role>) -> HashSet<Permission> {
        let mut result = self.permissions.clone();
        
        // 处理权限继承
        if let Some(parent_id) = &self.parent {
            if let Some(parent) = roles.get(parent_id) {
                result.extend(parent.get_effective_permissions(roles));
            }
        }
        
        result
    }
}

/// 用户组定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Group {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub roles: HashSet<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// 资源访问控制列表
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceACL {
    pub resource_id: String,
    pub resource_type: ResourceType,
    pub owner: String,
    pub roles: HashMap<String, HashSet<Permission>>,
    pub groups: HashMap<String, HashSet<Permission>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub user_permissions: HashMap<String, HashSet<Permission>>,
}

/// 会话定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub user_id: String,
    pub roles: HashSet<String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

impl Session {
    /// 检查会话是否过期
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }
}

/// 临时权限定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempPermission {
    pub id: String,
    pub user_id: String,
    pub resource_id: String,
    pub permissions: HashSet<Permission>,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// 权限管理器
#[derive(Debug)]
pub struct PermissionManager {
    db: DB,
    roles: HashMap<String, Role>,
    groups: HashMap<String, Group>,
    resource_acls: HashMap<String, ResourceACL>,
    user_roles: HashMap<String, HashSet<String>>,
    user_groups: HashMap<String, HashSet<String>>,
    group_roles: HashMap<String, HashSet<String>>,
    sessions: HashMap<String, Session>,
    temp_permissions: HashMap<String, TempPermission>,
}

impl PermissionManager {
    /// 创建新的权限管理器
    pub fn new(path: &str) -> Result<Self> {
        // 创建权限数据库目录
        let path = Path::new(path);
        if !path.exists() {
            std::fs::create_dir_all(path)?;
        }
        
        // 数据库路径
        let db_path = path.join("permission_db");
        
        // 数据库选项
        let mut options = Options::default();
        options.create_if_missing(true);
        options.create_missing_column_families(true);
        
        // 定义列族
        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_ROLES, Options::default()),
            ColumnFamilyDescriptor::new(CF_ACLS, Options::default()),
            ColumnFamilyDescriptor::new(CF_USER_ROLES, Options::default()),
            ColumnFamilyDescriptor::new(CF_SESSIONS, Options::default()),
            ColumnFamilyDescriptor::new(CF_GROUPS, Options::default()),
            ColumnFamilyDescriptor::new(CF_USER_GROUPS, Options::default()),
            ColumnFamilyDescriptor::new(CF_GROUP_ROLES, Options::default()),
            ColumnFamilyDescriptor::new(CF_TEMP_PERMISSIONS, Options::default()),
            ColumnFamilyDescriptor::new(CF_RESOURCE_ACLS, Options::default()),
        ];
        
        // 打开或创建数据库
        let db = if db_path.exists() {
            DB::open_cf_descriptors(&options, db_path, cf_descriptors)?
        } else {
            DB::open_cf_descriptors(&options, db_path, cf_descriptors)?
        };
        
        let mut manager = Self {
            db,
            roles: HashMap::new(),
            groups: HashMap::new(),
            resource_acls: HashMap::new(),
            user_roles: HashMap::new(),
            user_groups: HashMap::new(),
            group_roles: HashMap::new(),
            sessions: HashMap::new(),
            temp_permissions: HashMap::new(),
        };
        
        // 加载数据
        manager.load_data()?;
        
        Ok(manager)
    }

    /// 创建内存权限管理器（用于内存存储 / 测试场景）
    pub fn new_in_memory() -> Result<Self> {
        // 使用系统临时目录创建独立的权限数据库目录
        let temp_dir = std::env::temp_dir().join(format!("vecmind_permissions_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir)?;
        // 复用现有的基于路径的构造逻辑
        let path_str = temp_dir
            .to_str()
            .ok_or_else(|| Error::Configuration("Invalid temp permissions path".to_string()))?;
        Self::new(path_str)
    }
    
    /// 从数据库加载所有权限数据
    fn load_data(&mut self) -> Result<()> {
        // 加载角色
        if let Some(roles_cf) = self.db.cf_handle(CF_ROLES) {
            let iter = self.db.iterator_cf(&roles_cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (_, value) = item?;
                let role: Role = bincode::deserialize(&value)?;
                self.roles.insert(role.id.clone(), role);
            }
        }
        
        // 加载用户-角色关系
        if let Some(user_roles_cf) = self.db.cf_handle(CF_USER_ROLES) {
            let iter = self.db.iterator_cf(&user_roles_cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key, value) = item?;
                let user_id = String::from_utf8(key.to_vec())?;
                let roles: HashSet<String> = bincode::deserialize(&value)?;
                self.user_roles.insert(user_id, roles);
            }
        }
        
        // 加载组
        if let Some(groups_cf) = self.db.cf_handle(CF_GROUPS) {
            let iter = self.db.iterator_cf(&groups_cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (_, value) = item?;
                let group: Group = bincode::deserialize(&value)?;
                self.groups.insert(group.id.clone(), group);
            }
        }
        
        // 加载用户-组关系
        if let Some(user_groups_cf) = self.db.cf_handle(CF_USER_GROUPS) {
            let iter = self.db.iterator_cf(&user_groups_cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key, value) = item?;
                let user_id = String::from_utf8(key.to_vec())?;
                let groups: HashSet<String> = bincode::deserialize(&value)?;
                self.user_groups.insert(user_id, groups);
            }
        }
        
        // 加载组-角色关系
        if let Some(group_roles_cf) = self.db.cf_handle(CF_GROUP_ROLES) {
            let iter = self.db.iterator_cf(&group_roles_cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key, value) = item?;
                let group_id = String::from_utf8(key.to_vec())?;
                let roles: HashSet<String> = bincode::deserialize(&value)?;
                self.group_roles.insert(group_id, roles);
            }
        }
        
        // 加载资源ACL
        if let Some(resource_acls_cf) = self.db.cf_handle(CF_RESOURCE_ACLS) {
            let iter = self.db.iterator_cf(&resource_acls_cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (_, value) = item?;
                let acl: ResourceACL = bincode::deserialize(&value)?;
                self.resource_acls.insert(acl.resource_id.clone(), acl);
            }
        }
        
        // 加载会话
        if let Some(sessions_cf) = self.db.cf_handle(CF_SESSIONS) {
            let iter = self.db.iterator_cf(&sessions_cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (_, value) = item?;
                let session: Session = bincode::deserialize(&value)?;
                if !session.is_expired() {
                    self.sessions.insert(session.id.clone(), session);
                }
            }
        }
        
        // 加载临时权限
        if let Some(temp_permissions_cf) = self.db.cf_handle(CF_TEMP_PERMISSIONS) {
            let iter = self.db.iterator_cf(&temp_permissions_cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (_, value) = item?;
                let temp_perm: TempPermission = bincode::deserialize(&value)?;
                if Utc::now() < temp_perm.expires_at {
                    self.temp_permissions.insert(temp_perm.id.clone(), temp_perm);
                }
            }
        }
        
        Ok(())
    }

    /// 创建角色
    pub fn create_role(
        &mut self,
        name: String,
        permissions: HashSet<Permission>,
        parent: Option<String>,
    ) -> Result<String> {
        // 验证父角色是否存在
        if let Some(parent_id) = &parent {
            if !self.roles.contains_key(parent_id) {
                return Err(Error::InvalidInput(format!("父角色不存在: {}", parent_id)));
            }
        }
        
        let role_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let role = Role {
            id: role_id.clone(),
            name,
            permissions,
            parent,
            created_at: now,
            updated_at: now,
        };
        
        // 保存到数据库
        if let Some(roles_cf) = self.db.cf_handle(CF_ROLES) {
            let encoded = bincode::serialize(&role)?;
            self.db.put_cf(&roles_cf, role_id.as_bytes(), encoded)?;
        } else {
            return Err(Error::StorageError("找不到角色列族".to_string()));
        }
        
        // 更新内存
        self.roles.insert(role_id.clone(), role);
        
        Ok(role_id)
    }

    /// 更新角色
    pub fn update_role(
        &mut self,
        role_id: &str,
        name: Option<String>,
        new_parent: Option<String>,
        permissions: Option<Vec<Permission>>,
    ) -> Result<()> {
        let mut role = match self.roles.get(role_id) {
            Some(r) => r.clone(),
            None => return Err(Error::NotFound(format!("角色不存在: {}", role_id))),
        };
        
        // 检查新父角色是否存在
        if let Some(parent_id) = &new_parent {
            if !self.roles.contains_key(parent_id) && parent_id != "" {
                return Err(Error::InvalidInput(format!("父角色不存在: {}", parent_id)));
            }
        }
        
        // 更新字段
        if let Some(name_val) = name {
            role.name = name_val;
        }
        
        if let Some(parent_val) = new_parent {
            role.parent = if parent_val.is_empty() { None } else { Some(parent_val) };
        }
        
        if let Some(perms) = permissions {
            role.permissions = perms.into_iter().collect();
        }
        
        role.updated_at = Utc::now();
        
        // 保存到数据库
        if let Some(roles_cf) = self.db.cf_handle(CF_ROLES) {
            let encoded = bincode::serialize(&role)?;
            self.db.put_cf(&roles_cf, role_id.as_bytes(), encoded)?;
        } else {
            return Err(Error::StorageError("找不到角色列族".to_string()));
        }
        
        // 更新内存
        self.roles.insert(role_id.to_string(), role);
        
        Ok(())
    }

    /// 分配角色给用户
    pub fn assign_role_to_user(&mut self, user_id: &str, role_id: &str) -> Result<()> {
        // 检查角色是否存在
        if !self.roles.contains_key(role_id) {
            return Err(Error::NotFound(format!("角色不存在: {}", role_id)));
        }
        
        // 获取用户当前角色
        let mut user_roles = self.user_roles.entry(user_id.to_string())
            .or_insert_with(HashSet::new)
            .clone();
        
        // 添加新角色
        user_roles.insert(role_id.to_string());
        
        // 保存到数据库
        if let Some(user_roles_cf) = self.db.cf_handle(CF_USER_ROLES) {
            let encoded = bincode::serialize(&user_roles)?;
            self.db.put_cf(&user_roles_cf, user_id.as_bytes(), encoded)?;
        } else {
            return Err(Error::StorageError("找不到用户角色列族".to_string()));
        }
        
        // 更新内存
        self.user_roles.insert(user_id.to_string(), user_roles);
        
        Ok(())
    }

    /// 创建会话
    pub fn create_session(&mut self, user_id: String, ttl_seconds: i64) -> Result<String> {
        let session_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        
        // 获取用户角色
        let roles = self.user_roles.get(&user_id)
            .map(|r| r.clone())
            .unwrap_or_else(HashSet::new);
        
        let session = Session {
            id: session_id.clone(),
            user_id,
            roles,
            created_at: now,
            expires_at: now + Duration::seconds(ttl_seconds),
        };
        
        // 保存到数据库
        if let Some(sessions_cf) = self.db.cf_handle(CF_SESSIONS) {
            let encoded = bincode::serialize(&session)?;
            self.db.put_cf(&sessions_cf, session_id.as_bytes(), encoded)?;
        } else {
            return Err(Error::StorageError("找不到会话列族".to_string()));
        }
        
        // 更新内存
        self.sessions.insert(session_id.clone(), session);
        
        Ok(session_id)
    }

    /// 获取会话
    fn get_session(&self, session_id: &str) -> Result<&Session> {
        self.sessions.get(session_id)
            .ok_or_else(|| Error::NotFound(format!("会话不存在: {}", session_id)))
    }

    /// 检查会话是否有效
    pub fn is_session_valid(&self, session_id: &str) -> Result<bool> {
        match self.sessions.get(session_id) {
            Some(session) => Ok(!session.is_expired()),
            None => Ok(false),
        }
    }

    /// 从会话中获取用户ID
    pub fn get_user_from_session(&self, session_id: &str) -> Result<String> {
        let session = self.get_session(session_id)?;
        Ok(session.user_id.clone())
    }

    /// 检查资源ACL是否存在
    pub fn resource_acl_exists(&self, resource_id: &str) -> Result<bool> {
        Ok(self.resource_acls.contains_key(resource_id))
    }

    /// 获取资源ACL
    fn get_resource_acl(&self, resource_id: &str) -> Result<&ResourceACL> {
        self.resource_acls.get(resource_id)
            .ok_or_else(|| Error::NotFound(format!("资源ACL不存在: {}", resource_id)))
    }

    /// 创建资源ACL
    pub fn create_resource_acl(
        &mut self,
        resource_id: String,
        resource_type: ResourceType,
        owner: String,
    ) -> Result<()> {
        let now = Utc::now();
        let acl = ResourceACL {
            resource_id: resource_id.clone(),
            resource_type,
            owner,
            roles: HashMap::new(),
            groups: HashMap::new(),
            created_at: now,
            updated_at: now,
            user_permissions: HashMap::new(),
        };
        
        // 保存到数据库
        if let Some(resource_acls_cf) = self.db.cf_handle(CF_RESOURCE_ACLS) {
            let encoded = bincode::serialize(&acl)?;
            self.db.put_cf(&resource_acls_cf, resource_id.as_bytes(), encoded)?;
        } else {
            return Err(Error::StorageError("找不到资源ACL列族".to_string()));
        }
        
        // 更新内存
        self.resource_acls.insert(resource_id, acl);
        
        Ok(())
    }

    /// 检查权限
    pub fn check_permission(
        &self,
        session_id: &str,
        resource_id: &str,
        permission: Permission,
    ) -> Result<bool> {
        // 检查会话是否有效
        let session = match self.get_session(session_id) {
            Ok(s) => s,
            Err(_) => return Ok(false), // 会话不存在
        };
        
        if session.is_expired() {
            return Ok(false); // 会话已过期
        }
        
        let user_id = &session.user_id;
        
        // 检查资源ACL是否存在
        let acl = match self.get_resource_acl(resource_id) {
            Ok(a) => a,
            Err(_) => return Ok(false), // 资源ACL不存在
        };
        
        // 资源所有者具有所有权限
        if acl.owner == *user_id {
            return Ok(true);
        }
        
        // 检查用户直接权限
        if let Some(perms) = acl.user_permissions.get(user_id) {
            if perms.contains(&permission) {
                return Ok(true);
            }
            if perms.contains(&Permission::Admin) {
                return Ok(true);
            }
            if perms.contains(&Permission::Deny) {
                return Ok(false);
            }
        }
        
        // 检查角色权限
        for role_id in &session.roles {
            if let Some(role) = self.roles.get(role_id) {
                let effective_perms = role.get_effective_permissions(&self.roles);
                if effective_perms.contains(&permission) {
                    return Ok(true);
                }
                if effective_perms.contains(&Permission::Admin) {
                    return Ok(true);
                }
                if effective_perms.contains(&Permission::Deny) {
                    return Ok(false);
                }
            }
        }
        
        // 默认拒绝
        Ok(false)
    }
} 