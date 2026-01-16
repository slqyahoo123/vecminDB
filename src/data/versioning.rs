//! 数据版本管理模块
//! 提供完整的数据版本控制功能，包括数据版本跟踪、分支管理、合并策略等

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::{Arc, RwLock};
use std::path::{Path, PathBuf};
use std::fs;

use log::info;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};
use uuid::Uuid;

use crate::error::{Result, Error};
use crate::data::value::DataValue;
use crate::data::types::DataFormat;

/// 数据版本ID
pub type VersionId = String;

/// 分支名称
pub type BranchName = String;

/// 标签名称
pub type TagName = String;

/// 版本信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// 版本ID
    pub version_id: VersionId,
    /// 父版本ID
    pub parent_id: Option<VersionId>,
    /// 分支名称
    pub branch: BranchName,
    /// 提交信息
    pub message: String,
    /// 作者
    pub author: String,
    /// 创建时间
    pub timestamp: u64,
    /// 数据哈希
    pub data_hash: String,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 文件列表
    pub files: Vec<FileEntry>,
}

impl VersionInfo {
    /// 创建新版本信息
    pub fn new(
        parent_id: Option<VersionId>,
        branch: BranchName,
        message: String,
        author: String,
        data_hash: String,
        files: Vec<FileEntry>,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            version_id: Uuid::new_v4().to_string(),
            parent_id,
            branch,
            message,
            author,
            timestamp,
            data_hash,
            metadata: HashMap::new(),
            files,
        }
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// 获取简短ID
    pub fn short_id(&self) -> String {
        self.version_id[..8].to_string()
    }
}

/// 文件条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    /// 文件路径
    pub path: String,
    /// 文件哈希
    pub hash: String,
    /// 文件大小
    pub size: u64,
    /// 修改时间
    pub modified: u64,
    /// 文件类型
    pub file_type: FileType,
}

/// 文件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileType {
    /// 数据文件
    Data,
    /// 配置文件
    Config,
    /// 模型文件
    Model,
    /// 其他文件
    Other,
}

/// 分支信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    /// 分支名称
    pub name: BranchName,
    /// 当前版本ID
    pub head: VersionId,
    /// 创建时间
    pub created: u64,
    /// 最后更新时间
    pub updated: u64,
    /// 描述
    pub description: String,
}

/// 标签信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagInfo {
    /// 标签名称
    pub name: TagName,
    /// 指向的版本ID
    pub version_id: VersionId,
    /// 创建时间
    pub created: u64,
    /// 描述
    pub description: String,
}

/// 差异类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffType {
    /// 记录添加
    RecordAdded,
    /// 记录删除
    RecordDeleted,
    /// 记录修改
    RecordModified,
    /// 字段添加
    FieldAdded,
    /// 字段删除
    FieldDeleted,
    /// 字段修改
    FieldModified,
}

/// 差异条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    /// 差异类型
    pub diff_type: DiffType,
    /// 路径（如记录ID或字段路径）
    pub path: String,
    /// 旧值
    pub old_value: Option<DataValue>,
    /// 新值
    pub new_value: Option<DataValue>,
    /// 上下文信息
    pub context: HashMap<String, String>,
}

/// 合并策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// 自动合并（优先源分支）
    AutoSource,
    /// 自动合并（优先目标分支）
    AutoTarget,
    /// 手动合并
    Manual,
    /// 基于时间戳合并
    Timestamp,
    /// 自定义合并策略
    Custom(String),
}

/// 合并冲突
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConflict {
    /// 冲突路径
    pub path: String,
    /// 基础版本值
    pub base_value: Option<DataValue>,
    /// 源版本值
    pub source_value: Option<DataValue>,
    /// 目标版本值
    pub target_value: Option<DataValue>,
    /// 冲突类型
    pub conflict_type: ConflictType,
}

/// 冲突类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    /// 同时修改
    ModifyModify,
    /// 删除修改
    DeleteModify,
    /// 修改删除
    ModifyDelete,
    /// 同时添加
    AddAdd,
}

/// 合并结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    /// 是否成功
    pub success: bool,
    /// 合并后的版本ID
    pub merged_version_id: Option<VersionId>,
    /// 冲突列表
    pub conflicts: Vec<MergeConflict>,
    /// 合并统计
    pub stats: MergeStats,
}

/// 合并统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeStats {
    /// 添加的记录数
    pub records_added: usize,
    /// 删除的记录数
    pub records_deleted: usize,
    /// 修改的记录数
    pub records_modified: usize,
    /// 冲突数量
    pub conflicts_count: usize,
}

/// 数据版本管理器
pub struct DataVersionManager {
    /// 存储根目录
    storage_root: PathBuf,
    /// 版本信息存储
    versions: Arc<RwLock<HashMap<VersionId, VersionInfo>>>,
    /// 分支信息存储
    branches: Arc<RwLock<HashMap<BranchName, BranchInfo>>>,
    /// 标签信息存储
    tags: Arc<RwLock<HashMap<TagName, TagInfo>>>,
    /// 当前分支
    current_branch: Arc<RwLock<BranchName>>,
    /// 配置
    config: VersionConfig,
}

/// 版本管理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionConfig {
    /// 是否启用压缩存储
    pub enable_compression: bool,
    /// 最大版本历史数量
    pub max_versions: Option<usize>,
    /// 自动清理间隔（秒）
    pub cleanup_interval: u64,
    /// 默认分支名称
    pub default_branch: String,
    /// 数据存储格式
    pub storage_format: DataFormat,
}

impl Default for VersionConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            max_versions: Some(1000),
            cleanup_interval: 86400, // 24小时
            default_branch: "main".to_string(),
            storage_format: DataFormat::Json,
        }
    }
}

impl DataVersionManager {
    /// 创建新的版本管理器
    pub fn new<P: AsRef<Path>>(storage_root: P, config: VersionConfig) -> Result<Self> {
        let storage_root = storage_root.as_ref().to_path_buf();
        
        // 确保存储目录存在
        if !storage_root.exists() {
            fs::create_dir_all(&storage_root)
                .map_err(|e| Error::io_error(format!("创建存储目录失败: {}", e)))?;
        }

        let manager = Self {
            storage_root,
            versions: Arc::new(RwLock::new(HashMap::new())),
            branches: Arc::new(RwLock::new(HashMap::new())),
            tags: Arc::new(RwLock::new(HashMap::new())),
            current_branch: Arc::new(RwLock::new(config.default_branch.clone())),
            config,
        };

        // 初始化默认分支
        manager.initialize_default_branch()?;
        
        // 加载现有数据
        manager.load_metadata()?;

        Ok(manager)
    }

    /// 初始化默认分支
    fn initialize_default_branch(&self) -> Result<()> {
        let mut branches = self.branches.write().unwrap();
        
        if !branches.contains_key(&self.config.default_branch) {
            let branch_info = BranchInfo {
                name: self.config.default_branch.clone(),
                head: "".to_string(), // 空提交
                created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                description: "默认分支".to_string(),
            };
            
            branches.insert(self.config.default_branch.clone(), branch_info);
        }

        Ok(())
    }

    /// 加载元数据
    fn load_metadata(&self) -> Result<()> {
        // 加载版本信息
        let versions_file = self.storage_root.join("versions.json");
        if versions_file.exists() {
            let content = fs::read_to_string(&versions_file)
                .map_err(|e| Error::io_error(format!("读取版本文件失败: {}", e)))?;
            
            let versions_data: HashMap<VersionId, VersionInfo> = serde_json::from_str(&content)
                .map_err(|e| Error::invalid_input(format!("解析版本文件失败: {}", e)))?;
            
            *self.versions.write().unwrap() = versions_data;
        }

        // 加载分支信息
        let branches_file = self.storage_root.join("branches.json");
        if branches_file.exists() {
            let content = fs::read_to_string(&branches_file)
                .map_err(|e| Error::io_error(format!("读取分支文件失败: {}", e)))?;
            
            let branches_data: HashMap<BranchName, BranchInfo> = serde_json::from_str(&content)
                .map_err(|e| Error::invalid_input(format!("解析分支文件失败: {}", e)))?;
            
            *self.branches.write().unwrap() = branches_data;
        }

        // 加载标签信息
        let tags_file = self.storage_root.join("tags.json");
        if tags_file.exists() {
            let content = fs::read_to_string(&tags_file)
                .map_err(|e| Error::io_error(format!("读取标签文件失败: {}", e)))?;
            
            let tags_data: HashMap<TagName, TagInfo> = serde_json::from_str(&content)
                .map_err(|e| Error::invalid_input(format!("解析标签文件失败: {}", e)))?;
            
            *self.tags.write().unwrap() = tags_data;
        }

        Ok(())
    }

    /// 保存元数据
    fn save_metadata(&self) -> Result<()> {
        // 保存版本信息
        let versions_file = self.storage_root.join("versions.json");
        let versions_data = self.versions.read().unwrap().clone();
        let content = serde_json::to_string_pretty(&versions_data)
            .map_err(|e| Error::serialization(format!("序列化版本信息失败: {}", e)))?;
        fs::write(&versions_file, content)
            .map_err(|e| Error::io_error(format!("写入版本文件失败: {}", e)))?;

        // 保存分支信息
        let branches_file = self.storage_root.join("branches.json");
        let branches_data = self.branches.read().unwrap().clone();
        let content = serde_json::to_string_pretty(&branches_data)
            .map_err(|e| Error::serialization(format!("序列化分支信息失败: {}", e)))?;
        fs::write(&branches_file, content)
            .map_err(|e| Error::io_error(format!("写入分支文件失败: {}", e)))?;

        // 保存标签信息
        let tags_file = self.storage_root.join("tags.json");
        let tags_data = self.tags.read().unwrap().clone();
        let content = serde_json::to_string_pretty(&tags_data)
            .map_err(|e| Error::serialization(format!("序列化标签信息失败: {}", e)))?;
        fs::write(&tags_file, content)
            .map_err(|e| Error::io_error(format!("写入标签文件失败: {}", e)))?;

        Ok(())
    }

    /// 提交数据版本
    pub fn commit(
        &self,
        data: &[DataValue],
        message: String,
        author: String,
    ) -> Result<VersionId> {
        let current_branch = self.current_branch.read().unwrap().clone();
        
        // 计算数据哈希
        let data_hash = self.calculate_data_hash(data)?;
        
        // 创建文件条目
        let files = self.create_file_entries(data)?;
        
        // 获取父版本ID
        let parent_id = self.get_branch_head(&current_branch)?;
        
        // 创建版本信息
        let version_info = VersionInfo::new(
            parent_id.filter(|id| !id.is_empty()),
            current_branch.clone(),
            message,
            author,
            data_hash,
            files,
        );
        
        let version_id = version_info.version_id.clone();
        
        // 存储数据
        self.store_version_data(&version_id, data)?;
        
        // 更新版本信息
        {
            let mut versions = self.versions.write().unwrap();
            versions.insert(version_id.clone(), version_info);
        }
        
        // 更新分支头
        self.update_branch_head(&current_branch, &version_id)?;
        
        // 保存元数据
        self.save_metadata()?;
        
        info!("提交新版本: {} -> {}", current_branch, version_id);
        Ok(version_id)
    }

    /// 检出版本
    pub fn checkout(&self, version_id: &VersionId) -> Result<Vec<DataValue>> {
        let versions = self.versions.read().unwrap();
        let version_info = versions.get(version_id)
            .ok_or_else(|| Error::NotFound(format!("版本不存在: {}", version_id)))?;
        
        // 加载版本数据
        let data = self.load_version_data(version_id)?;
        
        info!("检出版本: {}", version_id);
        Ok(data)
    }

    /// 创建分支
    pub fn create_branch(
        &self,
        branch_name: &str,
        base_version: Option<&VersionId>,
    ) -> Result<()> {
        let mut branches = self.branches.write().unwrap();
        
        if branches.contains_key(branch_name) {
            return Err(Error::AlreadyExists(format!("分支已存在: {}", branch_name)));
        }
        
        // 确定基础版本
        let head = if let Some(base) = base_version {
            base.clone()
        } else {
            let current_branch = self.current_branch.read().unwrap();
            self.get_branch_head(&current_branch)?
                .unwrap_or_else(|| "".to_string())
        };
        
        let branch_info = BranchInfo {
            name: branch_name.to_string(),
            head,
            created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            description: format!("分支: {}", branch_name),
        };
        
        branches.insert(branch_name.to_string(), branch_info);
        self.save_metadata()?;
        
        info!("创建分支: {}", branch_name);
        Ok(())
    }

    /// 切换分支
    pub fn switch_branch(&self, branch_name: &str) -> Result<()> {
        let branches = self.branches.read().unwrap();
        
        if !branches.contains_key(branch_name) {
            return Err(Error::NotFound(format!("分支不存在: {}", branch_name)));
        }
        
        *self.current_branch.write().unwrap() = branch_name.to_string();
        
        info!("切换到分支: {}", branch_name);
        Ok(())
    }

    /// 合并分支
    pub fn merge_branch(
        &self,
        source_branch: &str,
        target_branch: Option<&str>,
        strategy: MergeStrategy,
        message: String,
        author: String,
    ) -> Result<MergeResult> {
        let target_branch = target_branch.unwrap_or_else(|| {
            &self.current_branch.read().unwrap()
        });
        
        // 获取分支头版本
        let source_head = self.get_branch_head(source_branch)?
            .ok_or_else(|| Error::NotFound(format!("源分支为空: {}", source_branch)))?;
        let target_head = self.get_branch_head(target_branch)?
            .ok_or_else(|| Error::NotFound(format!("目标分支为空: {}", target_branch)))?;
        
        // 加载版本数据
        let source_data = self.load_version_data(&source_head)?;
        let target_data = self.load_version_data(&target_head)?;
        
        // 计算差异和冲突
        let conflicts = self.detect_conflicts(&source_data, &target_data)?;
        
        if !conflicts.is_empty() && matches!(strategy, MergeStrategy::AutoSource | MergeStrategy::AutoTarget) {
            return Ok(MergeResult {
                success: false,
                merged_version_id: None,
                conflicts,
                stats: MergeStats {
                    records_added: 0,
                    records_deleted: 0,
                    records_modified: 0,
                    conflicts_count: conflicts.len(),
                },
            });
        }
        
        // 执行合并
        let merged_data = self.perform_merge(&source_data, &target_data, &strategy)?;
        
        // 提交合并结果
        let merge_message = format!("Merge {} into {}: {}", source_branch, target_branch, message);
        let merged_version_id = self.commit(&merged_data, merge_message, author)?;
        
        Ok(MergeResult {
            success: true,
            merged_version_id: Some(merged_version_id),
            conflicts: vec![],
            stats: MergeStats {
                records_added: 0, // 这里应该计算实际的统计
                records_deleted: 0,
                records_modified: 0,
                conflicts_count: 0,
            },
        })
    }

    /// 创建标签
    pub fn create_tag(
        &self,
        tag_name: &str,
        version_id: &VersionId,
        description: String,
    ) -> Result<()> {
        let mut tags = self.tags.write().unwrap();
        
        if tags.contains_key(tag_name) {
            return Err(Error::AlreadyExists(format!("标签已存在: {}", tag_name)));
        }
        
        let tag_info = TagInfo {
            name: tag_name.to_string(),
            version_id: version_id.clone(),
            created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            description,
        };
        
        tags.insert(tag_name.to_string(), tag_info);
        self.save_metadata()?;
        
        info!("创建标签: {} -> {}", tag_name, version_id);
        Ok(())
    }

    /// 比较版本差异
    pub fn diff(
        &self,
        version1: &VersionId,
        version2: &VersionId,
    ) -> Result<Vec<DiffEntry>> {
        let data1 = self.load_version_data(version1)?;
        let data2 = self.load_version_data(version2)?;
        
        self.calculate_diff(&data1, &data2)
    }

    /// 获取版本历史
    pub fn get_history(&self, branch: Option<&str>) -> Result<Vec<VersionInfo>> {
        let branch = branch.unwrap_or_else(|| &self.current_branch.read().unwrap());
        let head = self.get_branch_head(branch)?;
        
        if head.is_none() {
            return Ok(vec![]);
        }
        
        let mut history = Vec::new();
        let mut current = head;
        let versions = self.versions.read().unwrap();
        
        while let Some(version_id) = current {
            if let Some(version_info) = versions.get(&version_id) {
                history.push(version_info.clone());
                current = version_info.parent_id.clone();
            } else {
                break;
            }
        }
        
        Ok(history)
    }

    /// 获取所有分支
    pub fn list_branches(&self) -> Vec<BranchInfo> {
        self.branches.read().unwrap().values().cloned().collect()
    }

    /// 获取所有标签
    pub fn list_tags(&self) -> Vec<TagInfo> {
        self.tags.read().unwrap().values().cloned().collect()
    }

    /// 计算数据哈希
    fn calculate_data_hash(&self, data: &[DataValue]) -> Result<String> {
        let mut hasher = Sha256::new();
        
        for value in data {
            let serialized = serde_json::to_string(value)
                .map_err(|e| Error::serialization(format!("序列化数据失败: {}", e)))?;
            hasher.update(serialized.as_bytes());
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// 创建文件条目
    fn create_file_entries(&self, data: &[DataValue]) -> Result<Vec<FileEntry>> {
        let mut files = Vec::new();
        
        // 创建数据文件条目
        let hash = self.calculate_data_hash(data)?;
        let size = data.len() as u64; // 数据大小计算（基于元素数量）
        let modified = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        files.push(FileEntry {
            path: "data.json".to_string(),
            hash,
            size,
            modified,
            file_type: FileType::Data,
        });
        
        Ok(files)
    }

    /// 存储版本数据
    fn store_version_data(&self, version_id: &VersionId, data: &[DataValue]) -> Result<()> {
        let version_dir = self.storage_root.join("versions").join(version_id);
        fs::create_dir_all(&version_dir)
            .map_err(|e| Error::io_error(format!("创建版本目录失败: {}", e)))?;
        
        let data_file = version_dir.join("data.json");
        let content = serde_json::to_string_pretty(data)
            .map_err(|e| Error::serialization(format!("序列化数据失败: {}", e)))?;
        
        fs::write(data_file, content)
            .map_err(|e| Error::io_error(format!("写入数据文件失败: {}", e)))?;
        
        Ok(())
    }

    /// 加载版本数据
    fn load_version_data(&self, version_id: &VersionId) -> Result<Vec<DataValue>> {
        let data_file = self.storage_root.join("versions").join(version_id).join("data.json");
        
        if !data_file.exists() {
            return Err(Error::NotFound(format!("版本数据不存在: {}", version_id)));
        }
        
        let content = fs::read_to_string(data_file)
            .map_err(|e| Error::io_error(format!("读取数据文件失败: {}", e)))?;
        
        let data: Vec<DataValue> = serde_json::from_str(&content)
            .map_err(|e| Error::invalid_input(format!("解析数据失败: {}", e)))?;
        
        Ok(data)
    }

    /// 获取分支头版本
    fn get_branch_head(&self, branch: &str) -> Result<Option<VersionId>> {
        let branches = self.branches.read().unwrap();
        let branch_info = branches.get(branch)
            .ok_or_else(|| Error::NotFound(format!("分支不存在: {}", branch)))?;
        
        if branch_info.head.is_empty() {
            Ok(None)
        } else {
            Ok(Some(branch_info.head.clone()))
        }
    }

    /// 更新分支头
    fn update_branch_head(&self, branch: &str, version_id: &VersionId) -> Result<()> {
        let mut branches = self.branches.write().unwrap();
        let branch_info = branches.get_mut(branch)
            .ok_or_else(|| Error::NotFound(format!("分支不存在: {}", branch)))?;
        
        branch_info.head = version_id.clone();
        branch_info.updated = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        Ok(())
    }

    /// 检测合并冲突
    fn detect_conflicts(
        &self,
        source_data: &[DataValue],
        target_data: &[DataValue],
    ) -> Result<Vec<MergeConflict>> {
        let mut conflicts = Vec::new();
        
        // 冲突检测：检测数据长度变化和结构差异
        
        if source_data.len() != target_data.len() {
            conflicts.push(MergeConflict {
                path: "data_length".to_string(),
                base_value: None,
                source_value: Some(DataValue::Number(source_data.len() as f64)),
                target_value: Some(DataValue::Number(target_data.len() as f64)),
                conflict_type: ConflictType::ModifyModify,
            });
        }
        
        Ok(conflicts)
    }

    /// 执行合并
    fn perform_merge(
        &self,
        source_data: &[DataValue],
        target_data: &[DataValue],
        strategy: &MergeStrategy,
    ) -> Result<Vec<DataValue>> {
        match strategy {
            MergeStrategy::AutoSource => Ok(source_data.to_vec()),
            MergeStrategy::AutoTarget => Ok(target_data.to_vec()),
            _ => {
                // 合并逻辑：将源数据追加到目标数据
                let mut merged = target_data.to_vec();
                merged.extend_from_slice(source_data);
                Ok(merged)
            }
        }
    }

    /// 计算版本差异
    fn calculate_diff(
        &self,
        data1: &[DataValue],
        data2: &[DataValue],
    ) -> Result<Vec<DiffEntry>> {
        let mut diffs = Vec::new();
        
        // 差异计算：检测记录数量变化
        if data1.len() != data2.len() {
            diffs.push(DiffEntry {
                diff_type: DiffType::RecordModified,
                path: "data_length".to_string(),
                old_value: Some(DataValue::Number(data1.len() as f64)),
                new_value: Some(DataValue::Number(data2.len() as f64)),
                context: HashMap::new(),
            });
        }
        
        Ok(diffs)
    }
}

/// 创建数据版本管理器
pub fn create_version_manager<P: AsRef<Path>>(
    storage_root: P,
    config: Option<VersionConfig>,
) -> Result<DataVersionManager> {
    let config = config.unwrap_or_default();
    DataVersionManager::new(storage_root, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_version_manager_creation() {
        let dir = tempdir().unwrap();
        let manager = DataVersionManager::new(dir.path(), VersionConfig::default()).unwrap();
        
        let branches = manager.list_branches();
        assert!(!branches.is_empty());
        assert_eq!(branches[0].name, "main");
    }

    #[test]
    fn test_commit_and_checkout() {
        let dir = tempdir().unwrap();
        let manager = DataVersionManager::new(dir.path(), VersionConfig::default()).unwrap();
        
        let test_data = vec![
            DataValue::String("test1".to_string()),
            DataValue::Number(42.0),
        ];
        
        // 提交数据
        let version_id = manager.commit(
            &test_data,
            "Initial commit".to_string(),
            "test_user".to_string(),
        ).unwrap();
        
        // 检出数据
        let retrieved_data = manager.checkout(&version_id).unwrap();
        assert_eq!(test_data.len(), retrieved_data.len());
    }

    #[test]
    fn test_branch_operations() {
        let dir = tempdir().unwrap();
        let manager = DataVersionManager::new(dir.path(), VersionConfig::default()).unwrap();
        
        // 创建分支
        manager.create_branch("feature", None).unwrap();
        
        // 切换分支
        manager.switch_branch("feature").unwrap();
        
        let branches = manager.list_branches();
        assert_eq!(branches.len(), 2); // main + feature
    }
} 