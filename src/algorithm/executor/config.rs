use serde::{Serialize, Deserialize};
use std::collections::HashSet;
use std::time::Duration;
use std::path::PathBuf;

use crate::algorithm::types::{ResourceLimits, SandboxSecurityLevel, NetworkPolicy, FilesystemPolicy};

/// 沙箱类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxType {
    /// 本地进程 - 直接在当前进程中执行
    LocalProcess,
    /// 隔离进程 - 在独立进程中执行
    IsolatedProcess,
    /// 进程沙箱
    Process,
    /// WebAssembly沙箱
    Wasm,
    /// Docker容器沙箱
    Docker,
}

impl Default for SandboxType {
    fn default() -> Self {
        SandboxType::LocalProcess
    }
}

// NetworkPolicy 现在从 types.rs 导入

/// 只读文件系统挂载项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadOnlyFsEntry {
    pub host_path: PathBuf,
    pub guest_path: PathBuf,
}

/// 可写文件系统挂载项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritableFsEntry {
    pub host_path: PathBuf,
    pub guest_path: PathBuf,
}

// FilesystemPolicy 现在从 types.rs 导入

/// 沙箱配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// 沙箱类型
    pub sandbox_type: SandboxType,
    /// 安全级别
    pub security_level: SandboxSecurityLevel,
    /// 网络访问策略
    pub network_policy: NetworkPolicy,
    /// 文件系统访问策略
    pub filesystem_policy: FilesystemPolicy,
    /// 超时时间
    pub timeout: Duration,
    /// 额外参数
    pub extra_params: std::collections::HashMap<String, String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            sandbox_type: SandboxType::default(),
            security_level: SandboxSecurityLevel::default(),
            network_policy: NetworkPolicy::Deny,
            filesystem_policy: FilesystemPolicy::Denied,
            timeout: Duration::from_secs(30),
            extra_params: std::collections::HashMap::new(),
        }
    }
}

/// WASM安全配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSecurityConfig {
    /// 最大允许的内存页数
    pub max_memory_pages: u32,
    /// 最大允许的表大小
    pub max_table_size: u32,
    /// 最大允许的导入函数数量
    pub max_imports: usize,
    /// 最大允许的导出函数数量
    pub max_exports: usize,
    /// 允许的导入模块名称
    pub allowed_import_modules: HashSet<String>,
    /// 禁止导入函数名称
    pub forbidden_import_functions: HashSet<String>,
}

impl Default for WasmSecurityConfig {
    fn default() -> Self {
        let mut allowed_import_modules = HashSet::new();
        allowed_import_modules.insert("wasi_snapshot_preview1".to_string());
        allowed_import_modules.insert("env".to_string());
        
        let mut forbidden_import_functions = HashSet::new();
        // 禁止的系统功能
        forbidden_import_functions.insert("proc_exit".to_string());
        forbidden_import_functions.insert("process_exit".to_string());
        forbidden_import_functions.insert("exit".to_string());
        forbidden_import_functions.insert("abort".to_string());
        // 禁止的文件系统操作
        forbidden_import_functions.insert("path_open".to_string());
        forbidden_import_functions.insert("path_create_directory".to_string());
        forbidden_import_functions.insert("path_remove_directory".to_string());
        forbidden_import_functions.insert("path_unlink_file".to_string());
        
        Self {
            max_memory_pages: 1024, // 64MB
            max_table_size: 1024,
            max_imports: 100,
            max_exports: 100,
            allowed_import_modules,
            forbidden_import_functions,
        }
    }
}

/// 执行器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// 任务ID
    pub task_id: Option<String>,
    /// 沙箱配置
    pub sandbox_config: SandboxConfig,
    /// 资源限制
    pub resource_limits: ResourceLimits,
    /// 安全级别
    pub security_level: SandboxSecurityLevel,
    /// 环境变量
    pub environment_variables: std::collections::HashMap<String, String>,
    /// 超时时间（毫秒）
    pub timeout_ms: u64,
    /// 工作目录
    pub work_dir: PathBuf,
    /// 是否启用指标收集
    pub enable_metrics: bool,
    /// 是否启用调试模式
    pub debug_mode: bool,
    /// 最大并发任务数
    pub max_concurrent_tasks: Option<u32>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            task_id: None,
            sandbox_config: SandboxConfig::default(),
            resource_limits: ResourceLimits::default(),
            security_level: SandboxSecurityLevel::default(),
            environment_variables: std::collections::HashMap::new(),
            timeout_ms: 60000,
            work_dir: std::env::temp_dir().join("vecmind_executor"),
            enable_metrics: true,
            debug_mode: false,
            max_concurrent_tasks: Some(num_cpus::get() as u32),
        }
    }
}

/// 配置专用网络策略（针对配置管理的特定需求）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConfigNetworkPolicy {
    /// 拒绝所有网络访问
    Deny,
    /// 仅允许访问指定的主机列表 (包括端口)
    AllowedHosts(Vec<String>),
    /// 允许所有网络访问 (请谨慎使用)
    AllowAll,
}

impl Default for ConfigNetworkPolicy {
    fn default() -> Self {
        ConfigNetworkPolicy::Deny
    }
}

/// 配置专用文件系统策略（针对配置管理的特定需求）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConfigFilesystemPolicy {
    /// 严格只读访问
    ReadOnly,
    /// 仅允许写入临时目录
    TempWrite,
    /// 允许在配置指定的目录读写
    ConfiguredPaths(Vec<String>),
    /// 完全禁止文件系统访问
    Denied,
}

impl Default for ConfigFilesystemPolicy {
    fn default() -> Self {
        ConfigFilesystemPolicy::ReadOnly
    }
} 