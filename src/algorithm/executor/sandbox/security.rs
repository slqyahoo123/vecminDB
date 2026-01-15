use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use log::{debug, warn, error};
use chrono::{DateTime, Utc};

use crate::error::{Error, Result};
use crate::algorithm::types::{SandboxSecurityLevel, ResourceLimits};
use crate::algorithm::executor::sandbox::error::SandboxError;
use crate::algorithm::executor::sandbox::types::SecurityContext;

/// 系统调用过滤器
pub struct SyscallFilter {
    /// 允许的系统调用
    allowed_syscalls: HashSet<i32>,
    /// 是否使用默认允许列表
    use_default_allowlist: bool,
    /// 安全级别
    security_level: SandboxSecurityLevel,
}

impl SyscallFilter {
    /// 创建系统调用过滤器
    pub fn new(security_level: SandboxSecurityLevel) -> Self {
        let mut filter = Self {
            allowed_syscalls: HashSet::new(),
            use_default_allowlist: true,
            security_level,
        };
        
        filter.initialize_default_allowlist();
        filter
    }
    
    /// 初始化默认允许的系统调用列表
    fn initialize_default_allowlist(&mut self) {
        // 基本系统调用
        let base_syscalls = vec![
            0,   // read
            1,   // write
            3,   // close
            4,   // stat
            5,   // fstat
            6,   // lstat
            9,   // mmap
            10,  // mprotect
            11,  // munmap
            12,  // brk
            21,  // access
            59,  // execve (可控情况下才允许)
            60,  // exit
            61,  // wait4
            63,  // uname
            89,  // readlink
            231, // exit_group
        ];
        
        // 根据安全级别设置允许的系统调用
        match self.security_level {
            SandboxSecurityLevel::Low => {
                // 低安全级别允许大部分系统调用，但仍然限制敏感操作
                for i in 0..300 {
                    self.allowed_syscalls.insert(i);
                }
                
                // 禁止一些危险的系统调用
                let dangerous = vec![
                    41,  // socket
                    42,  // connect
                    56,  // clone
                    57,  // fork
                    58,  // vfork
                    85,  // creat
                    319, // memfd_create
                    322, // execveat
                ];
                
                for syscall in dangerous {
                    self.allowed_syscalls.remove(&syscall);
                }
            },
            SandboxSecurityLevel::Standard => {
                // 标准安全级别允许基本操作
                for syscall in base_syscalls {
                    self.allowed_syscalls.insert(syscall);
                }
                
                // 允许一些额外的系统调用
                let additional = vec![
                    13,  // rt_sigaction
                    14,  // rt_sigprocmask
                    16,  // ioctl (限制性使用)
                    25,  // mremap
                    28,  // madvise
                    35,  // nanosleep
                    39,  // getpid
                    96,  // gettimeofday
                    202, // futex
                    228, // clock_gettime
                ];
                
                for syscall in additional {
                    self.allowed_syscalls.insert(syscall);
                }
            },
            SandboxSecurityLevel::High => {
                // 高安全级别只允许最基本的操作
                for syscall in base_syscalls {
                    if syscall != 59 { // 不允许execve
                        self.allowed_syscalls.insert(syscall);
                    }
                }
            },
            SandboxSecurityLevel::Maximum => {
                // 最高安全级别极大限制系统调用
                let minimal = vec![
                    0,   // read
                    1,   // write
                    3,   // close
                    9,   // mmap
                    10,  // mprotect
                    11,  // munmap
                    60,  // exit
                    231, // exit_group
                ];
                
                for syscall in minimal {
                    self.allowed_syscalls.insert(syscall);
                }
            },
        }
    }
    
    /// 检查系统调用是否允许
    pub fn is_allowed(&self, syscall: i32) -> bool {
        self.allowed_syscalls.contains(&syscall)
    }
    
    /// 添加允许的系统调用
    pub fn allow_syscall(&mut self, syscall: i32) {
        self.allowed_syscalls.insert(syscall);
    }
    
    /// 添加多个允许的系统调用
    pub fn allow_syscalls(&mut self, syscalls: &[i32]) {
        for syscall in syscalls {
            self.allowed_syscalls.insert(*syscall);
        }
    }
    
    /// 移除允许的系统调用
    pub fn deny_syscall(&mut self, syscall: i32) {
        self.allowed_syscalls.remove(&syscall);
    }
    
    /// 获取系统调用名称
    pub fn get_syscall_name(&self, syscall: i32) -> String {
        match syscall {
            0 => "read".to_string(),
            1 => "write".to_string(),
            2 => "open".to_string(),
            3 => "close".to_string(),
            4 => "stat".to_string(),
            5 => "fstat".to_string(),
            6 => "lstat".to_string(),
            9 => "mmap".to_string(),
            10 => "mprotect".to_string(),
            11 => "munmap".to_string(),
            12 => "brk".to_string(),
            // 其他系统调用省略...
            _ => format!("unknown({})", syscall),
        }
    }
}

/// 网络访问控制
pub struct NetworkAccessControl {
    /// 是否允许网络访问
    allow_network: bool,
    /// 允许的网络域名/IP
    allowed_hosts: HashSet<String>,
    /// 允许的端口
    allowed_ports: HashSet<u16>,
    /// 允许的协议
    allowed_protocols: HashSet<String>,
    /// 是否只允许出站连接
    outbound_only: bool,
    /// 最大连接数
    max_connections: usize,
}

impl NetworkAccessControl {
    /// 创建网络访问控制
    pub fn new(security_level: SandboxSecurityLevel) -> Self {
        let mut control = Self {
            allow_network: false,
            allowed_hosts: HashSet::new(),
            allowed_ports: HashSet::new(),
            allowed_protocols: HashSet::new(),
            outbound_only: true,
            max_connections: 10,
        };
        
        // 根据安全级别设置网络访问控制
        match security_level {
            SandboxSecurityLevel::Low => {
                control.allow_network = true;
                control.outbound_only = false;
                control.max_connections = 100;
                control.allowed_protocols.insert("tcp".to_string());
                control.allowed_protocols.insert("udp".to_string());
                control.allowed_protocols.insert("http".to_string());
                control.allowed_protocols.insert("https".to_string());
            },
            SandboxSecurityLevel::Standard => {
                control.allow_network = true;
                control.outbound_only = true;
                control.max_connections = 20;
                control.allowed_protocols.insert("http".to_string());
                control.allowed_protocols.insert("https".to_string());
            },
            SandboxSecurityLevel::High => {
                control.allow_network = false;
            },
            SandboxSecurityLevel::Maximum => {
                control.allow_network = false;
            },
        }
        
        control
    }
    
    /// 检查主机是否允许访问
    pub fn is_host_allowed(&self, host: &str) -> bool {
        if !self.allow_network {
            return false;
        }
        
        if self.allowed_hosts.is_empty() {
            return true;
        }
        
        self.allowed_hosts.contains(host)
    }
    
    /// 检查端口是否允许访问
    pub fn is_port_allowed(&self, port: u16) -> bool {
        if !self.allow_network {
            return false;
        }
        
        if self.allowed_ports.is_empty() {
            return true;
        }
        
        self.allowed_ports.contains(&port)
    }
    
    /// 检查协议是否允许使用
    pub fn is_protocol_allowed(&self, protocol: &str) -> bool {
        if !self.allow_network {
            return false;
        }
        
        self.allowed_protocols.contains(protocol)
    }
    
    /// 添加允许的主机
    pub fn allow_host(&mut self, host: &str) {
        self.allowed_hosts.insert(host.to_string());
    }
    
    /// 添加允许的端口
    pub fn allow_port(&mut self, port: u16) {
        self.allowed_ports.insert(port);
    }
    
    /// 添加允许的协议
    pub fn allow_protocol(&mut self, protocol: &str) {
        self.allowed_protocols.insert(protocol.to_string());
    }
}

/// 文件系统访问控制
pub struct FileSystemAccessControl {
    /// 是否允许文件系统访问
    allow_filesystem: bool,
    /// 允许的路径
    allowed_paths: HashSet<PathBuf>,
    /// 路径映射（沙箱内路径 -> 实际路径）
    path_mapping: HashMap<PathBuf, PathBuf>,
    /// 是否只读
    read_only: bool,
    /// 最大文件大小
    max_file_size: usize,
    /// 最大总文件大小
    max_total_size: usize,
}

impl FileSystemAccessControl {
    /// 创建文件系统访问控制
    pub fn new(security_level: SandboxSecurityLevel) -> Self {
        let mut control = Self {
            allow_filesystem: false,
            allowed_paths: HashSet::new(),
            path_mapping: HashMap::new(),
            read_only: true,
            max_file_size: 10 * 1024 * 1024, // 10MB
            max_total_size: 100 * 1024 * 1024, // 100MB
        };
        
        // 根据安全级别设置文件系统访问控制
        match security_level {
            SandboxSecurityLevel::Low => {
                control.allow_filesystem = true;
                control.read_only = false;
                control.max_file_size = 100 * 1024 * 1024; // 100MB
                control.max_total_size = 1024 * 1024 * 1024; // 1GB
            },
            SandboxSecurityLevel::Standard => {
                control.allow_filesystem = true;
                control.read_only = true;
                control.max_file_size = 50 * 1024 * 1024; // 50MB
                control.max_total_size = 500 * 1024 * 1024; // 500MB
            },
            SandboxSecurityLevel::High => {
                control.allow_filesystem = false;
            },
            SandboxSecurityLevel::Maximum => {
                control.allow_filesystem = false;
            },
        }
        
        control
    }
    
    /// 检查路径是否允许访问
    pub fn is_path_allowed(&self, path: &PathBuf) -> bool {
        if !self.allow_filesystem {
            return false;
        }
        
        if self.allowed_paths.is_empty() {
            return true;
        }
        
        // 检查路径是否在允许列表中
        for allowed in &self.allowed_paths {
            if path.starts_with(allowed) {
                return true;
            }
        }
        
        false
    }
    
    /// 检查操作是否允许
    pub fn is_operation_allowed(&self, write_operation: bool) -> bool {
        if !self.allow_filesystem {
            return false;
        }
        
        if write_operation && self.read_only {
            return false;
        }
        
        true
    }
    
    /// 映射路径（从沙箱内路径到实际路径）
    pub fn map_path(&self, sandbox_path: &PathBuf) -> Option<PathBuf> {
        if !self.allow_filesystem {
            return None;
        }
        
        if let Some(real_path) = self.path_mapping.get(sandbox_path) {
            return Some(real_path.clone());
        }
        
        // 尝试部分路径映射
        for (sandbox, real) in &self.path_mapping {
            if sandbox_path.starts_with(sandbox) {
                if let Ok(relative) = sandbox_path.strip_prefix(sandbox) {
                    return Some(real.join(relative));
                }
            }
        }
        
        Some(sandbox_path.clone())
    }
    
    /// 添加允许的路径
    pub fn allow_path(&mut self, path: PathBuf) {
        self.allowed_paths.insert(path);
    }
    
    /// 添加路径映射
    pub fn add_path_mapping(&mut self, sandbox_path: PathBuf, real_path: PathBuf) {
        self.path_mapping.insert(sandbox_path, real_path);
    }
}

/// 增强型安全上下文
pub struct EnhancedSecurityContext {
    /// 基础安全上下文
    pub base_context: SecurityContext,
    /// 系统调用过滤器
    pub syscall_filter: SyscallFilter,
    /// 网络访问控制
    pub network_control: NetworkAccessControl,
    /// 文件系统访问控制
    pub filesystem_control: FileSystemAccessControl,
    /// 是否启用地址空间布局随机化(ASLR)
    pub enable_aslr: bool,
    /// 是否启用seccomp
    pub enable_seccomp: bool,
    /// 是否启用apparmor/selinux
    pub enable_lsm: bool,
    /// 是否启用cgroups资源限制
    pub enable_cgroups: bool,
    /// 是否启用命名空间隔离
    pub enable_namespaces: bool,
    /// 命名空间配置
    pub namespace_config: NamespaceConfig,
    /// 是否启用chroot
    pub enable_chroot: bool,
    /// 内存保护策略
    pub memory_protection: MemoryProtectionPolicy,
    /// 进程行为监控
    pub process_monitor: ProcessMonitor,
    /// 异常行为检测
    pub anomaly_detector: AnomalyDetector,
    /// 是否启用系统调用审计
    pub enable_syscall_auditing: bool,
    /// 是否启用写时复制保护
    pub enable_cow_protection: bool,
    /// 是否启用特权降级
    pub enable_privilege_dropping: bool,
}

/// 内存保护策略
#[derive(Debug, Clone)]
pub struct MemoryProtectionPolicy {
    /// 是否启用数据执行保护(DEP)
    pub enable_dep: bool,
    /// 是否启用堆栈保护
    pub enable_stack_protection: bool,
    /// 是否启用堆安全检查
    pub enable_heap_protection: bool,
    /// 是否启用指针混淆
    pub enable_pointer_obfuscation: bool,
    /// 内存页保护设置
    pub page_protection: HashMap<String, PageProtection>,
    /// 敏感数据区域
    pub sensitive_regions: Vec<MemoryRegion>,
}

/// 内存页保护类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageProtection {
    /// 只读
    ReadOnly,
    /// 读写
    ReadWrite,
    /// 可执行
    Executable,
    /// 不可访问
    NoAccess,
}

/// 内存区域
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// 区域名称
    pub name: String,
    /// 地址范围
    pub address_range: (usize, usize),
    /// 保护类型
    pub protection: PageProtection,
}

impl MemoryProtectionPolicy {
    /// 创建新的内存保护策略
    pub fn new(security_level: SandboxSecurityLevel) -> Self {
        let mut policy = Self {
            enable_dep: true,
            enable_stack_protection: true,
            enable_heap_protection: true,
            enable_pointer_obfuscation: false,
            page_protection: HashMap::new(),
            sensitive_regions: Vec::new(),
        };
        
        // 根据安全级别配置内存保护
        match security_level {
            SandboxSecurityLevel::Low => {
                // 基本保护
                policy.enable_pointer_obfuscation = false;
            },
            SandboxSecurityLevel::Standard => {
                // 标准保护
                policy.enable_pointer_obfuscation = true;
            },
            SandboxSecurityLevel::High | SandboxSecurityLevel::Maximum => {
                // 高级保护
                policy.enable_pointer_obfuscation = true;
                
                // 添加一些特定区域保护
                policy.sensitive_regions.push(MemoryRegion {
                    name: "config_area".to_string(),
                    address_range: (0, 0), // 运行时确定
                    protection: PageProtection::ReadOnly,
                });
            },
        }
        
        policy
    }
    
    /// 检查内存访问是否允许
    pub fn is_memory_access_allowed(&self, address: usize, is_write: bool, is_exec: bool) -> bool {
        // 检查敏感区域
        for region in &self.sensitive_regions {
            if address >= region.address_range.0 && address <= region.address_range.1 {
                match region.protection {
                    PageProtection::ReadOnly => return !is_write && !is_exec,
                    PageProtection::ReadWrite => return !is_exec,
                    PageProtection::Executable => return true,
                    PageProtection::NoAccess => return false,
                }
            }
        }
        
        // 默认允许
        true
    }
    
    /// 添加敏感内存区域
    pub fn add_sensitive_region(&mut self, region: MemoryRegion) {
        self.sensitive_regions.push(region);
    }
}

/// 进程行为监控
#[derive(Debug, Clone)]
pub struct ProcessMonitor {
    /// 是否启用监控
    pub enabled: bool,
    /// 监控间隔(毫秒)
    pub interval_ms: u64,
    /// CPU使用率阈值(百分比)
    pub cpu_threshold: f64,
    /// 内存使用率阈值(百分比)
    pub memory_threshold: f64,
    /// 文件描述符阈值
    pub fd_threshold: usize,
    /// 线程数量阈值
    pub thread_threshold: usize,
    /// 系统调用频率阈值(每秒)
    pub syscall_rate_threshold: usize,
    /// 监控回调
    pub callback: Option<Arc<dyn Fn(&ProcessStatus) + Send + Sync>>,
}

/// 进程状态
#[derive(Debug, Clone)]
pub struct ProcessStatus {
    /// 进程ID
    pub pid: u32,
    /// CPU使用率
    pub cpu_usage: f64,
    /// 内存使用
    pub memory_usage: usize,
    /// 虚拟内存大小
    pub virtual_memory_size: usize,
    /// 文件描述符数量
    pub fd_count: usize,
    /// 线程数量
    pub thread_count: usize,
    /// 运行时间(秒)
    pub run_time: f64,
    /// 系统调用计数
    pub syscall_count: HashMap<i32, usize>,
    /// 状态时间戳
    pub timestamp: DateTime<Utc>,
}

impl ProcessMonitor {
    /// 创建新的进程监控器
    pub fn new(security_level: SandboxSecurityLevel) -> Self {
        let mut monitor = Self {
            enabled: true,
            interval_ms: 1000,
            cpu_threshold: 90.0,
            memory_threshold: 90.0,
            fd_threshold: 100,
            thread_threshold: 10,
            syscall_rate_threshold: 1000,
            callback: None,
        };
        
        // 根据安全级别配置监控
        match security_level {
            SandboxSecurityLevel::Low => {
                // 低监控
                monitor.interval_ms = 5000;
                monitor.cpu_threshold = 95.0;
                monitor.memory_threshold = 95.0;
                monitor.fd_threshold = 500;
                monitor.thread_threshold = 50;
                monitor.syscall_rate_threshold = 5000;
            },
            SandboxSecurityLevel::Standard => {
                // 标准监控
                monitor.interval_ms = 1000;
                monitor.cpu_threshold = 90.0;
                monitor.memory_threshold = 90.0;
                monitor.fd_threshold = 200;
                monitor.thread_threshold = 20;
                monitor.syscall_rate_threshold = 2000;
            },
            SandboxSecurityLevel::High => {
                // 高监控
                monitor.interval_ms = 500;
                monitor.cpu_threshold = 80.0;
                monitor.memory_threshold = 80.0;
                monitor.fd_threshold = 100;
                monitor.thread_threshold = 10;
                monitor.syscall_rate_threshold = 1000;
            },
            SandboxSecurityLevel::Maximum => {
                // 最大监控
                monitor.interval_ms = 200;
                monitor.cpu_threshold = 70.0;
                monitor.memory_threshold = 70.0;
                monitor.fd_threshold = 50;
                monitor.thread_threshold = 5;
                monitor.syscall_rate_threshold = 500;
            },
        }
        
        monitor
    }
    
    /// 设置监控回调
    pub fn set_callback<F>(&mut self, callback: F)
    where
        F: Fn(&ProcessStatus) + Send + Sync + 'static,
    {
        self.callback = Some(Arc::new(callback));
    }
    
    /// 检查进程状态
    pub fn check_status(&self, status: &ProcessStatus) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        // 检查CPU使用率
        if status.cpu_usage > self.cpu_threshold {
            return Err(Error::resource_exhausted(
                format!("CPU使用率超过阈值: {:.2}% > {:.2}%", status.cpu_usage, self.cpu_threshold)
            ));
        }
        
        // 检查内存使用率
        let memory_percent = (status.memory_usage as f64 / status.virtual_memory_size as f64) * 100.0;
        if memory_percent > self.memory_threshold {
            return Err(Error::resource_exhausted(
                format!("内存使用率超过阈值: {:.2}% > {:.2}%", memory_percent, self.memory_threshold)
            ));
        }
        
        // 检查文件描述符
        if status.fd_count > self.fd_threshold {
            return Err(Error::resource_exhausted(
                format!("文件描述符数量超过阈值: {} > {}", status.fd_count, self.fd_threshold)
            ));
        }
        
        // 检查线程数量
        if status.thread_count > self.thread_threshold {
            return Err(Error::resource_exhausted(
                format!("线程数量超过阈值: {} > {}", status.thread_count, self.thread_threshold)
            ));
        }
        
        // 检查系统调用频率
        let total_syscalls: usize = status.syscall_count.values().sum();
        let syscall_rate = (total_syscalls as f64 / status.run_time) as usize;
        if syscall_rate > self.syscall_rate_threshold {
            return Err(Error::resource_exhausted(
                format!("系统调用频率超过阈值: {} > {} 每秒", syscall_rate, self.syscall_rate_threshold)
            ));
        }
        
        // 执行回调
        if let Some(callback) = &self.callback {
            callback(status);
        }
        
        Ok(())
    }
}

/// 异常行为检测器
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// 是否启用
    pub enabled: bool,
    /// 检测阈值
    pub thresholds: HashMap<String, f64>,
    /// 历史行为模式
    pub behavior_patterns: Vec<BehaviorPattern>,
    /// 异常处理策略
    pub anomaly_policy: AnomalyPolicy,
}

/// 行为模式
#[derive(Debug, Clone)]
pub struct BehaviorPattern {
    /// 模式名称
    pub name: String,
    /// 特征向量
    pub features: HashMap<String, f64>,
    /// 模式权重
    pub weight: f64,
}

/// 异常处理策略
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyPolicy {
    /// 仅记录
    LogOnly,
    /// 发出警告
    Warn,
    /// 限制资源
    Throttle,
    /// 终止进程
    Terminate,
}

impl AnomalyDetector {
    /// 创建新的异常检测器
    pub fn new(security_level: SandboxSecurityLevel) -> Self {
        let mut detector = Self {
            enabled: true,
            thresholds: HashMap::new(),
            behavior_patterns: Vec::new(),
            anomaly_policy: AnomalyPolicy::LogOnly,
        };
        
        // 设置基本阈值
        detector.thresholds.insert("syscall_entropy".to_string(), 0.8);
        detector.thresholds.insert("file_access_pattern".to_string(), 0.7);
        detector.thresholds.insert("network_activity".to_string(), 0.6);
        detector.thresholds.insert("memory_access_pattern".to_string(), 0.8);
        
        // 根据安全级别配置策略
        match security_level {
            SandboxSecurityLevel::Low => {
                detector.anomaly_policy = AnomalyPolicy::LogOnly;
            },
            SandboxSecurityLevel::Standard => {
                detector.anomaly_policy = AnomalyPolicy::Warn;
            },
            SandboxSecurityLevel::High => {
                detector.anomaly_policy = AnomalyPolicy::Throttle;
            },
            SandboxSecurityLevel::Maximum => {
                detector.anomaly_policy = AnomalyPolicy::Terminate;
            },
        }
        
        detector
    }
    
    /// 检测异常行为
    pub fn detect_anomaly(&self, features: &HashMap<String, f64>) -> Result<Option<String>> {
        if !self.enabled {
            return Ok(None);
        }
        
        // 计算异常分数
        let mut anomaly_score = 0.0;
        let mut pattern_matches = 0;
        
        for pattern in &self.behavior_patterns {
            let mut pattern_score = 0.0;
            let mut feature_count = 0;
            
            for (feature, value) in &pattern.features {
                if let Some(current) = features.get(feature) {
                    let diff = (current - value).abs();
                    pattern_score += diff;
                    feature_count += 1;
                }
            }
            
            if feature_count > 0 {
                pattern_score /= feature_count as f64;
                
                // 低分表示更好的匹配
                if pattern_score < 0.3 {
                    pattern_matches += 1;
                }
            }
        }
        
        // 如果没有匹配任何已知模式，可能是异常
        if pattern_matches == 0 && !self.behavior_patterns.is_empty() {
            anomaly_score = 1.0;
        }
        
        // 检查特定的异常特征
        for (feature, threshold) in &self.thresholds {
            if let Some(value) = features.get(feature) {
                if value > threshold {
                    anomaly_score = anomaly_score.max(*value / threshold);
                }
            }
        }
        
        // 如果异常分数超过阈值，返回异常
        if anomaly_score > 0.5 {
            let message = format!("检测到异常行为: 分数={:.2}", anomaly_score);
            
            // 根据策略处理异常
            match self.anomaly_policy {
                AnomalyPolicy::LogOnly => {
                    debug!("{}", message);
                },
                AnomalyPolicy::Warn => {
                    warn!("{}", message);
                },
                AnomalyPolicy::Throttle => {
                    warn!("{} - 资源将被限制", message);
                    // 实际的限制逻辑在调用者中实现
                },
                AnomalyPolicy::Terminate => {
                    error!("{} - 进程将被终止", message);
                    return Err(crate::algorithm::executor::sandbox::error::security_violation(
                        &format!(
                            "anomaly_detected: 检测到异常行为，安全策略要求终止: 分数={:.2}",
                            anomaly_score
                        ),
                    ));
                },
            }
            
            return Ok(Some(message));
        }
        
        Ok(None)
    }
    
    /// 添加行为模式
    pub fn add_behavior_pattern(&mut self, pattern: BehaviorPattern) {
        self.behavior_patterns.push(pattern);
    }
    
    /// 设置异常阈值
    pub fn set_threshold(&mut self, feature: &str, threshold: f64) {
        self.thresholds.insert(feature.to_string(), threshold);
    }
}

impl EnhancedSecurityContext {
    /// 创建新的增强安全上下文
    pub fn new(security_level: SandboxSecurityLevel) -> Self {
        let base_context = SecurityContext {
            security_level,
            allow_network: security_level == SandboxSecurityLevel::Low || 
                          security_level == SandboxSecurityLevel::Standard,
            allow_filesystem: security_level == SandboxSecurityLevel::Low || 
                             security_level == SandboxSecurityLevel::Standard,
            allow_syscalls: security_level == SandboxSecurityLevel::Low || 
                           security_level == SandboxSecurityLevel::Standard,
            ..Default::default()
        };
        
        Self {
            base_context,
            syscall_filter: SyscallFilter::new(security_level),
            network_control: NetworkAccessControl::new(security_level),
            filesystem_control: FileSystemAccessControl::new(security_level),
            enable_aslr: true,
            enable_seccomp: security_level != SandboxSecurityLevel::Low,
            enable_lsm: security_level == SandboxSecurityLevel::High || 
                       security_level == SandboxSecurityLevel::Maximum,
            enable_cgroups: true,
            enable_namespaces: security_level != SandboxSecurityLevel::Low,
            namespace_config: NamespaceConfig::new(security_level),
            enable_chroot: security_level == SandboxSecurityLevel::High || 
                          security_level == SandboxSecurityLevel::Maximum,
            memory_protection: MemoryProtectionPolicy::new(security_level),
            process_monitor: ProcessMonitor::new(security_level),
            anomaly_detector: AnomalyDetector::new(security_level),
            enable_syscall_auditing: match security_level {
                SandboxSecurityLevel::Low => false,
                SandboxSecurityLevel::Standard => false,
                SandboxSecurityLevel::High | SandboxSecurityLevel::Maximum => true,
            },
            enable_cow_protection: match security_level {
                SandboxSecurityLevel::Low => false,
                SandboxSecurityLevel::Standard => true,
                SandboxSecurityLevel::High | SandboxSecurityLevel::Maximum => true,
            },
            enable_privilege_dropping: match security_level {
                SandboxSecurityLevel::Low => false,
                SandboxSecurityLevel::Standard => true,
                SandboxSecurityLevel::High | SandboxSecurityLevel::Maximum => true,
            },
        }
    }
    
    /// 检查操作安全性
    pub fn check_operation_safety(&self, operation_type: &str, details: &HashMap<String, String>) -> Result<()> {
        debug!("检查操作安全性: {}, 详情: {:?}", operation_type, details);
        
        match operation_type {
            "syscall" => {
                if let Some(syscall_str) = details.get("syscall") {
                    if let Ok(syscall) = syscall_str.parse::<i32>() {
                        if !self.syscall_filter.is_allowed(syscall) {
                            let syscall_name = self.syscall_filter.get_syscall_name(syscall);
                            let msg = format!("系统调用 {} ({}) 不被允许", syscall_name, syscall);
                            error!("{}", msg);
                            return Err(SandboxError::SecurityViolation(msg).into());
                        }
                    }
                }
            },
            "network" => {
                if let Some(host) = details.get("host") {
                    if !self.network_control.is_host_allowed(host) {
                        let msg = format!("网络主机 {} 不被允许访问", host);
                        error!("{}", msg);
                        return Err(SandboxError::SecurityViolation(msg).into());
                    }
                }
                
                if let Some(port_str) = details.get("port") {
                    if let Ok(port) = port_str.parse::<u16>() {
                        if !self.network_control.is_port_allowed(port) {
                            let msg = format!("网络端口 {} 不被允许访问", port);
                            error!("{}", msg);
                            return Err(SandboxError::SecurityViolation(msg).into());
                        }
                    }
                }
                
                if let Some(protocol) = details.get("protocol") {
                    if !self.network_control.is_protocol_allowed(protocol) {
                        let msg = format!("网络协议 {} 不被允许使用", protocol);
                        error!("{}", msg);
                        return Err(SandboxError::SecurityViolation(msg).into());
                    }
                }
            },
            "filesystem" => {
                if let Some(path_str) = details.get("path") {
                    let path = PathBuf::from(path_str);
                    if !self.filesystem_control.is_path_allowed(&path) {
                        let msg = format!("文件系统路径 {} 不被允许访问", path_str);
                        error!("{}", msg);
                        return Err(SandboxError::SecurityViolation(msg).into());
                    }
                }
                
                if let Some(operation) = details.get("operation") {
                    let is_write = operation == "write" || operation == "create" || 
                                   operation == "delete" || operation == "modify";
                    if !self.filesystem_control.is_operation_allowed(is_write) {
                        let msg = format!("文件系统操作 {} 不被允许", operation);
                        error!("{}", msg);
                        return Err(SandboxError::SecurityViolation(msg).into());
                    }
                }
            },
            "memory" => {
                // 内存操作安全检查
                if let Some(addr_str) = details.get("address") {
                    if let Ok(addr) = u64::from_str_radix(addr_str.trim_start_matches("0x"), 16) {
                        // 检查内存地址是否在允许范围内
                        // 实现内存安全检查
                        if !self.memory_protection.is_memory_access_allowed(addr as usize, false, false) {
                            let msg = format!("内存地址访问违规: {:#x}", addr);
                            error!("{}", msg);
                            return Err(SandboxError::SecurityViolation(msg).into());
                        }
                        
                        // 检查地址是否在敏感区域内
                        for region in &self.memory_protection.sensitive_regions {
                            let (start, end) = region.address_range;
                            if addr as usize >= start && (addr as usize) < end {
                                let msg = format!("访问敏感内存区域: {} (地址: {:#x})", region.name, addr);
                                error!("{}", msg);
                                return Err(SandboxError::SecurityViolation(msg).into());
                            }
                        }
                    }
                }
            },
            "resource" => {
                // 资源使用安全检查
                // 可根据传入的资源使用详情进行限制
            },
            "memory_access" => {
                if let Some(address_str) = details.get("address") {
                    if let Ok(address) = address_str.parse::<usize>() {
                        let is_write = details.get("is_write").map_or(false, |v| v == "true");
                        let is_exec = details.get("is_exec").map_or(false, |v| v == "true");
                        
                        if !self.memory_protection.is_memory_access_allowed(address, is_write, is_exec) {
                            return Err(Error::permission_denied(
                                "memory_access_violation",
                                &format!("内存访问违规: 地址={:#x}, 写入={}, 执行={}", address, is_write, is_exec)
                            ));
                        }
                    }
                }
            },
            "process_status" => {
                // 构建特征向量
                let mut features = HashMap::new();
                
                for (key, value) in details {
                    if key == "cpu_usage" || key == "memory_usage" || key.ends_with("_rate") {
                        if let Ok(val) = value.parse::<f64>() {
                            features.insert(key.clone(), val);
                        }
                    }
                }
                
                // 检测异常行为
                if let Ok(Some(message)) = self.anomaly_detector.detect_anomaly(&features) {
                    warn!("安全警告: {}", message);
                    
                    // 如果策略要求，可能会返回错误
                    if self.anomaly_detector.anomaly_policy == AnomalyPolicy::Terminate {
                        return Err(crate::algorithm::executor::sandbox::error::security_violation(
                            &format!("anomaly_detected: 异常行为导致操作终止: {}", message),
                        ));
                    }
                }
            },
            _ => {
                warn!("未知的操作类型: {}", operation_type);
            }
        }
        
        Ok(())
    }
    
    /// 监控进程状态
    pub fn monitor_process(&self, status: &ProcessStatus) -> Result<()> {
        // 如果监控已启用，执行检查
        if self.process_monitor.enabled {
            self.process_monitor.check_status(status)?;
        }
        
        Ok(())
    }
    
    /// 获取安全上下文的综合安全级别评分(0-100)
    pub fn get_security_score(&self) -> u8 {
        let mut score = 0;
        
        // 基础设置
        if self.enable_seccomp { score += 10; }
        if self.enable_namespaces { score += 10; }
        if self.enable_lsm { score += 10; }
        if self.enable_cgroups { score += 5; }
        if self.enable_chroot { score += 5; }
        
        // 增强特性
        if self.memory_protection.enable_dep { score += 5; }
        if self.memory_protection.enable_stack_protection { score += 5; }
        if self.memory_protection.enable_heap_protection { score += 5; }
        if self.memory_protection.enable_pointer_obfuscation { score += 5; }
        
        // 监控和检测
        if self.process_monitor.enabled { score += 5; }
        if self.anomaly_detector.enabled { score += 5; }
        if self.enable_syscall_auditing { score += 5; }
        if self.enable_cow_protection { score += 5; }
        if self.enable_privilege_dropping { score += 5; }
        
        // 网络控制
        if !self.network_control.allow_network { score += 10; }
        
        // 确保不超过100
        score.min(100)
    }
}

/// 命名空间配置
pub struct NamespaceConfig {
    /// 是否启用PID命名空间
    pub enable_pid_ns: bool,
    /// 是否启用网络命名空间
    pub enable_net_ns: bool,
    /// 是否启用挂载命名空间
    pub enable_mount_ns: bool,
    /// 是否启用IPC命名空间
    pub enable_ipc_ns: bool,
    /// 是否启用UTS命名空间
    pub enable_uts_ns: bool,
    /// 是否启用用户命名空间
    pub enable_user_ns: bool,
}

impl NamespaceConfig {
    /// 创建新的命名空间配置
    pub fn new(security_level: SandboxSecurityLevel) -> Self {
        let mut config = Self {
            enable_pid_ns: false,
            enable_net_ns: false,
            enable_mount_ns: false,
            enable_ipc_ns: false,
            enable_uts_ns: false,
            enable_user_ns: false,
        };
        
        match security_level {
            SandboxSecurityLevel::Low => {
                config.enable_pid_ns = false;
                config.enable_net_ns = false;
                config.enable_mount_ns = true;
                config.enable_ipc_ns = false;
                config.enable_uts_ns = true;
                config.enable_user_ns = false;
            },
            SandboxSecurityLevel::Standard => {
                config.enable_pid_ns = true;
                config.enable_net_ns = true;
                config.enable_mount_ns = true;
                config.enable_ipc_ns = true;
                config.enable_uts_ns = true;
                config.enable_user_ns = false;
            },
            SandboxSecurityLevel::High | SandboxSecurityLevel::Maximum => {
                config.enable_pid_ns = true;
                config.enable_net_ns = true;
                config.enable_mount_ns = true;
                config.enable_ipc_ns = true;
                config.enable_uts_ns = true;
                config.enable_user_ns = true;
            },
        }
        
        config
    }
}

/// 为沙箱创建增强型安全上下文
pub fn create_enhanced_security_context(
    security_level: SandboxSecurityLevel,
    resource_limits: &ResourceLimits
) -> EnhancedSecurityContext {
    let mut context = EnhancedSecurityContext::new(security_level);
    
    // 根据资源限制调整安全上下文
    context.base_context.memory_limit_bytes = resource_limits.max_memory;
    context.base_context.cpu_time_limit_ms = resource_limits.max_cpu_time;
    context.base_context.disk_io_limit_bytes = Some(resource_limits.max_disk_io);
    
    // 添加常用临时目录到允许路径
    context.filesystem_control.allow_path(PathBuf::from("/tmp"));
    context.filesystem_control.allow_path(PathBuf::from("/var/tmp"));
    
    // 允许访问一些常用域名（仅在低安全级别和标准安全级别）
    if security_level == SandboxSecurityLevel::Low || security_level == SandboxSecurityLevel::Standard {
        context.network_control.allow_host("api.example.com");
        context.network_control.allow_host("cdn.example.com");
        
        // 允许常用端口
        context.network_control.allow_port(80);
        context.network_control.allow_port(443);
    }
    
    context
} 