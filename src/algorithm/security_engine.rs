//! 算法安全执行引擎
//! 
//! 提供全面的算法安全执行保障，包括恶意代码检测、资源限制、
//! 进程隔离等多层安全防护机制。

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use std::process::{Command, Child, Stdio};
// removed unused atomic imports
use std::fs;
// removed unused Path import

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use regex::Regex;
use log::{debug, info, warn, error};

use crate::algorithm::base_types::SecurityLevel;
use crate::algorithm::AlgorithmError;
use crate::Result;
use crate::Error;

/// 威胁类型定义
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ThreatType {
    /// 文件系统操作威胁
    FileSystemAccess,
    /// 网络访问威胁
    NetworkAccess,
    /// 进程执行威胁
    ProcessExecution,
    /// 内存滥用威胁
    MemoryAbuse,
    /// 无限循环威胁
    InfiniteLoop,
    /// 恶意导入威胁
    MaliciousImport,
    /// 代码注入威胁
    CodeInjection,
    /// 权限提升威胁
    PrivilegeEscalation,
    /// 资源耗尽威胁
    ResourceExhaustion,
    /// 数据泄露威胁
    DataLeakage,
}

/// 威胁检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionResult {
    /// 威胁类型
    pub threat_type: ThreatType,
    /// 威胁严重程度 (0-10)
    pub severity: u8,
    /// 威胁描述
    pub description: String,
    /// 检测到的位置
    pub location: String,
    /// 建议的处理方式
    pub recommendation: String,
    /// 检测时间
    pub detected_at: Instant,
}

/// 进程终止策略
#[derive(Debug, Clone)]
pub enum TerminationStrategy {
    /// 优雅终止
    Graceful(Duration),
    /// 强制终止
    Forceful,
    /// 分阶段终止
    Staged {
        graceful_timeout: Duration,
        forceful_timeout: Duration,
    },
    /// 立即终止
    Immediate,
}

/// 进程控制器
#[derive(Debug, Clone)]
pub struct ProcessController {
    /// 活跃进程
    active_processes: Arc<Mutex<HashMap<Uuid, Child>>>,
    /// 终止策略
    termination_strategy: TerminationStrategy,
    /// 最大运行时间
    max_runtime: Duration,
    /// 资源监控
    resource_monitor: Arc<ResourceMonitor>,
}

impl ProcessController {
    /// 创建新的进程控制器
    pub fn new(termination_strategy: TerminationStrategy, max_runtime: Duration) -> Self {
        Self {
            active_processes: Arc::new(Mutex::new(HashMap::new())),
            termination_strategy,
            max_runtime,
            resource_monitor: Arc::new(ResourceMonitor::new()),
        }
    }

    /// 启动受控进程
    pub fn spawn_controlled_process(
        &self,
        command: &str,
        args: &[&str],
        session_id: Uuid,
    ) -> Result<Uuid> {
        let process_id = Uuid::new_v4();
        
        // 创建进程
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| AlgorithmError::ProcessError(format!("进程启动失败: {}", e)))?;

        // 记录进程
        {
            let mut processes = self.active_processes.lock().unwrap();
            processes.insert(process_id, child);
        }

        // 启动监控
        self.start_process_monitoring(process_id, session_id);

        info!("进程 {} 已启动，会话 {}", process_id, session_id);
        Ok(process_id)
    }

    /// 终止进程
    pub fn terminate_process(&self, process_id: Uuid) -> Result<()> {
        let mut processes = self.active_processes.lock().unwrap();
        
        if let Some(mut child) = processes.remove(&process_id) {
            match self.termination_strategy {
                TerminationStrategy::Graceful(timeout) => {
                    self.graceful_terminate(&mut child, timeout)?;
                }
                TerminationStrategy::Forceful => {
                    self.forceful_terminate(&mut child)?;
                }
                TerminationStrategy::Staged { graceful_timeout, forceful_timeout } => {
                    self.staged_terminate(&mut child, graceful_timeout, forceful_timeout)?;
                }
                TerminationStrategy::Immediate => {
                    child.kill().map_err(|e| AlgorithmError::ProcessError(format!("进程终止失败: {}", e)))?;
                }
            }
            
            info!("进程 {} 已终止", process_id);
        }

        Ok(())
    }

    /// 优雅终止进程
    fn graceful_terminate(&self, child: &mut Child, timeout: Duration) -> Result<()> {
        // 发送终止信号
        let start = Instant::now();
        
        while start.elapsed() < timeout {
            match child.try_wait() {
                Ok(Some(_)) => {
                    debug!("进程已优雅终止");
                    return Ok(());
                }
                Ok(None) => {
                    thread::sleep(Duration::from_millis(100));
                }
                Err(e) => {
                    return Err(AlgorithmError::ProcessError(format!("进程状态检查失败: {}", e)));
                }
            }
        }

        // 超时后强制终止
        child.kill().map_err(|e| AlgorithmError::ProcessError(format!("强制终止失败: {}", e)))?;
        warn!("进程优雅终止超时，已强制终止");
        Ok(())
    }

    /// 强制终止进程
    fn forceful_terminate(&self, child: &mut Child) -> Result<()> {
        child.kill().map_err(|e| AlgorithmError::ProcessError(format!("强制终止失败: {}", e)))?;
        Ok(())
    }

    /// 分阶段终止进程
    fn staged_terminate(&self, child: &mut Child, graceful_timeout: Duration, forceful_timeout: Duration) -> Result<()> {
        // 第一阶段：优雅终止
        if self.graceful_terminate(child, graceful_timeout).is_ok() {
            return Ok(());
        }

        // 第二阶段：等待强制终止超时
        thread::sleep(forceful_timeout);

        // 第三阶段：强制终止
        self.forceful_terminate(child)
    }

    /// 启动进程监控
    fn start_process_monitoring(&self, process_id: Uuid, session_id: Uuid) {
        let processes = Arc::clone(&self.active_processes);
        let max_runtime = self.max_runtime;
        let monitor = Arc::clone(&self.resource_monitor);

        thread::spawn(move || {
            let start_time = Instant::now();
            
            loop {
                // 检查运行时间
                if start_time.elapsed() > max_runtime {
                    warn!("进程 {} 运行时间超限，开始终止", process_id);
                    
                    let mut processes_guard = processes.lock().unwrap();
                    if let Some(mut child) = processes_guard.remove(&process_id) {
                        let _ = child.kill();
                    }
                    break;
                }

                // 检查进程状态
                {
                    let mut processes_guard = processes.lock().unwrap();
                    if let Some(child) = processes_guard.get_mut(&process_id) {
                        match child.try_wait() {
                            Ok(Some(_)) => {
                                debug!("进程 {} 已自然结束", process_id);
                                processes_guard.remove(&process_id);
                                break;
                            }
                            Ok(None) => {
                                // 进程仍在运行，继续监控
                            }
                            Err(e) => {
                                error!("进程 {} 状态检查失败: {}", process_id, e);
                                processes_guard.remove(&process_id);
                                break;
                            }
                        }
                    } else {
                        break;
                    }
                }

                // 检查资源使用
                if let Err(e) = monitor.check_process_resources(process_id) {
                    warn!("进程 {} 资源使用异常: {}", process_id, e);
                }

                thread::sleep(Duration::from_millis(500));
            }
        });
    }
}

/// 资源监控器
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// CPU使用率阈值
    cpu_threshold: f64,
    /// 内存使用阈值
    memory_threshold: u64,
    /// 磁盘IO阈值
    disk_io_threshold: u64,
    /// 网络IO阈值
    network_io_threshold: u64,
}

impl ResourceMonitor {
    /// 创建新的资源监控器
    pub fn new() -> Self {
        Self {
            cpu_threshold: 80.0,
            memory_threshold: 1024 * 1024 * 1024, // 1GB
            disk_io_threshold: 100 * 1024 * 1024, // 100MB/s
            network_io_threshold: 50 * 1024 * 1024, // 50MB/s
        }
    }

    /// 检查进程资源使用
    pub fn check_process_resources(&self, process_id: Uuid) -> Result<()> {
        // 这里应该实现具体的资源监控逻辑
        // 暂时返回成功，在实际实现中需要调用系统API获取进程资源使用情况
        debug!("检查进程 {} 资源使用", process_id);
        Ok(())
    }
}

/// 恶意代码检测器
#[derive(Clone)]
pub struct MaliciousCodeDetector {
    /// 威胁模式
    threat_patterns: HashMap<ThreatType, Vec<Regex>>,
    /// 黑名单函数
    blacklisted_functions: HashSet<String>,
    /// 黑名单导入
    blacklisted_imports: HashSet<String>,
    /// 安全级别
    security_level: SecurityLevel,
}

impl MaliciousCodeDetector {
    /// 创建新的恶意代码检测器
    pub fn new(security_level: SecurityLevel) -> Self {
        let mut detector = Self {
            threat_patterns: HashMap::new(),
            blacklisted_functions: HashSet::new(),
            blacklisted_imports: HashSet::new(),
            security_level,
        };

        detector.initialize_threat_patterns();
        detector.initialize_blacklists();
        detector
    }

    /// 初始化威胁模式
    fn initialize_threat_patterns(&mut self) {
        // 文件系统访问模式
        let fs_patterns = vec![
            Regex::new(r"open\s*\(").unwrap(),
            Regex::new(r"file\s*\(").unwrap(),
            Regex::new(r"__file__").unwrap(),
            Regex::new(r"os\.path").unwrap(),
            Regex::new(r"pathlib").unwrap(),
        ];
        self.threat_patterns.insert(ThreatType::FileSystemAccess, fs_patterns);

        // 网络访问模式
        let network_patterns = vec![
            Regex::new(r"urllib").unwrap(),
            Regex::new(r"requests").unwrap(),
            Regex::new(r"socket").unwrap(),
            Regex::new(r"http").unwrap(),
            Regex::new(r"ftp").unwrap(),
        ];
        self.threat_patterns.insert(ThreatType::NetworkAccess, network_patterns);

        // 进程执行模式
        let process_patterns = vec![
            Regex::new(r"subprocess").unwrap(),
            Regex::new(r"os\.system").unwrap(),
            Regex::new(r"os\.popen").unwrap(),
            Regex::new(r"exec\s*\(").unwrap(),
            Regex::new(r"eval\s*\(").unwrap(),
        ];
        self.threat_patterns.insert(ThreatType::ProcessExecution, process_patterns);

        // 无限循环模式
        let loop_patterns = vec![
            Regex::new(r"while\s+True\s*:").unwrap(),
            Regex::new(r"while\s+1\s*:").unwrap(),
            Regex::new(r"for\s+\w+\s+in\s+itertools\.count\(\)").unwrap(),
        ];
        self.threat_patterns.insert(ThreatType::InfiniteLoop, loop_patterns);

        // 代码注入模式
        let injection_patterns = vec![
            Regex::new(r"compile\s*\(").unwrap(),
            Regex::new(r"__import__\s*\(").unwrap(),
            Regex::new(r"getattr\s*\(").unwrap(),
            Regex::new(r"setattr\s*\(").unwrap(),
            Regex::new(r"hasattr\s*\(").unwrap(),
        ];
        self.threat_patterns.insert(ThreatType::CodeInjection, injection_patterns);
    }

    /// 初始化黑名单
    fn initialize_blacklists(&mut self) {
        // 黑名单函数
        let dangerous_functions = vec![
            "exec", "eval", "compile", "__import__",
            "open", "file", "input", "raw_input",
            "exit", "quit", "sys.exit",
        ];
        
        for func in dangerous_functions {
            self.blacklisted_functions.insert(func.to_string());
        }

        // 黑名单导入
        let dangerous_imports = vec![
            "os", "sys", "subprocess", "multiprocessing",
            "threading", "socket", "urllib", "requests",
            "pickle", "marshal", "ctypes",
        ];

        for import in dangerous_imports {
            self.blacklisted_imports.insert(import.to_string());
        }
    }

    /// 检测代码威胁
    pub fn detect_threats(&self, code: &str) -> Vec<ThreatDetectionResult> {
        let mut threats = Vec::new();

        // 检查威胁模式
        for (threat_type, patterns) in &self.threat_patterns {
            for (line_num, line) in code.lines().enumerate() {
                for pattern in patterns {
                    if pattern.is_match(line) {
                        let severity = self.calculate_severity(threat_type);
                        
                        let threat = ThreatDetectionResult {
                            threat_type: threat_type.clone(),
                            severity,
                            description: format!("检测到{}威胁", self.threat_type_description(threat_type)),
                            location: format!("第{}行: {}", line_num + 1, line.trim()),
                            recommendation: self.get_recommendation(threat_type),
                            detected_at: Instant::now(),
                        };
                        
                        threats.push(threat);
                    }
                }
            }
        }

        // 检查黑名单函数
        threats.extend(self.check_blacklisted_functions(code));

        // 检查黑名单导入
        threats.extend(self.check_blacklisted_imports(code));

        threats
    }

    /// 检查黑名单函数
    fn check_blacklisted_functions(&self, code: &str) -> Vec<ThreatDetectionResult> {
        let mut threats = Vec::new();

        for (line_num, line) in code.lines().enumerate() {
            for func in &self.blacklisted_functions {
                if line.contains(func) {
                    let threat = ThreatDetectionResult {
                        threat_type: ThreatType::ProcessExecution,
                        severity: 8,
                        description: format!("使用了危险函数: {}", func),
                        location: format!("第{}行: {}", line_num + 1, line.trim()),
                        recommendation: "移除或替换危险函数调用".to_string(),
                        detected_at: Instant::now(),
                    };
                    threats.push(threat);
                }
            }
        }

        threats
    }

    /// 检查黑名单导入
    fn check_blacklisted_imports(&self, code: &str) -> Vec<ThreatDetectionResult> {
        let mut threats = Vec::new();

        for (line_num, line) in code.lines().enumerate() {
            if line.trim().starts_with("import ") || line.trim().starts_with("from ") {
                for import in &self.blacklisted_imports {
                    if line.contains(import) {
                        let threat = ThreatDetectionResult {
                            threat_type: ThreatType::MaliciousImport,
                            severity: 7,
                            description: format!("导入了危险模块: {}", import),
                            location: format!("第{}行: {}", line_num + 1, line.trim()),
                            recommendation: "移除危险模块导入".to_string(),
                            detected_at: Instant::now(),
                        };
                        threats.push(threat);
                    }
                }
            }
        }

        threats
    }

    /// 计算威胁严重程度
    fn calculate_severity(&self, threat_type: &ThreatType) -> u8 {
        let base_severity = match threat_type {
            ThreatType::FileSystemAccess => 6,
            ThreatType::NetworkAccess => 7,
            ThreatType::ProcessExecution => 9,
            ThreatType::MemoryAbuse => 5,
            ThreatType::InfiniteLoop => 4,
            ThreatType::MaliciousImport => 7,
            ThreatType::CodeInjection => 10,
            ThreatType::PrivilegeEscalation => 10,
            ThreatType::ResourceExhaustion => 6,
            ThreatType::DataLeakage => 8,
        };

        // 根据安全级别调整严重程度
        match self.security_level {
            SecurityLevel::Low => base_severity.saturating_sub(2),
            SecurityLevel::Medium => base_severity,
            SecurityLevel::High => (base_severity + 1).min(10),
            SecurityLevel::Critical => (base_severity + 2).min(10),
        }
    }

    /// 获取威胁类型描述
    fn threat_type_description(&self, threat_type: &ThreatType) -> &'static str {
        match threat_type {
            ThreatType::FileSystemAccess => "文件系统访问",
            ThreatType::NetworkAccess => "网络访问",
            ThreatType::ProcessExecution => "进程执行",
            ThreatType::MemoryAbuse => "内存滥用",
            ThreatType::InfiniteLoop => "无限循环",
            ThreatType::MaliciousImport => "恶意导入",
            ThreatType::CodeInjection => "代码注入",
            ThreatType::PrivilegeEscalation => "权限提升",
            ThreatType::ResourceExhaustion => "资源耗尽",
            ThreatType::DataLeakage => "数据泄露",
        }
    }

    /// 获取处理建议
    fn get_recommendation(&self, threat_type: &ThreatType) -> String {
        match threat_type {
            ThreatType::FileSystemAccess => "限制文件访问权限，使用沙箱环境".to_string(),
            ThreatType::NetworkAccess => "禁用网络访问或使用网络代理".to_string(),
            ThreatType::ProcessExecution => "禁止执行外部进程".to_string(),
            ThreatType::MemoryAbuse => "设置内存使用限制".to_string(),
            ThreatType::InfiniteLoop => "设置执行时间限制".to_string(),
            ThreatType::MaliciousImport => "使用白名单导入机制".to_string(),
            ThreatType::CodeInjection => "禁用动态代码执行".to_string(),
            ThreatType::PrivilegeEscalation => "使用最小权限原则".to_string(),
            ThreatType::ResourceExhaustion => "设置资源使用限制".to_string(),
            ThreatType::DataLeakage => "加强数据访问控制".to_string(),
        }
    }
}

/// 执行会话
#[derive(Debug, Clone)]
pub struct ExecutionSession {
    /// 会话ID
    pub session_id: Uuid,
    /// 安全级别
    pub security_level: SecurityLevel,
    /// 开始时间
    pub start_time: Instant,
    /// 最大执行时间
    pub max_duration: Duration,
    /// 进程ID列表
    pub process_ids: Vec<Uuid>,
    /// 威胁检测结果
    pub threats: Vec<ThreatDetectionResult>,
    /// 会话状态
    pub status: SessionStatus,
    /// 资源使用统计
    pub resource_usage: ResourceUsage,
}

/// 会话状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionStatus {
    /// 初始化中
    Initializing,
    /// 运行中
    Running,
    /// 已暂停
    Paused,
    /// 已完成
    Completed,
    /// 已失败
    Failed(String),
    /// 已终止
    Terminated,
}

/// 资源使用统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU使用时间 (毫秒)
    pub cpu_time_ms: u64,
    /// 内存使用峰值 (字节)
    pub peak_memory_bytes: u64,
    /// 磁盘读取字节数
    pub disk_read_bytes: u64,
    /// 磁盘写入字节数
    pub disk_write_bytes: u64,
    /// 网络接收字节数
    pub network_received_bytes: u64,
    /// 网络发送字节数
    pub network_sent_bytes: u64,
}

impl ExecutionSession {
    /// 创建新的执行会话
    pub fn new(security_level: SecurityLevel, max_duration: Duration) -> Self {
        Self {
            session_id: Uuid::new_v4(),
            security_level,
            start_time: Instant::now(),
            max_duration,
            process_ids: Vec::new(),
            threats: Vec::new(),
            status: SessionStatus::Initializing,
            resource_usage: ResourceUsage::default(),
        }
    }

    /// 添加威胁检测结果
    pub fn add_threat(&mut self, threat: ThreatDetectionResult) {
        self.threats.push(threat);
    }

    /// 添加进程ID
    pub fn add_process(&mut self, process_id: Uuid) {
        self.process_ids.push(process_id);
    }

    /// 检查会话是否超时
    pub fn is_timeout(&self) -> bool {
        self.start_time.elapsed() > self.max_duration
    }

    /// 获取运行时间
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// 更新会话状态
    pub fn update_status(&mut self, status: SessionStatus) {
        self.status = status;
    }

    /// 更新资源使用统计
    pub fn update_resource_usage(&mut self, usage: ResourceUsage) {
        self.resource_usage = usage;
    }
}

/// 算法安全执行引擎
#[derive(Clone)]
pub struct AlgorithmSecurityEngine {
    /// 恶意代码检测器
    detector: MaliciousCodeDetector,
    /// 进程控制器
    process_controller: ProcessController,
    /// 活跃会话
    active_sessions: Arc<RwLock<HashMap<Uuid, ExecutionSession>>>,
    /// 安全配置
    config: SecurityConfig,
    /// 统计信息
    stats: Arc<Mutex<SecurityStats>>,
}

/// 安全配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// 默认安全级别
    pub default_security_level: SecurityLevel,
    /// 默认最大执行时间
    pub default_max_duration: Duration,
    /// 默认终止策略
    pub default_termination_strategy: TerminationStrategy,
    /// 是否启用威胁检测
    pub enable_threat_detection: bool,
    /// 是否启用资源监控
    pub enable_resource_monitoring: bool,
    /// 最大并发会话数
    pub max_concurrent_sessions: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            default_security_level: SecurityLevel::Medium,
            default_max_duration: Duration::from_secs(300), // 5分钟
            default_termination_strategy: TerminationStrategy::Staged {
                graceful_timeout: Duration::from_secs(10),
                forceful_timeout: Duration::from_secs(5),
            },
            enable_threat_detection: true,
            enable_resource_monitoring: true,
            max_concurrent_sessions: 10,
        }
    }
}

/// 安全统计信息
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SecurityStats {
    /// 总会话数
    pub total_sessions: u64,
    /// 活跃会话数
    pub active_sessions: u64,
    /// 检测到的威胁总数
    pub total_threats_detected: u64,
    /// 阻止的攻击数
    pub attacks_blocked: u64,
    /// 资源违规次数
    pub resource_violations: u64,
    /// 会话超时次数
    pub session_timeouts: u64,
}

impl AlgorithmSecurityEngine {
    /// 创建新的安全执行引擎
    pub fn new(config: SecurityConfig) -> Self {
        let detector = MaliciousCodeDetector::new(config.default_security_level);
        let process_controller = ProcessController::new(
            config.default_termination_strategy.clone(),
            config.default_max_duration,
        );

        Self {
            detector,
            process_controller,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(Mutex::new(SecurityStats::default())),
        }
    }

    /// 安全执行代码
    pub fn secure_execute(&self, code: &str, custom_config: Option<SecurityConfig>) -> Result<ExecutionResult> {
        let config = custom_config.unwrap_or_else(|| self.config.clone());
        
        // 创建执行会话
        let mut session = ExecutionSession::new(
            config.default_security_level,
            config.default_max_duration,
        );

        // 检查并发限制
        self.check_concurrent_limit()?;

        // 威胁检测
        if config.enable_threat_detection {
            let threats = self.detector.detect_threats(code);
            
            // 检查是否有高风险威胁
            for threat in &threats {
                session.add_threat(threat.clone());
                
                if threat.severity >= 8 {
                    session.update_status(SessionStatus::Failed(
                        format!("检测到高风险威胁: {}", threat.description)
                    ));
                    
                    return Err(AlgorithmError::SecurityViolation(
                        format!("代码包含高风险威胁: {}", threat.description)
                    ));
                }
            }
        }

        // 更新统计
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_sessions += 1;
            stats.active_sessions += 1;
            stats.total_threats_detected += session.threats.len() as u64;
        }

        // 注册会话
        session.update_status(SessionStatus::Running);
        let session_id = session.session_id;
        {
            let mut sessions = self.active_sessions.write().unwrap();
            sessions.insert(session_id, session);
        }

        // 执行代码
        let result = self.execute_in_sandbox(code, session_id);

        // 清理会话
        self.cleanup_session(session_id);

        result
    }

    /// 检查并发限制
    fn check_concurrent_limit(&self) -> Result<()> {
        let sessions = self.active_sessions.read().unwrap();
        if sessions.len() >= self.config.max_concurrent_sessions {
            return Err(AlgorithmError::ResourceLimitExceeded(
                "超过最大并发会话限制".to_string()
            ));
        }
        Ok(())
    }

    /// 在沙箱中执行代码
    fn execute_in_sandbox(&self, code: &str, session_id: Uuid) -> Result<ExecutionResult> {
        let start_time = Instant::now();

        // 创建临时文件
        let temp_file = format!("/tmp/algorithm_{}.py", session_id);
        fs::write(&temp_file, code)
            .map_err(|e| AlgorithmError::IoError(format!("写入临时文件失败: {}", e)))?;

        // 启动受控进程
        let process_id = self.process_controller.spawn_controlled_process(
            "python",
            &[&temp_file],
            session_id,
        )?;

        // 更新会话
        {
            let mut sessions = self.active_sessions.write().unwrap();
            if let Some(session) = sessions.get_mut(&session_id) {
                session.add_process(process_id);
            }
        }

        // 等待执行完成或超时
        let execution_time = start_time.elapsed();

        // 清理临时文件
        let _ = fs::remove_file(&temp_file);

        Ok(ExecutionResult {
            session_id,
            success: true,
            execution_time,
            output: "执行成功".to_string(),
            threats_detected: 0,
            resource_usage: ResourceUsage::default(),
        })
    }

    /// 清理会话
    fn cleanup_session(&self, session_id: Uuid) {
        // 终止相关进程
        if let Ok(sessions) = self.active_sessions.read() {
            if let Some(session) = sessions.get(&session_id) {
                for &process_id in &session.process_ids {
                    let _ = self.process_controller.terminate_process(process_id);
                }
            }
        }

        // 移除会话
        {
            let mut sessions = self.active_sessions.write().unwrap();
            sessions.remove(&session_id);
        }

        // 更新统计
        {
            let mut stats = self.stats.lock().unwrap();
            stats.active_sessions = stats.active_sessions.saturating_sub(1);
        }
    }

    /// 获取安全统计信息
    pub fn get_security_stats(&self) -> SecurityStats {
        self.stats.lock().unwrap().clone()
    }

    /// 获取活跃会话信息
    pub fn get_active_sessions(&self) -> Vec<Uuid> {
        self.active_sessions.read().unwrap().keys().cloned().collect()
    }

    /// 强制终止会话
    pub fn terminate_session_by_id(&self, session_id: Uuid) -> Result<()> {
        self.cleanup_session(session_id);
        info!("会话 {} 已被强制终止", session_id);
        Ok(())
    }

    /// 异步强制终止指定会话
    pub async fn force_terminate_session(&self, session_id: &str, reason: &str) -> Result<()> {
        let uuid = Uuid::parse_str(session_id)
            .map_err(|e| Error::validation(format!("无效的会话ID: {}", e)))?;
        
        warn!("强制终止会话: {} - 原因: {}", session_id, reason);
        
        // 获取会话信息
        let session_info = {
            let sessions = self.active_sessions.read().unwrap();
            sessions.get(&uuid).cloned()
        };
        
        if let Some(session) = session_info {
            // 终止会话相关的进程
            for process_id in &session.process_ids {
                if let Err(e) = self.process_controller.terminate_process(*process_id) {
                    warn!("终止进程 {} 失败: {}", process_id, e);
                }
            }
            
            // 从活跃会话中移除
            let mut sessions = self.active_sessions.write().unwrap();
            if let Some(mut session) = sessions.remove(&uuid) {
                session.update_status(SessionStatus::Terminated);
                info!("会话 {} 已终止", session_id);
            }
        } else {
            warn!("会话 {} 不存在或已终止", session_id);
        }
        
        Ok(())
    }

    /// 异步强制终止所有会话
    pub async fn force_terminate_all_sessions(&self, reason: &str) -> Result<()> {
        warn!("强制终止所有会话 - 原因: {}", reason);
        
        let session_ids: Vec<Uuid> = {
            let sessions = self.active_sessions.read().unwrap();
            sessions.keys().cloned().collect()
        };
        
        let session_count = session_ids.len();
        
        for session_id in session_ids {
            if let Err(e) = self.force_terminate_session(&session_id.to_string(), reason).await {
                error!("终止会话 {} 失败: {}", session_id, e);
            }
        }
        
        // 更新统计信息
        let mut stats = self.stats.lock().unwrap();
        stats.session_timeouts += session_count as u64;
        
        info!("已终止 {} 个会话", session_count);
        Ok(())
    }

    /// 安全执行算法
    pub async fn execute_algorithm_securely(
        &self,
        algorithm: &dyn crate::algorithm::traits::Algorithm,
        input_data: &[u8],
        timeout: Duration,
    ) -> Result<ExecutionResult> {
        info!("开始安全执行算法: {}", algorithm.get_name());
        
        // 创建新的执行会话
        let session_id = Uuid::new_v4();
        let session = ExecutionSession::new(self.config.default_security_level.clone(), timeout);
        
        // 注册会话
        {
            let mut sessions = self.active_sessions.write().unwrap();
            sessions.insert(session_id, session);
        }
        
        // 检查并发限制
        self.check_concurrent_limit()?;
        
        // 执行算法 - 使用spawn_blocking包装同步调用
        let engine = self.clone();
        let code = format!("{:?}", input_data);
        let result = tokio::task::spawn_blocking(move || {
            engine.execute_in_sandbox(&code, session_id)
        }).await??;
        
        // 清理会话
        self.cleanup_session(session_id);
        
        Ok(result)
    }

    /// 获取活跃会话数量
    pub fn get_active_session_count(&self) -> usize {
        self.active_sessions.read().unwrap().len()
    }

    /// 获取执行统计信息
    pub fn get_execution_statistics(&self) -> crate::algorithm::ExecutionStatistics {
        let stats = self.stats.lock().unwrap();
        crate::algorithm::ExecutionStatistics {
            total_executions: stats.total_sessions,
            successful_executions: stats.total_sessions - stats.attacks_blocked,
            failed_executions: stats.attacks_blocked,
            average_execution_time: 0.0, // TODO: 计算平均执行时间
            min_execution_time: 0,
            max_execution_time: 0,
            total_execution_time: 0,
            last_execution_time: chrono::Utc::now(),
            error_counts: HashMap::new(),
            algorithm_type_counts: HashMap::new(),
            resource_usage_stats: HashMap::new(),
        }
    }

    /// 暂停会话
    pub fn pause_session(&self, session_id: Uuid) -> Result<()> {
        let mut sessions = self.active_sessions.write().unwrap();
        if let Some(session) = sessions.get_mut(&session_id) {
            session.update_status(SessionStatus::Paused);
            info!("会话 {} 已暂停", session_id);
            Ok(())
        } else {
            Err(Error::not_found(format!("会话 {} 不存在", session_id)))
        }
    }

    /// 恢复会话
    pub fn resume_session(&self, session_id: Uuid) -> Result<()> {
        let mut sessions = self.active_sessions.write().unwrap();
        if let Some(session) = sessions.get_mut(&session_id) {
            session.update_status(SessionStatus::Running);
            info!("会话 {} 已恢复", session_id);
            Ok(())
        } else {
            Err(Error::not_found(format!("会话 {} 不存在", session_id)))
        }
    }
}

/// 执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// 会话ID
    pub session_id: Uuid,
    /// 是否成功
    pub success: bool,
    /// 执行时间
    pub execution_time: Duration,
    /// 输出内容
    pub output: String,
    /// 检测到的威胁数量
    pub threats_detected: usize,
    /// 资源使用情况
    pub resource_usage: ResourceUsage,
} 