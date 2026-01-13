use crate::error::{Error, Result};
use crate::algorithm::types::{Algorithm, AlgorithmType, SandboxSecurityLevel, ResourceLimits};
use crate::algorithm::executor::config::{SandboxConfig, WasmSecurityConfig};
use crate::algorithm::validator::{AlgorithmValidator, ValidationReport};
use std::collections::{HashMap, HashSet};
use std::sync::{RwLock, Mutex};
use log::{info, warn, error, debug};
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// 安全策略分级
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityPolicyLevel {
    /// 低安全级别(生产环境隔离使用)
    Low,
    /// 标准安全级别(常规生产环境)
    Standard,
    /// 高安全级别(敏感环境)
    High,
    /// 极高安全级别(金融/医疗等特殊场景)
    Strict,
}

impl Default for SecurityPolicyLevel {
    fn default() -> Self {
        Self::Standard
    }
}

/// 威胁级别枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreatLevel {
    /// 最小威胁
    Minimal,
    /// 低级威胁
    Low,
    /// 中等威胁
    Medium,
    /// 高级威胁
    High,
    /// 严重威胁
    Critical,
}

impl Default for ThreatLevel {
    fn default() -> Self {
        Self::Minimal
    }
}

impl From<u32> for ThreatLevel {
    fn from(score: u32) -> Self {
        match score {
            0..=10 => ThreatLevel::Minimal,
            11..=30 => ThreatLevel::Low,
            31..=60 => ThreatLevel::Medium,
            61..=85 => ThreatLevel::High,
            _ => ThreatLevel::Critical,
        }
    }
}

impl From<ThreatLevel> for SecurityPolicyLevel {
    fn from(threat: ThreatLevel) -> Self {
        match threat {
            ThreatLevel::Minimal => SecurityPolicyLevel::Low,
            ThreatLevel::Low => SecurityPolicyLevel::Standard,
            ThreatLevel::Medium => SecurityPolicyLevel::Standard,
            ThreatLevel::High => SecurityPolicyLevel::High,
            ThreatLevel::Critical => SecurityPolicyLevel::Strict,
        }
    }
}

impl ThreatLevel {
    /// 获取威胁级别的数值分数
    pub fn score(&self) -> u32 {
        match self {
            ThreatLevel::Minimal => 5,
            ThreatLevel::Low => 20,
            ThreatLevel::Medium => 45,
            ThreatLevel::High => 75,
            ThreatLevel::Critical => 95,
        }
    }
    
    /// 获取威胁级别的描述
    pub fn description(&self) -> &'static str {
        match self {
            ThreatLevel::Minimal => "最小威胁 - 可信算法或基础操作",
            ThreatLevel::Low => "低级威胁 - 标准算法执行",
            ThreatLevel::Medium => "中等威胁 - 复杂算法或有限资源访问",
            ThreatLevel::High => "高级威胁 - 网络访问或文件系统操作",
            ThreatLevel::Critical => "严重威胁 - 系统级操作或恶意行为",
        }
    }
    
    /// 判断是否应该阻止执行
    pub fn should_block(&self) -> bool {
        matches!(self, ThreatLevel::Critical)
    }
    
    /// 判断是否需要额外审查
    pub fn requires_review(&self) -> bool {
        matches!(self, ThreatLevel::High | ThreatLevel::Critical)
    }
}

/// 全局安全策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// 安全策略等级
    pub level: SecurityPolicyLevel,
    /// 是否允许自定义算法
    pub allow_custom_algorithms: bool,
    /// 是否允许网络访问
    pub allow_network: bool,
    /// 是否允许文件系统访问
    pub allow_filesystem: bool,
    /// 是否允许外部库
    pub allow_external_libraries: bool,
    /// 允许的环境变量
    pub allowed_env_vars: Vec<String>,
    /// 允许的文件系统路径
    pub allowed_paths: Vec<String>,
    /// 允许的导入模块
    pub allowed_imports: HashSet<String>,
    /// 禁止的导入模块
    pub forbidden_imports: HashSet<String>,
    /// 最大内存使用(MB)
    pub max_memory_mb: usize,
    /// 最大CPU时间(ms)
    pub max_cpu_time_ms: u64,
    /// 最大WASM二进制大小(MB)
    pub max_wasm_size_mb: usize,
    /// 最大代码大小(KB)
    pub max_code_size_kb: usize,
    /// 算法执行超时(秒)
    pub execution_timeout_seconds: u64,
    /// 资源监控间隔(ms)
    pub monitoring_interval_ms: u64,
    /// 自定义安全规则
    pub custom_rules: HashMap<String, String>,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        let mut allowed_imports = HashSet::new();
        allowed_imports.insert("wasi_snapshot_preview1".to_string());
        allowed_imports.insert("env".to_string());
        
        let mut forbidden_imports = HashSet::new();
        forbidden_imports.insert("proc_exit".to_string());
        forbidden_imports.insert("path_open".to_string());
        forbidden_imports.insert("sock_open".to_string());
        
        Self {
            level: SecurityPolicyLevel::Standard,
            allow_custom_algorithms: true,
            allow_network: false,
            allow_filesystem: false,
            allow_external_libraries: false,
            allowed_env_vars: vec!["TEMP".to_string()],
            allowed_paths: vec![],
            allowed_imports,
            forbidden_imports,
            max_memory_mb: 1024,
            max_cpu_time_ms: 30000,
            max_wasm_size_mb: 10,
            max_code_size_kb: 1024,
            execution_timeout_seconds: 60,
            monitoring_interval_ms: 1000,
            custom_rules: HashMap::new(),
        }
    }
}

/// 安全审计事件类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityAuditEventType {
    /// 算法创建
    AlgorithmCreation,
    /// 算法执行
    AlgorithmExecution,
    /// 安全策略违规
    SecurityViolation,
    /// 资源限制超出
    ResourceExceeded,
    /// 安全策略更改
    PolicyChange,
    /// 访问拒绝
    AccessDenied,
}

/// 安全审计事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditEvent {
    /// 事件ID
    pub id: String,
    /// 事件类型
    pub event_type: SecurityAuditEventType,
    /// 事件时间
    pub timestamp: i64,
    /// 相关算法ID
    pub algorithm_id: Option<String>,
    /// 相关任务ID
    pub task_id: Option<String>,
    /// 事件描述
    pub description: String,
    /// 事件详情
    pub details: HashMap<String, String>,
    /// 事件严重程度
    pub severity: String,
}

/// 安全策略管理器
pub struct SecurityPolicyManager {
    /// 全局安全策略
    policy: RwLock<SecurityPolicy>,
    /// 算法类型策略映射
    algorithm_type_policies: RwLock<HashMap<AlgorithmType, SecurityPolicy>>,
    /// 算法ID策略映射
    algorithm_policies: RwLock<HashMap<String, SecurityPolicy>>,
    /// 安全审计日志
    audit_log: Mutex<Vec<SecurityAuditEvent>>,
    /// 可信算法ID列表
    trusted_algorithms: RwLock<HashSet<String>>,
    /// 黑名单算法ID列表
    blacklisted_algorithms: RwLock<HashSet<String>>,
    /// 算法验证器
    validator: AlgorithmValidator,
}

impl SecurityPolicyManager {
    /// 创建新的安全策略管理器
    pub fn new(policy: SecurityPolicy) -> Self {
        Self {
            policy: RwLock::new(policy.clone()),
            algorithm_type_policies: RwLock::new(HashMap::new()),
            algorithm_policies: RwLock::new(HashMap::new()),
            audit_log: Mutex::new(Vec::new()),
            trusted_algorithms: RwLock::new(HashSet::new()),
            blacklisted_algorithms: RwLock::new(HashSet::new()),
            validator: AlgorithmValidator::new(),
        }
    }
    
    /// 创建默认安全策略管理器
    pub fn new_default() -> Self {
        Self::new(SecurityPolicy::default())
    }
    
    /// 获取算法的安全策略（兼容方法）
    pub fn get_policy(&self, algorithm: &Algorithm) -> Result<SecurityPolicy> {
        Ok(self.get_policy_for_algorithm(algorithm))
    }
    
    /// 获取全局安全策略
    pub fn get_global_policy(&self) -> SecurityPolicy {
        self.policy.read().unwrap().clone()
    }
    
    /// 设置全局安全策略
    pub fn set_global_policy(&self, policy: SecurityPolicy) -> Result<()> {
        let mut current = self.policy.write().unwrap();
        *current = policy.clone();
        
        // 记录安全策略变更事件
        let event = SecurityAuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: SecurityAuditEventType::PolicyChange,
            timestamp: chrono::Utc::now().timestamp(),
            algorithm_id: None,
            task_id: None,
            description: "全局安全策略已更新".to_string(),
            details: HashMap::new(),
            severity: "INFO".to_string(),
        };
        
        self.add_audit_event(event)?;
        
        info!("全局安全策略已更新");
        Ok(())
    }
    
    /// 为算法类型设置安全策略
    pub fn set_policy_for_algorithm_type(&self, algorithm_type: AlgorithmType, policy: SecurityPolicy) -> Result<()> {
        let mut policies = self.algorithm_type_policies.write().unwrap();
        policies.insert(algorithm_type, policy);
        
        info!("已为算法类型 {:?} 设置安全策略", algorithm_type);
        Ok(())
    }
    
    /// 为指定算法设置安全策略
    pub fn set_policy_for_algorithm(&self, algorithm_id: &str, policy: SecurityPolicy) -> Result<()> {
        let mut policies = self.algorithm_policies.write().unwrap();
        policies.insert(algorithm_id.to_string(), policy);
        
        // 记录安全策略变更事件
        let event = SecurityAuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: SecurityAuditEventType::PolicyChange,
            timestamp: chrono::Utc::now().timestamp(),
            algorithm_id: Some(algorithm_id.to_string()),
            task_id: None,
            description: format!("算法 {} 的安全策略已更新", algorithm_id),
            details: HashMap::new(),
            severity: "INFO".to_string(),
        };
        
        self.add_audit_event(event)?;
        
        info!("已为算法 {} 设置安全策略", algorithm_id);
        Ok(())
    }
    
    /// 获取算法的安全策略
    pub fn get_policy_for_algorithm(&self, algorithm: &Algorithm) -> SecurityPolicy {
        // 1. 检查算法特定策略
        let algorithm_policies = self.algorithm_policies.read().unwrap();
        if let Some(policy) = algorithm_policies.get(&algorithm.id) {
            return policy.clone();
        }
        
        // 2. 检查算法类型策略
        let type_policies = self.algorithm_type_policies.read().unwrap();
        if let Some(policy) = type_policies.get(&algorithm.algorithm_type) {
            return policy.clone();
        }
        
        // 3. 使用全局策略
        self.get_global_policy()
    }
    
    /// 验证算法安全性
    pub fn validate_algorithm_security(&self, algorithm: &Algorithm) -> Result<ValidationReport> {
        // 首先检查黑名单
        if self.is_algorithm_blacklisted(&algorithm.id) {
            return Err(Error::algorithm(format!("算法 {} 在黑名单中", algorithm.id)));
        }
        
        // 如果在可信列表中，则跳过详细验证
        if self.is_algorithm_trusted(&algorithm.id) {
            return Ok(ValidationReport {
                algorithm_id: algorithm.id.clone(),
                passed: true,
                issues: Vec::new(),
                warnings: Vec::new(),
                security_score: 100.0,
                metadata: HashMap::new(),
            });
        }
        
        // 获取适用的安全策略
        let policy = self.get_policy_for_algorithm(algorithm);
        
        // 根据策略配置验证器
        let validator = self.configure_validator(policy)?;
        
        // 执行算法验证
        let report = validator.validate(algorithm)?;
        
        // 记录验证结果
        if !report.passed {
            let event = SecurityAuditEvent {
                id: uuid::Uuid::new_v4().to_string(),
                event_type: SecurityAuditEventType::SecurityViolation,
                timestamp: chrono::Utc::now().timestamp(),
                algorithm_id: Some(algorithm.id.clone()),
                task_id: None,
                description: format!("算法 {} 安全验证失败", algorithm.id),
                details: report.issues.iter()
                    .map(|i| (i.code.clone(), i.message.clone()))
                    .collect(),
                severity: "WARNING".to_string(),
            };
            
            self.add_audit_event(event)?;
        }
        
        Ok(report)
    }
    
    /// 验证算法更新
    pub fn validate_algorithm_update(&self, algorithm: &Algorithm) -> Result<ValidationReport> {
        self.validate_algorithm_security(algorithm)
    }
    
    /// 配置沙箱环境
    pub fn configure_sandbox(&self, algorithm: &Algorithm) -> Result<SandboxConfig> {
        let policy = self.get_policy_for_algorithm(algorithm);
        
        // 根据安全策略和算法类型确定沙箱安全级别
        let security_level = self.determine_security_level(algorithm, &policy);
        
        let mut sandbox_config = SandboxConfig {
            security_level,
            allow_network: policy.allow_network,
            allow_filesystem: policy.allow_filesystem,
            allowed_env_vars: policy.allowed_env_vars.clone(),
            allowed_paths: policy.allowed_paths.clone(),
            allow_stdin: false,
            allow_stdout: true,
            allow_stderr: true,
            allowed_libraries: Vec::new(),
            use_real_time: false,
            preallocated_memory: Some(policy.max_memory_mb * 1024 * 1024),
        };
        
        // 根据安全级别进一步调整配置
        match security_level {
            SandboxSecurityLevel::Low => {
                // 低安全级别允许更多权限
                sandbox_config.allow_stdin = true;
                sandbox_config.allowed_libraries = vec!["libc.so.6".to_string()];
            },
            SandboxSecurityLevel::Medium => {
                // 中等安全级别有限制权限
                if policy.level == SecurityPolicyLevel::Low {
                    sandbox_config.allowed_libraries = vec!["libc.so.6".to_string()];
                }
            },
            SandboxSecurityLevel::High => {
                // 高安全级别严格限制权限
                sandbox_config.allow_network = false;
                sandbox_config.allow_filesystem = false;
                sandbox_config.allowed_env_vars.clear();
                sandbox_config.allowed_paths.clear();
            },
        }
        
        Ok(sandbox_config)
    }
    
    /// 配置资源限制
    pub fn configure_resource_limits(&self, algorithm: &Algorithm) -> Result<ResourceLimits> {
        let policy = self.get_policy_for_algorithm(algorithm);
        
        // 基于策略设置资源限制
        let resource_limits = ResourceLimits {
            max_memory: policy.max_memory_mb * 1024 * 1024,
            max_cpu_time: policy.max_cpu_time_ms,
            max_code_size: policy.max_code_size_kb * 1024,
            max_output_size: 10 * 1024 * 1024, // 10MB
            max_instruction_count: 1_000_000_000, // 10亿指令
            track_memory: true,
            track_cpu_time: true,
            track_instructions: true,
        };
        
        Ok(resource_limits)
    }
    
    /// 为算法配置WASM安全参数
    pub fn configure_wasm_security(&self, algorithm: &Algorithm) -> Result<WasmSecurityConfig> {
        debug!("为算法 {} 配置WASM安全参数", algorithm.id);
        
        // 获取算法适用的安全策略
        let policy = self.get_policy_for_algorithm(algorithm);
        
        let mut wasm_config = WasmSecurityConfig {
            max_memory_pages: (policy.max_memory_mb * 1024 * 1024 / 65536) as u32, // 每页64KB
            max_table_size: 10000, // 默认值
            max_imports: 100,
            max_exports: 100,
            allowed_import_modules: policy.allowed_imports.clone(),
            forbidden_import_functions: policy.forbidden_imports.clone(),
        };
        
        // 根据算法类型调整参数
        match algorithm.algorithm_type {
            AlgorithmType::Classification | AlgorithmType::Regression => {
                // 机器学习算法可能需要更多内存
                wasm_config.max_memory_pages = wasm_config.max_memory_pages.max(2000);
            }
            AlgorithmType::DimensionReduction => {
                // 降维算法需要额外的内存
                wasm_config.max_memory_pages = wasm_config.max_memory_pages.max(2500);
            }
            AlgorithmType::Custom => {
                // 自定义算法使用默认限制，但根据安全级别可能更严格
                if policy.level == SecurityPolicyLevel::High || policy.level == SecurityPolicyLevel::Strict {
                    wasm_config.max_memory_pages = wasm_config.max_memory_pages.min(500);
                    wasm_config.max_table_size = 5000;
                    wasm_config.max_imports = 50;
                    wasm_config.max_exports = 50;
                    
                    // 添加更多的禁止函数
                    wasm_config.forbidden_import_functions.insert("random_get".to_string());
                    wasm_config.forbidden_import_functions.insert("clock_time_get".to_string());
                }
            }
            _ => {}
        }
        
        // 记录审计信息
        let mut details = HashMap::new();
        details.insert("max_memory_pages".to_string(), wasm_config.max_memory_pages.to_string());
        details.insert("max_table_size".to_string(), wasm_config.max_table_size.to_string());
        details.insert("max_imports".to_string(), wasm_config.max_imports.to_string());
        details.insert("max_exports".to_string(), wasm_config.max_exports.to_string());
        
        let event = SecurityAuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: SecurityAuditEventType::AlgorithmExecution,
            timestamp: chrono::Utc::now().timestamp(),
            algorithm_id: Some(algorithm.id.clone()),
            task_id: None,
            description: "WASM安全配置已生成".to_string(),
            details,
            severity: "INFO".to_string(),
        };
        
        self.add_audit_event(event)?;
        
        Ok(wasm_config)
    }
    
    /// 添加可信算法
    pub fn add_trusted_algorithm(&self, algorithm_id: &str) -> Result<()> {
        let mut trusted = self.trusted_algorithms.write().unwrap();
        trusted.insert(algorithm_id.to_string());
        
        info!("算法 {} 已添加至可信列表", algorithm_id);
        Ok(())
    }
    
    /// 移除可信算法
    pub fn remove_trusted_algorithm(&self, algorithm_id: &str) -> Result<()> {
        let mut trusted = self.trusted_algorithms.write().unwrap();
        trusted.remove(algorithm_id);
        
        info!("算法 {} 已从可信列表移除", algorithm_id);
        Ok(())
    }
    
    /// 检查算法是否可信
    pub fn is_algorithm_trusted(&self, algorithm_id: &str) -> bool {
        let trusted = self.trusted_algorithms.read().unwrap();
        trusted.contains(algorithm_id)
    }
    
    /// 添加黑名单算法
    pub fn add_blacklisted_algorithm(&self, algorithm_id: &str) -> Result<()> {
        let mut blacklisted = self.blacklisted_algorithms.write().unwrap();
        blacklisted.insert(algorithm_id.to_string());
        
        // 记录黑名单操作
        let event = SecurityAuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: SecurityAuditEventType::PolicyChange,
            timestamp: chrono::Utc::now().timestamp(),
            algorithm_id: Some(algorithm_id.to_string()),
            task_id: None,
            description: format!("算法 {} 已添加至黑名单", algorithm_id),
            details: HashMap::new(),
            severity: "WARNING".to_string(),
        };
        
        self.add_audit_event(event)?;
        
        info!("算法 {} 已添加至黑名单", algorithm_id);
        Ok(())
    }
    
    /// 移除黑名单算法
    pub fn remove_blacklisted_algorithm(&self, algorithm_id: &str) -> Result<()> {
        let mut blacklisted = self.blacklisted_algorithms.write().unwrap();
        blacklisted.remove(algorithm_id);
        
        info!("算法 {} 已从黑名单移除", algorithm_id);
        Ok(())
    }
    
    /// 检查算法是否在黑名单中
    pub fn is_algorithm_blacklisted(&self, algorithm_id: &str) -> bool {
        let blacklisted = self.blacklisted_algorithms.read().unwrap();
        blacklisted.contains(algorithm_id)
    }
    
    /// 记录算法执行事件
    pub fn log_algorithm_execution(&self, algorithm_id: &str, task_id: &str) -> Result<()> {
        let event = SecurityAuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: SecurityAuditEventType::AlgorithmExecution,
            timestamp: chrono::Utc::now().timestamp(),
            algorithm_id: Some(algorithm_id.to_string()),
            task_id: Some(task_id.to_string()),
            description: format!("算法 {} 执行开始", algorithm_id),
            details: HashMap::new(),
            severity: "INFO".to_string(),
        };
        
        self.add_audit_event(event)?;
        
        Ok(())
    }
    
    /// 记录资源超限事件
    pub fn log_resource_exceeded(&self, algorithm_id: &str, task_id: &str, resource: &str, value: String) -> Result<()> {
        let mut details = HashMap::new();
        details.insert("resource".to_string(), resource.to_string());
        details.insert("value".to_string(), value);
        
        let event = SecurityAuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: SecurityAuditEventType::ResourceExceeded,
            timestamp: chrono::Utc::now().timestamp(),
            algorithm_id: Some(algorithm_id.to_string()),
            task_id: Some(task_id.to_string()),
            description: format!("算法 {} 资源 {} 超限", algorithm_id, resource),
            details,
            severity: "WARNING".to_string(),
        };
        
        self.add_audit_event(event)?;
        
        warn!("算法 {} 任务 {} 资源 {} 超限", algorithm_id, task_id, resource);
        Ok(())
    }
    
    /// 获取算法的安全审计日志
    pub fn get_audit_log_for_algorithm(&self, algorithm_id: &str) -> Result<Vec<SecurityAuditEvent>> {
        let log = self.audit_log.lock().unwrap();
        
        let filtered = log.iter()
            .filter(|event| event.algorithm_id.as_ref().map_or(false, |id| id == algorithm_id))
            .cloned()
            .collect();
            
        Ok(filtered)
    }
    
    /// 获取近期安全审计日志
    pub fn get_recent_audit_log(&self, limit: usize) -> Result<Vec<SecurityAuditEvent>> {
        let log = self.audit_log.lock().unwrap();
        
        let mut events = log.clone();
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp)); // 按时间倒序
        
        Ok(events.into_iter().take(limit).collect())
    }
    
    /// 根据算法类型和安全策略确定沙箱安全级别
    fn determine_security_level(&self, algorithm: &Algorithm, policy: &SecurityPolicy) -> SandboxSecurityLevel {
        // 首先检查元数据中是否有明确指定的安全级别
        if let Some(level) = algorithm.metadata.get("security_level") {
            match level.as_str() {
                "low" => return SandboxSecurityLevel::Low,
                "medium" => return SandboxSecurityLevel::Medium,
                "high" => return SandboxSecurityLevel::High,
                _ => {} // 使用默认分配逻辑
            }
        }
        
        // 根据安全策略等级确定基础安全级别
        let base_level = match policy.level {
            SecurityPolicyLevel::Low => SandboxSecurityLevel::Low,
            SecurityPolicyLevel::Standard => SandboxSecurityLevel::Medium,
            SecurityPolicyLevel::High | SecurityPolicyLevel::Strict => SandboxSecurityLevel::High,
        };
        
        // 根据算法类型进行调整
        match algorithm.algorithm_type {
            AlgorithmType::Custom => {
                // 自定义算法需要更高安全级别
                match base_level {
                    SandboxSecurityLevel::Low => SandboxSecurityLevel::Medium,
                    other => other,
                }
            },
            AlgorithmType::AnomalyDetection | AlgorithmType::Recommendation => {
                // 异常检测和推荐算法可能需要更多资源
                base_level
            },
            _ => {
                // 其他标准算法类型
                base_level
            }
        }
    }
    
    /// 根据安全策略配置验证器
    fn configure_validator(&self, policy: SecurityPolicy) -> Result<AlgorithmValidator> {
        let mut validator = self.validator.clone();
        
        // 配置验证器参数
        validator.with_max_code_size(policy.max_code_size_kb * 1024)
            .with_network_access(policy.allow_network)
            .with_filesystem_access(policy.allow_filesystem)
            .with_validation_timeout(Duration::from_secs(30));
            
        Ok(validator)
    }
    
    /// 添加审计事件
    fn add_audit_event(&self, event: SecurityAuditEvent) -> Result<()> {
        let mut log = self.audit_log.lock().unwrap();
        log.push(event);
        
        // 限制日志大小，防止内存泄漏
        if log.len() > 10000 {
            log.drain(0..5000);
        }
        
        Ok(())
    }
    
    /// 记录安全违规事件
    pub fn log_security_violation(
        &self,
        algorithm_id: &str,
        violation_type: &str,
        description: &str,
        details: &[(&str, &str)]
    ) -> Result<()> {
        let mut detail_map = HashMap::new();
        detail_map.insert("violation_type".to_string(), violation_type.to_string());
        
        for (key, value) in details {
            detail_map.insert(key.to_string(), value.to_string());
        }
        
        let event = SecurityAuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: SecurityAuditEventType::SecurityViolation,
            timestamp: chrono::Utc::now().timestamp(),
            algorithm_id: Some(algorithm_id.to_string()),
            task_id: None,
            description: description.to_string(),
            details: detail_map,
            severity: "WARNING".to_string(),
        };
        
        self.add_audit_event(event)?;
        
        warn!("算法 {} 安全违规: {}", algorithm_id, description);
        Ok(())
    }
    
    /// 记录执行失败事件
    pub fn log_execution_failure(
        &self,
        algorithm_id: &str,
        task_id: &str,
        error_message: &str
    ) -> Result<()> {
        let mut details = HashMap::new();
        details.insert("error".to_string(), error_message.to_string());
        
        let event = SecurityAuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: SecurityAuditEventType::AlgorithmExecution,
            timestamp: chrono::Utc::now().timestamp(),
            algorithm_id: Some(algorithm_id.to_string()),
            task_id: Some(task_id.to_string()),
            description: format!("算法 {} 执行失败", algorithm_id),
            details,
            severity: "ERROR".to_string(),
        };
        
        self.add_audit_event(event)?;
        
        error!("算法 {} 任务 {} 执行失败: {}", algorithm_id, task_id, error_message);
        Ok(())
    }
    
    /// 评估算法安全性（API兼容方法）
    pub async fn assess_algorithm_security(
        &self,
        algorithm: &Algorithm,
        parameters: &HashMap<String, serde_json::Value>,
    ) -> Result<SecurityAssessment> {
        // 获取算法的安全策略
        let policy = self.get_policy_for_algorithm(algorithm);
        
        // 执行安全验证
        let validation_report = self.validate_algorithm_security(algorithm)?;
        
        // 基于验证结果和策略创建安全评估
        let mut security_score = if validation_report.passed {
            validation_report.security_score as f64
        } else {
            0.0
        };
        
        // 根据策略调整安全分数
        match policy.level {
            SecurityPolicyLevel::Strict => security_score *= 0.8, // 更严格评估
            SecurityPolicyLevel::High => security_score *= 0.9,
            SecurityPolicyLevel::Standard => {}, // 保持原分数
            SecurityPolicyLevel::Low => security_score *= 1.1, // 稍微宽松
        }
        
        // 确保分数不超过100
        security_score = security_score.min(100.0);
        
        let risk_level = match security_score as u32 {
            90..=100 => "低",
            70..=89 => "中等",
            50..=69 => "高",
            _ => "极高",
        };
        
        // 检查参数安全性
        let mut parameter_issues = Vec::new();
        for (key, value) in parameters {
            if key.contains("password") || key.contains("secret") || key.contains("token") {
                parameter_issues.push(format!("参数 {} 可能包含敏感信息", key));
            }
            
            if let serde_json::Value::String(s) = value {
                if s.len() > 10000 {
                    parameter_issues.push(format!("参数 {} 值过长，可能存在风险", key));
                }
            }
        }
        
        Ok(SecurityAssessment {
            algorithm_id: algorithm.id.clone(),
            security_score,
            risk_level: risk_level.to_string(),
            policy_compliance: validation_report.passed,
            identified_risks: validation_report.issues.iter()
                .map(|issue| issue.message.clone())
                .chain(parameter_issues)
                .collect(),
            recommendations: validation_report.warnings.iter()
                .map(|warning| warning.clone())
                .collect(),
            assessment_time: chrono::Utc::now(),
            metadata: validation_report.metadata,
        })
    }
    

    

    

    

    

    

    

    
    /// 记录策略变更事件
    pub fn log_policy_change(
        &self,
        algorithm_id: &str,
        description: &str,
        details: &HashMap<String, String>,
    ) -> Result<()> {
        let event = SecurityAuditEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: SecurityAuditEventType::PolicyChange,
            timestamp: chrono::Utc::now().timestamp(),
            algorithm_id: Some(algorithm_id.to_string()),
            task_id: None,
            description: description.to_string(),
            details: details.clone(),
            severity: "INFO".to_string(),
        };
        
        self.add_audit_event(event)
    }
}

/// 安全评估结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAssessment {
    /// 算法ID
    pub algorithm_id: String,
    /// 安全评分 (0-100)
    pub security_score: f64,
    /// 风险级别
    pub risk_level: String,
    /// 策略合规性
    pub policy_compliance: bool,
    /// 识别的风险
    pub identified_risks: Vec<String>,
    /// 建议
    pub recommendations: Vec<String>,
    /// 评估时间
    pub assessment_time: chrono::DateTime<chrono::Utc>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

/// 创建标准安全策略管理器
pub fn create_standard_security_manager() -> SecurityPolicyManager {
    SecurityPolicyManager::new(SecurityPolicy::default())
}

/// 创建高安全策略管理器
pub fn create_high_security_manager() -> SecurityPolicyManager {
    let policy = SecurityPolicy {
        level: SecurityPolicyLevel::High,
        allow_custom_algorithms: false,
        allow_network: false,
        allow_filesystem: false,
        allow_external_libraries: false,
        allowed_env_vars: vec![],
        allowed_paths: vec![],
        allowed_imports: {
            let mut set = HashSet::new();
            set.insert("wasi_snapshot_preview1".to_string());
            set.insert("env".to_string());
            set
        },
        forbidden_imports: {
            let mut set = HashSet::new();
            set.insert("proc_exit".to_string());
            set.insert("path_open".to_string());
            set.insert("sock_open".to_string());
            set.insert("fd_write".to_string());
            set.insert("fd_read".to_string());
            set
        },
        max_memory_mb: 512,
        max_cpu_time_ms: 15000,
        max_wasm_size_mb: 5,
        max_code_size_kb: 512,
        execution_timeout_seconds: 30,
        monitoring_interval_ms: 500,
        custom_rules: HashMap::new(),
    };
    
    SecurityPolicyManager::new(policy)
} 