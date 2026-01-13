use crate::algorithm::security::{
    SecurityPolicyManager, SecurityPolicy, SecurityPolicyLevel, SecurityAuditEventType
};
use crate::algorithm::types::{
    Algorithm, AlgorithmType, SandboxSecurityLevel, ResourceUsage, ResourceLimits, SandboxConfig, NetworkPolicy, FilesystemPolicy
};
// removed unused SandboxType import
use crate::error::{Error, Result};

use std::sync::Arc;
use std::collections::{HashMap, HashSet};

/// 安全风险评分
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    /// 低风险
    Low,
    /// 中等风险
    Medium,
    /// 高风险
    High,
    /// 非常高风险
    VeryHigh,
}

impl From<u32> for RiskLevel {
    fn from(score: u32) -> Self {
        match score {
            0..=25 => RiskLevel::Low,
            26..=50 => RiskLevel::Medium,
            51..=75 => RiskLevel::High,
            _ => RiskLevel::VeryHigh,
        }
    }
}

impl From<RiskLevel> for SecurityPolicyLevel {
    fn from(risk: RiskLevel) -> Self {
        match risk {
            RiskLevel::Low => SecurityPolicyLevel::Low,
            RiskLevel::Medium => SecurityPolicyLevel::Standard,
            RiskLevel::High => SecurityPolicyLevel::High,
            RiskLevel::VeryHigh => SecurityPolicyLevel::Strict,
        }
    }
}

/// 风险评估结果
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    /// 算法ID
    pub algorithm_id: String,
    /// 风险级别
    pub risk_level: RiskLevel,
    /// 风险评分 (0-100)
    pub risk_score: u32,
    /// 评估时间
    pub assessment_time: chrono::DateTime<chrono::Utc>,
    /// 评估细节
    pub details: HashMap<String, String>,
    /// 建议的安全策略级别
    pub suggested_policy_level: SecurityPolicyLevel,
}

/// 自动安全调整系统
pub struct AutoSecurityAdjuster {
    /// 安全策略管理器
    security_manager: Arc<SecurityPolicyManager>,
    /// 风险评估历史
    assessment_history: HashMap<String, Vec<RiskAssessment>>,
    /// 已知危险函数集合
    dangerous_functions: HashSet<String>,
    /// 已知危险导入模块
    dangerous_imports: HashSet<String>,
    /// 资源使用阈值
    resource_thresholds: ResourceThresholds,
    /// 自动调整配置
    config: AutoAdjustConfig,
}

/// 资源使用阈值配置
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    /// CPU时间阈值(毫秒)
    pub cpu_time_threshold_ms: u64,
    /// 内存使用阈值(字节)
    pub memory_threshold_bytes: usize,
    /// 网络请求数阈值
    pub network_requests_threshold: usize,
    /// 文件操作数阈值
    pub file_operations_threshold: usize,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            cpu_time_threshold_ms: 10000, // 10秒
            memory_threshold_bytes: 100 * 1024 * 1024, // 100MB
            network_requests_threshold: 50,
            file_operations_threshold: 100,
        }
    }
}

/// 自动调整配置
#[derive(Debug, Clone)]
pub struct AutoAdjustConfig {
    /// 是否启用自动调整
    pub enabled: bool,
    /// 调整间隔(秒)
    pub adjustment_interval_secs: u64,
    /// 连续违规阈值
    pub violation_threshold: usize,
    /// 提升级别冷却期(小时)
    pub upgrade_cooldown_hours: u64,
    /// 降级级别冷却期(小时)
    pub downgrade_cooldown_hours: u64,
    /// 是否允许通知
    pub allow_notifications: bool,
}

impl Default for AutoAdjustConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            adjustment_interval_secs: 3600, // 1小时
            violation_threshold: 3,
            upgrade_cooldown_hours: 24, // 1天
            downgrade_cooldown_hours: 168, // 7天
            allow_notifications: true,
        }
    }
}

impl AutoSecurityAdjuster {
    /// 创建新的自动安全调整系统
    pub fn new(security_manager: Arc<SecurityPolicyManager>) -> Self {
        let mut dangerous_functions = HashSet::new();
        dangerous_functions.insert("eval".to_string());
        dangerous_functions.insert("exec".to_string());
        dangerous_functions.insert("Function".to_string());
        dangerous_functions.insert("setTimeout".to_string());
        dangerous_functions.insert("setInterval".to_string());
        
        let mut dangerous_imports = HashSet::new();
        dangerous_imports.insert("fs".to_string());
        dangerous_imports.insert("net".to_string());
        dangerous_imports.insert("http".to_string());
        dangerous_imports.insert("child_process".to_string());
        
        Self {
            security_manager,
            assessment_history: HashMap::new(),
            dangerous_functions,
            dangerous_imports,
            resource_thresholds: ResourceThresholds::default(),
            config: AutoAdjustConfig::default(),
        }
    }
    
    /// 创建简化的自动安全调整系统
    pub fn new_simple() -> Self {
        let security_manager = Arc::new(SecurityPolicyManager::new_default());
        Self::new(security_manager)
    }
    
    /// 记录算法执行情况，用于安全调整
    pub async fn record_execution(
        &self,
        algorithm_id: &str,
        execution_time: f64,
        resource_usage: &ResourceUsage,
    ) -> Result<()> {
        // 检查是否超过资源阈值
        let mut violations = Vec::new();
        
        if resource_usage.memory_usage > self.resource_thresholds.memory_threshold_bytes as u64 {
            violations.push(("memory", resource_usage.memory_usage.to_string()));
        }
        
        if (execution_time * 1000.0) as u64 > self.resource_thresholds.cpu_time_threshold_ms {
            violations.push(("cpu_time", execution_time.to_string()));
        }
        
        // 记录违规情况
        for (resource_type, value) in violations {
            self.security_manager.log_resource_exceeded(
                algorithm_id,
                &format!("exec_{}", chrono::Utc::now().timestamp()),
                resource_type,
                value,
            )?;
        }
        
        Ok(())
    }
    
    /// 设置自动调整配置
    pub fn with_config(mut self, config: AutoAdjustConfig) -> Self {
        self.config = config;
        self
    }
    
    /// 设置资源阈值
    pub fn with_resource_thresholds(mut self, thresholds: ResourceThresholds) -> Self {
        self.resource_thresholds = thresholds;
        self
    }
    
    /// 添加危险函数
    pub fn add_dangerous_function(&mut self, function_name: &str) {
        self.dangerous_functions.insert(function_name.to_string());
    }
    
    /// 添加危险导入模块
    pub fn add_dangerous_import(&mut self, import_name: &str) {
        self.dangerous_imports.insert(import_name.to_string());
    }
    
    /// 评估算法风险
    pub fn assess_algorithm_risk(&mut self, algorithm: &Algorithm) -> Result<RiskAssessment> {
        // 基础风险分数
        let mut risk_score = 0;
        let mut details = HashMap::new();
        
        // 1. 根据算法类型评估风险
        match algorithm.algorithm_type {
            AlgorithmType::Custom => {
                risk_score += 25;
                details.insert("type_risk".to_string(), "Custom algorithm types pose higher risk".to_string());
            },
            AlgorithmType::Classification | AlgorithmType::Regression => {
                risk_score += 10;
                details.insert("type_risk".to_string(), "Standard ML algorithm types pose moderate risk".to_string());
            },
            _ => {
                risk_score += 15;
                details.insert("type_risk".to_string(), "Specialized algorithm types pose elevated risk".to_string());
            }
        }
        
        // 2. 检查元数据中的风险指标
        if let Some(network_access) = algorithm.metadata.get("has_network_access") {
            if network_access == "true" {
                risk_score += 20;
                details.insert("network_risk".to_string(), "Algorithm requests network access".to_string());
            }
        }
        
        if let Some(filesystem_access) = algorithm.metadata.get("has_filesystem_access") {
            if filesystem_access == "true" {
                risk_score += 15;
                details.insert("filesystem_risk".to_string(), "Algorithm requests filesystem access".to_string());
            }
        }
        
        // 3. 分析WASM二进制中的风险
        if let Ok(wasm_risk) = self.analyze_wasm_risks(&algorithm.code) {
            risk_score += wasm_risk.0;
            details.insert("wasm_risk".to_string(), wasm_risk.1);
        }
        
        // 4. 检查过去的审计日志
        let audit_logs = self.security_manager.get_audit_log_for_algorithm(&algorithm.id)?;
        let violations_count = audit_logs.iter()
            .filter(|log| matches!(log.event_type, 
                SecurityAuditEventType::SecurityViolation | 
                SecurityAuditEventType::ResourceExceeded))
            .count();
        
        if violations_count > 0 {
            let risk_increase = std::cmp::min(violations_count * 5, 30);
            risk_score += risk_increase as u32;
            details.insert("violation_history".to_string(), 
                format!("Algorithm has {} previous security violations", violations_count));
        }
        
        // 5. 检查是否已经在黑名单中
        if self.security_manager.is_algorithm_blacklisted(&algorithm.id) {
            risk_score += 50;
            details.insert("blacklist".to_string(), "Algorithm is blacklisted".to_string());
        }
        
        // 6. 检查是否在可信列表中
        if self.security_manager.is_algorithm_trusted(&algorithm.id) {
            risk_score = risk_score.saturating_sub(30);
            details.insert("trusted".to_string(), "Algorithm is in trusted list".to_string());
        }
        
        // 限制分数在0-100范围内
        risk_score = risk_score.clamp(0, 100);
        
        // 确定风险级别
        let risk_level: RiskLevel = risk_score.into();
        
        // 根据风险级别确定建议的安全策略级别
        let suggested_policy_level = risk_level.into();
        
        let assessment = RiskAssessment {
            algorithm_id: algorithm.id.clone(),
            risk_level,
            risk_score,
            assessment_time: chrono::Utc::now(),
            details,
            suggested_policy_level,
        };
        
        // 保存评估结果
        self.add_assessment_to_history(&algorithm.id, assessment.clone());
        
        Ok(assessment)
    }
    
    /// 分析WASM二进制文件的风险
    fn analyze_wasm_risks(&self, wasm_binary: &[u8]) -> Result<(u32, String)> {
        // 验证WASM二进制文件
        if wasm_binary.len() < 8 {
            return Err(Error::invalid_argument("Invalid WASM binary: too short"));
        }
        
        // 检查WASM魔数
        if !wasm_binary.starts_with(&[0x00, 0x61, 0x73, 0x6D]) {
            return Err(Error::invalid_argument("Invalid WASM binary: wrong magic number"));
        }
        
        // TODO: 需要在Cargo.toml中添加wasmparser依赖
        /*
        use wasmparser::{Parser, Chunk, SectionReader, Section};
        
        // WASM检查相关代码
        */
        
        // 解析和分析WASM
        let (risk_score, reasons) = self.parse_and_analyze_wasm(wasm_binary)?;
        
        // 生成详细的风险报告
        let report = reasons.join("\n");
        
        Ok((risk_score, report))
    }
    
    /// 解析和分析WASM二进制文件
    fn parse_and_analyze_wasm(&self, wasm_binary: &[u8]) -> Result<(u32, Vec<String>)> {
        #[cfg(feature = "wasmtime")]
        use wasmparser::{Parser, Payload};
        
        let mut risk_score = 0;
        let mut reasons = Vec::new();
        
        // 创建WASM解析器
        let parser = Parser::new(0).parse_all(wasm_binary);
        
        // 遍历所有section
        for payload in parser {
            match payload? {
                Payload::MemorySection(memory_reader) => {
                    // 检查内存限制
                    for memory_result in memory_reader {
                        let memory = memory_result?;
                        if memory.initial > 1000 || memory.maximum.unwrap_or(0) > 10000 {
                            risk_score += 20;
                            reasons.push("High memory limits detected".to_string());
                        }
                    }
                },
                Payload::TableSection(table_reader) => {
                    // 检查表大小限制
                    for table_result in table_reader {
                        let table = table_result?;
                        if table.initial > 1000 || table.maximum.unwrap_or(0) > 10000 {
                            risk_score += 15;
                            reasons.push("Large table size detected".to_string());
                        }
                    }
                },
                Payload::FunctionSection(function_reader) => {
                    // 检查函数数量
                    let function_count = function_reader.count();
                    if function_count > 1000 {
                        risk_score += 10;
                        reasons.push("Large number of functions detected".to_string());
                    }
                },
                Payload::ImportSection(import_reader) => {
                    // 检查导入函数
                    for import_result in import_reader {
                        let import = import_result?;
                        if self.dangerous_imports.contains(&import.module.to_string()) {
                            risk_score += 25;
                            reasons.push(format!("Dangerous import detected: {}", import.module));
                        }
                    }
                },
                Payload::ExportSection(export_reader) => {
                    // 检查导出函数
                    for export_result in export_reader {
                        let export = export_result?;
                        if self.dangerous_functions.contains(&export.name.to_string()) {
                            risk_score += 30;
                            reasons.push(format!("Dangerous export detected: {}", export.name));
                        }
                    }
                },
                Payload::CodeSection(code_reader) => {
                    // 检查代码大小
                    for func_body_result in code_reader {
                        let func_body = func_body_result?;
                        let operators = func_body.get_operators_reader();
                        if operators.get_binary_reader().bytes_remaining() > 1000 {
                            risk_score += 15;
                            reasons.push("Large function body detected".to_string());
                        }
                    }
                },
                _ => {}
            }
        }
        
        // 确保风险分数在0-100范围内
        risk_score = risk_score.min(100);
        
        Ok((risk_score, reasons))
    }
    
    /// 将评估结果添加到历史记录
    fn add_assessment_to_history(&mut self, algorithm_id: &str, assessment: RiskAssessment) {
        self.assessment_history
            .entry(algorithm_id.to_string())
            .or_insert_with(Vec::new)
            .push(assessment);
        
        // 限制历史记录大小
        if let Some(history) = self.assessment_history.get_mut(algorithm_id) {
            if history.len() > 20 {
                history.sort_by(|a, b| b.assessment_time.cmp(&a.assessment_time));
                history.truncate(20);
            }
        }
    }
    
    /// 获取算法的风险评估历史
    pub fn get_assessment_history(&self, algorithm_id: &str) -> Vec<RiskAssessment> {
        self.assessment_history
            .get(algorithm_id)
            .cloned()
            .unwrap_or_default()
    }
    
    /// 获取最近的风险评估
    pub fn get_latest_assessment(&self, algorithm_id: &str) -> Option<RiskAssessment> {
        self.assessment_history
            .get(algorithm_id)
            .and_then(|history| {
                history.iter()
                    .max_by_key(|assessment| assessment.assessment_time)
                    .cloned()
            })
    }
    
    /// 自动调整算法安全策略
    pub fn auto_adjust_security_policy(&self, algorithm: &Algorithm) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }
        
        // 获取最新的风险评估
        let latest_assessment = match self.get_latest_assessment(&algorithm.id) {
            Some(assessment) => assessment,
            None => return Ok(false), // 没有评估结果，不调整
        };
        
        // 获取当前策略
        let current_policy = self.security_manager.get_policy_for_algorithm(algorithm)?;
        
        // 如果当前策略级别与建议级别不同，考虑调整
        if current_policy.level != latest_assessment.suggested_policy_level {
            // 查看最后一次策略变更时间
            let audit_logs = self.security_manager.get_audit_log_for_algorithm(&algorithm.id)?;
            let last_policy_change = audit_logs.iter()
                .filter(|log| log.event_type == SecurityAuditEventType::PolicyChange)
                .max_by_key(|log| log.timestamp);
            
            let now = chrono::Utc::now().timestamp();
            
            if let Some(last_change) = last_policy_change {
                let hours_since_change = (now - last_change.timestamp) / 3600;
                
                // 检查冷却期
                if self.is_policy_upgrade(&current_policy.level, &latest_assessment.suggested_policy_level) {
                    // 提高安全级别 - 应用较短的冷却期
                    if hours_since_change < self.config.upgrade_cooldown_hours {
                        return Ok(false);
                    }
                } else {
                    // 降低安全级别 - 应用较长的冷却期
                    if hours_since_change < self.config.downgrade_cooldown_hours {
                        return Ok(false);
                    }
                }
            }
            
            // 创建新的安全策略，复制当前策略但更改级别
            let mut new_policy = current_policy.clone();
            new_policy.level = latest_assessment.suggested_policy_level;
            
            // 根据安全级别调整其他策略属性
            self.adjust_policy_attributes(&mut new_policy);
            
            // 应用新策略
            self.security_manager.set_policy_for_algorithm(&algorithm.id, new_policy)?;
            
            // 记录策略变更事件
            let details = HashMap::from([
                ("previous_level".to_string(), format!("{:?}", current_policy.level)),
                ("new_level".to_string(), format!("{:?}", latest_assessment.suggested_policy_level)),
                ("reason".to_string(), format!("Auto-adjusted based on risk score: {}", latest_assessment.risk_score)),
            ]);
            
            self.security_manager.log_policy_change(
                &algorithm.id,
                "Auto security policy adjustment",
                &details
            )?;
            
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// 检查是否是策略升级(增加限制)
    fn is_policy_upgrade(&self, current: &SecurityPolicyLevel, suggested: &SecurityPolicyLevel) -> bool {
        match (current, suggested) {
            (SecurityPolicyLevel::Low, _) => *suggested != SecurityPolicyLevel::Low,
            (SecurityPolicyLevel::Standard, SecurityPolicyLevel::High | SecurityPolicyLevel::Strict) => true,
            (SecurityPolicyLevel::High, SecurityPolicyLevel::Strict) => true,
            _ => false,
        }
    }
    
    /// 根据安全级别调整策略的其他属性
    fn adjust_policy_attributes(&self, policy: &mut SecurityPolicy) {
        match policy.level {
            SecurityPolicyLevel::Low => {
                policy.allow_custom_algorithms = true;
                policy.allow_network_access = true;
                policy.filesystem_access = Some(vec!["./".to_string()]);
            },
            SecurityPolicyLevel::Standard => {
                policy.allow_custom_algorithms = true;
                policy.allow_network_access = false;
                policy.filesystem_access = Some(vec!["./data".to_string()]);
            },
            SecurityPolicyLevel::High => {
                policy.allow_custom_algorithms = true;
                policy.allow_network_access = false;
                policy.filesystem_access = None;
            },
            SecurityPolicyLevel::Strict => {
                policy.allow_custom_algorithms = false;
                policy.allow_network_access = false;
                policy.filesystem_access = None;
            },
        }
    }
    
    /// 扫描所有算法并自动调整策略
    pub fn scan_and_adjust_all(&self, algorithms: &[Algorithm]) -> Result<Vec<String>> {
        let mut adjusted_algorithms = Vec::new();
        
        for algorithm in algorithms {
            if self.auto_adjust_security_policy(algorithm)? {
                adjusted_algorithms.push(algorithm.id.clone());
            }
        }
        
        Ok(adjusted_algorithms)
    }
    
    /// 根据算法和安全评估调整安全配置（API兼容方法）
    pub async fn adjust_security_config(
        &self,
        algorithm: &Algorithm,
        security_assessment: &crate::algorithm::security::SecurityAssessment,
    ) -> Result<AdjustedSecurityConfig> {
        // 获取当前安全策略
        let current_policy = self.security_manager.get_policy_for_algorithm(algorithm);
        
        // 基于风险评估调整配置
        let security_level = match security_assessment.risk_level.as_str() {
            "低" => SandboxSecurityLevel::Low,
            "中等" => SandboxSecurityLevel::Medium,
            "高" => SandboxSecurityLevel::High,
            "极高" => SandboxSecurityLevel::Strict,
            _ => SandboxSecurityLevel::Strict,
        };
        
        // 创建资源限制
        let resource_limits = ResourceLimits {
            max_memory_usage: match security_level {
                SandboxSecurityLevel::Low => 2048 * 1024 * 1024,  // 2GB
                SandboxSecurityLevel::Standard => 1536 * 1024 * 1024, // 1.5GB
                SandboxSecurityLevel::Medium => 1024 * 1024 * 1024, // 1GB
                SandboxSecurityLevel::High => 512 * 1024 * 1024,   // 512MB
                SandboxSecurityLevel::Strict => 256 * 1024 * 1024, // 256MB
            },
            max_execution_time_ms: match security_level {
                SandboxSecurityLevel::Low => 300_000,    // 5 minutes
                SandboxSecurityLevel::Standard => 240_000, // 4 minutes
                SandboxSecurityLevel::Medium => 180_000, // 3 minutes
                SandboxSecurityLevel::High => 60_000,    // 1 minute
                SandboxSecurityLevel::Strict => 30_000, // 30 seconds
            },
            max_cpu_usage: match security_level {
                SandboxSecurityLevel::Low => 90.0,
                SandboxSecurityLevel::Standard => 85.0,
                SandboxSecurityLevel::Medium => 80.0,
                SandboxSecurityLevel::High => 60.0,
                SandboxSecurityLevel::Strict => 30.0,
            },
            max_disk_io: Some(match security_level {
                SandboxSecurityLevel::Low => 10 * 1024 * 1024 * 1024,  // 10GB
                SandboxSecurityLevel::Standard => 7 * 1024 * 1024 * 1024,  // 7GB
                SandboxSecurityLevel::Medium => 5 * 1024 * 1024 * 1024, // 5GB
                SandboxSecurityLevel::High => 1024 * 1024 * 1024,       // 1GB
                SandboxSecurityLevel::Strict => 512 * 1024 * 1024,     // 512MB
            }),
            max_network_io: Some(match security_level {
                SandboxSecurityLevel::Low => 100 * 1024 * 1024,   // 100MB
                SandboxSecurityLevel::Standard => 75 * 1024 * 1024,   // 75MB
                SandboxSecurityLevel::Medium => 50 * 1024 * 1024, // 50MB
                SandboxSecurityLevel::High => 10 * 1024 * 1024,   // 10MB
                SandboxSecurityLevel::Strict => 0,               // No network
            }),
            max_gpu_usage: Some(match security_level {
                SandboxSecurityLevel::Low => 90.0,
                SandboxSecurityLevel::Standard => 80.0,
                SandboxSecurityLevel::Medium => 70.0,
                SandboxSecurityLevel::High => 50.0,
                SandboxSecurityLevel::Strict => 0.0,
            }),
            max_gpu_memory_usage: Some(match security_level {
                SandboxSecurityLevel::Low => 4 * 1024 * 1024 * 1024,  // 4GB
                SandboxSecurityLevel::Standard => 3 * 1024 * 1024 * 1024,  // 3GB
                SandboxSecurityLevel::Medium => 2 * 1024 * 1024 * 1024, // 2GB
                SandboxSecurityLevel::High => 1024 * 1024 * 1024,       // 1GB
                SandboxSecurityLevel::Strict => 0,                     // No GPU
            }),
            max_memory_bytes: match security_level {
                SandboxSecurityLevel::Low => 2048 * 1024 * 1024,  // 2GB
                SandboxSecurityLevel::Standard => 1536 * 1024 * 1024, // 1.5GB
                SandboxSecurityLevel::Medium => 1024 * 1024 * 1024, // 1GB
                SandboxSecurityLevel::High => 512 * 1024 * 1024,   // 512MB
                SandboxSecurityLevel::Strict => 256 * 1024 * 1024, // 256MB
            },
            max_cpu_time_seconds: match security_level {
                SandboxSecurityLevel::Low => 300,    // 5 minutes
                SandboxSecurityLevel::Standard => 240, // 4 minutes
                SandboxSecurityLevel::Medium => 180, // 3 minutes
                SandboxSecurityLevel::High => 60,    // 1 minute
                SandboxSecurityLevel::Strict => 30, // 30 seconds
            },
            max_gpu_memory_bytes: Some(match security_level {
                SandboxSecurityLevel::Low => 4 * 1024 * 1024 * 1024,  // 4GB
                SandboxSecurityLevel::Standard => 3 * 1024 * 1024 * 1024,  // 3GB
                SandboxSecurityLevel::Medium => 2 * 1024 * 1024 * 1024, // 2GB
                SandboxSecurityLevel::High => 1024 * 1024 * 1024,       // 1GB
                SandboxSecurityLevel::Strict => 0,                     // No GPU
            }),
            max_network_bandwidth_bps: Some(match security_level {
                SandboxSecurityLevel::Low => 100 * 1024 * 1024,   // 100MB/s
                SandboxSecurityLevel::Standard => 75 * 1024 * 1024,   // 75MB/s
                SandboxSecurityLevel::Medium => 50 * 1024 * 1024, // 50MB/s
                SandboxSecurityLevel::High => 10 * 1024 * 1024,   // 10MB/s
                SandboxSecurityLevel::Strict => 0,               // No network
            }),
        };
        
        // 创建沙箱配置
        let sandbox_config = SandboxConfig {
            sandbox_type: convert_executor_to_types_sandbox_type(match security_level {
                SandboxSecurityLevel::Low => crate::algorithm::executor::config::SandboxType::LocalProcess,
                SandboxSecurityLevel::Standard => crate::algorithm::executor::config::SandboxType::Process,
                SandboxSecurityLevel::Medium => crate::algorithm::executor::config::SandboxType::Process,
                SandboxSecurityLevel::High => crate::algorithm::executor::config::SandboxType::IsolatedProcess,
                SandboxSecurityLevel::Strict => crate::algorithm::executor::config::SandboxType::Docker,
            }),
            security_level: security_level,
            network_policy: match security_level {
                SandboxSecurityLevel::Low => NetworkPolicy::Allowed,
                SandboxSecurityLevel::Standard => NetworkPolicy::Restricted(vec!["localhost".to_string(), "127.0.0.1".to_string()]),
                SandboxSecurityLevel::Medium => NetworkPolicy::Restricted(vec!["localhost".to_string()]),
                SandboxSecurityLevel::High | SandboxSecurityLevel::Strict => NetworkPolicy::Denied,
            },
            filesystem_policy: match security_level {
                SandboxSecurityLevel::Low => FilesystemPolicy::Full,
                SandboxSecurityLevel::Standard => FilesystemPolicy::Restricted(vec!["/tmp".to_string(), "/var/tmp".to_string()]),
                SandboxSecurityLevel::Medium => FilesystemPolicy::Restricted(vec!["/tmp".to_string()]),
                SandboxSecurityLevel::High => FilesystemPolicy::ReadOnly,
                SandboxSecurityLevel::Strict => FilesystemPolicy::ReadOnly,
            },
            memory_limit_mb: match security_level {
                SandboxSecurityLevel::Low => 2048,
                SandboxSecurityLevel::Standard => 1536,
                SandboxSecurityLevel::Medium => 1024,
                SandboxSecurityLevel::High => 512,
                SandboxSecurityLevel::Strict => 256,
            },
            cpu_limit_percent: match security_level {
                SandboxSecurityLevel::Low => 90.0,
                SandboxSecurityLevel::Standard => 85.0,
                SandboxSecurityLevel::Medium => 80.0,
                SandboxSecurityLevel::High => 60.0,
                SandboxSecurityLevel::Strict => 30.0,
            },
            timeout_seconds: match security_level {
                SandboxSecurityLevel::Low => 300,
                SandboxSecurityLevel::Standard => 240,
                SandboxSecurityLevel::Medium => 180,
                SandboxSecurityLevel::High => 60,
                SandboxSecurityLevel::Strict => 30,
            },
            environment_variables: std::collections::HashMap::new(),
            working_directory: None,
            allowed_syscalls: Some(vec!["read".to_string(), "write".to_string(), "open".to_string(), "close".to_string()]),
            blocked_syscalls: Some(vec!["execve".to_string(), "fork".to_string(), "socket".to_string()]),
        };
        
        // 记录配置调整
        let new_policy_level = match security_assessment.risk_level.as_str() {
            "低" => SecurityPolicyLevel::Low,
            "中等" => SecurityPolicyLevel::Standard,
            "高" => SecurityPolicyLevel::High,
            "极高" => SecurityPolicyLevel::Strict,
            _ => SecurityPolicyLevel::Standard,
        };
        
        if current_policy.level != new_policy_level {
            let mut details = HashMap::new();
            details.insert("previous_level".to_string(), format!("{:?}", current_policy.level));
            details.insert("new_level".to_string(), format!("{:?}", security_level));
            details.insert("risk_score".to_string(), security_assessment.security_score.to_string());
            
            self.security_manager.log_policy_change(
                &algorithm.id,
                &format!("根据风险评估自动调整安全配置，风险级别: {}", security_assessment.risk_level),
                &details,
            )?;
        }
        
        Ok(AdjustedSecurityConfig {
            security_level,
            resource_limits,
            sandbox_config,
            adjustment_reason: format!(
                "基于风险评估自动调整 - 风险级别: {}, 评分: {:.1}",
                security_assessment.risk_level, security_assessment.security_score
            ),
            applied_at: chrono::Utc::now(),
        })
    }
}

/// 调整后的安全配置
pub struct AdjustedSecurityConfig {
    /// 安全级别
    pub security_level: SandboxSecurityLevel,
    /// 资源限制
    pub resource_limits: ResourceLimits,
    /// 沙箱配置
    pub sandbox_config: SandboxConfig,
    /// 调整原因
    pub adjustment_reason: String,
    /// 应用时间
    pub applied_at: chrono::DateTime<chrono::Utc>,
}

// 转换函数：将执行器的SandboxType转换为通用types的SandboxType
fn convert_executor_to_types_sandbox_type(executor_type: crate::algorithm::executor::config::SandboxType) -> crate::algorithm::types::SandboxType {
    match executor_type {
        crate::algorithm::executor::config::SandboxType::LocalProcess => crate::algorithm::types::SandboxType::Process,
        crate::algorithm::executor::config::SandboxType::IsolatedProcess => crate::algorithm::types::SandboxType::Container,
        crate::algorithm::executor::config::SandboxType::Process => crate::algorithm::types::SandboxType::Process,
        crate::algorithm::executor::config::SandboxType::Wasm => crate::algorithm::types::SandboxType::Virtual,
        crate::algorithm::executor::config::SandboxType::Docker => crate::algorithm::types::SandboxType::Container,
    }
} 