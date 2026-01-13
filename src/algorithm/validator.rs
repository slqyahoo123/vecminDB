use crate::algorithm::types::{Algorithm, AlgorithmTask, ResourceLimits};
use crate::error::{Error, Result};
use std::collections::HashMap;
use log::{warn, error, debug};
use regex::Regex;
use std::time::Duration;
use std::collections::HashSet;
use serde_json;
use rand;

// 添加对WASM相关模块的引用
#[cfg(feature = "wasmtime")]
use wasmparser::{Parser, Payload, Validator, WasmFeatures};
use sha2::{Sha256, Digest};
use crate::algorithm::wasm;

// 引入生产级的安全策略和资源限制类型
use crate::algorithm::security::{SecurityPolicy, SecurityPolicyLevel};
use crate::algorithm::executor::sandbox::types::{SecurityContext};

// 添加缺失的依赖
use chrono;
use base64::{Engine as _, engine::general_purpose};

/// 算法验证器
/// 负责检查用户提交的算法是否符合安全标准
#[derive(Debug)]
pub struct AlgorithmValidator {
    /// 最大允许的代码大小（字节）
    max_code_size: usize,
    /// 禁止使用的API列表
    forbidden_apis: Vec<String>,
    /// 安全策略
    security_policy: SecurityPolicy,
    /// 资源限制
    resource_limits: ResourceLimits,
    /// 是否允许网络访问
    allow_network: bool,
    /// 是否允许文件系统访问
    allow_filesystem: bool,
    /// 最大验证超时时间
    validation_timeout: Duration,
    /// 安全检查器
    security_checker: Option<SecurityChecker>,
    /// WASM安全检查配置
    wasm_security_config: WasmSecurityConfig,
    /// 高级验证规则
    advanced_rules: AdvancedValidationRules,
    /// 沙箱配置
    sandbox_config: SecurityContext,
}

/// 高级验证规则
#[derive(Debug, Clone)]
pub struct AdvancedValidationRules {
    /// 最大函数复杂度
    pub max_function_complexity: usize,
    /// 最大嵌套层级
    pub max_nesting_level: usize,
    /// 最大循环数量
    pub max_loops_count: usize,
    /// 禁止的代码模式
    pub forbidden_patterns: Vec<String>,
    /// 强制安全注解
    pub require_security_annotations: bool,
    /// 数据流分析
    pub enable_dataflow_analysis: bool,
    /// 控制流分析
    pub enable_controlflow_analysis: bool,
}

impl Default for AdvancedValidationRules {
    fn default() -> Self {
        Self {
            max_function_complexity: 50,
            max_nesting_level: 10,
            max_loops_count: 20,
            forbidden_patterns: vec![
                r"eval\s*\(".to_string(),
                r"exec\s*\(".to_string(),
                r"__import__\s*\(".to_string(),
                r"globals\s*\(\s*\)".to_string(),
                r"locals\s*\(\s*\)".to_string(),
                r"setattr\s*\(".to_string(),
                r"getattr\s*\(".to_string(),
                r"hasattr\s*\(".to_string(),
                r"delattr\s*\(".to_string(),
                r"compile\s*\(".to_string(),
                r"__.*__".to_string(), // 魔术方法
            ],
            require_security_annotations: false,
            enable_dataflow_analysis: true,
            enable_controlflow_analysis: true,
        }
    }
}

/// WASM安全检查配置
#[derive(Debug, Clone)]
pub struct WasmSecurityConfig {
    /// 最大允许的WASM二进制大小（字节）
    max_wasm_size: usize,
    /// 最大允许的内存页数
    max_memory_pages: u32,
    /// 最大允许的表项数
    max_table_size: u32,
    /// 最大允许的导入函数数量
    max_imports: usize,
    /// 最大允许的导出函数数量
    max_exports: usize,
    /// 允许的导入模块名称
    allowed_import_modules: HashSet<String>,
    /// 禁止的导入函数名称
    forbidden_import_functions: HashSet<String>,
    /// 最大全局变量数量
    max_globals: usize,
    /// 最大函数数量
    max_functions: usize,
    /// 最大代码段大小
    max_code_section_size: usize,
    /// 最大数据段大小
    max_data_section_size: usize,
    /// 是否允许多值返回
    allow_multi_value: bool,
    /// 是否允许批量内存操作
    allow_bulk_memory: bool,
    /// 是否允许引用类型
    allow_reference_types: bool,
    /// 是否允许SIMD指令
    allow_simd: bool,
    /// 是否允许线程功能
    allow_threads: bool,
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
        // 禁止的网络操作
        forbidden_import_functions.insert("sock_open".to_string());
        forbidden_import_functions.insert("sock_connect".to_string());
        forbidden_import_functions.insert("sock_listen".to_string());
        
        Self {
            max_wasm_size: 10 * 1024 * 1024, // 10MB
            max_memory_pages: 1000,          // 约64MB内存
            max_table_size: 10000,
            max_imports: 100,
            max_exports: 100,
            allowed_import_modules,
            forbidden_import_functions,
            max_globals: 1000,
            max_functions: 10000,
            max_code_section_size: 8 * 1024 * 1024, // 8MB
            max_data_section_size: 2 * 1024 * 1024, // 2MB
            allow_multi_value: false,
            allow_bulk_memory: false,
            allow_reference_types: false,
            allow_simd: false,
            allow_threads: false,
        }
    }
}

impl Default for AlgorithmValidator {
    fn default() -> Self {
        let security_policy = SecurityPolicy::default();
        let resource_limits = ResourceLimits::default();
        let sandbox_config = SecurityContext::default();
        
        Self {
            max_code_size: 1024 * 1024, // 1MB
            forbidden_apis: vec![
                "std::process".to_string(),
                "std::env".to_string(),
                "std::fs::write".to_string(),
                "std::fs::create_dir".to_string(),
                "std::fs::remove".to_string(),
                "std::net".to_string(),
                "unsafe".to_string(),
                "libc::".to_string(),
                "winapi::".to_string(),
                "syscall".to_string(),
                "asm!".to_string(),
                "global_asm!".to_string(),
            ],
            security_policy,
            resource_limits,
            allow_network: false,
            allow_filesystem: false,
            validation_timeout: Duration::from_secs(30),
            security_checker: None,
            wasm_security_config: WasmSecurityConfig::default(),
            advanced_rules: AdvancedValidationRules::default(),
            sandbox_config,
        }
    }
}

impl AlgorithmValidator {
    /// 创建新的算法验证器
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 创建具有特定安全级别的验证器
    pub fn with_security_level(security_level: SecurityPolicyLevel) -> Self {
        let mut validator = Self::new();
        
        // 根据安全级别调整配置
        match security_level {
            SecurityPolicyLevel::Low => {
                validator.security_policy.level = SecurityPolicyLevel::Low;
                validator.security_policy.allow_network = true;
                validator.security_policy.allow_filesystem = true;
                validator.security_policy.max_memory_mb = 2048;
                validator.security_policy.max_cpu_time_ms = 120000;
                validator.advanced_rules.max_function_complexity = 100;
                validator.advanced_rules.max_nesting_level = 15;
            },
            SecurityPolicyLevel::Standard => {
                validator.security_policy.level = SecurityPolicyLevel::Standard;
                validator.security_policy.allow_network = false;
                validator.security_policy.allow_filesystem = false;
                validator.security_policy.max_memory_mb = 1024;
                validator.security_policy.max_cpu_time_ms = 60000;
            },
            SecurityPolicyLevel::High => {
                validator.security_policy.level = SecurityPolicyLevel::High;
                validator.security_policy.allow_network = false;
                validator.security_policy.allow_filesystem = false;
                validator.security_policy.max_memory_mb = 512;
                validator.security_policy.max_cpu_time_ms = 30000;
                validator.advanced_rules.max_function_complexity = 25;
                validator.advanced_rules.max_nesting_level = 5;
                validator.advanced_rules.require_security_annotations = true;
            },
            SecurityPolicyLevel::Strict => {
                validator.security_policy.level = SecurityPolicyLevel::Strict;
                validator.security_policy.allow_network = false;
                validator.security_policy.allow_filesystem = false;
                validator.security_policy.max_memory_mb = 256;
                validator.security_policy.max_cpu_time_ms = 15000;
                validator.advanced_rules.max_function_complexity = 15;
                validator.advanced_rules.max_nesting_level = 3;
                validator.advanced_rules.require_security_annotations = true;
                validator.advanced_rules.enable_dataflow_analysis = true;
                validator.advanced_rules.enable_controlflow_analysis = true;
            },
        }
        
        // 同步更新资源限制
        validator.resource_limits.max_memory = validator.security_policy.max_memory_mb * 1024 * 1024;
        validator.resource_limits.max_cpu_time = validator.security_policy.max_cpu_time_ms;
        validator.resource_limits.max_code_size = validator.security_policy.max_code_size_kb * 1024;
        
        // 同步更新沙箱配置
        validator.sandbox_config.allow_network = validator.security_policy.allow_network;
        validator.sandbox_config.allow_filesystem = validator.security_policy.allow_filesystem;
        validator.sandbox_config.memory_limit_bytes = validator.resource_limits.max_memory;
        validator.sandbox_config.cpu_time_limit_ms = validator.resource_limits.max_cpu_time;
        
        validator
    }
    
    /// 设置最大代码大小
    pub fn with_max_code_size(mut self, size: usize) -> Self {
        self.max_code_size = size;
        self.resource_limits.max_code_size = size;
        self
    }
    
    /// 设置禁止使用的API
    pub fn with_forbidden_apis(mut self, apis: Vec<String>) -> Self {
        self.forbidden_apis = apis;
        self
    }
    
    /// 设置安全策略
    pub fn with_security_policy(mut self, policy: SecurityPolicy) -> Self {
        self.security_policy = policy;
        // 同步更新其他配置
        self.allow_network = self.security_policy.allow_network;
        self.allow_filesystem = self.security_policy.allow_filesystem;
        self.resource_limits.max_memory = self.security_policy.max_memory_mb * 1024 * 1024;
        self.resource_limits.max_cpu_time = self.security_policy.max_cpu_time_ms;
        self
    }
    
    /// 设置资源限制
    pub fn with_resource_limits(mut self, limits: ResourceLimits) -> Self {
        self.resource_limits = limits;
        // 同步更新沙箱配置
        self.sandbox_config.memory_limit_bytes = self.resource_limits.max_memory;
        self.sandbox_config.cpu_time_limit_ms = self.resource_limits.max_cpu_time;
        self
    }
    
    /// 设置是否允许网络访问
    pub fn with_network_access(mut self, allow: bool) -> Self {
        self.allow_network = allow;
        self.security_policy.allow_network = allow;
        self.sandbox_config.allow_network = allow;
        self
    }
    
    /// 设置是否允许文件系统访问
    pub fn with_filesystem_access(mut self, allow: bool) -> Self {
        self.allow_filesystem = allow;
        self.security_policy.allow_filesystem = allow;
        self.sandbox_config.allow_filesystem = allow;
        self
    }
    
    /// 设置验证超时时间
    pub fn with_validation_timeout(mut self, timeout: Duration) -> Self {
        self.validation_timeout = timeout;
        self
    }
    
    /// 设置高级验证规则
    pub fn with_advanced_rules(mut self, rules: AdvancedValidationRules) -> Self {
        self.advanced_rules = rules;
        self
    }
    
    /// 添加安全检查器
    pub fn with_security_checker(mut self, checker: SecurityChecker) -> Self {
        self.security_checker = Some(checker);
        self
    }
    
    /// 获取当前安全策略
    pub fn get_security_policy(&self) -> &SecurityPolicy {
        &self.security_policy
    }
    
    /// 获取当前资源限制
    pub fn get_resource_limits(&self) -> &ResourceLimits {
        &self.resource_limits
    }
    
    /// 获取沙箱配置
    pub fn get_sandbox_config(&self) -> &SecurityContext {
        &self.sandbox_config
    }
    
    /// 验证算法
    pub fn validate(&self, algorithm: &Algorithm) -> Result<ValidationReport> {
        let start_time = std::time::Instant::now();
        
        let mut report = ValidationReport {
            algorithm_id: algorithm.id.clone(),
            passed: true,
            security_score: 100.0,
            issues: Vec::new(),
            warnings: Vec::new(),
            metadata: HashMap::new(),
        };
        
        // 添加验证开始信息
        report.metadata.insert("validation_start_time".to_string(), 
                              chrono::Utc::now().to_rfc3339());
        report.metadata.insert("validator_version".to_string(), 
                              env!("CARGO_PKG_VERSION").to_string());
        report.metadata.insert("security_level".to_string(), 
                              format!("{:?}", self.security_policy.level));
        
        // 1. 验证基本属性
        self.validate_basic_attributes(algorithm, &mut report)?;
        
        // 2. 运行高级验证规则
        self.validate_advanced_rules(algorithm, &mut report)?;
        
        // 3. 运行安全检查
        if let Some(checker) = &self.security_checker {
            checker.check_security(algorithm, &mut report)?;
        } else {
            // 使用内置安全检查
            self.run_builtin_security_checks(algorithm, &mut report)?;
        }
        
        // 4. 验证WASM代码（如果适用）
        if algorithm.metadata.get("type").map(|t| t == "wasm").unwrap_or(false) {
            if let Some(wasm_code) = algorithm.metadata.get("wasm_binary") {
                let wasm_bytes = general_purpose::STANDARD.decode(wasm_code)
                    .map_err(|e| Error::validation(format!("Invalid WASM binary encoding: {}", e)))?;
                let wasm_report = self.validate_wasm(&algorithm.id, &wasm_bytes)?;
                
                // 合并WASM验证结果
                report.issues.extend(wasm_report.issues);
                report.warnings.extend(wasm_report.warnings);
                if !wasm_report.passed {
                    report.passed = false;
                }
                report.security_score = report.security_score.min(wasm_report.security_score);
            }
        }
        
        // 5. 测试在沙箱中运行
        self.test_run_in_sandbox(algorithm, &mut report)?;
        
        // 6. 计算最终安全分数和通过状态
        let error_count = report.issues.iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count();
            
        let warning_count = report.issues.iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .count();
            
        if error_count > 0 {
            report.passed = false;
            report.security_score = 0.0;
        } else {
            // 根据警告数量调整安全分数
            let penalty = (warning_count as f32) * 5.0;
            report.security_score = (100.0 - penalty).max(0.0);
        }
        
        // 添加验证完成信息
        let validation_duration = start_time.elapsed();
        report.metadata.insert("validation_duration_ms".to_string(), 
                              validation_duration.as_millis().to_string());
        report.metadata.insert("validation_end_time".to_string(), 
                              chrono::Utc::now().to_rfc3339());
        report.metadata.insert("total_issues".to_string(), 
                              report.issues.len().to_string());
        report.metadata.insert("error_count".to_string(), 
                              error_count.to_string());
        report.metadata.insert("warning_count".to_string(), 
                              warning_count.to_string());
        
        debug!("Algorithm validation completed for {} in {:?}: passed={}, score={:.2}", 
               algorithm.id, validation_duration, report.passed, report.security_score);
        
        Ok(report)
    }
    
    /// 验证代码大小
    fn validate_code_size(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        let code_size = algorithm.code.len();
        debug!("代码大小: {} 字节", code_size);
        
        if code_size > self.max_code_size {
            report.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "CODE_SIZE_EXCEEDED".to_string(),
                message: format!("代码大小 ({} 字节) 超过最大限制 ({} 字节)", 
                                 code_size, self.max_code_size),
            });
        } else if code_size > self.max_code_size / 2 {
            report.warnings.push(format!(
                "代码大小 ({} 字节) 接近最大限制 ({} 字节)", 
                code_size, self.max_code_size
            ));
        }
        
        Ok(())
    }
    
    /// 验证代码中是否包含禁止的API
    fn validate_forbidden_apis(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        for api in &self.forbidden_apis {
            // 使用正则表达式匹配，以支持更复杂的模式
            let pattern = format!(r"\b{}\b", regex::escape(api));
            let regex = Regex::new(&pattern).map_err(|e| {
                error!("正则表达式编译错误: {} ", e);
                Error::Internal(format!("正则表达式编译错误: {} ", e))
            })?;
            
            if regex.is_match(&algorithm.code) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "FORBIDDEN_API".to_string(),
                    message: format!("代码中包含禁止使用的API: {} ", api),
                });
            }
        }
        
        Ok(())
    }
    
    /// 验证代码语法
    fn validate_syntax(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        debug!("开始验证代码语法");
        
        // 使用临时文件保存代码
        let temp_dir = std::env::temp_dir();
        let file_name = format!("algorithm_syntax_check_{}.rs", algorithm.id);
        let file_path = temp_dir.join(file_name);
        
        std::fs::write(&file_path, &algorithm.code)
            .map_err(|e| Error::internal(format!("写入临时文件失败: {}", e)))?;
        
        // 使用rustc检查语法
        let output = std::process::Command::new("rustc")
            .arg("--edition=2021")
            .arg("--error-format=json")
            .arg("--emit=metadata")
            .arg("-Z")
            .arg("no-codegen")
            .arg(&file_path)
            .output()
            .map_err(|e| Error::internal(format!("运行语法检查失败: {}", e)))?;
        
        // 删除临时文件
        if let Err(e) = std::fs::remove_file(&file_path) {
            warn!("删除临时文件失败: {}", e);
        }
        
        // 检查编译输出
        if !output.status.success() {
            let error_output = String::from_utf8_lossy(&output.stderr);
            
            // 解析JSON输出
            if let Ok(errors) = serde_json::from_str::<Vec<serde_json::Value>>(&error_output) {
                for error in errors {
                    if let Some(message) = error.get("message").and_then(|m| m.get("message")).and_then(|m| m.as_str()) {
                        report.issues.push(ValidationIssue {
                            severity: IssueSeverity::Error,
                            code: "SYNTAX_ERROR".to_string(),
                            message: message.to_string(),
                        });
                    }
                }
            } else {
                // 如果无法解析JSON输出，使用原始错误信息
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "SYNTAX_ERROR".to_string(),
                    message: format!("代码语法错误: {}", error_output),
                });
            }
            
            // 添加语法错误的详细信息到警告中
            report.warnings.push(format!(
                "代码语法检查失败，请修正语法错误后重试。编译器输出: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        } else {
            debug!("代码语法检查通过");
        }
        
        Ok(())
    }
    
    /// 验证资源使用
    fn validate_resource_usage(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        debug!("验证资源使用");
        
        // 1. 检查代码中的循环和递归，防止潜在的无限循环
        let code = &algorithm.code;
        
        // 检查潜在的无限循环
        let loop_patterns = vec![
            r"loop\s*{(?![^{}]*break)",  // 没有break的loop
            r"while\s+true\s*{(?![^{}]*break)",  // 没有break的while true
            r"for\s+.+\s+in\s+.+\s*{(?![^{}]*break)",  // 没有break的for循环
        ];
        
        for pattern in loop_patterns {
            let regex = Regex::new(pattern).map_err(|e| {
                error!("正则表达式编译错误: {} ", e);
                Error::internal(format!("正则表达式编译错误: {} ", e))
            })?;
            
            if regex.is_match(code) {
                report.warnings.push(format!(
                    "检测到可能的无限循环，请确保所有循环都有明确的终止条件 "
                ));
                break;
            }
        }
        
        // 2. 检查可能的内存泄漏
        let leak_patterns = vec![
            r"Box::leak",
            r"std::mem::forget",
            r"core::mem::forget",
            r"ManuallyDrop",
        ];
        
        for pattern in leak_patterns {
            let regex = Regex::new(pattern).map_err(|e| {
                error!("正则表达式编译错误: {} ", e);
                Error::internal(format!("正则表达式编译错误: {} ", e))
            })?;
            
            if regex.is_match(code) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "POTENTIAL_MEMORY_LEAK".to_string(),
                    message: format!("代码中使用了可能导致内存泄漏的API: {} ", pattern),
                });
            }
        }
        
        // 3. 检查大型内存分配
        let large_allocation_patterns = vec![
            r"vec!\[[^\]]{1000,}\]",  // 大型向量字面量
            r"[\[\(](\s*0\s*,\s*){1000,}",  // 大型数组或元组
            r"\[\s*0\s*;\s*[0-9]{5,}\s*\]",  // 使用重复语法的大型数组
        ];
        
        for pattern in large_allocation_patterns {
            let regex = Regex::new(pattern).map_err(|e| {
                error!("正则表达式编译错误: {} ", e);
                Error::internal(format!("正则表达式编译错误: {} ", e))
            })?;
            
            if regex.is_match(code) {
                report.warnings.push(format!(
                    "检测到大型内存分配，可能导致性能问题或内存不足 "
                ));
                break;
            }
        }
        
        Ok(())
    }
    
    /// 验证网络访问
    fn validate_network_access(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if !self.allow_network {
            // 检查是否包含网络相关API
            let network_patterns = [
                r"\bTcpStream\b",
                r"\bUdpSocket\b",
                r"\bconnect\s*\(",
                r"\blisten\s*\(",
                r"\bhyper::",
                r"\breqwest::",
                r"\bcurl::",
                r"\bsocket\s*\(",
            ];
            
            for pattern in &network_patterns {
                if Regex::new(pattern).unwrap().is_match(&algorithm.code) {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "NETWORK_ACCESS_FORBIDDEN".to_string(),
                        message: "代码尝试进行网络访问，但当前策略不允许".to_string(),
                    });
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    /// 验证文件系统访问
    fn validate_filesystem_access(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if !self.allow_filesystem {
            // 检查是否包含文件系统相关API
            let fs_patterns = [
                r"\bFile::",
                r"\bOpenOptions\b",
                r"\bfs::",
                r"\bopen\s*\(",
                r"\bread\s*\(",
                r"\bwrite\s*\(",
                r"\bcreate\s*\(",
                r"\bremove\s*\(",
            ];
            
            for pattern in &fs_patterns {
                if Regex::new(pattern).unwrap().is_match(&algorithm.code) {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "FILESYSTEM_ACCESS_FORBIDDEN".to_string(),
                        message: "代码尝试进行文件系统访问，但当前策略不允许".to_string(),
                    });
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    /// 验证算法接口兼容性
    fn validate_interface_compatibility(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        // 检查是否实现了必要的接口函数
        let required_functions = match algorithm.task {
            AlgorithmTask::Classification => vec!["predict", "train"],
            AlgorithmTask::Regression => vec!["predict", "train"],
            AlgorithmTask::Clustering => vec!["cluster", "fit"],
            AlgorithmTask::Embedding => vec!["embed", "fit"],
            AlgorithmTask::Custom => vec![],  // 自定义任务没有固定接口要求
        };
        
        for func in required_functions {
            let pattern = format!(r"fn\s+{}\s*\(", func);
            if !Regex::new(&pattern).unwrap().is_match(&algorithm.code) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "MISSING_INTERFACE".to_string(),
                    message: format!("缺少必要的接口函数: {}", func),
                });
            }
        }
        
        Ok(())
    }
    
    /// 在安全沙箱中测试运行算法
    fn test_run_in_sandbox(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        debug!("在安全沙箱中测试运行算法 {}", algorithm.metadata.name);
        
        // 解析算法代码
        let parsed_code = match self.parse_algorithm_code(algorithm) {
            Ok(code) => code,
            Err(e) => {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "CODE_PARSE_ERROR".to_string(),
                    message: format!("算法代码解析失败: {}", e),
                });
                return Ok(());
            }
        };
        
        // 创建资源限制
        let resource_limits = ResourceLimits {
            max_memory: self.resource_limits.max_memory,
            max_cpu_time: self.validation_timeout,
            max_disk_io: self.resource_limits.max_disk_io,
        };
        
        // 创建权限配置
        let permissions = SandboxPermissions {
            allow_network: self.allow_network,
            allow_filesystem: self.allow_filesystem,
            allow_syscalls: self.security_policy.allowed_syscalls.clone(),
        };
        
        // 准备测试数据
        let test_input = self.prepare_test_data(algorithm)?;
        
        // 安全漏洞检测
        self.detect_security_vulnerabilities(algorithm, report)?;
        
        // 数据泄露检测
        self.detect_data_leakage(algorithm, report)?;
        
        // 恶意代码检测
        self.detect_malicious_code(algorithm, report)?;
        
        // 依赖分析
        self.analyze_dependencies(algorithm, report)?;
        
        // 记录开始时间
        let start_time = std::time::Instant::now();
        
        // 尝试在沙箱中执行算法
        let execution_result = match self.execute_in_sandbox(&parsed_code, &test_input, &resource_limits, &permissions) {
            Ok(result) => result,
            Err(e) => {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "SANDBOX_EXECUTION_ERROR".to_string(),
                    message: format!("沙箱执行失败: {}", e),
                });
                return Ok(());
            }
        };
        
        // 计算执行时间
        let execution_time_ms = start_time.elapsed().as_millis();
        
        // 检查执行结果
        let sandbox_test_passed = execution_result.success;
        let resource_exceeded = execution_result.resource_exceeded;
        
        // 根据执行结果添加验证问题
        if !sandbox_test_passed {
            report.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "SANDBOX_TEST_FAILED".to_string(),
                message: format!("安全沙箱测试失败: {}", execution_result.error_message.unwrap_or_else(|| "未知错误".to_string())),
            });
        }
        
        if resource_exceeded {
            let resource_type = if execution_result.memory_exceeded {
                "内存"
            } else if execution_result.cpu_exceeded {
                "CPU时间"
            } else if execution_result.io_exceeded {
                "磁盘IO"
            } else {
                "未知资源"
            };
            
            report.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "RESOURCE_LIMIT_EXCEEDED".to_string(),
                message: format!("算法执行超出{}限制", resource_type),
            });
        }
        
        // 添加权限违规警告
        if !self.allow_network && execution_result.attempted_network_access {
            report.issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                code: "NETWORK_ACCESS_ATTEMPT".to_string(),
                message: "算法尝试进行网络访问，但已被阻止".to_string(),
            });
        }
        
        if !self.allow_filesystem && execution_result.attempted_filesystem_access {
            report.issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                code: "FILESYSTEM_ACCESS_ATTEMPT".to_string(),
                message: "算法尝试进行文件系统访问，但已被阻止".to_string(),
            });
        }
        
        // 添加执行时间警告
        if execution_time_ms > 1000 {
            report.warnings.push(format!(
                "算法执行时间较长: {} ms", execution_time_ms
            ));
        }
        
        // 添加内存使用警告
        if execution_result.peak_memory > 1024 * 1024 * 50 { // 50MB
            report.warnings.push(format!(
                "算法内存使用较高: {} MB", execution_result.peak_memory / (1024 * 1024)
            ));
        }
        
        // 分析执行结果
        self.analyze_execution_result(&execution_result, report)?;
        
        Ok(())
    }
    
    /// 解析算法代码
    fn parse_algorithm_code(&self, algorithm: &Algorithm) -> Result<ParsedCode> {
        // 在实际实现中，这里应该解析算法代码
        // 简化实现返回一个代表已解析代码的占位符
        Ok(ParsedCode {
            ast: "算法AST表示".to_string(),
            functions: vec!["main".to_string(), "process".to_string()],
            imports: vec![],
        })
    }
    
    /// 准备测试数据
    fn prepare_test_data(&self, algorithm: &Algorithm) -> Result<Vec<u8>> {
        debug!("准备算法测试数据");
        
        // 根据算法类型准备不同的测试数据
        match algorithm.task {
            AlgorithmTask::Classification => {
                // 创建分类问题的测试数据
                let mut data = Vec::new();
                
                // 添加10个特征变量
                for _ in 0..10 {
                    data.extend_from_slice(&(rand::random::<f32>() * 10.0).to_ne_bytes());
                }
                
                Ok(data)
            },
            AlgorithmTask::Regression => {
                // 创建回归问题的测试数据
                let mut data = Vec::new();
                
                // 创建线性关系的数据
                for i in 0..20 {
                    let x = i as f32 / 10.0;
                    data.extend_from_slice(&x.to_ne_bytes());
                    data.extend_from_slice(&(2.0 * x + 1.0 + rand::random::<f32>() * 0.1).to_ne_bytes());
                }
                
                Ok(data)
            },
            AlgorithmTask::Clustering => {
                // 创建聚类问题的测试数据
                let mut data = Vec::new();
                
                // 创建两个簇的数据
                for _ in 0..15 {
                    // 第一个簇
                    data.extend_from_slice(&(rand::random::<f32>() * 2.0).to_ne_bytes());
                    data.extend_from_slice(&(rand::random::<f32>() * 2.0).to_ne_bytes());
                }
                
                for _ in 0..15 {
                    // 第二个簇
                    data.extend_from_slice(&(5.0 + rand::random::<f32>() * 2.0).to_ne_bytes());
                    data.extend_from_slice(&(5.0 + rand::random::<f32>() * 2.0).to_ne_bytes());
                }
                
                Ok(data)
            },
            AlgorithmTask::Embedding => {
                // 创建嵌入问题的测试数据
                let text_samples = [
                    "这是第一个测试样本",
                    "这是另一个样本，用于测试嵌入算法",
                    "我们需要多样化的文本来测试",
                    "自然语言处理是AI的重要分支",
                ];
                
                let mut data = Vec::new();
                for text in &text_samples {
                    data.extend_from_slice(text.as_bytes());
                    data.push(0); // 添加分隔符
                }
                
                Ok(data)
            },
            AlgorithmTask::Custom => {
                // 对于自定义任务，创建通用测试数据
                let mut data = Vec::new();
                
                // 添加整数数据
                for i in 0..20 {
                    data.extend_from_slice(&i.to_ne_bytes());
                }
                
                // 添加浮点数据
                for i in 0..20 {
                    let f = i as f32 * 1.5;
                    data.extend_from_slice(&f.to_ne_bytes());
                }
                
                // 添加简单的字符串数据
                data.extend_from_slice("测试数据字符串".as_bytes());
                
                Ok(data)
            },
        }
    }
    
    /// 在沙箱中执行算法
    fn execute_in_sandbox(
        &self, 
        code: &ParsedCode, 
        input: &[u8], 
        limits: &ResourceLimits, 
        permissions: &SandboxPermissions
    ) -> Result<ExecutionResult> {
        // 在实际实现中，这里应该创建一个安全的沙箱环境并执行代码
        // 此处为简化实现，返回模拟的执行结果
        
        // 模拟执行过程
        let success = true; // 默认成功
        let resource_exceeded = false; // 默认资源充足
        
        Ok(ExecutionResult {
            success,
            resource_exceeded,
            memory_exceeded: false,
            cpu_exceeded: false,
            io_exceeded: false,
            peak_memory: 1024 * 1024 * 20, // 20MB
            execution_time_ms: 150,
            output: vec![5, 4, 3, 2, 1],
            error_message: None,
            attempted_network_access: false,
            attempted_filesystem_access: false,
        })
    }
    
    /// 安全漏洞检测
    fn detect_security_vulnerabilities(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        debug!("检查算法安全漏洞");
        
        // 检查常见的安全漏洞模式
        let vulnerability_patterns = [
            (r"eval\s*\(", "使用了eval函数，可能导致代码注入"),
            (r"exec\s*\(", "使用了exec函数，可能导致命令执行"),
            (r"unsafe\s+{", "使用了unsafe代码块，可能导致内存安全问题"),
            (r"std::ptr::(read|write)", "直接内存操作，可能导致内存安全问题"),
            (r"transmute\s*[<\(]", "使用了类型转换，可能导致类型安全问题"),
            (r"libc::(system|popen)", "调用系统命令，存在命令注入风险"),
            (r"rand::thread_rng\(\).gen_range", "使用不安全的随机数生成器"),
        ];
        
        for (pattern, message) in &vulnerability_patterns {
            if Regex::new(pattern).unwrap().is_match(&algorithm.code) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "SECURITY_VULNERABILITY".to_string(),
                    message: message.to_string(),
                });
            }
        }
        
        // 检查SQL注入风险
        if algorithm.code.contains("SELECT") || algorithm.code.contains("INSERT") || 
           algorithm.code.contains("UPDATE") || algorithm.code.contains("DELETE") {
            if !algorithm.code.contains("prepare") && !algorithm.code.contains("bind_param") {
                report.warnings.push(
                    "检测到SQL语句，但未发现参数化查询，可能存在SQL注入风险".to_string()
                );
            }
        }
        
        // 检查正则表达式DDoS风险
        if algorithm.code.contains("Regex::new") && 
           (algorithm.code.contains(".*") || algorithm.code.contains(".+")) {
            report.warnings.push(
                "检测到贪婪模式正则表达式，可能导致ReDoS攻击".to_string()
            );
        }
        
        Ok(())
    }
    
    /// 数据泄露检测
    fn detect_data_leakage(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        debug!("检查数据泄露风险");
        
        // 检查是否存在输出敏感数据的风险
        let data_leak_patterns = [
            (r"println!\s*\(\s*.*password", "打印密码相关信息"),
            (r"println!\s*\(\s*.*token", "打印token相关信息"),
            (r"println!\s*\(\s*.*key", "打印密钥相关信息"),
            (r"fs::write\s*\(.*password", "写入密码相关信息到文件"),
            (r"fs::write\s*\(.*token", "写入token相关信息到文件"),
            (r"fs::write\s*\(.*key", "写入密钥相关信息到文件"),
        ];
        
        for (pattern, message) in &data_leak_patterns {
            if Regex::new(pattern).unwrap().is_match(&algorithm.code) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "DATA_LEAKAGE_RISK".to_string(),
                    message: format!("可能的数据泄露风险: {}", message),
                });
            }
        }
        
        // 检查网络数据传输
        if Regex::new(r"TcpStream|UdpSocket|reqwest").unwrap().is_match(&algorithm.code) {
            if !algorithm.code.contains("encrypt") && !algorithm.code.contains("https://") {
                report.warnings.push(
                    "检测到网络数据传输，但未发现加密措施，可能导致数据泄露".to_string()
                );
            }
        }
        
        Ok(())
    }
    
    /// 恶意代码检测
    fn detect_malicious_code(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        debug!("检查恶意代码");
        
        // 检查常见的恶意代码模式
        let malicious_patterns = [
            (r"fork\s*\(", "可能尝试创建子进程"),
            (r"daemon\s*\(", "可能尝试创建守护进程"),
            (r"setsid\s*\(", "可能尝试创建新会话"),
            (r"rm\s+-rf|unlink\s*\(", "可能尝试删除文件"),
            (r"chmod\s+777", "可能尝试修改文件权限"),
            (r"nc\s+-e|netcat", "可能尝试创建反向shell"),
            (r"crontab\s+-e", "可能尝试修改计划任务"),
        ];
        
        for (pattern, message) in &malicious_patterns {
            if Regex::new(pattern).unwrap().is_match(&algorithm.code) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "MALICIOUS_CODE".to_string(),
                    message: format!("检测到可能的恶意代码: {}", message),
                });
            }
        }
        
        // 检查密码破解尝试
        if algorithm.code.contains("password") && 
           (algorithm.code.contains("brute") || algorithm.code.contains("crack")) {
            report.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "UNAUTHORIZED_ACTION".to_string(),
                message: "检测到可能的密码破解尝试".to_string(),
            });
        }
        
        // 检查挖矿代码
        let crypto_mining_keywords = ["miner", "mining", "bitcoin", "ethereum", "monero", "hashrate"];
        for keyword in &crypto_mining_keywords {
            if algorithm.code.contains(keyword) {
                report.warnings.push(
                    format!("检测到可能与加密货币挖矿相关的关键词: {}", keyword)
                );
            }
        }
        
        Ok(())
    }
    
    /// 依赖分析
    fn analyze_dependencies(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        debug!("分析算法依赖");
        
        // 提取extern crate和use语句
        let extern_regex = Regex::new(r"extern\s+crate\s+([a-zA-Z0-9_]+)").unwrap();
        let use_regex = Regex::new(r"use\s+([a-zA-Z0-9_:]+)").unwrap();
        
        let mut dependencies = HashSet::new();
        
        for capture in extern_regex.captures_iter(&algorithm.code) {
            if let Some(dependency) = capture.get(1) {
                dependencies.insert(dependency.as_str().to_string());
            }
        }
        
        for capture in use_regex.captures_iter(&algorithm.code) {
            if let Some(path) = capture.get(1) {
                let path_str = path.as_str();
                let parts: Vec<&str> = path_str.split("::").collect();
                if !parts.is_empty() {
                    dependencies.insert(parts[0].to_string());
                }
            }
        }
        
        // 检查未授权的依赖
        let unauthorized_dependencies = [
            "tokio_rusqlite", // 可能尝试访问SQLite数据库
            "diesel", // 可能尝试访问数据库
            "mongodb", // 可能尝试访问MongoDB
            "ssh2", // 可能尝试SSH连接
            "lettre", // 可能尝试发送邮件
        ];
        
        for dep in &unauthorized_dependencies {
            if dependencies.contains(*dep) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "UNAUTHORIZED_DEPENDENCY".to_string(),
                    message: format!("使用了未授权的依赖: {}", dep),
                });
            }
        }
        
        // 检查过时或有安全漏洞的依赖版本
        // 在实际环境中，这里应该连接到漏洞数据库进行检查
        if algorithm.metadata.contains_key("dependencies") {
            let deps_str = &algorithm.metadata["dependencies"];
            if deps_str.contains("openssl = \"0.9") || 
               deps_str.contains("chrono < \"0.4.20") ||
               deps_str.contains("time < \"0.2") {
                report.warnings.push(
                    "算法依赖包含已知存在安全漏洞的版本".to_string()
                );
            }
        }
        
        // 记录依赖数量
        if dependencies.len() > 10 {
            report.warnings.push(
                format!("算法使用了大量依赖({})，可能增加安全风险", dependencies.len())
            );
        }
        
        Ok(())
    }
    
    /// 分析执行结果
    fn analyze_execution_result(&self, result: &ExecutionResult, report: &mut ValidationReport) -> Result<()> {
        debug!("分析算法执行结果");
        
        // 检查输出结果大小
        if result.output.len() > 1024 * 1024 { // 1MB
            report.warnings.push(
                format!("算法输出结果较大 ({} KB)，可能导致内存问题", result.output.len() / 1024)
            );
        }
        
        // 检查算法稳定性
        if result.success && result.execution_time_ms < 10 {
            report.warnings.push(
                "算法执行时间过短，可能未进行实质性计算".to_string()
            );
        }
        
        // 检查资源使用效率
        let memory_efficiency = result.peak_memory as f64 / result.output.len() as f64;
        if memory_efficiency > 100.0 && result.output.len() > 1000 {
            report.warnings.push(
                format!("算法内存使用效率较低，每字节输出消耗 {:.2} 字节内存", memory_efficiency)
            );
        }
        
        Ok(())
    }

    /// 验证WASM二进制代码
    pub fn validate_wasm(&self, algorithm_id: &str, wasm_binary: &[u8]) -> Result<ValidationReport> {
        let mut report = ValidationReport {
            algorithm_id: algorithm_id.to_string(),
            passed: true,
            security_score: 100.0,
            issues: Vec::new(),
            warnings: Vec::new(),
            metadata: HashMap::new(),
        };

        // 1. 检查WASM二进制大小
        if wasm_binary.len() > self.wasm_security_config.max_wasm_size {
            report.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "WASM_SIZE_EXCEEDED".to_string(),
                message: format!("WASM二进制大小 ({} 字节) 超过最大限制 ({} 字节)", 
                                wasm_binary.len(), self.wasm_security_config.max_wasm_size),
            });
            report.passed = false;
        }

        // 2. 验证WASM二进制格式
        if let Err(e) = self.validate_wasm_format(wasm_binary, &mut report) {
            error!("验证WASM格式失败: {}", e);
            report.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "WASM_FORMAT_INVALID".to_string(),
                message: format!("WASM二进制格式无效: {}", e),
            });
            report.passed = false;
            return Ok(report);
        }

        // 3. 检查WASM安全规则
        if let Err(e) = self.check_wasm_security_rules(wasm_binary, &mut report) {
            error!("WASM安全规则检查失败: {}", e);
            report.issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "WASM_SECURITY_CHECK_FAILED".to_string(),
                message: format!("WASM安全规则检查失败: {}", e),
            });
            report.passed = false;
            return Ok(report);
        }

        // 4. 计算WASM二进制哈希值，用于缓存和安全审计
        let mut hasher = Sha256::new();
        hasher.update(wasm_binary);
        let hash = hasher.finalize();
        let hash_hex = format!("{:x}", hash);
        report.metadata.insert("wasm_hash".to_string(), hash_hex);

        // 根据问题数量调整安全分数
        let error_count = report.issues.iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .count();
            
        if error_count > 0 {
            report.passed = false;
            report.security_score = (100.0 - (error_count as f32 * 10.0)).max(0.0);
        }
        
        let warning_count = report.warnings.len();
        if warning_count > 0 {
            report.security_score -= (warning_count as f32 * 2.0).min(20.0);
            report.security_score = report.security_score.max(0.0);
        }

        Ok(report)
    }

    /// 验证WASM格式
    fn validate_wasm_format(&self, wasm_binary: &[u8], report: &mut ValidationReport) -> Result<()> {
        // 配置验证器功能支持
        let features = WasmFeatures {
            mutable_global: true,
            saturating_float_to_int: true,
            sign_extension: true,
            reference_types: true,
            multi_value: true,
            bulk_memory: true,
            simd: true,
            relaxed_simd: true,
            threads: true,
            tail_call: true,
            exceptions: true,
            memory64: true,
            extended_const: true,
            component_model: false,  // 暂不启用组件模型
            memory_control: true,
            gc: true,
            ..WasmFeatures::default()
        };
        
        let mut validator = Validator::new_with_features(features);
        
        // 解析并验证WASM模块
        for payload in Parser::new(0).parse_all(wasm_binary) {
            match payload {
                Ok(payload) => {
                    if let Err(e) = validator.payload(&payload) {
                        report.issues.push(ValidationIssue {
                            severity: IssueSeverity::Error,
                            code: "WASM_VALIDATION_ERROR".to_string(),
                            message: format!("WASM验证错误: {}", e),
                        });
                        return Err(Error::algorithm(format!("WASM格式验证失败: {}", e)));
                    }
                    
                    // 在这里添加对特定部分的检查
                    match payload {
                        Payload::MemorySection(reader) => {
                            for memory_result in reader {
                                if let Ok(memory) = memory_result {
                                    if memory.initial > self.wasm_security_config.max_memory_pages {
                                        report.issues.push(ValidationIssue {
                                            severity: IssueSeverity::Error,
                                            code: "WASM_MEMORY_EXCEEDED".to_string(),
                                            message: format!("WASM内存页数 ({}) 超过最大限制 ({})",
                                                           memory.initial, self.wasm_security_config.max_memory_pages),
                                        });
                                    }
                                }
                            }
                        },
                        Payload::TableSection(reader) => {
                            for table_result in reader {
                                if let Ok(table) = table_result {
                                    if table.initial > self.wasm_security_config.max_table_size {
                                        report.issues.push(ValidationIssue {
                                            severity: IssueSeverity::Error,
                                            code: "WASM_TABLE_EXCEEDED".to_string(),
                                            message: format!("WASM表大小 ({}) 超过最大限制 ({})",
                                                           table.initial, self.wasm_security_config.max_table_size),
                                        });
                                    }
                                }
                            }
                        },
                        Payload::ImportSection(reader) => {
                            let mut import_count = 0;
                            
                            for import_result in reader {
                                if let Ok(import) = import_result {
                                    import_count += 1;
                                    
                                    // 检查导入模块是否在允许列表中
                                    if !self.wasm_security_config.allowed_import_modules.contains(&import.module.to_string()) {
                                        report.issues.push(ValidationIssue {
                                            severity: IssueSeverity::Error,
                                            code: "WASM_FORBIDDEN_IMPORT_MODULE".to_string(),
                                            message: format!("WASM导入了禁止的模块: {}", import.module),
                                        });
                                    }
                                    
                                    // 检查导入函数是否在禁止列表中
                                    if self.wasm_security_config.forbidden_import_functions.contains(&import.name.to_string()) {
                                        report.issues.push(ValidationIssue {
                                            severity: IssueSeverity::Error,
                                            code: "WASM_FORBIDDEN_IMPORT_FUNCTION".to_string(),
                                            message: format!("WASM导入了禁止的函数: {}.{}", import.module, import.name),
                                        });
                                    }
                                }
                            }
                            
                            if import_count > self.wasm_security_config.max_imports {
                                report.issues.push(ValidationIssue {
                                    severity: IssueSeverity::Warning,
                                    code: "WASM_IMPORTS_EXCEEDED".to_string(),
                                    message: format!("WASM导入函数数量 ({}) 超过建议限制 ({})",
                                                   import_count, self.wasm_security_config.max_imports),
                                });
                            }
                        },
                        Payload::ExportSection(reader) => {
                            let mut export_count = 0;
                            
                            for export_result in reader {
                                if let Ok(_) = export_result {
                                    export_count += 1;
                                }
                            }
                            
                            if export_count > self.wasm_security_config.max_exports {
                                report.issues.push(ValidationIssue {
                                    severity: IssueSeverity::Warning,
                                    code: "WASM_EXPORTS_EXCEEDED".to_string(),
                                    message: format!("WASM导出函数数量 ({}) 超过建议限制 ({})",
                                                   export_count, self.wasm_security_config.max_exports),
                                });
                            }
                        },
                        _ => {}
                    }
                },
                Err(e) => {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "WASM_PARSE_ERROR".to_string(),
                        message: format!("WASM解析错误: {}", e),
                    });
                    return Err(Error::algorithm(format!("WASM解析失败: {}", e)));
                }
            }
        }
        
        Ok(())
    }

    /// 集成wasm模块中的安全规则检查
    fn check_wasm_security_rules(&self, wasm_binary: &[u8], report: &mut ValidationReport) -> Result<()> {
        let security_report = wasm::check_security(wasm_binary)?;
        
        // 检查是否通过安全验证
        if !security_report.is_safe {
            report.passed = false;
            
            // 添加错误到报告
            for error in security_report.errors {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "WASM_SECURITY_ERROR".to_string(),
                    message: error,
                });
            }
            
            // 添加警告
            for warning in security_report.warnings {
                report.warnings.push(warning);
            }
            
            return Err(Error::security("WASM模块未通过安全检查".to_string()));
        }
        
        Ok(())
    }

    /// 验证高级规则
    fn validate_advanced_rules(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        // 1. 检查函数复杂度
        self.check_function_complexity(algorithm, report)?;
        
        // 2. 检查嵌套层级
        self.check_nesting_level(algorithm, report)?;
        
        // 3. 检查循环数量
        self.check_loops_count(algorithm, report)?;
        
        // 4. 检查禁止模式
        self.check_forbidden_patterns(algorithm, report)?;
        
        // 5. 检查安全注解（如果启用）
        if self.advanced_rules.require_security_annotations {
            self.check_security_annotations(algorithm, report)?;
        }
        
        // 6. 数据流分析（如果启用）
        if self.advanced_rules.enable_dataflow_analysis {
            self.analyze_dataflow(algorithm, report)?;
        }
        
        // 7. 控制流分析（如果启用）
        if self.advanced_rules.enable_controlflow_analysis {
            self.analyze_controlflow(algorithm, report)?;
        }
        
        Ok(())
    }
    
    /// 检查函数复杂度
    fn check_function_complexity(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            // 简单的复杂度计算：计算条件分支、循环等
            let complexity_patterns = vec![
                r"if\s+", r"else\s+", r"elif\s+", r"while\s+", r"for\s+",
                r"try\s+", r"except\s+", r"catch\s+", r"switch\s+",
                r"case\s+", r"&&", r"\|\|", r"\?", r":"
            ];
            
            for pattern in complexity_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid complexity pattern: {}", e)))?;
                let count = regex.find_iter(code).count();
                
                if count > self.advanced_rules.max_function_complexity {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "COMPLEXITY_TOO_HIGH".to_string(),
                        message: format!("Function complexity too high: {} > {} for pattern '{}'", 
                                       count, self.advanced_rules.max_function_complexity, pattern),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查嵌套层级
    fn check_nesting_level(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            let mut max_nesting = 0;
            let mut current_nesting = 0;
            
            for line in code.lines() {
                let trimmed = line.trim();
                
                // 检查增加嵌套的关键词
                if trimmed.starts_with("if ") || trimmed.starts_with("for ") || 
                   trimmed.starts_with("while ") || trimmed.starts_with("try:") ||
                   trimmed.contains("function") || trimmed.contains("{") {
                    current_nesting += 1;
                    max_nesting = max_nesting.max(current_nesting);
                }
                
                // 检查减少嵌套的关键词
                if trimmed.starts_with("end") || trimmed.contains("}") {
                    current_nesting = current_nesting.saturating_sub(1);
                }
            }
            
            if max_nesting > self.advanced_rules.max_nesting_level {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "NESTING_TOO_DEEP".to_string(),
                    message: format!("Nesting level too deep: {} > {}", 
                                   max_nesting, self.advanced_rules.max_nesting_level),
                });
            }
        }
        
        Ok(())
    }
    
    /// 检查循环数量
    fn check_loops_count(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            let loop_patterns = vec![r"for\s+", r"while\s+", r"loop\s+", r"do\s+"];
            let mut total_loops = 0;
            
            for pattern in loop_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid loop pattern: {}", e)))?;
                total_loops += regex.find_iter(code).count();
            }
            
            if total_loops > self.advanced_rules.max_loops_count {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "TOO_MANY_LOOPS".to_string(),
                    message: format!("Too many loops: {} > {}", 
                                   total_loops, self.advanced_rules.max_loops_count),
                });
            }
        }
        
        Ok(())
    }
    
    /// 检查禁止的代码模式
    fn check_forbidden_patterns(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            for pattern in &self.advanced_rules.forbidden_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid forbidden pattern: {}", e)))?;
                
                if let Some(matched) = regex.find(code) {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "FORBIDDEN_PATTERN".to_string(),
                        message: format!("Forbidden pattern found: '{}' at position {}", 
                                       matched.as_str(), matched.start()),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查安全注解
    fn check_security_annotations(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            // 检查是否存在安全相关的注解或注释
            let security_annotations = vec![
                r"@security", r"@safe", r"@trusted", r"#\s*security:", r"//\s*security:",
                r"@permission", r"@access", r"@sandbox"
            ];
            
            let mut has_security_annotation = false;
            
            for annotation in security_annotations {
                let regex = Regex::new(annotation)
                    .map_err(|e| Error::validation(format!("Invalid security annotation pattern: {}", e)))?;
                
                if regex.is_match(code) {
                    has_security_annotation = true;
                    break;
                }
            }
            
            if !has_security_annotation {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "MISSING_SECURITY_ANNOTATIONS".to_string(),
                    message: "Algorithm lacks required security annotations".to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    /// 数据流分析
    fn analyze_dataflow(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            // 简单的数据流分析：检查变量使用和赋值
            let mut variables = HashSet::new();
            let mut used_variables = HashSet::new();
            
            // 查找变量声明
            let var_declaration_patterns = vec![
                r"(\w+)\s*=", r"let\s+(\w+)", r"var\s+(\w+)", r"const\s+(\w+)"
            ];
            
            for pattern in var_declaration_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid variable pattern: {}", e)))?;
                
                for capture in regex.captures_iter(code) {
                    if let Some(var_name) = capture.get(1) {
                        variables.insert(var_name.as_str().to_string());
                    }
                }
            }
            
            // 查找变量使用
            for var in &variables {
                let usage_regex = Regex::new(&format!(r"\b{}\b", regex::escape(var)))
                    .map_err(|e| Error::validation(format!("Invalid usage pattern: {}", e)))?;
                
                if usage_regex.find_iter(code).count() > 1 { // 声明+使用至少2次
                    used_variables.insert(var.clone());
                }
            }
            
            // 检查未使用的变量
            for var in &variables {
                if !used_variables.contains(var) {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Info,
                        code: "UNUSED_VARIABLE".to_string(),
                        message: format!("Variable '{}' is declared but never used", var),
                    });
                }
            }
            
            // 检查潜在的数据泄露
            let sensitive_operations = vec![
                r"print\s*\(", r"console\.log\s*\(", r"log\s*\(", r"debug\s*\(",
                r"alert\s*\(", r"confirm\s*\(", r"prompt\s*\("
            ];
            
            for pattern in sensitive_operations {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid sensitive operation pattern: {}", e)))?;
                
                if regex.is_match(code) {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        code: "POTENTIAL_DATA_LEAK".to_string(),
                        message: format!("Potential data leakage through: {}", pattern),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// 控制流分析
    fn analyze_controlflow(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            // 检查无限循环风险
            let infinite_loop_patterns = vec![
                r"while\s+true", r"while\s+1", r"for\s*\(\s*;\s*;\s*\)",
                r"loop\s*\{", r"while\s*\(\s*true\s*\)"
            ];
            
            for pattern in infinite_loop_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid infinite loop pattern: {}", e)))?;
                
                if regex.is_match(code) {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        code: "POTENTIAL_INFINITE_LOOP".to_string(),
                        message: format!("Potential infinite loop detected: {}", pattern),
                    });
                }
            }
            
            // 检查递归调用深度
            if let Some(algo_name) = algorithm.metadata.get("name") {
                let recursive_pattern = format!(r"\b{}\s*\(", regex::escape(algo_name));
                let regex = Regex::new(&recursive_pattern)
                    .map_err(|e| Error::validation(format!("Invalid recursive pattern: {}", e)))?;
                
                let recursive_calls = regex.find_iter(code).count();
                if recursive_calls > 0 {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        code: "RECURSIVE_CALLS".to_string(),
                        message: format!("Function contains {} recursive calls", recursive_calls),
                    });
                }
            }
            
            // 检查异常处理
            let exception_patterns = vec![r"try\s*\{", r"catch\s*\(", r"except\s*:", r"finally\s*\{"];
            let mut has_exception_handling = false;
            
            for pattern in exception_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid exception pattern: {}", e)))?;
                
                if regex.is_match(code) {
                    has_exception_handling = true;
                    break;
                }
            }
            
            // 如果代码复杂但缺少异常处理，给出建议
            if code.lines().count() > 20 && !has_exception_handling {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Info,
                    code: "MISSING_EXCEPTION_HANDLING".to_string(),
                    message: "Complex algorithm should include exception handling".to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    /// 运行内置安全检查
    fn run_builtin_security_checks(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        // 1. 检查危险导入
        self.check_dangerous_imports(algorithm, report)?;
        
        // 2. 检查系统调用
        self.check_system_calls(algorithm, report)?;
        
        // 3. 检查网络访问
        self.check_network_usage(algorithm, report)?;
        
        // 4. 检查文件系统访问
        self.check_filesystem_usage(algorithm, report)?;
        
        // 5. 检查内存分配
        self.check_memory_allocation(algorithm, report)?;
        
        // 6. 检查加密相关操作
        self.check_cryptographic_operations(algorithm, report)?;
        
        Ok(())
    }
    
    /// 检查危险导入
    fn check_dangerous_imports(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            let dangerous_imports = vec![
                r"import\s+os", r"import\s+sys", r"import\s+subprocess",
                r"import\s+socket", r"import\s+urllib", r"import\s+requests",
                r"from\s+os\s+import", r"from\s+sys\s+import", 
                r"from\s+subprocess\s+import", r"from\s+socket\s+import",
                r"#include\s*<unistd\.h>", r"#include\s*<sys/", 
                r"use\s+std::process", r"use\s+std::fs", r"use\s+std::net"
            ];
            
            for pattern in dangerous_imports {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid import pattern: {}", e)))?;
                
                if regex.is_match(code) {
                    // 检查是否在允许列表中
                    let is_allowed = self.security_policy.allowed_imports.iter()
                        .any(|allowed| pattern.contains(allowed));
                    
                    if !is_allowed {
                        report.issues.push(ValidationIssue {
                            severity: IssueSeverity::Error,
                            code: "DANGEROUS_IMPORT".to_string(),
                            message: format!("Dangerous import detected: {}", pattern),
                        });
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查系统调用
    fn check_system_calls(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            let system_call_patterns = vec![
                r"system\s*\(", r"exec\s*\(", r"shell\s*\(", r"popen\s*\(",
                r"os\.system", r"subprocess\.", r"Runtime\.getRuntime",
                r"ProcessBuilder", r"syscall", r"__asm__"
            ];
            
            for pattern in system_call_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid system call pattern: {}", e)))?;
                
                if regex.is_match(code) {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "SYSTEM_CALL_DETECTED".to_string(),
                        message: format!("System call detected: {}", pattern),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查网络使用
    fn check_network_usage(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            let network_patterns = vec![
                r"socket\s*\(", r"connect\s*\(", r"bind\s*\(", r"listen\s*\(",
                r"urllib\.", r"requests\.", r"http\.", r"https\.",
                r"fetch\s*\(", r"XMLHttpRequest", r"WebSocket",
                r"TcpStream::", r"UdpSocket::", r"reqwest::"
            ];
            
            for pattern in network_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid network pattern: {}", e)))?;
                
                if regex.is_match(code) {
                    if !self.security_policy.allow_network {
                        report.issues.push(ValidationIssue {
                            severity: IssueSeverity::Error,
                            code: "NETWORK_ACCESS_DENIED".to_string(),
                            message: format!("Network access not allowed: {}", pattern),
                        });
                    } else {
                        report.issues.push(ValidationIssue {
                            severity: IssueSeverity::Info,
                            code: "NETWORK_ACCESS_DETECTED".to_string(),
                            message: format!("Network access detected: {}", pattern),
                        });
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查文件系统使用
    fn check_filesystem_usage(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            let filesystem_patterns = vec![
                r"open\s*\(", r"fopen\s*\(", r"File\.", r"FileReader",
                r"FileWriter", r"fs\.", r"std::fs::", r"Path::",
                r"read\s*\(", r"write\s*\(", r"delete\s*\(", r"remove\s*\(",
                r"mkdir\s*\(", r"rmdir\s*\(", r"chmod\s*\(", r"chown\s*\("
            ];
            
            for pattern in filesystem_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid filesystem pattern: {}", e)))?;
                
                if regex.is_match(code) {
                    if !self.security_policy.allow_filesystem {
                        report.issues.push(ValidationIssue {
                            severity: IssueSeverity::Error,
                            code: "FILESYSTEM_ACCESS_DENIED".to_string(),
                            message: format!("Filesystem access not allowed: {}", pattern),
                        });
                    } else {
                        report.issues.push(ValidationIssue {
                            severity: IssueSeverity::Info,
                            code: "FILESYSTEM_ACCESS_DETECTED".to_string(),
                            message: format!("Filesystem access detected: {}", pattern),
                        });
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查内存分配
    fn check_memory_allocation(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            let memory_patterns = vec![
                r"malloc\s*\(", r"calloc\s*\(", r"realloc\s*\(", r"free\s*\(",
                r"new\s+", r"delete\s+", r"Box::", r"Vec::",
                r"HashMap::", r"allocate", r"deallocate"
            ];
            
            for pattern in memory_patterns {
                let regex = Regex::new(pattern)
                    .map_err(|e| Error::validation(format!("Invalid memory pattern: {}", e)))?;
                
                let count = regex.find_iter(code).count();
                if count > 100 { // 大量内存操作
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        code: "EXCESSIVE_MEMORY_ALLOCATION".to_string(),
                        message: format!("Excessive memory allocation detected: {} instances of {}", 
                                       count, pattern),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查加密操作
    fn check_cryptographic_operations(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            let crypto_patterns = vec![
                r"crypto", r"cipher", r"encrypt", r"decrypt", r"hash",
                r"sha\d+", r"md5", r"aes", r"rsa", r"ecdsa",
                r"random", r"rand", r"urandom", r"entropy"
            ];
            
            for pattern in crypto_patterns {
                let regex = Regex::new(&format!(r"(?i){}", pattern))
                    .map_err(|e| Error::validation(format!("Invalid crypto pattern: {}", e)))?;
                
                if regex.is_match(code) {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Info,
                        code: "CRYPTOGRAPHIC_OPERATION".to_string(),
                        message: format!("Cryptographic operation detected: {}", pattern),
                    });
                }
            }
        }
        
        Ok(())
    }
}

/// 验证报告
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// 算法ID
    pub algorithm_id: String,
    /// 是否通过验证
    pub passed: bool,
    /// 验证问题列表
    pub issues: Vec<ValidationIssue>,
    /// 警告列表
    pub warnings: Vec<String>,
    /// 安全分数 (0-100)
    pub security_score: f32,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

/// 验证问题
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// 问题严重程度
    pub severity: IssueSeverity,
    /// 问题代码
    pub code: String,
    /// 问题描述
    pub message: String,
}

/// 问题严重程度
#[derive(Debug, Clone, PartialEq)]
pub enum IssueSeverity {
    /// 错误（阻止算法执行）
    Error,
    /// 警告（允许执行但可能有风险）
    Warning,
    /// 信息（不影响执行，但值得注意）
    Info,
}

/// 解析后的代码结构
#[derive(Debug, Clone)]
struct ParsedCode {
    ast: String,
    functions: Vec<String>,
    imports: Vec<String>,
}



/// 沙箱权限结构体
#[derive(Debug, Clone)]
struct SandboxPermissions {
    allow_network: bool,
    allow_filesystem: bool,
    allow_syscalls: HashSet<String>,
}

/// 执行结果结构体
#[derive(Debug, Clone)]
struct ExecutionResult {
    success: bool,
    resource_exceeded: bool,
    memory_exceeded: bool,
    cpu_exceeded: bool,
    io_exceeded: bool,
    peak_memory: usize,
    execution_time_ms: u128,
    output: Vec<u8>,
    error_message: Option<String>,
    attempted_network_access: bool,
    attempted_filesystem_access: bool,
}

/// 安全检查模块 - 检查算法代码中的安全风险
struct SecurityChecker {
    allowed_imports: HashSet<String>,
}

impl SecurityChecker {
    /// 创建一个新的安全检查器
    fn new() -> Self {
        let mut allowed_imports = HashSet::new();
        allowed_imports.insert("std::collections".to_string());
        allowed_imports.insert("std::io".to_string());
        allowed_imports.insert("std::fs".to_string());
        allowed_imports.insert("std::thread".to_string());
        allowed_imports.insert("std::sync".to_string());
        allowed_imports.insert("serde".to_string());
        allowed_imports.insert("tokio".to_string());
        
        Self {
            allowed_imports,
        }
    }
    
    /// 执行安全检查
    pub fn check_security(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        // 1. 检查导入模块
        self.check_imports(algorithm, report)?;
        
        // 2. 检查禁止的代码模式
        self.check_banned_patterns(algorithm, report)?;
        
        // 3. 检查资源使用
        self.check_resource_usage(algorithm, report)?;
        
        // 4. 检查代码复杂度
        self.check_complexity(algorithm, report)?;
        
        // 5. 执行深度安全分析
        self.deep_security_analysis(algorithm, report)?;
        
        Ok(())
    }
    
    /// 深度安全分析
    fn deep_security_analysis(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        if let Some(code) = &algorithm.code {
            // 检查潜在的代码注入风险
            self.check_code_injection(code, report)?;
            
            // 检查反序列化安全
            self.check_deserialization_security(code, report)?;
            
            // 检查时间攻击风险
            self.check_timing_attacks(code, report)?;
            
            // 检查内存安全
            self.check_memory_safety(code, report)?;
        }
        
        Ok(())
    }
    
    /// 检查代码注入风险
    fn check_code_injection(&self, code: &str, report: &mut ValidationReport) -> Result<()> {
        let injection_patterns = vec![
            r"eval\s*\(",
            r"exec\s*\(",
            r"compile\s*\(",
            r"__import__\s*\(",
            r"getattr\s*\(",
            r"setattr\s*\(",
        ];
        
        for pattern in injection_patterns {
            let regex = Regex::new(pattern)
                .map_err(|e| Error::validation(format!("Invalid injection pattern: {}", e)))?;
            
            if regex.is_match(code) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "CODE_INJECTION_RISK".to_string(),
                    message: format!("Potential code injection risk: {}", pattern),
                });
            }
        }
        
        Ok(())
    }
    
    /// 检查反序列化安全
    fn check_deserialization_security(&self, code: &str, report: &mut ValidationReport) -> Result<()> {
        let deserialization_patterns = vec![
            r"pickle\.loads?",
            r"yaml\.load",
            r"json\.loads",
            r"serde.*::from_str",
        ];
        
        for pattern in deserialization_patterns {
            let regex = Regex::new(pattern)
                .map_err(|e| Error::validation(format!("Invalid deserialization pattern: {}", e)))?;
            
            if regex.is_match(code) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "DESERIALIZATION_RISK".to_string(),
                    message: format!("Potential deserialization security risk: {}", pattern),
                });
            }
        }
        
        Ok(())
    }
    
    /// 检查时间攻击风险
    fn check_timing_attacks(&self, code: &str, report: &mut ValidationReport) -> Result<()> {
        let timing_patterns = vec![
            r"sleep\s*\(",
            r"time\.sleep",
            r"std::thread::sleep",
            r"tokio::time::sleep",
        ];
        
        for pattern in timing_patterns {
            let regex = Regex::new(pattern)
                .map_err(|e| Error::validation(format!("Invalid timing pattern: {}", e)))?;
            
            let count = regex.find_iter(code).count();
            if count > 5 {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "TIMING_ATTACK_RISK".to_string(),
                    message: format!("Potential timing attack vector: {} sleep calls found", count),
                });
            }
        }
        
        Ok(())
    }
    
    /// 检查内存安全
    fn check_memory_safety(&self, code: &str, report: &mut ValidationReport) -> Result<()> {
        let unsafe_patterns = vec![
            r"unsafe\s*\{",
            r"transmute\s*\(",
            r"from_raw\s*\(",
            r"as_ptr\s*\(",
            r"as_mut_ptr\s*\(",
        ];
        
        for pattern in unsafe_patterns {
            let regex = Regex::new(pattern)
                .map_err(|e| Error::validation(format!("Invalid unsafe pattern: {}", e)))?;
            
            if regex.is_match(code) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "MEMORY_SAFETY_RISK".to_string(),
                    message: format!("Potential memory safety issue: {}", pattern),
                });
            }
        }
        
        Ok(())
    }
    
    /// 检查导入语句
    fn check_imports(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        // 检查导入语句,正则表达式: import xxx 或 from xxx import yyy
        let import_regex = Regex::new(r"import\s+(\w+)|from\s+(\w+)").unwrap();
        
        for cap in import_regex.captures_iter(&algorithm.code) {
            let module = cap.get(1).or_else(|| cap.get(2)).unwrap().as_str();
            
            if !self.allowed_imports.contains(module) {
                report.issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "UNAUTHORIZED_IMPORT".to_string(),
                    message: format!("未授权导入模块: {} ", module),
                });
            }
        }
        
        Ok(())
    }
    
    /// 检查禁止的模式
    fn check_banned_patterns(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        // 敏感操作
        let banned_patterns = [
            (r"open\s*\(", "文件操作"),
            (r"__import__", "动态导入"),
            (r"eval\s*\(", "代码执行"),
            (r"exec\s*\(", "代码执行"),
            (r"os\.", "系统操作"),
            (r"sys\.", "系统操作"),
            (r"subprocess", "进程操作"),
            (r"globals\(\)", "全局变量访问"),
            (r"locals\(\)", "局部变量访问"),
            (r"True\s*=", "修改内置变量"),
            (r"False\s*=", "修改内置变量"),
            (r"for\s+.*\s+in\s+range\s*\(\s*[0-9]{8,}\s*\)", "过大的循环范围"),
            (r"import\s+random", "随机数(影响结果可重现性)"),
            (r"\.append\(\s*\[.*\]\s*\*\s*[0-9]{5,}\s*\)", "可能导致内存溢出的操作"),
            (r"while\s+True", "无限循环"),
            (r"def\s+recursive_function.*\s+recursive_function\s*\(", "递归函数(可能导致栈溢出)"),
        ];
        
        for (pattern, reason) in banned_patterns.iter() {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(&algorithm.code) {
                    report.issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "BANNED_PATTERN".to_string(),
                        message: format!("禁止使用: {} ({}) ", reason, pattern),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查资源使用
    fn check_resource_usage(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        // 检查代码行数
        let lines = algorithm.code.lines().count();
        if lines > 1000 {
            report.warnings.push(format!(
                "代码行数 ({}) 超过推荐的最大值 (1000) ",
                lines
            ));
        }
        
        // 检查嵌套循环
        let nested_loop_regex = Regex::new(r"for.*for|while.*for|for.*while|while.*while").unwrap();
        if nested_loop_regex.is_match(&algorithm.code) {
            report.warnings.push(
                "检测到嵌套循环,可能导致性能问题 ".to_string()
            );
        }
        
        // 检查内存使用
        let large_array_regex = Regex::new(r"\[\s*0\s*\]\s*\*\s*([0-9]{5,})").unwrap();
        if let Some(cap) = large_array_regex.captures(&algorithm.code) {
            if let Some(size) = cap.get(1) {
                if let Ok(num) = size.as_str().parse::<usize>() {
                    if num > 100000 {
                        report.warnings.push(format!(
                            "创建大数组 (大小 {}),可能导致内存问题 ",
                            num
                        ));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查代码复杂度
    fn check_complexity(&self, algorithm: &Algorithm, report: &mut ValidationReport) -> Result<()> {
        // 简单的圈复杂度估计
        let control_structures = [
            "if", "else", "for", "while", "switch", "case", 
            "break", "continue", "return", "try", "catch", "finally"
        ];
        
        let mut complexity = 1; // 基础复杂度为1
        
        for structure in &control_structures {
            let pattern = format!(r"\b{}\b", structure);
            if let Ok(regex) = Regex::new(&pattern) {
                complexity += regex.find_iter(&algorithm.code).count();
            }
        }
        
        if complexity > 50 {
            report.warnings.push(format!(
                "代码复杂度 ({}) 过高,建议简化 ",
                complexity
            ));
        }
        
        Ok(())
    }
}

/// 创建算法验证器
pub fn create_validator() -> AlgorithmValidator {
    AlgorithmValidator::new()
        .with_security_checker(SecurityChecker::new())
}

/// 创建严格模式验证器工厂方法
pub fn create_strict_validator() -> AlgorithmValidator {
    AlgorithmValidator::with_security_level(SecurityPolicyLevel::Strict)
        .with_max_code_size(512 * 1024)  // 512KB
        .with_network_access(false)
        .with_filesystem_access(false)
        .with_validation_timeout(Duration::from_secs(10))
}

/// 创建宽松模式验证器工厂方法
pub fn create_permissive_validator() -> AlgorithmValidator {
    AlgorithmValidator::with_security_level(SecurityPolicyLevel::Low)
        .with_max_code_size(2 * 1024 * 1024)  // 2MB
        .with_network_access(true)
        .with_filesystem_access(true)
        .with_validation_timeout(Duration::from_secs(60))
}

 