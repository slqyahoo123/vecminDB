use crate::error::{Error, Result};
// removed unused imports: WasmSecurityReport, WasmModuleResult, ResourceLimits
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use std::thread;
use log::{info, warn, error, debug};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// WASM执行状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmExecutionStatus {
    /// 准备中
    Preparing,
    /// 加载中
    Loading,
    /// 验证中
    Validating,
    /// 实例化中
    Instantiating,
    /// 执行中
    Executing,
    /// 已完成
    Completed,
    /// 被终止
    Terminated,
    /// 失败
    Failed,
}

/// WASM安全配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSecurityConfig {
    /// 最大内存页数 (每页64KB)
    pub max_memory_pages: u32,
    /// 最大表元素数
    pub max_table_elements: u32,
    /// 最大全局变量数
    pub max_globals: u32,
    /// 最大函数数
    pub max_functions: u32,
    /// 最大导入数
    pub max_imports: u32,
    /// 最大导出数
    pub max_exports: u32,
    /// 是否允许内存增长
    pub allow_memory_growth: bool,
    /// 是否启用燃料限制
    pub enable_fuel_limit: bool,
    /// 燃料限制
    pub fuel_limit: u64,
    /// 允许的导入模块
    pub allowed_import_modules: HashSet<String>,
    /// 禁止的导入函数
    pub forbidden_import_functions: HashSet<String>,
    /// 是否启用WASI
    pub enable_wasi: bool,
    /// WASI配置
    pub wasi_config: Option<WasiConfig>,
    /// 执行超时(毫秒)
    pub execution_timeout_ms: u64,
    /// 是否启用栈溢出保护
    pub enable_stack_overflow_protection: bool,
    /// 栈大小限制(字节)
    pub max_stack_size: usize,
}

impl Default for WasmSecurityConfig {
    fn default() -> Self {
        let mut allowed_modules = HashSet::new();
        allowed_modules.insert("env".to_string());
        allowed_modules.insert("wasi_snapshot_preview1".to_string());
        
        let mut forbidden_functions = HashSet::new();
        forbidden_functions.insert("proc_exit".to_string());
        forbidden_functions.insert("sock_recv".to_string());
        forbidden_functions.insert("sock_send".to_string());
        forbidden_functions.insert("path_open".to_string());
        
        Self {
            max_memory_pages: 1024,    // 64MB
            max_table_elements: 1000,
            max_globals: 100,
            max_functions: 1000,
            max_imports: 100,
            max_exports: 100,
            allow_memory_growth: false,
            enable_fuel_limit: true,
            fuel_limit: 1_000_000_000, // 10亿燃料单位
            allowed_import_modules: allowed_modules,
            forbidden_import_functions: forbidden_functions,
            enable_wasi: true,
            wasi_config: Some(WasiConfig::default()),
            execution_timeout_ms: 30000, // 30秒
            enable_stack_overflow_protection: true,
            max_stack_size: 1024 * 1024, // 1MB
        }
    }
}

/// WASI安全配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasiConfig {
    /// 是否允许文件系统访问
    pub allow_filesystem: bool,
    /// 允许的目录
    pub allowed_directories: Vec<String>,
    /// 是否只读
    pub readonly: bool,
    /// 是否允许网络访问
    pub allow_network: bool,
    /// 允许的网络地址
    pub allowed_addresses: Vec<String>,
    /// 环境变量白名单
    pub allowed_env_vars: Vec<String>,
    /// 是否允许随机数生成
    pub allow_random: bool,
    /// 是否允许时间访问
    pub allow_time: bool,
}

impl Default for WasiConfig {
    fn default() -> Self {
        Self {
            allow_filesystem: false,
            allowed_directories: vec!["/tmp".to_string()],
            readonly: true,
            allow_network: false,
            allowed_addresses: Vec::new(),
            allowed_env_vars: vec!["PATH".to_string()],
            allow_random: true,
            allow_time: true,
        }
    }
}

/// WASM运行时统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRuntimeStats {
    /// 已用燃料
    pub fuel_consumed: u64,
    /// 内存使用(字节)
    pub memory_usage: usize,
    /// 函数调用次数
    pub function_calls: u64,
    /// 异常次数
    pub exceptions: u64,
    /// 执行时间(毫秒)
    pub execution_time_ms: u64,
    /// 指令执行次数
    pub instructions_executed: u64,
    /// 内存分配次数
    pub memory_allocations: u64,
    /// 表访问次数
    pub table_accesses: u64,
}

impl Default for WasmRuntimeStats {
    fn default() -> Self {
        Self {
            fuel_consumed: 0,
            memory_usage: 0,
            function_calls: 0,
            exceptions: 0,
            execution_time_ms: 0,
            instructions_executed: 0,
            memory_allocations: 0,
            table_accesses: 0,
        }
    }
}

/// WASM安全事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSecurityEvent {
    /// 事件类型
    pub event_type: WasmSecurityEventType,
    /// 事件时间
    pub timestamp: DateTime<Utc>,
    /// 事件描述
    pub description: String,
    /// 相关数据
    pub data: HashMap<String, String>,
    /// 严重程度
    pub severity: WasmSecuritySeverity,
}

/// WASM安全事件类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmSecurityEventType {
    /// 非法导入
    IllegalImport,
    /// 内存访问越界
    MemoryOutOfBounds,
    /// 栈溢出
    StackOverflow,
    /// 燃料耗尽
    FuelExhausted,
    /// 执行超时
    ExecutionTimeout,
    /// 非法函数调用
    IllegalFunctionCall,
    /// 权限违规
    PermissionViolation,
    /// 资源耗尽
    ResourceExhaustion,
    /// 异常终止
    AbnormalTermination,
}

/// WASM安全严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmSecuritySeverity {
    /// 信息
    Info,
    /// 警告
    Warning,
    /// 错误
    Error,
    /// 严重
    Critical,
}

/// WASM安全执行上下文
pub struct WasmSecurityContext {
    /// 安全配置
    pub config: WasmSecurityConfig,
    /// 运行时统计
    pub stats: Arc<RwLock<WasmRuntimeStats>>,
    /// 安全事件
    pub events: Arc<Mutex<Vec<WasmSecurityEvent>>>,
    /// 执行状态
    pub status: Arc<RwLock<WasmExecutionStatus>>,
    /// 开始时间
    pub start_time: Instant,
    /// 是否被终止
    pub terminated: Arc<RwLock<bool>>,
}

impl WasmSecurityContext {
    /// 创建新的安全上下文
    pub fn new(config: WasmSecurityConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(WasmRuntimeStats::default())),
            events: Arc::new(Mutex::new(Vec::new())),
            status: Arc::new(RwLock::new(WasmExecutionStatus::Preparing)),
            start_time: Instant::now(),
            terminated: Arc::new(RwLock::new(false)),
        }
    }
    
    /// 记录安全事件
    pub fn log_security_event(&self, event: WasmSecurityEvent) {
        warn!("WASM安全事件: {:?} - {}", event.event_type, event.description);
        self.events.lock().unwrap().push(event);
    }
    
    /// 更新运行时统计
    pub fn update_stats<F>(&self, updater: F) 
    where 
        F: FnOnce(&mut WasmRuntimeStats),
    {
        updater(&mut self.stats.write().unwrap());
    }
    
    /// 检查是否应该终止
    pub fn should_terminate(&self) -> bool {
        *self.terminated.read().unwrap() || 
        self.start_time.elapsed().as_millis() > self.config.execution_timeout_ms as u128
    }
    
    /// 强制终止
    pub fn force_terminate(&self, reason: &str) {
        warn!("强制终止WASM执行: {}", reason);
        *self.terminated.write().unwrap() = true;
        *self.status.write().unwrap() = WasmExecutionStatus::Terminated;
        
        self.log_security_event(WasmSecurityEvent {
            event_type: WasmSecurityEventType::AbnormalTermination,
            timestamp: Utc::now(),
            description: format!("强制终止: {}", reason),
            data: HashMap::new(),
            severity: WasmSecuritySeverity::Critical,
        });
    }
}

/// WASM安全执行器
pub struct WasmSecurityExecutor {
    /// 默认安全配置
    default_config: WasmSecurityConfig,
    /// 活动执行上下文
    active_contexts: Arc<RwLock<HashMap<String, Arc<WasmSecurityContext>>>>,
    /// 执行统计
    execution_stats: Arc<RwLock<WasmExecutionStatistics>>,
}

/// WASM执行统计
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct WasmExecutionStatistics {
    /// 总执行次数
    pub total_executions: u64,
    /// 成功执行次数
    pub successful_executions: u64,
    /// 失败执行次数
    pub failed_executions: u64,
    /// 安全终止次数
    pub security_terminations: u64,
    /// 超时次数
    pub timeout_count: u64,
    /// 平均执行时间(毫秒)
    pub average_execution_time_ms: f64,
    /// 平均燃料消耗
    pub average_fuel_consumption: f64,
    /// 平均内存使用(字节)
    pub average_memory_usage: f64,
}

impl WasmSecurityExecutor {
    /// 创建新的安全执行器
    pub fn new() -> Self {
        Self {
            default_config: WasmSecurityConfig::default(),
            active_contexts: Arc::new(RwLock::new(HashMap::new())),
            execution_stats: Arc::new(RwLock::new(WasmExecutionStatistics::default())),
        }
    }
    
    /// 使用自定义配置创建
    pub fn with_config(config: WasmSecurityConfig) -> Self {
        Self {
            default_config: config,
            active_contexts: Arc::new(RwLock::new(HashMap::new())),
            execution_stats: Arc::new(RwLock::new(WasmExecutionStatistics::default())),
        }
    }
    
    /// 安全执行WASM模块
    pub async fn execute_secure(
        &self,
        execution_id: &str,
        wasm_binary: &[u8],
        function_name: &str,
        args: &[u8],
        custom_config: Option<WasmSecurityConfig>,
    ) -> Result<SecureWasmExecutionResult> {
        let config = custom_config.unwrap_or_else(|| self.default_config.clone());
        let context = Arc::new(WasmSecurityContext::new(config));
        
        info!("开始安全执行WASM模块: {}", execution_id);
        
        // 1. 安全验证
        self.validate_wasm_security(wasm_binary, &context).await?;
        
        // 2. 注册执行上下文
        self.active_contexts.write().unwrap().insert(execution_id.to_string(), context.clone());
        
        // 3. 执行WASM
        let result = self.execute_wasm_with_monitoring(execution_id, wasm_binary, function_name, args, context.clone()).await;
        
        // 4. 清理上下文
        self.active_contexts.write().unwrap().remove(execution_id);
        
        // 5. 更新统计
        self.update_execution_statistics(&result).await;
        
        result
    }
    
    /// 验证WASM安全性
    async fn validate_wasm_security(
        &self,
        wasm_binary: &[u8],
        context: &WasmSecurityContext,
    ) -> Result<()> {
        *context.status.write().unwrap() = WasmExecutionStatus::Validating;
        
        debug!("验证WASM模块安全性，大小: {} 字节", wasm_binary.len());
        
        // 1. 基本格式验证
        if wasm_binary.len() < 8 {
            return Err(Error::validation_error("WASM二进制文件太小"));
        }
        
        // 检查魔数
        if &wasm_binary[0..4] != &[0x00, 0x61, 0x73, 0x6D] {
            return Err(Error::validation_error("无效的WASM魔数"));
        }
        
        // 检查版本
        if &wasm_binary[4..8] != &[0x01, 0x00, 0x00, 0x00] {
            return Err(Error::validation_error("不支持的WASM版本"));
        }
        
        // 2. 模块结构分析
        self.analyze_wasm_structure(wasm_binary, context).await?;
        
        // 3. 导入安全检查
        self.validate_imports(wasm_binary, context).await?;
        
        // 4. 内存和表限制检查
        self.validate_memory_and_tables(wasm_binary, context).await?;
        
        // 5. 函数安全检查
        self.validate_functions(wasm_binary, context).await?;
        
        info!("WASM模块安全验证通过");
        Ok(())
    }
    
    /// 分析WASM模块结构
    async fn analyze_wasm_structure(
        &self,
        wasm_binary: &[u8],
        _context: &WasmSecurityContext,
    ) -> Result<()> {
        debug!("分析WASM模块结构");
        
        // 这里应该实现真正的WASM二进制解析
        // 暂时使用简化的实现
        
        let mut offset = 8; // 跳过魔数和版本
        let mut sections_found = HashSet::new();
        
        while offset < wasm_binary.len() {
            if offset + 1 >= wasm_binary.len() {
                break;
            }
            
            let section_id = wasm_binary[offset];
            offset += 1;
            
            // 读取段长度（LEB128编码，这里简化处理）
            let mut section_size = 0;
            let mut shift = 0;
            
            while offset < wasm_binary.len() && shift < 32 {
                let byte = wasm_binary[offset];
                offset += 1;
                
                section_size |= ((byte & 0x7F) as usize) << shift;
                
                if (byte & 0x80) == 0 {
                    break;
                }
                
                shift += 7;
            }
            
            sections_found.insert(section_id);
            
            // 检查段大小是否合理
            if section_size > wasm_binary.len() - offset {
                return Err(Error::validation_error("WASM段大小无效"));
            }
            
            offset += section_size;
        }
        
        debug!("发现WASM段: {:?}", sections_found);
        Ok(())
    }
    
    /// 验证导入
    async fn validate_imports(
        &self,
        wasm_binary: &[u8],
        context: &WasmSecurityContext,
    ) -> Result<()> {
        debug!("验证WASM导入");
        
        // 这里应该解析导入段并验证
        // 暂时使用简化的检查
        
        let binary_str = String::from_utf8_lossy(wasm_binary);
        
        // 检查禁止的导入函数
        for forbidden_func in &context.config.forbidden_import_functions {
            if binary_str.contains(forbidden_func) {
                context.log_security_event(WasmSecurityEvent {
                    event_type: WasmSecurityEventType::IllegalImport,
                    timestamp: Utc::now(),
                    description: format!("检测到禁止的导入函数: {}", forbidden_func),
                    data: HashMap::from([("function".to_string(), forbidden_func.clone())]),
                    severity: WasmSecuritySeverity::Error,
                });
                
                return Err(Error::security_violation(format!("禁止的导入函数: {}", forbidden_func)));
            }
        }
        
        Ok(())
    }
    
    /// 验证内存和表
    async fn validate_memory_and_tables(
        &self,
        wasm_binary: &[u8],
        context: &WasmSecurityContext,
    ) -> Result<()> {
        debug!("验证WASM内存和表限制");
        
        // 这里应该解析内存段和表段
        // 暂时使用简化的检查
        
        // 检查文件大小作为简单的资源限制
        let max_size = context.config.max_memory_pages as usize * 64 * 1024; // 每页64KB
        if wasm_binary.len() > max_size {
            return Err(Error::validation_error(format!(
                "WASM模块过大: {} > {}", 
                wasm_binary.len(), 
                max_size
            )));
        }
        
        Ok(())
    }
    
    /// 验证函数
    async fn validate_functions(
        &self,
        _wasm_binary: &[u8],
        _context: &WasmSecurityContext,
    ) -> Result<()> {
        debug!("验证WASM函数");
        
        // 这里应该解析函数段并验证
        // 包括检查危险的指令序列
        
        Ok(())
    }
    
    /// 带监控的WASM执行
    async fn execute_wasm_with_monitoring(
        &self,
        execution_id: &str,
        wasm_binary: &[u8],
        function_name: &str,
        args: &[u8],
        context: Arc<WasmSecurityContext>,
    ) -> Result<SecureWasmExecutionResult> {
        *context.status.write().unwrap() = WasmExecutionStatus::Executing;
        
        info!("开始监控执行WASM: {}", execution_id);
        
        // 启动监控线程
        let monitoring_context = context.clone();
        let monitoring_handle = thread::spawn(move || {
            Self::monitoring_thread(monitoring_context);
        });
        
        // 执行WASM（这里应该使用真正的WASM运行时）
        let execution_result = self.execute_wasm_module(wasm_binary, function_name, args, context.clone()).await;
        
        // 停止监控
        context.force_terminate("执行完成");
        if let Err(e) = monitoring_handle.join() {
            warn!("监控线程结束异常: {:?}", e);
        }
        
        // 生成执行结果
        let success = execution_result.is_ok();
        let stats = context.stats.read().unwrap().clone();
        let events = context.events.lock().unwrap().clone();
        
        *context.status.write().unwrap() = if success {
            WasmExecutionStatus::Completed
        } else {
            WasmExecutionStatus::Failed
        };
        
        Ok(SecureWasmExecutionResult {
            execution_id: execution_id.to_string(),
            success,
            output: execution_result.unwrap_or_default(),
            stats,
            security_events: events,
            execution_time_ms: context.start_time.elapsed().as_millis() as u64,
        })
    }
    
    /// 执行WASM模块
    async fn execute_wasm_module(
        &self,
        _wasm_binary: &[u8],
        function_name: &str,
        _args: &[u8],
        _context: Arc<WasmSecurityContext>,
    ) -> Result<Vec<u8>> {
        debug!("执行WASM模块，函数: {}", function_name);
        
        // 这里应该使用真正的WASM运行时（如wasmtime、wasmer等）
        // 暂时返回模拟结果
        
        // 模拟执行过程中的统计更新
        _context.update_stats(|stats| {
            stats.function_calls += 1;
            stats.instructions_executed += 1000;
            stats.memory_usage = 1024 * 1024; // 1MB
            stats.fuel_consumed += 1000;
        });
        
        // 模拟输出
        Ok(b"WASM execution result".to_vec())
    }
    
    /// 监控线程
    fn monitoring_thread(context: Arc<WasmSecurityContext>) {
        debug!("启动WASM监控线程");
        
        let check_interval = Duration::from_millis(100);
        
        while !context.should_terminate() {
            // 检查燃料消耗
            let stats = context.stats.read().unwrap();
            if context.config.enable_fuel_limit && stats.fuel_consumed > context.config.fuel_limit {
                context.log_security_event(WasmSecurityEvent {
                    event_type: WasmSecurityEventType::FuelExhausted,
                    timestamp: Utc::now(),
                    description: "燃料耗尽".to_string(),
                    data: HashMap::from([
                        ("consumed".to_string(), stats.fuel_consumed.to_string()),
                        ("limit".to_string(), context.config.fuel_limit.to_string()),
                    ]),
                    severity: WasmSecuritySeverity::Error,
                });
                
                context.force_terminate("燃料耗尽");
                break;
            }
            
            // 检查内存使用
            if stats.memory_usage > context.config.max_memory_pages as usize * 64 * 1024 {
                context.log_security_event(WasmSecurityEvent {
                    event_type: WasmSecurityEventType::MemoryOutOfBounds,
                    timestamp: Utc::now(),
                    description: "内存使用超限".to_string(),
                    data: HashMap::from([
                        ("usage".to_string(), stats.memory_usage.to_string()),
                        ("limit".to_string(), (context.config.max_memory_pages as usize * 64 * 1024).to_string()),
                    ]),
                    severity: WasmSecuritySeverity::Error,
                });
                
                context.force_terminate("内存超限");
                break;
            }
            
            drop(stats);
            thread::sleep(check_interval);
        }
        
        debug!("WASM监控线程结束");
    }
    
    /// 强制终止指定执行
    pub async fn force_terminate_execution(&self, execution_id: &str, reason: &str) -> Result<()> {
        if let Some(context) = self.active_contexts.read().unwrap().get(execution_id) {
            context.force_terminate(reason);
            
            // 更新统计
            self.execution_stats.write().unwrap().security_terminations += 1;
            
            info!("已强制终止WASM执行: {}, 原因: {}", execution_id, reason);
        }
        
        Ok(())
    }
    
    /// 强制终止所有执行
    pub async fn force_terminate_all(&self, reason: &str) -> Result<()> {
        warn!("强制终止所有WASM执行，原因: {}", reason);
        
        let execution_ids: Vec<String> = {
            self.active_contexts.read().unwrap().keys().cloned().collect()
        };
        
        for execution_id in execution_ids {
            if let Err(e) = self.force_terminate_execution(&execution_id, reason).await {
                error!("强制终止WASM执行失败 {}: {}", execution_id, e);
            }
        }
        
        Ok(())
    }
    
    /// 获取活动执行数量
    pub fn get_active_execution_count(&self) -> usize {
        self.active_contexts.read().unwrap().len()
    }
    
    /// 获取执行统计
    pub fn get_execution_statistics(&self) -> WasmExecutionStatistics {
        self.execution_stats.read().unwrap().clone()
    }
    
    /// 更新执行统计
    async fn update_execution_statistics(&self, result: &Result<SecureWasmExecutionResult>) {
        let mut stats = self.execution_stats.write().unwrap();
        stats.total_executions += 1;
        
        match result {
            Ok(exec_result) => {
                if exec_result.success {
                    stats.successful_executions += 1;
                } else {
                    stats.failed_executions += 1;
                }
                
                // 更新平均值
                let total = stats.total_executions as f64;
                let current = (total - 1.0) / total;
                let new = 1.0 / total;
                
                stats.average_execution_time_ms = stats.average_execution_time_ms * current + exec_result.execution_time_ms as f64 * new;
                stats.average_fuel_consumption = stats.average_fuel_consumption * current + exec_result.stats.fuel_consumed as f64 * new;
                stats.average_memory_usage = stats.average_memory_usage * current + exec_result.stats.memory_usage as f64 * new;
            },
            Err(_) => {
                stats.failed_executions += 1;
            }
        }
    }
}

/// 安全WASM执行结果
#[derive(Debug, Clone)]
pub struct SecureWasmExecutionResult {
    /// 执行ID
    pub execution_id: String,
    /// 是否成功
    pub success: bool,
    /// 输出数据
    pub output: Vec<u8>,
    /// 运行时统计
    pub stats: WasmRuntimeStats,
    /// 安全事件
    pub security_events: Vec<WasmSecurityEvent>,
    /// 执行时间(毫秒)
    pub execution_time_ms: u64,
}

/// 创建默认WASM安全执行器
pub fn create_default_wasm_executor() -> WasmSecurityExecutor {
    WasmSecurityExecutor::new()
}

/// 创建高安全级别WASM执行器
pub fn create_high_security_wasm_executor() -> WasmSecurityExecutor {
    let mut config = WasmSecurityConfig::default();
    config.max_memory_pages = 256;      // 16MB
    config.max_functions = 100;
    config.max_imports = 10;
    config.fuel_limit = 100_000_000;    // 1亿燃料单位
    config.execution_timeout_ms = 10000; // 10秒
    config.allow_memory_growth = false;
    config.enable_wasi = false;
    
    WasmSecurityExecutor::with_config(config)
}

/// 创建受限的WASM执行器（用于不受信任的代码）
pub fn create_restricted_wasm_executor() -> WasmSecurityExecutor {
    let mut config = WasmSecurityConfig::default();
    config.max_memory_pages = 64;       // 4MB
    config.max_functions = 50;
    config.max_imports = 5;
    config.fuel_limit = 10_000_000;     // 1千万燃料单位
    config.execution_timeout_ms = 5000;  // 5秒
    config.allow_memory_growth = false;
    config.enable_wasi = false;
    config.allowed_import_modules.clear();
    config.allowed_import_modules.insert("env".to_string());
    
    WasmSecurityExecutor::with_config(config)
} 