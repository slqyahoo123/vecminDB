use std::path::Path;
use std::time::Duration;
use std::sync::{Arc, RwLock};
use std::sync::atomic::Ordering;
use async_trait::async_trait;
#[cfg(feature = "tempfile")]
use tempfile::TempDir;
use uuid::Uuid;
#[cfg(feature = "wasmtime")]
use wasmtime::{Engine as WasmEngine, Store as WasmStore, Module, Linker};
use log::{debug, warn};
use tokio::time::timeout as tokio_timeout;
#[cfg(feature = "wasmtime")]
use wasmtime_wasi::WasiCtxBuilder;

use crate::{Error, Result};
use crate::algorithm::executor::config::{ExecutorConfig, SandboxConfig};
use crate::algorithm::types::ResourceUsage;
use crate::algorithm::executor::sandbox::interface::Sandbox;
use crate::algorithm::executor::sandbox::result::SandboxResult;
use crate::algorithm::types::SandboxStatus;
use crate::algorithm::executor::sandbox::error::SandboxError;
use crate::algorithm::executor::sandbox::environment::ExecutionEnvironment;

/// WebAssembly沙箱实现
#[derive(Debug)]
pub struct WasmSandbox {
    id: String,
    config: ExecutorConfig,
    engine: WasmEngine,
    temp_dir: Option<TempDir>,
    status: RwLock<SandboxStatus>,
    environment: Option<Arc<tokio::sync::RwLock<ExecutionEnvironment>>>,
}

impl WasmSandbox {
    pub async fn new(config: &ExecutorConfig) -> Result<Self> {
        let id = format!("wasm-sandbox-{}", Uuid::new_v4());
        let temp_dir = TempDir::new().ok();
        
        // 创建WASM引擎
        let engine = WasmEngine::default();
        
        Ok(Self {
            id,
            config: config.clone(),
            engine,
            temp_dir,
            status: RwLock::new(SandboxStatus::Uninitialized),
            environment: None,
        })
    }
    
    fn set_status(&self, status: SandboxStatus) -> Result<()> {
        let mut current_status = self.status.write().map_err(|_| {
            Error::lock("无法获取沙箱状态锁")
        })?;
        *current_status = status;
        Ok(())
    }
    
    fn create_environment(&mut self) -> Result<Arc<tokio::sync::RwLock<ExecutionEnvironment>>> {
        // 创建沙箱安全配置
        let sandbox_config = SandboxConfig {
            allow_network: Some(self.config.resource_limits.network_enabled),
            allow_filesystem: Some(self.config.resource_limits.io_enabled),
            allowed_env_vars: vec![],
            allowed_paths: vec![],
            allow_stdin: Some(false),
            allow_stdout: Some(true),
            allow_stderr: Some(true),
            security_level: self.config.security_level.clone(),
            ..Default::default()
        };
        
        // 创建执行环境
        let mut env = ExecutionEnvironment::new(sandbox_config, &self.config.task_id);
        
        // 设置临时目录
        if let Some(dir) = &self.temp_dir {
            env.temp_dir = TempDir::new().ok();
        }
        
        let env_arc = Arc::new(tokio::sync::RwLock::new(env));
        self.environment = Some(env_arc.clone());
        
        Ok(env_arc)
    }
    
    /// 获取执行结果
    #[cfg(feature = "wasmtime")]
    fn get_execution_result(&self, instance: &wasmtime::Instance, store: &mut wasmtime::Store<()>, result_code: u32) -> Result<Vec<u8>, SandboxError> {
        let result_ptr_fn = instance.get_func(&mut *store, "get_result_ptr")
            .or_else(|| instance.get_func(&mut *store, "get_output_ptr"))
            .ok_or(SandboxError::MissingExport("get_result_ptr or get_output_ptr".to_string()))?;

        let result_len_fn = instance.get_func(&mut *store, "get_result_len")
            .or_else(|| instance.get_func(&mut *store, "get_output_len"))
            .ok_or(SandboxError::MissingExport("get_result_len or get_output_len".to_string()))?;

        let mut result_ptr_val = [wasmtime::Val::I32(0)];
        result_ptr_fn.call(&mut *store, &[], &mut result_ptr_val)
            .map_err(|e| SandboxError::Execution(format!("Failed to call result_ptr function: {}", e)))?;

        let mut result_len_val = [wasmtime::Val::I32(0)];
        result_len_fn.call(&mut *store, &[], &mut result_len_val)
            .map_err(|e| SandboxError::Execution(format!("Failed to call result_len function: {}", e)))?;

        let result_ptr = match result_ptr_val[0] {
            wasmtime::Val::I32(ptr) => ptr as u32,
            _ => return Err(SandboxError::InvalidOutput("Expected i32 result pointer".to_string())),
        };

        let result_len = match result_len_val[0] {
            wasmtime::Val::I32(len) => len as u32,
            _ => return Err(SandboxError::InvalidOutput("Expected i32 result length".to_string())),
        };

        let memory = instance.get_memory(&mut *store, "memory")
            .ok_or(SandboxError::MissingExport("memory".to_string()))?;

        let max_size = memory.data_size(&mut *store);
        if result_ptr as usize + result_len as usize > max_size {
            return Err(SandboxError::InvalidOutput("Result pointer out of bounds".to_string()));
        }

        let mut result_data = vec![0u8; result_len as usize];
        memory.read(&mut *store, result_ptr as usize, &mut result_data)
            .map_err(|e| SandboxError::Execution(format!("Failed to read result from memory: {}", e)))?;

        Ok(result_data)
    }
    
    /// 禁用网络功能
    #[cfg(feature = "wasmtime")]
    fn disable_networking(&self, linker: &mut wasmtime::Linker<()>) -> Result<(), SandboxError> {
        debug!("为沙箱禁用网络功能: {}", self.id);
        
        // 注册禁用的网络函数
        #[cfg(feature = "wasmtime")]
        let disabled_network_fn = |_: &mut wasmtime::Caller<'_, ()>, _args: &[wasmtime::Val], _results: &mut [wasmtime::Val]| {
            Err(wasmtime::Trap::new("网络访问被禁止"))
        };
        
        // 禁用套接字操作
        let network_funcs = [
            "socket", "connect", "bind", "listen", "accept", 
            "getsockname", "getpeername", "socketpair", "shutdown",
            "getsockopt", "setsockopt", "connect", "poll", "send", 
            "recv", "sendto", "recvfrom", "sendmsg", "recvmsg"
        ];
        
        for func_name in &network_funcs {
            if let Err(e) = linker.func_wrap("wasi_snapshot_preview1", func_name, disabled_network_fn) {
                warn!("无法禁用网络函数 {}: {}", func_name, e);
                // 继续禁用其他函数
            }
        }
        
        Ok(())
    }
    
    /// 限制文件系统访问
    #[cfg(feature = "wasmtime")]
    fn restrict_filesystem_access(&self, linker: &mut wasmtime::Linker<()>, allowed_paths: &[String]) -> Result<(), SandboxError> {
        debug!("为沙箱限制文件系统访问: {}", self.id);
        
        // 因为WASI不支持直接拦截特定路径，我们采用完全禁用的策略
        // 对于正式实现，可以考虑使用更复杂的WASI自定义实现
        
        // 禁用的文件系统函数
        let fs_funcs = [
            "path_open", "path_create_directory", "path_unlink_file", 
            "path_rename", "path_link", "path_symlink", "path_readlink",
            "fd_filestat_get", "fd_filestat_set_size", "fd_filestat_set_times",
            "path_filestat_get", "path_filestat_set_times"
        ];
        
        #[cfg(feature = "wasmtime")]
        let disabled_fs_fn = |_: &mut wasmtime::Caller<'_, ()>, _args: &[wasmtime::Val], _results: &mut [wasmtime::Val]| {
            Err(wasmtime::Trap::new("文件系统访问被禁止"))
        };
        
        for func_name in &fs_funcs {
            if let Err(e) = linker.func_wrap("wasi_snapshot_preview1", func_name, disabled_fs_fn) {
                warn!("无法禁用文件系统函数 {}: {}", func_name, e);
                // 继续禁用其他函数
            }
        }
        
        Ok(())
    }
}

#[async_trait]
impl Sandbox for WasmSandbox {
    fn id(&self) -> &str {
        &self.id
    }
    
    async fn prepare(&self) -> Result<()> {
        debug!("【WasmSandbox】准备环境: {}", self.id);
        
        if let Some(env) = &self.environment {
            let mut env_guard = env.write().await;
            env_guard.create_wasm_environment()?;
            
            // 启动资源监控
            env_guard.start_monitoring().await?;
        }
        
        self.set_status(SandboxStatus::Ready)?;
        Ok(())
    }
    
    #[cfg(feature = "wasmtime")]
    async fn execute(&self, code: &[u8], input: &[u8], timeout: Duration) -> Result<SandboxResult> {
        debug!("【WasmSandbox】执行代码: {}, 输入大小: {}, 超时: {:?}", 
              self.id, input.len(), timeout);
        
        if self.environment.is_none() {
            return Err(Error::failed_precondition("沙箱环境未准备"));
        }
        
        self.set_status(SandboxStatus::Running)?;
        let start_time = std::time::Instant::now();
        
        // 获取环境
        let env_arc = self.environment.as_ref().unwrap();
        let env = env_arc.read().await;
        
        // 检查代码安全性
        let validation_result = self.validate_code(code).await?;
        let has_warnings = !validation_result.is_empty();
        
        // 准备模块
        let module_result = tokio::task::spawn_blocking(move || {
            Module::from_binary(&env.wasm_engine.as_ref().unwrap(), code)
        }).await;
        
        let module = match module_result {
            Ok(Ok(module)) => module,
            Ok(Err(e)) => {
                self.set_status(SandboxStatus::Failed)?;
                return Err(SandboxError::WasmCompilation(e.to_string()).into());
            },
            Err(e) => {
                self.set_status(SandboxStatus::Failed)?;
                return Err(Error::internal(format!("编译任务失败: {}", e)));
            }
        };
        
        // 创建超时任务
        let timeout_result = tokio_timeout(timeout, async {
            // 创建存储和Linker
            let mut store = WasmStore::new(&self.engine, ());
            let mut linker = Linker::new(&self.engine);
            
            // 设置WASI
            let wasi_ctx = WasiCtxBuilder::new()
                .inherit_stdio()
                .build();
            
            // 设置WASI上下文到store
            store.set_wasi(wasi_ctx);
            
            // 添加WASI接口到链接器
            if let Err(e) = wasmtime_wasi::add_to_linker(&mut linker, |s| s) {
                return Err(SandboxError::InitializationFailed(format!("配置WASI失败: {}", e)));
            }
            
            // 增加安全限制
            if !env.security_context.allow_network {
                // 禁用网络相关功能
                self.disable_networking(&mut linker)?;
            }
            
            if !env.security_context.allow_filesystem {
                // 禁用文件系统访问，仅允许预设的路径
                self.restrict_filesystem_access(&mut linker, &env.security_context.allowed_paths)?;
            }
            
            // 实例化模块
            let instance = match linker.instantiate(&mut store, &module) {
                Ok(instance) => instance,
                Err(e) => {
                    return Err(SandboxError::WasmInstantiation(format!("实例化模块失败: {}", e)));
                }
            };
            
            // 使用内存管理器
            let mut memory_manager = super::super::utils::WasmMemoryManager::new(&instance, &mut store);
            
            // 将输入数据写入内存
            let (input_ptr, input_len) = match memory_manager.allocate_and_write(input) {
                Ok(result) => result,
                Err(e) => return Err(SandboxError::MemoryAccess(format!("写入输入数据失败: {}", e))),
            };
            
            // 查找入口函数
            let entry_function = instance.get_func(&mut store, "_start")
                .or_else(|| instance.get_func(&mut store, "main"))
                .or_else(|| instance.get_func(&mut store, "run")) 
                .or_else(|| instance.get_func(&mut store, "execute"))
                .ok_or_else(|| SandboxError::FunctionNotFound("找不到入口函数".to_string()))?;
            
            // 准备参数和结果
            let mut args = [
                wasmtime::Val::I32(input_ptr as i32),
                wasmtime::Val::I32(input_len as i32)
            ];
            let mut results = [wasmtime::Val::I32(0)]; // 存储返回值
            
            // 执行入口函数
            match entry_function.call(&mut store, &args, &mut results) {
                Ok(_) => {
                    // 获取返回值
                    let result_code = match results[0].unwrap_i32() {
                        -1 => return Err(SandboxError::ExecutionFailed("WASM执行失败".to_string())),
                        code => code,
                    };
                    
                    // 获取结果数据
                    let result_data = self.get_execution_result(&instance, &mut store, result_code as u32)?;
                    
                    // 解析结果
                    let stdout = match String::from_utf8(result_data) {
                        Ok(s) => s,
                        Err(e) => {
                            return Err(SandboxError::ExecutionFailed(
                                format!("无法解析返回的UTF-8字符串: {}", e)
                            ));
                        }
                    };
                    
                    // 获取错误输出
                    let stderr = env.get_stderr();
                    
                    // 计算资源使用情况
                    let memory_usage = memory_manager.get_total_allocated_size() as usize;
                    env.memory_bytes.store(memory_usage as u64, Ordering::SeqCst);
                    
                    let resource_usage = ResourceUsage {
                        cpu_usage_percent: 100.0, // 近似值
                        memory_usage_bytes: memory_usage,
                        peak_memory_bytes: memory_usage,
                        io_read_bytes: env.disk_read_bytes.load(Ordering::SeqCst) as usize,
                        io_write_bytes: env.disk_write_bytes.load(Ordering::SeqCst) as usize,
                        network_bytes: env.network_bytes.load(Ordering::SeqCst) as usize,
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                    };
                    
                    let mut result = SandboxResult::success(
                        stdout,
                        stderr,
                        start_time.elapsed().as_millis() as u64,
                        resource_usage,
                    );
                    
                    // 添加警告信息
                    for warning in &validation_result {
                        result.add_warning(warning);
                    }
                    
                    Ok(result)
                },
                Err(e) => {
                    // 处理执行错误
                    if e.to_string().contains("out of memory") || e.to_string().contains("memory limit") {
                        Err(SandboxError::ResourceExceeded(format!("内存超限: {}", e)))
                    } else if e.to_string().contains("execution timeout") {
                        Err(SandboxError::Timeout(timeout.as_millis() as u64))
                    } else {
                        Err(SandboxError::WasmExecution(format!("执行失败: {}", e)))
                    }
                }
            }
        }).await;
        
        // 处理执行结果
        let result = match timeout_result {
            Ok(Ok(mut result)) => {
                // 添加警告信息
                for warning in validation_result {
                    result.add_warning(&warning);
                }
                result
            },
            Ok(Err(e)) => {
                self.set_status(SandboxStatus::Failed)?;
                return Err(e.into());
            },
            Err(_) => {
                // 执行超时
                self.set_status(SandboxStatus::Failed)?;
                return Ok(SandboxResult::timeout(
                    timeout.as_millis() as u64,
                    env.get_stdout(),
                    env.get_stderr(),
                    env.resource_usage.clone(),
                ));
            }
        };
        
        self.set_status(SandboxStatus::Completed)?;
        Ok(result)
    }
    
    #[cfg(not(feature = "wasmtime"))]
    async fn execute(&self, _code: &[u8], _input: &[u8], _timeout: Duration) -> Result<SandboxResult> {
        Err(Error::internal("WASM support requires 'wasmtime' feature".to_string()))
    }
    
    async fn cleanup(&self) -> Result<()> {
        debug!("【WasmSandbox】清理资源: {}", self.id);
        
        if let Some(env) = &self.environment {
            let mut env_guard = env.write().await;
            
            // 停止资源监控
            env_guard.stop_monitoring().await?;
            
            // 清理资源
            env_guard.cleanup()?;
        }
        
        self.set_status(SandboxStatus::Cleaned)?;
        Ok(())
    }
    
    fn supports_file_type(&self, file_type: &str) -> bool {
        match file_type.to_lowercase().as_str() {
            "wasm" | "wat" => true,
            _ => false,
        }
    }
    
    async fn load_file(&self, src_path: &Path, sandbox_path: &str) -> Result<()> {
        if let Some(env) = &self.environment {
            let env_guard = env.read().await;
            
            // 检查权限
            if !env_guard.security_context.allow_filesystem {
                return Err(Error::permission_denied("文件系统访问被禁止"));
            }
            
            // 验证沙箱路径安全性
            if sandbox_path.contains("..") || sandbox_path.starts_with('/') {
                return Err(Error::invalid_argument("不安全的沙箱路径"));
            }
            
            // 计算目标路径
            let target_path = match &env_guard.temp_dir {
                Some(dir) => dir.path().join(sandbox_path),
                None => return Err(Error::failed_precondition("临时目录未初始化")),
            };
            
            // 创建父目录
            if let Some(parent) = target_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            
            // 复制文件
            tokio::fs::copy(src_path, &target_path).await?;
            
            // 更新磁盘写入统计
            if let Ok(metadata) = tokio::fs::metadata(&target_path).await {
                let size = metadata.len() as u64;
                env_guard.disk_write_bytes.fetch_add(size, Ordering::SeqCst);
            }
            
            Ok(())
        } else {
            Err(Error::failed_precondition("沙箱环境未初始化"))
        }
    }
    
    async fn save_file(&self, sandbox_path: &str, dest_path: &Path) -> Result<()> {
        if let Some(env) = &self.environment {
            let env_guard = env.read().await;
            
            // 检查权限
            if !env_guard.security_context.allow_filesystem {
                return Err(Error::permission_denied("文件系统访问被禁止"));
            }
            
            // 验证沙箱路径安全性
            if sandbox_path.contains("..") || sandbox_path.starts_with('/') {
                return Err(Error::invalid_argument("不安全的沙箱路径"));
            }
            
            // 计算源路径
            let source_path = match &env_guard.temp_dir {
                Some(dir) => dir.path().join(sandbox_path),
                None => return Err(Error::failed_precondition("临时目录未初始化")),
            };
            
            // 检查源文件是否存在
            if !source_path.exists() {
                return Err(Error::not_found(format!("沙箱文件不存在: {}", sandbox_path)));
            }
            
            // 创建父目录
            if let Some(parent) = dest_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            
            // 复制文件
            tokio::fs::copy(&source_path, dest_path).await?;
            
            // 更新磁盘读取统计
            if let Ok(metadata) = tokio::fs::metadata(&source_path).await {
                let size = metadata.len() as u64;
                env_guard.disk_read_bytes.fetch_add(size, Ordering::SeqCst);
            }
            
            Ok(())
        } else {
            Err(Error::failed_precondition("沙箱环境未初始化"))
        }
    }
    
    async fn cancel(&self) -> Result<()> {
        debug!("【WasmSandbox】取消执行: {}", self.id);
        
        if let Some(env) = &self.environment {
            let env_guard = env.read().await;
            env_guard.cancel();
            
            // 设置状态为已取消
            self.set_status(SandboxStatus::Failed)?;
        }
        
        Ok(())
    }
    
    async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        if let Some(env) = &self.environment {
            let env_guard = env.read().await;
            env_guard.get_resource_usage().await
        } else {
            Err(Error::failed_precondition("沙箱环境未初始化"))
        }
    }
    
    async fn validate_code(&self, code: &[u8]) -> Result<Vec<String>> {
        let mut warnings = Vec::new();
        
        // 检查文件大小
        if code.len() > 5 * 1024 * 1024 {
            warnings.push("WASM模块大小超过5MB，可能影响性能".to_string());
        }
        
        // 检查WASM魔数
        if code.len() < 4 || code[0..4] != [0x00, 0x61, 0x73, 0x6D] {
            return Err(Error::invalid_argument("无效的WASM模块"));
        }
        
        // 这里可以添加更多的代码验证逻辑
        // 例如，扫描危险操作，检查内存使用等
        
        Ok(warnings)
    }
    
    async fn set_env_var(&self, name: &str, value: &str) -> Result<()> {
        if let Some(env) = &self.environment {
            let env_guard = env.read().await;
            
            // 检查环境变量名是否在允许列表中
            if !env_guard.security_context.allowed_env_vars.contains(&name.to_string()) {
                return Err(Error::permission_denied(format!("环境变量不允许设置: {}", name)));
            }
            
            // 在实际实现中，这里应该设置WASI环境变量
            debug!("【WasmSandbox】设置环境变量: {}={}", name, value);
            Ok(())
        } else {
            Err(Error::failed_precondition("沙箱环境未初始化"))
        }
    }
    
    async fn get_status(&self) -> Result<SandboxStatus> {
        let status = self.status.read().map_err(|_| {
            Error::lock("无法获取沙箱状态锁")
        })?;
        Ok(status.clone())
    }

    async fn get_all_algorithm_definitions(&self) -> Result<std::collections::HashMap<String, crate::core::interfaces::AlgorithmDefinition>> {
        Ok(std::collections::HashMap::new())
    }

    async fn execute_algorithm(&self, _algorithm: &crate::core::interfaces::AlgorithmDefinition, _data: &crate::data::DataBatch) -> Result<crate::algorithm::types::ExecutionResult> {
        Ok(crate::algorithm::types::ExecutionResult::success(vec![]))
    }

    async fn prepare_input_data(&self, _data: &crate::data::DataBatch) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }

    async fn process_sandbox_result(&self, _algorithm_id: &str, _sandbox_result: crate::algorithm::executor::sandbox::result::SandboxResult) -> Result<crate::algorithm::types::ExecutionResult> {
        Ok(crate::algorithm::types::ExecutionResult::success(vec![]))
    }
} 