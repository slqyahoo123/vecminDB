#[cfg(feature = "wasmtime")]
use std::sync::Arc;
#[cfg(feature = "wasmtime")]
use std::time::Duration;
#[cfg(feature = "wasmtime")]
use wasmtime::{Module, Linker};
#[cfg(feature = "wasmtime")]
use wasmtime_wasi::{WasiCtx};
#[cfg(feature = "wasmtime")]
use std::sync::atomic::Ordering;
#[cfg(feature = "wasmtime")]
use log::{debug, warn};

use crate::Result;
#[cfg(feature = "wasmtime")]
use crate::algorithm::types::{ResourceLimits};
#[cfg(feature = "wasmtime")]
use crate::algorithm::executor::sandbox::environment::ExecutionEnvironment;
#[cfg(feature = "wasmtime")]
use crate::algorithm::executor::sandbox::error::SandboxError;
#[cfg(feature = "wasmtime")]
use crate::algorithm::executor::sandbox::result::SandboxResult;
#[cfg(feature = "wasmtime")]
use crate::algorithm::executor::config::SandboxConfig;

/// 强制执行资源限制
///
/// 在一个循环中监控资源使用情况，如果超出限制则返回错误。
#[cfg(feature = "wasmtime")]
async fn enforce_resource_limits(
    env: &Arc<ExecutionEnvironment>,
    limits: &ResourceLimits,
) -> Result<(), SandboxError> {
    loop {
        // 检查内存限制
        if let Some(limit_mb) = limits.max_memory_mb() {
            let current_usage = env.memory_bytes.load(Ordering::SeqCst);
            if current_usage > (limit_mb * 1024 * 1024) as u64 {
                let msg = format!("内存使用超出限制: {} > {}MB", current_usage, limit_mb);
                if let Ok(mut usage) = env.resource_usage.try_lock() {
                    usage.set_limit_exceeded("memory", &msg);
                }
                return Err(SandboxError::ResourceExceeded(msg));
            }
        }

        // 检查CPU限制
        if let Some(_limit_percent) = limits.max_cpu_percent() {
            // 注意: CPU限制的精确执行需要更复杂的平台相关代码。
            // 这里我们暂时使用一个占位符。
        }

        // 短暂休眠以避免繁忙循环
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// 在沙箱中执行WASM模块
#[cfg(feature = "wasmtime")]
pub async fn execute_in_sandbox(
    module: &Arc<wasmtime::Module>,
    input: &[u8],
    config: &SandboxConfig,
    limits: ResourceLimits,
    timeout: Duration,
) -> Result<SandboxResult> {
    debug!("开始在沙箱执行WASM模块, 输入大小: {}, 超时: {:?}", input.len(), timeout);
    
    // 创建沙箱ID
    let sandbox_id = uuid::Uuid::new_v4().to_string();
    let start_time = std::time::Instant::now();
    
    // 创建执行环境
    let env = Arc::new(ExecutionEnvironment::new(config.clone(), &sandbox_id));
    
    // 启动资源监控
    if let Err(e) = env.start_monitoring().await {
        warn!("无法启动资源监控: {}", e);
    }
    
    // 将WASM执行和资源限制检查并发执行
    #[cfg(feature = "wasmtime")]
    let wasm_execution_future = async {
        let env_clone = env.clone();
        
        // 创建WASM运行环境
        if let Err(e) = env_clone.create_wasm_environment() {
            return Err(SandboxError::InitializationFailed(format!("创建WASM环境失败: {}", e)));
        }
        
        let engine = env_clone.get_wasm_engine()
            .ok_or_else(|| SandboxError::InitializationFailed("WASM引擎未初始化".to_string()))?;
            
        let mut store = env_clone.take_wasm_store()
            .ok_or_else(|| SandboxError::InitializationFailed("WASM存储未初始化".to_string()))?;
        
        // 创建Linker
        let mut linker = Linker::<WasiCtx>::new(&engine);
        
        // 添加WASI支持 - 使用正确的API
        add_to_linker(&mut linker, &mut store)
            .map_err(|e| SandboxError::InitializationFailed(format!("添加WASI支持失败: {}", e)))?;
        
        // 实例化模块
        let instance = linker.instantiate(&mut store, &module)
            .map_err(|e| SandboxError::WasmInstantiation(format!("实例化WASM模块失败: {}", e)))?;
        
        // 分配内存并写入输入数据
        let (input_ptr, input_len) = copy_input_to_memory(&instance, &mut store, input)
            .map_err(|e| SandboxError::MemoryAccess(format!("分配或写入内存失败: {}", e)))?;
        
        // 查找入口函数
        let entry_fn = instance.get_func(&mut store, "run")
            .ok_or_else(|| SandboxError::FunctionNotFound("找不到 'run' 入口函数".to_string()))?;
        
        // 创建参数和结果缓冲区
        let args = [
            wasmtime::Val::I32(input_ptr as i32),
            wasmtime::Val::I32(input_len as i32)
        ];
        let mut results = [wasmtime::Val::I32(0)];
        
        // 调用入口函数
        entry_fn.call(&mut store, &args, &mut results)
            .map_err(|e| SandboxError::WasmExecution(format!("执行WASM函数失败: {}", e)))?;
        
        // 获取结果指针和长度
        let result_ptr_fn = instance.get_func(&mut store, "get_result_ptr")
            .ok_or_else(|| SandboxError::FunctionNotFound("get_result_ptr函数不存在".to_string()))?;
        let result_len_fn = instance.get_func(&mut store, "get_result_len")
            .ok_or_else(|| SandboxError::FunctionNotFound("get_result_len函数不存在".to_string()))?;

        let mut result_ptr_val = [wasmtime::Val::I32(0)];
        result_ptr_fn.call(&mut store, &[], &mut result_ptr_val)
            .map_err(|e| SandboxError::WasmExecution(format!("获取结果指针失败: {}", e)))?;

        let mut result_len_val = [wasmtime::Val::I32(0)];
        result_len_fn.call(&mut store, &[], &mut result_len_val)
            .map_err(|e| SandboxError::WasmExecution(format!("获取结果长度失败: {}", e)))?;

        let result_ptr = result_ptr_val[0].unwrap_i32() as u32;
        let result_len = result_len_val[0].unwrap_i32() as u32;

        // 读取结果数据
        let result_data = read_from_memory(&instance, &mut store, result_ptr, result_len)
            .map_err(|e| SandboxError::MemoryAccess(format!("读取结果数据失败: {}", e)))?;
        
        // 转换结果为字符串
        let stdout = String::from_utf8(result_data)
            .map_err(|e| SandboxError::ExecutionFailed(format!("结果不是有效的UTF-8字符串: {}", e)))?;
        
        let stderr = env_clone.get_stderr();
        
        // 计算资源使用情况
        let final_usage = {
            let usage = env.resource_usage.lock().unwrap();
            let mut updated_usage = usage.clone();
            updated_usage.execution_time_ms = start_time.elapsed().as_millis() as u64;
            updated_usage
        };

        Ok(SandboxResult::success(
            stdout,
            stderr,
            start_time.elapsed().as_millis() as u64,
            final_usage,
        ))
    };

    let limits_check_future = enforce_resource_limits(&env, &limits);

    #[cfg(feature = "wasmtime")]
    let execution_result = tokio::select! {
        res = tokio::time::timeout(timeout, wasm_execution_future) => {
            match res {
                Ok(Ok(result)) => Ok(result),
                Ok(Err(e)) => Err(e),
                Err(_) => {
                    let usage = {
                        let usage_guard = env.resource_usage.lock().unwrap();
                        let mut updated_usage = usage_guard.clone();
                        updated_usage.set_limit_exceeded("timeout", &format!("执行超过最大允许时间: {:?}", timeout));
                        updated_usage
                    };
                    Ok(SandboxResult::timeout(
                        timeout.as_millis() as u64,
                        env.get_stdout(),
                        env.get_stderr(),
                        usage,
                    ))
                }
            }
        },
        res = limits_check_future => {
            match res {
                Ok(_) => unreachable!(), // 循环永不正常退出
                Err(e) => Err(e),
            }
        }
    };
    
    #[cfg(not(feature = "wasmtime"))]
    let execution_result: Result<SandboxResult> = Err(Error::feature_not_enabled("wasmtime"));
    
    // 停止资源监控
    if let Err(e) = env.stop_monitoring().await {
        warn!("停止资源监控失败: {}", e);
    }
    
    // 清理资源
    if let Err(e) = env.cleanup() {
        warn!("清理沙箱资源失败: {}", e);
    }
    
    // 处理最终结果
    match execution_result {
        Ok(result) => {
             debug!(
                "WASM模块执行完成, 执行时间: {}ms, 输出大小: {}", 
                result.execution_time_ms, 
                result.stdout.len()
            );
            Ok(result)
        },
        Err(e) => {
            error!("WASM模块执行失败: {}", e);
            Err(e.into())
        }
    }
}

/// 复制输入数据到WASM内存
#[cfg(feature = "wasmtime")]
pub fn copy_input_to_memory(instance: &wasmtime::Instance, store: &mut wasmtime::Store<WasiCtx>, input: &[u8]) -> Result<(u32, u32)> {
    debug!("复制输入数据到WASM内存, 大小: {}", input.len());
    
    // 1. 获取WASM内存
    let memory = match instance.get_memory(store, "memory") {
        Some(mem) => mem,
        None => return Err(Error::internal("无法获取WASM内存")),
    };
    
    // 2. 获取内存分配函数
    let alloc_fn = match instance.get_func(store, "alloc") {
        Some(func) => func,
        None => {
            // 尝试使用其他常见的分配函数名
            instance.get_func(store, "malloc")
                .or_else(|| instance.get_func(store, "allocate"))
                .or_else(|| instance.get_func(store, "__wbindgen_malloc"))
                .ok_or_else(|| Error::not_found("找不到内存分配函数，缺少alloc、malloc、allocate或__wbindgen_malloc"))?
        }
    };
    
    // 3. 调用内存分配函数
    let mut result = [wasmtime::Val::I32(0)];
    if let Err(e) = alloc_fn.call(store, &[wasmtime::Val::I32(input.len() as i32)], &mut result) {
        return Err(Error::execution(format!("调用内存分配函数失败: {}", e)));
    }
    
    // 提取分配的内存指针
    let input_ptr = match result[0].unwrap_i32() {
        0 => return Err(Error::resource_exhausted("内存分配失败，返回空指针")),
        ptr => ptr as u32,
    };
    
    // 4. 将输入数据写入分配的内存
    if let Err(e) = memory.write(store, input_ptr as usize, input) {
        // 尝试释放已分配的内存
        if let Err(e) = free_memory(instance, store, input_ptr, input.len() as u32) {
            warn!("释放已分配内存失败: {}", e);
        }
        
        return Err(Error::out_of_memory(format!("写入WASM内存失败: {}", e)));
    }
    
    // 5. 返回内存指针和长度
    debug!("成功将数据复制到WASM内存，指针: {}, 长度: {}", input_ptr, input.len());
    Ok((input_ptr, input.len() as u32))
}

/// 从WASM内存读取数据
#[cfg(feature = "wasmtime")]
pub fn read_from_memory(instance: &wasmtime::Instance, store: &mut wasmtime::Store<wasmtime_wasi::WasiCtx>, ptr: u32, len: u32) -> Result<Vec<u8>> {
    debug!("从WASM内存读取数据, 指针: {}, 长度: {}", ptr, len);
    
    // 1. 获取WASM内存
    let memory = match instance.get_memory(store, "memory") {
        Some(mem) => mem,
        None => return Err(Error::internal("无法获取WASM内存")),
    };
    
    // 2. 验证内存访问范围
    let max_size = memory.data_size(store);
    if (ptr as usize) + (len as usize) > max_size {
        return Err(Error::invalid_argument(format!(
            "内存访问越界: 尝试读取 {} 字节，起始于 {}，但内存大小只有 {}", 
            len, ptr, max_size
        )));
    }
    
    // 3. 分配缓冲区
    let mut buffer = vec![0u8; len as usize];
    
    // 4. 读取内存
    if let Err(e) = memory.read(store, ptr as usize, &mut buffer) {
        return Err(Error::out_of_memory(format!("读取WASM内存失败: {}", e)));
    }
    
    debug!("成功从WASM内存读取数据, 大小: {}", buffer.len());
    Ok(buffer)
}

/// 从WASM内存释放数据
#[cfg(feature = "wasmtime")]
pub fn free_memory(instance: &wasmtime::Instance, store: &mut wasmtime::Store<WasiCtx>, ptr: u32, len: u32) -> Result<()> {
    debug!("释放WASM内存, 指针: {}, 长度: {}", ptr, len);
    
    // 尝试各种常见的内存释放函数名
    let free_fn = instance.get_func(&mut *store, "free")
        .or_else(|| instance.get_func(&mut *store, "dealloc"))
        .or_else(|| instance.get_func(&mut *store, "deallocate"))
        .or_else(|| instance.get_func(&mut *store, "__wbindgen_free"));
    
    if let Some(func) = free_fn {
        if let Err(e) = func.call(&mut *store, &[wasmtime::Val::I32(ptr as i32)], &mut []) {
            return Err(Error::execution(format!("调用内存释放函数失败: {}", e)));
        }
        debug!("成功释放WASM内存");
        Ok(())
    } else {
        // 某些WASM模块可能没有显式的释放函数
        // 对于这种情况，我们可以依赖WASM的自动内存管理
        debug!("未找到内存释放函数，将依赖WASM内存管理");
        Ok(())
    }
}

/// 安全的WASM内存访问封装
#[cfg(feature = "wasmtime")]
pub struct WasmMemoryManager<'a> {
    instance: &'a wasmtime::Instance,
    store: &'a mut wasmtime::Store<wasmtime_wasi::WasiCtx>,
    allocated_blocks: Vec<(u32, u32)>, // (指针, 长度)
}

#[cfg(feature = "wasmtime")]
impl<'a> WasmMemoryManager<'a> {
    /// 创建新的内存管理器
    pub fn new(instance: &'a wasmtime::Instance, store: &'a mut wasmtime::Store<wasmtime_wasi::WasiCtx>) -> Self {
        Self {
            instance,
            store,
            allocated_blocks: Vec::new(),
        }
    }
    
    /// 分配并写入数据
    pub fn allocate_and_write(&mut self, data: &[u8]) -> Result<(u32, u32)> {
        let (ptr, len) = copy_input_to_memory(self.instance, self.store, data)?;
        self.allocated_blocks.push((ptr, len));
        Ok((ptr, len))
    }
    
    /// 读取数据
    pub fn read(&mut self, ptr: u32, len: u32) -> Result<Vec<u8>> {
        read_from_memory(self.instance, self.store, ptr, len)
    }
    
    /// 手动释放单个内存块
    pub fn free(&mut self, ptr: u32, len: u32) -> Result<()> {
        if let Some(index) = self.allocated_blocks.iter().position(|&block| block == (ptr, len)) {
            self.allocated_blocks.remove(index);
        }
        free_memory(self.instance, self.store, ptr, len)
    }
    
    /// 获取当前已分配的总内存大小（字节）
    pub fn get_total_allocated_size(&self) -> u64 {
        self.allocated_blocks.iter()
            .map(|(_, len)| *len as u64)
            .sum()
    }
    
    /// 获取当前分配的内存块数量
    pub fn get_block_count(&self) -> usize {
        self.allocated_blocks.len()
    }
    
    /// 释放所有已分配的内存
    pub fn free_all(&mut self) -> Result<()> {
        let blocks = std::mem::take(&mut self.allocated_blocks);
        for (ptr, len) in blocks {
            if let Err(e) = free_memory(self.instance, self.store, ptr, len) {
                warn!("释放内存块失败: 指针 {}, 长度 {}, 错误: {}", ptr, len, e);
                // 继续尝试释放其他块
            }
        }
        Ok(())
    }
}

#[cfg(feature = "wasmtime")]
impl<'a> Drop for WasmMemoryManager<'a> {
    fn drop(&mut self) {
        // 释放所有尚未手动释放的内存块
        for (ptr, len) in self.allocated_blocks.drain(..) {
            if let Err(e) = free_memory(self.instance, self.store, ptr, len) {
                warn!("自动释放WASM内存失败: 指针 {}, 长度 {}, 错误: {}", ptr, len, e);
            }
        }
    }
}

/// 向 Wasmtime 链接器添加主机函数
/// 
/// 这个函数负责将各种主机提供的功能绑定到 WASM 实例中，
/// 使得运行在沙箱中的算法可以访问受控的系统功能。
#[cfg(feature = "wasmtime")]
pub fn add_to_linker<T>(
    linker: &mut wasmtime::Linker<T>, 
    _store: &mut wasmtime::Store<T>
) -> Result<(), anyhow::Error> 
where 
    T: Send + 'static,
{
    // 添加基本 I/O 函数
    add_io_functions(linker)?;
    
    // 添加内存管理函数
    add_memory_functions(linker)?;
    
    // 添加数学计算函数
    add_math_functions(linker)?;
    
    // 添加时间相关函数
    add_time_functions(linker)?;
    
    // 添加日志记录函数
    add_logging_functions(linker)?;
    
    // 添加资源监控函数
    add_monitoring_functions(linker)?;
    
    log::info!("成功向链接器添加了所有主机函数");
    Ok(())
}

/// 添加 I/O 相关函数
fn add_io_functions<T>(linker: &mut wasmtime::Linker<T>) -> Result<(), anyhow::Error> 
where 
    T: Send + 'static,
{
    // 安全的打印函数
    linker.func_wrap("env", "host_print", |caller: wasmtime::Caller<'_, T>, ptr: i32, len: i32| {
        if let Ok(memory) = caller.get_export("memory") {
            if let Some(memory) = memory.into_memory() {
                let data = memory.data(&caller);
                let start = ptr as usize;
                let end = start + len as usize;
                
                if end <= data.len() {
                    if let Ok(text) = std::str::from_utf8(&data[start..end]) {
                        print!("{}", text);
                    }
                }
            }
        }
    })?;
    
    // 安全的读取输入函数（受限）
    linker.func_wrap("env", "host_read_input", |_caller: wasmtime::Caller<'_, T>| -> i32 {
        // 在生产环境中，这里应该返回错误或空值
        // 目前返回0表示没有可用输入
        0
    })?;
    
    Ok(())
}

/// 添加内存管理相关函数
fn add_memory_functions<T>(linker: &mut wasmtime::Linker<T>) -> Result<(), anyhow::Error> 
where 
    T: Send + 'static,
{
    // 内存使用情况查询
    linker.func_wrap("env", "host_get_memory_usage", |_caller: wasmtime::Caller<'_, T>| -> i32 {
        // 返回当前进程的内存使用情况（KB）
        if let Ok(usage) = get_current_memory_usage() {
            usage as i32
        } else {
            -1
        }
    })?;
    
    // 垃圾回收触发（如果WASM运行时支持）
    linker.func_wrap("env", "host_gc", |_caller: wasmtime::Caller<'_, T>| {
        // 这里可以触发主机端的内存回收
        // 具体实现取决于运行时环境
    })?;
    
    Ok(())
}

/// 添加数学计算相关函数
fn add_math_functions<T>(linker: &mut wasmtime::Linker<T>) -> Result<(), anyhow::Error> 
where 
    T: Send + 'static,
{
    // 高精度数学函数（如果WASM不支持某些函数）
    linker.func_wrap("env", "host_sin", |_caller: wasmtime::Caller<'_, T>, x: f64| -> f64 {
        x.sin()
    })?;
    
    linker.func_wrap("env", "host_cos", |_caller: wasmtime::Caller<'_, T>, x: f64| -> f64 {
        x.cos()
    })?;
    
    linker.func_wrap("env", "host_sqrt", |_caller: wasmtime::Caller<'_, T>, x: f64| -> f64 {
        x.sqrt()
    })?;
    
    linker.func_wrap("env", "host_pow", |_caller: wasmtime::Caller<'_, T>, base: f64, exp: f64| -> f64 {
        base.powf(exp)
    })?;
    
    Ok(())
}

/// 添加时间相关函数
fn add_time_functions<T>(linker: &mut wasmtime::Linker<T>) -> Result<(), anyhow::Error> 
where 
    T: Send + 'static,
{
    // 获取当前时间戳（毫秒）
    linker.func_wrap("env", "host_get_timestamp", |_caller: wasmtime::Caller<'_, T>| -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    })?;
    
    // 高精度睡眠函数
    linker.func_wrap("env", "host_sleep_ms", |_caller: wasmtime::Caller<'_, T>, ms: i32| {
        if ms > 0 && ms <= 1000 { // 限制睡眠时间最多1秒
            std::thread::sleep(std::time::Duration::from_millis(ms as u64));
        }
    })?;
    
    Ok(())
}

/// 添加日志记录相关函数
fn add_logging_functions<T>(linker: &mut wasmtime::Linker<T>) -> Result<(), anyhow::Error> 
where 
    T: Send + 'static,
{
    // 结构化日志记录
    linker.func_wrap("env", "host_log_info", |caller: wasmtime::Caller<'_, T>, ptr: i32, len: i32| {
        if let Ok(memory) = caller.get_export("memory") {
            if let Some(memory) = memory.into_memory() {
                let data = memory.data(&caller);
                let start = ptr as usize;
                let end = start + len as usize;
                
                if end <= data.len() {
                    if let Ok(text) = std::str::from_utf8(&data[start..end]) {
                        log::info!("[WASM] {}", text);
                    }
                }
            }
        }
    })?;
    
    linker.func_wrap("env", "host_log_warn", |caller: wasmtime::Caller<'_, T>, ptr: i32, len: i32| {
        if let Ok(memory) = caller.get_export("memory") {
            if let Some(memory) = memory.into_memory() {
                let data = memory.data(&caller);
                let start = ptr as usize;
                let end = start + len as usize;
                
                if end <= data.len() {
                    if let Ok(text) = std::str::from_utf8(&data[start..end]) {
                        log::warn!("[WASM] {}", text);
                    }
                }
            }
        }
    })?;
    
    linker.func_wrap("env", "host_log_error", |caller: wasmtime::Caller<'_, T>, ptr: i32, len: i32| {
        if let Ok(memory) = caller.get_export("memory") {
            if let Some(memory) = memory.into_memory() {
                let data = memory.data(&caller);
                let start = ptr as usize;
                let end = start + len as usize;
                
                if end <= data.len() {
                    if let Ok(text) = std::str::from_utf8(&data[start..end]) {
                        log::error!("[WASM] {}", text);
                    }
                }
            }
        }
    })?;
    
    Ok(())
}

/// 添加资源监控相关函数
fn add_monitoring_functions<T>(linker: &mut wasmtime::Linker<T>) -> Result<(), anyhow::Error> 
where 
    T: Send + 'static,
{
    // CPU 使用率查询
    linker.func_wrap("env", "host_get_cpu_usage", |_caller: wasmtime::Caller<'_, T>| -> f32 {
        // 返回当前CPU使用率（百分比）
        get_current_cpu_usage().unwrap_or(0.0)
    })?;
    
    // 磁盘空间查询
    linker.func_wrap("env", "host_get_disk_space", |_caller: wasmtime::Caller<'_, T>| -> i64 {
        // 返回可用磁盘空间（字节）
        get_available_disk_space().unwrap_or(0)
    })?;
    
    // 网络状态查询
    linker.func_wrap("env", "host_get_network_status", |_caller: wasmtime::Caller<'_, T>| -> i32 {
        // 返回网络连接状态（0=无连接，1=已连接）
        if is_network_available() { 1 } else { 0 }
    })?;
    
    Ok(())
}

/// 获取当前内存使用情况（KB）（生产级实现：使用 sysinfo 查询真实系统内存）
fn get_current_memory_usage() -> Result<u64, anyhow::Error> {
    #[cfg(feature = "sysinfo")]
    {
        use sysinfo::{System, SystemExt, ProcessExt, Pid};
        
        let mut system = System::new();
        system.refresh_process(Pid::from(std::process::id() as usize));
        
        if let Some(process) = system.process(Pid::from(std::process::id() as usize)) {
            // 返回内存使用量（KB）
            Ok(process.memory() / 1024)
        } else {
            // 如果无法获取进程信息，返回系统总内存使用量
            system.refresh_memory();
            Ok(system.used_memory() / 1024)
        }
    }
    
    #[cfg(not(feature = "sysinfo"))]
    {
        // 降级实现：使用平台特定的方法
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            use std::io::BufRead;
            
            if let Ok(file) = fs::File::open("/proc/self/status") {
                let reader = std::io::BufReader::new(file);
                for line in reader.lines() {
                    if let Ok(line) = line {
                        if line.starts_with("VmRSS:") {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 2 {
                                if let Ok(kb) = parts[1].parse::<u64>() {
                                    return Ok(kb);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 如果所有方法都失败，返回错误
        Err(anyhow::anyhow!("无法获取内存使用情况：sysinfo 功能未启用且平台特定方法失败"))
    }
}

/// 获取当前CPU使用率（生产级实现：使用 sysinfo 查询真实CPU使用率）
fn get_current_cpu_usage() -> Result<f32, anyhow::Error> {
    #[cfg(feature = "sysinfo")]
    {
        use sysinfo::{System, SystemExt, ProcessExt, Pid};
        
        let mut system = System::new();
        system.refresh_process(Pid::from(std::process::id() as usize));
        
        if let Some(process) = system.process(Pid::from(std::process::id() as usize)) {
            Ok(process.cpu_usage())
        } else {
            // 如果无法获取进程信息，返回系统全局CPU使用率
            system.refresh_cpu();
            std::thread::sleep(std::time::Duration::from_millis(100));
            system.refresh_cpu();
            Ok(system.global_cpu_info().cpu_usage())
        }
    }
    
    #[cfg(not(feature = "sysinfo"))]
    {
        // 降级实现：返回0.0表示无法获取
        Ok(0.0)
    }
}

/// 获取可用磁盘空间（生产级实现：使用系统调用查询真实磁盘空间）
fn get_available_disk_space() -> Result<i64, anyhow::Error> {
    // 获取当前工作目录
    let current_dir = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("无法获取当前目录: {}", e))?;
    
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::mem;
        use std::os::unix::ffi::OsStrExt;
        
        #[repr(C)]
        struct StatVfs {
            f_bsize: u64,
            f_frsize: u64,
            f_blocks: u64,
            f_bfree: u64,
            f_bavail: u64,
            f_files: u64,
            f_ffree: u64,
            f_favail: u64,
            f_fsid: u64,
            f_flag: u64,
            f_namemax: u64,
        }
        
        extern "C" {
            fn statvfs(path: *const i8, buf: *mut StatVfs) -> i32;
        }
        
        let path_cstr = CString::new(current_dir.as_os_str().as_bytes())
            .map_err(|e| anyhow::anyhow!("路径转换失败: {}", e))?;
        
        let mut statvfs_buf: StatVfs = unsafe { mem::zeroed() };
        let result = unsafe { statvfs(path_cstr.as_ptr(), &mut statvfs_buf) };
        
        if result == 0 {
            let block_size = statvfs_buf.f_frsize;
            let available = statvfs_buf.f_bavail * block_size;
            Ok(available as i64)
        } else {
            Err(anyhow::anyhow!("statvfs调用失败: {}", std::io::Error::last_os_error()))
        }
    }
    
    #[cfg(windows)]
    {
        use std::ffi::OsStr;
        use std::iter::once;
        use std::os::windows::ffi::OsStrExt;
        
        // 获取驱动器根路径
        let path_str = current_dir.to_string_lossy();
        let drive_letter = if path_str.len() >= 2 && path_str.chars().nth(1) == Some(':') {
            path_str.chars().next().unwrap().to_ascii_uppercase()
        } else {
            return Err(anyhow::anyhow!("无法确定Windows驱动器字母"));
        };
        
        let drive_path = format!("{}:\\", drive_letter);
        
        // 转换为宽字符
        let wide_path: Vec<u16> = OsStr::new(&drive_path)
            .encode_wide()
            .chain(once(0))
            .collect();
        
        let mut free_bytes_available = 0u64;
        
        extern "system" {
            fn GetDiskFreeSpaceExW(
                directory_name: *const u16,
                free_bytes_available: *mut u64,
                _total_bytes: *mut u64,
                _total_free_bytes: *mut u64,
            ) -> i32;
        }
        
        let result = unsafe {
            GetDiskFreeSpaceExW(
                wide_path.as_ptr(),
                &mut free_bytes_available,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        
        if result != 0 {
            Ok(free_bytes_available as i64)
        } else {
            Err(anyhow::anyhow!("GetDiskFreeSpaceExW调用失败: {}", std::io::Error::last_os_error()))
        }
    }
    
    #[cfg(not(any(unix, windows)))]
    {
        Err(anyhow::anyhow!("当前平台不支持磁盘空间查询"))
    }
}

/// 检查网络连接状态（生产级实现：尝试连接外部服务验证网络可用性）
fn is_network_available() -> bool {
    // 方法1：检查是否有活动的网络接口
    #[cfg(feature = "sysinfo")]
    {
        use sysinfo::{System, SystemExt};
        
        let mut system = System::new();
        system.refresh_networks_list();
        system.refresh_networks();
        
        // 检查是否有活动的网络接口
        for (_interface_name, network) in system.networks() {
            if network.total_received() > 0 || network.total_transmitted() > 0 {
                return true;
            }
        }
    }
    
    // 方法2：尝试解析DNS（轻量级检查）
    use std::net::ToSocketAddrs;
    if "google.com:80".to_socket_addrs().is_ok() {
        return true;
    }
    
    // 如果所有检查都失败，返回false
    false
} 