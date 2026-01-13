use std::path::Path;
use std::time::Duration;
use async_trait::async_trait;
use log::debug;

use crate::Result;
use crate::algorithm::executor::config::ExecutorConfig;
use crate::algorithm::types::ResourceUsage;
use crate::algorithm::executor::sandbox::interface::Sandbox;
use crate::algorithm::executor::sandbox::result::SandboxResult;
use crate::algorithm::types::SandboxStatus;
use crate::algorithm::executor::sandbox::error::SandboxError;

/// 示例沙箱实现(开发用)
#[derive(Debug)]
pub struct DummySandbox {
    id: String,
    config: ExecutorConfig,
    status: SandboxStatus,
}

impl DummySandbox {
    pub fn new(id: &str, config: &ExecutorConfig) -> Self {
        Self {
            id: id.to_string(),
            config: config.clone(),
            status: SandboxStatus::Uninitialized,
        }
    }
}

#[async_trait]
impl Sandbox for DummySandbox {
    fn id(&self) -> &str {
        &self.id
    }
    
    async fn prepare(&self) -> Result<()> {
        debug!("【DummySandbox】准备环境: {}", self.id);
        Ok(())
    }
    
    async fn execute(&self, _code: &[u8], _input: &[u8], timeout: Duration) -> Result<SandboxResult> {
        debug!("【DummySandbox】执行代码: {}, 超时: {:?}", self.id, timeout);
        
        // 模拟执行代码
        let execution_time_ms = 100;
        
        if execution_time_ms > timeout.as_millis() as u64 {
            return Err(SandboxError::Timeout(timeout.as_millis() as u64).into());
        }
        
        Ok(SandboxResult::success(
            "{\"result\": \"success\", \"data\": {\"value\": 42}}".to_string(),
            "执行完成".to_string(),
            execution_time_ms,
            ResourceUsage {
                cpu_usage_percent: 10.0,
                memory_usage_bytes: 1024 * 1024,
                peak_memory_bytes: 2 * 1024 * 1024,
                io_read_bytes: 0,
                io_write_bytes: 0,
                network_bytes: 0,
                execution_time_ms,
                limits_exceeded: false,
                exceeded_resource: None,
                code_size_bytes: 0,
                instruction_count: 0,
                output_size_bytes: 0,
            },
        ))
    }
    
    async fn cleanup(&self) -> Result<()> {
        debug!("【DummySandbox】清理资源: {}", self.id);
        Ok(())
    }
    
    fn supports_file_type(&self, file_type: &str) -> bool {
        debug!("【DummySandbox】检查文件类型支持: {}, 类型: {}", self.id, file_type);
        true
    }
    
    async fn load_file(&self, src_path: &Path, sandbox_path: &str) -> Result<()> {
        debug!("【DummySandbox】加载文件: {} -> {}", src_path.display(), sandbox_path);
        Ok(())
    }
    
    async fn save_file(&self, sandbox_path: &str, dest_path: &Path) -> Result<()> {
        debug!("【DummySandbox】保存文件: {} -> {}", sandbox_path, dest_path.display());
        Ok(())
    }
    
    async fn cancel(&self) -> Result<()> {
        debug!("【DummySandbox】取消执行: {}", self.id);
        Ok(())
    }
    
    async fn get_resource_usage(&self) -> Result<ResourceUsage> {
        Ok(ResourceUsage {
            cpu_usage_percent: 10.0,
            memory_usage_bytes: 1024 * 1024,
            peak_memory_bytes: 2 * 1024 * 1024,
            io_read_bytes: 0,
            io_write_bytes: 0,
            network_bytes: 0,
            execution_time_ms: 100,
            limits_exceeded: false,
            exceeded_resource: None,
            code_size_bytes: 0,
            instruction_count: 0,
            output_size_bytes: 0,
        })
    }
    
    async fn validate_code(&self, _code: &[u8]) -> Result<Vec<String>> {
        Ok(vec!["代码验证通过".to_string()])
    }
    
    async fn set_env_var(&self, name: &str, value: &str) -> Result<()> {
        debug!("【DummySandbox】设置环境变量: {}={}", name, value);
        Ok(())
    }
    
    async fn get_status(&self) -> Result<SandboxStatus> {
        Ok(self.status.clone())
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