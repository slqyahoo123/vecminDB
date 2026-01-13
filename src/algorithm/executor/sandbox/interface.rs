use std::path::Path;
use std::time::Duration;
use async_trait::async_trait;
use crate::Result;
use crate::algorithm::types::ResourceUsage;
use super::result::SandboxResult;
use crate::algorithm::types::SandboxStatus;

/// 沙箱接口
#[async_trait]
pub trait Sandbox: std::fmt::Debug + Send + Sync {
    /// 获取沙箱ID
    fn id(&self) -> &str;
    
    /// 准备沙箱环境
    async fn prepare(&self) -> Result<()>;
    
    /// 异步执行代码
    async fn execute(&self, code: &[u8], input: &[u8], timeout: Duration) -> Result<SandboxResult>;
    
    /// 清理沙箱资源
    async fn cleanup(&self) -> Result<()>;
    
    /// 是否支持指定的文件类型
    fn supports_file_type(&self, file_type: &str) -> bool;
    
    /// 加载文件到沙箱
    async fn load_file(&self, src_path: &Path, sandbox_path: &str) -> Result<()>;
    
    /// 从沙箱保存文件
    async fn save_file(&self, sandbox_path: &str, dest_path: &Path) -> Result<()>;
    
    /// 取消执行
    async fn cancel(&self) -> Result<()>;
    
    /// 获取资源使用情况
    async fn get_resource_usage(&self) -> Result<ResourceUsage>;
    
    /// 验证代码安全性
    async fn validate_code(&self, code: &[u8]) -> Result<Vec<String>>;
    
    /// 设置环境变量
    async fn set_env_var(&self, name: &str, value: &str) -> Result<()>;
    
    /// 获取沙箱状态
    async fn get_status(&self) -> Result<SandboxStatus>;
    
    /// 获取所有算法定义
    async fn get_all_algorithm_definitions(&self) -> Result<std::collections::HashMap<String, crate::core::interfaces::AlgorithmDefinition>>;
    
    /// 执行算法定义
    async fn execute_algorithm(&self, algorithm: &crate::core::interfaces::AlgorithmDefinition, data: &crate::data::DataBatch) -> Result<crate::algorithm::types::ExecutionResult>;
    
    /// 准备输入数据
    async fn prepare_input_data(&self, data: &crate::data::DataBatch) -> Result<Vec<u8>>;
    
    /// 处理沙箱执行结果
    async fn process_sandbox_result(&self, algorithm_id: &str, sandbox_result: super::result::SandboxResult) -> Result<crate::algorithm::types::ExecutionResult>;
} 