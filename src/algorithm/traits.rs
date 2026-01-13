use std::collections::HashMap;
use async_trait::async_trait;
use crate::error::Result;
use crate::algorithm::types::AlgorithmType;

/// 通用算法接口特征
pub trait Algorithm: Send + Sync + std::fmt::Debug {
    /// 获取算法ID
    fn get_id(&self) -> &str;
    
    /// 获取算法名称
    fn get_name(&self) -> &str;
    
    /// 获取算法描述
    fn get_description(&self) -> Option<&str>;
    
    /// 获取算法版本
    fn get_version(&self) -> u32;
    
    /// 执行算法
    fn execute(&self, input: &[u8]) -> Result<Vec<u8>>;
    
    /// 验证算法
    fn validate(&self) -> Result<()>;
    
    /// 获取算法类型
    fn get_algorithm_type(&self) -> &AlgorithmType;
    
    /// 获取算法元数据
    fn get_metadata(&self) -> &HashMap<String, String>;
    
    /// 获取算法依赖
    fn get_dependencies(&self) -> &[String];
    
    /// 获取创建时间
    fn get_created_at(&self) -> i64;
    
    /// 获取更新时间
    fn get_updated_at(&self) -> i64;
    
    /// 获取算法类型（兼容旧API）
    fn get_type(&self) -> AlgorithmType {
        *self.get_algorithm_type()
    }
    
    /// 获取算法类型（新方法名）
    fn algorithm_type(&self) -> AlgorithmType {
        self.get_type()
    }
    
    /// 获取算法描述（兼容API）
    fn description(&self) -> Option<String> {
        self.get_description().map(|s| s.to_string())
    }
    
    /// 获取算法代码（默认实现）
    fn get_code(&self) -> &str {
        ""
    }
    
    /// 获取算法配置（默认实现）
    fn get_config(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    
    /// 获取算法参数（用于存储）
    fn get_parameters(&self) -> HashMap<String, String> {
        self.get_config()
    }
    
    /// 序列化算法配置
    fn serialize_config(&self) -> Result<Vec<u8>> {
        let config = self.get_config();
        bincode::serialize(&config)
            .map_err(|e| crate::error::Error::SerializationError(e.to_string()))
    }
    
    /// 设置算法配置（默认实现）
    fn set_config(&mut self, _config: HashMap<String, String>) {
        // 默认实现为空，具体类型可以重写
    }
    
    /// 应用算法（默认实现）
    fn apply(&self, _params: &HashMap<String, String>) -> Result<serde_json::Value> {
        Ok(serde_json::Value::Null)
    }
    
    /// 获取算法参数
    fn get_params(&self) -> Result<serde_json::Value> {
        let config = self.get_config();
        serde_json::to_value(config)
            .map_err(|e| crate::error::Error::SerializationError(e.to_string()))
    }
}

/// 算法验证器接口
#[async_trait]
pub trait ValidationInterface {
    async fn validate(&self, algorithm: &dyn Algorithm) -> Result<ValidationReport>;
}

/// 验证报告
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub security_score: u8,
}

/// 用于支持旧版API的适配器
#[async_trait]
pub trait AlgorithmTrait: Send + Sync {
    fn get_id(&self) -> &str;
    fn get_name(&self) -> &str;
    fn get_type(&self) -> AlgorithmType;
    async fn apply(&self, model_id: &str, params: HashMap<String, String>) -> Result<String>;
    /// 添加参数验证方法
    async fn validate_parameters(&self, params: &HashMap<String, String>) -> Result<()>;
}

/// Box<dyn Algorithm>的Algorithm trait实现
impl Algorithm for Box<dyn Algorithm> {
    fn get_id(&self) -> &str {
        (**self).get_id()
    }
    
    fn get_name(&self) -> &str {
        (**self).get_name()
    }
    
    fn get_description(&self) -> Option<&str> {
        (**self).get_description()
    }
    
    fn get_version(&self) -> u32 {
        (**self).get_version()
    }
    
    fn execute(&self, input: &[u8]) -> Result<Vec<u8>> {
        (**self).execute(input)
    }
    
    fn validate(&self) -> Result<()> {
        (**self).validate()
    }
    
    fn get_algorithm_type(&self) -> &AlgorithmType {
        (**self).get_algorithm_type()
    }
    
    fn get_metadata(&self) -> &HashMap<String, String> {
        (**self).get_metadata()
    }
    
    fn get_dependencies(&self) -> &[String] {
        (**self).get_dependencies()
    }
    
    fn get_created_at(&self) -> i64 {
        (**self).get_created_at()
    }
    
    fn get_updated_at(&self) -> i64 {
        (**self).get_updated_at()
    }
    
    fn get_code(&self) -> &str {
        (**self).get_code()
    }
    
    fn get_config(&self) -> HashMap<String, String> {
        (**self).get_config()
    }
    
    fn set_config(&mut self, config: HashMap<String, String>) {
        (**self).set_config(config)
    }
    
    fn apply(&self, params: &HashMap<String, String>) -> Result<serde_json::Value> {
        (**self).apply(params)
    }
}

/// 动态克隆工具
pub mod dyn_clone {
    use std::any::Any;
    
    pub trait DynClone: Any {
        fn clone_box(&self) -> Box<dyn DynClone>;
    }
    
    impl<T: Clone + 'static> DynClone for T {
        fn clone_box(&self) -> Box<dyn DynClone> {
            Box::new(self.clone())
        }
    }
} 