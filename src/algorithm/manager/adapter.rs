// 算法特性适配器模块
// 提供算法特性的适配实现

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::algorithm::{AlgorithmTrait, AlgorithmType};

/// 算法trait适配器
#[derive(Clone)]
pub struct AlgorithmTraitAdapter {
    /// 算法ID
    id: String,
    /// 算法名称
    name: String,
    /// 算法类型
    algorithm_type: AlgorithmType,
    /// 管理器引用
    manager: Arc<RwLock<crate::algorithm::manager::AlgorithmManager>>,
}

impl AlgorithmTraitAdapter {
    /// 创建新的适配器
    pub fn new(
        id: String,
        name: String,
        algorithm_type: AlgorithmType,
        manager: Arc<RwLock<crate::algorithm::manager::AlgorithmManager>>,
    ) -> Self {
        Self {
            id,
            name,
            algorithm_type,
            manager,
        }
    }
    
    /// 从字符串解析算法类型
    fn parse_algorithm_type(type_str: &str) -> Result<AlgorithmType> {
        match type_str.to_lowercase().as_str() {
            "classification" => Ok(AlgorithmType::Classification),
            "regression" => Ok(AlgorithmType::Regression),
            "clustering" => Ok(AlgorithmType::Clustering),
            "dimensionreduction" => Ok(AlgorithmType::DimensionReduction),
            "anomalydetection" => Ok(AlgorithmType::AnomalyDetection),
            "recommendation" => Ok(AlgorithmType::Recommendation),
            "custom" => Ok(AlgorithmType::Custom),
            _ => Err(Error::invalid_argument(format!("不支持的算法类型: {}", type_str))),
        }
    }
}

#[async_trait]
impl AlgorithmTrait for AlgorithmTraitAdapter {
    fn get_id(&self) -> &str {
        &self.id
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn get_type(&self) -> AlgorithmType {
        self.algorithm_type.clone()
    }
    
    async fn apply(&self, model_id: &str, params: HashMap<String, String>) -> Result<String> {
        // 将params字符串值转换为Value
        let mut json_params = HashMap::new();
        for (k, v) in params {
            // 尝试将字符串解析为JSON值，如果失败则作为字符串处理
            let value = match serde_json::from_str::<Value>(&v) {
                Ok(parsed) => parsed,
                Err(_) => Value::String(v),
            };
            json_params.insert(k, value);
        }
        
        // 获取管理器
        let manager = self.manager.read().map_err(|e| 
            Error::internal(format!("无法获取算法管理器锁: {}", e))
        )?;
        
        // 执行算法：仅使用现有字段构造配置
        let config = crate::algorithm::AlgorithmApplyConfig {
            parameters: json_params,
            execution_config: None,
            resource_limits: None,
            timeout_seconds: None,
        };
        
        let result = manager.apply_algorithm(&self.id, model_id, &config).await?;
        
        // 返回输出模型ID
        Ok(result.output_model_id)
    }
    
    /// 验证参数
    async fn validate_parameters(&self, params: &HashMap<String, String>) -> Result<()> {
        // 获取算法管理器
        let manager = self.manager.read()
            .map_err(|e| Error::internal(format!("无法获取算法管理器锁: {}", e)))?;
        
        // 获取算法
        let algorithm = manager.get_algorithm_simple(&self.id)?;
        
        // 获取必需参数列表
        let required_params = match algorithm.get_required_parameters() {
            Some(params) => params,
            None => vec![]
        };
        
        // 验证必需参数是否都已提供
        for param in &required_params {
            if !params.contains_key(param) {
                return Err(Error::validation(format!("缺少必需参数: {}", param)));
            }
        }
        
        // 检查参数是否允许
        for (param_name, _) in params {
            if !algorithm.is_parameter_allowed(param_name) {
                return Err(Error::invalid_argument(format!("不支持的参数: {}", param_name)));
            }
        }
        
        Ok(())
    }
} 