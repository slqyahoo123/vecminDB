use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use std::sync::{Arc, RwLock};
use crate::error::{Result, Error};
use crate::compat::tensor::TensorData;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

/// 通用参数管理接口
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Parameters {
    /// 参数ID
    pub id: Option<String>,
    /// 目标ID（如模型ID等）
    pub target_id: Option<String>,
    /// 参数版本
    pub version: String,
    /// 创建时间
    pub created_at: Option<u64>,
    /// 更新时间
    pub updated_at: Option<u64>,
    /// 参数映射，存储字符串键到参数值的映射
    pub data: HashMap<String, ParameterValue>,
    /// 张量数据映射，用于模型参数
    pub tensor_data: Option<HashMap<String, TensorData>>,
    /// 张量数据映射（为了向后兼容）
    pub tensors: Option<HashMap<String, TensorData>>,
    /// 梯度数据（可选）
    pub gradients: Option<HashMap<String, TensorData>>,
    /// 优化器状态（可选）
    pub optimizer_state: Option<HashMap<String, TensorData>>,
    /// 元数据信息
    pub metadata: HashMap<String, String>,
}

/// 参数值类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParameterValue {
    /// 数值类型
    Number(f64),
    /// 字符串类型
    String(String),
    /// 布尔类型
    Boolean(bool),
    /// 向量类型
    Vector(Vec<f32>),
    /// 矩阵类型
    Matrix(Vec<Vec<f32>>),
    /// 嵌套参数
    Nested(Box<Parameters>),
    /// 张量数据引用
    TensorRef(String),
    /// 整数类型
    Integer(i64),
    /// 浮点数类型
    Float(f64),
    /// 数组类型
    Array(Vec<ParameterValue>),
    /// 对象类型
    Object(HashMap<String, ParameterValue>),
}

impl Parameters {
    /// 创建新的参数集
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            id: None,
            target_id: None,
            version: version.into(),
            created_at: None,
            updated_at: None,
            data: HashMap::new(),
            tensor_data: None,
            tensors: None,
            gradients: None,
            optimizer_state: None,
            metadata: HashMap::new(),
        }
    }

    /// 创建具有ID的参数集
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// 创建具有目标ID的参数集
    pub fn with_target_id(mut self, target_id: impl Into<String>) -> Self {
        self.target_id = Some(target_id.into());
        self
    }

    /// 初始化时间戳
    pub fn init_timestamps(&mut self) -> &mut Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        self.created_at = Some(now);
        self.updated_at = Some(now);
        self
    }

    /// 更新时间戳
    pub fn update_timestamp(&mut self) -> &mut Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        self.updated_at = Some(now);
        self
    }

    /// 添加一个参数
    pub fn set(&mut self, key: impl Into<String>, value: ParameterValue) -> &mut Self {
        self.data.insert(key.into(), value);
        self.update_timestamp();
        self
    }

    /// 设置张量数据
    pub fn set_tensor_data(&mut self, tensor_data: HashMap<String, TensorData>) -> &mut Self {
        self.tensor_data = Some(tensor_data);
        self.update_timestamp();
        self
    }

    /// 设置梯度数据
    pub fn set_gradients(&mut self, gradients: HashMap<String, TensorData>) -> &mut Self {
        self.gradients = Some(gradients);
        self.update_timestamp();
        self
    }

    /// 设置优化器状态
    pub fn set_optimizer_state(&mut self, state: HashMap<String, TensorData>) -> &mut Self {
        self.optimizer_state = Some(state);
        self.update_timestamp();
        self
    }

    /// 更新张量数据
    pub fn update_tensor_data(&mut self, tensor_data: HashMap<String, TensorData>) -> Result<()> {
        if let Some(data) = &mut self.tensor_data {
            *data = tensor_data;
        } else {
            self.tensor_data = Some(tensor_data);
        }
        self.update_timestamp();
        Ok(())
    }

    /// 使用梯度更新张量参数
    pub fn update_with_gradients(&mut self, gradients: HashMap<String, TensorData>, learning_rate: f32) -> Result<()> {
        if let Some(tensor_data) = &mut self.tensor_data {
            for (key, grad) in gradients.iter() {
                if let Some(param) = tensor_data.get_mut(key) {
                    // 简单的梯度下降更新
                    for (p, g) in param.data.iter_mut().zip(grad.data.iter()) {
                        *p -= learning_rate * g;
                    }
                } else {
                    return Err(Error::model(format!("Parameter key not found: {}", key)));
                }
            }
            
            self.update_timestamp();
            
            // 存储梯度信息
            self.gradients = Some(gradients);
            
            Ok(())
        } else {
            Err(Error::model("No tensor data available for gradient update".to_string()))
        }
    }

    /// 获取参数值
    pub fn get(&self, key: &str) -> Option<&ParameterValue> {
        self.data.get(key)
    }

    /// 获取数值类型参数
    pub fn get_number(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(ParameterValue::Number(n)) => Some(*n),
            _ => None,
        }
    }

    /// 获取字符串类型参数
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(ParameterValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// 获取布尔类型参数
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(ParameterValue::Boolean(b)) => Some(*b),
            _ => None,
        }
    }

    /// 获取向量类型参数
    pub fn get_vector(&self, key: &str) -> Option<&Vec<f32>> {
        match self.get(key) {
            Some(ParameterValue::Vector(v)) => Some(v),
            _ => None,
        }
    }

    /// 获取矩阵类型参数
    pub fn get_matrix(&self, key: &str) -> Option<&Vec<Vec<f32>>> {
        match self.get(key) {
            Some(ParameterValue::Matrix(m)) => Some(m),
            _ => None,
        }
    }

    /// 获取张量数据
    pub fn get_tensor_data(&self) -> Option<&HashMap<String, TensorData>> {
        self.tensor_data.as_ref()
    }

    /// 获取梯度数据
    pub fn get_gradients(&self) -> Option<&HashMap<String, TensorData>> {
        self.gradients.as_ref()
    }

    /// 获取优化器状态
    pub fn get_optimizer_state(&self) -> Option<&HashMap<String, TensorData>> {
        self.optimizer_state.as_ref()
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.metadata.insert(key.into(), value.into());
        self.update_timestamp();
        self
    }

    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// 创建新版本
    pub fn create_new_version(&self) -> Self {
        let mut new_params = self.clone();
        
        // 更新版本号
        let version_parts: Vec<&str> = self.version.split('.').collect();
        if version_parts.len() == 3 {
            if let Ok(patch) = version_parts[2].parse::<u32>() {
                new_params.version = format!("{}.{}.{}", version_parts[0], version_parts[1], patch + 1);
            }
        }
        
        // 更新ID和时间戳
        new_params.id = Some(Uuid::new_v4().to_string());
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        new_params.created_at = Some(now);
        new_params.updated_at = Some(now);
        
        new_params
    }

    /// 序列化为JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| Error::data(format!("Failed to serialize parameters: {}", e)))
    }

    /// 从JSON反序列化
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| Error::data(format!("Failed to deserialize parameters: {}", e)))
    }

    /// 序列化为二进制数据
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| Error::data(format!("Failed to serialize parameters: {}", e)))
    }
    
    /// 从二进制数据反序列化
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data)
            .map_err(|e| Error::data(format!("Failed to deserialize parameters: {}", e)))
    }

    /// 合并另一个参数集
    pub fn merge(&mut self, other: &Parameters) -> &mut Self {
        for (k, v) in &other.data {
            self.data.insert(k.clone(), v.clone());
        }
        
        // 合并张量数据
        if let Some(other_tensor_data) = &other.tensor_data {
            if let Some(tensor_data) = &mut self.tensor_data {
                for (k, v) in other_tensor_data {
                    tensor_data.insert(k.clone(), v.clone());
                }
            } else {
                self.tensor_data = Some(other_tensor_data.clone());
            }
        }
        
        // 合并梯度数据
        if let Some(other_gradients) = &other.gradients {
            if let Some(gradients) = &mut self.gradients {
                for (k, v) in other_gradients {
                    gradients.insert(k.clone(), v.clone());
                }
            } else {
                self.gradients = Some(other_gradients.clone());
            }
        }
        
        // 合并优化器状态
        if let Some(other_optimizer_state) = &other.optimizer_state {
            if let Some(optimizer_state) = &mut self.optimizer_state {
                for (k, v) in other_optimizer_state {
                    optimizer_state.insert(k.clone(), v.clone());
                }
            } else {
                self.optimizer_state = Some(other_optimizer_state.clone());
            }
        }
        
        for (k, v) in &other.metadata {
            self.metadata.insert(k.clone(), v.clone());
        }
        
        self.update_timestamp();
        self
    }

    /// 转换为模型参数
    pub fn to_model_parameters(&self, model_id: String) -> Result<crate::model::parameters::ModelParameters> {
        crate::model::parameters::ModelParameters::from_parameters(self, model_id)
    }
    
    /// 从模型参数创建
    pub fn from_model_parameters(params: &crate::model::parameters::ModelParameters) -> Self {
        params.to_parameters()
    }

    /// 获取参数总数量（包括所有类型的参数）
    pub fn len(&self) -> usize {
        let mut count = self.data.len();
        
        // 计算张量数据的参数数量
        if let Some(tensor_data) = &self.tensor_data {
            count += tensor_data.len();
        }
        
        // 计算梯度的参数数量
        if let Some(gradients) = &self.gradients {
            count += gradients.len();
        }
        
        // 计算优化器状态的参数数量
        if let Some(optimizer_state) = &self.optimizer_state {
            count += optimizer_state.len();
        }
        
        count
    }
    
    /// 检查参数集是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty() 
            && self.tensor_data.as_ref().map_or(true, |t| t.is_empty())
            && self.gradients.as_ref().map_or(true, |g| g.is_empty())
            && self.optimizer_state.as_ref().map_or(true, |o| o.is_empty())
    }
}

/// 线程安全的参数管理器
#[derive(Debug, Clone)]
pub struct ParameterManager {
    parameters: Arc<RwLock<Parameters>>,
}

impl ParameterManager {
    /// 创建新的参数管理器
    pub fn new(version: impl Into<String>) -> Self {
        let mut params = Parameters::new(version);
        params.init_timestamps();
        
        Self {
            parameters: Arc::new(RwLock::new(params)),
        }
    }

    /// 创建具有ID的参数管理器
    pub fn with_id(self, id: impl Into<String>) -> Self {
        let mut params = self.parameters.write().unwrap();
        params.id = Some(id.into());
        drop(params);
        self
    }

    /// 创建具有目标ID的参数管理器
    pub fn with_target_id(self, target_id: impl Into<String>) -> Self {
        let mut params = self.parameters.write().unwrap();
        params.target_id = Some(target_id.into());
        drop(params);
        self
    }

    /// 从现有参数创建
    pub fn from_parameters(parameters: Parameters) -> Self {
        Self {
            parameters: Arc::new(RwLock::new(parameters)),
        }
    }

    /// 设置参数
    pub fn set(&self, key: impl Into<String>, value: ParameterValue) -> Result<()> {
        let mut params = self.parameters.write().map_err(|_| Error::lock("Failed to acquire write lock for parameters"))?;
        params.set(key, value);
        Ok(())
    }

    /// 设置张量数据
    pub fn set_tensor_data(&self, tensor_data: HashMap<String, TensorData>) -> Result<()> {
        let mut params = self.parameters.write().map_err(|_| Error::lock("Failed to acquire write lock for parameters"))?;
        params.set_tensor_data(tensor_data);
        Ok(())
    }

    /// 设置梯度数据
    pub fn set_gradients(&self, gradients: HashMap<String, TensorData>) -> Result<()> {
        let mut params = self.parameters.write().map_err(|_| Error::lock("Failed to acquire write lock for parameters"))?;
        params.set_gradients(gradients);
        Ok(())
    }

    /// 设置优化器状态
    pub fn set_optimizer_state(&self, state: HashMap<String, TensorData>) -> Result<()> {
        let mut params = self.parameters.write().map_err(|_| Error::lock("Failed to acquire write lock for parameters"))?;
        params.set_optimizer_state(state);
        Ok(())
    }

    /// 获取参数
    pub fn get(&self, key: &str) -> Result<Option<ParameterValue>> {
        let params = self.parameters.read().map_err(|_| Error::lock("Failed to acquire read lock for parameters"))?;
        Ok(params.get(key).cloned())
    }

    /// 获取张量数据
    pub fn get_tensor_data(&self) -> Result<Option<HashMap<String, TensorData>>> {
        let params = self.parameters.read().map_err(|_| Error::lock("Failed to acquire read lock for parameters"))?;
        Ok(params.get_tensor_data().cloned())
    }

    /// 获取所有参数的克隆
    pub fn get_all(&self) -> Result<Parameters> {
        let params = self.parameters.read().map_err(|_| Error::lock("Failed to acquire read lock for parameters"))?;
        Ok(params.clone())
    }

    /// 添加元数据
    pub fn add_metadata(&self, key: impl Into<String>, value: impl Into<String>) -> Result<()> {
        let mut params = self.parameters.write().map_err(|_| Error::lock("Failed to acquire write lock for parameters"))?;
        params.add_metadata(key, value);
        Ok(())
    }

    /// 序列化为JSON
    pub fn to_json(&self) -> Result<String> {
        let params = self.parameters.read().map_err(|_| Error::lock("Failed to acquire read lock for parameters"))?;
        params.to_json()
    }

    /// 从JSON更新
    pub fn update_from_json(&self, json: &str) -> Result<()> {
        let new_params = Parameters::from_json(json)?;
        let mut params = self.parameters.write().map_err(|_| Error::lock("Failed to acquire write lock for parameters"))?;
        params.merge(&new_params);
        Ok(())
    }

    /// 使用梯度更新张量参数
    pub fn update_with_gradients(&self, gradients: HashMap<String, TensorData>, learning_rate: f32) -> Result<()> {
        let mut params = self.parameters.write().map_err(|_| Error::lock("Failed to acquire write lock for parameters"))?;
        params.update_with_gradients(gradients, learning_rate)
    }

    /// 转换为模型参数对象
    pub fn to_model_parameters(&self) -> Result<crate::model::parameters::ModelParameters> {
        let params = self.parameters.read().map_err(|_| Error::lock("Failed to acquire read lock for parameters"))?;
        params.to_model_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameters() {
        let mut params = Parameters::new("1.0.0");
        params.set("name", ParameterValue::String("value".to_string()));
        params.set("age", ParameterValue::Integer(30));
        
        assert_eq!(params.get_string("name"), Some("value"));
        assert_eq!(params.get_integer("age"), Some(30));
        assert_eq!(params.get_float("height"), None);
    }

    #[test]
    fn test_parameter_manager() {
        let mut manager = ParameterManager::new("1.0.0");
        manager.set("name", ParameterValue::String("value".to_string()));
        
        assert_eq!(
            manager.get("name"), 
            Some(ParameterValue::String("value".to_string()))
        );
    }
} 