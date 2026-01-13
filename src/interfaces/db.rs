use crate::Result;
use serde_json::Value;

/// 查询请求类型
#[derive(Debug, Clone)]
pub enum QueryRequest {
    /// 根据ID获取模型
    GetModelById {
        model_id: String,
    },
    /// 根据ID获取数据Schema
    GetDataSchemaById {
        schema_id: String,
    },
    /// 根据条件查询数据
    QueryData {
        query: String,
        params: Option<Value>,
    },
    /// 获取算法信息
    GetAlgorithmInfo {
        algorithm_id: String,
    },
    /// 自定义查询
    Custom {
        query_type: String,
        parameters: Value,
    },
}

/// 查询结果数据
#[derive(Debug, Clone)]
pub enum QueryResultData {
    /// 模型数据
    Model(Option<crate::model::Model>),
    /// Schema数据
    Schema(Option<crate::data::DataSchema>),
    /// 数据批次
    DataBatch(Option<crate::data::DataBatch>),
    /// 算法信息
    Algorithm(Option<Value>),
    /// 字符串结果
    String(String),
    /// JSON结果
    Json(Value),
    /// 布尔结果
    Boolean(bool),
    /// 数字结果
    Number(f64),
    /// 空结果
    Empty,
}

/// 查询响应
#[derive(Debug, Clone)]
pub struct QueryResponse {
    /// 结果数据
    pub data: QueryResultData,
}

/// 数据库接口
/// 
/// 定义与数据库交互的抽象接口，供storage模块使用
pub trait DatabaseInterface: Send + Sync {
    /// 执行查询
    fn execute_query(&self, request: QueryRequest) -> Result<QueryResponse>;
    
    /// 事件通知
    fn notify_event(&self, event_type: &str, data: Value) -> Result<()>;
    
    /// 验证查询是否有效
    fn validate_query(&self, query: &str) -> Result<bool>;
    
    /// 获取数据库状态
    fn get_status(&self) -> Result<Value>;
    
    /// 获取数据库统计信息
    fn get_statistics(&self) -> Result<Value>;
    
    /// 数据导入
    fn import_data(&self, source: &str, format: &str, options: Option<Value>) -> Result<String>;
    
    /// 数据导出
    fn export_data(&self, target: &str, format: &str, query: Option<&str>, options: Option<Value>) -> Result<String>;
    
    /// 执行索引操作
    fn perform_index_operation(&self, operation: &str, index_name: &str, options: Option<Value>) -> Result<()>;
    
    /// 执行维护操作
    fn perform_maintenance(&self, operation: &str, options: Option<Value>) -> Result<()>;
} 