use crate::Result;
use crate::compat::{ModelArchitecture, ModelParameters, TrainingMetrics, Model};
use crate::storage::engine::implementation::StorageOptions;
use crate::storage::models::{ModelInfo, ModelMetrics};
use crate::interfaces::storage::{StorageTransaction, IsolationLevel};
use std::collections::HashMap;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use crate::core::{InferenceResultDetail, InferenceResult};

/// 存储服务接口
/// 
/// 定义存储服务的核心功能，包括模型参数、架构、训练状态等的存储和获取
pub trait StorageService: Send + Sync {
    /// 保存模型参数
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `params`: 模型参数对象
    fn save_model_parameters(&self, model_id: &str, params: &ModelParameters) -> Result<()>;
    
    /// 获取模型参数
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型参数，如果不存在则返回None
    fn get_model_parameters(&self, model_id: &str) -> Result<Option<ModelParameters>>;
    
    /// 保存模型架构
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `arch`: 模型架构对象
    fn save_model_architecture(&self, model_id: &str, arch: &ModelArchitecture) -> Result<()>;
    
    /// 获取模型架构
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型架构，如果不存在则返回None
    fn get_model_architecture(&self, model_id: &str) -> Result<Option<ModelArchitecture>>;
    
    /// 保存推理结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `result`: 推理结果对象
    fn save_inference_result(&self, model_id: &str, result: &crate::core::InferenceResult) -> Result<()>;
    
    /// 获取推理结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回推理结果，如果不存在则返回None
    fn get_inference_result(&self, model_id: &str) -> Result<Option<crate::core::InferenceResult>>;
    
    /// 存储键值对
    /// 
    /// # 参数
    /// - `key`: 存储键
    /// - `value`: 存储值
    async fn store(&self, key: &str, value: &[u8]) -> Result<()>;
    
    /// 检索键值对
    /// 
    /// # 参数
    /// - `key`: 存储键
    /// 
    /// # 返回值
    /// 返回存储的值，如果不存在则返回None
    async fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// 删除键值对
    /// 
    /// # 参数
    /// - `key`: 存储键
    async fn delete(&self, key: &str) -> Result<()>;
    
    /// 检查键是否存在
    /// 
    /// # 参数
    /// - `key`: 存储键
    /// 
    /// # 返回值
    /// 返回键是否存在
    async fn exists(&self, key: &str) -> Result<bool>;
    
    /// 列出所有键
    /// 
    /// # 参数
    /// - `prefix`: 键前缀
    /// 
    /// # 返回值
    /// 返回匹配前缀的所有键
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>>;
    
    /// 批量存储键值对
    /// 
    /// # 参数
    /// - `items`: 键值对列表
    async fn batch_store(&self, items: &[(String, Vec<u8>)]) -> Result<()>;
    
    /// 批量检索键值对
    /// 
    /// # 参数
    /// - `keys`: 键列表
    /// 
    /// # 返回值
    /// 返回键值对映射
    async fn batch_retrieve(&self, keys: &[String]) -> Result<HashMap<String, Option<Vec<u8>>>>;
    
    /// 批量删除键值对
    /// 
    /// # 参数
    /// - `keys`: 键列表
    async fn batch_delete(&self, keys: &[String]) -> Result<()>;
    
    /// 创建事务
    /// 
    /// # 返回值
    /// 返回事务对象
    async fn transaction(&self) -> Result<Box<dyn StorageTransaction + Send + Sync>>;
    
    /// 创建带隔离级别的事务
    /// 
    /// # 参数
    /// - `isolation_level`: 隔离级别
    /// 
    /// # 返回值
    /// 返回事务对象
    async fn transaction_with_isolation(&self, isolation_level: IsolationLevel) -> Result<Box<dyn StorageTransaction + Send + Sync>>;
    
    /// 获取数据集大小
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 返回数据集大小
    async fn get_dataset_size(&self, dataset_id: &str) -> Result<usize>;
    
    /// 获取数据集块
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// - `offset`: 偏移量
    /// - `limit`: 限制数量
    /// 
    /// # 返回值
    /// 返回数据集块
    async fn get_dataset_chunk(&self, dataset_id: &str, offset: usize, limit: usize) -> Result<Vec<u8>>;
    
    /// 列出推理结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回推理结果列表
    fn list_inference_results(&self, model_id: &str) -> Result<Vec<InferenceResult>>;
    
    /// 获取特定推理结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `inference_id`: 推理唯一标识符
    /// 
    /// # 返回值
    /// 返回推理结果，如果不存在则返回None
    fn get_specific_inference_result(&self, model_id: &str, inference_id: &str) -> Result<Option<InferenceResult>>;
    
    /// 保存详细推理结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `inference_id`: 推理唯一标识符
    /// - `result`: 推理结果对象
    /// - `detail`: 推理结果详情对象
    /// - `processing_time`: 处理时间
    fn save_detailed_inference_result(
        &self,
        model_id: &str,
        inference_id: &str,
        result: &InferenceResult,
        detail: &InferenceResultDetail,
        processing_time: u64,
    ) -> Result<()>;
    
    /// 保存模型信息
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `info`: 模型信息对象
    fn save_model_info(&self, model_id: &str, info: &ModelInfo) -> Result<()>;
    
    /// 获取模型信息
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型信息，如果不存在则返回None
    fn get_model_info(&self, model_id: &str) -> Result<Option<ModelInfo>>;
    
    /// 保存模型指标
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `metrics`: 模型指标对象
    fn save_model_metrics(&self, model_id: &str, metrics: &ModelMetrics) -> Result<()>;
    
    /// 获取模型指标
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型指标，如果不存在则返回None
    fn get_model_metrics(&self, model_id: &str) -> Result<Option<ModelMetrics>>;
    
    /// 获取模型
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型，如果不存在则返回None
    fn get_model(&self, model_id: &str) -> Result<Option<Model>>;
    
    /// 保存模型
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `model`: 模型对象
    fn save_model(&self, model_id: &str, model: &Model) -> Result<()>;
    
    /// 检查模型是否存在
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型是否存在
    fn model_exists(&self, model_id: &str) -> Result<bool>;
    
    /// 检查是否有模型
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回是否有模型
    fn has_model(&self, model_id: &str) -> Result<bool>;
    
    /// 获取数据集
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 返回数据集，如果不存在则返回None
    fn get_dataset(&self, dataset_id: &str) -> Result<Option<crate::data::DataBatch>>;
}

/// 存储引擎接口
/// 
/// 定义存储引擎的基础操作，包括键值存储、数据集管理、模型管理等功能
pub trait StorageEngine: Send + Sync {
    /// 获取键值对应的值
    /// 
    /// # 参数
    /// - `key`: 键的字节数组
    /// 
    /// # 返回值
    /// 返回对应的值，如果不存在则返回None
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    
    /// 设置键值对
    /// 
    /// # 参数
    /// - `key`: 键的字节数组
    /// - `value`: 值的字节数组
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()>;
    
    /// 删除键值对
    /// 
    /// # 参数
    /// - `key`: 键的字节数组
    fn delete(&self, key: &[u8]) -> Result<()>;
    
    /// 获取前缀扫描迭代器
    /// 
    /// # 参数
    /// - `prefix`: 前缀字节数组
    /// 
    /// # 返回值
    /// 返回迭代器，用于遍历匹配前缀的键值对
    fn scan_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Result<(Vec<u8>, Vec<u8>)>> + '_>;
    
    /// 设置存储引擎选项
    /// 
    /// # 参数
    /// - `options`: 存储选项配置
    fn set_options(&mut self, options: &StorageOptions) -> Result<()>;
    
    /// 检查数据集是否存在
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集是否存在
    fn dataset_exists(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + '_>>;
    
    /// 获取数据集数据
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集的原始字节数据
    fn get_dataset_data(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + '_>>;
    
    /// 获取数据集元数据
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集的元数据JSON对象
    fn get_dataset_metadata(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send + '_>>;
    
    /// 保存数据集数据
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// - `data`: 数据集的原始字节数据
    fn save_dataset_data(&self, dataset_id: &str, data: &[u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
    
    /// 保存数据集元数据
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// - `metadata`: 数据集的元数据JSON对象
    fn save_dataset_metadata(&self, dataset_id: &str, metadata: &serde_json::Value) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
    
    /// 获取数据集模式
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集的模式定义JSON对象
    fn get_dataset_schema(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send + '_>>;
    
    /// 保存数据集模式
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// - `schema`: 数据集的模式定义JSON对象
    fn save_dataset_schema(&self, dataset_id: &str, schema: &serde_json::Value) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
    
    /// 删除完整数据集
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    fn delete_dataset_complete(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
    
    /// 列出所有数据集
    /// 
    /// # 返回值
    /// 异步返回所有数据集的ID列表
    fn list_datasets(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + '_>>;
    
    /// 存储数据
    /// 
    /// # 参数
    /// - `key`: 存储键
    /// - `data`: 数据字节数组
    fn store(&self, key: &str, data: &[u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
    
    /// 根据过滤条件列出模型
    /// 
    /// # 参数
    /// - `filters`: 过滤条件映射
    /// - `limit`: 限制返回数量
    /// - `offset`: 偏移量
    /// 
    /// # 返回值
    /// 异步返回符合条件的模型列表
    fn list_models_with_filters(
        &self,
        filters: std::collections::HashMap<String, String>,
        limit: usize,
        offset: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<crate::model::Model>>> + Send + '_>>;
    
    /// 统计模型数量
    /// 
    /// # 返回值
    /// 异步返回模型总数
    fn count_models(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>>;
    
    /// 获取数据批次
    /// 
    /// # 参数
    /// - `batch_id`: 批次唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据批次对象
    fn get_data_batch(&self, batch_id: &str) -> Pin<Box<dyn Future<Output = Result<crate::data::DataBatch>> + Send + '_>>;
    
    /// 获取批次数据
    /// 
    /// # 参数
    /// - `batch_id`: 批次唯一标识符
    /// 
    /// # 返回值
    /// 异步返回批次数据对象
    fn get_batch_data(&self, batch_id: &str) -> Pin<Box<dyn Future<Output = Result<crate::data::DataBatch>> + Send + '_>>;
    
    /// 保存处理后的批次
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `batch`: 处理后的数据批次
    fn save_processed_batch(
        &self,
        model_id: &str,
        batch: &crate::data::ProcessedBatch,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
    
    /// 获取训练指标历史
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 异步返回训练指标历史列表
    fn get_training_metrics_history(
        &self,
        model_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<TrainingMetrics>>> + Send + '_>>;
    
    /// 记录训练指标
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `metrics`: 训练指标对象
    fn record_training_metrics(
        &self,
        model_id: &str,
        metrics: &TrainingMetrics,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
    
    /// 查询数据集
    /// 
    /// # 参数
    /// - `name`: 数据集名称
    /// - `limit`: 限制返回数量
    /// - `offset`: 偏移量
    /// 
    /// # 返回值
    /// 异步返回查询结果JSON数组
    fn query_dataset(
        &self,
        name: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<serde_json::Value>>> + Send + '_>>;
    
    /// 检查键是否存在
    /// 
    /// # 参数
    /// - `key`: 键的字节数组
    /// 
    /// # 返回值
    /// 返回键是否存在
    fn exists(&self, key: &[u8]) -> Result<bool>;
    
    /// 获取数据集大小（字节数）
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集的大小（字节数）
    fn get_dataset_size(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>>;
    
    /// 获取数据集指定区间的字节数据
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// - `start`: 起始偏移量
    /// - `end`: 结束偏移量
    /// 
    /// # 返回值
    /// 异步返回指定区间的字节数据
    fn get_dataset_chunk(&self, dataset_id: &str, start: usize, end: usize) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + '_>>;
    
    /// 关闭存储引擎
    /// 
    /// # 返回值
    /// 关闭操作的结果
    fn close(&self) -> Result<()>;
    
    /// 保存处理后的数据
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `data`: 处理后的数据
    fn save_processed_data(
        &self,
        model_id: &str,
        data: &[Vec<f32>],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>>;
}

/// 数据集存储接口
pub trait DatasetStorageInterface: Send + Sync {
    /// 检查数据集是否存在
    fn dataset_exists(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<bool>> + Send;
    
    /// 获取数据集数据
    fn get_dataset_data(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Vec<u8>>> + Send;
    
    /// 获取数据集元数据
    fn get_dataset_metadata(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;
    
    /// 保存数据集数据
    fn save_dataset_data(&self, dataset_id: &str, data: &[u8]) -> impl std::future::Future<Output = Result<()>> + Send;
    
    /// 保存数据集元数据
    fn save_dataset_metadata(&self, dataset_id: &str, metadata: &Value) -> impl std::future::Future<Output = Result<()>> + Send;
    
    /// 获取数据集模式
    fn get_dataset_schema(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;
    
    /// 保存数据集模式
    fn save_dataset_schema(&self, dataset_id: &str, schema: &Value) -> impl std::future::Future<Output = Result<()>> + Send;
    
    /// 删除数据集
    fn delete_dataset(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<()>> + Send;
    
    /// 列出数据集
    fn list_datasets(&self) -> impl std::future::Future<Output = Result<Vec<String>>> + Send;
    
    /// 获取数据集信息
    fn get_dataset_info(&self, id: &str) -> impl std::future::Future<Output = Result<Option<Value>>> + Send;
    
    /// 保存数据集信息
    fn save_dataset_info(&self, dataset: &Value) -> impl std::future::Future<Output = Result<()>> + Send;
    
    /// 查询数据集
    fn query_dataset(
        &self,
        name: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Vec<Value>>> + Send;
    
    /// 获取数据集统计信息
    fn get_dataset_stats(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;
    
    /// 验证数据集
    fn validate_dataset(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Value>> + Send;
    
    /// 获取数据集大小（字节数）
    fn get_dataset_size(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<usize>> + Send;
    
    /// 获取数据集指定区间的字节数据
    fn get_dataset_chunk(&self, dataset_id: &str, start: usize, end: usize) -> impl std::future::Future<Output = Result<Vec<u8>>> + Send;
}

/// 存储监控服务接口
pub trait StorageMonitoringInterface: Send + Sync {
    /// 获取系统统计信息
    fn get_stats(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + '_>>;
    
    /// 获取计数器值
    fn get_counter(&self, counter_name: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64>> + Send + '_>>;
    
    /// 递增计数器
    fn increment_counter(&self, counter_name: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<u64>> + Send + '_>>;
    
    /// 统计模型数量
    fn count_models(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize>> + Send + '_>>;
    
    /// 按类型统计模型数量
    fn count_models_by_type(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<HashMap<String, usize>>> + Send + '_>>;
    
    /// 获取最近的模型
    fn get_recent_models(&self, limit: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Value>>> + Send + '_>>;
    
    /// 统计任务数量
    fn count_tasks(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize>> + Send + '_>>;
    
    /// 按状态统计任务数量
    fn count_tasks_by_status(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<HashMap<String, usize>>> + Send + '_>>;
    
    /// 获取最近的任务
    fn get_recent_tasks(&self, limit: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Value>>> + Send + '_>>;
    
    /// 获取日志
    fn get_logs(&self, level: &str, limit: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Value>>> + Send + '_>>;
    
    /// 统计活跃任务数量
    fn count_active_tasks(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize>> + Send + '_>>;
    
    /// 获取API统计信息
    fn get_api_stats(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + '_>>;
    
    /// 检查系统健康状态
    fn check_health(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send + '_>>;
}