/*!
数据导出模块

提供数据批次、数据集等核心数据类型的统一导出接口。
该模块用于简化数据类型的使用，避免复杂的模块路径引用。
*/

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// 数据批次
/// 
/// 用于机器学习训练的数据批次结构，包含数据、标签和元信息。
/// 这是训练过程中最常用的数据结构。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBatch {
    /// 批次ID
    pub id: String,
    /// 批次数据（每个样本为一个特征向量）
    pub data: Vec<Vec<f32>>,
    /// 批次标签
    pub labels: Vec<f32>,
    /// 批次大小
    pub batch_size: usize,
    /// 批次索引
    pub batch_index: usize,
    /// 特征维度
    pub feature_dim: usize,
    /// 序列长度（用于序列数据）
    pub sequence_length: Option<usize>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
}

impl DataBatch {
    /// 创建新的数据批次
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            data: Vec::new(),
            labels: Vec::new(),
            batch_size: 0,
            batch_index: 0,
            feature_dim: 0,
            sequence_length: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// 创建带ID的数据批次
    pub fn with_id(id: String) -> Self {
        Self {
            id,
            data: Vec::new(),
            labels: Vec::new(),
            batch_size: 0,
            batch_index: 0,
            feature_dim: 0,
            sequence_length: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// 创建带数据的批次
    pub fn with_data(data: Vec<Vec<f32>>, labels: Vec<f32>) -> Self {
        let batch_size = data.len();
        let feature_dim = if !data.is_empty() { data[0].len() } else { 0 };
        
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            data,
            labels,
            batch_size,
            batch_index: 0,
            feature_dim,
            sequence_length: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    /// 设置数据
    pub fn set_data(&mut self, data: Vec<Vec<f32>>) {
        self.batch_size = data.len();
        self.feature_dim = if !data.is_empty() { data[0].len() } else { 0 };
        self.data = data;
    }

    /// 设置标签
    pub fn set_labels(&mut self, labels: Vec<f32>) {
        self.labels = labels;
    }

    /// 设置批次大小
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    /// 设置批次索引
    pub fn set_batch_index(&mut self, batch_index: usize) {
        self.batch_index = batch_index;
    }

    /// 获取数据
    pub fn get_data(&self) -> &Vec<Vec<f32>> {
        &self.data
    }

    /// 获取标签
    pub fn get_labels(&self) -> &Vec<f32> {
        &self.labels
    }

    /// 检查批次是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 获取样本数量
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 添加样本
    pub fn add_sample(&mut self, sample: Vec<f32>, label: f32) {
        // 检查特征维度一致性
        if !self.data.is_empty() && sample.len() != self.feature_dim {
            log::warn!("样本特征维度不一致: 期望 {}, 实际 {}", self.feature_dim, sample.len());
        }
        
        self.data.push(sample);
        self.labels.push(label);
        self.batch_size = self.data.len();
        
        if self.feature_dim == 0 && !self.data.is_empty() {
            self.feature_dim = self.data[0].len();
        }
    }

    /// 清空批次
    pub fn clear(&mut self) {
        self.data.clear();
        self.labels.clear();
        self.batch_size = 0;
        self.feature_dim = 0;
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// 分割批次
    pub fn split(&self, chunk_size: usize) -> Vec<DataBatch> {
        let mut batches = Vec::new();
        
        for (i, chunk) in self.data.chunks(chunk_size).enumerate() {
            let label_chunk = if i * chunk_size < self.labels.len() {
                let end = std::cmp::min((i + 1) * chunk_size, self.labels.len());
                self.labels[i * chunk_size..end].to_vec()
            } else {
                Vec::new()
            };

            let mut batch = DataBatch::with_data(chunk.to_vec(), label_chunk);
            batch.batch_index = i;
            batch.metadata = self.metadata.clone();
            batches.push(batch);
        }
        
        batches
    }

    /// 合并批次
    pub fn merge(batches: Vec<DataBatch>) -> DataBatch {
        let mut merged_data = Vec::new();
        let mut merged_labels = Vec::new();
        let mut metadata = HashMap::new();

        for batch in batches {
            merged_data.extend(batch.data);
            merged_labels.extend(batch.labels);
            
            // 合并元数据
            for (key, value) in batch.metadata {
                metadata.insert(key, value);
            }
        }

        let mut result = DataBatch::with_data(merged_data, merged_labels);
        result.metadata = metadata;
        result
    }

    /// 转换为张量格式
    pub fn to_tensor_format(&self) -> (Vec<f32>, Vec<usize>) {
        let mut flattened = Vec::new();
        for sample in &self.data {
            flattened.extend(sample);
        }
        let shape = vec![self.batch_size, self.feature_dim];
        (flattened, shape)
    }

    /// 从张量格式创建
    pub fn from_tensor_format(data: Vec<f32>, shape: Vec<usize>, labels: Vec<f32>) -> Self {
        if shape.len() != 2 {
            log::error!("形状必须是2D: [batch_size, feature_dim]");
            return Self::new();
        }

        let batch_size = shape[0];
        let feature_dim = shape[1];
        let mut batch_data = Vec::new();

        for i in 0..batch_size {
            let start = i * feature_dim;
            let end = start + feature_dim;
            if end <= data.len() {
                batch_data.push(data[start..end].to_vec());
            }
        }

        Self::with_data(batch_data, labels)
    }

    /// 验证数据一致性
    pub fn validate(&self) -> Result<(), String> {
        // 检查数据和标签数量一致性
        if !self.labels.is_empty() && self.data.len() != self.labels.len() {
            return Err(format!("数据和标签数量不匹配: {} vs {}", self.data.len(), self.labels.len()));
        }

        // 检查特征维度一致性
        for (i, sample) in self.data.iter().enumerate() {
            if sample.len() != self.feature_dim {
                return Err(format!("样本 {} 特征维度不匹配: {} vs {}", i, sample.len(), self.feature_dim));
            }
        }

        // 检查批次大小一致性
        if self.batch_size != self.data.len() {
            return Err(format!("批次大小不匹配: {} vs {}", self.batch_size, self.data.len()));
        }

        Ok(())
    }

    /// 从ManagerDataset创建DataBatch
    pub fn from_manager_dataset(dataset: crate::data::manager::ManagerDataset) -> Result<Self, crate::error::Error> {
        let mut batch = Self::new();
        
        // 设置基本信息
        batch.id = dataset.id.clone();
        batch.batch_size = dataset.size;
        batch.batch_index = 0;
        
        // 设置元数据
        batch.add_metadata("name".to_string(), dataset.name);
        batch.add_metadata("format".to_string(), dataset.format);
        batch.add_metadata("dataset_type".to_string(), format!("{:?}", dataset.dataset_type));
        batch.add_metadata("created_at".to_string(), dataset.created_at.to_string());
        batch.add_metadata("updated_at".to_string(), dataset.updated_at.to_string());
        
        // 合并其他元数据
        for (key, value) in dataset.metadata {
            batch.add_metadata(key, value);
        }
        
        // 初始化空数据（实际数据加载将在后续步骤中进行）
        batch.data = Vec::new();
        batch.labels = Vec::new();
        batch.feature_dim = 0;
        
        Ok(batch)
    }
}

impl Default for DataBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// 数据样本
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSample {
    /// 样本ID
    pub id: String,
    /// 特征数据
    pub features: Vec<f32>,
    /// 标签
    pub label: Option<f32>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl DataSample {
    /// 创建新样本
    pub fn new(features: Vec<f32>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            features,
            label: None,
            metadata: HashMap::new(),
        }
    }

    /// 创建带标签的样本
    pub fn with_label(features: Vec<f32>, label: f32) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            features,
            label: Some(label),
            metadata: HashMap::new(),
        }
    }
}

/// 数据迭代器
pub struct DataBatchIterator {
    batches: Vec<DataBatch>,
    current_index: usize,
}

impl DataBatchIterator {
    /// 创建新的迭代器
    pub fn new(batches: Vec<DataBatch>) -> Self {
        Self {
            batches,
            current_index: 0,
        }
    }

    /// 重置迭代器
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// 获取总批次数
    pub fn len(&self) -> usize {
        self.batches.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
}

impl Iterator for DataBatchIterator {
    type Item = DataBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.batches.len() {
            let batch = self.batches[self.current_index].clone();
            self.current_index += 1;
            Some(batch)
        } else {
            None
        }
    }
}

/// 数据统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics {
    /// 总样本数
    pub total_samples: usize,
    /// 特征维度
    pub feature_dimension: usize,
    /// 类别数（分类任务）
    pub num_classes: Option<usize>,
    /// 特征统计
    pub feature_stats: Vec<FeatureStatistics>,
    /// 标签分布
    pub label_distribution: HashMap<String, usize>,
}

/// 特征统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStatistics {
    /// 特征索引
    pub index: usize,
    /// 最小值
    pub min: f32,
    /// 最大值
    pub max: f32,
    /// 平均值
    pub mean: f32,
    /// 标准差
    pub std: f32,
    /// 缺失值数量
    pub missing_count: usize,
} 