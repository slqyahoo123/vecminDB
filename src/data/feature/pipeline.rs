// Feature Pipeline Module
// 特征处理管道模块

use std::sync::Arc;
use std::collections::HashMap;
use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::data::feature::vector::{Vector, VectorBatch};

/// 管道处理阶段trait
pub trait PipelineStage: Send + Sync + Debug {
    /// 获取处理阶段名称
    fn name(&self) -> &str;
    
    /// 获取处理阶段描述
    fn description(&self) -> &str;
    
    /// 获取处理阶段参数
    fn parameters(&self) -> &HashMap<String, String>;
    
    /// 处理单个特征向量
    fn process_vector(&mut self, vector: &mut Vector) -> Result<()>;
    
    /// 处理特征向量批次
    fn process_batch(&mut self, batch: &mut VectorBatch) -> Result<()> {
        for vector in &mut batch.vectors {
            self.process_vector(vector)?;
        }
        Ok(())
    }
    
    /// 初始化处理阶段
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// 重置处理阶段状态
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// 特征处理管道配置
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// 配置名称
    pub name: String,
    
    /// 配置参数
    pub parameters: HashMap<String, String>,
}

impl PipelineConfig {
    /// 创建新的管道配置
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parameters: HashMap::new(),
        }
    }
    
    /// 添加配置参数
    pub fn with_parameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
    
    /// 获取布尔参数值
    pub fn get_bool(&self, key: &str, default: bool) -> bool {
        match self.parameters.get(key) {
            Some(value) => value.to_lowercase() == "true" || value == "1",
            None => default,
        }
    }
    
    /// 获取整数参数值
    pub fn get_int(&self, key: &str, default: i64) -> i64 {
        match self.parameters.get(key) {
            Some(value) => value.parse().unwrap_or(default),
            None => default,
        }
    }
    
    /// 获取浮点数参数值
    pub fn get_float(&self, key: &str, default: f64) -> f64 {
        match self.parameters.get(key) {
            Some(value) => value.parse().unwrap_or(default),
            None => default,
        }
    }
    
    /// 获取字符串参数值
    pub fn get_string(&self, key: &str, default: &str) -> String {
        match self.parameters.get(key) {
            Some(value) => value.clone(),
            None => default.to_string(),
        }
    }
}

/// 特征处理管道
#[derive(Debug)]
pub struct FeaturePipeline {
    /// 管道名称
    pub name: String,
    
    /// 管道描述
    pub description: String,
    
    /// 处理阶段
    stages: Vec<Box<dyn PipelineStage>>,
    
    /// 管道是否已初始化
    initialized: bool,
}

impl FeaturePipeline {
    /// 创建新的特征处理管道
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            stages: Vec::new(),
            initialized: false,
        }
    }
    
    /// 添加处理阶段
    pub fn add_stage(&mut self, stage: Box<dyn PipelineStage>) -> &mut Self {
        self.stages.push(stage);
        self.initialized = false;
        self
    }
    
    /// 初始化管道
    pub fn initialize(&mut self) -> Result<&mut Self> {
        for stage in &mut self.stages {
            stage.initialize()?;
        }
        self.initialized = true;
        Ok(self)
    }
    
    /// 处理特征向量
    pub fn process_vector(&mut self, vector: &mut Vector) -> Result<&mut Self> {
        if !self.initialized {
            self.initialize()?;
        }
        
        for stage in &mut self.stages {
            stage.process_vector(vector)?;
        }
        
        Ok(self)
    }
    
    /// 处理特征向量批次
    pub fn process_batch(&mut self, batch: &mut VectorBatch) -> Result<&mut Self> {
        if !self.initialized {
            self.initialize()?;
        }
        
        for stage in &mut self.stages {
            stage.process_batch(batch)?;
        }
        
        Ok(self)
    }
    
    /// 重置管道状态
    pub fn reset(&mut self) -> Result<&mut Self> {
        for stage in &mut self.stages {
            stage.reset()?;
        }
        Ok(self)
    }
    
    /// 获取处理阶段数量
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
    
    /// 检查管道是否为空
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }
    
    /// 获取管道处理阶段名称列表
    pub fn stage_names(&self) -> Vec<String> {
        self.stages.iter()
            .map(|stage| stage.name().to_string())
            .collect()
    }
}

/// 归一化处理阶段
#[derive(Debug)]
pub struct NormalizationStage {
    /// 归一化方法
    method: String,
    
    /// 处理阶段参数
    parameters: HashMap<String, String>,
}

impl NormalizationStage {
    /// 创建新的归一化处理阶段
    pub fn new(method: impl Into<String>) -> Self {
        let method = method.into();
        let mut parameters = HashMap::new();
        parameters.insert("method".to_string(), method.clone());
        
        Self {
            method,
            parameters,
        }
    }
    
    /// L1归一化（曼哈顿范数）
    fn l1_normalize(&self, vector: &mut Vector) -> Result<()> {
        let norm = vector.l1_norm();
        if norm > 0.0 && !norm.is_nan() {
            for value in &mut vector.values {
                *value /= norm;
            }
        }
        Ok(())
    }
    
    /// L2归一化（欧几里得范数）
    fn l2_normalize(&self, vector: &mut Vector) -> Result<()> {
        let norm = vector.l2_norm();
        if norm > 0.0 && !norm.is_nan() {
            for value in &mut vector.values {
                *value /= norm;
            }
        }
        Ok(())
    }
    
    /// MinMax归一化（缩放到[0,1]区间）
    fn minmax_normalize(&self, vector: &mut Vector) -> Result<()> {
        if vector.values.is_empty() {
            return Ok(());
        }
        
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        
        for &value in &vector.values {
            if value < min_val {
                min_val = value;
            }
            if value > max_val {
                max_val = value;
            }
        }
        
        let range = max_val - min_val;
        if range > 0.0 && !range.is_nan() {
            for value in &mut vector.values {
                *value = (*value - min_val) / range;
            }
        }
        
        Ok(())
    }
    
    /// Z-Score归一化（标准化）
    fn zscore_normalize(&self, vector: &mut Vector) -> Result<()> {
        if vector.values.is_empty() {
            return Ok(());
        }
        
        // 计算均值
        let mean = vector.values.iter().sum::<f32>() / vector.values.len() as f32;
        
        // 计算标准差
        let variance = vector.values.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>() / vector.values.len() as f32;
            
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 && !std_dev.is_nan() {
            for value in &mut vector.values {
                *value = (*value - mean) / std_dev;
            }
        }
        
        Ok(())
    }
}

impl PipelineStage for NormalizationStage {
    fn name(&self) -> &str {
        "normalization"
    }
    
    fn description(&self) -> &str {
        "对特征向量进行归一化处理"
    }
    
    fn parameters(&self) -> &HashMap<String, String> {
        &self.parameters
    }
    
    fn process_vector(&mut self, vector: &mut Vector) -> Result<()> {
        match self.method.as_str() {
            "l1" => self.l1_normalize(vector),
            "l2" => self.l2_normalize(vector),
            "minmax" => self.minmax_normalize(vector),
            "zscore" => self.zscore_normalize(vector),
            _ => Err(format!("不支持的归一化方法: {}", self.method).into()),
        }
    }
}

/// 特征选择处理阶段
#[derive(Debug)]
pub struct FeatureSelectionStage {
    /// 选择的维度集合
    dimensions: Vec<usize>,
    
    /// 处理阶段参数
    parameters: HashMap<String, String>,
}

impl FeatureSelectionStage {
    /// 创建新的特征选择处理阶段
    pub fn new(dimensions: Vec<usize>) -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("dimensions".to_string(), 
            dimensions.iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(","));
        
        Self {
            dimensions,
            parameters,
        }
    }
    
    /// 从范围创建
    pub fn from_range(start: usize, end: usize) -> Self {
        let dimensions = (start..end).collect();
        Self::new(dimensions)
    }
    
    /// 从索引列表创建
    pub fn from_indices(indices: &[usize]) -> Self {
        Self::new(indices.to_vec())
    }
}

impl PipelineStage for FeatureSelectionStage {
    fn name(&self) -> &str {
        "feature_selection"
    }
    
    fn description(&self) -> &str {
        "选择特定维度的特征"
    }
    
    fn parameters(&self) -> &HashMap<String, String> {
        &self.parameters
    }
    
    fn process_vector(&mut self, vector: &mut Vector) -> Result<()> {
        if self.dimensions.is_empty() {
            return Ok(());
        }
        
        // 验证维度是否有效
        for &dim in &self.dimensions {
            if dim >= vector.dimension {
                return Err(format!(
                    "维度索引越界: {} >= {}", 
                    dim, 
                    vector.dimension
                ).into());
            }
        }
        
        // 选择指定维度的特征
        let mut selected_values = Vec::with_capacity(self.dimensions.len());
        for &dim in &self.dimensions {
            selected_values.push(vector.values[dim]);
        }
        
        // 更新向量
        vector.values = selected_values;
        vector.dimension = self.dimensions.len();
        
        Ok(())
    }
} 
} 