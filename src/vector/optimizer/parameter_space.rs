// 参数空间模块
// 定义参数取值范围和可能的参数组合

use crate::vector::index::IndexType;
use rand::Rng;
use std::collections::HashMap;
use std::fmt::{self, Display};
use serde::{Serialize, Deserialize};

/// 优化维度定义
/// 表示一个可优化的参数及其范围
#[derive(Debug, Clone)]
pub struct OptimizationDimension {
    /// 参数名称
    pub name: String,
    /// 参数范围
    pub range: ParameterRange,
    /// 参数说明
    pub description: Option<String>,
    /// 是否为连续参数
    pub is_continuous: bool,
}

/// 参数类型
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterType {
    /// 整数类型，带最小值和最大值
    Integer(i64, i64),
    /// 浮点类型，带最小值和最大值
    Float(f64, f64),
    /// 分类类型，带可能的值列表
    Categorical(Vec<ParameterValue>),
}

/// 参数值
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    /// 整数值
    Integer(i64),
    /// 浮点值
    Float(f64),
    /// 分类值 (存储为字符串)
    Categorical(String),
}

impl Display for ParameterValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(v) => write!(f, "{}", v),
            Self::Float(v) => write!(f, "{}", v),
            Self::Categorical(v) => write!(f, "{}", v),
        }
    }
}

/// 参数信息
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// 参数名称
    name: String,
    /// 参数类型
    parameter_type: ParameterType,
    /// 参数描述
    description: Option<String>,
}

/// 参数定义
#[derive(Debug, Clone)]
pub struct Parameter {
    /// 参数信息
    info: ParameterInfo,
    /// 默认值
    default_value: Option<ParameterValue>,
}

impl Parameter {
    /// 创建新参数
    pub fn new(name: String, parameter_type: ParameterType) -> Self {
        Self {
            info: ParameterInfo {
                name,
                parameter_type,
                description: None,
            },
            default_value: None,
        }
    }
    
    /// 设置描述
    pub fn with_description(mut self, description: String) -> Self {
        self.info.description = Some(description);
        self
    }
    
    /// 设置默认值
    pub fn with_default(mut self, value: ParameterValue) -> Self {
        self.default_value = Some(value);
        self
    }
    
    /// 获取参数名称
    pub fn name(&self) -> &str {
        &self.info.name
    }
    
    /// 获取参数类型
    pub fn parameter_type(&self) -> &ParameterType {
        &self.info.parameter_type
    }
    
    /// 获取参数描述
    pub fn description(&self) -> Option<&str> {
        self.info.description.as_deref()
    }
    
    /// 获取默认值
    pub fn default_value(&self) -> Option<&ParameterValue> {
        self.default_value.as_ref()
    }
}

/// 参数范围
#[derive(Debug, Clone)]
pub struct ParameterRange {
    /// 参数名称
    pub name: String,
    /// 最小值
    pub min: f64,
    /// 最大值
    pub max: f64,
    /// 步长（用于网格搜索）
    pub step: f64,
    /// 是否为整数
    pub is_integer: bool,
    /// 是否为对数尺度
    pub is_log_scale: bool,
}

/// 参数空间
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    /// 参数范围列表
    pub ranges: Vec<ParameterRange>,
    /// 参数列表
    parameters: Vec<Parameter>,
}

impl ParameterSpace {
    /// 创建HNSW索引的参数空间
    pub fn hnsw() -> Self {
        let ranges = vec![
            ParameterRange {
                name: "M".to_string(),
                min: 4.0,
                max: 64.0,
                step: 4.0,
                is_integer: true,
                is_log_scale: false,
            },
            ParameterRange {
                name: "ef_construction".to_string(),
                min: 40.0,
                max: 400.0,
                step: 40.0,
                is_integer: true,
                is_log_scale: false,
            },
            ParameterRange {
                name: "ef".to_string(),
                min: 10.0,
                max: 200.0,
                step: 10.0,
                is_integer: true,
                is_log_scale: false,
            },
        ];
        Self { 
            ranges,
            parameters: Vec::new(),
        }
    }

    /// 创建IVF索引的参数空间
    pub fn ivf() -> Self {
        let ranges = vec![
            ParameterRange {
                name: "nlist".to_string(),
                min: 4.0,
                max: 256.0,
                step: 4.0,
                is_integer: true,
                is_log_scale: true,
            },
            ParameterRange {
                name: "nprobe".to_string(),
                min: 1.0,
                max: 64.0,
                step: 1.0,
                is_integer: true,
                is_log_scale: false,
            },
        ];
        Self { 
            ranges,
            parameters: Vec::new(),
        }
    }

    /// 创建PQ索引的参数空间
    pub fn pq() -> Self {
        let ranges = vec![
            ParameterRange {
                name: "m".to_string(),
                min: 1.0,
                max: 16.0,
                step: 1.0,
                is_integer: true,
                is_log_scale: false,
            },
            ParameterRange {
                name: "nbits".to_string(),
                min: 4.0,
                max: 12.0,
                step: 1.0,
                is_integer: true,
                is_log_scale: false,
            },
            ParameterRange {
                name: "nlist".to_string(),
                min: 4.0,
                max: 256.0,
                step: 4.0,
                is_integer: true,
                is_log_scale: true,
            },
            ParameterRange {
                name: "nprobe".to_string(),
                min: 1.0,
                max: 32.0,
                step: 1.0,
                is_integer: true,
                is_log_scale: false,
            },
        ];
        Self { 
            ranges,
            parameters: Vec::new(),
        }
    }

    /// 创建LSH索引的参数空间
    pub fn lsh() -> Self {
        let ranges = vec![
            ParameterRange {
                name: "nbits".to_string(),
                min: 4.0,
                max: 32.0,
                step: 4.0,
                is_integer: true,
                is_log_scale: false,
            },
            ParameterRange {
                name: "ntables".to_string(),
                min: 1.0,
                max: 32.0,
                step: 1.0,
                is_integer: true,
                is_log_scale: false,
            },
            ParameterRange {
                name: "multi_probe".to_string(),
                min: 0.0,
                max: 16.0,
                step: 1.0,
                is_integer: true,
                is_log_scale: false,
            },
        ];
        Self { 
            ranges,
            parameters: Vec::new(),
        }
    }

    /// 根据索引类型创建参数空间
    pub fn for_index_type(index_type: IndexType) -> Self {
        match index_type {
            IndexType::HNSW => Self::hnsw(),
            IndexType::IVF => Self::ivf(),
            IndexType::PQ => Self::pq(),
            IndexType::LSH => Self::lsh(),
            IndexType::Flat => Self { ranges: vec![], parameters: Vec::new() }, // Flat索引没有参数
            _ => Self { ranges: vec![], parameters: Vec::new() },  // 默认返回空参数空间
        }
    }

    /// 生成随机参数
    pub fn random_params(&self, rng: &mut impl Rng) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        
        for range in &self.ranges {
            let value = if range.is_log_scale {
                let log_min = range.min.ln();
                let log_max = range.max.ln();
                let log_value = rng.gen_range(log_min..log_max);
                log_value.exp()
            } else {
                rng.gen_range(range.min..range.max)
            };
            
            let value = if range.is_integer {
                value.round()
            } else {
                value
            };
            
            params.insert(range.name.clone(), value);
        }
        
        params
    }

    /// 生成网格参数集
    pub fn grid_params(&self) -> Vec<HashMap<String, f64>> {
        let mut result = Vec::new();
        let mut current = HashMap::new();
        self.generate_grid_params(&mut result, &mut current, 0);
        result
    }

    /// 递归生成网格参数集（辅助方法）
    fn generate_grid_params(&self, result: &mut Vec<HashMap<String, f64>>, current: &mut HashMap<String, f64>, index: usize) {
        if index >= self.ranges.len() {
            result.push(current.clone());
            return;
        }
        
        let range = &self.ranges[index];
        let mut value = range.min;
        
        while value <= range.max {
            current.insert(range.name.clone(), value);
            self.generate_grid_params(result, current, index + 1);
            value += range.step;
        }
        
        current.remove(&range.name);
    }

    /// 迭代所有参数
    pub fn parameters(&self) -> impl Iterator<Item = &Parameter> {
        self.parameters.iter()
    }
    
    /// 添加参数
    pub fn add_parameter(&mut self, parameter: Parameter) {
        self.parameters.push(parameter);
    }
} 