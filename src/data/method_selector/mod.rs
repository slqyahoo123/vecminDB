// Method Selector Module
// 方法选择器模块
// 负责基于数据特性自动选择最适合的处理方法

// 重新导出子模块
pub mod analyzer;
pub mod config;
pub mod evaluator;
pub mod rules;
pub mod selector;
pub mod types;

// 重新导出关键类型
pub use types::{
    // MethodSelector, DataAnalyzer, MethodEvaluator, RuleBuilder,
    MethodEvaluation, MethodSelectorConfig, DataCharacteristics,
    DomainRule, PerformanceDataPoint,
};
// 直接导入这些类型
// 未在本模块使用，去除未使用导入
// mongodb-specific selection criteria not used here
// 未在本模块使用，去除未使用导入

// 直接导入TextFieldStats
pub use crate::data::text_features::TextFieldStats;

pub use selector::MethodSelector;
pub use analyzer::DataAnalyzer;
pub use evaluator::MethodEvaluator;
pub use rules::RuleBuilder;
pub use crate::data::text_features::config::TextFeatureMethod;

// 统计类型由各子模块按需导入；此处不直接依赖

// 默认方法选择器实例
use serde_json::Value;
use std::sync::{Arc, Mutex};
use once_cell::sync::Lazy;

/// 全局默认方法选择器
static DEFAULT_SELECTOR: Lazy<Arc<Mutex<MethodSelector>>> = Lazy::new(|| {
    Arc::new(Mutex::new(MethodSelector::new(MethodSelectorConfig::default())))
});

/// 使用默认选择器选择最佳方法
pub async fn select_best_method(data: &[Value], field: &str) -> TextFeatureMethod {
    let mut selector = DEFAULT_SELECTOR.lock().unwrap();
    selector.select_best_method(data, field).await
}

/// 使用自定义配置选择最佳方法
pub async fn select_best_method_with_config(
    data: &[Value], 
    field: &str, 
    config: MethodSelectorConfig
) -> TextFeatureMethod {
    let mut selector = MethodSelector::new(config);
    selector.select_best_method(data, field).await
}

/// 创建新的方法选择器
pub fn new_selector(config: MethodSelectorConfig) -> MethodSelector {
    MethodSelector::new(config)
}

/// 使用默认配置创建新的方法选择器
pub fn new_default_selector() -> MethodSelector {
    MethodSelector::new(MethodSelectorConfig::default())
}

// 引入项目模块
// 未在本模块直接使用这些类型，去除未使用导入

// use std::collections::HashMap; // not used in this module