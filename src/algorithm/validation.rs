/// 算法验证模块
/// 
/// 提供算法的安全性、正确性、性能等方面的验证功能

use std::time::Duration;
use serde::{Serialize, Deserialize};
use crate::error::{Result};
use crate::algorithm::types::{Algorithm, AlgorithmType};

/// 算法验证器
pub struct AlgorithmValidator {
    /// 验证规则集合
    rules: Vec<Box<dyn ValidationRule>>,
    /// 验证配置
    config: ValidationConfig,
}

/// 验证配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// 启用安全验证
    pub enable_security_validation: bool,
    /// 启用性能验证
    pub enable_performance_validation: bool,
    /// 启用语法验证
    pub enable_syntax_validation: bool,
    /// 验证超时时间
    pub validation_timeout: Duration,
    /// 最大内存使用
    pub max_memory_usage: usize,
    /// 最大CPU使用率
    pub max_cpu_usage: f32,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_security_validation: true,
            enable_performance_validation: true,
            enable_syntax_validation: true,
            validation_timeout: Duration::from_secs(30),
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
            max_cpu_usage: 0.8, // 80%
        }
    }
}

/// 验证报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// 验证是否通过
    pub is_valid: bool,
    /// 验证问题列表
    pub issues: Vec<ValidationIssue>,
    /// 安全评分 (0-100)
    pub security_score: u8,
    /// 性能评分 (0-100)
    pub performance_score: u8,
    /// 验证耗时
    pub validation_time: Duration,
    /// 验证器版本
    pub validator_version: String,
    /// 验证时间戳
    pub validated_at: chrono::DateTime<chrono::Utc>,
}

/// 验证问题
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// 问题严重程度
    pub severity: ValidationSeverity,
    /// 问题分类
    pub category: ValidationCategory,
    /// 问题代码
    pub code: String,
    /// 问题描述
    pub description: String,
    /// 建议修复方案
    pub suggestion: Option<String>,
    /// 问题位置
    pub location: Option<CodeLocation>,
}

/// 验证严重程度
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    /// 信息
    Info,
    /// 警告
    Warning,
    /// 错误
    Error,
    /// 严重错误
    Critical,
}

/// 验证分类
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationCategory {
    /// 安全性
    Security,
    /// 性能
    Performance,
    /// 语法
    Syntax,
    /// 逻辑
    Logic,
    /// 资源使用
    ResourceUsage,
    /// 兼容性
    Compatibility,
}

/// 代码位置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeLocation {
    /// 文件名
    pub file: Option<String>,
    /// 行号
    pub line: Option<u32>,
    /// 列号
    pub column: Option<u32>,
    /// 函数名
    pub function: Option<String>,
}

/// 验证规则接口
pub trait ValidationRule: Send + Sync {
    /// 规则名称
    fn name(&self) -> &str;
    
    /// 规则描述
    fn description(&self) -> &str;
    
    /// 规则严重程度
    fn severity(&self) -> ValidationSeverity;
    
    /// 验证算法
    fn validate(&self, algorithm: &Algorithm) -> Result<Vec<ValidationIssue>>;
    
    /// 是否启用
    fn is_enabled(&self) -> bool {
        true
    }
}

impl AlgorithmValidator {
    /// 创建新的验证器
    pub fn new() -> Self {
        let mut validator = Self {
            rules: Vec::new(),
            config: ValidationConfig::default(),
        };
        
        // 添加默认验证规则
        validator.add_default_rules();
        validator
    }
    
    /// 使用自定义配置创建验证器
    pub fn with_config(config: ValidationConfig) -> Self {
        let mut validator = Self {
            rules: Vec::new(),
            config,
        };
        
        validator.add_default_rules();
        validator
    }
    
    /// 添加验证规则
    pub fn add_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.rules.push(rule);
    }
    
    /// 验证算法
    pub fn validate(&self, algorithm: &Algorithm) -> Result<ValidationReport> {
        let start_time = std::time::Instant::now();
        let mut all_issues = Vec::new();
        let mut security_score = 100u8;
        let mut performance_score = 100u8;
        
        // 执行所有验证规则
        for rule in &self.rules {
            if !rule.is_enabled() {
                continue;
            }
            
            match rule.validate(algorithm) {
                Ok(issues) => {
                    for issue in &issues {
                        // 根据问题严重程度调整评分
                        match issue.severity {
                            ValidationSeverity::Critical => {
                                if issue.category == ValidationCategory::Security {
                                    security_score = security_score.saturating_sub(30);
                                } else if issue.category == ValidationCategory::Performance {
                                    performance_score = performance_score.saturating_sub(30);
                                }
                            },
                            ValidationSeverity::Error => {
                                if issue.category == ValidationCategory::Security {
                                    security_score = security_score.saturating_sub(20);
                                } else if issue.category == ValidationCategory::Performance {
                                    performance_score = performance_score.saturating_sub(20);
                                }
                            },
                            ValidationSeverity::Warning => {
                                if issue.category == ValidationCategory::Security {
                                    security_score = security_score.saturating_sub(10);
                                } else if issue.category == ValidationCategory::Performance {
                                    performance_score = performance_score.saturating_sub(10);
                                }
                            },
                            ValidationSeverity::Info => {
                                if issue.category == ValidationCategory::Security {
                                    security_score = security_score.saturating_sub(5);
                                } else if issue.category == ValidationCategory::Performance {
                                    performance_score = performance_score.saturating_sub(5);
                                }
                            },
                        }
                    }
                    all_issues.extend(issues);
                },
                Err(e) => {
                    // 验证规则执行失败
                    all_issues.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        category: ValidationCategory::Logic,
                        code: "VALIDATION_RULE_FAILED".to_string(),
                        description: format!("验证规则 '{}' 执行失败: {}", rule.name(), e),
                        suggestion: Some("检查验证规则配置或联系系统管理员".to_string()),
                        location: None,
                    });
                }
            }
        }
        
        let validation_time = start_time.elapsed();
        let is_valid = !all_issues.iter().any(|issue| 
            matches!(issue.severity, ValidationSeverity::Critical | ValidationSeverity::Error)
        );
        
        Ok(ValidationReport {
            is_valid,
            issues: all_issues,
            security_score,
            performance_score,
            validation_time,
            validator_version: env!("CARGO_PKG_VERSION").to_string(),
            validated_at: chrono::Utc::now().timestamp(),
        })
    }
    
    /// 添加默认验证规则
    fn add_default_rules(&mut self) {
        if self.config.enable_security_validation {
            self.add_rule(Box::new(SecurityValidationRule));
            self.add_rule(Box::new(MemoryAccessValidationRule));
            self.add_rule(Box::new(NetworkAccessValidationRule));
        }
        
        if self.config.enable_performance_validation {
            self.add_rule(Box::new(PerformanceValidationRule));
            self.add_rule(Box::new(ResourceUsageValidationRule));
        }
        
        if self.config.enable_syntax_validation {
            self.add_rule(Box::new(SyntaxValidationRule));
            self.add_rule(Box::new(TypeSafetyValidationRule));
        }
    }
}

/// 安全验证规则
pub struct SecurityValidationRule;

impl ValidationRule for SecurityValidationRule {
    fn name(&self) -> &str {
        "Security Validation"
    }
    
    fn description(&self) -> &str {
        "检查算法的安全性，包括恶意代码、权限滥用等"
    }
    
    fn severity(&self) -> ValidationSeverity {
        ValidationSeverity::Critical
    }
    
    fn validate(&self, algorithm: &Algorithm) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();
        
        // 检查算法名称中的可疑模式
        if algorithm.name.contains("eval") || algorithm.name.contains("exec") {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                category: ValidationCategory::Security,
                code: "SUSPICIOUS_NAME".to_string(),
                description: "算法名称包含可疑的执行相关关键词".to_string(),
                suggestion: Some("避免使用可能引起安全担忧的命名".to_string()),
                location: None,
            });
        }
        
        // 检查算法类型的安全性
        match algorithm.algorithm_type {
            AlgorithmType::CustomScript => {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    category: ValidationCategory::Security,
                    code: "CUSTOM_SCRIPT_RISK".to_string(),
                    description: "自定义脚本算法可能存在安全风险".to_string(),
                    suggestion: Some("确保脚本在安全沙箱中执行".to_string()),
                    location: None,
                });
            },
            _ => {}
        }
        
        Ok(issues)
    }
}

/// 内存访问验证规则
pub struct MemoryAccessValidationRule;

impl ValidationRule for MemoryAccessValidationRule {
    fn name(&self) -> &str {
        "Memory Access Validation"
    }
    
    fn description(&self) -> &str {
        "检查算法的内存访问模式"
    }
    
    fn severity(&self) -> ValidationSeverity {
        ValidationSeverity::Error
    }
    
    fn validate(&self, _algorithm: &Algorithm) -> Result<Vec<ValidationIssue>> {
        // 简化实现，实际应该分析算法的内存访问模式
        Ok(Vec::new())
    }
}

/// 网络访问验证规则
pub struct NetworkAccessValidationRule;

impl ValidationRule for NetworkAccessValidationRule {
    fn name(&self) -> &str {
        "Network Access Validation"
    }
    
    fn description(&self) -> &str {
        "检查算法是否进行网络访问"
    }
    
    fn severity(&self) -> ValidationSeverity {
        ValidationSeverity::Warning
    }
    
    fn validate(&self, _algorithm: &Algorithm) -> Result<Vec<ValidationIssue>> {
        // 简化实现，实际应该检查网络访问模式
        Ok(Vec::new())
    }
}

/// 性能验证规则
pub struct PerformanceValidationRule;

impl ValidationRule for PerformanceValidationRule {
    fn name(&self) -> &str {
        "Performance Validation"
    }
    
    fn description(&self) -> &str {
        "检查算法的性能特征"
    }
    
    fn severity(&self) -> ValidationSeverity {
        ValidationSeverity::Warning
    }
    
    fn validate(&self, algorithm: &Algorithm) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();
        
        // 检查算法复杂度
        if let Some(metadata) = &algorithm.metadata {
            if let Some(complexity) = metadata.get("complexity") {
                if complexity.contains("O(n^3)") || complexity.contains("O(2^n)") {
                    issues.push(ValidationIssue {
                        severity: ValidationSeverity::Warning,
                        category: ValidationCategory::Performance,
                        code: "HIGH_COMPLEXITY".to_string(),
                        description: "算法时间复杂度较高".to_string(),
                        suggestion: Some("考虑优化算法或增加输入数据大小限制".to_string()),
                        location: None,
                    });
                }
            }
        }
        
        Ok(issues)
    }
}

/// 资源使用验证规则
pub struct ResourceUsageValidationRule;

impl ValidationRule for ResourceUsageValidationRule {
    fn name(&self) -> &str {
        "Resource Usage Validation"
    }
    
    fn description(&self) -> &str {
        "检查算法的资源使用情况"
    }
    
    fn severity(&self) -> ValidationSeverity {
        ValidationSeverity::Warning
    }
    
    fn validate(&self, _algorithm: &Algorithm) -> Result<Vec<ValidationIssue>> {
        // 简化实现，实际应该分析算法的资源使用模式
        Ok(Vec::new())
    }
}

/// 语法验证规则
pub struct SyntaxValidationRule;

impl ValidationRule for SyntaxValidationRule {
    fn name(&self) -> &str {
        "Syntax Validation"
    }
    
    fn description(&self) -> &str {
        "检查算法代码的语法正确性"
    }
    
    fn severity(&self) -> ValidationSeverity {
        ValidationSeverity::Error
    }
    
    fn validate(&self, _algorithm: &Algorithm) -> Result<Vec<ValidationIssue>> {
        // 简化实现，实际应该进行语法解析
        Ok(Vec::new())
    }
}

/// 类型安全验证规则
pub struct TypeSafetyValidationRule;

impl ValidationRule for TypeSafetyValidationRule {
    fn name(&self) -> &str {
        "Type Safety Validation"
    }
    
    fn description(&self) -> &str {
        "检查算法的类型安全性"
    }
    
    fn severity(&self) -> ValidationSeverity {
        ValidationSeverity::Error
    }
    
    fn validate(&self, _algorithm: &Algorithm) -> Result<Vec<ValidationIssue>> {
        // 简化实现，实际应该进行类型检查
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::types::AlgorithmType;
    
    #[test]
    fn test_validator_creation() {
        let validator = AlgorithmValidator::new();
        assert!(!validator.rules.is_empty());
    }
    
    #[test]
    fn test_validation_with_safe_algorithm() {
        let validator = AlgorithmValidator::new();
        let algorithm = Algorithm {
            id: "test_algo".to_string(),
            name: "safe_algorithm".to_string(),
            description: Some("A safe test algorithm".to_string()),
            algorithm_type: AlgorithmType::MachineLearning,
            metadata: None,
            created_at: chrono::Utc::now().timestamp(),
            updated_at: chrono::Utc::now().timestamp(),
        };
        
        let report = validator.validate(&algorithm).unwrap();
        assert!(report.security_score > 80);
        assert!(report.performance_score > 80);
    }
    
    #[test]
    fn test_validation_with_suspicious_algorithm() {
        let validator = AlgorithmValidator::new();
        let algorithm = Algorithm {
            id: "test_algo".to_string(),
            name: "eval_algorithm".to_string(),
            description: Some("An algorithm with suspicious name".to_string()),
            algorithm_type: AlgorithmType::CustomScript,
            metadata: None,
            created_at: chrono::Utc::now().timestamp(),
            updated_at: chrono::Utc::now().timestamp(),
        };
        
        let report = validator.validate(&algorithm).unwrap();
        assert!(!report.issues.is_empty());
        assert!(report.security_score < 100);
    }
} 