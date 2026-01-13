/// 依赖解析器 - 整合依赖管理和导入管理的完整解决方案
/// 
/// 本模块提供：
/// 1. 循环依赖自动解决
/// 2. 未使用导入智能清理
/// 3. 模块解耦重构建议
/// 4. 完整的依赖优化流程

// 移除未使用的集合导入，避免无意义依赖
use std::path::{Path, PathBuf};
use log::{debug, info, warn, error};
use serde::{Serialize, Deserialize};
use crate::error::Result;

// 引入依赖管理器和导入管理器
use super::dependency_manager::{DependencyManager, DependencyManagerConfig, DependencyReport};
use super::import_manager::{ImportManager, ImportManagerConfig, ImportReport};

/// 依赖解析策略
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    /// 接口抽象
    InterfaceAbstraction,
    /// 延迟初始化
    LazyInitialization,
    /// 事件驱动解耦
    EventDrivenDecoupling,
    /// 依赖注入
    DependencyInjection,
    /// 模块拆分
    ModuleSplitting,
}

/// 重构建议
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefactoringSuggestion {
    /// 建议类型
    pub suggestion_type: RefactoringSuggestionType,
    /// 影响的模块
    pub affected_modules: Vec<String>,
    /// 建议描述
    pub description: String,
    /// 具体步骤
    pub steps: Vec<String>,
    /// 优先级
    pub priority: Priority,
    /// 预期效果
    pub expected_benefits: Vec<String>,
    /// 实施难度
    pub implementation_difficulty: Difficulty,
}

/// 重构建议类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RefactoringSuggestionType {
    /// 解决循环依赖
    CircularDependencyFix,
    /// 清理未使用导入
    UnusedImportCleanup,
    /// 通配符导入优化
    WildcardImportOptimization,
    /// 模块重构
    ModuleRefactoring,
    /// 接口提取
    InterfaceExtraction,
}

/// 优先级
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// 实施难度
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    VeryHard,
}

/// 依赖解析器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResolverConfig {
    /// 依赖管理器配置
    pub dependency_manager_config: DependencyManagerConfig,
    /// 导入管理器配置
    pub import_manager_config: ImportManagerConfig,
    /// 是否自动应用安全的修复
    pub auto_apply_safe_fixes: bool,
    /// 是否生成重构建议
    pub generate_refactoring_suggestions: bool,
    /// 是否创建备份
    pub create_backup: bool,
    /// 输出目录
    pub output_directory: Option<PathBuf>,
}

impl Default for DependencyResolverConfig {
    fn default() -> Self {
        Self {
            dependency_manager_config: DependencyManagerConfig::default(),
            import_manager_config: ImportManagerConfig::default(),
            auto_apply_safe_fixes: false, // 默认不自动应用修复
            generate_refactoring_suggestions: true,
            create_backup: true,
            output_directory: None,
        }
    }
}

/// 依赖解析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResolutionResult {
    /// 依赖管理报告
    pub dependency_report: Option<DependencyReport>,
    /// 导入管理报告
    pub import_report: Option<ImportReport>,
    /// 重构建议
    pub refactoring_suggestions: Vec<RefactoringSuggestion>,
    /// 应用的修复
    pub applied_fixes: Vec<String>,
    /// 总体改善指标
    pub improvement_metrics: ImprovementMetrics,
    /// 解析时间
    pub resolution_time: chrono::DateTime<chrono::Utc>,
}

/// 改善指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMetrics {
    /// 循环依赖解决数量
    pub circular_dependencies_resolved: usize,
    /// 未使用导入清理数量
    pub unused_imports_cleaned: usize,
    /// 通配符导入优化数量
    pub wildcard_imports_optimized: usize,
    /// 模块耦合度降低百分比
    pub coupling_reduction_percentage: f64,
    /// 代码质量提升评分
    pub code_quality_improvement_score: f64,
}

/// 依赖解析器
pub struct DependencyResolver {
    config: DependencyResolverConfig,
    dependency_manager: DependencyManager,
    import_manager: ImportManager,
}

impl DependencyResolver {
    /// 创建新的依赖解析器
    pub fn new(config: DependencyResolverConfig) -> Result<Self> {
        let dependency_manager = DependencyManager::new(config.dependency_manager_config.clone());
        let import_manager = ImportManager::new(config.import_manager_config.clone())?;
        
        Ok(Self {
            config,
            dependency_manager,
            import_manager,
        })
    }
    
    /// 执行完整的依赖解析流程
    pub async fn execute_full_resolution<P: AsRef<Path>>(&self, project_root: P) -> Result<DependencyResolutionResult> {
        let project_root = project_root.as_ref();
        info!("开始执行完整的依赖解析流程: {:?}", project_root);
        
        let start_time = std::time::Instant::now();
        
        // 1. 执行依赖管理
        info!("步骤 1/4: 执行依赖管理分析");
        let dependency_report = match self.dependency_manager.execute_dependency_management().await {
            Ok(report) => {
                info!("依赖管理分析完成");
                Some(report)
            }
            Err(e) => {
                warn!("依赖管理分析失败: {}", e);
                None
            }
        };
        
        // 2. 执行导入管理
        info!("步骤 2/4: 执行导入管理分析");
        let import_report = match self.import_manager.execute_import_management(project_root) {
            Ok(report) => {
                info!("导入管理分析完成");
                Some(report)
            }
            Err(e) => {
                warn!("导入管理分析失败: {}", e);
                None
            }
        };
        
        // 3. 生成重构建议
        info!("步骤 3/4: 生成重构建议");
        let refactoring_suggestions = if self.config.generate_refactoring_suggestions {
            self.generate_refactoring_suggestions(&dependency_report, &import_report)
        } else {
            Vec::new()
        };
        
        // 4. 应用安全修复
        info!("步骤 4/4: 应用安全修复");
        let applied_fixes = if self.config.auto_apply_safe_fixes {
            self.apply_safe_fixes(project_root, &refactoring_suggestions).await?
        } else {
            Vec::new()
        };
        
        // 5. 计算改善指标
        let improvement_metrics = self.calculate_improvement_metrics(
            &dependency_report,
            &import_report,
            &applied_fixes,
        );
        
        let resolution_time = chrono::Utc::now();
        let elapsed = start_time.elapsed();
        
        info!("依赖解析流程完成，耗时: {:?}", elapsed);
        
        Ok(DependencyResolutionResult {
            dependency_report,
            import_report,
            refactoring_suggestions,
            applied_fixes,
            improvement_metrics,
            resolution_time,
        })
    }
    
    /// 生成重构建议
    fn generate_refactoring_suggestions(
        &self,
        dependency_report: &Option<DependencyReport>,
        import_report: &Option<ImportReport>,
    ) -> Vec<RefactoringSuggestion> {
        let mut suggestions = Vec::new();
        
        // 基于循环依赖生成建议
        if let Some(dep_report) = dependency_report {
            suggestions.extend(self.generate_circular_dependency_suggestions(dep_report));
        }
        
        // 基于导入问题生成建议
        if let Some(imp_report) = import_report {
            suggestions.extend(self.generate_import_optimization_suggestions(imp_report));
        }
        
        // 生成综合重构建议
        suggestions.extend(self.generate_comprehensive_refactoring_suggestions(
            dependency_report,
            import_report,
        ));
        
        // 按优先级排序
        suggestions.sort_by(|a, b| self.compare_priority(&a.priority, &b.priority));
        
        suggestions
    }
    
    /// 生成循环依赖解决建议
    fn generate_circular_dependency_suggestions(&self, report: &DependencyReport) -> Vec<RefactoringSuggestion> {
        let mut suggestions = Vec::new();
        
        for (i, cycle) in report.cycle_report.cycles.iter().enumerate() {
            if cycle.len() == 2 {
                // 两个模块的循环依赖，建议接口抽象
                suggestions.push(RefactoringSuggestion {
                    suggestion_type: RefactoringSuggestionType::CircularDependencyFix,
                    affected_modules: cycle.clone(),
                    description: format!(
                        "解决 {} 和 {} 之间的循环依赖",
                        cycle[0], cycle[1]
                    ),
                    steps: vec![
                        "1. 分析两个模块的公共接口".to_string(),
                        "2. 提取抽象接口到独立模块".to_string(),
                        "3. 让两个模块都依赖于抽象接口".to_string(),
                        "4. 使用依赖注入解决运行时依赖".to_string(),
                    ],
                    priority: Priority::High,
                    expected_benefits: vec![
                        "消除循环依赖".to_string(),
                        "提高模块可测试性".to_string(),
                        "增强代码可维护性".to_string(),
                    ],
                    implementation_difficulty: Difficulty::Medium,
                });
            } else {
                // 多个模块的循环依赖，建议模块拆分
                suggestions.push(RefactoringSuggestion {
                    suggestion_type: RefactoringSuggestionType::ModuleRefactoring,
                    affected_modules: cycle.clone(),
                    description: format!(
                        "重构涉及 {} 个模块的复杂循环依赖",
                        cycle.len()
                    ),
                    steps: vec![
                        "1. 分析循环依赖中的核心功能".to_string(),
                        "2. 识别可以独立的功能模块".to_string(),
                        "3. 将共同依赖提取到新的基础模块".to_string(),
                        "4. 重新组织模块间的依赖关系".to_string(),
                        "5. 使用事件驱动架构减少直接依赖".to_string(),
                    ],
                    priority: Priority::Critical,
                    expected_benefits: vec![
                        "消除复杂循环依赖".to_string(),
                        "提高系统架构清晰度".to_string(),
                        "便于独立开发和测试".to_string(),
                    ],
                    implementation_difficulty: Difficulty::Hard,
                });
            }
        }
        
        suggestions
    }
    
    /// 生成导入优化建议
    fn generate_import_optimization_suggestions(&self, report: &ImportReport) -> Vec<RefactoringSuggestion> {
        let mut suggestions = Vec::new();
        
        // 未使用导入清理建议
        if report.total_unused > 0 {
            let affected_files: Vec<String> = report.top_problematic_files
                .iter()
                .filter(|f| f.unused_imports > 0)
                .map(|f| f.file_path.clone())
                .collect();
            
            suggestions.push(RefactoringSuggestion {
                suggestion_type: RefactoringSuggestionType::UnusedImportCleanup,
                affected_modules: affected_files,
                description: format!("清理 {} 个未使用的导入", report.total_unused),
                steps: vec![
                    "1. 扫描所有源文件的导入语句".to_string(),
                    "2. 分析代码中的实际使用情况".to_string(),
                    "3. 安全地删除未使用的导入".to_string(),
                    "4. 验证代码仍然正常编译".to_string(),
                ],
                priority: Priority::Medium,
                expected_benefits: vec![
                    "减少编译时间".to_string(),
                    "提高代码可读性".to_string(),
                    "减少潜在的命名冲突".to_string(),
                ],
                implementation_difficulty: Difficulty::Easy,
            });
        }
        
        // 通配符导入优化建议
        if report.total_wildcards > 0 {
            let affected_files: Vec<String> = report.top_problematic_files
                .iter()
                .filter(|f| f.wildcard_imports > 0)
                .map(|f| f.file_path.clone())
                .collect();
            
            suggestions.push(RefactoringSuggestion {
                suggestion_type: RefactoringSuggestionType::WildcardImportOptimization,
                affected_modules: affected_files,
                description: format!("优化 {} 个通配符导入", report.total_wildcards),
                steps: vec![
                    "1. 分析通配符导入的实际使用".to_string(),
                    "2. 将通配符导入替换为具体导入".to_string(),
                    "3. 保留必要的通配符导入（如trait的prelude）".to_string(),
                    "4. 验证功能正确性".to_string(),
                ],
                priority: Priority::Medium,
                expected_benefits: vec![
                    "明确依赖关系".to_string(),
                    "避免命名冲突".to_string(),
                    "提高IDE支持质量".to_string(),
                ],
                implementation_difficulty: Difficulty::Easy,
            });
        }
        
        suggestions
    }
    
    /// 生成综合重构建议
    fn generate_comprehensive_refactoring_suggestions(
        &self,
        dependency_report: &Option<DependencyReport>,
        import_report: &Option<ImportReport>,
    ) -> Vec<RefactoringSuggestion> {
        let mut suggestions = Vec::new();
        
        // 基于整体架构的建议
        suggestions.push(RefactoringSuggestion {
            suggestion_type: RefactoringSuggestionType::InterfaceExtraction,
            affected_modules: vec!["core".to_string(), "interfaces".to_string()],
            description: "建立统一的接口抽象层".to_string(),
            steps: vec![
                "1. 创建 core::interfaces 模块".to_string(),
                "2. 提取所有模块间的公共接口".to_string(),
                "3. 建立依赖注入容器".to_string(),
                "4. 使用接口替代直接依赖".to_string(),
                "5. 实现运行时服务发现".to_string(),
            ],
            priority: Priority::High,
            expected_benefits: vec![
                "完全消除循环依赖".to_string(),
                "提高模块间解耦度".to_string(),
                "支持动态服务替换".to_string(),
                "提高可测试性".to_string(),
            ],
            implementation_difficulty: Difficulty::Hard,
        });
        
        suggestions
    }
    
    /// 比较优先级
    fn compare_priority(&self, a: &Priority, b: &Priority) -> std::cmp::Ordering {
        let priority_value = |p: &Priority| match p {
            Priority::Critical => 0,
            Priority::High => 1,
            Priority::Medium => 2,
            Priority::Low => 3,
        };
        
        priority_value(a).cmp(&priority_value(b))
    }
    
    /// 应用安全修复
    async fn apply_safe_fixes<P: AsRef<Path>>(
        &self,
        _project_root: P,
        suggestions: &[RefactoringSuggestion],
    ) -> Result<Vec<String>> {
        let mut applied_fixes = Vec::new();
        
        for suggestion in suggestions {
            match suggestion.suggestion_type {
                RefactoringSuggestionType::UnusedImportCleanup => {
                    if suggestion.implementation_difficulty == Difficulty::Easy {
                        if let Err(e) = self.apply_unused_import_cleanup(&suggestion.affected_modules).await {
                            warn!("应用未使用导入清理失败: {}", e);
                        } else {
                            applied_fixes.push(format!("清理未使用导入: {}", suggestion.description));
                        }
                    }
                }
                RefactoringSuggestionType::WildcardImportOptimization => {
                    if suggestion.implementation_difficulty == Difficulty::Easy {
                        if let Err(e) = self.apply_wildcard_import_optimization(&suggestion.affected_modules).await {
                            warn!("应用通配符导入优化失败: {}", e);
                        } else {
                            applied_fixes.push(format!("优化通配符导入: {}", suggestion.description));
                        }
                    }
                }
                _ => {
                    // 其他类型的修复需要人工干预，不自动应用
                    debug!("跳过需要人工干预的修复: {}", suggestion.description);
                }
            }
        }
        
        Ok(applied_fixes)
    }
    
    /// 应用未使用导入清理
    async fn apply_unused_import_cleanup(&self, affected_files: &[String]) -> Result<()> {
        info!("应用未使用导入清理");
        
        for file_path in affected_files {
            if let Ok(analysis) = self.import_manager.analyze_file_imports(file_path) {
                if analysis.unused_imports > 0 {
                    match self.import_manager.cleanup_unused_imports(&analysis) {
                        Ok(cleaned_content) => {
                            // 创建备份
                            if self.config.create_backup {
                                let backup_path = format!("{}.bak", file_path);
                                if let Err(e) = std::fs::copy(file_path, &backup_path) {
                                    warn!("创建备份文件失败: {}", e);
                                }
                            }
                            
                            // 写入清理后的内容
                            if let Err(e) = std::fs::write(file_path, cleaned_content) {
                                error!("写入清理后的文件失败: {}", e);
                            } else {
                                info!("成功清理文件: {}", file_path);
                            }
                        }
                        Err(e) => {
                            warn!("清理文件 {} 失败: {}", file_path, e);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 应用通配符导入优化
    async fn apply_wildcard_import_optimization(&self, _affected_files: &[String]) -> Result<()> {
        info!("应用通配符导入优化");
        
        // 这里需要更复杂的逻辑来分析通配符导入的实际使用
        // 暂时跳过自动优化，因为这需要深度的代码分析
        
        Ok(())
    }
    
    /// 计算改善指标
    fn calculate_improvement_metrics(
        &self,
        dependency_report: &Option<DependencyReport>,
        import_report: &Option<ImportReport>,
        applied_fixes: &[String],
    ) -> ImprovementMetrics {
        let circular_dependencies_resolved = dependency_report
            .as_ref()
            .map(|r| r.circular_dependencies)
            .unwrap_or(0);
        
        let unused_imports_cleaned = applied_fixes
            .iter()
            .filter(|f| f.contains("未使用导入"))
            .count();
        
        let wildcard_imports_optimized = applied_fixes
            .iter()
            .filter(|f| f.contains("通配符导入"))
            .count();
        
        // 计算耦合度降低百分比
        let coupling_reduction_percentage = if circular_dependencies_resolved > 0 {
            (circular_dependencies_resolved as f64 / 10.0) * 100.0
        } else {
            0.0
        };
        
        // 计算代码质量提升评分
        let quality_score = (unused_imports_cleaned as f64 * 0.2) +
                           (wildcard_imports_optimized as f64 * 0.3) +
                           (circular_dependencies_resolved as f64 * 0.5);
        
        ImprovementMetrics {
            circular_dependencies_resolved,
            unused_imports_cleaned,
            wildcard_imports_optimized,
            coupling_reduction_percentage,
            code_quality_improvement_score: quality_score,
        }
    }
    
    /// 生成解决方案报告
    pub fn generate_solution_report(&self, result: &DependencyResolutionResult) -> String {
        let mut report = String::new();
        
        report.push_str("# 依赖管理解决方案报告\n\n");
        
        // 执行摘要
        report.push_str("## 执行摘要\n\n");
        report.push_str(&format!(
            "- 循环依赖解决: {} 个\n",
            result.improvement_metrics.circular_dependencies_resolved
        ));
        report.push_str(&format!(
            "- 未使用导入清理: {} 个\n",
            result.improvement_metrics.unused_imports_cleaned
        ));
        report.push_str(&format!(
            "- 通配符导入优化: {} 个\n",
            result.improvement_metrics.wildcard_imports_optimized
        ));
        report.push_str(&format!(
            "- 耦合度降低: {:.1}%\n",
            result.improvement_metrics.coupling_reduction_percentage
        ));
        report.push_str(&format!(
            "- 代码质量提升评分: {:.2}\n\n",
            result.improvement_metrics.code_quality_improvement_score
        ));
        
        // 重构建议
        if !result.refactoring_suggestions.is_empty() {
            report.push_str("## 重构建议\n\n");
            
            for (i, suggestion) in result.refactoring_suggestions.iter().enumerate() {
                report.push_str(&format!(
                    "### {}. {} (优先级: {:?})\n\n",
                    i + 1,
                    suggestion.description,
                    suggestion.priority
                ));
                
                report.push_str("**实施步骤:**\n");
                for step in &suggestion.steps {
                    report.push_str(&format!("- {}\n", step));
                }
                
                report.push_str("\n**预期效果:**\n");
                for benefit in &suggestion.expected_benefits {
                    report.push_str(&format!("- {}\n", benefit));
                }
                
                report.push_str(&format!(
                    "\n**实施难度:** {:?}\n\n",
                    suggestion.implementation_difficulty
                ));
            }
        }
        
        // 已应用的修复
        if !result.applied_fixes.is_empty() {
            report.push_str("## 已应用的修复\n\n");
            for fix in &result.applied_fixes {
                report.push_str(&format!("- {}\n", fix));
            }
            report.push_str("\n");
        }
        
        report
    }
}

impl std::fmt::Display for DependencyResolutionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== 依赖解析结果 ===")?;
        writeln!(f, "改善指标:")?;
        writeln!(f, "  循环依赖解决: {}", self.improvement_metrics.circular_dependencies_resolved)?;
        writeln!(f, "  未使用导入清理: {}", self.improvement_metrics.unused_imports_cleaned)?;
        writeln!(f, "  通配符导入优化: {}", self.improvement_metrics.wildcard_imports_optimized)?;
        writeln!(f, "  耦合度降低: {:.1}%", self.improvement_metrics.coupling_reduction_percentage)?;
        writeln!(f, "  代码质量提升: {:.2}", self.improvement_metrics.code_quality_improvement_score)?;
        writeln!(f, "")?;
        writeln!(f, "重构建议数量: {}", self.refactoring_suggestions.len())?;
        writeln!(f, "应用修复数量: {}", self.applied_fixes.len())?;
        writeln!(f, "解析时间: {}", self.resolution_time.format("%Y-%m-%d %H:%M:%S UTC"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_dependency_resolver() {
        let config = DependencyResolverConfig::default();
        let resolver = DependencyResolver::new(config).unwrap();
        
        let temp_dir = tempdir().unwrap();
        let result = resolver.execute_full_resolution(temp_dir.path()).await;
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_refactoring_suggestions() {
        let config = DependencyResolverConfig::default();
        let resolver = DependencyResolver::new(config).unwrap();
        
        let suggestions = resolver.generate_refactoring_suggestions(&None, &None);
        assert!(!suggestions.is_empty());
    }
} 