/// 导入管理器 - 解决未使用导入堆积问题
/// 
/// 本模块提供：
/// 1. 未使用导入检测
/// 2. 导入清理和优化
/// 3. 智能导入管理
/// 4. 导入依赖分析

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::fs;
use std::io::BufRead;
use regex::Regex;
use serde::{Serialize, Deserialize};
use log::{debug, info, warn};
use crate::error::{Error, Result};

/// 导入类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImportType {
    /// 标准库导入
    Standard,
    /// 第三方依赖导入
    External,
    /// 项目内部导入
    Internal,
    /// 通配符导入
    Wildcard,
    /// 条件导入
    Conditional,
}

/// 导入项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportItem {
    /// 导入路径
    pub path: String,
    /// 导入项目
    pub items: Vec<String>,
    /// 导入类型
    pub import_type: ImportType,
    /// 是否为通配符导入
    pub is_wildcard: bool,
    /// 在文件中的行号
    pub line_number: usize,
    /// 原始导入语句
    pub original_statement: String,
    /// 是否被使用
    pub is_used: bool,
    /// 使用次数
    pub usage_count: usize,
    /// 使用位置
    pub usage_locations: Vec<usize>,
}

/// 导入分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportAnalysisResult {
    /// 文件路径
    pub file_path: String,
    /// 总导入数
    pub total_imports: usize,
    /// 未使用导入数
    pub unused_imports: usize,
    /// 通配符导入数
    pub wildcard_imports: usize,
    /// 导入项列表
    pub imports: Vec<ImportItem>,
    /// 建议的优化操作
    pub suggestions: Vec<String>,
    /// 分析时间
    pub analysis_time: chrono::DateTime<chrono::Utc>,
}

/// 导入管理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportManagerConfig {
    /// 是否自动清理未使用导入
    pub auto_cleanup_unused: bool,
    /// 是否转换通配符导入
    pub convert_wildcard_imports: bool,
    /// 是否启用智能导入建议
    pub enable_smart_suggestions: bool,
    /// 排除的文件模式
    pub exclude_patterns: Vec<String>,
    /// 保留的通配符导入
    pub preserve_wildcard_imports: Vec<String>,
    /// 是否生成详细报告
    pub generate_detailed_report: bool,
}

impl Default for ImportManagerConfig {
    fn default() -> Self {
        Self {
            auto_cleanup_unused: false, // 默认不自动清理，避免意外删除
            convert_wildcard_imports: true,
            enable_smart_suggestions: true,
            exclude_patterns: vec![
                "*.rs.bak".to_string(),
                "target/*".to_string(),
                "*/tests/*".to_string(),
            ],
            preserve_wildcard_imports: vec![
                "rayon::prelude::*".to_string(),
                "serde::*".to_string(),
                "tokio::prelude::*".to_string(),
            ],
            generate_detailed_report: true,
        }
    }
}

/// 导入管理器
pub struct ImportManager {
    config: ImportManagerConfig,
    import_patterns: HashMap<String, Regex>,
    usage_patterns: HashMap<String, Regex>,
}

impl ImportManager {
    /// 创建新的导入管理器
    pub fn new(config: ImportManagerConfig) -> Result<Self> {
        let mut manager = Self {
            config,
            import_patterns: HashMap::new(),
            usage_patterns: HashMap::new(),
        };
        
        manager.initialize_patterns()?;
        Ok(manager)
    }
    
    /// 初始化正则表达式模式
    fn initialize_patterns(&mut self) -> Result<()> {
        // 导入语句模式
        self.import_patterns.insert(
            "use_statement".to_string(),
            Regex::new(r"^use\s+([^;]+);").unwrap(),
        );
        
        self.import_patterns.insert(
            "wildcard_import".to_string(),
            Regex::new(r"use\s+([^:]+)::\*;").unwrap(),
        );
        
        self.import_patterns.insert(
            "selective_import".to_string(),
            Regex::new(r"use\s+([^:]+)::\{([^}]+)\};").unwrap(),
        );
        
        self.import_patterns.insert(
            "simple_import".to_string(),
            Regex::new(r"use\s+([^:]+::)?([^:;]+);").unwrap(),
        );
        
        // 使用模式
        self.usage_patterns.insert(
            "identifier_usage".to_string(),
            Regex::new(r"\b([A-Z][a-zA-Z0-9_]*)\b").unwrap(),
        );
        
        self.usage_patterns.insert(
            "function_call".to_string(),
            Regex::new(r"\b([a-z_][a-zA-Z0-9_]*)\s*\(").unwrap(),
        );
        
        self.usage_patterns.insert(
            "macro_usage".to_string(),
            Regex::new(r"\b([a-z_][a-zA-Z0-9_]*!)\s*\(").unwrap(),
        );
        
        Ok(())
    }
    
    /// 分析单个文件的导入
    pub fn analyze_file_imports<P: AsRef<Path>>(&self, file_path: P) -> Result<ImportAnalysisResult> {
        let file_path = file_path.as_ref();
        let content = fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content.lines().collect();
        
        let mut imports = Vec::new();
        let mut used_items = HashSet::new();
        
        // 1. 提取所有导入语句
        for (line_num, line) in lines.iter().enumerate() {
            if let Some(import_item) = self.parse_import_statement(line, line_num + 1) {
                imports.push(import_item);
            }
        }
        
        // 2. 分析代码中的使用情况
        for (line_num, line) in lines.iter().enumerate() {
            if line.trim_start().starts_with("use ") {
                continue; // 跳过导入语句行
            }
            
            // 查找标识符使用
            if let Some(pattern) = self.usage_patterns.get("identifier_usage") {
                for cap in pattern.captures_iter(line) {
                    if let Some(identifier) = cap.get(1) {
                        used_items.insert(identifier.as_str().to_string());
                    }
                }
            }
            
            // 查找函数调用
            if let Some(pattern) = self.usage_patterns.get("function_call") {
                for cap in pattern.captures_iter(line) {
                    if let Some(func_name) = cap.get(1) {
                        used_items.insert(func_name.as_str().to_string());
                    }
                }
            }
            
            // 查找宏使用
            if let Some(pattern) = self.usage_patterns.get("macro_usage") {
                for cap in pattern.captures_iter(line) {
                    if let Some(macro_name) = cap.get(1) {
                        used_items.insert(macro_name.as_str().to_string());
                    }
                }
            }
        }
        
        // 3. 标记使用状态
        for import in &mut imports {
            import.is_used = self.check_import_usage(import, &used_items, &content);
            if import.is_used {
                import.usage_count = self.count_usage_occurrences(import, &content);
                import.usage_locations = self.find_usage_locations(import, &lines);
            }
        }
        
        // 4. 生成分析结果
        let unused_imports = imports.iter().filter(|i| !i.is_used).count();
        let wildcard_imports = imports.iter().filter(|i| i.is_wildcard).count();
        let suggestions = self.generate_suggestions(&imports);
        
        Ok(ImportAnalysisResult {
            file_path: file_path.to_string_lossy().to_string(),
            total_imports: imports.len(),
            unused_imports,
            wildcard_imports,
            imports,
            suggestions,
            analysis_time: chrono::Utc::now(),
        })
    }
    
    /// 解析导入语句
    fn parse_import_statement(&self, line: &str, line_number: usize) -> Option<ImportItem> {
        let trimmed = line.trim();
        
        // 检查是否为导入语句
        if !trimmed.starts_with("use ") {
            return None;
        }
        
        // 通配符导入
        if let Some(pattern) = self.import_patterns.get("wildcard_import") {
            if let Some(cap) = pattern.captures(trimmed) {
                if let Some(path) = cap.get(1) {
                    return Some(ImportItem {
                        path: path.as_str().to_string(),
                        items: vec!["*".to_string()],
                        import_type: ImportType::Wildcard,
                        is_wildcard: true,
                        line_number,
                        original_statement: line.to_string(),
                        is_used: false,
                        usage_count: 0,
                        usage_locations: Vec::new(),
                    });
                }
            }
        }
        
        // 选择性导入
        if let Some(pattern) = self.import_patterns.get("selective_import") {
            if let Some(cap) = pattern.captures(trimmed) {
                if let Some(path) = cap.get(1) {
                    if let Some(items_str) = cap.get(2) {
                        let items: Vec<String> = items_str.as_str()
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                        
                        return Some(ImportItem {
                            path: path.as_str().to_string(),
                            items,
                            import_type: self.classify_import_type(path.as_str()),
                            is_wildcard: false,
                            line_number,
                            original_statement: line.to_string(),
                            is_used: false,
                            usage_count: 0,
                            usage_locations: Vec::new(),
                        });
                    }
                }
            }
        }
        
        // 简单导入
        if let Some(pattern) = self.import_patterns.get("use_statement") {
            if let Some(cap) = pattern.captures(trimmed) {
                if let Some(full_path) = cap.get(1) {
                    let path_str = full_path.as_str();
                    let parts: Vec<&str> = path_str.split("::").collect();
                    let item_name = parts.last().unwrap_or(&"").to_string();
                    
                    return Some(ImportItem {
                        path: path_str.to_string(),
                        items: vec![item_name],
                        import_type: self.classify_import_type(path_str),
                        is_wildcard: false,
                        line_number,
                        original_statement: line.to_string(),
                        is_used: false,
                        usage_count: 0,
                        usage_locations: Vec::new(),
                    });
                }
            }
        }
        
        None
    }
    
    /// 分类导入类型
    fn classify_import_type(&self, path: &str) -> ImportType {
        if path.starts_with("std::") || path.starts_with("core::") {
            ImportType::Standard
        } else if path.starts_with("crate::") || path.starts_with("super::") || path.starts_with("self::") {
            ImportType::Internal
        } else if path.contains("::*") {
            ImportType::Wildcard
        } else {
            ImportType::External
        }
    }
    
    /// 检查导入是否被使用
    fn check_import_usage(&self, import: &ImportItem, used_items: &HashSet<String>, content: &str) -> bool {
        if import.is_wildcard {
            // 通配符导入比较复杂，暂时标记为已使用
            return true;
        }
        
        for item in &import.items {
            if used_items.contains(item) {
                return true;
            }
            
            // 检查是否有重命名使用
            let as_pattern = format!("{}\\s+as\\s+(\\w+)", regex::escape(item));
            if let Ok(re) = Regex::new(&as_pattern) {
                if re.is_match(content) {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// 计算使用次数
    fn count_usage_occurrences(&self, import: &ImportItem, content: &str) -> usize {
        let mut count = 0;
        
        for item in &import.items {
            let pattern = format!(r"\b{}\b", regex::escape(item));
            if let Ok(re) = Regex::new(&pattern) {
                count += re.find_iter(content).count();
            }
        }
        
        count
    }
    
    /// 查找使用位置
    fn find_usage_locations(&self, import: &ImportItem, lines: &[&str]) -> Vec<usize> {
        let mut locations = Vec::new();
        
        for (line_num, line) in lines.iter().enumerate() {
            if line.trim_start().starts_with("use ") {
                continue; // 跳过导入语句
            }
            
            for item in &import.items {
                let pattern = format!(r"\b{}\b", regex::escape(item));
                if let Ok(re) = Regex::new(&pattern) {
                    if re.is_match(line) {
                        locations.push(line_num + 1);
                    }
                }
            }
        }
        
        locations
    }
    
    /// 生成优化建议
    fn generate_suggestions(&self, imports: &[ImportItem]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        let unused_count = imports.iter().filter(|i| !i.is_used).count();
        if unused_count > 0 {
            suggestions.push(format!("发现 {} 个未使用的导入，建议删除", unused_count));
        }
        
        let wildcard_count = imports.iter().filter(|i| i.is_wildcard).count();
        if wildcard_count > 0 {
            suggestions.push(format!("发现 {} 个通配符导入，建议替换为具体导入", wildcard_count));
        }
        
        // 检查重复导入
        let mut path_counts: HashMap<String, usize> = HashMap::new();
        for import in imports {
            *path_counts.entry(import.path.clone()).or_insert(0) += 1;
        }
        
        let duplicate_paths: Vec<_> = path_counts.iter()
            .filter(|(_, &count)| count > 1)
            .collect();
        
        if !duplicate_paths.is_empty() {
            suggestions.push("发现重复的导入路径，建议合并".to_string());
        }
        
        // 检查可以合并的导入
        let mut base_paths: HashMap<String, Vec<&ImportItem>> = HashMap::new();
        for import in imports {
            if let Some(base) = import.path.rfind("::") {
                let base_path = &import.path[..base];
                base_paths.entry(base_path.to_string()).or_insert_with(Vec::new).push(import);
            }
        }
        
        for (base_path, items) in base_paths {
            if items.len() > 1 && items.iter().all(|i| !i.is_wildcard) {
                suggestions.push(format!("可以将来自 {} 的多个导入合并", base_path));
            }
        }
        
        suggestions
    }
    
    /// 分析整个项目的导入
    pub fn analyze_project_imports<P: AsRef<Path>>(&self, project_root: P) -> Result<Vec<ImportAnalysisResult>> {
        let project_root = project_root.as_ref();
        let mut results = Vec::new();
        
        self.walk_rust_files(project_root, &mut |file_path| {
            match self.analyze_file_imports(file_path) {
                Ok(result) => {
                    if result.total_imports > 0 {
                        results.push(result);
                    }
                }
                Err(e) => {
                    warn!("分析文件 {:?} 失败: {}", file_path, e);
                }
            }
        })?;
        
        Ok(results)
    }
    
    /// 遍历Rust文件
    fn walk_rust_files<F>(&self, dir: &Path, callback: &mut F) -> Result<()>
    where
        F: FnMut(&Path),
    {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.is_dir() {
                    // 检查是否应该排除这个目录
                    if self.should_exclude_path(&path) {
                        continue;
                    }
                    self.walk_rust_files(&path, callback)?;
                } else if path.extension().map_or(false, |ext| ext == "rs") {
                    if !self.should_exclude_path(&path) {
                        callback(&path);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 检查路径是否应该排除
    fn should_exclude_path(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        
        for pattern in &self.config.exclude_patterns {
            if self.match_pattern(&path_str, pattern) {
                return true;
            }
        }
        
        false
    }
    
    /// 简单的模式匹配
    fn match_pattern(&self, text: &str, pattern: &str) -> bool {
        if pattern.contains('*') {
            // 简单的通配符匹配
            let parts: Vec<&str> = pattern.split('*').collect();
            if parts.len() == 2 {
                return text.starts_with(parts[0]) && text.ends_with(parts[1]);
            }
        }
        
        text == pattern
    }
    
    /// 清理未使用的导入
    pub fn cleanup_unused_imports(&self, analysis_result: &ImportAnalysisResult) -> Result<String> {
        if !self.config.auto_cleanup_unused {
            return Err(Error::invalid_input("自动清理未启用"));
        }
        
        let content = fs::read_to_string(&analysis_result.file_path)?;
        let lines: Vec<&str> = content.lines().collect();
        let mut cleaned_lines = Vec::new();
        
        for (line_num, line) in lines.iter().enumerate() {
            let should_remove = analysis_result.imports.iter().any(|import| 
                import.line_number == line_num + 1 && !import.is_used
            );
            
            if !should_remove {
                cleaned_lines.push(*line);
            } else {
                debug!("删除未使用的导入: {}", line.trim());
            }
        }
        
        Ok(cleaned_lines.join("\n"))
    }
    
    /// 转换通配符导入
    pub fn convert_wildcard_imports(&self, analysis_result: &ImportAnalysisResult, usage_info: &HashMap<String, Vec<String>>) -> Result<String> {
        if !self.config.convert_wildcard_imports {
            return Err(Error::invalid_input("通配符转换未启用"));
        }
        
        let content = fs::read_to_string(&analysis_result.file_path)?;
        let lines: Vec<&str> = content.lines().collect();
        let mut converted_lines = Vec::new();
        
        for (line_num, line) in lines.iter().enumerate() {
            let mut converted_line = line.to_string();
            
            for import in &analysis_result.imports {
                if import.line_number == line_num + 1 && import.is_wildcard {
                    // 检查是否应该保留这个通配符导入
                    if self.config.preserve_wildcard_imports.contains(&import.path) {
                        continue;
                    }
                    
                    // 查找具体使用的项目
                    if let Some(used_items) = usage_info.get(&import.path) {
                        if !used_items.is_empty() {
                            let specific_imports = used_items.join(", ");
                            converted_line = format!("use {}::{{{}}};", import.path, specific_imports);
                            debug!("转换通配符导入: {} -> {}", line.trim(), converted_line.trim());
                        }
                    }
                }
            }
            
            converted_lines.push(converted_line);
        }
        
        Ok(converted_lines.join("\n"))
    }
    
    /// 生成导入报告
    pub fn generate_import_report(&self, results: &[ImportAnalysisResult]) -> ImportReport {
        let total_files = results.len();
        let total_imports: usize = results.iter().map(|r| r.total_imports).sum();
        let total_unused: usize = results.iter().map(|r| r.unused_imports).sum();
        let total_wildcards: usize = results.iter().map(|r| r.wildcard_imports).sum();
        
        let files_with_issues: Vec<_> = results.iter()
            .filter(|r| r.unused_imports > 0 || r.wildcard_imports > 0)
            .map(|r| r.file_path.clone())
            .collect();
        
        let mut top_problematic_files: Vec<_> = results.iter()
            .filter(|r| r.unused_imports > 0 || r.wildcard_imports > 0)
            .collect();
        top_problematic_files.sort_by_key(|r| r.unused_imports + r.wildcard_imports * 2);
        top_problematic_files.reverse();
        top_problematic_files.truncate(10);
        
        ImportReport {
            total_files,
            total_imports,
            total_unused,
            total_wildcards,
            files_with_issues,
            top_problematic_files: top_problematic_files.into_iter().cloned().collect(),
            generated_at: chrono::Utc::now(),
        }
    }
    
    /// 执行完整的导入管理流程
    pub fn execute_import_management<P: AsRef<Path>>(&self, project_root: P) -> Result<ImportReport> {
        info!("开始执行导入管理流程");
        
        // 1. 分析所有文件的导入
        let analysis_results = self.analyze_project_imports(project_root)?;
        
        // 2. 生成报告
        let report = self.generate_import_report(&analysis_results);
        
        info!("导入管理流程执行完成");
        info!("{}", report);
        
        // 3. 如果启用了自动清理，执行清理操作
        if self.config.auto_cleanup_unused {
            for result in &analysis_results {
                if result.unused_imports > 0 {
                    match self.cleanup_unused_imports(result) {
                        Ok(cleaned_content) => {
                            info!("清理文件: {}", result.file_path);
                            // 这里可以选择是否写回文件
                            // fs::write(&result.file_path, cleaned_content)?;
                        }
                        Err(e) => {
                            warn!("清理文件 {} 失败: {}", result.file_path, e);
                        }
                    }
                }
            }
        }
        
        Ok(report)
    }
}

/// 导入报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportReport {
    /// 总文件数
    pub total_files: usize,
    /// 总导入数
    pub total_imports: usize,
    /// 总未使用导入数
    pub total_unused: usize,
    /// 总通配符导入数
    pub total_wildcards: usize,
    /// 有问题的文件列表
    pub files_with_issues: Vec<String>,
    /// 最有问题的文件（前10个）
    pub top_problematic_files: Vec<ImportAnalysisResult>,
    /// 生成时间
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

impl std::fmt::Display for ImportReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== 导入管理报告 ===")?;
        writeln!(f, "分析文件数: {}", self.total_files)?;
        writeln!(f, "总导入数: {}", self.total_imports)?;
        writeln!(f, "未使用导入数: {} ({:.1}%)", 
                self.total_unused, 
                self.total_unused as f64 / self.total_imports as f64 * 100.0)?;
        writeln!(f, "通配符导入数: {} ({:.1}%)", 
                self.total_wildcards,
                self.total_wildcards as f64 / self.total_imports as f64 * 100.0)?;
        writeln!(f, "有问题的文件数: {}", self.files_with_issues.len())?;
        writeln!(f, "")?;
        
        if !self.top_problematic_files.is_empty() {
            writeln!(f, "最需要优化的文件:")?;
            for (i, file_result) in self.top_problematic_files.iter().take(5).enumerate() {
                writeln!(f, "  {}. {} (未使用: {}, 通配符: {})", 
                        i + 1, 
                        file_result.file_path,
                        file_result.unused_imports,
                        file_result.wildcard_imports)?;
            }
        }
        
        writeln!(f, "")?;
        writeln!(f, "生成时间: {}", self.generated_at.format("%Y-%m-%d %H:%M:%S UTC"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;
    
    #[test]
    fn test_import_parsing() {
        let config = ImportManagerConfig::default();
        let manager = ImportManager::new(config).unwrap();
        
        // 测试通配符导入
        let wildcard_import = manager.parse_import_statement("use std::collections::*;", 1);
        assert!(wildcard_import.is_some());
        assert!(wildcard_import.unwrap().is_wildcard);
        
        // 测试选择性导入
        let selective_import = manager.parse_import_statement("use std::collections::{HashMap, HashSet};", 2);
        assert!(selective_import.is_some());
        let import = selective_import.unwrap();
        assert!(!import.is_wildcard);
        assert_eq!(import.items.len(), 2);
    }
    
    #[test]
    fn test_import_analysis() {
        let config = ImportManagerConfig::default();
        let manager = ImportManager::new(config).unwrap();
        
        // 创建临时文件
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.rs");
        
        let content = r#"
use std::collections::HashMap;
use std::vec::Vec;
use unused_module::SomeType;

fn main() {
    let map = HashMap::new();
    let vec = Vec::new();
}
"#;
        
        fs::write(&file_path, content).unwrap();
        
        let result = manager.analyze_file_imports(&file_path).unwrap();
        assert_eq!(result.total_imports, 3);
        assert_eq!(result.unused_imports, 1); // unused_module::SomeType
    }
} 