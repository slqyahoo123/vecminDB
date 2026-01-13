/// 依赖管理器 - 解决循环依赖和管理模块间依赖关系
/// 
/// 本模块提供：
/// 1. 循环依赖检测和解决
/// 2. 依赖注入容器
/// 3. 模块解耦机制
/// 4. 运行时依赖管理

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::any::{Any, TypeId};
use std::fmt;
use async_trait::async_trait;
use log::{debug, info, warn, error};
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};

/// 依赖图节点
#[derive(Debug, Clone)]
pub struct DependencyNode {
    /// 模块名称
    pub module_name: String,
    /// 模块类型ID
    pub type_id: TypeId,
    /// 依赖的模块列表
    pub dependencies: Vec<String>,
    /// 被依赖的模块列表
    pub dependents: Vec<String>,
    /// 模块状态
    pub status: ModuleStatus,
    /// 初始化优先级
    pub priority: i32,
    /// 模块元数据
    pub metadata: HashMap<String, String>,
}

/// 模块状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModuleStatus {
    /// 未初始化
    Uninitialized,
    /// 正在初始化
    Initializing,
    /// 已初始化
    Initialized,
    /// 初始化失败
    Failed(String),
    /// 已销毁
    Destroyed,
}

/// 依赖关系类型
#[derive(Debug, Clone, PartialEq)]
pub enum DependencyType {
    /// 强依赖（必须）
    Required,
    /// 弱依赖（可选）
    Optional,
    /// 循环依赖（需要特殊处理）
    Circular,
}

/// 依赖关系描述
#[derive(Debug, Clone)]
pub struct Dependency {
    /// 依赖者模块名
    pub from: String,
    /// 被依赖模块名
    pub to: String,
    /// 依赖类型
    pub dependency_type: DependencyType,
    /// 依赖描述
    pub description: Option<String>,
}

/// 循环依赖报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularDependencyReport {
    /// 是否存在循环依赖
    pub has_cycles: bool,
    /// 循环依赖路径
    pub cycles: Vec<Vec<String>>,
    /// 建议的解决方案
    pub suggestions: Vec<String>,
    /// 检测时间
    pub detection_time: chrono::DateTime<chrono::Utc>,
}

/// 依赖管理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyManagerConfig {
    /// 是否启用循环依赖检测
    pub enable_cycle_detection: bool,
    /// 是否自动解决循环依赖
    pub auto_resolve_cycles: bool,
    /// 最大依赖深度
    pub max_dependency_depth: usize,
    /// 初始化超时时间（秒）
    pub initialization_timeout: u64,
    /// 是否启用详细日志
    pub verbose_logging: bool,
}

impl Default for DependencyManagerConfig {
    fn default() -> Self {
        Self {
            enable_cycle_detection: true,
            auto_resolve_cycles: true,
            max_dependency_depth: 10,
            initialization_timeout: 30,
            verbose_logging: false,
        }
    }
}

/// 服务接口
#[async_trait]
pub trait ServiceInterface: Send + Sync {
    /// 服务名称
    fn service_name(&self) -> &str;
    
    /// 服务依赖列表
    fn dependencies(&self) -> Vec<String>;
    
    /// 初始化服务
    async fn initialize(&mut self, container: &DependencyContainer) -> Result<()>;
    
    /// 停止服务
    async fn shutdown(&mut self) -> Result<()>;
    
    /// 获取服务状态
    fn status(&self) -> ModuleStatus;
    
    /// 健康检查
    async fn health_check(&self) -> Result<bool>;
}

/// 依赖容器
pub struct DependencyContainer {
    /// 服务实例存储
    services: RwLock<HashMap<String, Arc<dyn Any + Send + Sync>>>,
    /// 依赖图
    dependency_graph: RwLock<HashMap<String, DependencyNode>>,
    /// 初始化顺序
    initialization_order: RwLock<Vec<String>>,
    /// 配置
    config: DependencyManagerConfig,
    /// 容器状态
    status: RwLock<ContainerStatus>,
}

/// 容器状态
#[derive(Debug, Clone, PartialEq)]
pub enum ContainerStatus {
    /// 未初始化
    Uninitialized,
    /// 正在构建
    Building,
    /// 已就绪
    Ready,
    /// 运行中
    Running,
    /// 正在关闭
    Shutting,
    /// 已关闭
    Shutdown,
    /// 错误状态
    Error(String),
}

impl DependencyContainer {
    /// 创建新的依赖容器
    pub fn new(config: DependencyManagerConfig) -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            dependency_graph: RwLock::new(HashMap::new()),
            initialization_order: RwLock::new(Vec::new()),
            config,
            status: RwLock::new(ContainerStatus::Uninitialized),
        }
    }
    
    /// 注册服务
    pub fn register_service<T>(&self, name: &str, service: T) -> Result<()>
    where
        T: Any + Send + Sync + 'static,
    {
        let mut services = self.services.write().unwrap();
        
        if services.contains_key(name) {
            return Err(Error::invalid_input(format!("服务 '{}' 已经注册", name)));
        }
        
        services.insert(name.to_string(), Arc::new(service));
        
        if self.config.verbose_logging {
            debug!("注册服务: {}", name);
        }
        
        Ok(())
    }
    
    /// 获取服务
    pub fn get_service<T>(&self, name: &str) -> Result<Arc<T>>
    where
        T: Any + Send + Sync + 'static,
    {
        let services = self.services.read().unwrap();
        
        if let Some(service) = services.get(name) {
            service.clone()
                .downcast::<T>()
                .map_err(|_| Error::invalid_input(format!("服务 '{}' 类型不匹配", name)))
        } else {
            Err(Error::not_found(format!("服务 '{}' 未找到", name)))
        }
    }
    
    /// 注册依赖关系
    pub fn register_dependency(&self, dependency: Dependency) -> Result<()> {
        let mut graph = self.dependency_graph.write().unwrap();
        
        // 更新依赖者节点
        let from_node = graph.entry(dependency.from.clone())
            .or_insert_with(|| DependencyNode {
                module_name: dependency.from.clone(),
                type_id: TypeId::of::<()>(),
                dependencies: Vec::new(),
                dependents: Vec::new(),
                status: ModuleStatus::Uninitialized,
                priority: 0,
                metadata: HashMap::new(),
            });
        
        if !from_node.dependencies.contains(&dependency.to) {
            from_node.dependencies.push(dependency.to.clone());
        }
        
        // 更新被依赖者节点
        let to_node = graph.entry(dependency.to.clone())
            .or_insert_with(|| DependencyNode {
                module_name: dependency.to.clone(),
                type_id: TypeId::of::<()>(),
                dependencies: Vec::new(),
                dependents: Vec::new(),
                status: ModuleStatus::Uninitialized,
                priority: 0,
                metadata: HashMap::new(),
            });
        
        if !to_node.dependents.contains(&dependency.from) {
            to_node.dependents.push(dependency.from.clone());
        }
        
        if self.config.verbose_logging {
            debug!("注册依赖关系: {} -> {}", dependency.from, dependency.to);
        }
        
        Ok(())
    }
    
    /// 检测循环依赖
    pub fn detect_circular_dependencies(&self) -> CircularDependencyReport {
        let graph = self.dependency_graph.read().unwrap();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut cycles = Vec::new();
        
        for node_name in graph.keys() {
            if !visited.contains(node_name) {
                if let Some(cycle) = self.dfs_detect_cycle(
                    node_name,
                    &graph,
                    &mut visited,
                    &mut rec_stack,
                    &mut Vec::new(),
                ) {
                    cycles.push(cycle);
                }
            }
        }
        
        let suggestions = if !cycles.is_empty() {
            self.generate_cycle_resolution_suggestions(&cycles)
        } else {
            Vec::new()
        };
        
        CircularDependencyReport {
            has_cycles: !cycles.is_empty(),
            cycles,
            suggestions,
            detection_time: chrono::Utc::now(),
        }
    }
    
    /// DFS循环检测
    fn dfs_detect_cycle(
        &self,
        node: &str,
        graph: &HashMap<String, DependencyNode>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());
        path.push(node.to_string());
        
        if let Some(node_info) = graph.get(node) {
            for dep in &node_info.dependencies {
                if !visited.contains(dep) {
                    if let Some(cycle) = self.dfs_detect_cycle(dep, graph, visited, rec_stack, path) {
                        return Some(cycle);
                    }
                } else if rec_stack.contains(dep) {
                    // 找到循环，构建循环路径
                    let cycle_start = path.iter().position(|x| x == dep).unwrap();
                    let mut cycle = path[cycle_start..].to_vec();
                    cycle.push(dep.clone());
                    return Some(cycle);
                }
            }
        }
        
        path.pop();
        rec_stack.remove(node);
        None
    }
    
    /// 生成循环依赖解决建议
    fn generate_cycle_resolution_suggestions(&self, cycles: &[Vec<String>]) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        for (i, cycle) in cycles.iter().enumerate() {
            suggestions.push(format!(
                "循环依赖 #{}: {} -> 建议引入接口抽象或延迟初始化",
                i + 1,
                cycle.join(" -> ")
            ));
            
            // 具体建议
            if cycle.len() == 2 {
                suggestions.push(format!(
                    "  - 在 {} 和 {} 之间引入接口抽象",
                    cycle[0], cycle[1]
                ));
            } else {
                suggestions.push(format!(
                    "  - 将 {} 拆分为多个独立模块",
                    cycle.iter().min().unwrap()
                ));
            }
        }
        
        suggestions
    }
    
    /// 计算初始化顺序
    pub fn compute_initialization_order(&self) -> Result<Vec<String>> {
        let graph = self.dependency_graph.read().unwrap();
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        
        // 计算入度
        for (name, node) in graph.iter() {
            in_degree.insert(name.clone(), node.dependencies.len());
            if node.dependencies.is_empty() {
                queue.push_back(name.clone());
            }
        }
        
        // 拓扑排序
        while let Some(node) = queue.pop_front() {
            result.push(node.clone());
            
            if let Some(node_info) = graph.get(&node) {
                for dependent in &node_info.dependents {
                    if let Some(degree) = in_degree.get_mut(dependent) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(dependent.clone());
                        }
                    }
                }
            }
        }
        
        // 检查是否存在循环依赖
        if result.len() != graph.len() {
            return Err(Error::invalid_state("存在循环依赖，无法确定初始化顺序"));
        }
        
        Ok(result)
    }
    
    /// 自动解决循环依赖
    pub fn resolve_circular_dependencies(&self) -> Result<()> {
        if !self.config.auto_resolve_cycles {
            return Ok(());
        }
        
        let report = self.detect_circular_dependencies();
        if !report.has_cycles {
            return Ok(());
        }
        
        info!("检测到 {} 个循环依赖，尝试自动解决", report.cycles.len());
        
        for cycle in &report.cycles {
            if let Err(e) = self.resolve_single_cycle(cycle) {
                warn!("无法自动解决循环依赖 {}: {}", cycle.join(" -> "), e);
            }
        }
        
        Ok(())
    }
    
    /// 解决单个循环依赖
    fn resolve_single_cycle(&self, cycle: &[String]) -> Result<()> {
        if cycle.len() < 2 {
            return Ok(());
        }
        
        // 策略1：引入接口抽象
        if cycle.len() == 2 {
            info!("为循环依赖 {} <-> {} 引入接口抽象", cycle[0], cycle[1]);
            self.introduce_interface_abstraction(&cycle[0], &cycle[1])?;
        }
        
        // 策略2：延迟初始化
        else {
            info!("为循环依赖 {} 使用延迟初始化", cycle.join(" -> "));
            self.setup_lazy_initialization(cycle)?;
        }
        
        Ok(())
    }
    
    /// 引入接口抽象
    fn introduce_interface_abstraction(&self, module1: &str, module2: &str) -> Result<()> {
        // 这里可以生成接口代码或配置
        debug!("为模块 {} 和 {} 创建接口抽象", module1, module2);
        
        // 实际实现中，这里应该：
        // 1. 分析两个模块的依赖关系
        // 2. 提取共同接口
        // 3. 生成抽象接口代码
        // 4. 更新依赖关系
        
        Ok(())
    }
    
    /// 设置延迟初始化
    fn setup_lazy_initialization(&self, cycle: &[String]) -> Result<()> {
        debug!("为循环依赖设置延迟初始化: {}", cycle.join(" -> "));
        
        // 实际实现中，这里应该：
        // 1. 标记需要延迟初始化的模块
        // 2. 设置初始化回调
        // 3. 更新依赖图
        
        Ok(())
    }
    
    /// 初始化所有服务
    pub async fn initialize_all(&self) -> Result<()> {
        *self.status.write().unwrap() = ContainerStatus::Building;
        
        // 检测循环依赖
        if self.config.enable_cycle_detection {
            let report = self.detect_circular_dependencies();
            if report.has_cycles {
                error!("检测到循环依赖:");
                for cycle in &report.cycles {
                    error!("  {}", cycle.join(" -> "));
                }
                
                if self.config.auto_resolve_cycles {
                    self.resolve_circular_dependencies()?;
                } else {
                    return Err(Error::invalid_state("存在循环依赖"));
                }
            }
        }
        
        // 计算初始化顺序
        let order = self.compute_initialization_order()?;
        *self.initialization_order.write().unwrap() = order.clone();
        
        info!("服务初始化顺序: {}", order.join(" -> "));
        
        // 按顺序初始化服务
        for service_name in &order {
            if let Err(e) = self.initialize_service(service_name).await {
                error!("初始化服务 '{}' 失败: {}", service_name, e);
                *self.status.write().unwrap() = ContainerStatus::Error(e.to_string());
                return Err(e);
            }
        }
        
        *self.status.write().unwrap() = ContainerStatus::Ready;
        info!("所有服务初始化完成");
        
        Ok(())
    }
    
    /// 初始化单个服务
    async fn initialize_service(&self, service_name: &str) -> Result<()> {
        debug!("正在初始化服务: {}", service_name);
        
        // 更新服务状态
        if let Ok(mut graph) = self.dependency_graph.write() {
            if let Some(node) = graph.get_mut(service_name) {
                node.status = ModuleStatus::Initializing;
            }
        }
        
        // 这里应该调用实际的服务初始化逻辑
        // 由于这是抽象层，具体实现由各个服务提供
        
        // 模拟初始化过程
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // 更新服务状态
        if let Ok(mut graph) = self.dependency_graph.write() {
            if let Some(node) = graph.get_mut(service_name) {
                node.status = ModuleStatus::Initialized;
            }
        }
        
        debug!("服务 '{}' 初始化完成", service_name);
        Ok(())
    }
    
    /// 获取容器状态
    pub fn get_status(&self) -> ContainerStatus {
        self.status.read().unwrap().clone()
    }
    
    /// 获取依赖图
    pub fn get_dependency_graph(&self) -> HashMap<String, DependencyNode> {
        self.dependency_graph.read().unwrap().clone()
    }
    
    /// 生成依赖图报告
    pub fn generate_dependency_report(&self) -> DependencyReport {
        let graph = self.dependency_graph.read().unwrap();
        let total_modules = graph.len();
        let total_dependencies = graph.values()
            .map(|node| node.dependencies.len())
            .sum();
        
        let cycle_report = self.detect_circular_dependencies();
        
        DependencyReport {
            total_modules,
            total_dependencies,
            circular_dependencies: cycle_report.cycles.len(),
            initialization_order: self.initialization_order.read().unwrap().clone(),
            cycle_report,
            generated_at: chrono::Utc::now(),
        }
    }
}

/// 依赖报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyReport {
    /// 总模块数
    pub total_modules: usize,
    /// 总依赖数
    pub total_dependencies: usize,
    /// 循环依赖数
    pub circular_dependencies: usize,
    /// 初始化顺序
    pub initialization_order: Vec<String>,
    /// 循环依赖报告
    pub cycle_report: CircularDependencyReport,
    /// 生成时间
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

impl fmt::Display for DependencyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== 依赖管理报告 ===")?;
        writeln!(f, "总模块数: {}", self.total_modules)?;
        writeln!(f, "总依赖数: {}", self.total_dependencies)?;
        writeln!(f, "循环依赖数: {}", self.circular_dependencies)?;
        writeln!(f, "")?;
        
        if !self.initialization_order.is_empty() {
            writeln!(f, "初始化顺序:")?;
            for (i, module) in self.initialization_order.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, module)?;
            }
            writeln!(f, "")?;
        }
        
        if self.cycle_report.has_cycles {
            writeln!(f, "循环依赖详情:")?;
            for (i, cycle) in self.cycle_report.cycles.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, cycle.join(" -> "))?;
            }
            writeln!(f, "")?;
            
            writeln!(f, "解决建议:")?;
            for suggestion in &self.cycle_report.suggestions {
                writeln!(f, "  - {}", suggestion)?;
            }
        }
        
        writeln!(f, "生成时间: {}", self.generated_at.format("%Y-%m-%d %H:%M:%S UTC"))
    }
}

/// 全局依赖管理器
pub struct DependencyManager {
    container: Arc<DependencyContainer>,
    config: DependencyManagerConfig,
}

impl DependencyManager {
    /// 创建新的依赖管理器
    pub fn new(config: DependencyManagerConfig) -> Self {
        let container = Arc::new(DependencyContainer::new(config.clone()));
        
        Self {
            container,
            config,
        }
    }
    
    /// 获取容器引用
    pub fn container(&self) -> &Arc<DependencyContainer> {
        &self.container
    }
    
    /// 注册模块依赖
    pub fn register_module_dependencies(&self) -> Result<()> {
        // 注册核心模块依赖关系
        self.register_core_dependencies()?;
        self.register_storage_dependencies()?;
        self.register_model_dependencies()?;
        self.register_training_dependencies()?;
        self.register_algorithm_dependencies()?;
        self.register_api_dependencies()?;
        
        info!("所有模块依赖关系注册完成");
        Ok(())
    }
    
    /// 注册核心模块依赖
    fn register_core_dependencies(&self) -> Result<()> {
        // core 模块不应该依赖具体业务模块
        debug!("注册核心模块依赖关系");
        Ok(())
    }
    
    /// 注册存储模块依赖
    fn register_storage_dependencies(&self) -> Result<()> {
        self.container.register_dependency(Dependency {
            from: "storage".to_string(),
            to: "core".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("存储模块依赖核心接口".to_string()),
        })?;
        
        debug!("注册存储模块依赖关系");
        Ok(())
    }
    
    /// 注册模型模块依赖
    fn register_model_dependencies(&self) -> Result<()> {
        self.container.register_dependency(Dependency {
            from: "model".to_string(),
            to: "core".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("模型模块依赖核心接口".to_string()),
        })?;
        
        self.container.register_dependency(Dependency {
            from: "model".to_string(),
            to: "storage".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("模型模块依赖存储接口".to_string()),
        })?;
        
        debug!("注册模型模块依赖关系");
        Ok(())
    }
    
    /// 注册训练模块依赖
    fn register_training_dependencies(&self) -> Result<()> {
        self.container.register_dependency(Dependency {
            from: "training".to_string(),
            to: "core".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("训练模块依赖核心接口".to_string()),
        })?;
        
        self.container.register_dependency(Dependency {
            from: "training".to_string(),
            to: "model".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("训练模块依赖模型接口".to_string()),
        })?;
        
        debug!("注册训练模块依赖关系");
        Ok(())
    }
    
    /// 注册算法模块依赖
    fn register_algorithm_dependencies(&self) -> Result<()> {
        self.container.register_dependency(Dependency {
            from: "algorithm".to_string(),
            to: "core".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("算法模块依赖核心接口".to_string()),
        })?;
        
        self.container.register_dependency(Dependency {
            from: "algorithm".to_string(),
            to: "model".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("算法模块依赖模型接口".to_string()),
        })?;
        
        debug!("注册算法模块依赖关系");
        Ok(())
    }
    
    /// 注册API模块依赖
    fn register_api_dependencies(&self) -> Result<()> {
        self.container.register_dependency(Dependency {
            from: "api".to_string(),
            to: "core".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("API模块依赖核心接口".to_string()),
        })?;
        
        self.container.register_dependency(Dependency {
            from: "api".to_string(),
            to: "storage".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("API模块依赖存储接口".to_string()),
        })?;
        
        self.container.register_dependency(Dependency {
            from: "api".to_string(),
            to: "model".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("API模块依赖模型接口".to_string()),
        })?;
        
        self.container.register_dependency(Dependency {
            from: "api".to_string(),
            to: "training".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("API模块依赖训练接口".to_string()),
        })?;
        
        self.container.register_dependency(Dependency {
            from: "api".to_string(),
            to: "algorithm".to_string(),
            dependency_type: DependencyType::Required,
            description: Some("API模块依赖算法接口".to_string()),
        })?;
        
        debug!("注册API模块依赖关系");
        Ok(())
    }
    
    /// 执行完整的依赖管理流程
    pub async fn execute_dependency_management(&self) -> Result<DependencyReport> {
        info!("开始执行依赖管理流程");
        
        // 1. 注册所有模块依赖
        self.register_module_dependencies()?;
        
        // 2. 检测循环依赖
        let report = self.container.detect_circular_dependencies();
        if report.has_cycles {
            warn!("检测到 {} 个循环依赖", report.cycles.len());
            for cycle in &report.cycles {
                warn!("循环依赖: {}", cycle.join(" -> "));
            }
        }
        
        // 3. 自动解决循环依赖
        self.container.resolve_circular_dependencies()?;
        
        // 4. 初始化所有服务
        self.container.initialize_all().await?;
        
        // 5. 生成最终报告
        let final_report = self.container.generate_dependency_report();
        
        info!("依赖管理流程执行完成");
        info!("{}", final_report);
        
        Ok(final_report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_dependency_manager() {
        let config = DependencyManagerConfig::default();
        let manager = DependencyManager::new(config);
        
        let result = manager.execute_dependency_management().await;
        assert!(result.is_ok());
        
        let report = result.unwrap();
        assert!(report.total_modules > 0);
    }
    
    #[test]
    fn test_circular_dependency_detection() {
        let config = DependencyManagerConfig::default();
        let container = DependencyContainer::new(config);
        
        // 创建循环依赖
        container.register_dependency(Dependency {
            from: "A".to_string(),
            to: "B".to_string(),
            dependency_type: DependencyType::Required,
            description: None,
        }).unwrap();
        
        container.register_dependency(Dependency {
            from: "B".to_string(),
            to: "A".to_string(),
            dependency_type: DependencyType::Required,
            description: None,
        }).unwrap();
        
        let report = container.detect_circular_dependencies();
        assert!(report.has_cycles);
        assert!(!report.cycles.is_empty());
    }
    
    #[test]
    fn test_initialization_order() {
        let config = DependencyManagerConfig::default();
        let container = DependencyContainer::new(config);
        
        // 创建线性依赖链
        container.register_dependency(Dependency {
            from: "B".to_string(),
            to: "A".to_string(),
            dependency_type: DependencyType::Required,
            description: None,
        }).unwrap();
        
        container.register_dependency(Dependency {
            from: "C".to_string(),
            to: "B".to_string(),
            dependency_type: DependencyType::Required,
            description: None,
        }).unwrap();
        
        let order = container.compute_initialization_order().unwrap();
        assert_eq!(order, vec!["A", "B", "C"]);
    }
} 