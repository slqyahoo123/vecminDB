//! DSL解析模块
//!
//! 提供算法DSL（领域特定语言）解析和执行功能

use crate::error::{Error, Result};
// TODO: wasm模块中未定义WasmInstruction，暂时注释相关导入
// use crate::algorithm::wasm::WasmInstruction;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use log::{debug, warn};
use serde_json;

/// WASM指令
#[derive(Debug, Clone)]
pub enum WasmInstruction {
    /// 注释
    Comment(String),
    /// i32常量
    I32Const(i32),
    /// i32加载
    I32Load(u32),
    /// i32存储
    I32Store(u32),
    /// i32加法
    I32Add,
    /// i32减法
    I32Sub,
    /// i32乘法
    I32Mul,
    /// i32除法
    I32Div,
    /// 函数调用
    Call(String),
    /// 返回
    Return,
}

/// DSL配置
#[derive(Debug, Clone)]
pub struct DslConfig {
    /// 最大操作数
    pub max_operations: usize,
    /// 最大矩阵操作数
    pub max_matrix_operations: usize,
    /// 黑名单导入
    pub blacklisted_imports: Vec<String>,
}

impl Default for DslConfig {
    fn default() -> Self {
        Self {
            max_operations: 1000,
            max_matrix_operations: 100,
            blacklisted_imports: vec![
                "system".to_string(),
                "fs".to_string(),
                "net".to_string(),
                "process".to_string(),
            ],
        }
    }
}

/// 节点类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// 输入节点
    Input,
    /// 输出节点
    Output,
    /// 操作节点
    Operation,
    /// 组合节点，用于表示一个节点既是输入又是输出，或具有多种功能
    Combined,
}

/// 激活函数类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// ReLU激活函数
    ReLU,
    /// Sigmoid激活函数
    Sigmoid,
    /// Tanh激活函数
    Tanh,
    /// Softmax激活函数
    Softmax,
}

/// 操作类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    /// 加法
    Add,
    /// 减法
    Subtract,
    /// 乘法
    Multiply,
    /// 除法
    Divide,
    /// 矩阵乘法
    MatMul,
    /// 激活函数
    Activation(Activation),
    /// 转置
    Transpose,
    /// 重塑
    Reshape,
    /// 连接
    Concat,
    /// 切片
    Slice,
    /// 自定义操作
    Custom(String),
}

/// 算法节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmNode {
    /// 节点ID
    pub id: usize,
    /// 节点类型
    pub node_type: NodeType,
    /// 节点名称
    pub name: String,
    /// 输入节点ID
    pub inputs: Vec<usize>,
    /// 操作类型
    pub operation: Option<Operation>,
    /// 参数
    pub parameters: HashMap<String, String>,
}

/// 算法AST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmAst {
    /// 节点列表
    pub nodes: Vec<AlgorithmNode>,
}

/// DSL解析器
#[derive(Debug)]
pub struct DslParser {
    /// 配置
    config: DslConfig,
}

impl DslParser {
    /// 创建新的解析器
    pub fn new(config: DslConfig) -> Self {
        Self { config }
    }
    
    /// 使用默认配置创建解析器
    pub fn default() -> Self {
        Self { config: DslConfig::default() }
    }
    
    /// 解析DSL代码为AST
    pub fn parse_dsl_to_ast(&self, dsl_code: &str) -> Result<AlgorithmAst> {
        // 创建基本AST结构
        let mut nodes = Vec::new();
        let mut current_node_id = 0;
        
        // 简单解析DSL代码的每一行
        for line in dsl_code.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue; // 跳过空行和注释
            }
            
            // 简单解析，实际中需要更复杂的语法分析
            if let Some(node) = self.parse_dsl_line(line, current_node_id)? {
                nodes.push(node);
                current_node_id += 1;
            }
        }
        
        if nodes.is_empty() {
            return Err(Error::algorithm("无法解析算法DSL代码，结果为空"));
        }
        
        Ok(AlgorithmAst { nodes })
    }
    
    /// 解析单行DSL
    fn parse_dsl_line(&self, line: &str, id: usize) -> Result<Option<AlgorithmNode>> {
        // 简化的行解析逻辑
        if line.starts_with("input") {
            // 解析输入节点
            let parts: Vec<&str> = line.splitn(3, ' ').collect();
            if parts.len() < 2 {
                return Err(Error::algorithm(format!("无效的输入定义: {}", line)));
            }
            let name = parts[1].trim_matches(|c| c == ' ' || c == ':' || c == ';');
            return Ok(Some(AlgorithmNode {
                id,
                node_type: NodeType::Input,
                name: name.to_string(),
                inputs: Vec::new(),
                operation: None,
                parameters: HashMap::new(),
            }));
        } else if line.starts_with("output") {
            // 解析输出节点
            let parts: Vec<&str> = line.splitn(3, ' ').collect();
            if parts.len() < 2 {
                return Err(Error::algorithm(format!("无效的输出定义: {}", line)));
            }
            let name = parts[1].trim_matches(|c| c == ' ' || c == ':' || c == ';');
            return Ok(Some(AlgorithmNode {
                id,
                node_type: NodeType::Output,
                name: name.to_string(),
                inputs: Vec::new(),
                operation: None,
                parameters: HashMap::new(),
            }));
        } else if line.contains("=") {
            // 解析操作节点
            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() != 2 {
                return Err(Error::algorithm(format!("无效的操作定义: {}", line)));
            }
            
            let name = parts[0].trim();
            let operation_str = parts[1].trim_matches(|c| c == ' ' || c == ';');
            
            // 简单操作解析
            if let Some(op) = self.parse_operation(operation_str)? {
                let mut parameters = HashMap::new();
                
                // 提取参数，简化处理
                if operation_str.contains("(") && operation_str.contains(")") {
                    let params_start = operation_str.find('(').unwrap();
                    let params_end = operation_str.rfind(')').unwrap();
                    let params_str = &operation_str[params_start+1..params_end];
                    
                    for param in params_str.split(',') {
                        if param.contains(":") {
                            let param_parts: Vec<&str> = param.splitn(2, ':').collect();
                            if param_parts.len() == 2 {
                                let key = param_parts[0].trim();
                                let value = param_parts[1].trim();
                                parameters.insert(key.to_string(), value.to_string());
                            }
                        }
                    }
                }
                
                return Ok(Some(AlgorithmNode {
                    id,
                    node_type: NodeType::Operation,
                    name: name.to_string(),
                    inputs: Vec::new(), // 输入关系需要后续处理
                    operation: Some(op),
                    parameters,
                }));
            }
        }
        
        // 无法识别的行
        Ok(None)
    }
    
    /// 解析操作字符串
    fn parse_operation(&self, operation_str: &str) -> Result<Option<Operation>> {
        let op_name = if let Some(idx) = operation_str.find('(') {
            operation_str[..idx].trim()
        } else {
            operation_str.trim()
        };
        
        match op_name {
            "add" => Ok(Some(Operation::Add)),
            "subtract" => Ok(Some(Operation::Subtract)),
            "multiply" => Ok(Some(Operation::Multiply)),
            "divide" => Ok(Some(Operation::Divide)),
            "matmul" => Ok(Some(Operation::MatMul)),
            "relu" => Ok(Some(Operation::Activation(Activation::ReLU))),
            "sigmoid" => Ok(Some(Operation::Activation(Activation::Sigmoid))),
            "tanh" => Ok(Some(Operation::Activation(Activation::Tanh))),
            "softmax" => Ok(Some(Operation::Activation(Activation::Softmax))),
            "transpose" => Ok(Some(Operation::Transpose)),
            "reshape" => Ok(Some(Operation::Reshape)),
            "concat" => Ok(Some(Operation::Concat)),
            "slice" => Ok(Some(Operation::Slice)),
            "custom" => Ok(Some(Operation::Custom("custom".to_string()))),
            _ => {
                if op_name.starts_with("custom_") {
                    Ok(Some(Operation::Custom(op_name.to_string())))
                } else {
                    Err(Error::algorithm(format!("未知操作: {}", op_name)))
                }
            }
        }
    }
    
    /// 验证AST是否符合安全规则
    pub fn validate_ast(&self, ast: &AlgorithmAst) -> Result<()> {
        // 1. 检查是否有输入和输出节点
        let has_input = ast.nodes.iter().any(|node| node.node_type == NodeType::Input);
        let has_output = ast.nodes.iter().any(|node| node.node_type == NodeType::Output);
        
        if !has_input || !has_output {
            return Err(Error::algorithm("算法必须至少有一个输入节点和一个输出节点"));
        }
        
        // 2. 检查图中是否有循环
        // 简化实现，更完整的检查应使用图算法
        if self.has_cycles(ast) {
            return Err(Error::algorithm("算法图中检测到循环依赖"));
        }
        
        // 3. 检查操作安全性
        for node in &ast.nodes {
            if let Some(op) = &node.operation {
                self.validate_operation(op)?;
            }
        }
        
        // 4. 检查资源使用
        self.check_resource_usage(ast)?;
        
        Ok(())
    }
    
    /// 验证操作是否安全
    fn validate_operation(&self, operation: &Operation) -> Result<()> {
        match operation {
            Operation::Custom(name) => {
                // 检查自定义操作的安全性
                if self.is_operation_blacklisted(name) {
                    return Err(Error::security(format!("不允许使用的操作: {}", name)));
                }
            },
            Operation::MatMul => {
                // 矩阵乘法是资源密集型操作，可能需要特殊检查
                debug!("检测到矩阵乘法操作，注意性能影响");
            },
            _ => {
                // 其他操作通常是安全的
            }
        }
        
        Ok(())
    }
    
    /// 检查操作是否在黑名单中
    fn is_operation_blacklisted(&self, name: &str) -> bool {
        self.config.blacklisted_imports.iter().any(|blocked| name.contains(blocked))
    }
    
    /// 检查资源使用情况
    fn check_resource_usage(&self, ast: &AlgorithmAst) -> Result<()> {
        // 按类型计算操作数
        let mut op_counts = HashMap::new();
        
        for node in &ast.nodes {
            if let Some(op) = &node.operation {
                let type_name = format!("{:?}", op);
                *op_counts.entry(type_name).or_insert(0) += 1;
            }
        }
        
        // 检查操作数是否过多
        let total_ops = ast.nodes.len();
        if total_ops > self.config.max_operations {
            return Err(Error::algorithm(format!(
                "操作数超出限制: {} 操作，限制为 {}",
                total_ops,
                self.config.max_operations
            )));
        }
        
        // 检查潜在的昂贵操作
        if let Some(matmul_count) = op_counts.get("MatMul") {
            if *matmul_count > self.config.max_matrix_operations {
                return Err(Error::algorithm(format!(
                    "矩阵乘法次数过多: {}，限制为 {}",
                    matmul_count,
                    self.config.max_matrix_operations
                )));
            }
        }
        
        Ok(())
    }
    
    /// 检查AST中是否有循环
    fn has_cycles(&self, ast: &AlgorithmAst) -> bool {
        // 简化实现，实际应使用图算法如深度优先遍历检测循环
        // 这个简化版本只检查直接的自我引用
        for node in &ast.nodes {
            if node.inputs.contains(&node.id) {
                return true;
            }
        }
        
        false
    }
    
    /// 从AST生成WASM代码
    pub fn generate_wasm_from_ast(&self, ast: &AlgorithmAst) -> Result<Vec<u8>> {
        // 将AST转换为WASM指令
        let instructions = self.convert_ast_to_instructions(ast)?;
        
        // 编译指令为WASM二进制
        let wasm_binary = self.compile_instructions(instructions)?;
        
        // 验证生成的WASM
        self.validate_generated_wasm(&wasm_binary)?;
        
        Ok(wasm_binary)
    }
    
    /// 将AST转换为WASM指令
    fn convert_ast_to_instructions(&self, ast: &AlgorithmAst) -> Result<Vec<WasmInstruction>> {
        let mut instructions = Vec::new();
        let mut node_outputs = HashMap::new();
        
        // 添加模块注释
        instructions.push(WasmInstruction::Comment("生成的WASM模块".to_string()));
        instructions.push(WasmInstruction::Comment("从算法DSL自动转换".to_string()));
        
        // 进行拓扑排序
        let sorted_nodes = self.topological_sort(ast)?;
        
        // 生成每个节点的指令
        for node_id in sorted_nodes {
            if let Some(node) = ast.nodes.iter().find(|n| n.id == node_id) {
                self.generate_node_instructions(node, &mut instructions)?;
            }
        }
        
        // 添加返回指令
        instructions.push(WasmInstruction::Return);
        
        Ok(instructions)
    }
    
    /// 拓扑排序
    fn topological_sort(&self, ast: &AlgorithmAst) -> Result<Vec<usize>> {
        // 简化实现，仅按ID排序
        // 实际应该实现完整的拓扑排序算法
        let mut node_ids: Vec<usize> = ast.nodes.iter().map(|n| n.id).collect();
        node_ids.sort();
        Ok(node_ids)
    }
    
    /// 为节点生成指令
    fn generate_node_instructions(&self, node: &AlgorithmNode, instructions: &mut Vec<WasmInstruction>) -> Result<()> {
        // 添加注释
        instructions.push(WasmInstruction::Comment(format!("节点: {} ({})", node.name, node.id)));
        
        match node.node_type {
            NodeType::Input => {
                // 输入节点加载数据
                instructions.push(WasmInstruction::Comment("加载输入数据".to_string()));
                instructions.push(WasmInstruction::I32Const(0)); // 内存偏移
                instructions.push(WasmInstruction::I32Load(0)); // 加载数据
            },
            NodeType::Output => {
                // 输出节点存储数据
                instructions.push(WasmInstruction::Comment("存储输出数据".to_string()));
                instructions.push(WasmInstruction::I32Const(0)); // 内存偏移
                instructions.push(WasmInstruction::I32Store(0)); // 存储数据
            },
            NodeType::Operation => {
                // 操作节点执行计算
                if let Some(op) = &node.operation {
                    self.generate_operation_instructions(op, node, instructions)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// 为操作生成指令
    fn generate_operation_instructions(&self, op: &Operation, node: &AlgorithmNode, instructions: &mut Vec<WasmInstruction>) -> Result<()> {
        match op {
            Operation::Add => {
                instructions.push(WasmInstruction::Comment("加法操作".to_string()));
                instructions.push(WasmInstruction::I32Add);
            },
            Operation::Subtract => {
                instructions.push(WasmInstruction::Comment("减法操作".to_string()));
                instructions.push(WasmInstruction::I32Sub);
            },
            Operation::Multiply => {
                instructions.push(WasmInstruction::Comment("乘法操作".to_string()));
                instructions.push(WasmInstruction::I32Mul);
            },
            Operation::Divide => {
                instructions.push(WasmInstruction::Comment("除法操作".to_string()));
                instructions.push(WasmInstruction::I32Div);
            },
            Operation::MatMul => {
                instructions.push(WasmInstruction::Comment("矩阵乘法操作".to_string()));
                // 矩阵乘法需要调用特殊函数
                instructions.push(WasmInstruction::Call("matmul".to_string()));
            },
            Operation::Activation(activation) => {
                match activation {
                    Activation::ReLU => {
                        instructions.push(WasmInstruction::Comment("ReLU激活".to_string()));
                        instructions.push(WasmInstruction::Call("relu".to_string()));
                    },
                    Activation::Sigmoid => {
                        instructions.push(WasmInstruction::Comment("Sigmoid激活".to_string()));
                        instructions.push(WasmInstruction::Call("sigmoid".to_string()));
                    },
                    Activation::Tanh => {
                        instructions.push(WasmInstruction::Comment("Tanh激活".to_string()));
                        instructions.push(WasmInstruction::Call("tanh".to_string()));
                    },
                    Activation::Softmax => {
                        instructions.push(WasmInstruction::Comment("Softmax激活".to_string()));
                        instructions.push(WasmInstruction::Call("softmax".to_string()));
                    },
                }
            },
            Operation::Transpose => {
                instructions.push(WasmInstruction::Comment("转置操作".to_string()));
                instructions.push(WasmInstruction::Call("transpose".to_string()));
            },
            Operation::Reshape => {
                instructions.push(WasmInstruction::Comment("重塑操作".to_string()));
                instructions.push(WasmInstruction::Call("reshape".to_string()));
            },
            Operation::Concat => {
                instructions.push(WasmInstruction::Comment("连接操作".to_string()));
                instructions.push(WasmInstruction::Call("concat".to_string()));
            },
            Operation::Slice => {
                instructions.push(WasmInstruction::Comment("切片操作".to_string()));
                instructions.push(WasmInstruction::Call("slice".to_string()));
            },
            Operation::Custom(name) => {
                instructions.push(WasmInstruction::Comment(format!("自定义操作: {}", name)));
                instructions.push(WasmInstruction::Call(name.clone()));
            },
        }
        
        Ok(())
    }
    
    /// 编译指令为WASM二进制
    fn compile_instructions(&self, instructions: Vec<WasmInstruction>) -> Result<Vec<u8>> {
        // 实际实现应该使用WASM编译库
        // 这里返回一个占位符二进制
        warn!("WASM编译功能尚未完全实现");
        
        // 生成一个简单的WASM模块头作为占位符
        Ok(vec![
            0x00, 0x61, 0x73, 0x6D, // WASM魔数
            0x01, 0x00, 0x00, 0x00, // 版本号 1
        ])
    }
    
    /// 验证生成的WASM
    fn validate_generated_wasm(&self, wasm_binary: &[u8]) -> Result<()> {
        // 检查WASM魔数
        if wasm_binary.len() < 8 || &wasm_binary[0..4] != &[0x00, 0x61, 0x73, 0x6D] {
            return Err(Error::algorithm("无效的WASM二进制"));
        }
        
        // 检查版本
        if &wasm_binary[4..8] != &[0x01, 0x00, 0x00, 0x00] {
            return Err(Error::algorithm("不支持的WASM版本"));
        }
        
        // 简化验证，实际应使用专门的WASM验证库
        Ok(())
    }
    
    /// 获取导入黑名单
    fn get_blacklisted_imports(&self) -> Vec<&str> {
        self.config.blacklisted_imports.iter().map(|s| s.as_str()).collect()
    }
}

/// DSL执行上下文
pub struct DSLContext {
    input: Option<serde_json::Value>,
    output: Option<serde_json::Value>,
}

impl DSLContext {
    pub fn new() -> Self {
        Self { input: None, output: None }
    }
    pub fn set_input(&mut self, input: serde_json::Value) {
        self.input = Some(input);
    }
    pub fn set_output(&mut self, output: serde_json::Value) {
        self.output = Some(output);
    }
    pub fn get_input(&self) -> Option<&serde_json::Value> {
        self.input.as_ref()
    }
    pub fn get_output(&self) -> Option<&serde_json::Value> {
        self.output.as_ref()
    }
}

/// DSL执行器
pub struct DSLExecutor;

impl DSLExecutor {
    pub fn new() -> Self {
        Self
    }
    pub fn execute(&self, dsl_code: &str, context: &mut DSLContext) -> Result<serde_json::Value> {
        // 解析DSL为AST
        let parser = DslParser::default();
        let ast = parser.parse_dsl_to_ast(dsl_code)?;
        parser.validate_ast(&ast)?;
        // 这里应有更复杂的执行逻辑，暂用输入数据作为输出
        let input = context.get_input().cloned().unwrap_or_default();
        context.set_output(input.clone());
        Ok(input)
    }
} 