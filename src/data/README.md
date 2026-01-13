# 数据模块目录结构

本目录包含AI数据库的数据处理相关模块，负责数据加载、处理、转换、特征提取和存储等功能。

## 目录结构

```
data/
├── adaptive_weights/    # 自适应权重调整模块
├── backup/              # 备份文件目录
├── connector/           # 数据库连接器
├── loader/              # 数据加载器
├── method_selector/     # 方法选择器
├── multimodal/          # 多模态特征提取
├── pipeline/            # 数据处理管道
├── processor/           # 数据处理器
├── schema/              # 数据模式定义
├── tests/               # 测试模块
├── text_features/       # 文本特征提取
├── utils/               # 工具函数
└── mod.rs               # 模块导出
```

## 模块说明

### adaptive_weights
提供自适应权重算法，用于动态调整特征提取和处理过程中的权重。

### connector
提供与各种数据库的连接和交互功能，支持MySQL、PostgreSQL等多种数据库。

### loader
负责从各种数据源加载数据，支持CSV、JSON等多种格式。

### method_selector
提供智能算法选择功能，根据数据特性自动选择最佳特征提取方法。

### multimodal
提供多模态数据处理功能，支持文本、图像、视频等多种数据类型的联合特征提取。

### pipeline
提供数据处理管道，用于构建和执行复杂的数据处理流程。

### processor
提供通用数据处理功能，如转换、清洗、归一化等。

### schema
提供数据模式定义和验证功能，确保数据结构的一致性。

### text_features
提供文本特征提取功能，包括基础特征提取和高级语义提取。

### utils
提供各种工具函数，如并行处理、数据验证等。

## 使用示例

```rust
use crate::data::{DataConfig, DataBatch, text_features::TextFeatureExtractor};

// 创建数据加载配置
let config = DataConfig::new()
    .with_source("/path/to/data.csv")
    .with_batch_size(64)
    .with_shuffle(true);

// 加载数据
let data_batch = loader::load_data(&config).await?;

// 提取特征
let feature_config = TextFeatureConfig::new()
    .with_method(Method::TfIdf)
    .with_dimension(100);
    
let extractor = TextFeatureExtractor::from_config(&feature_config)?;
let features = extractor.extract(&data_batch)?;
```

## 开发指南

1. 所有新功能应遵循模块化设计原则
2. 每个模块应提供清晰的API和完整的文档
3. 所有公共组件应通过mod.rs导出
4. 测试文件应放在tests目录下
5. 工具函数应放在utils目录下 