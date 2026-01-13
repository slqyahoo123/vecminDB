use crate::data::{
    DatabaseType, DatabaseConfig, QueryParams, QueryParam, SortDirection,
    DatabaseConnector, WriteMode, DatabaseConnectorFactory, DatabaseManager,
    DataBatch, DataFormat, DataPipelineConfig, DataLoader, DataProcessor
};
use crate::data::connector::{
    WriteMode, DatabaseConnectorFactory, DatabaseManager
};
use crate::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;
use uuid;

/// 数据库连接器使用示例
pub async fn connector_example() -> Result<()> {
    println!("开始数据库连接器示例...");
    
    // 初始化数据库管理器
    let mut db_manager = DatabaseManager::new();
    
    // 创建MySQL连接配置
    let mysql_config = DatabaseConfig {
        db_type: DatabaseType::MySQL,
        connection_string: "mysql://localhost:3306/testdb".to_string(),
        username: Some("root".to_string()),
        password: Some("password".to_string()),
        database: Some("testdb".to_string()),
        table: Some("users".to_string()),
        pool_size: Some(10),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    
    // 创建MySQL连接器
    match DatabaseConnectorFactory::create_connector(mysql_config) {
        Ok(connector) => {
            db_manager.add_connector("mysql", connector);
        },
        Err(e) => {
            println!("无法创建MySQL连接器: {}", e);
            // 示例中继续执行，实际应用中可能需要处理错误
        }
    }
    
    // 创建PostgreSQL连接配置
    let pg_config = DatabaseConfig {
        db_type: DatabaseType::PostgreSQL,
        connection_string: "postgres://localhost:5432/testdb".to_string(),
        username: Some("postgres".to_string()),
        password: Some("password".to_string()),
        database: Some("testdb".to_string()),
        table: Some("products".to_string()),
        pool_size: Some(10),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    
    // 创建PostgreSQL连接器
    match DatabaseConnectorFactory::create_connector(pg_config) {
        Ok(connector) => {
            db_manager.add_connector("postgres", connector);
        },
        Err(e) => {
            println!("无法创建PostgreSQL连接器: {}", e);
            // 示例中继续执行，实际应用中可能需要处理错误
        }
    }
    
    // 创建MongoDB连接配置
    let mongo_config = DatabaseConfig {
        db_type: DatabaseType::MongoDB,
        connection_string: "mongodb://localhost:27017".to_string(),
        username: Some("admin".to_string()),
        password: Some("password".to_string()),
        database: Some("testdb".to_string()),
        table: Some("documents".to_string()),
        pool_size: Some(10),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    
    // 创建MongoDB连接器
    match DatabaseConnectorFactory::create_connector(mongo_config) {
        Ok(connector) => {
            db_manager.add_connector("mongodb", connector);
        },
        Err(e) => {
            println!("无法创建MongoDB连接器: {}", e);
            // 示例中继续执行，实际应用中可能需要处理错误
        }
    }
    
    // 连接所有数据库
    match db_manager.connect_all().await {
        Ok(_) => println!("所有数据库连接成功"),
        Err(e) => println!("数据库连接失败: {}", e),
    }
    
    // MySQL示例
    if let Some(mysql_connector) = db_manager.get_connector("mysql") {
        // 使用MySQL连接器执行查询
        let query_params = QueryParams {
            query: "SELECT * FROM users WHERE age > ?".to_string(),
            params: vec![QueryParam::Integer(18)],
            limit: Some(10),
            offset: Some(0),
            sort_by: Some("name".to_string()),
            sort_direction: Some(SortDirection::Ascending),
        };
        
        let results_future = mysql_connector.query(&query_params);
        let pinned_future = Box::pin(results_future);
        if let Ok(result) = pinned_future.await {
            println!("MySQL查询结果: {} 行", result.batch_size());
            if let Some(metadata) = result.metadata.as_ref() {
                if let Some(columns) = metadata.get("columns") {
                    println!("  列：{}", columns);
                }
            }
        }
    }
    
    // MongoDB示例
    if let Some(mongo_connector) = db_manager.get_connector("mongodb") {
        // 使用MongoDB连接器执行查询
        let query_params = QueryParams {
            query: r#"{"age": {"$gt": 18}}"#.to_string(),
            params: vec![],
            limit: Some(10),
            offset: Some(0),
            sort_by: Some("name".to_string()),
            sort_direction: Some(SortDirection::Ascending),
        };
        
        match mongo_connector.query(&query_params).await {
            Ok(batch) => {
                println!("MongoDB查询结果: {} 行", batch.batch_size());
                // 处理查询结果
            },
            Err(e) => println!("MongoDB查询失败: {}", e),
        }
    }
    
    // PostgreSQL示例
    if let Some(pg_connector) = db_manager.get_connector("postgres") {
        // 获取PostgreSQL表结构
        match pg_connector.get_schema(Some("products")).await {
            Ok(schema) => {
                println!("PostgreSQL表结构: {} 字段", schema.fields.len());
                // 处理表结构信息
                for field in schema.fields {
                    println!("  字段: {}, 类型: {:?}", field.name, field.field_type);
                }
            },
            Err(e) => println!("获取PostgreSQL表结构失败: {}", e),
        }
    }
    
    // 断开所有数据库连接
    db_manager.disconnect_all().await?;
    
    println!("数据库连接器示例完成");
    Ok(())
}

/// 多数据源集成示例
pub async fn multi_source_integration_example() -> Result<()> {
    println!("开始多数据源集成示例...");
    
    // 创建数据库管理器
    let mut db_manager = DatabaseManager::new();
    
    // 添加多个数据源
    let mysql_config = DatabaseConfig {
        db_type: DatabaseType::MySQL,
        connection_string: "mysql://localhost:3306".to_string(),
        username: Some("root".to_string()),
        password: Some("password".to_string()),
        database: Some("users_db".to_string()),
        table: None,
        pool_size: Some(10),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    
    let mongo_config = DatabaseConfig {
        db_type: DatabaseType::MongoDB,
        connection_string: "mongodb://localhost:27017".to_string(),
        username: Some("admin".to_string()),
        password: Some("password".to_string()),
        database: Some("products_db".to_string()),
        table: None,
        pool_size: Some(10),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    
    // 添加连接器，注意这里需要使用factory创建连接器再添加
    let mysql_connector = DatabaseConnectorFactory::create_connector(mysql_config)?;
    db_manager.add_connector("users_db", mysql_connector);
    
    let mongo_connector = DatabaseConnectorFactory::create_connector(mongo_config)?;
    db_manager.add_connector("products_db", mongo_connector);
    
    // 连接所有数据库
    db_manager.connect_all().await?;
    
    // 从用户数据库获取数据
    let users_data = if let Some(users_connector) = db_manager.get_connector("users_db") {
        // 使用MySQL连接器执行查询
        let query_params = QueryParams {
            query: "SELECT id, name, email FROM users WHERE status = ?".to_string(),
            params: vec![QueryParam::String("active".to_string())],
            limit: Some(1000),
            offset: None,
            sort_by: None,
            sort_direction: None,
        };
        
        users_connector.query(&query_params).await?
    } else {
        return Err(Error::invalid_argument("用户数据库连接器不存在".to_string()));
    };
    
    println!("从MySQL获取了 {} 个用户", users_data.batch_size());
    
    // 从产品数据库获取数据
    let products_data = if let Some(products_connector) = db_manager.get_connector("products_db") {
        // 使用MongoDB连接器执行查询
        let query_params = QueryParams {
            query: r#"{"category": "electronics", "in_stock": true}"#.to_string(),
            params: vec![],
            limit: Some(1000),
            offset: None,
            sort_by: Some("price".to_string()),
            sort_direction: Some(SortDirection::Ascending),
        };
        
        products_connector.query(&query_params).await?
    } else {
        return Err(Error::invalid_argument("产品数据库连接器不存在".to_string()));
    };
    
    println!("从MongoDB获取了 {} 个产品", products_data.batch_size());
    
    // 在这里可以进行数据集成和处理
    // 例如，将用户数据和产品数据合并，创建一个用户-产品推荐数据集
    println!("数据集成和处理...");
    
    // 断开所有连接
    db_manager.disconnect_all().await?;
    
    println!("多数据源集成示例完成");
    Ok(())
}

/// 数据库模式自动发现示例
pub async fn schema_discovery_example() -> Result<()> {
    println!("开始数据库模式自动发现示例...");
    
    // 创建数据库管理器
    let mut db_manager = DatabaseManager::new();
    
    // 配置数据库连接
    let db_config = DatabaseConfig {
        db_type: DatabaseType::PostgreSQL,
        connection_string: "postgresql://localhost:5432".to_string(),
        username: Some("postgres".to_string()),
        password: Some("password".to_string()),
        database: Some("analytics_db".to_string()),
        table: None,
        pool_size: Some(5),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    
    // 创建连接器后添加到管理器
    let db_connector = DatabaseConnectorFactory::create_connector(db_config)?;
    db_manager.add_connector("analytics", db_connector);
    
    // 连接数据库
    db_manager.connect_all().await?;
    
    // 获取连接器
    if let Some(connector) = db_manager.get_connector("analytics") {
        // 使用PostgreSQL连接器执行查询
        // 获取所有表的模式
        let tables = vec!["users", "events", "sessions", "conversions"];
        let mut schemas = HashMap::new();
        
        for table in tables {
            let schema = connector.get_schema(Some(table)).await?;
            println!("表 {} 的模式: {} 个字段", table, schema.fields.len());
            
            // 打印字段信息
            for field in &schema.fields {
                println!("  - 字段: {}, 类型: {:?}, 可空: {}", 
                    field.name, field.field_type, !field.required);
            }
            
            schemas.insert(table.to_string(), schema);
        }
        
        // 在这里可以进行模式分析和处理
        // 例如，检查表之间的关系，识别主键和外键等
        println!("模式分析和处理...");
    }
    
    // 断开连接
    db_manager.disconnect_all().await?;
    
    println!("数据库模式自动发现示例完成");
    Ok(())
}

/// 多数据库连接示例
pub async fn multi_database_example() -> Result<()> {
    // 创建数据库管理器
    let mut manager = DatabaseManager::new();
    
    // 添加MySQL连接器
    let mysql_config = DatabaseConfig {
        db_type: DatabaseType::MySQL,
        connection_string: "mysql://localhost:3306".to_string(),
        username: Some("root".to_string()),
        password: Some("password".to_string()),
        database: Some("test_db".to_string()),
        table: Some("users".to_string()),
        pool_size: Some(10),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    // 使用create_connector创建连接器，然后传递给add_connector
    match DatabaseConnectorFactory::create_connector(mysql_config) {
        Ok(connector) => {
            manager.add_connector("mysql", connector);
        },
        Err(e) => {
            println!("无法创建MySQL连接器: {}", e);
        }
    }
    
    // 添加PostgreSQL连接器
    let postgres_config = DatabaseConfig {
        db_type: DatabaseType::PostgreSQL,
        connection_string: "postgres://localhost:5432".to_string(),
        username: Some("postgres".to_string()),
        password: Some("password".to_string()),
        database: Some("test_db".to_string()),
        table: Some("products".to_string()),
        pool_size: Some(10),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    // 使用create_connector创建连接器，然后传递给add_connector
    match DatabaseConnectorFactory::create_connector(postgres_config) {
        Ok(connector) => {
            manager.add_connector("postgres", connector);
        },
        Err(e) => {
            println!("无法创建PostgreSQL连接器: {}", e);
        }
    }
    
    // 添加MongoDB连接器
    let mongodb_config = DatabaseConfig {
        db_type: DatabaseType::MongoDB,
        connection_string: "mongodb://localhost:27017".to_string(),
        username: None,
        password: None,
        database: Some("test_db".to_string()),
        table: Some("orders".to_string()),
        pool_size: None,
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    // 使用create_connector创建连接器，然后传递给add_connector
    match DatabaseConnectorFactory::create_connector(mongodb_config) {
        Ok(connector) => {
            manager.add_connector("mongodb", connector);
        },
        Err(e) => {
            println!("无法创建MongoDB连接器: {}", e);
        }
    }
    
    // 添加Redis连接器
    let redis_config = DatabaseConfig {
        db_type: DatabaseType::Redis,
        connection_string: "redis://localhost:6379".to_string(),
        username: None,
        password: None,
        database: Some("0".to_string()),
        table: None,
        pool_size: None,
        timeout: Some(10),
        extra_params: HashMap::new(),
    };
    // 使用create_connector创建连接器，然后传递给add_connector
    match DatabaseConnectorFactory::create_connector(redis_config) {
        Ok(connector) => {
            manager.add_connector("redis", connector);
        },
        Err(e) => {
            println!("无法创建Redis连接器: {}", e);
        }
    }
    
    // 添加Elasticsearch连接器
    let elasticsearch_config = DatabaseConfig {
        db_type: DatabaseType::Elasticsearch,
        connection_string: "http://localhost:9200".to_string(),
        username: None,
        password: None,
        database: None,
        table: Some("documents".to_string()),
        pool_size: None,
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    // 使用create_connector创建连接器，然后传递给add_connector
    match DatabaseConnectorFactory::create_connector(elasticsearch_config) {
        Ok(connector) => {
            manager.add_connector("elasticsearch", connector);
        },
        Err(e) => {
            println!("无法创建Elasticsearch连接器: {}", e);
        }
    }
    
    // 连接所有数据库
    manager.connect_all().await?;
    
    // 测试所有连接
    let connection_status = manager.test_all_connections().await?;
    println!("连接状态：");
    for (name, status) in connection_status {
        println!("  {}: {}", name, if status { "连接成功" } else { "连接失败" });
    }
    
    // 从MySQL查询用户数据
    if let Some(mysql_connector) = manager.get_connector("mysql") {
        // 使用MySQL连接器执行查询
        let query_params = QueryParams {
            query: "SELECT * FROM users LIMIT 10".to_string(),
            params: vec![],
            limit: Some(10),
            offset: None,
            sort_by: Some("id".to_string()),
            sort_direction: Some(SortDirection::Ascending),
        };
        
        let result = mysql_connector.query(&query_params).await?;
        println!("\nMySQL用户数据：");
        println!("  行数：{}", result.batch_size());
        println!("  列：{}", result.metadata.get("columns").unwrap_or(&"".to_string()));
    }
    
    // 从PostgreSQL查询产品数据
    if let Some(postgres_connector) = manager.get_connector("postgres") {
        // 使用PostgreSQL连接器执行查询
        let query_params = QueryParams {
            query: "SELECT * FROM products WHERE price > $1".to_string(),
            params: vec![QueryParam::Float(10.0)],
            limit: Some(5),
            offset: None,
            sort_by: Some("price".to_string()),
            sort_direction: Some(SortDirection::Descending),
        };
        
        let results_future = postgres_connector.query(&query_params);
        let pinned_future = Box::pin(results_future);
        if let Ok(result) = pinned_future.await {
            println!("PostgreSQL查询结果: {} 行", result.batch_size());
            if let Some(metadata) = result.metadata.as_ref() {
                if let Some(columns) = metadata.get("columns") {
                    println!("  列：{}", columns);
                }
            }
        }
    }
    
    // 从MongoDB查询订单数据
    if let Some(mongodb_connector) = manager.get_connector("mongodb") {
        // 直接使用连接器，不需要.read()
        let query_params = QueryParams {
            query: "{ \"order_status\": \"pending\" }".to_string(),
            params: vec![],
            limit: Some(10),
            offset: None,
            sort_by: None,
            sort_direction: None,
        };
        
        let results_future = mongodb_connector.query(&query_params);
        let pinned_future = Box::pin(results_future);
        match pinned_future.await {
            Ok(result) => {
                println!("MongoDB订单查询结果: {} 行", result.batch_size());
                // 处理结果
            },
            Err(e) => {
                println!("MongoDB查询失败: {}", e);
            }
        }
    }
    
    // 从Redis获取数据
    if let Some(redis_connector) = manager.get_connector("redis") {
        // 直接使用连接器，不需要.read()
        let query_params = QueryParams {
            query: "GET user:profile:123".to_string(),
            params: vec![],
            limit: None,
            offset: None,
            sort_by: None,
            sort_direction: None,
        };
        
        let results_future = redis_connector.query(&query_params);
        let pinned_future = Box::pin(results_future);
        match pinned_future.await {
            Ok(result) => {
                println!("Redis数据: {}", result.batch_size());
                // 处理结果
            },
            Err(e) => {
                println!("Redis查询失败: {}", e);
            }
        }
    }
    
    // 从Elasticsearch搜索文档
    if let Some(elasticsearch_connector) = manager.get_connector("elasticsearch") {
        // 直接使用连接器，不需要.read()
        let query_params = QueryParams {
            query: "{\"query\":{\"match_all\":{}}}".to_string(),
            params: vec![],
            limit: Some(10),
            offset: None,
            sort_by: None,
            sort_direction: None,
        };
        
        let results_future = elasticsearch_connector.query(&query_params);
        let pinned_future = Box::pin(results_future);
        match pinned_future.await {
            Ok(result) => {
                println!("Elasticsearch搜索结果: {} 行", result.batch_size());
                // 处理结果
            },
            Err(e) => {
                println!("Elasticsearch搜索失败: {}", e);
            }
        }
    }
    
    // 断开所有连接
    manager.disconnect_all().await?;
    println!("\n已断开所有数据库连接");
    
    Ok(())
}

/// 断开所有数据库连接
pub async fn disconnect_all_databases() -> Result<()> {
    println!("正在断开所有数据库连接...");
    
    // 创建数据库管理器
    let mut manager = DatabaseManager::new();
    
    // 添加各种数据库连接器
    let databases = vec![
        ("mysql", DatabaseType::MySQL, "mysql://localhost:3306"),
        ("postgres", DatabaseType::PostgreSQL, "postgres://localhost:5432"),
        ("sqlite", DatabaseType::SQLite, "sqlite:///tmp/test.db"),
        ("mongodb", DatabaseType::MongoDB, "mongodb://localhost:27017"),
        ("redis", DatabaseType::Redis, "redis://localhost:6379"),
        ("elasticsearch", DatabaseType::Elasticsearch, "http://localhost:9200"),
    ];
    
    // 添加所有连接器
    for (name, db_type, conn_string) in databases {
        let config = DatabaseConfig {
            db_type,
            connection_string: conn_string.to_string(),
            username: None,
            password: None,
            database: None,
            table: None,
            pool_size: None,
            timeout: Some(5),
            extra_params: HashMap::new(),
        };
        
        // 使用create_connector创建连接器，然后传递给add_connector
        match DatabaseConnectorFactory::create_connector(config) {
            Ok(connector) => {
                manager.add_connector(name, connector);
            },
            Err(e) => {
                println!("添加 {} 连接器失败: {}", name, e);
            }
        }
    }
    
    // 断开所有连接
    if let Err(e) = manager.disconnect_all().await {
        println!("断开连接失败: {}", e);
    } else {
        println!("已成功断开所有数据库连接");
    }
    
    Ok(())
}

/// 数据库与文件集成示例
pub async fn database_file_integration_example() -> Result<()> {
    println!("开始数据库与文件集成示例...");
    
    // 创建数据库管理器
    let db_manager = Arc::new(DatabaseManager::new());
    
    // 添加MySQL连接器
    let mysql_config = DatabaseConfig {
        db_type: DatabaseType::MySQL,
        connection_string: "mysql://localhost:3306".to_string(),
        username: Some("root".to_string()),
        password: Some("password".to_string()),
        database: Some("test_db".to_string()),
        table: Some("users".to_string()),
        pool_size: Some(10),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    // 使用create_connector创建连接器，然后传递给add_connector
    match DatabaseConnectorFactory::create_connector(mysql_config) {
        Ok(connector) => {
            db_manager.add_connector("mysql", connector);
        },
        Err(e) => {
            println!("无法创建MySQL连接器: {}", e);
        }
    }
    
    // 添加PostgreSQL连接器
    let postgres_config = DatabaseConfig {
        db_type: DatabaseType::PostgreSQL,
        connection_string: "postgres://localhost:5432".to_string(),
        username: Some("postgres".to_string()),
        password: Some("password".to_string()),
        database: Some("test_db".to_string()),
        table: Some("products".to_string()),
        pool_size: Some(10),
        timeout: Some(30),
        extra_params: HashMap::new(),
    };
    // 使用create_connector创建连接器，然后传递给add_connector
    match DatabaseConnectorFactory::create_connector(postgres_config) {
        Ok(connector) => {
            db_manager.add_connector("postgres", connector);
        },
        Err(e) => {
            println!("无法创建PostgreSQL连接器: {}", e);
        }
    }
    
    // 创建数据管道配置
    let pipeline_config = DataPipelineConfig {
        data_dir: "data".to_string(),
        cache_dir: "cache".to_string(),
        max_cache_size: 1024 * 1024 * 1024, // 1GB
        batch_size: 32,
        shuffle_buffer: 10000,
        num_workers: 4,
    };
    
    // 创建数据加载器，并设置数据库管理器
    let loader = DataLoader::with_database_manager(pipeline_config, db_manager.clone())?;
    
    // 创建临时目录用于存储数据
    let temp_dir = std::env::temp_dir().join("vecmind_example");
    tokio::fs::create_dir_all(&temp_dir).await?;
    
    println!("临时目录: {}", temp_dir.display());
    
    // 1. 从CSV文件加载数据
    let csv_path = temp_dir.join("sample.csv");
    
    // 创建示例CSV文件
    let csv_content = "id,name,age,email\n1,Alice,30,alice@example.com\n2,Bob,25,bob@example.com\n3,Charlie,35,charlie@example.com";
    tokio::fs::write(&csv_path, csv_content).await?;
    
    let csv_target = temp_dir.join("csv_data");
    let (csv_size, _) = loader.load(
        csv_path.to_str().unwrap(),
        csv_target.to_str().unwrap(),
        DataFormat::CSV
    ).await?;
    
    println!("从CSV加载了 {} 行数据", csv_size);
    
    // 2. 从JSON文件加载数据
    let json_path = temp_dir.join("sample.json");
    
    // 创建示例JSON文件
    let json_content = r#"[
        {"id": 1, "product": "Laptop", "price": 1200, "in_stock": true},
        {"id": 2, "product": "Phone", "price": 800, "in_stock": true},
        {"id": 3, "product": "Tablet", "price": 500, "in_stock": false}
    ]"#;
    tokio::fs::write(&json_path, json_content).await?;
    
    let json_target = temp_dir.join("json_data");
    let (json_size, _) = loader.load(
        json_path.to_str().unwrap(),
        json_target.to_str().unwrap(),
        DataFormat::JSON
    ).await?;
    
    println!("从JSON加载了 {} 行数据", json_size);
    
    // 3. 从MySQL数据库加载数据
    // 注意：这里使用URL格式，而不是直接使用连接器
    // 这样可以测试DataLoader的数据库URL解析功能
    let mysql_url = "mysql://localhost:3306?table=users&query=SELECT id, name, email FROM {table} LIMIT 10";
    let mysql_target = temp_dir.join("mysql_data");
    
    // 连接数据库
    db_manager.connect_all().await?;
    
    let (mysql_size, _) = loader.load(
        mysql_url,
        mysql_target.to_str().unwrap(),
        DataFormat::Custom("Database".to_string()) // 这个参数会被忽略，因为URL会被识别为数据库URL
    ).await?;
    
    println!("从MySQL加载了 {} 行数据", mysql_size);
    
    // 4. 从PostgreSQL数据库加载数据
    let postgres_url = "postgres://localhost:5432?table=products&query=SELECT id, product, price FROM {table} WHERE price > 100";
    let postgres_target = temp_dir.join("postgres_data");
    
    // 连接数据库
    db_manager.connect_all().await?;
    
    let (postgres_size, _) = loader.load(
        postgres_url,
        postgres_target.to_str().unwrap(),
        DataFormat::Custom("Database".to_string()) // 这个参数会被忽略
    ).await?;
    
    println!("从PostgreSQL加载了 {} 行数据", postgres_size);
    
    // 5. 数据处理示例 - 将CSV和数据库数据合并
    println!("\n数据处理示例 - 合并数据源");
    
    // 创建数据处理器
    let processor = DataProcessor::new()?;
    
    // 从CSV和MySQL加载的数据创建批次
    let csv_batch = loader.get_batch(csv_target.to_str().unwrap(), 10).await?;
    let mysql_batch = loader.get_batch(mysql_target.to_str().unwrap(), 10).await?;
    
    println!("CSV批次大小: {}", csv_batch.batch_size());
    println!("MySQL批次大小: {}", mysql_batch.batch_size());
    
    // 合并批次（简单示例）
    let mut merged_features = csv_batch.get_features().clone();
    merged_features.extend(mysql_batch.get_features().clone());
    
    let merged_batch = DataBatch::new(merged_features);
    
    println!("合并后批次大小: {}", merged_batch.batch_size());
    
    // 断开数据库连接
    db_manager.disconnect_all().await?;
    
    // 清理临时文件
    tokio::fs::remove_dir_all(&temp_dir).await?;
    
    println!("数据库与文件集成示例完成");
    Ok(())
} 