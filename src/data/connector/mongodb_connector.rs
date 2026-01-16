// 注意：mongodb 特性未在 Cargo.toml 中定义，整个文件已用块注释包裹以避免 cfg 警告
// 如需启用 MongoDB 连接器，请在 Cargo.toml 中添加 mongodb 特性，然后取消下面的注释
/*
#[cfg(feature = "mongodb")]
// connector/mongodb_connector.rs - MongoDB鏁版嵁搴撹繛鎺ュ櫒瀹炵幇

#[cfg(feature = "mongodb")]
use std::collections::HashMap;
#[cfg(feature = "mongodb")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "mongodb")]
use crate::data::{DataBatch, DataSchema, FieldDefinition};
#[cfg(feature = "mongodb")]
use crate::data::schema::FieldType;
#[cfg(feature = "mongodb")]
use crate::compat::tensor::TensorData;
#[cfg(feature = "mongodb")]
use crate::Result;
#[cfg(feature = "mongodb")]
use crate::Error;
#[cfg(feature = "mongodb")]
use super::types::{DatabaseConnector, DatabaseConfig, DatabaseType, QueryParams, WriteMode, QueryParam, SortDirection};
#[cfg(feature = "mongodb")]
use mongodb::Client;
#[cfg(feature = "mongodb")]
use mongodb::options::{ClientOptions, Credential, FindOptions, UpdateOptions};
#[cfg(all(feature = "mongodb", feature = "bson"))]
use bson::{Document, Bson, oid::ObjectId, doc};
#[cfg(feature = "mongodb")]
use async_trait::async_trait;
#[cfg(feature = "mongodb")]
use futures::stream::StreamExt;
#[cfg(feature = "mongodb")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "mongodb")]
use serde_json::{Value, json};
#[cfg(feature = "mongodb")]
use tokio::sync::Mutex as TokioMutex;

/// MongoDB客户端
#[cfg(feature = "mongodb")]
#[derive(Clone)]
pub struct MongoDBClient {
    pub connection_string: Option<String>,
    pub client: Option<mongodb::Client>,
}

#[cfg(feature = "mongodb")]
impl MongoDBClient {
    /// 创建新的MongoDB客户端
    pub fn new(connection_string: Option<String>) -> Self {
        Self {
            connection_string,
            client: None,
        }
    }
    
    /// 列出数据库名称
    pub async fn list_database_names(&self) -> Result<Vec<String>> {
        // 模拟实现
        println!("列出MongoDB数据库名称");
        Ok(vec!["admin".to_string(), "test".to_string(), "local".to_string()])
    }
    
    /// 获取数据库引用
    pub fn database(&self, db_name: &str) -> MongoDBDatabase {
        MongoDBDatabase::new(db_name.to_string())
    }
    
    /// 执行查询
    pub async fn query(&self, collection: &str, filter: &str, sort: Option<(&str, bool)>, limit: Option<u64>, skip: Option<u64>) -> Result<Vec<HashMap<String, String>>> {
        let client = self.client.as_ref().ok_or_else(|| Error::not_found("MongoDB客户端未连接"))?;
        
        // 解析过滤条件
        let filter_doc = mongodb::bson::Document::parse_document(filter)?;
        
        let mut options = FindOptions::default();
        
        // 设置排序
        if let Some((sort_by, ascending)) = sort {
            let sort_direction = if ascending { 1 } else { -1 };
            options.sort = Some(doc! { sort_by: sort_direction });
        }
        
        // 设置限制
        if let Some(limit_val) = limit {
            options.limit = Some(limit_val as i64);
        }
        
        // 设置跳过
        if let Some(skip_val) = skip {
            options.skip = Some(skip_val as u64);
        }
        
        // 执行查询
        let collection = client.database("test").collection::<Document>(collection);
        let mut cursor = collection.find(filter_doc, options).await?;
        
        let mut results = Vec::new();
        
        // 处理结果
        while let Some(result) = cursor.next().await {
            match result {
                Ok(doc) => {
                    let mut doc = HashMap::new();
                    for (k, v) in doc.iter() {
                        doc.insert(k.to_string(), format!("{}", v));
                    }
                    results.push(doc);
                }
                Err(e) => return Err(Error::invalid_data(format!("MongoDB查询错误: {}", e))),
            }
        }
        
        Ok(results)
    }
    
    /// 插入文档
    pub async fn insert_one(&self, collection: &str, document: HashMap<String, String>) -> Result<String> {
        let client = self.client.as_ref().ok_or_else(|| Error::not_found("MongoDB客户端未连接"))?;
        
        // 转换为BSON文档
        let mut doc = Document::new();
        for (k, v) in document {
            doc.insert(k, v);
        }
        
        // 插入文档
        let collection = client.database("test").collection::<Document>(collection);
        let result = collection.insert_one(doc, None).await?;
        
        // 获取ID
        let id = result.inserted_id.as_object_id().map(|oid| oid.to_hex()).unwrap_or_default();
        
        Ok(id)
    }
    
    /// 插入多个文档
    pub async fn insert_many(&self, collection: &str, documents: Vec<HashMap<String, String>>) -> Result<Vec<String>> {
        let client = self.client.as_ref().ok_or_else(|| Error::not_found("MongoDB客户端未连接"))?;
        
        // 转换为BSON文档
        let mut docs = Vec::new();
        for document in documents {
            let mut doc = Document::new();
            for (k, v) in document {
                doc.insert(k, v);
            }
            docs.push(doc);
        }
        
        // 插入文档
        let collection = client.database("test").collection::<Document>(collection);
        let result = collection.insert_many(docs, None).await?;
        
        // 获取ID列表
        let mut ids = Vec::new();
        for (_, id) in result.inserted_ids {
            let id_str = id.as_object_id().map(|oid| oid.to_hex()).unwrap_or_default();
            ids.push(id_str);
        }
        
        Ok(ids)
    }
    
    /// 更新文档
    pub async fn update_one(&self, collection: &str, filter: HashMap<String, String>, update: HashMap<String, String>) -> Result<bool> {
        let client = self.client.as_ref().ok_or_else(|| Error::not_found("MongoDB客户端未连接"))?;
        
        // 转换为BSON文档
        let mut filter_doc = Document::new();
        for (k, v) in filter {
            filter_doc.insert(k, v);
        }
        
        let mut update_doc = Document::new();
        let mut set_doc = Document::new();
        for (k, v) in update {
            set_doc.insert(k, v);
        }
        update_doc.insert("$set", set_doc);
        
        // 更新文档
        let collection = client.database("test").collection::<Document>(collection);
        let result = collection.update_one(filter_doc, update_doc, None).await?;
        
        Ok(result.modified_count > 0)
    }
    
    /// 删除文档
    pub async fn delete_one(&self, collection: &str, filter: HashMap<String, String>) -> Result<bool> {
        let client = self.client.as_ref().ok_or_else(|| Error::not_found("MongoDB客户端未连接"))?;
        
        // 转换为BSON文档
        let mut filter_doc = Document::new();
        for (k, v) in filter {
            filter_doc.insert(k, v);
        }
        
        // 删除文档
        let collection = client.database("test").collection::<Document>(collection);
        let result = collection.delete_one(filter_doc, None).await?;
        
        Ok(result.deleted_count > 0)
    }
    
    /// 获取集合结构
    pub async fn get_collection_schema(&self, collection: &str) -> Result<Vec<FieldDefinition>> {
        // 这里应该实际查询MongoDB集合的结构
        println!("获取MongoDB集合结构: {}", collection);
        
        // 模拟字段定义
        let mut fields = Vec::new();
        fields.push(FieldDefinition {
            name: "_id".to_string(),
            field_type: FieldType::Text,
            required: true,
            description: Some("文档ID".to_string()),
            default_value: None,
            constraints: None,
        });
        
        fields.push(FieldDefinition {
            name: "name".to_string(),
            field_type: FieldType::Text,
            required: false,
            description: Some("名称".to_string()),
            default_value: None,
            constraints: None,
        });
        
        fields.push(FieldDefinition {
            name: "value".to_string(),
            field_type: FieldType::Numeric,
            required: false,
            description: Some("数值".to_string()),
            default_value: None,
            constraints: None,
        });
        
        Ok(fields)
    }
    
    /// 测试连接
    pub async fn test_connection(&self) -> Result<bool> {
        if let Some(client) = &self.client {
            match client.list_database_names(None, None).await {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        } else {
            Ok(false)
        }
    }

    fn list_database_names<'life0>(&'life0 self) -> Box<dyn std::future::Future<Output = Result<Vec<String>>> + Send + 'life0> {
        Box::new(async move {
            if let Some(client) = &self.client {
                let names = client.list_database_names(None, None).await
                    .map_err(|e| Error::not_connected(format!("MongoDB列出数据库错误: {}", e)))?;
                Ok(names)
            } else {
                return Err(Error::not_connected("MongoDB连接未初始化"));
            }
        })
    }
    
    fn list_collection_names<'life0, 'life1>(
        &'life0 self,
        database_name: &'life1 str,
    ) -> Box<dyn std::future::Future<Output = Result<Vec<String>>> + Send + 'life0>
    where
        'life1: 'life0
    {
        Box::new(async move {
            if let Some(client) = &self.client {
                let database = client.database(database_name);
                let names = database.list_collection_names(None).await
                    .map_err(|e| Error::not_connected(format!("MongoDB列出集合错误: {}", e)))?;
                Ok(names)
            } else {
                return Err(Error::not_connected("MongoDB连接未初始化"));
            }
        })
    }
}

/// MongoDB数据库
#[cfg(feature = "mongodb")]
pub struct MongoDBDatabase {
    name: String,
}

#[cfg(feature = "mongodb")]
impl MongoDBDatabase {
    /// 创建新的MongoDB数据库引用
    pub fn new(name: String) -> Self {
        Self { name }
    }
    
    /// 获取集合引用
    pub fn collection<T>(&self, collection_name: &str) -> MongoDBCollection<T> {
        MongoDBCollection::new(self.name.clone(), collection_name.to_string())
    }
}

/// MongoDB集合
#[cfg(feature = "mongodb")]
pub struct MongoDBCollection<T> {
    db_name: String,
    collection_name: String,
    _phantom: std::marker::PhantomData<T>,
}

#[cfg(feature = "mongodb")]
impl<T> MongoDBCollection<T> {
    /// 创建新的MongoDB集合引用
    pub fn new(db_name: String, collection_name: String) -> Self {
        Self { 
            db_name, 
            collection_name,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// 获取数据库名称
    pub fn get_db_name(&self) -> &str {
        &self.db_name
    }
    
    /// 获取集合名称
    pub fn get_collection_name(&self) -> &str {
        &self.collection_name
    }
}

/// MongoDB连接器
#[cfg(feature = "mongodb")]
pub struct MongoDBConnector {
    config: DatabaseConfig,
    client: Option<Client>,
    active_database: String,
}

#[cfg(feature = "mongodb")]
impl MongoDBConnector {
    /// 创建新的MongoDB连接器
    pub fn new(config: DatabaseConfig) -> Result<Self> {
        let active_database = config.database.clone().unwrap_or_else(|| "test".to_string());
        
        Ok(Self {
            config,
            client: None,
            active_database,
        })
    }
    
    /// 从QueryParams构建MongoDB查询
    fn build_filter(params: &QueryParams) -> String {
        // 简单地使用查询字符串
        params.query.clone()
    }
    
    /// 将MongoDB查询结果转换为DataBatch
    fn convert_results(results: Vec<HashMap<String, String>>) -> Result<DataBatch> {
        if results.is_empty() {
            // 如果没有结果，返回空批次
            return Ok(DataBatch::empty());
        }
        
        // 获取字段名
        let fields = results[0].keys().cloned().collect::<Vec<_>>();
        
        // 转换为特征数据
        let mut features = Vec::new();
        
        for item in results {
            let mut values = Vec::new();
            
            for field in &fields {
                if let Some(value) = item.get(field) {
                    // 尝试将值转换为浮点数
                    match value.parse::<f32>() {
                        Ok(val) => values.push(val),
                        Err(_) => {
                            // 如果无法解析为数字，使用默认值0
                            values.push(0.0);
                        }
                    }
                } else {
                    // 缺失值使用0填充
                    values.push(0.0);
                }
            }
            
            features.push(values);
        }
        
        Ok(DataBatch::new(features))
    }
    
    /// 解析表名为MongoDB集合名
    fn parse_table_name(table_name: &str) -> &str {
        // 提取集合名，移除路径
        let parts: Vec<&str> = table_name.split('/').collect();
        parts.last().unwrap_or(&table_name)
    }
    
    /// 将DataBatch转换为MongoDB文档列表
    fn convert_batch_to_documents(&self, batch: &DataBatch) -> Result<Vec<HashMap<String, String>>> {
        // 获取原始数据
        let data = batch.get_data();
        
        // 创建文档列表
        let mut documents = Vec::new();
        
        // 确保有字段名
        let field_names = match batch.get_field_names() {
            Some(names) => names.clone(),
            None => {
                // 如果没有名称，使用默认名称
                let mut default_names = Vec::new();
                if !data.is_empty() {
                    for i in 0..data[0].len() {
                        default_names.push(format!("field_{}", i));
                    }
                }
                default_names
            }
        };
        
        // 将数据转换为文档
        for row in data {
            let mut doc = HashMap::new();
            
            for (i, value) in row.iter().enumerate() {
                if i < field_names.len() {
                    doc.insert(field_names[i].clone(), value.to_string());
                } else {
                    doc.insert(format!("field_{}", i), value.to_string());
                }
            }
            
            documents.push(doc);
        }
        
        Ok(documents)
    }

    fn get_schema<'life0, 'life1>(&'life0 self, table_name: Option<&'life1 str>) -> Box<dyn std::future::Future<Output = Result<DataSchema>> + Send + 'life0>
    where 
        'life1: 'life0
    {
        Box::new(async move {
            let client = match &self.client {
                Some(client) => client,
                None => return Err(Error::not_found("MongoDB未连接")),
            };
            
            // 获取集合名
            let coll_name = match table_name.or_else(|| self.config.table.as_deref()) {
                Some(name) => name,
                None => return Err(Error::invalid_input("必须提供集合名")),
            };
            
            // 获取数据库名
            let db_name = &self.active_database;
            
            // 创建模式
            let collection_name = table_name.unwrap_or("default_collection");
            let mut schema = DataSchema::new(collection_name, "1.0");
            
            // 获取集合
            let collection = client.database(db_name).collection::<Document>(coll_name);
            
            // 这里应该从实际集合中获取样本文档推断模式
            // 但我们使用模拟数据
            println!("获取MongoDB集合模式: 数据库: {}, 集合: {}", db_name, coll_name);
            
            // 模拟字段定义
            let mut fields = Vec::new();
            fields.push(FieldDefinition {
                name: "_id".to_string(),
                field_type: FieldType::Text,
                required: true,
                description: Some("文档ID".to_string()),
                default_value: None,
                constraints: None,
            });
            
            fields.push(FieldDefinition {
                name: "name".to_string(),
                field_type: FieldType::Text,
                required: false,
                description: Some("名称".to_string()),
                default_value: None,
                constraints: None,
            });
            
            fields.push(FieldDefinition {
                name: "value".to_string(),
                field_type: FieldType::Numeric,
                required: false,
                description: Some("数值".to_string()),
                default_value: None,
                constraints: None,
            });
            
            schema.add_fields(fields);
            
            Ok(schema)
        })
    }

    /// 获取MongoDB客户端
    async fn get_client(&self) -> Result<&Client> {
        match &self.client {
            Some(client) => Ok(client),
            None => Err(Error::not_found("MongoDB客户端未初始化")),
        }
    }
}

#[cfg(feature = "mongodb")]
#[async_trait]
impl DatabaseConnector for MongoDBConnector {
    async fn connect(&mut self) -> Result<()> {
        // 获取连接字符串
        let connection_string = self.config.connection_string.clone();
        
        // 创建客户端
        let client = Client::with_uri_str(&connection_string).await?;
        
        // 连接成功
        self.client = Some(client);
        
        Ok(())
    }
    
    async fn disconnect(&mut self) -> Result<()> {
        // MongoDB客户端会在销毁时自动断开连接
        self.client = None;
        Ok(())
    }
    
    async fn test_connection(&self) -> Result<bool> {
        if let Some(client) = &self.client {
            // 尝试获取数据库列表
            match client.list_database_names(None, None).await {
                Ok(_) => Ok(true),
                Err(e) => Err(Error::connection(format!("MongoDB连接测试失败: {}", e))),
            }
        } else {
            Ok(false)
        }
    }
    
    async fn query(&self, params: &QueryParams) -> Result<DataBatch> {
        if let Some(client) = &self.client {
            // 获取数据库名
            let db_name = self.active_database.as_str();
            
            // 获取集合名
            let collection_name = match self.config.table.as_deref() {
                Some(name) => name,
                None => return Err(Error::invalid_input("必须提供集合名")),
            };
            
            // 构建查询条件
            let filter = Self::build_filter(params);
            
            // 解析为BSON文档
            let filter_doc = Document::parse_document(&filter)?;
            
            // 构建排序
            let sort_doc = match (&params.sort_by, &params.sort_direction) {
                (Some(sort_by), Some(direction)) => {
                    let sort_val = match direction {
                        SortDirection::Ascending => 1,
                        SortDirection::Descending => -1,
                    };
                    Some(doc! { sort_by: sort_val })
                },
                _ => None,
            };
            
            // 设置选项
            let mut options = FindOptions::default();
            
            // 设置排序
            if let Some(sort) = sort_doc {
                options.sort = Some(sort);
            }
            
            // 设置限制
            if let Some(limit) = params.limit {
                options.limit = Some(limit as i64);
            }
            
            // 设置跳过
            if let Some(offset) = params.offset {
                options.skip = Some(offset as u64);
            }
            
            // 执行查询
            let collection = client.database(db_name).collection::<Document>(collection_name);
            let mut cursor = collection.find(filter_doc, options).await?;
            
            let mut results = Vec::new();
            
            // 处理结果
            while let Some(result) = cursor.next().await {
                match result {
                    Ok(doc) => {
                        let mut map = HashMap::new();
                        for (k, v) in doc.iter() {
                            map.insert(k.to_string(), v.to_string());
                        }
                        results.push(map);
                    }
                    Err(e) => return Err(Error::invalid_data(format!("MongoDB查询错误: {}", e))),
                }
            }
            
            // 转换为DataBatch
            Self::convert_results(results)
        } else {
            Err(Error::not_found("MongoDB未连接"))
        }
    }
    
    fn get_schema<'life0, 'life1>(&'life0 self, table_name: Option<&'life1 str>) -> Box<dyn std::future::Future<Output = Result<DataSchema>> + Send + 'life0>
    where
        'life1: 'life0,
    {
        self.get_schema(table_name)
    }
    
    async fn write_data(&self, batch: &DataBatch, table_name: &str, mode: WriteMode) -> Result<usize> {
        if let Some(client) = &self.client {
            // 获取数据库名
            let db_name = self.active_database.as_str();
            
            // 获取集合名
            let collection_name = Self::parse_table_name(table_name);
            
            // 将DataBatch转换为文档列表
            let documents = self.convert_batch_to_documents(batch)?;
            
            // 获取集合
            let collection = client.database(db_name).collection::<Document>(collection_name);
            
            match mode {
                WriteMode::Insert => {
                    // 将文档转换为BSON
                    let mut bson_docs = Vec::new();
                    for doc in documents {
                        let mut bson_doc = Document::new();
                        for (k, v) in doc {
                            bson_doc.insert(k, v);
                        }
                        bson_docs.push(bson_doc);
                    }
                    
                    // 插入文档
                    let result = collection.insert_many(bson_docs, None).await?;
                    
                    Ok(result.inserted_ids.len())
                },
                WriteMode::Update => {
                    // 需要有ID字段来更新
                    let mut count = 0;
                    
                    for doc in documents {
                        if let Some(id_str) = doc.get("_id") {
                            // 提取ID
                            let filter = doc! { "_id": id_str };
                            
                            // 创建更新文档
                            let mut update_doc = Document::new();
                            for (k, v) in doc {
                                if k != "_id" {
                                    update_doc.insert(k, v);
                                }
                            }
                            
                            let update = doc! { "$set": update_doc };
                            
                            // 更新文档
                            let result = collection.update_one(filter, update, None).await?;
                            
                            count += result.modified_count as usize;
                        }
                    }
                    
                    Ok(count)
                },
                WriteMode::Upsert => {
                    // 需要有ID字段来upsert
                    let mut count = 0;
                    
                    for doc in documents {
                        if let Some(id_str) = doc.get("_id") {
                            // 提取ID
                            let filter = doc! { "_id": id_str };
                            
                            // 创建更新文档
                            let mut update_doc = Document::new();
                            for (k, v) in doc {
                                if k != "_id" {
                                    update_doc.insert(k, v);
                                }
                            }
                            
                            let update = doc! { "$set": update_doc };
                            
                            // 设置upsert选项
                            let mut options = UpdateOptions::default();
                            options.upsert = Some(true);
                            
                            // 更新文档
                            let result = collection.update_one(filter, update, options).await?;
                            
                            count += (result.modified_count + result.upserted_id.map(|_| 1).unwrap_or(0)) as usize;
                        } else {
                            // 没有ID，执行插入
                            let mut bson_doc = Document::new();
                            for (k, v) in doc {
                                bson_doc.insert(k, v);
                            }
                            
                            let result = collection.insert_one(bson_doc, None).await?;
                            
                            if result.inserted_id.as_object_id().is_some() {
                                count += 1;
                            }
                        }
                    }
                    
                    Ok(count)
                },
                WriteMode::Replace => {
                    // 删除集合中的所有文档
                    let _delete_result = collection.delete_many(Document::new(), None).await?;
                    
                    // 将文档转换为BSON
                    let mut bson_docs = Vec::new();
                    for doc in documents {
                        let mut bson_doc = Document::new();
                        for (k, v) in doc {
                            bson_doc.insert(k, v);
                        }
                        bson_docs.push(bson_doc);
                    }
                    
                    // 插入文档
                    let result = collection.insert_many(bson_docs, None).await?;
                    
                    Ok(result.inserted_ids.len())
                },
                WriteMode::Append => {
                    // 将文档转换为BSON
                    let mut bson_docs = Vec::new();
                    for doc in documents {
                        let mut bson_doc = Document::new();
                        for (k, v) in doc {
                            bson_doc.insert(k, v);
                        }
                        bson_docs.push(bson_doc);
                    }
                    
                    // 插入文档
                    let result = collection.insert_many(bson_docs, None).await?;
                    
                    Ok(result.inserted_ids.len())
                },
            }
        } else {
            Err(Error::not_found("MongoDB未连接"))
        }
    }
    
    fn get_type(&self) -> DatabaseType {
        DatabaseType::MongoDB
    }
    
    fn get_config(&self) -> &DatabaseConfig {
        &self.config
    }
} */ 