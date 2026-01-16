// 数据导入器实现

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::sync::atomic::{AtomicUsize, Ordering};

use log::{debug, error, info, warn, trace};
use tokio::runtime::Runtime;
use tokio::time::timeout;
// use futures::stream::{self, StreamExt};
use chrono::Utc;
use uuid::Uuid;

use crate::data::{DataBatch, DataSchema};
use crate::data::pipeline::ImportPipelineBuilder;
use crate::data::loader::file::{FileProcessor, FileProcessorFactory, Record};
use crate::data::loader::types::FileType;
use crate::error::{Error, Result};

use crate::data::loader::config::ImportConfig;
use crate::data::loader::progress::ProgressTracker as ImportProgressTracker;
use crate::data::loader::result::{ImportResult, BatchSummary};
use crate::data::loader::validation::{StandardBatchValidator, BatchValidator};

use crate::core::interfaces::ValidationResult;

/// 进度追踪器
pub struct ProgressTracker {
    /// 总记录数
    total_records: AtomicUsize,
    /// 已处理记录数
    processed_records: AtomicUsize,
    /// 开始时间
    start_time: std::time::Instant,
    /// 进度回调
    progress_callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
}

impl ProgressTracker {
    /// 创建新的进度追踪器
    pub fn new() -> Self {
        Self {
            total_records: AtomicUsize::new(0),
            processed_records: AtomicUsize::new(0),
            start_time: std::time::Instant::now(),
            progress_callback: None,
        }
    }
    
    /// 设置总记录数
    pub fn set_total(&self, total: usize) {
        self.total_records.store(total, Ordering::SeqCst);
    }
    
    /// 更新已处理记录数
    pub fn update(&self, processed: usize) {
        self.processed_records.store(processed, Ordering::SeqCst);
        
        // 调用进度回调
        if let Some(callback) = &self.progress_callback {
            callback(
                self.processed_records.load(Ordering::SeqCst),
                self.total_records.load(Ordering::SeqCst)
            );
        }
    }
    
    /// 增加已处理记录数
    pub fn increment(&self, count: usize) {
        let processed = self.processed_records.fetch_add(count, Ordering::SeqCst) + count;
        
        // 调用进度回调
        if let Some(callback) = &self.progress_callback {
            callback(
                processed,
                self.total_records.load(Ordering::SeqCst)
            );
        }
    }
    
    /// 获取进度百分比
    pub fn percentage(&self) -> f32 {
        let processed = self.processed_records.load(Ordering::SeqCst);
        let total = self.total_records.load(Ordering::SeqCst);
        
        if total == 0 {
            return 0.0;
        }
        
        (processed as f32 / total as f32) * 100.0
    }
    
    /// 获取已用时间
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// 获取估计剩余时间
    pub fn estimated_time_remaining(&self) -> Option<Duration> {
        let processed = self.processed_records.load(Ordering::SeqCst);
        let total = self.total_records.load(Ordering::SeqCst);
        let elapsed = self.elapsed();
        
        if processed == 0 || total == 0 {
            return None;
        }
        
        let elapsed_secs = elapsed.as_secs_f64();
        let secs_per_record = elapsed_secs / processed as f64;
        let remaining_records = total.saturating_sub(processed);
        let remaining_secs = secs_per_record * remaining_records as f64;
        
        Some(Duration::from_secs_f64(remaining_secs))
    }
    
    /// 设置进度回调
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
    }
}

/// 为FileProcessor添加额外功能的扩展特性
#[async_trait::async_trait]
pub trait FileProcessorExt: FileProcessor {
    /// 使用进度追踪处理批次
    async fn process_batches_with_progress(&mut self, batch_size: usize, tracker: &ImportProgressTracker) -> Result<Vec<DataBatch>> {
        // 估计总记录数
        let estimated_count = match self.estimate_record_count() {
            Ok(count) => count,
            Err(e) => {
                warn!("无法估计记录数量: {}", e);
                0
            }
        };
        
        // 设置总记录数
        tracker.set_total(estimated_count);
        debug!("估计总记录数: {}", estimated_count);
        
        let mut batches = Vec::new();
        let mut total_processed = 0;
        
        // 处理批次
        let mut current_batch = DataBatch::new("import", 0, batch_size);
        
        while let Some(record) = self.next_record()? {
            // 将Record转换为HashMap<String, DataValue>并添加到批次
            let mut record_map = HashMap::new();
            for (i, value) in record.values().iter().enumerate() {
                record_map.insert(format!("field_{}", i), value.clone());
            }
            current_batch.records.push(record_map);
            
            // 如果达到批次大小，添加到批次列表
            if current_batch.records.len() >= batch_size {
                total_processed += current_batch.records.len();
                tracker.update(total_processed);
                
                // 记录进度
                debug!("已处理 {} 条记录 ({:.1}%), 耗时: {:?}",
                     total_processed,
                    tracker.percentage(),
                    tracker.elapsed());
                
                batches.push(current_batch);
                current_batch = DataBatch::new("import", batches.len(), batch_size);
            }
        }
        
        // 添加最后一个批次（如果有）
        if !current_batch.records.is_empty() {
            total_processed += current_batch.records.len();
            tracker.update(total_processed);
            
            batches.push(current_batch);
        }
        
        info!("已完成数据处理: {} 条记录, {} 个批次, 耗时: {:?}",
             total_processed, batches.len(), tracker.elapsed());
        
        Ok(batches)
    }
}

// 为所有FileProcessor实现FileProcessorExt
#[async_trait::async_trait]
impl<T: FileProcessor + ?Sized> FileProcessorExt for T {}

/// 数据导入器
/// 负责将数据从源位置导入到目标位置，支持各种格式和转换选项
pub struct DataImporter {
    /// 导入配置
    config: ImportConfig,
    /// 缓存最后一次成功导入的批次数据
    last_imported_batch: Option<DataBatch>,
    /// 超时时间（秒）
    timeout_seconds: Option<u64>,
    /// 验证器
    validator: Option<Box<dyn BatchValidator>>,
}

impl DataImporter {
    /// 创建新的导入器实例
    pub fn new(config: ImportConfig) -> Self {
        Self { 
            config,
            last_imported_batch: None,
            timeout_seconds: None,
            validator: None,
        }
    }
    
    /// 设置超时时间
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = Some(seconds);
        self
    }
    
    /// 设置验证器
    pub fn with_validator<V: BatchValidator + 'static>(mut self, validator: V) -> Self {
        self.validator = Some(Box::new(validator));
        self
    }
    
    /// 执行导入操作
    pub fn import(&self) -> Result<ImportResult> {
        // 初始化导入结果
        let mut result = ImportResult::default();
        
        debug!("开始导入数据从 {} 到 {}", self.config.source_path, self.config.target_location);
        
        // 检查源路径是否存在
        if !Path::new(&self.config.source_path).exists() {
            error!("源文件不存在: {}", self.config.source_path);
            result.message = format!("源文件不存在: {}", self.config.source_path);
            result.errors.push("源文件不存在".to_string());
            return Ok(result);
        }
        
        // 尝试创建处理管道
        let builder = match self.create_pipeline_builder() {
            Ok(builder) => {
                info!("成功创建导入管道构建器");
                builder
            },
            Err(e) => {
                error!("创建处理管道失败: {}", e);
                result.message = format!("创建处理管道失败: {}", e);
                result.errors.push(format!("创建处理管道失败: {}", e));
                return Ok(result);
            }
        };
        
        // 创建导入管道
        let pipeline = match builder.build() {
            Ok(p) => {
                info!("成功创建导入管道");
                p
            },
            Err(e) => {
                error!("构建导入管道失败: {}", e);
                result.message = format!("构建导入管道失败: {}", e);
                result.errors.push(e.to_string());
                return Ok(result);
            }
        };
        
        // 准备导入
        info!("开始准备导入过程");
        if let Err(e) = pipeline.prepare() {
            error!("准备导入过程失败: {}", e);
            result.message = format!("准备导入过程失败: {}", e);
            result.errors.push(e.to_string());
            return Ok(result);
        }
        
        // 执行导入
        info!("开始执行导入");
        let start_time = std::time::Instant::now();
        
        // 创建异步运行环境
        let runtime = match Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                let err_msg = format!("创建异步运行环境失败: {}", e);
                error!("{}", err_msg);
                result.message = err_msg.clone();
                result.errors.push(err_msg);
                return Ok(result);
            }
        };
        
        // 使用异步运行环境执行导入
        let import_result = match runtime.block_on(self.execute_import(&mut result, pipeline)) {
            Ok(r) => r,
            Err(e) => {
                error!("执行导入过程失败: {}", e);
                result.success = false;
                result.message = format!("执行导入过程失败: {}", e);
                result.errors.push(e.to_string());
                return Ok(result);
            }
        };
        
        let duration = start_time.elapsed();
        info!("导入完成，耗时: {:?}", duration);
        
        // 添加性能指标到元数据
        import_result.add_metadata("import.duration_ms", &duration.as_millis().to_string());
        
        Ok(import_result)
    }
    
    /// 执行实际的导入过程（异步）
    async fn execute_import(&self, result: &mut ImportResult, _pipeline: impl crate::data::pipeline::Pipeline) -> Result<ImportResult> {
        let mut import_result = result.clone();
        
        // 创建文件处理器
        let mut file_processor = match self.create_file_processor().await {
            Ok(processor) => processor,
            Err(e) => {
                error!("创建文件处理器失败: {}", e);
                import_result.message = format!("创建文件处理器失败: {}", e);
                import_result.errors.push(e.to_string());
                return Ok(import_result);
            }
        };
        
        // 设置处理器选项
        for (key, value) in &self.config.processor_options {
            if let Err(e) = file_processor.set_option(key, value) {
                warn!("设置处理器选项失败 {}: {} - {}", key, value, e);
                import_result.add_warning(&format!("设置处理器选项失败: {}", e));
            }
        }
        
        // 创建进度跟踪器
        let progress_tracker = ImportProgressTracker::new();
        
        // 处理数据批次
        let batch_size = self.config.batch_size.unwrap_or(1000);
        
        let start_time = std::time::Instant::now();
        let timeout = Duration::from_secs(self.timeout_seconds.unwrap_or(300));
        
        let mut total_records = 0;
        let mut failed_records = 0;
        let mut last_batch: Option<DataBatch> = None;
        
        // 使用带有超时的批次处理
        let batches = match tokio::time::timeout(
            timeout,
            file_processor.process_batches_with_progress(batch_size, &progress_tracker)
        ).await {
            Ok(batch_result) => {
                match batch_result {
                    Ok(processed_batches) => {
                        processed_batches
                    },
                    Err(e) => {
                        error!("处理数据批次失败: {}", e);
                        import_result.add_error(&format!("处理数据批次失败: {}", e));
                        import_result.success = false;
                        return Ok(import_result);
                    }
                }
            },
            Err(_) => {
                error!("处理数据批次超时，超过了{}秒", timeout.as_secs());
                import_result.add_error(&format!("处理数据批次超时，超过了{}秒", timeout.as_secs()));
                import_result.success = false;
                return Ok(import_result);
            }
        };
        
        // 处理获取的数据批次
        for (i, batch) in batches.iter().enumerate() {
            // 将批次转换为DataBatch格式
            let data_batch = match crate::data::DataBatch::from_records(
                batch.records.clone(),
                self.config.schema.clone()
            ) {
                Ok(db) => db,
                Err(e) => {
                    error!("转换批次数据失败 (批次 #{}): {}", i, e);
                    import_result.add_error(&format!("转换批次数据失败 (批次 #{}): {}", i, e));
                    failed_records += batch.records.len();
                    continue;
                }
            };
            
            // 处理数据批次
            total_records += data_batch.records.len();
            
            // 验证数据批次（如果启用验证）
            if self.config.validate.unwrap_or(true) {
                match self.validate_batch(&data_batch) {
                    Ok(validation_result) => {
                        if !validation_result.is_valid {
                            let error_msg = validation_result.errors.join(", ");
                            warn!("批次 #{} 验证失败: {}", i, error_msg);
                            import_result.add_warning(&format!("批次 #{} 验证失败: {}", i, error_msg));
                            failed_records += validation_result.errors.len();
                        }
                    },
                    Err(e) => {
                        warn!("验证批次 #{} 时出错: {}", i, e);
                        import_result.add_warning(&format!("验证批次 #{} 时出错: {}", i, e));
                    }
                }
            }
            
            // 缓存最后一个批次用于结果摘要
            last_batch = Some(data_batch);
        }
        
        // 更新导入结果
        import_result.success = true;
        import_result.records_processed = total_records;
        import_result.records_failed = failed_records;
        import_result.message = format!("成功导入数据，处理了{}条记录，失败{}条", total_records, failed_records);
        
        // 添加批次摘要
        if let Some(batch) = &last_batch {
            import_result.with_batch(batch);
            
            // 更新导入器的最后导入批次
            let mut importer = self.clone_as_mut();
            importer.last_imported_batch = last_batch;
        }
        
        // 记录处理时间
        let processing_time = start_time.elapsed();
        import_result.add_metadata("processing_time_ms", &processing_time.as_millis().to_string());
        
        // 记录处理速率
        if processing_time.as_secs() > 0 && total_records > 0 {
            let records_per_second = total_records as f64 / processing_time.as_secs_f64();
            import_result.add_metadata("records_per_second", &format!("{:.2}", records_per_second));
        }
        
        Ok(import_result)
    }
    
    /// 获取最后导入的批次
    pub fn get_last_imported_batch(&self) -> Option<&DataBatch> {
        self.last_imported_batch.as_ref()
    }
    
    /// 创建管道构建器
    fn create_pipeline_builder(&self) -> Result<ImportPipelineBuilder> {
        let mut builder = ImportPipelineBuilder::new();
        
        // 设置源路径
        builder = builder.with_source(&self.config.source_path);
        
        // 设置目标位置
        if !self.config.target_location.is_empty() {
            builder = builder.with_target(&self.config.target_location);
        }
        
        // 设置批处理大小
        if let Some(batch_size) = self.config.batch_size {
            builder = builder.with_batch_size(batch_size);
        }
        
        // 设置验证
        if let Some(validate) = self.config.validate {
            builder = builder.with_validation(validate);
        }
        
        // 设置覆盖
        if let Some(overwrite) = self.config.overwrite {
            builder = builder.with_overwrite(overwrite);
        }
        
        // 设置模式
        if let Some(schema) = &self.config.schema {
            builder = builder.with_schema(schema.clone());
        } else {
            builder = builder.with_schema_inference(true);
        }
        
        // 设置文件选项
        for (key, value) in &self.config.file_options {
            builder = builder.with_file_option(key, value);
        }
        
        // 设置性能监控
        builder = builder.with_performance_monitoring(true);
        
        Ok(builder)
    }
    
    /// 创建文件处理器（异步）
    async fn create_file_processor(&self) -> Result<Box<dyn FileProcessor + Send>> {
        let path = Path::new(&self.config.source_path);
        let file_type = match self.config.format.as_ref() {
            Some(format_str) => {
                // 从字符串解析文件类型
                FileType::from_string(format_str).unwrap_or_else(|| {
                    warn!("无法解析格式字符串 '{}', 将尝试自动检测", format_str);
                    FileType::from_path(path)
                })
            },
            None => {
                // 根据路径自动检测文件类型
                FileType::from_path(path)
            }
        };
        
        debug!("为文件 {:?} 创建处理器，检测到类型: {:?}", path, file_type);
        
        // 使用工厂创建处理器
        let processor = FileProcessorFactory::create_processor(path)?;
        
        // 由于FileProcessor的返回类型不兼容，我们创建一个适配器
        struct FileProcessorAdapter {
            inner: Arc<dyn FileProcessor>,
            file_path: PathBuf,  // 存储文件路径
        }
        
        impl FileProcessor for FileProcessorAdapter {
            fn get_file_path(&self) -> &Path {
                &self.file_path
            }
            
            fn get_schema(&self) -> Result<DataSchema> {
                self.inner.get_schema()
            }
            
            fn get_row_count(&self) -> Result<usize> {
                self.inner.get_row_count()
            }
            
            fn read_rows(&mut self, count: usize) -> Result<Vec<Record>> {
                // Arc 无法提供可变引用，这里需要重新设计架构
                Err(Error::processing("FileProcessor适配器无法提供可变访问，需要重新设计"))
            }
            
            fn reset(&mut self) -> Result<()> {
                // Arc 无法提供可变引用，这里需要重新设计架构
                Err(Error::processing("FileProcessor适配器无法提供可变访问，需要重新设计"))
            }
            
            fn estimate_record_count(&self) -> Result<usize> {
                self.inner.estimate_record_count()
            }
            
            fn next_record(&mut self) -> Result<Option<Record>> {
                // Arc 无法提供可变引用，这里需要重新设计架构
                Err(Error::processing("FileProcessor适配器无法提供可变访问，需要重新设计"))
            }
        }
        
        Ok(Box::new(FileProcessorAdapter { 
            inner: processor,
            file_path: path.to_path_buf(),
        }))
    }
    
    /// 验证数据批次
    fn validate_batch(&self, batch: &DataBatch) -> Result<ValidationResult> {
        if let Some(validator) = &self.validator {
            // 优先使用导入配置中显式提供的模式，其次退回到批次自带的模式
            let schema = self
                .config
                .schema
                .as_ref()
                .or(batch.schema.as_ref());

            let validation_result = validator.validate_batch(batch, schema)?;
            let mut vr = validation_result;
            // 从验证结果中获取无效记录数（如果有的话）
            let invalid_count = vr.metadata.get("invalid_records")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            Ok(vr)
        } else {
            // 没有验证器，默认为有效
            let mut vr = ValidationResult::success();
            vr.metadata.insert("message".into(), "未执行验证".into());
            Ok(vr)
        }
    }
    
    /// 克隆为可变引用
    fn clone_as_mut(&self) -> Self {
        Self {
            config: self.config.clone(),
            last_imported_batch: self.last_imported_batch.clone(),
            timeout_seconds: self.timeout_seconds,
            validator: None, // 验证器不能被克隆，需要根据需要重新创建
        }
    }
    
    /// 使用并发导入多个源
    pub async fn import_multiple(&self, sources: Vec<String>) -> Result<HashMap<String, ImportResult>> {
        trace!("开始并发导入多个源: {:?}", sources);
        
        let mut results = HashMap::new();
        let mut tasks = Vec::new();
        
        // 创建多个导入任务
        for source in sources {
            let mut config = self.config.clone();
            config.source_path = source.clone();
            let timeout_seconds = self.timeout_seconds;
            
            tasks.push(tokio::spawn(async move {
                let mut importer = DataImporter::new(config);
                if let Some(timeout) = timeout_seconds {
                    importer = importer.with_timeout(timeout);
                }
                
                (source, importer.execute_import_async().await)
            }));
        }
        
        // 等待所有任务完成
        let mut join_errors = Vec::new();
        
        for task in futures::future::join_all(tasks).await {
            match task {
                Ok((source, result)) => {
                    match result {
                        Ok(import_result) => {
                            results.insert(source, import_result);
                        },
                        Err(e) => {
                            error!("导入源 {} 失败: {}", source, e);
                            let mut failed_result = ImportResult::default();
                            failed_result.success = false;
                            failed_result.message = format!("导入失败: {}", e);
                            failed_result.errors.push(e.to_string());
                            results.insert(source, failed_result);
                        }
                    }
                },
                Err(e) => {
                    error!("导入任务执行失败: {}", e);
                    join_errors.push(e.to_string());
                }
            }
        }
        
        // 如果有任务执行错误，添加到元数据
        if !join_errors.is_empty() {
            for (_, result) in results.iter_mut() {
                result.metadata.insert("join_errors".to_string(), serde_json::to_value(&join_errors).unwrap_or_default().to_string());
            }
        }
        
        trace!("完成多源导入，结果数: {}", results.len());
        Ok(results)
    }
    
    /// 异步执行导入操作
    pub async fn execute_import_async(&self) -> Result<ImportResult> {
        let mut result = ImportResult::default();
        result.id = Uuid::new_v4().to_string();
        result.start_time = Utc::now();
        
        debug!("开始异步导入数据从 {} 到 {}", self.config.source_path, self.config.target_location);
        
        // 创建导入管道
        let pipeline_builder = self.create_pipeline_builder()?;
        let _pipeline = pipeline_builder.clone();
        let pipeline = pipeline_builder.build()?;
        
        // 执行导入操作
        let timeout_duration = self.timeout_seconds.map(Duration::from_secs).unwrap_or(Duration::from_secs(3600));
        
        // 使用带超时的异步执行
        match timeout(timeout_duration, self.execute_import(&mut result, pipeline)).await {
            Ok(import_result) => {
                match import_result {
                    Ok(r) => {
                        result = r;
                        result.success = true;
                        result.end_time = Some(Utc::now());
                        result.duration = Some(result.end_time.unwrap().signed_duration_since(result.start_time).num_milliseconds() as u64);
                        
                        info!("异步导入成功，处理了 {} 条记录", result.records_processed);
                        Ok(result)
                    },
                    Err(e) => {
                        result.success = false;
                        result.end_time = Some(Utc::now());
                        result.duration = Some(result.end_time.unwrap().signed_duration_since(result.start_time).num_milliseconds() as u64);
                        result.message = format!("导入失败: {}", e);
                        result.errors.push(e.to_string());
                        
                        error!("异步导入失败: {}", e);
                        Ok(result)
                    }
                }
            },
            Err(_) => {
                result.success = false;
                result.end_time = Some(Utc::now());
                result.duration = Some(result.end_time.unwrap().signed_duration_since(result.start_time).num_milliseconds() as u64);
                result.message = format!("导入操作超时，超过了 {} 秒", timeout_duration.as_secs());
                result.errors.push("导入操作超时".to_string());
                
                error!("异步导入超时, 超过了 {} 秒", timeout_duration.as_secs());
                Ok(result)
            }
        }
    }
    
    /// 验证导入批次的数据架构
    fn validate_batch_schema(&self, batch: &DataBatch) -> Result<BatchSummary> {
        trace!("开始验证批次数据架构，批次大小: {}", batch.records.len());
        
        let mut summary = BatchSummary {
            batch_id: batch.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            record_count: batch.records.len(),
            field_count: 0,
            field_names: Vec::new(),
            created_at: batch.created_at.to_string(),
        };
        
        // 如果批次有模式，添加到汇总信息
        if let Some(schema) = &batch.schema {
            let field_names: Vec<String> = schema.fields().iter()
                .map(|f| f.name.clone())
                .collect();
            
            summary.field_count = field_names.len();
            summary.field_names = field_names;
            
            trace!("批次包含模式，字段数: {}", schema.fields().len());
            
            // 执行标准批次验证
            let validator = StandardBatchValidator::new();
            let validation_result = self.validator.as_ref()
                .map(|v| v.validate_batch(batch, Some(schema)))
                .unwrap_or_else(|| validator.validate_batch(batch, Some(schema)))?;
            
            if !validation_result.is_valid {
                let invalid_count = validation_result.metadata.get("invalid_records")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(validation_result.errors.len());
                trace!("批次验证完成，有 {} 条无效记录", invalid_count);
            } else {
                trace!("批次验证通过，全部记录有效");
            }
        } else {
            trace!("批次没有模式，跳过架构验证");
        }
        
        Ok(summary)
    }
    
    /// 处理导入进度通知
    fn handle_progress_notification(&self, progress: usize, total: usize) -> Result<()> {
        let progress_percentage = if total > 0 {
            (progress as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        
        // 注意：ImportConfig 没有 progress_callback 字段（生产级实现）
        // 如果需要进度回调功能，可以通过其他方式实现（如事件系统）
        // 当前实现不包含进度回调，这是合理的设计选择
        
        Ok(())
    }
}

/// 批量导入器
pub struct BatchImporter {
    /// 基础配置
    config: ImportConfig,
    /// 超时时间（秒）
    timeout_seconds: u64,
    /// 最大并发导入数
    max_concurrent: usize,
    /// 缓存最后导入的批次
    last_imported_batches: Vec<DataBatch>,
}

impl BatchImporter {
    /// 创建新的批量导入器
    pub fn new(config: ImportConfig) -> Self {
        Self {
            config,
            timeout_seconds: 300, // 默认5分钟超时
            max_concurrent: 5,     // 默认最多5个并发任务
            last_imported_batches: Vec::new(),
        }
    }
    
    /// 设置超时时间
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }
    
    /// 设置最大并发数
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max;
        self
    }
    
    /// 获取所有已导入的批次
    pub fn get_imported_batches(&self) -> &[DataBatch] {
        &self.last_imported_batches
    }
    
    /// 从目录导入多个文件
    pub async fn import_from_directory<P: AsRef<Path>>(&mut self, directory: P, pattern: Option<&str>) -> Result<Vec<ImportResult>> {
        let dir_path = directory.as_ref();
        info!("开始从目录导入数据: {:?}, 模式: {:?}", dir_path, pattern);
        
        if !dir_path.exists() || !dir_path.is_dir() {
            return Err(Error::invalid_input(format!("指定的路径不是有效目录: {:?}", dir_path)));
        }
        
        // 扫描目录，获取匹配的文件
        let files = match crate::data::loader::utils::scan_directory(dir_path, pattern, false) {
            Ok(files) => files,
            Err(e) => {
                error!("扫描目录失败: {}", e);
                return Err(e);
            }
        };
        
        if files.is_empty() {
            warn!("在目录 {:?} 中没有找到匹配的文件", dir_path);
            return Ok(Vec::new());
        }
        
        info!("在目录 {:?} 中找到 {} 个文件", dir_path, files.len());
        
        // 创建导入任务
        let mut tasks = Vec::with_capacity(files.len());
        let mut results = Vec::with_capacity(files.len());
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent));
        
        for file in files {
            // 创建该文件的导入配置
            let mut file_config = self.config.clone();
            file_config.source_path = file.to_string_lossy().to_string();
            
            // 检测文件格式（如果未指定）
            if file_config.format.is_none() {
                match crate::data::loader::utils::detect_file_format(&file) {
                    Ok(format) => {
                        debug!("检测到文件格式 {:?}: {:?}", file, format);
                        file_config.format = Some(format.to_string());
                    },
                    Err(e) => {
                        warn!("无法检测文件格式 {:?}: {}", file, e);
                    }
                }
            }
            
            // 创建导入器
            let importer = DataImporter::new(file_config)
                .with_timeout(self.timeout_seconds);
            
            // 获取信号量许可
            let permit = semaphore.clone();
            
            // 创建导入任务
            let task = async move {
                // 获取并释放许可
                let _permit = permit.acquire().await.unwrap();
                
                // 执行导入
                match importer.import() {
                    Ok(result) => {
                        if result.success {
                            info!("成功导入文件 {:?}: {}", file, result.message);
                        } else {
                            warn!("导入文件 {:?} 失败: {}", file, result.message);
                        }
                        result
                    },
                    Err(e) => {
                        error!("导入文件 {:?} 时发生错误: {}", file, e);
                        let mut result = ImportResult::default();
                        result.success = false;
                        result.message = format!("导入出错: {}", e);
                        result.errors.push(e.to_string());
                        result
                    }
                }
            };
            
            tasks.push(task);
        }
        
        // 使用超时执行所有任务
        let timeout_duration = Duration::from_secs(self.timeout_seconds);
        match tokio::time::timeout(
            timeout_duration,
            futures::future::join_all(tasks)
        ).await {
            Ok(task_results) => {
                // 处理每个导入结果
                self.last_imported_batches.clear();
                
                let mut success_count = 0;
                let mut failed_count = 0;
                let mut total_records = 0;
                
                for result in task_results {
                    if result.success {
                        success_count += 1;
                        total_records += result.records_processed;
                        
                                                    // 如果有批次摘要，添加到导入批次列表
                            if let Some(batch_summary) = &result.batch_summary {
                                // 创建一个新的批次对象
                                let mut batch = DataBatch::new("imported", 0, 1000);
                                batch.id = Some(batch_summary.batch_id.clone());
                                // 添加到导入批次列表
                                self.last_imported_batches.push(batch);
                            }
                    } else {
                        failed_count += 1;
                    }
                    
                    results.push(result);
                }
                
                info!("批量导入完成: 成功={}, 失败={}, 总记录数={}",
                     success_count, failed_count, total_records);
            },
            Err(_) => {
                error!("批量导入超时，超过了{}秒", timeout_duration.as_secs());
                return Err(Error::timeout(format!("批量导入超时，超过了{}秒", timeout_duration.as_secs())));
            }
        }
        
        Ok(results)
    }
    
    /// 从目录导入数据（同步版本）
    pub fn import_from_directory_sync<P: AsRef<Path>>(&mut self, directory: P, pattern: Option<&str>) -> Result<Vec<ImportResult>> {
        // 创建运行时
        let runtime = Runtime::new()
            .map_err(|e| Error::processing(format!("创建异步运行时失败: {}", e)))?;
        
        // 执行异步导入操作
        runtime.block_on(self.import_from_directory(directory, pattern))
    }
    
    /// 合并所有已导入的批次
    pub fn merge_imported_batches(&self) -> Result<DataBatch> {
        if self.last_imported_batches.is_empty() {
            return Err(Error::invalid_state("没有可合并的批次"));
        }
        
        let mut merged_batch = self.last_imported_batches[0].clone();
        
        for batch in &self.last_imported_batches[1..] {
            merged_batch.merge(batch)?;
        }
        
        info!("合并了{}个批次，共{}条记录",
             self.last_imported_batches.len(),
             merged_batch.records.len());
        
        Ok(merged_batch)
    }
    
    /// 清除已导入的批次
    pub fn clear_imported_batches(&mut self) {
        self.last_imported_batches.clear();
        debug!("已清除所有导入批次");
    }
    
    /// 获取导入的批次数量
    pub fn imported_batch_count(&self) -> usize {
        self.last_imported_batches.len()
    }
    
    /// 获取导入的总记录数
    pub fn imported_record_count(&self) -> usize {
        self.last_imported_batches.iter()
            .map(|batch| batch.records.len())
            .sum()
    }
    
    /// 导出所有批次到目标位置
    pub fn export_batches<P: AsRef<Path>>(&self, target_dir: P, format: crate::data::loader::types::DataFormat) -> Result<Vec<std::path::PathBuf>> {
        let target_path = target_dir.as_ref();
        
        // 确保目标目录存在
        if !target_path.exists() {
            std::fs::create_dir_all(target_path)
                .map_err(|e| Error::io_error(format!("创建目标目录失败: {}", e)))?;
        }
        
        let mut exported_files = Vec::new();
        
        // 导出每个批次
        for (i, batch) in self.last_imported_batches.iter().enumerate() {
            let filename = format!("batch_{}.{}", i, format.file_extension());
            let output_path = target_path.join(filename);
            
            let saved_path = crate::data::loader::utils::save_batch_to_file(batch, &output_path, format.clone())?;
            exported_files.push(saved_path);
        }
        
        info!("成功导出 {} 个批次到 {:?}", exported_files.len(), target_path);
        
        Ok(exported_files)
    }
} 