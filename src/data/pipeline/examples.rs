// 数据流水线使用示例

use crate::data::pipeline::*;
use crate::data::DataBatch;
use crate::error::Result;
use std::collections::HashMap;

/// 基本的数据流水线使用示例
pub fn basic_pipeline_example() -> Result<()> {
    // 创建数据流水线配置
    let config = PipelineConfig {
        batch_size: 32,
        parallelism: 4,
        cache_enabled: true,
        shuffle: true,
        validation_split: 0.2,
        ..Default::default()
    };

    // 创建数据流水线
    let mut pipeline = DataPipeline::new(config);

    // 添加数据验证器
    pipeline.add_validator(BasicValidator::new(1, 100));

    // 创建测试数据
    let data = DataBatch {
        id: Some("test_batch".to_string()),
        features: vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ],
        labels: Some(vec![
            vec![0.0],
            vec![1.0],
            vec![0.0],
        ]),
        weights: Some(vec![1.0, 1.0, 1.0]),
        metadata: Some(HashMap::new()),
    };

    // 计算数据统计信息
    pipeline.calculate_stats(&data)?;

    // 处理数据
    let processed_data = pipeline.process_batch(data)?;

    // 生成数据质量报告
    let quality_report = pipeline.generate_quality_report(&processed_data)?;
    println!("数据完整度: {:.2}%", quality_report.completeness * 100.0);

    // 可视化数据
    let histogram = pipeline.visualize_data(&processed_data, VisualizationType::Histogram)?;
    let correlation = pipeline.visualize_data(&processed_data, VisualizationType::Correlation)?;

    // 输出可视化结果
    let histogram_json = pipeline.visualization_to_json(&histogram)?;
    let correlation_json = pipeline.visualization_to_json(&correlation)?;

    println!("直方图可视化: {}", histogram_json);
    println!("相关性矩阵可视化: {}", correlation_json);

    Ok(())
}

/// 异常检测示例
pub fn anomaly_detection_example() -> Result<()> {
    // 创建包含异常的测试数据
    let data = DataBatch {
        id: Some("anomaly_test".to_string()),
        features: vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![100.0, 200.0, 300.0],  // 异常值
            vec![4.0, 5.0, f32::NAN],   // 缺失值
        ],
        labels: None,
        weights: None,
        metadata: None,
    };
    
    // 创建Z-Score异常检测器
    let detector = ZScoreAnomalyDetector::new(3.0);
    
    // 检测异常
    let anomalies = detector.detect_anomalies(&data)?;
    
    // 输出异常信息
    println!("检测到 {} 个异常", anomalies.len());
    for anomaly in &anomalies {
        println!("{:?} 在特征 {} 行 {}: {}", 
            anomaly.anomaly_type,
            anomaly.feature_index,
            anomaly.record_index,
            anomaly.description);
    }
    
    Ok(())
}

/// 时序数据处理示例
pub fn time_series_example() -> Result<()> {
    // 创建时序数据（10个时间点，3个特征）
    let data = DataBatch {
        id: Some("time_series".to_string()),
        features: vec![
            vec![1.0, 10.0, 100.0],  // t=1
            vec![2.0, 20.0, 200.0],  // t=2
            vec![3.0, 30.0, 300.0],  // t=3
            vec![4.0, 40.0, 400.0],  // t=4
            vec![5.0, 50.0, 500.0],  // t=5
            vec![6.0, 60.0, 600.0],  // t=6
            vec![7.0, 70.0, 700.0],  // t=7
            vec![8.0, 80.0, 800.0],  // t=8
            vec![9.0, 90.0, 900.0],  // t=9
            vec![10.0, 100.0, 1000.0],  // t=10
        ],
        labels: Some(vec![
            vec![0.0], vec![0.0], vec![0.0], vec![0.0], vec![0.0],
            vec![1.0], vec![1.0], vec![1.0], vec![1.0], vec![1.0],
        ]),
        weights: None,
        metadata: None,
    };
    
    // 创建时序特征提取器配置
    let ts_config = TimeSeriesConfig {
        window_size: 5,  // 5个时间点的窗口
        stride: 1,       // 步长为1
        features: vec![
            TimeSeriesFeatureType::Mean,
            TimeSeriesFeatureType::Std,
            TimeSeriesFeatureType::Min,
            TimeSeriesFeatureType::Max,
            TimeSeriesFeatureType::Range,
        ],
        normalization: true,
    };
    
    // 创建时序特征提取器
    let ts_extractor = TimeSeriesFeatureExtractor::new(ts_config);
    
    // 处理时序数据
    let mut processed_data = data.clone();
    ts_extractor.process(&mut processed_data)?;
    
    // 输出处理结果
    println!("原始数据: {} 个时间点, {} 个特征", 
        data.features.len(),
        data.feature_dim().unwrap_or(0));
    
    println!("处理后数据: {} 个窗口, {} 个特征", 
        processed_data.features.len(),
        processed_data.feature_dim().unwrap_or(0));
    
    // 每个窗口计算了5种特征，每个特征有3个维度，因此特征维度会增加
    println!("每个样本的特征维度从 {} 增加到 {}", 
        data.feature_dim().unwrap_or(0),
        processed_data.feature_dim().unwrap_or(0));
    
    Ok(())
}

/// 数据质量报告示例
pub fn quality_report_example() -> Result<()> {
    // 创建数据流水线
    let pipeline = DataPipeline::default();
    
    // 创建包含异常的测试数据
    let data = DataBatch {
        id: Some("quality_test".to_string()),
        features: vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0],  // 重复行
            vec![7.0, 8.0, 9.0],
            vec![100.0, 200.0, 300.0],  // 异常值
            vec![4.0, 5.0, f32::NAN],  // 缺失值
        ],
        labels: None,
        weights: None,
        metadata: None,
    };
    
    // 生成数据质量报告
    let report = pipeline.generate_quality_report(&data)?;
    
    // 输出报告
    println!("数据质量报告:");
    println!("- 完整度: {:.2}%", report.completeness * 100.0);
    println!("- 一致性: {:.2}%", report.consistency * 100.0);
    println!("- 有效性: {:.2}%", report.validity * 100.0);
    println!("- 异常数: {}", report.anomalies.len());
    println!("- 总记录数: {}", report.total_records);
    println!("- 特征数: {}", report.total_features);
    
    // 将报告保存为JSON文件
    let json = pipeline.quality_report_to_json(&report)?;
    println!("JSON报告: {}", json);
    
    // 在实际应用中可以保存到文件
    // pipeline.save_quality_report(&report, "quality_report.json")?;
    
    Ok(())
}

/// 批处理生成器示例
pub fn batch_generator_example() -> Result<()> {
    // 创建数据流水线
    let pipeline = DataPipeline::new(
        PipelineConfig {
            batch_size: 2,  // 每批2个样本
            validation_split: 0.25,  // 25%数据作为验证集
            shuffle: true,   // 打乱数据
            ..Default::default()
        }
    );
    
    // 创建测试数据
    let data = DataBatch {
        id: Some("batch_test".to_string()),
        features: vec![
            vec![1.0, 2.0, 3.0],  // 样本1
            vec![4.0, 5.0, 6.0],  // 样本2
            vec![7.0, 8.0, 9.0],  // 样本3
            vec![10.0, 11.0, 12.0],  // 样本4
            vec![13.0, 14.0, 15.0],  // 样本5
            vec![16.0, 17.0, 18.0],  // 样本6
            vec![19.0, 20.0, 21.0],  // 样本7
            vec![22.0, 23.0, 24.0],  // 样本8
        ],
        labels: Some(vec![
            vec![0.0], vec![1.0], vec![0.0], vec![1.0],
            vec![0.0], vec![1.0], vec![0.0], vec![1.0],
        ]),
        weights: None,
        metadata: None,
    };
    
    // 创建批次生成器
    let mut generator = pipeline.create_batch_generator(data);
    
    // 输出数据集统计信息
    println!("训练集大小: {}", generator.train_size());
    println!("验证集大小: {}", generator.validation_size());
    println!("批次大小: {}", generator.batch_size());
    println!("总批次数: {}", generator.num_batches());
    
    // 遍历所有训练批次
    let mut batch_count = 0;
    println!("\n训练批次:");
    while let Some(batch) = generator.next_batch() {
        batch_count += 1;
        println!("批次 {}: {} 样本, 特征维度 {}", 
            batch_count, 
            batch.batch_size(), 
            batch.feature_dim().unwrap_or(0));
    }
    
    // 获取验证批次
    println!("\n验证批次:");
    if let Some(validation_batch) = generator.validation_batch() {
        println!("验证批次: {} 样本, 特征维度 {}", 
            validation_batch.batch_size(),
            validation_batch.feature_dim().unwrap_or(0));
    } else {
        println!("没有验证数据");
    }
    
    // 重置生成器并再次遍历
    generator.reset();
    println!("\n重置后再次遍历:");
    let mut batch_count = 0;
    while let Some(batch) = generator.next_batch() {
        batch_count += 1;
        if batch_count > 2 {  // 只输出前两个批次
            break;
        }
        println!("批次 {}: {} 样本", batch_count, batch.batch_size());
    }
    
    Ok(())
}

/// 完整的模型训练示例
pub fn model_training_example() -> Result<()> {
    // 创建数据流水线
    let mut pipeline = DataPipeline::default();
    
    // 创建训练数据
    let data = DataBatch {
        id: Some("training_data".to_string()),
        features: vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0],
        ],
        labels: Some(vec![
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            vec![0.0],
        ]),
        weights: Some(vec![1.0, 1.0, 1.0, 1.0, 1.0]),
        metadata: None,
    };
    
    // 计算数据统计信息
    pipeline.calculate_stats(&data)?;
    
    // 创建模型配置
    let model_config = ModelConfig {
        model_type: "linear".to_string(),
        hyperparameters: HashMap::from([
            ("learning_rate".to_string(), 0.01),
            ("regularization".to_string(), 0.001),
        ]),
        max_iterations: 100,
        learning_rate: 0.01,
        batch_size: 32,
        validation_split: 0.2,
    };
    
    // 训练模型
    let result = pipeline.train_model(&data, model_config)?;
    
    // 输出训练结果
    println!("模型训练完成:");
    println!("- 模型ID: {}", result.model_id);
    println!("- 训练时间: {}ms", result.training_time.as_millis());
    println!("- 准确率: {:.2}%", result.metrics.get("accuracy").unwrap_or(&0.0) * 100.0);
    println!("- 损失: {:.4}", result.metrics.get("loss").unwrap_or(&0.0));
    println!("- 迭代次数: {}", result.iterations);
    
    Ok(())
}

/// 运行所有示例
pub fn run_all_examples() -> Result<()> {
    println!("====== 运行基本数据流水线示例 ======");
    basic_pipeline_example()?;
    
    println!("\n====== 运行异常检测示例 ======");
    anomaly_detection_example()?;
    
    println!("\n====== 运行时序数据处理示例 ======");
    time_series_example()?;
    
    println!("\n====== 运行数据质量报告示例 ======");
    quality_report_example()?;
    
    println!("\n====== 运行批处理生成器示例 ======");
    batch_generator_example()?;
    
    println!("\n====== 运行模型训练示例 ======");
    model_training_example()?;
    
    println!("\n所有示例运行完成!");
    Ok(())
}

// 为测试添加一个main函数
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_pipeline() {
        basic_pipeline_example().unwrap();
    }
    
    #[test]
    fn test_anomaly_detection() {
        anomaly_detection_example().unwrap();
    }
    
    #[test]
    fn test_time_series() {
        time_series_example().unwrap();
    }
    
    #[test]
    fn test_quality_report() {
        quality_report_example().unwrap();
    }
    
    #[test]
    fn test_batch_generator() {
        batch_generator_example().unwrap();
    }
    
    #[test]
    fn test_model_training() {
        model_training_example().unwrap();
    }
} 