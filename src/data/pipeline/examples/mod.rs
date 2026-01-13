// src/data/pipeline/examples/mod.rs
//
// 示例模块入口
// 包含各种数据管道功能演示示例

mod performance_monitoring;

pub use performance_monitoring::{
    run_monitoring_example,
    run_measure_time_example,
    run_examples
};

/// 运行所有示例
pub fn run_all_examples() -> Result<(), Box<dyn std::error::Error>> {
    // 添加日志头信息
    println!("====================================");
    println!("      数据管道功能示例运行器");
    println!("====================================");

    // 运行性能监控示例
    println!("\n>> 运行性能监控示例");
    run_examples()?;

    // 在此处添加其他示例

    println!("\n所有示例执行完成！");
    Ok(())
} 