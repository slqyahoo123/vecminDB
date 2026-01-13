// 状态同步协议使用示例
//
// 演示如何在分布式环境中使用状态同步功能

use std::sync::Arc;
use std::time::Duration;
use std::thread;

use crate::error::Result;
use crate::storage::engine::{StorageEngine, Storage};
use crate::storage::config::StorageConfig;
use crate::storage::replication::state_sync::{
    StateSyncManager, StateSyncConfig, SyncOperationType, 
    SyncPriority, NodeRole, VerificationLevel
};
use crate::storage::replication::sync_protocol::{
    SyncProtocol, DefaultStateProvider, DefaultStateApplier
};

/// 运行状态同步示例
pub fn run_state_sync_example() -> Result<()> {
    println!("开始运行分布式状态同步示例...");
    
    // 创建两个存储节点：主节点和从节点
    let primary_config = StorageConfig {
        path: std::path::PathBuf::from("./data/node1"),
        engine_type: "rocksdb".to_string(),
        ..StorageConfig::default()
    };
    
    let secondary_config = StorageConfig {
        path: std::path::PathBuf::from("./data/node2"),
        engine_type: "rocksdb".to_string(),
        ..StorageConfig::default()
    };
    
    let primary_storage = Storage::new(primary_config)?;
    let secondary_storage = Storage::new(secondary_config)?;
    
    // 创建同步配置
    let sync_config = StateSyncConfig {
        sync_interval_ms: 1000,
        timeout_ms: 5000,
        retry_count: 3,
        retry_interval_ms: 500,
        batch_size: 100,
        enable_compression: true,
        enable_incremental_sync: true,
        max_concurrent_syncs: 3,
        heartbeat_interval_ms: 1000,
        node_timeout_ms: 3000,
        verification_level: VerificationLevel::Checksum,
    };
    
    // 创建主节点同步管理器
    let primary_sync_manager = StateSyncManager::with_impl(
        "node1", 
        Arc::new(primary_storage.clone()),
        Some(sync_config.clone())
    )?;
    
    // 设置主节点角色
    primary_sync_manager.set_local_role(NodeRole::Primary)?;
    
    // 创建从节点同步管理器
    let secondary_sync_manager = StateSyncManager::with_impl(
        "node2", 
        Arc::new(secondary_storage.clone()),
        Some(sync_config.clone())
    )?;
    
    // 设置从节点角色
    secondary_sync_manager.set_local_role(NodeRole::Secondary)?;
    
    // 启动同步协议
    primary_sync_manager.start_sync_protocol(Arc::new(primary_storage.clone()))?;
    secondary_sync_manager.start_sync_protocol(Arc::new(secondary_storage.clone()))?;
    
    println!("同步管理器和协议已启动");
    
    // 添加节点
    primary_sync_manager.add_node("node2", "localhost:8002", NodeRole::Secondary)?;
    secondary_sync_manager.add_node("node1", "localhost:8001", NodeRole::Primary)?;
    
    println!("节点已互相注册");
    
    // 在主节点上写入一些数据
    let test_data = [
        ("key1".as_bytes(), "value1".as_bytes()),
        ("key2".as_bytes(), "value2".as_bytes()),
        ("key3".as_bytes(), "value3".as_bytes()),
    ];
    
    for (key, value) in &test_data {
        primary_storage.put(*key, *value)?;
    }
    
    // 更新主节点版本
    primary_sync_manager.update_current_version(1)?;
    
    println!("主节点已写入测试数据并更新版本");
    
    // 启动同步过程
    let sync_id = primary_sync_manager.start_sync_with_node(
        "node2", 
        SyncOperationType::FullSync, 
        SyncPriority::High
    )?;
    
    println!("已启动同步会话: {}", sync_id);
    
    // 等待同步完成
    let sync_timeout = 5000; // 5秒超时
    primary_sync_manager.wait_for_sync_completion(&[sync_id], sync_timeout)?;
    
    // 验证从节点数据
    for (key, expected_value) in &test_data {
        match secondary_storage.get(*key)? {
            Some(value) => {
                if value == *expected_value {
                    println!("数据同步成功: {:?} -> {:?}", 
                             String::from_utf8_lossy(*key), 
                             String::from_utf8_lossy(*expected_value));
                } else {
                    println!("数据同步失败: {:?} 期望 {:?}, 得到 {:?}", 
                             String::from_utf8_lossy(*key),
                             String::from_utf8_lossy(*expected_value),
                             String::from_utf8_lossy(&value));
                }
            },
            None => {
                println!("数据同步失败: 未找到键 {:?}", String::from_utf8_lossy(*key));
            }
        }
    }
    
    // 测试增量同步
    let new_data = [
        ("key4".as_bytes(), "value4".as_bytes()),
        ("key5".as_bytes(), "value5".as_bytes()),
    ];
    
    for (key, value) in &new_data {
        primary_storage.put(*key, *value)?;
    }
    
    // 更新主节点版本
    primary_sync_manager.update_current_version(2)?;
    
    println!("主节点已写入新数据并更新版本");
    
    // 启动增量同步
    let incremental_sync_id = primary_sync_manager.create_incremental_sync("node2", 1)?;
    
    if incremental_sync_id != "no_sync_needed" {
        println!("已启动增量同步会话: {}", incremental_sync_id);
        
        // 等待同步完成
        primary_sync_manager.wait_for_sync_completion(&[incremental_sync_id], sync_timeout)?;
    } else {
        println!("无需增量同步");
    }
    
    // 验证从节点新数据
    for (key, expected_value) in &new_data {
        match secondary_storage.get(*key)? {
            Some(value) => {
                if value == *expected_value {
                    println!("增量同步成功: {:?} -> {:?}", 
                             String::from_utf8_lossy(*key), 
                             String::from_utf8_lossy(*expected_value));
                } else {
                    println!("增量同步失败: {:?} 期望 {:?}, 得到 {:?}", 
                             String::from_utf8_lossy(*key),
                             String::from_utf8_lossy(*expected_value),
                             String::from_utf8_lossy(&value));
                }
            },
            None => {
                println!("增量同步失败: 未找到键 {:?}", String::from_utf8_lossy(*key));
            }
        }
    }
    
    // 检测节点状态变化
    let status_changes = primary_sync_manager.detect_node_status_changes()?;
    println!("节点状态变化: {:?}", status_changes);
    
    // 计算同步效率
    let efficiency = primary_sync_manager.calculate_sync_efficiency()?;
    println!("同步效率: {:.2}%", efficiency * 100.0);
    
    // 获取同步统计信息
    let sync_stats = primary_sync_manager.get_sync_statistics()?;
    println!("同步统计信息: {:?}", sync_stats);
    
    // 停止同步协议
    thread::sleep(Duration::from_secs(1)); // 等待所有操作完成
    
    println!("状态同步示例运行完成");
    Ok(())
}

/// 运行状态同步示例的主函数
pub fn main() -> Result<()> {
    run_state_sync_example()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_sync() -> Result<()> {
        // 这里可以添加更简单的测试版本
        Ok(())
    }
} 