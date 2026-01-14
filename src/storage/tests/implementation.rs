#[cfg(test)]
use crate::storage::*;
#[cfg(all(test, feature = "tempfile"))]
use tempfile::TempDir;

#[cfg(all(test, feature = "tempfile"))]
pub fn create_test_storage() -> Result<(Storage, TempDir)> {
    let temp_dir = TempDir::new()?;
    let config = StorageConfig::new(temp_dir.path().to_str().unwrap().to_string());
    let storage = Storage::new(config)?;
    Ok((storage, temp_dir))
}

#[test]
pub fn test_storage_engine() -> Result<()> {
    let (storage, _temp_dir) = create_test_storage()?;
    
    // 存储和获取基本测试
    let test_key = "test_key";
    let test_value = "test_value".as_bytes().to_vec();
    
    // 测试插入数据
    storage.put(test_key.as_bytes(), &test_value)?;
    
    // 测试获取数据
    let retrieved = storage.get(test_key.as_bytes())?;
    assert!(retrieved.is_some(), "应该能够获取到存储的值");
    assert_eq!(retrieved.unwrap().as_ref(), test_value, "存取的值应相等");
    
    // 测试删除数据
    storage.delete(test_key.as_bytes())?;
    let after_delete = storage.get(test_key.as_bytes())?;
    assert!(after_delete.is_none(), "删除后应该获取不到值");
    
    Ok(())
}

#[test]
fn test_storage_basic_operations() -> Result<()> {
    let (storage, _temp_dir) = create_test_storage()?;
    
    // 测试基本的键值存储操作
    let key = "test_key";
    let value = "test_value";
    
    storage.put(key, value.as_bytes())?;
    let result = storage.get(key)?;
    
    assert!(result.is_some());
    assert_eq!(String::from_utf8(result.unwrap()).unwrap(), value);
    
    // 测试删除操作
    storage.delete(key)?;
    let result = storage.get(key)?;
    assert!(result.is_none());
    
    // 测试批量操作
    let mut batch = HashMap::new();
    for i in 0..10 {
        let key = format!("batch_key_{}", i);
        let value = format!("batch_value_{}", i);
        batch.insert(key, value.as_bytes().to_vec());
    }
    
    storage.put_batch(&batch)?;
    
    for i in 0..10 {
        let key = format!("batch_key_{}", i);
        let expected_value = format!("batch_value_{}", i);
        let result = storage.get(&key)?;
        assert!(result.is_some());
        assert_eq!(String::from_utf8(result.unwrap()).unwrap(), expected_value);
    }
    
    Ok(())
}

#[test]
fn test_cache_operations() -> Result<()> {
    let (storage, _temp_dir) = create_test_storage()?;
    
    // 测试缓存操作
    let cache = storage.cache();
    
    let key = "cache_test_key";
    let value = "cache_test_value";
    
    cache.put(key, value.as_bytes())?;
    let result = cache.get(key)?;
    
    assert!(result.is_some());
    assert_eq!(String::from_utf8(result.unwrap()).unwrap(), value);
    
    // 测试缓存TTL
    let ttl_key = "ttl_key";
    let ttl_value = "ttl_value";
    
    cache.put_with_ttl(ttl_key, ttl_value.as_bytes(), Duration::from_millis(100))?;
    
    // 验证值存在
    let result = cache.get(ttl_key)?;
    assert!(result.is_some());
    
    // 等待TTL过期
    thread::sleep(Duration::from_millis(200));
    
    // 验证值已过期
    let result = cache.get(ttl_key)?;
    assert!(result.is_none());
    
    Ok(())
}

#[test]
fn test_storage_permissions() -> Result<()> {
    let (storage, _temp_dir) = create_test_storage()?;
    
    let user_id = "user123";
    let resource_id = "resource456";
    let resource_type = "document";
    
    // 测试权限管理
    storage.permissions().grant(resource_type, resource_id, user_id, "read")?;
    storage.permissions().grant(resource_type, resource_id, user_id, "write")?;
    
    // 检查权限
    let has_read = storage.permissions().check(resource_type, resource_id, user_id, "read")?;
    let has_write = storage.permissions().check(resource_type, resource_id, user_id, "write")?;
    let has_delete = storage.permissions().check(resource_type, resource_id, user_id, "delete")?;
    
    assert!(has_read);
    assert!(has_write);
    assert!(!has_delete);
    
    // 测试取消权限
    storage.permissions().revoke(resource_type, resource_id, user_id, "write")?;
    let has_write_after_revoke = storage.permissions().check(resource_type, resource_id, user_id, "write")?;
    assert!(!has_write_after_revoke);
    
    // 测试获取用户所有权限
    storage.permissions().grant(resource_type, "another_resource", user_id, "admin")?;
    
    let user_permissions = storage.permissions().get_user_permissions(user_id, resource_type)?;
    assert_eq!(user_permissions.len(), 2);
    
    // 测试权限继承
    let group_id = "group789";
    
    // 将用户添加到组
    storage.permissions().add_to_group(user_id, group_id)?;
    
    // 给组授权
    storage.permissions().grant(resource_type, "group_resource", group_id, "view")?;
    
    // 检查用户是否继承了组的权限
    let has_group_perm = storage.permissions().check(resource_type, "group_resource", user_id, "view")?;
    assert!(has_group_perm);
    
    // 测试从组中移除用户
    storage.permissions().remove_from_group(user_id, group_id)?;
    
    // 检查权限是否已被移除
    let has_group_perm_after_remove = storage.permissions().check(resource_type, "group_resource", user_id, "view")?;
    assert!(!has_group_perm_after_remove);
    
    Ok(())
}

#[test]
fn test_column_families() -> Result<()> {
    let (storage, _temp_dir) = create_test_storage()?;
    
    // 测试不同列族的数据隔离
    let cf1 = "family1";
    let cf2 = "family2";
    
    // 创建列族
    storage.create_column_family(cf1)?;
    storage.create_column_family(cf2)?;
    
    // 在不同列族中存储相同的键
    let key = "same_key";
    let value1 = "value_for_cf1";
    let value2 = "value_for_cf2";
    
    storage.put_in_cf(cf1, key, value1.as_bytes())?;
    storage.put_in_cf(cf2, key, value2.as_bytes())?;
    
    // 从不同列族中读取
    let result1 = storage.get_from_cf(cf1, key)?;
    let result2 = storage.get_from_cf(cf2, key)?;
    
    assert!(result1.is_some());
    assert!(result2.is_some());
    assert_eq!(String::from_utf8(result1.unwrap()).unwrap(), value1);
    assert_eq!(String::from_utf8(result2.unwrap()).unwrap(), value2);
    
    Ok(())
}

#[test]
fn test_data_storage() -> Result<()> {
    let (storage, _temp_dir) = create_test_storage()?;
    
    // 测试存储二进制数据
    let data_id = "test_tensor";
    let tensor_data = TensorData {
        shape: vec![2, 3],
        data_type: DataType::Float32,
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_iter()
            .flat_map(|x| (x as f32).to_le_bytes().to_vec())
            .collect(),
    };
    
    storage.put_data(data_id, &tensor_data)?;
    
    // 读取数据
    let retrieved_data = storage.get_data::<TensorData>(data_id)?;
    assert!(retrieved_data.is_some());
    
    let retrieved_tensor = retrieved_data.unwrap();
    assert_eq!(retrieved_tensor.shape, tensor_data.shape);
    assert_eq!(retrieved_tensor.data_type, tensor_data.data_type);
    assert_eq!(retrieved_tensor.data, tensor_data.data);
    
    Ok(())
}

#[test]
fn test_persistence() -> Result<()> {
    let temp_dir = tempdir()?;
    let db_path = temp_dir.path().to_str().unwrap().to_string();
    
    // 创建存储并写入数据
    {
        let config = StorageConfig::new(db_path.clone());
        let storage = Storage::new(config)?;
        
        storage.put("persistent_key", "persistent_value".as_bytes())?;
        
        // 关闭存储实例
        drop(storage);
    }
    
    // 重新打开存储并验证数据持久性
    {
        let config = StorageConfig::new(db_path);
        let storage = Storage::new(config)?;
        
        let result = storage.get("persistent_key")?;
        assert!(result.is_some());
        assert_eq!(String::from_utf8(result.unwrap()).unwrap(), "persistent_value");
    }
    
    Ok(())
}

#[test]
fn test_data_compression() -> Result<()> {
    let (storage, _temp_dir) = create_test_storage()?;
    
    // 测试数据压缩
    let large_data = vec![0u8; 1000000]; // 1MB的数据
    
    let start = Instant::now();
    storage.put_compressed("compressed_key", &large_data)?;
    println!("压缩存储耗时: {:?}", start.elapsed());
    
    let start = Instant::now();
    let retrieved_data = storage.get_decompressed("compressed_key")?;
    println!("解压读取耗时: {:?}", start.elapsed());
    
    assert!(retrieved_data.is_some());
    assert_eq!(retrieved_data.unwrap(), large_data);
    
    // 比较压缩与非压缩数据大小
    let start = Instant::now();
    storage.put("uncompressed_key", &large_data)?;
    println!("非压缩存储耗时: {:?}", start.elapsed());
    
    let compressed_size = storage.get_data_size("compressed_key")?;
    let uncompressed_size = storage.get_data_size("uncompressed_key")?;
    
    println!("压缩数据大小: {} 字节", compressed_size);
    println!("非压缩数据大小: {} 字节", uncompressed_size);
    
    // 压缩数据应该小于原始数据
    assert!(compressed_size < uncompressed_size);
    
    Ok(())
}
