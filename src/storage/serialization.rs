//! 序列化工具模块
//! 
//! 提供高效的序列化/反序列化功能

use crate::Result;
use serde::{Serialize, de::DeserializeOwned};

/// 序列化格式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// Bincode（二进制，高效）
    Bincode,
    /// JSON（文本，可读）
    Json,
    /// MessagePack（二进制，紧凑）
    MessagePack,
}

/// 序列化器
pub struct Serializer {
    format: SerializationFormat,
}

impl Serializer {
    /// 创建新的序列化器
    pub fn new(format: SerializationFormat) -> Self {
        Self { format }
    }

    /// 创建默认序列化器（使用Bincode）
    pub fn default() -> Self {
        Self::new(SerializationFormat::Bincode)
    }

    /// 序列化数据
    pub fn serialize<T: Serialize>(&self, data: &T) -> Result<Vec<u8>> {
        match self.format {
            SerializationFormat::Bincode => {
                bincode::serialize(data)
                    .map_err(|e| crate::Error::serialization(format!("Bincode序列化失败: {}", e)))
            },
            SerializationFormat::Json => {
                serde_json::to_vec(data)
                    .map_err(|e| crate::Error::serialization(format!("JSON序列化失败: {}", e)))
            },
            SerializationFormat::MessagePack => {
                rmp_serde::to_vec(data)
                    .map_err(|e| crate::Error::serialization(format!("MessagePack序列化失败: {}", e)))
            },
        }
    }

    /// 反序列化数据
    pub fn deserialize<T: DeserializeOwned>(&self, data: &[u8]) -> Result<T> {
        match self.format {
            SerializationFormat::Bincode => {
                bincode::deserialize(data)
                    .map_err(|e| crate::Error::serialization(format!("Bincode反序列化失败: {}", e)))
            },
            SerializationFormat::Json => {
                serde_json::from_slice(data)
                    .map_err(|e| crate::Error::serialization(format!("JSON反序列化失败: {}", e)))
            },
            SerializationFormat::MessagePack => {
                rmp_serde::from_slice(data)
                    .map_err(|e| crate::Error::serialization(format!("MessagePack反序列化失败: {}", e)))
            },
        }
    }
}

/// 向量序列化工具
pub struct VectorSerializer;

impl VectorSerializer {
    /// 序列化f32向量为字节数组
    pub fn serialize_f32_vector(vector: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(vector.len() * 4);
        for &value in vector {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// 从字节数组反序列化f32向量
    pub fn deserialize_f32_vector(bytes: &[u8]) -> Result<Vec<f32>> {
        if bytes.len() % 4 != 0 {
            return Err(crate::Error::serialization(
                "无效的向量数据：字节长度不是4的倍数".to_string()
            ));
        }

        let mut vector = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            vector.push(value);
        }
        Ok(vector)
    }

    /// 序列化f64向量为字节数组
    pub fn serialize_f64_vector(vector: &[f64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(vector.len() * 8);
        for &value in vector {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// 从字节数组反序列化f64向量
    pub fn deserialize_f64_vector(bytes: &[u8]) -> Result<Vec<f64>> {
        if bytes.len() % 8 != 0 {
            return Err(crate::Error::serialization(
                "无效的向量数据：字节长度不是8的倍数".to_string()
            ));
        }

        let mut vector = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            let value = f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            vector.push(value);
        }
        Ok(vector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Serialize, serde::Deserialize)]
    struct TestData {
        id: u64,
        name: String,
        values: Vec<f32>,
    }

    #[test]
    fn test_bincode_serialization() {
        let serializer = Serializer::new(SerializationFormat::Bincode);
        let data = TestData {
            id: 123,
            name: "test".to_string(),
            values: vec![1.0, 2.0, 3.0],
        };

        let bytes = serializer.serialize(&data).unwrap();
        let deserialized: TestData = serializer.deserialize(&bytes).unwrap();

        assert_eq!(data, deserialized);
    }

    #[test]
    fn test_json_serialization() {
        let serializer = Serializer::new(SerializationFormat::Json);
        let data = TestData {
            id: 123,
            name: "test".to_string(),
            values: vec![1.0, 2.0, 3.0],
        };

        let bytes = serializer.serialize(&data).unwrap();
        let deserialized: TestData = serializer.deserialize(&bytes).unwrap();

        assert_eq!(data, deserialized);
    }

    #[test]
    fn test_vector_serialization_f32() {
        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes = VectorSerializer::serialize_f32_vector(&vector);
        let deserialized = VectorSerializer::deserialize_f32_vector(&bytes).unwrap();

        assert_eq!(vector, deserialized);
    }

    #[test]
    fn test_vector_serialization_f64() {
        let vector = vec![1.0f64, 2.0, 3.0, 4.0];
        let bytes = VectorSerializer::serialize_f64_vector(&vector);
        let deserialized = VectorSerializer::deserialize_f64_vector(&bytes).unwrap();

        assert_eq!(vector, deserialized);
    }

    #[test]
    fn test_invalid_vector_data() {
        // 不是4的倍数的字节数组
        let invalid_bytes = vec![1, 2, 3];
        let result = VectorSerializer::deserialize_f32_vector(&invalid_bytes);
        assert!(result.is_err());
    }
}



