// Data Operations Module
// 数据操作模块

use std::collections::HashMap;
use std::error::Error as StdError;
use serde_json::Value;
use crate::Result;
use crate::Error;
use crate::data::processor::config::ProcessorConfig as ImportedProcessorConfig;
use crate::data::pipeline::traits::RecordValue;
use crate::data::record::{Record, Value as RecordFieldValue};
use crate::data::value::DataValue;
use crate::data::DataFormat;
// remove unused sync imports; parsing is single-threaded here
use std::hash::Hasher;
use std::hash::Hash;

/// 数据解析器
pub struct DataParser {
    format: DataFormat,
}

impl DataParser {
    /// 创建新的数据解析器
    pub fn new(format: DataFormat) -> Self {
        Self { format }
    }

    fn record_value_to_data_value(value: RecordValue) -> DataValue {
        match value {
            RecordValue::String(s) => DataValue::String(s),
            RecordValue::Number(n) => DataValue::Number(n),
            RecordValue::Integer(i) => DataValue::Integer(i),
            RecordValue::Boolean(b) => DataValue::Boolean(b),
            RecordValue::Null => DataValue::Null,
            RecordValue::Array(arr) => {
                let mapped = arr.into_iter().map(Self::record_value_to_data_value).collect();
                DataValue::Array(mapped)
            }
            RecordValue::Object(obj) => {
                let mut map = HashMap::new();
                for (k, v) in obj {
                    map.insert(k, Self::record_value_to_data_value(v));
                }
                DataValue::Object(map)
            }
            RecordValue::Binary(b) => DataValue::Binary(b),
        }
    }

    fn build_record_from_map(record_data: HashMap<String, RecordValue>) -> Record {
        let mut record = Record::new();
        for (k, v) in record_data {
            let dv = Self::record_value_to_data_value(v);
            record.fields.insert(k, RecordFieldValue::Data(dv));
        }
        record
    }

    /// 解析数据
    pub fn parse_data(&self, data: &[u8], config: &ImportedProcessorConfig) -> Result<Vec<Record>> {
        let data_str = String::from_utf8_lossy(data);
        
        match self.format {
            DataFormat::CSV => self.parse_csv_data(&data_str, config),
            DataFormat::JSON => self.parse_json_data(&data_str, config),
            DataFormat::TSV => self.parse_tsv_data(&data_str, config),
            _ => Err(Error::invalid_data("不支持的数据格式"))
        }
    }

    /// 直接从 JSON Value 解析记录（用于自检/内联场景）
    pub fn parse_inline(&self, value: &serde_json::Value, config: &ImportedProcessorConfig) -> Result<Vec<Record>> {
        let bytes = serde_json::to_vec(value)
            .map_err(|e| Error::serialization(format!("无法序列化内联数据: {}", e)))?;
        self.parse_data(&bytes, config)
    }

    /// 解析CSV数据
    fn parse_csv_data(&self, data_str: &str, _config: &ImportedProcessorConfig) -> Result<Vec<Record>> {
        let lines: Vec<&str> = data_str.lines().collect();
        if lines.is_empty() {
            return Ok(Vec::new());
        }
        
        // 第一行作为标题
        let headers: Vec<&str> = lines[0].split(',').collect();
        let mut records = Vec::new();
        
        // 解析数据行
        for (line_num, line) in lines.iter().skip(1).enumerate() {
            let values: Vec<&str> = line.split(',').collect();
            
            if values.len() != headers.len() {
                log::warn!("第{}行列数不匹配", line_num + 2);
                continue;
            }
            
            let mut record_data = HashMap::new();
            for (i, value) in values.iter().enumerate() {
                if let Some(header) = headers.get(i) {
                    record_data.insert(header.to_string(), RecordValue::String(value.trim().to_string()));
                }
            }
            
            records.push(Self::build_record_from_map(record_data));
        }
        
        Ok(records)
    }
    
    /// 解析JSON数据
    fn parse_json_data(&self, data_str: &str, _config: &ImportedProcessorConfig) -> Result<Vec<Record>> {
        let json_value: serde_json::Value = serde_json::from_str(data_str)
            .map_err(|e| Error::invalid_data(&format!("JSON解析失败: {}", e)))?;
        
        match json_value {
            serde_json::Value::Array(arr) => {
                let mut records = Vec::new();
                for item in arr {
                    if let serde_json::Value::Object(obj) = item {
                        let mut record_data = HashMap::new();
                        for (key, value) in obj {
                            let record_value = self.json_value_to_record_value(value);
                            record_data.insert(key, record_value);
                        }
                        records.push(Self::build_record_from_map(record_data));
                    }
                }
                Ok(records)
            },
            serde_json::Value::Object(obj) => {
                let mut record_data = HashMap::new();
                for (key, value) in obj {
                    let record_value = self.json_value_to_record_value(value);
                    record_data.insert(key, record_value);
                }
                Ok(vec![Self::build_record_from_map(record_data)])
            },
            _ => Err(Error::invalid_data("JSON数据必须是对象或对象数组"))
        }
    }
    
    /// 解析TSV数据
    fn parse_tsv_data(&self, data_str: &str, config: &ImportedProcessorConfig) -> Result<Vec<Record>> {
        // TSV类似CSV，但使用制表符分隔
        let tsv_as_csv = data_str.replace('\t', ",");
        self.parse_csv_data(&tsv_as_csv, config)
    }
    
    /// JSON值转换为记录值
    fn json_value_to_record_value(&self, value: serde_json::Value) -> RecordValue {
        match value {
            serde_json::Value::String(s) => RecordValue::String(s),
            serde_json::Value::Number(n) => {
                if n.is_i64() {
                    RecordValue::Integer(n.as_i64().unwrap_or(0))
                } else {
                    RecordValue::Number(n.as_f64().unwrap_or(0.0))
                }
            },
            serde_json::Value::Bool(b) => RecordValue::Boolean(b),
            serde_json::Value::Null => RecordValue::Null,
            serde_json::Value::Array(arr) => {
                let items: Vec<RecordValue> = arr.into_iter()
                    .map(|v| self.json_value_to_record_value(v))
                    .collect();
                RecordValue::Array(items)
            },
            serde_json::Value::Object(obj) => {
                let mut map = HashMap::new();
                for (k, v) in obj {
                    map.insert(k, self.json_value_to_record_value(v));
                }
                RecordValue::Object(map)
            }
        }
    }
}

/// 数据转换器
pub struct DataConverter {
}

impl DataConverter {
    /// 创建新的数据转换器
    pub fn new() -> Self {
        Self {}
    }

    /// 内联转换（用于自检/就绪检查）
    pub fn convert_inline(&self, records: &[Record]) -> Result<Vec<Record>> {
        // 就绪检查阶段不做实际转换，直接克隆返回
        Ok(records.to_vec())
    }

    /// 编码转换
    pub fn convert_encoding(&self, data: &[u8], from_encoding: &str, to_encoding: &str) -> Result<Vec<u8>> {
        // 实际生产环境中应该使用专业的编码转换库如encoding_rs
        match (from_encoding, to_encoding) {
            ("utf-8", "utf-8") => Ok(data.to_vec()),
            ("gbk", "utf-8") => {
                // 这里应该使用专业的编码转换库
                // 为了示例，假设成功转换
                let converted_string = String::from_utf8_lossy(data).to_string();
                Ok(converted_string.into_bytes())
            },
            ("latin1", "utf-8") => {
                // 同样应该使用专业库
                let converted_string = String::from_utf8_lossy(data).to_string();
                Ok(converted_string.into_bytes())
            },
            _ => {
                log::warn!("不支持的编码转换: {} -> {}", from_encoding, to_encoding);
                Ok(data.to_vec())
            }
        }
    }

    /// 压缩数据
    pub fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // 使用LZ4或其他压缩算法
        // 这里提供一个简化的实现
        
        if data.len() < 1024 {
            // 小文件不压缩
            return Ok(data.to_vec());
        }
        
        // 实际实现应该使用压缩库如lz4_flex或flate2
        // 这里为了示例返回原数据
        log::debug!("压缩数据: {} 字节", data.len());
        Ok(data.to_vec())
    }

    /// 解压数据
    pub fn decompress_data(&self, compressed_data: &[u8]) -> Result<Vec<u8>> {
        // 解压缩实现
        log::debug!("解压数据: {} 字节", compressed_data.len());
        Ok(compressed_data.to_vec())
    }
}

/// 数据验证器
pub struct DataValidator {
    format: DataFormat,
}

impl DataValidator {
    /// 创建新的数据验证器
    pub fn new(format: DataFormat) -> Self {
        Self { format }
    }

    /// 验证和修复数据格式
    pub fn validate_and_fix_format(&self, data: &[u8]) -> Result<Vec<u8>> {
        let data_str = String::from_utf8_lossy(data);
        
        match self.format {
            DataFormat::CSV => self.validate_and_fix_csv(&data_str),
            DataFormat::JSON => self.validate_and_fix_json(&data_str),
            DataFormat::Parquet => self.validate_and_fix_parquet(data),
            DataFormat::Arrow => self.validate_and_fix_arrow(data),
            _ => Ok(data.to_vec()),
        }
    }
    
    /// 验证和修复CSV格式
    fn validate_and_fix_csv(&self, data_str: &str) -> Result<Vec<u8>> {
        let lines: Vec<&str> = data_str.lines().collect();
        
        if lines.is_empty() {
            return Err(Error::invalid_data("CSV文件为空"));
        }
        
        // 获取头部行作为参考
        let header_line = lines[0];
        let expected_columns = header_line.split(',').count();
        
        // 验证和修复每一行
        let mut fixed_lines = Vec::new();
        fixed_lines.push(header_line.to_string());
        
        for (line_num, line) in lines.iter().skip(1).enumerate() {
            let columns: Vec<&str> = line.split(',').collect();
            
            if columns.len() != expected_columns {
                log::warn!("第{}行列数不匹配，期望{}列，实际{}列", line_num + 2, expected_columns, columns.len());
                
                // 修复策略：补齐缺失列或截断多余列
                let mut fixed_columns = columns;
                if fixed_columns.len() < expected_columns {
                    // 补齐缺失列
                    for _ in fixed_columns.len()..expected_columns {
                        fixed_columns.push("");
                    }
                } else if fixed_columns.len() > expected_columns {
                    // 截断多余列
                    fixed_columns.truncate(expected_columns);
                }
                
                fixed_lines.push(fixed_columns.join(","));
            } else {
                fixed_lines.push(line.to_string());
            }
        }
        
        Ok(fixed_lines.join("\n").into_bytes())
    }
    
    /// 验证和修复JSON格式
    fn validate_and_fix_json(&self, data_str: &str) -> Result<Vec<u8>> {
        // 尝试解析JSON
        match serde_json::from_str::<serde_json::Value>(data_str) {
            Ok(_) => Ok(data_str.as_bytes().to_vec()),
            Err(e) => {
                log::warn!("JSON格式错误: {}", e);
                
                // 尝试修复常见的JSON问题
                let fixed_json = self.fix_common_json_issues(data_str)?;
                
                // 再次验证修复后的JSON
                match serde_json::from_str::<serde_json::Value>(&fixed_json) {
                    Ok(_) => Ok(fixed_json.into_bytes()),
                    Err(_) => Err(Error::invalid_data("无法修复JSON格式")),
                }
            }
        }
    }
    
    /// 修复常见的JSON问题
    fn fix_common_json_issues(&self, json_str: &str) -> Result<String> {
        let mut fixed = json_str.to_string();
        
        // 1. 修复尾部逗号
        fixed = fixed.replace(",}", "}").replace(",]", "]");
        
        // 2. 修复单引号
        fixed = fixed.replace("'", "\"");
        
        // 3. 确保字符串被正确引用
        // 这是一个简化的实现，实际应该使用更复杂的解析逻辑
        
        // 4. 处理未引用的键
        // 同样是简化实现
        
        Ok(fixed)
    }
    
    /// 验证Parquet格式
    fn validate_and_fix_parquet(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Parquet是二进制格式，需要使用parquet库进行验证
        // 这里提供一个基本的实现框架
        
        if data.len() < 4 {
            return Err(Error::invalid_data("Parquet文件太小"));
        }
        
        // 检查Parquet魔数
        if &data[0..4] != b"PAR1" && &data[data.len()-4..] != b"PAR1" {
            return Err(Error::invalid_data("不是有效的Parquet文件"));
        }
        
        // 基本验证通过
        Ok(data.to_vec())
    }
    
    /// 验证Arrow格式
    fn validate_and_fix_arrow(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Arrow格式验证
        if data.len() < 8 {
            return Err(Error::invalid_data("Arrow文件太小"));
        }
        
        // 基本验证（实际实现应该使用arrow库）
        Ok(data.to_vec())
    }

    /// 内联验证（用于自检/就绪检查）
    pub fn validate_inline(&self, records: &[Record]) -> Result<()> {
        if records.is_empty() {
            return Err(Error::invalid_data("内联数据为空"));
        }
        Ok(())
    }
}

/// 数据清理器
pub struct DataCleaner {
}

impl DataCleaner {
    /// 创建新的数据清理器
    pub fn new() -> Self {
        Self {}
    }

    /// 内联清理记录（用于自检/就绪检查）
    pub fn clean_inline(&self, records: &mut Vec<Record>) -> Result<()> {
        for record in records.iter_mut() {
            self.clean_record_data(record)?;
        }
        Ok(())
    }

    /// 数据清理
    pub fn clean_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut result = data.to_vec();
        
        // 1. 移除BOM标记
        if result.len() >= 3 && &result[0..3] == b"\xEF\xBB\xBF" {
            result = result[3..].to_vec();
        }
        
        // 2. 规范化换行符
        let data_str = String::from_utf8_lossy(&result);
        let normalized = data_str.replace("\r\n", "\n").replace("\r", "\n");
        result = normalized.into_bytes();
        
        // 3. 移除尾部空白
        while result.last() == Some(&b' ') || result.last() == Some(&b'\t') || result.last() == Some(&b'\n') {
            result.pop();
        }
        
        // 4. 确保文件以换行符结束（如果是文本格式）
        if !result.is_empty() && result.last() != Some(&b'\n') {
            result.push(b'\n');
        }
        
        Ok(result)
    }

    /// 清理记录数据
    pub fn clean_record_data(&self, record: &mut Record) -> Result<(), Box<dyn StdError>> {
        // 遍历记录字段，清理字符串相关的数据值
        for (_field_name, value) in record.fields.iter_mut() {
            match value {
                RecordFieldValue::Data(dv) => {
                    match dv {
                        DataValue::String(s) | DataValue::Text(s) => {
                            let trimmed = s.trim().to_string();
                            *s = trimmed;
                            // 移除控制字符（保留换行和制表符）
                            s.retain(|c| !c.is_control() || c == '\n' || c == '\t');
                        }
                        DataValue::Array(arr) => {
                            for item in arr.iter_mut() {
                                if let DataValue::String(s) | DataValue::Text(s) = item {
                                    let trimmed = s.trim().to_string();
                                    *s = trimmed;
                                    s.retain(|c| !c.is_control() || c == '\n' || c == '\t');
                                }
                            }
                        }
                        _ => {}
                    }
                }
                // 对嵌套记录可以递归清理（可选）
                RecordFieldValue::Record(inner) => {
                    self.clean_record_data(inner)?;
                }
                _ => {}
            }
        }
        
        Ok(())
    }

    /// 移除异常值
    pub fn remove_outliers(&self, records: &mut Vec<Record>, field_name: &str) -> Result<usize> {
        let mut numeric_values = Vec::new();
        
        // 收集数值数据（支持 Number / Float / Integer）
        for record in records.iter() {
            if let Some(field_val) = record.fields.get(field_name) {
                if let RecordFieldValue::Data(dv) = field_val {
                    match dv {
                        DataValue::Number(f) | DataValue::Float(f) => numeric_values.push(*f),
                        DataValue::Integer(i) => numeric_values.push(*i as f64),
                        _ => continue,
                    }
                }
            }
        }
        
        if numeric_values.is_empty() {
            return Ok(0);
        }
        
        // 计算四分位数
        numeric_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1_idx = numeric_values.len() / 4;
        let q3_idx = (numeric_values.len() * 3) / 4;
        
        let q1 = numeric_values[q1_idx];
        let q3 = numeric_values[q3_idx];
        let iqr = q3 - q1;
        
        // 定义异常值边界
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;
        
        // 移除异常值
        let original_len = records.len();
        records.retain(|record| {
            if let Some(field_val) = record.fields.get(field_name) {
                if let RecordFieldValue::Data(dv) = field_val {
                    match dv {
                        DataValue::Number(f) | DataValue::Float(f) => *f >= lower_bound && *f <= upper_bound,
                        DataValue::Integer(i) => {
                            let f = *i as f64;
                            f >= lower_bound && f <= upper_bound
                        }
                        _ => true,
                    }
                } else {
                    true
                }
            } else {
                true
            }
        });
        
        Ok(original_len - records.len())
    }

    /// 处理重复记录
    pub fn deduplicate_records(&self, records: &mut Vec<Record>) -> usize {
        let mut seen = std::collections::HashSet::new();
        let original_len = records.len();
        
        records.retain(|record| {
            // 创建记录的哈希
            let hash = self.calculate_record_hash(record);
            seen.insert(hash)
        });
        
        original_len - records.len()
    }

    /// 计算记录哈希（基于 data::record::Record 结构）
    fn calculate_record_hash(&self, record: &Record) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // 对字段按名称排序以确保一致性
        let mut fields: Vec<_> = record.fields.iter().collect();
        fields.sort_by_key(|(name, _)| *name);
        
        for (name, value) in fields {
            name.hash(&mut hasher);
            self.hash_record_field_value(value, &mut hasher);
        }
        
        hasher.finish()
    }
    
    /// 对 RecordFieldValue / DataValue 进行哈希
    fn hash_record_field_value(&self, value: &RecordFieldValue, hasher: &mut impl Hasher) {
        use std::hash::Hash;
        match value {
            RecordFieldValue::Data(dv) => {
                match dv {
                    DataValue::Null => 0u8.hash(hasher),
                    DataValue::Boolean(b) => b.hash(hasher),
                    DataValue::Integer(i) => i.hash(hasher),
                    DataValue::Float(f) | DataValue::Number(f) => f.to_bits().hash(hasher),
                    DataValue::String(s) | DataValue::Text(s) => s.hash(hasher),
                    DataValue::Array(arr) => {
                        arr.len().hash(hasher);
                        for item in arr {
                            self.hash_record_field_value(&RecordFieldValue::Data(item.clone()), hasher);
                        }
                    }
                    DataValue::Object(obj) => {
                        obj.len().hash(hasher);
                        let mut entries: Vec<_> = obj.iter().collect();
                        entries.sort_by_key(|(k, _)| *k);
                        for (k, v) in entries {
                            k.hash(hasher);
                            self.hash_record_field_value(&RecordFieldValue::Data(v.clone()), hasher);
                        }
                    }
                    DataValue::Binary(b) => b.hash(hasher),
                    DataValue::DateTime(s) => s.hash(hasher),
                    DataValue::Tensor(t) => {
                        // 粗略哈希：使用调试表示
                        format!("{:?}", t).hash(hasher);
                    }
                }
            }
            RecordFieldValue::Record(rec) => {
                // 对嵌套记录递归哈希
                let nested_hash = self.calculate_record_hash(rec);
                nested_hash.hash(hasher);
            }
            RecordFieldValue::Reference(id) => {
                id.hash(hasher);
            }
        }
    }
}

/// 数据统计工具
pub struct DataStatistics {
}

impl DataStatistics {
    /// 创建新的数据统计工具
    pub fn new() -> Self {
        Self {}
    }

    /// 内联统计（用于自检/就绪检查）
    pub fn compute_inline(&self, records: &[Record]) -> Result<HashMap<String, Value>> {
        Ok(self.calculate_basic_stats(records))
    }

    /// 计算基本统计信息
    pub fn calculate_basic_stats(&self, records: &[Record]) -> HashMap<String, Value> {
        let mut stats = HashMap::new();
        
        stats.insert("total_records".to_string(), Value::from(records.len()));
        
        if records.is_empty() {
            return stats;
        }
        
        // 统计字段信息
        let mut field_stats = HashMap::new();
        let mut field_counts = HashMap::new();
        
        for record in records {
            for (field_name, field_val) in &record.fields {
                *field_counts.entry(field_name.clone()).or_insert(0) += 1;
                
                let field_stat = field_stats.entry(field_name.clone()).or_insert_with(|| {
                    serde_json::json!({
                        "count": 0,
                        "null_count": 0,
                        "type_distribution": {}
                    })
                });
                
                if let Some(obj) = field_stat.as_object_mut() {
                    // 更新计数
                    if let Some(count) = obj.get_mut("count") {
                        *count = Value::from(count.as_u64().unwrap_or(0) + 1);
                    }
                    
                    // 基于 DataValue 统计类型和空值
                    let dv_opt = match field_val {
                        RecordFieldValue::Data(dv) => Some(dv),
                        _ => None,
                    };
                    
                    if let Some(dv) = dv_opt {
                        // 统计空值
                        if matches!(dv, DataValue::Null) {
                            if let Some(null_count) = obj.get_mut("null_count") {
                                *null_count = Value::from(null_count.as_u64().unwrap_or(0) + 1);
                            }
                        }
                        
                        // 统计类型分布
                        if let Some(type_dist) = obj.get_mut("type_distribution").and_then(|v| v.as_object_mut()) {
                            let type_name = match dv {
                                DataValue::Boolean(_) => "boolean",
                                DataValue::Integer(_) => "integer",
                                DataValue::Float(_) | DataValue::Number(_) => "number",
                                DataValue::String(_) | DataValue::Text(_) => "string",
                                DataValue::Array(_) => "array",
                                DataValue::Object(_) => "object",
                                DataValue::Binary(_) => "binary",
                                DataValue::DateTime(_) => "datetime",
                                DataValue::Null => "null",
                                DataValue::Tensor(_) => "tensor",
                            };
                            
                            let current = type_dist.get(type_name).and_then(|v| v.as_u64()).unwrap_or(0);
                            type_dist.insert(type_name.to_string(), Value::from(current + 1));
                        }
                    }
                }
            }
        }
        
        stats.insert("field_statistics".to_string(), serde_json::to_value(field_stats).unwrap_or_default());
        
        // 计算完整性统计
        let mut completeness = HashMap::new();
        for (field_name, count) in field_counts {
            let completeness_rate = (count as f64 / records.len() as f64) * 100.0;
            completeness.insert(field_name, Value::from(completeness_rate));
        }
        stats.insert("field_completeness".to_string(), serde_json::to_value(completeness).unwrap_or_default());
        
        stats
    }

    /// 计算数值字段统计
    pub fn calculate_numeric_stats(&self, records: &[Record], field_name: &str) -> Option<HashMap<String, f64>> {
        let mut values = Vec::new();
        
        for record in records {
            if let Some(field_val) = record.fields.get(field_name) {
                if let RecordFieldValue::Data(dv) = field_val {
                    match dv {
                        DataValue::Number(f) | DataValue::Float(f) => values.push(*f),
                        DataValue::Integer(i) => values.push(*i as f64),
                        _ => continue,
                    }
                }
            }
        }
        
        if values.is_empty() {
            return None;
        }
        
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let count = values.len() as f64;
        let sum: f64 = values.iter().sum();
        let mean = sum / count;
        
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / count;
        let std_dev = variance.sqrt();
        
        let min = values[0];
        let max = values[values.len() - 1];
        let median = if values.len() % 2 == 0 {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };
        
        let mut stats = HashMap::new();
        stats.insert("count".to_string(), count);
        stats.insert("mean".to_string(), mean);
        stats.insert("median".to_string(), median);
        stats.insert("std_dev".to_string(), std_dev);
        stats.insert("variance".to_string(), variance);
        stats.insert("min".to_string(), min);
        stats.insert("max".to_string(), max);
        stats.insert("sum".to_string(), sum);
        
        Some(stats)
    }
} 