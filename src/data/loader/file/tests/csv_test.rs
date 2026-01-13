#[cfg(test)]
mod csv_tests {
	use crate::data::loader::file::csv::CSVProcessor;
	use crate::data::loader::FileDataLoader;
	use crate::data::DataConfig;
	use crate::error::Result;
	use std::fs::{self, File};
	use std::io::Write;
	use tempfile::tempdir;

	// 创建测试CSV文件
	fn create_test_csv(content: &str) -> Result<(tempfile::TempDir, String)> {
		let dir = tempdir().unwrap();
		let file_path = dir.path().join("test.csv");
		let mut file = File::create(&file_path).unwrap();
		file.write_all(content.as_bytes()).unwrap();
		
		Ok((dir, file_path.to_string_lossy().to_string()))
	}

	#[test]
	fn test_detect_delimiter() -> Result<()> {
		// 测试不同的分隔符
		let test_cases = vec![
			// 逗号分隔
			("a,b,c\n1,2,3\n4,5,6", ','),
			// 分号分隔
			("a;b;c\n1;2;3\n4;5;6", ';'),
			// 制表符分隔
			("a\tb\tc\n1\t2\t3\n4\t5\t6", '\t'),
			// 竖线分隔
			("a|b|c\n1|2|3\n4|5|6", '|'),
		];

		for (content, expected_delimiter) in test_cases {
			let (dir, file_path) = create_test_csv(content)?;
			let _processor = CSVProcessor::new(&file_path, None)?;
			let delimiter = CSVProcessor::detect_delimiter(&file_path)?;
			
			assert_eq!(delimiter, expected_delimiter as u8);
			drop(dir); // 清理临时目录
		}

		Ok(())
	}

	#[test]
	fn test_infer_schema() -> Result<()> {
		// 创建带有各种数据类型的CSV
		let content = "Name,Age,IsActive,JoinDate,Score\n\
					  Alice,30,true,2020-01-15,95.5\n\
					  Bob,25,false,2021-03-20,87.3\n\
					  Charlie,42,true,2019-07-10,76.8";
		
		let (dir, file_path) = create_test_csv(content)?;
		
		// 使用处理器推断模式
		let processor = CSVProcessor::new(&file_path, None)?;
		let schema = processor.infer_schema()?;
		
		// 验证推断的模式
		let field_names = schema.field_names.unwrap();
		assert_eq!(field_names.len(), 5);
		assert_eq!(field_names[0], "Name");
		assert_eq!(field_names[1], "Age");
		assert_eq!(field_names[2], "IsActive");
		assert_eq!(field_names[3], "JoinDate");
		assert_eq!(field_names[4], "Score");
		
		// 验证字段类型
		let field_types = schema.field_types.unwrap();
		assert_eq!(field_types.len(), 5);
		// 这里验证类型，但是具体的值取决于LineParser的实现
		
		// 验证特征字段识别
		let feature_fields = schema.feature_fields.unwrap();
		assert!(feature_fields.contains(&"Age".to_string()));
		assert!(feature_fields.contains(&"IsActive".to_string()));
		assert!(feature_fields.contains(&"Score".to_string()));
		
		drop(dir); // 清理临时目录
		Ok(())
	}

	#[test]
	fn test_batch_read() -> Result<()> {
		let content = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9";
		let (dir, file_path) = create_test_csv(content)?;
		
		let processor = CSVProcessor::new(&file_path, None)?;
		let data = processor.read_batch(None)?;
		
		assert_eq!(data.len(), 3); // 3行数据
		assert_eq!(data[0], vec!["1", "2", "3"]);
		assert_eq!(data[1], vec!["4", "5", "6"]);
		assert_eq!(data[2], vec!["7", "8", "9"]);
		
		// 测试限制行数
		let limited_data = processor.read_batch(Some(2))?;
		assert_eq!(limited_data.len(), 2);
		
		drop(dir);
		Ok(())
	}

	#[test]
	fn test_stream_processing() -> Result<()> {
		let content = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n10,11,12";
		let (dir, file_path) = create_test_csv(content)?;
		
		let processor = CSVProcessor::new(&file_path, None)?;
		
		// 使用流式处理收集数据
		let mut all_data = Vec::new();
		let batch_size = 2;
		
		let processed_count = processor.process_stream(
			batch_size,
			|batch, _is_first| {
				all_data.extend(batch);
				Ok(true) // 继续处理
			}
		)?;
		
		assert_eq!(processed_count, 4); // 4行数据
		assert_eq!(all_data.len(), 4);
		
		// 验证处理到一半就停止
		let mut partial_data = Vec::new();
		let _ = processor.process_stream(
			batch_size,
			|batch, _is_first| {
				partial_data.extend(batch);
				Ok(false) // 处理一个批次后停止
			}
		)?;
		
		assert_eq!(partial_data.len(), batch_size);
		
		drop(dir);
		Ok(())
	}

	#[test]
	fn test_write_csv() -> Result<()> {
		let dir = tempdir().unwrap();
		let file_path = dir.path().join("output.csv");
		
		// 准备数据
		let headers = vec!["Name".to_string(), "Value".to_string()];
		let data = vec![
			vec!["Item1".to_string(), "100".to_string()],
			vec!["Item2".to_string(), "200".to_string()],
			vec!["Item3".to_string(), "300".to_string()],
		];
		
		// 写入数据
		let processor = CSVProcessor::new(file_path.to_str().unwrap(), None)?;
		processor.write_csv(
			&file_path,
			&data,
			Some(&headers)
		)?;
		
		// 读取并验证
		let content = fs::read_to_string(&file_path).unwrap();
		let expected = "Name,Value\nItem1,100\nItem2,200\nItem3,300\n";
		
		assert_eq!(content, expected);
		
		// 不带标题的测试
		let file_path2 = dir.path().join("output2.csv");
		processor.write_csv(&file_path2, &data, None)?;
		
		let content2 = fs::read_to_string(&file_path2).unwrap();
		let expected2 = "Item1,100\nItem2,200\nItem3,300\n";
		
		assert_eq!(content2, expected2);
		
		drop(dir);
		Ok(())
	}

	#[test]
	fn test_file_data_loader_integration() -> Result<()> {
		let content = "Name,Age,IsActive\nAlice,30,true\nBob,25,false";
		let (dir, file_path) = create_test_csv(content)?;
		
		let config = DataConfig {
			skip_header: true,
			..Default::default()
		};
		
		let loader = FileDataLoader::new(config);
		
		// 测试模式推断
		let schema = loader.infer_csv_schema(&file_path)?;
		assert_eq!(schema.field_names.unwrap().len(), 3);
		
		// 测试批量读取
		let data = loader.read_csv_batch(&file_path, None)?;
		assert_eq!(data.len(), 2);
		assert_eq!(data[0][0], "Alice");
		
		drop(dir);
		Ok(())
	}
} 