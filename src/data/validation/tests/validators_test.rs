// 验证器综合测试
use crate::data::{DataValue, Record};
use crate::data::validation::{Validator, ValidationError};
use crate::data::validation::range_validator::{
    int_range, float_range, length_range, exact_length,
    IntRangeValidator, FloatRangeValidator, LengthRangeValidator,
};
use crate::data::validation::pattern_validator::{
    regex, email, url, phone, date, time, datetime, custom,
    PatternValidator, PatternType,
};

#[test]
fn test_combined_validators() {
    // 创建一条测试记录
    let mut record = Record::new();
    record.insert("user_id".to_string(), DataValue::Integer(1001));
    record.insert("score".to_string(), DataValue::Number(85.5));
    record.insert("username".to_string(), DataValue::String("alex_zhang".to_string()));
    record.insert("email".to_string(), DataValue::String("alex.zhang@example.com".to_string()));
    record.insert("phone".to_string(), DataValue::String("13812345678".to_string()));
    record.insert("create_date".to_string(), DataValue::String("2023-05-15".to_string()));

    // 创建多个验证器
    let validators: Vec<Box<dyn Validator>> = vec![
        Box::new(int_range("user_id", 1000, 9999)),
        Box::new(float_range("score", 0.0, 100.0)),
        Box::new(length_range("username", 5, 20)),
        Box::new(email("email")),
        Box::new(phone("phone")),
        Box::new(date("create_date")),
    ];

    // 运行所有验证器
    for validator in &validators {
        let result = validator.validate(&record);
        assert!(result.is_ok(), "验证失败: {}", validator.description());
    }

    // 修改记录，使其不符合验证规则
    let mut invalid_record = record.clone();
    invalid_record.insert("user_id".to_string(), DataValue::Integer(100)); // 小于最小值
    invalid_record.insert("score".to_string(), DataValue::Number(120.0)); // 大于最大值
    invalid_record.insert("username".to_string(), DataValue::String("ab".to_string())); // 长度太短
    invalid_record.insert("email".to_string(), DataValue::String("invalid-email".to_string())); // 无效邮箱
    invalid_record.insert("phone".to_string(), DataValue::String("1381234".to_string())); // 无效电话
    invalid_record.insert("create_date".to_string(), DataValue::String("2023/05/15".to_string())); // 无效日期格式

    // 验证失败记录
    for validator in &validators {
        let result = validator.validate(&invalid_record);
        assert!(result.is_err(), "应该验证失败: {}", validator.description());
    }
}

#[test]
fn test_custom_validators() {
    // 创建自定义验证器
    
    // 1. 自定义整数范围验证器 - 只允许奇数
    let odd_number_validator = custom("odd_number", |value| {
        if let Ok(num) = value.parse::<i64>() {
            return num % 2 == 1;
        }
        false
    });
    
    // 2. 自定义密码强度验证器 - 至少8位，包含大小写字母、数字和特殊字符
    let strong_password_validator = regex("password", r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$").unwrap();
    
    // 3. 自定义中国身份证号码验证器
    let id_card_validator = regex("id_card", r"^[1-9]\d{5}(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[\dX]$").unwrap();

    // 测试奇数验证器
    let mut record = Record::new();
    record.insert("odd_number".to_string(), DataValue::String("123".to_string()));
    assert!(odd_number_validator.validate(&record).is_ok());
    
    let mut record = Record::new();
    record.insert("odd_number".to_string(), DataValue::String("124".to_string()));
    assert!(odd_number_validator.validate(&record).is_err());

    // 测试密码强度验证器
    let mut record = Record::new();
    record.insert("password".to_string(), DataValue::String("Passw0rd!".to_string()));
    assert!(strong_password_validator.validate(&record).is_ok());
    
    let mut record = Record::new();
    record.insert("password".to_string(), DataValue::String("weakpwd".to_string()));
    assert!(strong_password_validator.validate(&record).is_err());

    // 测试身份证号码验证器
    let mut record = Record::new();
    record.insert("id_card".to_string(), DataValue::String("110101199001011234".to_string()));
    assert!(id_card_validator.validate(&record).is_ok());
    
    let mut record = Record::new();
    record.insert("id_card".to_string(), DataValue::String("1101011990010".to_string()));
    assert!(id_card_validator.validate(&record).is_err());
}

#[test]
fn test_validator_composition() {
    // 测试同一字段使用多个验证器
    let record = create_test_user_record();
    
    // 用户名验证 - 长度 + 模式
    let username_validators = vec![
        Box::new(length_range("username", 3, 20)),
        Box::new(regex("username", r"^[a-zA-Z][a-zA-Z0-9_]*$").unwrap()),
    ];
    
    // 电话号码验证 - 中国 + 国际格式
    let phone_validators = vec![
        Box::new(phone("phone")), // 中国手机号
        // 自定义国际电话格式验证
        Box::new(regex("int_phone", r"^\+[0-9]{1,3}-[0-9]{3,14}$").unwrap()),
    ];
    
    // 验证所有用户名验证器
    for validator in &username_validators {
        assert!(validator.validate(&record).is_ok(), 
                "用户名验证失败: {}", validator.description());
    }
    
    // 验证中国手机号格式
    assert!(phone_validators[0].validate(&record).is_ok(), 
            "中国手机号验证失败");
            
    // 验证国际电话格式
    let mut intl_record = record.clone();
    intl_record.insert("int_phone".to_string(), 
                      DataValue::String("+86-13812345678".to_string()));
    assert!(phone_validators[1].validate(&intl_record).is_ok(), 
            "国际电话验证失败");
}

#[test]
fn test_error_messages() {
    // 测试自定义错误消息
    let record = Record::new(); // 空记录
    
    // 带自定义错误消息的验证器
    let validator = int_range("age", 18, 100)
        .with_error_message("年龄必须在18-100岁之间");
    
    // 验证失败
    let result = validator.validate(&record);
    assert!(result.is_err());
    
    // 检查错误消息
    if let Err(err) = result {
        assert_eq!(err.message(), "年龄必须在18-100岁之间");
    }
    
    // 具有条件错误消息的验证器
    let mut record = Record::new();
    record.insert("age".to_string(), DataValue::Integer(10));
    
    let result = validator.validate(&record);
    assert!(result.is_err());
    
    if let Err(err) = result {
        assert_eq!(err.message(), "年龄必须在18-100岁之间");
    }
}

// 辅助函数 - 创建测试用户记录
fn create_test_user_record() -> Record {
    let mut record = Record::new();
    record.insert("user_id".to_string(), DataValue::Integer(10001));
    record.insert("username".to_string(), DataValue::String("alex_zhang".to_string()));
    record.insert("email".to_string(), DataValue::String("alex.zhang@example.com".to_string()));
    record.insert("phone".to_string(), DataValue::String("13812345678".to_string()));
    record.insert("age".to_string(), DataValue::Integer(30));
    record.insert("score".to_string(), DataValue::Number(85.5));
    record.insert("is_active".to_string(), DataValue::Boolean(true));
    record.insert("create_date".to_string(), DataValue::String("2023-05-15".to_string()));
    record
} 