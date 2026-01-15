use std::collections::HashMap;
use std::time::SystemTime;
use serde::{Serialize, Deserialize};

/// 安全上下文
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    pub user_id: String,
    pub session_id: String,
    pub permissions: Vec<Permission>,
    pub security_level: SecurityLevel,
    pub audit_trail: Vec<AuditEvent>,
    pub created_at: SystemTime,
    pub expires_at: Option<SystemTime>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

/// 安全级别
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecurityLevel {
    Public,
    Internal,
    Confidential,
    Secret,
    TopSecret,
}

/// 权限定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    pub resource: String,
    pub actions: Vec<Action>,
    pub conditions: Vec<Condition>,
    pub granted_at: SystemTime,
    pub granted_by: String,
    pub expires_at: Option<SystemTime>,
}

/// 操作类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Read,
    Write,
    Delete,
    Execute,
    Admin,
    Create,
    Update,
    List,
    Custom(String),
}

/// 条件定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub condition_type: ConditionType,
    pub operator: Operator,
    pub value: String,
    pub case_sensitive: bool,
}

/// 条件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    TimeRange,
    IpAddress,
    UserAgent,
    ResourcePath,
    Custom(String),
}

/// 操作符
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    GreaterThan,
    LessThan,
    InRange,
    NotInRange,
}

/// 审计事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: String,
    pub event_type: AuditEventType,
    pub timestamp: SystemTime,
    pub user_id: String,
    pub resource: String,
    pub action: Action,
    pub result: AuditResult,
    pub details: HashMap<String, String>,
    pub risk_score: f64,
}

/// 审计事件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    SystemAccess,
    SecurityViolation,
    ConfigurationChange,
    Custom(String),
}

/// 审计结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure,
    Blocked,
    Warning,
}

/// 安全策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    pub policy_id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub rules: Vec<SecurityRule>,
    pub enforcement_level: EnforcementLevel,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub created_by: String,
    pub is_active: bool,
}

/// 安全规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRule {
    pub rule_id: String,
    pub name: String,
    pub description: String,
    pub rule_type: SecurityRuleType,
    pub conditions: Vec<RuleCondition>,
    pub actions: Vec<RuleAction>,
    pub priority: i32,
    pub is_enabled: bool,
}

/// 安全规则类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityRuleType {
    AccessControl,
    DataProtection,
    AnomalyDetection,
    ComplianceCheck,
    ThreatPrevention,
    Custom(String),
}

/// 规则条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    pub field: String,
    pub operator: Operator,
    pub value: String,
    pub negate: bool,
}

/// 规则动作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuleAction {
    Allow,
    Deny,
    Log,
    Alert,
    Quarantine,
    Encrypt,
    Decrypt,
    Custom(String),
}

/// 执行级别
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnforcementLevel {
    Advisory,
    Warning,
    Enforced,
    Strict,
}

/// 加密配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub algorithm: EncryptionAlgorithm,
    pub key_size: usize,
    pub mode: EncryptionMode,
    pub padding: PaddingScheme,
    pub key_derivation: KeyDerivationFunction,
}

/// 加密算法
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES,
    ChaCha20,
    RSA,
    ECC,
    Custom(String),
}

/// 加密模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionMode {
    ECB,
    CBC,
    CTR,
    GCM,
    XTS,
}

/// 填充方案
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingScheme {
    PKCS7,
    OAEP,
    PSS,
    None,
}

/// 密钥派生函数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyDerivationFunction {
    PBKDF2,
    Scrypt,
    Argon2,
    HKDF,
}

/// 威胁检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionResult {
    pub threat_id: String,
    pub threat_type: ThreatType,
    pub severity: ThreatSeverity,
    pub confidence: f64,
    pub description: String,
    pub affected_resources: Vec<String>,
    pub mitigation_suggestions: Vec<String>,
    pub detected_at: SystemTime,
    pub metadata: HashMap<String, String>,
}

/// 威胁类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatType {
    UnauthorizedAccess,
    DataExfiltration,
    PrivilegeEscalation,
    AnomalousActivity,
    MaliciousCode,
    SystemCompromise,
    Custom(String),
}

/// 威胁严重程度
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl SecurityContext {
    /// 创建新的安全上下文
    pub fn new(user_id: String, session_id: String, security_level: SecurityLevel) -> Self {
        Self {
            user_id,
            session_id,
            permissions: Vec::new(),
            security_level,
            audit_trail: Vec::new(),
            created_at: SystemTime::now(),
            expires_at: None,
            ip_address: None,
            user_agent: None,
        }
    }

    /// 添加权限
    pub fn add_permission(&mut self, permission: Permission) {
        self.permissions.push(permission);
    }

    /// 检查权限
    pub fn has_permission(&self, resource: &str, action: &Action) -> bool {
        self.permissions.iter().any(|perm| {
            perm.resource == resource && perm.actions.contains(action) && !self.is_permission_expired(perm)
        })
    }

    /// 检查权限是否过期
    fn is_permission_expired(&self, permission: &Permission) -> bool {
        if let Some(expires_at) = permission.expires_at {
            SystemTime::now() > expires_at
        } else {
            false
        }
    }

    /// 添加审计事件
    pub fn add_audit_event(&mut self, event: AuditEvent) {
        self.audit_trail.push(event);
    }

    /// 验证安全上下文
    pub fn validate(&self) -> bool {
        // 检查是否过期
        if let Some(expires_at) = self.expires_at {
            if SystemTime::now() > expires_at {
                return false;
            }
        }

        // 检查权限是否有效
        self.permissions.iter().all(|perm| !self.is_permission_expired(perm))
    }

    /// 升级安全级别
    pub fn upgrade_security_level(&mut self, new_level: SecurityLevel) -> Result<(), String> {
        if new_level > self.security_level {
            self.security_level = new_level;
            Ok(())
        } else {
            Err("Cannot downgrade security level".to_string())
        }
    }
}

impl SecurityPolicy {
    /// 创建新的安全策略
    pub fn new(policy_id: String, name: String, created_by: String) -> Self {
        let now = SystemTime::now();
        Self {
            policy_id,
            name,
            description: String::new(),
            version: "1.0.0".to_string(),
            rules: Vec::new(),
            enforcement_level: EnforcementLevel::Enforced,
            created_at: now,
            updated_at: now,
            created_by,
            is_active: true,
        }
    }

    /// 添加规则
    pub fn add_rule(&mut self, rule: SecurityRule) {
        self.rules.push(rule);
        self.updated_at = SystemTime::now();
    }

    /// 移除规则
    pub fn remove_rule(&mut self, rule_id: &str) -> bool {
        let initial_len = self.rules.len();
        self.rules.retain(|rule| rule.rule_id != rule_id);
        if self.rules.len() < initial_len {
            self.updated_at = SystemTime::now();
            true
        } else {
            false
        }
    }

    /// 评估策略
    pub fn evaluate(&self, context: &SecurityContext) -> Vec<RuleAction> {
        let mut actions = Vec::new();
        
        for rule in &self.rules {
            if rule.is_enabled && self.evaluate_rule(rule, context) {
                actions.extend(rule.actions.clone());
            }
        }
        
        actions
    }

    /// 评估单个规则
    fn evaluate_rule(&self, rule: &SecurityRule, context: &SecurityContext) -> bool {
        rule.conditions.iter().all(|condition| {
            self.evaluate_condition(condition, context)
        })
    }

    /// 评估条件
    fn evaluate_condition(&self, condition: &RuleCondition, _context: &SecurityContext) -> bool {
        // 基础条件评估：支持等值比较
        // 完整实现应支持所有操作符（Equals, NotEquals, GreaterThan, LessThan等）
        // 并根据 SecurityContext 中的实际值进行评估
        match condition.operator {
            Operator::Equals => condition.value == "true",
            _ => true, // 其他操作符默认返回 true，完整实现需要根据上下文值进行评估
        }
    }
}

impl ThreatDetectionResult {
    /// 创建新的威胁检测结果
    pub fn new(threat_type: ThreatType, severity: ThreatSeverity, description: String) -> Self {
        Self {
            threat_id: uuid::Uuid::new_v4().to_string(),
            threat_type,
            severity,
            confidence: 0.0,
            description,
            affected_resources: Vec::new(),
            mitigation_suggestions: Vec::new(),
            detected_at: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// 添加受影响的资源
    pub fn add_affected_resource(&mut self, resource: String) {
        self.affected_resources.push(resource);
    }

    /// 添加缓解建议
    pub fn add_mitigation_suggestion(&mut self, suggestion: String) {
        self.mitigation_suggestions.push(suggestion);
    }

    /// 设置置信度
    pub fn set_confidence(&mut self, confidence: f64) {
        self.confidence = confidence.clamp(0.0, 1.0);
    }
} 