use serde::{Serialize, Deserialize};

/// Algorithm Type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlgorithmType {
    Classification,
    Regression,
    Clustering,
    DimensionReduction,
    AnomalyDetection,
    Recommendation,
    MachineLearning,
    DataProcessing,
    FeatureExtraction,
    Optimization,
    Wasm,
    Custom,
}

/// Algorithm Optimization Type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlgorithmOptimizationType {
    Performance,
    Memory,
    Accuracy,
    Speed,
    PowerEfficiency,
    Custom(String),
}

impl std::fmt::Display for AlgorithmType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgorithmType::Classification => write!(f, "classification"),
            AlgorithmType::Regression => write!(f, "regression"),
            AlgorithmType::Clustering => write!(f, "clustering"),
            AlgorithmType::DimensionReduction => write!(f, "dimension_reduction"),
            AlgorithmType::AnomalyDetection => write!(f, "anomaly_detection"),
            AlgorithmType::Recommendation => write!(f, "recommendation"),
            AlgorithmType::MachineLearning => write!(f, "machine_learning"),
            AlgorithmType::DataProcessing => write!(f, "data_processing"),
            AlgorithmType::FeatureExtraction => write!(f, "feature_extraction"),
            AlgorithmType::Optimization => write!(f, "optimization"),
            AlgorithmType::Wasm => write!(f, "wasm"),
            AlgorithmType::Custom => write!(f, "custom"),
        }
    }
}

impl AlgorithmType {
    pub fn to_string(&self) -> String {
        match self {
            AlgorithmType::Classification => "classification".to_string(),
            AlgorithmType::Regression => "regression".to_string(),
            AlgorithmType::Clustering => "clustering".to_string(),
            AlgorithmType::DimensionReduction => "dimension_reduction".to_string(),
            AlgorithmType::AnomalyDetection => "anomaly_detection".to_string(),
            AlgorithmType::Recommendation => "recommendation".to_string(),
            AlgorithmType::MachineLearning => "machine_learning".to_string(),
            AlgorithmType::DataProcessing => "data_processing".to_string(),
            AlgorithmType::FeatureExtraction => "feature_extraction".to_string(),
            AlgorithmType::Optimization => "optimization".to_string(),
            AlgorithmType::Wasm => "wasm".to_string(),
            AlgorithmType::Custom => "custom".to_string(),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AlgorithmType::Classification => "classification",
            AlgorithmType::Regression => "regression",
            AlgorithmType::Clustering => "clustering",
            AlgorithmType::DimensionReduction => "dimension_reduction",
            AlgorithmType::AnomalyDetection => "anomaly_detection",
            AlgorithmType::Recommendation => "recommendation",
            AlgorithmType::MachineLearning => "machine_learning",
            AlgorithmType::DataProcessing => "data_processing",
            AlgorithmType::FeatureExtraction => "feature_extraction",
            AlgorithmType::Optimization => "optimization",
            AlgorithmType::Wasm => "wasm",
            AlgorithmType::Custom => "custom",
        }
    }
} 