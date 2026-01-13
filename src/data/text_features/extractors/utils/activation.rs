use ndarray::{Array1, Array2, Array3};

/// 激活函数类型
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    /// ReLU激活函数
    ReLU,
    /// LeakyReLU激活函数
    LeakyReLU,
    /// ELU激活函数
    ELU,
    /// SELU激活函数
    SELU,
    /// GELU激活函数
    GELU,
    /// SiLU激活函数
    SiLU,
    /// Mish激活函数
    Mish,
}

/// 激活函数实现
pub struct Activation {
    /// 激活函数类型
    activation_type: ActivationType,
    /// 参数
    params: Vec<f32>,
}

impl Activation {
    /// 创建新的激活函数
    pub fn new(activation_type: ActivationType, params: Vec<f32>) -> Self {
        Self {
            activation_type,
            params,
        }
    }

    /// 应用激活函数
    pub fn apply(&self, x: &Array1<f32>) -> Array1<f32> {
        match self.activation_type {
            ActivationType::ReLU => self.relu(x),
            ActivationType::LeakyReLU => self.leaky_relu(x),
            ActivationType::ELU => self.elu(x),
            ActivationType::SELU => self.selu(x),
            ActivationType::GELU => self.gelu(x),
            ActivationType::SiLU => self.silu(x),
            ActivationType::Mish => self.mish(x),
        }
    }

    /// ReLU激活函数
    fn relu(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|&v| v.max(0.0))
    }

    /// LeakyReLU激活函数
    fn leaky_relu(&self, x: &Array1<f32>) -> Array1<f32> {
        let alpha = self.params.get(0).copied().unwrap_or(0.01);
        x.map(|&v| if v > 0.0 { v } else { alpha * v })
    }

    /// ELU激活函数
    fn elu(&self, x: &Array1<f32>) -> Array1<f32> {
        let alpha = self.params.get(0).copied().unwrap_or(1.0);
        x.map(|&v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
    }

    /// SELU激活函数
    fn selu(&self, x: &Array1<f32>) -> Array1<f32> {
        let alpha = self.params.get(0).copied().unwrap_or(1.6733);
        let scale = self.params.get(1).copied().unwrap_or(1.0507);
        x.map(|&v| if v > 0.0 { scale * v } else { scale * alpha * (v.exp() - 1.0) })
    }

    /// GELU激活函数
    fn gelu(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|&v| 0.5 * v * (1.0 + (v / 2.0f32.sqrt()).tanh()))
    }

    /// SiLU激活函数
    fn silu(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|&v| v / (1.0 + (-v).exp()))
    }

    /// Mish激活函数
    fn mish(&self, x: &Array1<f32>) -> Array1<f32> {
        x.map(|&v| v * (1.0 + v.exp()).ln().tanh())
    }
}

impl Default for Activation {
    fn default() -> Self {
        Self::new(ActivationType::ReLU, vec![])
    }
} 