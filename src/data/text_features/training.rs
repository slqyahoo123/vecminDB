// src/data/text_features/training.rs
//
// Transformer 训练模块

use std::sync::{Arc, Mutex};
// NOTE: Training functionality is disabled in vecminDB. Keep imports minimal.
use super::error::TransformerError;
use super::config::TransformerConfig;
use super::model::{TransformerModel, TrainingExample, TrainingResult};
use super::encoder::Encoder;

/// 训练配置
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// 学习率
    pub learning_rate: f32,
    /// 批次大小
    pub batch_size: usize,
    /// 训练轮数
    pub epochs: usize,
    /// 验证比例
    pub validation_split: f32,
    /// 早停耐心值
    pub early_stopping_patience: usize,
    /// 是否使用梯度裁剪
    pub use_gradient_clipping: bool,
    /// 梯度裁剪阈值
    pub gradient_clip_threshold: f32,
    /// 是否使用学习率调度
    pub use_learning_rate_scheduling: bool,
    /// 权重衰减
    pub weight_decay: f32,
    /// 是否使用混合精度训练
    pub use_mixed_precision: bool,
    /// 是否使用数据增强
    pub use_data_augmentation: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            validation_split: 0.2,
            early_stopping_patience: 10,
            use_gradient_clipping: true,
            gradient_clip_threshold: 1.0,
            use_learning_rate_scheduling: true,
            weight_decay: 0.0001,
            use_mixed_precision: false,
            use_data_augmentation: false,
        }
    }
}

/// 训练器
pub struct Trainer {
    /// 训练配置
    config: TrainingConfig,
    /// 模型
    model: Arc<Mutex<TransformerModel>>,
    /// 优化器
    optimizer: Box<dyn Optimizer>,
    /// 损失函数
    loss_function: Box<dyn LossFunction>,
    /// 学习率调度器
    scheduler: Option<Box<dyn LearningRateScheduler>>,
    /// 训练历史
    history: TrainingHistory,
    /// 早停监控器
    early_stopping: EarlyStopping,
}

impl Trainer {
    /// 创建新的训练器
    pub fn new(
        config: TrainingConfig,
        model: TransformerModel,
        optimizer: Box<dyn Optimizer>,
        loss_function: Box<dyn LossFunction>,
    ) -> Self {
        let scheduler: Option<Box<dyn LearningRateScheduler>> = if config.use_learning_rate_scheduling {
            Some(Box::new(StepLR::new(config.learning_rate, 30, 0.1)) as Box<dyn LearningRateScheduler>)
        } else {
            None
        };
        
        Self {
            config,
            model: Arc::new(Mutex::new(model)),
            optimizer,
            loss_function,
            scheduler,
            history: TrainingHistory::new(),
            early_stopping: EarlyStopping::new(),
        }
    }

    /// 通过 TransformerConfig 与 Encoder 进行快速构造（最小闭环接线）
    pub fn from_configs(
        training: TrainingConfig,
        model_cfg: TransformerConfig,
        encoder: Option<Encoder>,
        optimizer: Box<dyn Optimizer>,
        loss_function: Box<dyn LossFunction>,
    ) -> Result<Self, TransformerError> {
        let mut model = TransformerModel::new(model_cfg)?;
        if let Some(enc) = encoder {
            // 将外部编码器接入模型的特征提取器
            model.set_encoder(enc);
        }
        Ok(Self::new(training, model, optimizer, loss_function))
    }
    
    /// 训练模型
    ///
    /// 注意：vecminDB 是一个向量数据库，不提供模型训练功能。
    /// 训练功能已被禁用，调用此方法将返回错误。
    pub fn train(&mut self, training_data: Vec<TrainingExample>) -> Result<TrainingResult, TransformerError> {
        // Training is not supported in vecminDB - this is a vector database, not a training platform.
        // We return an explicit error here and intentionally do NOT execute any training loop logic.
        Err(TransformerError::ComputationError(format!(
            "model training is not supported in vecminDB (attempted to train with {} examples)",
            training_data.len()
        )))
    }
    
    /// 训练单个epoch
    fn train_epoch(&mut self, data: &[TrainingExample]) -> Result<EpochMetrics, TransformerError> {
        let mut total_loss = 0.0;
        let mut total_steps = 0;
        
        // 数据增强
        let augmented_data = if self.config.use_data_augmentation {
            self.augment_data(data)?
        } else {
            data.to_vec()
        };
        
        // 批次训练
        for batch in augmented_data.chunks(self.config.batch_size) {
            let batch_loss = self.train_batch(batch)?;
            total_loss += batch_loss;
            total_steps += 1;
        }
        
        let avg_loss = total_loss / total_steps as f32;
        
        Ok(EpochMetrics {
            loss: avg_loss,
            accuracy: 0.0, // Training not supported in vector database - always 0.0
            steps: total_steps,
        })
    }
    
    /// 训练单个批次
    fn train_batch(&mut self, batch: &[TrainingExample]) -> Result<f32, TransformerError> {
        let mut model = self.model.lock().unwrap();
        
        // 前向传播
        let mut batch_loss = 0.0;
        let mut gradients = Vec::new();
        
        for example in batch {
            // 处理输入
            let processed = model.process_text(&example.input)?;
            
            // 前向传播
            let prediction = self.forward_pass(&processed.encoded)?;
            
            // 计算损失
            let loss = self.loss_function.compute(&prediction, &example.target)?;
            batch_loss += loss;
            
            // Backpropagation (training not supported in vecminDB)
            let grad = self.backward_pass(&prediction, &example.target)?;
            gradients.push(grad);
        }
        
        // 平均梯度
        let avg_gradients = self.average_gradients(&gradients)?;
        
        // 梯度裁剪
        let clipped_gradients = if self.config.use_gradient_clipping {
            self.clip_gradients(&avg_gradients)?
        } else {
            avg_gradients
        };
        
        // 更新参数
        self.optimizer.update(&clipped_gradients)?;
        
        Ok(batch_loss / batch.len() as f32)
    }
    
    /// 验证单个epoch
    fn validate_epoch(&self, data: &[TrainingExample]) -> Result<EpochMetrics, TransformerError> {
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;
        
        for batch in data.chunks(self.config.batch_size) {
            for example in batch {
                let model = self.model.lock().unwrap();
                let processed = model.process_text(&example.input)?;
                let prediction = self.forward_pass(&processed.encoded)?;
                
                let loss = self.loss_function.compute(&prediction, &example.target)?;
                total_loss += loss;
                
                // Calculate accuracy (training not supported in vecminDB)
                if self.is_correct_prediction(&prediction, &example.target) {
                    correct_predictions += 1;
                }
                total_predictions += 1;
            }
        }
        
        let avg_loss = total_loss / data.len() as f32;
        let accuracy = correct_predictions as f32 / total_predictions as f32;
        
        Ok(EpochMetrics {
            loss: avg_loss,
            accuracy,
            steps: data.len(),
        })
    }
    
    /// 前向传播
    fn forward_pass(&self, input: &[f32]) -> Result<Vec<f32>, TransformerError> {
        // 简化的前向传播实现
        // 在实际应用中，这里应该使用真正的神经网络前向传播
        Ok(input.to_vec())
    }
    
    /// 反向传播
    fn backward_pass(&self, prediction: &[f32], target: &[f32]) -> Result<Vec<f32>, TransformerError> {
        // 简化的反向传播实现
        // 计算梯度
        let mut gradients = Vec::with_capacity(prediction.len());
        
        for (pred, targ) in prediction.iter().zip(target.iter()) {
            let gradient = 2.0 * (pred - targ); // MSE损失的梯度
            gradients.push(gradient);
        }
        
        Ok(gradients)
    }
    
    /// 平均梯度
    fn average_gradients(&self, gradients: &[Vec<f32>]) -> Result<Vec<f32>, TransformerError> {
        if gradients.is_empty() {
            return Err(TransformerError::computation_error("梯度列表为空"));
        }
        
        let grad_len = gradients[0].len();
        let mut avg_gradients = vec![0.0; grad_len];
        
        for gradient in gradients {
            if gradient.len() != grad_len {
                return Err(TransformerError::dimension_mismatch(
                    format!("{}", gradient.len()),
                    format!("{}", grad_len)
                ));
            }
            
            for (i, &grad) in gradient.iter().enumerate() {
                avg_gradients[i] += grad;
            }
        }
        
        // 计算平均值
        let num_gradients = gradients.len() as f32;
        for grad in &mut avg_gradients {
            *grad /= num_gradients;
        }
        
        Ok(avg_gradients)
    }
    
    /// 梯度裁剪
    fn clip_gradients(&self, gradients: &[f32]) -> Result<Vec<f32>, TransformerError> {
        let threshold = self.config.gradient_clip_threshold;
        let mut clipped = Vec::with_capacity(gradients.len());
        
        for &grad in gradients {
            let clipped_grad = if grad.abs() > threshold {
                if grad > 0.0 {
                    threshold
                } else {
                    -threshold
                }
            } else {
                grad
            };
            clipped.push(clipped_grad);
        }
        
        Ok(clipped)
    }
    
    /// 判断预测是否正确
    fn is_correct_prediction(&self, prediction: &[f32], target: &[f32]) -> bool {
        if prediction.len() != target.len() {
            return false;
        }
        
        // 简化的正确性判断
        // 在实际应用中，这里应该根据具体任务类型来判断
        let mut total_diff = 0.0;
        for (pred, targ) in prediction.iter().zip(target.iter()) {
            total_diff += (pred - targ).abs();
        }
        
        let avg_diff = total_diff / prediction.len() as f32;
        avg_diff < 0.1 // 阈值可调整
    }
    
    /// 分割数据
    fn split_data(&self, data: Vec<TrainingExample>) -> Result<(Vec<TrainingExample>, Vec<TrainingExample>), TransformerError> {
        let split_idx = (data.len() as f32 * (1.0 - self.config.validation_split)) as usize;
        
        if split_idx == 0 || split_idx >= data.len() {
            return Err(TransformerError::InputError("数据分割失败".to_string()));
        }
        
        let (train, val) = data.split_at(split_idx);
        Ok((train.to_vec(), val.to_vec()))
    }
    
    /// 数据增强
    fn augment_data(&self, data: &[TrainingExample]) -> Result<Vec<TrainingExample>, TransformerError> {
        let mut augmented = Vec::new();
        
        for example in data {
            // 添加原始数据
            augmented.push(example.clone());
            
            // 简单的数据增强：添加噪声
            let augmented_example = self.add_noise_to_example(example)?;
            augmented.push(augmented_example);
        }
        
        Ok(augmented)
    }
    
    /// 为示例添加噪声
    fn add_noise_to_example(&self, example: &TrainingExample) -> Result<TrainingExample, TransformerError> {
        let mut noisy_target = example.target.clone();
        
        // 添加随机噪声
        for value in &mut noisy_target {
            let noise = (rand::random::<f32>() - 0.5) * 0.1;
            *value += noise;
        }
        
        Ok(TrainingExample {
            input: example.input.clone(),
            target: noisy_target,
            label: example.label.clone(),
        })
    }
    
    /// 保存最佳模型
    fn save_best_model(&self) -> Result<(), TransformerError> {
        let model = self.model.lock().unwrap();
        model.save_model("best_model.json")
    }
    
    /// 加载最佳模型
    fn load_best_model(&self) -> Result<(), TransformerError> {
        let best_model = TransformerModel::load_model("best_model.json")?;
        let mut model = self.model.lock().unwrap();
        *model = best_model;
        Ok(())
    }
    
    /// 获取训练历史
    pub fn get_history(&self) -> &TrainingHistory {
        &self.history
    }
    
    /// 获取配置
    pub fn get_config(&self) -> &TrainingConfig {
        &self.config
    }
}

/// 优化器特征
pub trait Optimizer: Send + Sync {
    /// 更新参数
    fn update(&mut self, gradients: &[f32]) -> Result<(), TransformerError>;
    /// 获取学习率
    fn get_learning_rate(&self) -> f32;
    /// 设置学习率
    fn set_learning_rate(&mut self, lr: f32);
}

/// SGD优化器
#[derive(Debug)]
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    velocity: Vec<f32>,
}

impl SGD {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, gradients: &[f32]) -> Result<(), TransformerError> {
        if self.velocity.is_empty() {
            self.velocity = vec![0.0; gradients.len()];
        }
        
        for (i, &gradient) in gradients.iter().enumerate() {
            self.velocity[i] = self.momentum * self.velocity[i] + self.learning_rate * gradient;
        }
        
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// Adam优化器
#[derive(Debug)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, gradients: &[f32]) -> Result<(), TransformerError> {
        if self.m.is_empty() {
            self.m = vec![0.0; gradients.len()];
            self.v = vec![0.0; gradients.len()];
        }
        
        self.t += 1;
        
        for (i, &gradient) in gradients.iter().enumerate() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * gradient;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * gradient * gradient;
            
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));
            
            // 参数更新计算（不实际应用，因为训练功能已禁用）
            let _update = self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
        
        Ok(())
    }
    
    fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

/// 损失函数特征
pub trait LossFunction: Send + Sync {
    /// 计算损失
    fn compute(&self, prediction: &[f32], target: &[f32]) -> Result<f32, TransformerError>;
    /// 计算梯度
    fn gradient(&self, prediction: &[f32], target: &[f32]) -> Result<Vec<f32>, TransformerError>;
}

/// 均方误差损失
#[derive(Debug)]
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn compute(&self, prediction: &[f32], target: &[f32]) -> Result<f32, TransformerError> {
        if prediction.len() != target.len() {
            return Err(TransformerError::dimension_mismatch(
                format!("{}", prediction.len()),
                format!("{}", target.len())
            ));
        }
        
        let mut loss = 0.0;
        for (pred, targ) in prediction.iter().zip(target.iter()) {
            let diff = pred - targ;
            loss += diff * diff;
        }
        
        Ok(loss / prediction.len() as f32)
    }
    
    fn gradient(&self, prediction: &[f32], target: &[f32]) -> Result<Vec<f32>, TransformerError> {
        if prediction.len() != target.len() {
            return Err(TransformerError::dimension_mismatch(
                format!("{}", prediction.len()),
                format!("{}", target.len())
            ));
        }
        
        let mut gradients = Vec::with_capacity(prediction.len());
        for (pred, targ) in prediction.iter().zip(target.iter()) {
            gradients.push(2.0 * (pred - targ));
        }
        
        Ok(gradients)
    }
}

/// 交叉熵损失
#[derive(Debug)]
pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn compute(&self, prediction: &[f32], target: &[f32]) -> Result<f32, TransformerError> {
        if prediction.len() != target.len() {
            return Err(TransformerError::dimension_mismatch(
                format!("{}", prediction.len()),
                format!("{}", target.len())
            ));
        }
        
        let mut loss = 0.0;
        for (pred, targ) in prediction.iter().zip(target.iter()) {
            let pred_clamped = pred.max(1e-7).min(1.0 - 1e-7);
            loss -= targ * pred_clamped.ln() + (1.0 - targ) * (1.0 - pred_clamped).ln();
        }
        
        Ok(loss / prediction.len() as f32)
    }
    
    fn gradient(&self, prediction: &[f32], target: &[f32]) -> Result<Vec<f32>, TransformerError> {
        if prediction.len() != target.len() {
            return Err(TransformerError::dimension_mismatch(
                format!("{}", prediction.len()),
                format!("{}", target.len())
            ));
        }
        
        let mut gradients = Vec::with_capacity(prediction.len());
        for (pred, targ) in prediction.iter().zip(target.iter()) {
            let pred_clamped = pred.max(1e-7).min(1.0 - 1e-7);
            gradients.push((pred_clamped - targ) / (pred_clamped * (1.0 - pred_clamped)));
        }
        
        Ok(gradients)
    }
}

/// 学习率调度器特征
pub trait LearningRateScheduler: Send + Sync {
    /// 更新学习率
    fn step(&mut self);
    /// 获取当前学习率
    fn get_learning_rate(&self) -> f32;
}

/// 步进学习率调度器
#[derive(Debug)]
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_step: usize,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl LearningRateScheduler for StepLR {
    fn step(&mut self) {
        self.current_step += 1;
    }
    
    fn get_learning_rate(&self) -> f32 {
        let factor = (self.current_step / self.step_size) as f32;
        self.initial_lr * self.gamma.powi(factor as i32)
    }
}

/// 训练历史
#[derive(Debug)]
pub struct TrainingHistory {
    /// 训练轮数
    pub epochs: Vec<usize>,
    /// 训练损失
    pub train_losses: Vec<f32>,
    /// 验证损失
    pub val_losses: Vec<f32>,
    /// 训练准确率
    pub train_accuracies: Vec<f32>,
    /// 验证准确率
    pub val_accuracies: Vec<f32>,
    /// 总步数
    pub total_steps: usize,
}

impl TrainingHistory {
    pub fn new() -> Self {
        Self {
            epochs: Vec::new(),
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
            total_steps: 0,
        }
    }
    
    pub fn add_epoch(&mut self, epoch: usize, train_metrics: EpochMetrics, val_metrics: EpochMetrics) {
        self.epochs.push(epoch);
        self.train_losses.push(train_metrics.loss);
        self.val_losses.push(val_metrics.loss);
        self.train_accuracies.push(train_metrics.accuracy);
        self.val_accuracies.push(val_metrics.accuracy);
        self.total_steps += train_metrics.steps;
    }
}

/// 早停监控器
#[derive(Debug)]
pub struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    counter: usize,
    best_loss: f32,
}

impl EarlyStopping {
    pub fn new() -> Self {
        Self {
            patience: 10,
            min_delta: 1e-4,
            counter: 0,
            best_loss: f32::INFINITY,
        }
    }
    
    pub fn check(&mut self, val_loss: f32) -> bool {
        if val_loss < self.best_loss - self.min_delta {
            self.best_loss = val_loss;
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }
}

/// Epoch指标
impl std::fmt::Debug for Trainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trainer")
            .field("config", &self.config)
            .field("model", &"Arc<Mutex<TransformerModel>>")
            .field("optimizer", &"Box<dyn Optimizer>")
            .field("loss_function", &"Box<dyn LossFunction>")
            .field("scheduler", &self.scheduler.as_ref().map(|_| "Box<dyn LearningRateScheduler>"))
            .field("history", &self.history)
            .field("early_stopping", &self.early_stopping)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// 损失
    pub loss: f32,
    /// 准确率
    pub accuracy: f32,
    /// 步数
    pub steps: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default();
        let model = TransformerModel::new(TransformerConfig::default()).unwrap();
        let optimizer = Box::new(SGD::new(0.001, 0.9));
        let loss_function = Box::new(MSELoss);
        
        let trainer = Trainer::new(config, model, optimizer, loss_function);
        assert_eq!(trainer.config.learning_rate, 0.001);
    }

    #[test]
    fn test_optimizer_sgd() {
        let mut optimizer = SGD::new(0.01, 0.9);
        let gradients = vec![1.0, -1.0, 0.5];
        
        let result = optimizer.update(&gradients);
        assert!(result.is_ok());
        assert_eq!(optimizer.get_learning_rate(), 0.01);
    }

    #[test]
    fn test_loss_function_mse() {
        let loss_fn = MSELoss;
        let prediction = vec![1.0, 2.0, 3.0];
        let target = vec![1.1, 1.9, 3.1];
        
        let loss = loss_fn.compute(&prediction, &target);
        assert!(loss.is_ok());
        assert!(loss.unwrap() > 0.0);
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let mut scheduler = StepLR::new(0.1, 10, 0.5);
        
        assert_eq!(scheduler.get_learning_rate(), 0.1);
        
        for _ in 0..10 {
            scheduler.step();
        }
        
        assert_eq!(scheduler.get_learning_rate(), 0.05);
    }
} 