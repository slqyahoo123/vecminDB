// src/data/text_features/encoder.rs
//
// Transformer 编码器模块

use std::collections::HashMap;
use super::error::TransformerError;
use super::config::TransformerConfig;
use super::tokenizer::Tokenizer;

/// 编码器配置
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// 是否使用位置编码
    pub use_positional_encoding: bool,
    /// 是否使用注意力机制
    pub use_attention: bool,
    /// 是否使用残差连接
    pub use_residual: bool,
    /// 是否使用层归一化
    pub use_layer_norm: bool,
    /// Dropout率
    pub dropout_rate: f32,
    /// 是否启用梯度检查点
    pub use_gradient_checkpointing: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            use_positional_encoding: true,
            use_attention: true,
            use_residual: true,
            use_layer_norm: true,
            dropout_rate: 0.1,
            use_gradient_checkpointing: false,
        }
    }
}

/// 编码器
#[derive(Debug, Clone)]
pub struct Encoder {
    /// 配置
    config: EncoderConfig,
    /// 分词器
    tokenizer: Tokenizer,
    /// 词汇表大小
    vocab_size: usize,
    /// 隐藏层大小
    hidden_size: usize,
    /// 最大序列长度
    max_seq_length: usize,
    /// 注意力头数
    num_heads: usize,
    /// 层数
    num_layers: usize,
    /// 前馈网络维度
    feed_forward_dim: usize,
    /// 位置编码
    positional_encoding: Option<Vec<Vec<f32>>>,
    /// 层权重
    layer_weights: HashMap<String, Vec<f32>>,
}

impl Encoder {
    /// 创建新的编码器
    pub fn new(config: EncoderConfig, tokenizer: Tokenizer, model_config: &TransformerConfig) -> Self {
        let vocab_size = tokenizer.vocabulary().size();
        let hidden_size = model_config.hidden_size;
        let max_seq_length = model_config.max_seq_length;
        let num_heads = model_config.num_heads;
        let num_layers = model_config.num_layers;
        let feed_forward_dim = model_config.feed_forward_dim;
        
        let mut encoder = Self {
            config,
            tokenizer,
            vocab_size,
            hidden_size,
            max_seq_length,
            num_heads,
            num_layers,
            feed_forward_dim,
            positional_encoding: None,
            layer_weights: HashMap::new(),
        };
        
        // 初始化位置编码
        if encoder.config.use_positional_encoding {
            encoder.initialize_positional_encoding();
        }
        
        // 初始化层权重
        encoder.initialize_layer_weights();
        
        encoder
    }
    
    /// 初始化位置编码
    fn initialize_positional_encoding(&mut self) {
        let mut pos_encoding = Vec::with_capacity(self.max_seq_length);
        
        for pos in 0..self.max_seq_length {
            let mut encoding = Vec::with_capacity(self.hidden_size);
            
            for i in 0..self.hidden_size {
                let angle = pos as f32 / (10000.0_f32).powf((2 * i) as f32 / self.hidden_size as f32);
                if i % 2 == 0 {
                    encoding.push(angle.sin());
                } else {
                    encoding.push(angle.cos());
                }
            }
            
            pos_encoding.push(encoding);
        }
        
        self.positional_encoding = Some(pos_encoding);
    }
    
    /// 初始化层权重
    fn initialize_layer_weights(&mut self) {
        // 初始化嵌入层权重
        let embedding_size = self.vocab_size * self.hidden_size;
        let mut embedding_weights = Vec::with_capacity(embedding_size);
        for _ in 0..embedding_size {
            embedding_weights.push((rand::random::<f32>() - 0.5) * 0.02);
        }
        self.layer_weights.insert("embedding".to_string(), embedding_weights);
        
        // 初始化每一层的权重
        for layer_idx in 0..self.num_layers {
            // 自注意力层权重
            let qkv_size = self.hidden_size * self.hidden_size * 3;
            let mut qkv_weights = Vec::with_capacity(qkv_size);
            for _ in 0..qkv_size {
                qkv_weights.push((rand::random::<f32>() - 0.5) * 0.02);
            }
            self.layer_weights.insert(
                format!("layer_{}_self_attn_qkv", layer_idx),
                qkv_weights
            );
            
            // 输出投影权重
            let output_size = self.hidden_size * self.hidden_size;
            let mut output_weights = Vec::with_capacity(output_size);
            for _ in 0..output_size {
                output_weights.push((rand::random::<f32>() - 0.5) * 0.02);
            }
            self.layer_weights.insert(
                format!("layer_{}_self_attn_output", layer_idx),
                output_weights
            );
            
            // 前馈网络权重
            let ffn_input_size = self.hidden_size * self.feed_forward_dim;
            let mut ffn_input_weights = Vec::with_capacity(ffn_input_size);
            for _ in 0..ffn_input_size {
                ffn_input_weights.push((rand::random::<f32>() - 0.5) * 0.02);
            }
            self.layer_weights.insert(
                format!("layer_{}_ffn_input", layer_idx),
                ffn_input_weights
            );
            
            let ffn_output_size = self.feed_forward_dim * self.hidden_size;
            let mut ffn_output_weights = Vec::with_capacity(ffn_output_size);
            for _ in 0..ffn_output_size {
                ffn_output_weights.push((rand::random::<f32>() - 0.5) * 0.02);
            }
            self.layer_weights.insert(
                format!("layer_{}_ffn_output", layer_idx),
                ffn_output_weights
            );
            
            // 层归一化权重
            let ln_size = self.hidden_size * 2; // weight + bias
            let mut ln_weights = Vec::with_capacity(ln_size);
            for i in 0..self.hidden_size {
                ln_weights.push(1.0); // weight
            }
            for _ in 0..self.hidden_size {
                ln_weights.push(0.0); // bias
            }
            self.layer_weights.insert(
                format!("layer_{}_ln1", layer_idx),
                ln_weights.clone()
            );
            self.layer_weights.insert(
                format!("layer_{}_ln2", layer_idx),
                ln_weights
            );
        }
        
        // 最终层归一化
        let final_ln_size = self.hidden_size * 2;
        let mut final_ln_weights = Vec::with_capacity(final_ln_size);
        for i in 0..self.hidden_size {
            final_ln_weights.push(1.0); // weight
        }
        for _ in 0..self.hidden_size {
            final_ln_weights.push(0.0); // bias
        }
        self.layer_weights.insert("final_ln".to_string(), final_ln_weights);
    }
    
    /// 编码文本
    pub fn encode(&self, text: &str) -> Result<Vec<f32>, TransformerError> {
        // 1. 分词
        let tokens = self.tokenizer.tokenize(text);
        if tokens.is_empty() {
            return Err(TransformerError::InputError("输入文本为空".to_string()));
        }
        
        // 2. 转换为token ID
        let mut token_ids = Vec::with_capacity(tokens.len());
        for token in &tokens {
            let id = self.tokenizer.vocabulary().get_id(token)
                .unwrap_or(self.tokenizer.vocabulary().get_unk_id());
            token_ids.push(id);
        }
        
        // 3. 限制序列长度
        if token_ids.len() > self.max_seq_length {
            token_ids.truncate(self.max_seq_length);
        }
        
        // 4. 获取token嵌入
        let embeddings = self.get_token_embeddings(&token_ids)?;
        
        // 5. 添加位置编码
        let encoded = if self.config.use_positional_encoding {
            self.add_positional_encoding(&embeddings)?
        } else {
            embeddings
        };
        
        // 6. 通过Transformer层
        let mut hidden_states = encoded;
        for layer_idx in 0..self.num_layers {
            hidden_states = self.transformer_layer(&hidden_states, layer_idx)?;
        }
        
        // 7. 最终层归一化
        let final_output = if self.config.use_layer_norm {
            self.layer_norm(&hidden_states, "final_ln")?
        } else {
            hidden_states
        };
        
        // 8. 池化得到最终表示
        let pooled_output = self.pool_output(&final_output)?;
        
        Ok(pooled_output)
    }
    
    /// 获取token嵌入
    fn get_token_embeddings(&self, token_ids: &[usize]) -> Result<Vec<f32>, TransformerError> {
        let embedding_weights = self.layer_weights.get("embedding")
            .ok_or_else(|| TransformerError::config_error("嵌入权重未找到"))?;
        
        let mut embeddings = Vec::with_capacity(token_ids.len() * self.hidden_size);
        
        for &token_id in token_ids {
            if token_id >= self.vocab_size {
                return Err(TransformerError::InputError(format!("无效的token ID: {}", token_id)));
            }
            
            let start_idx = token_id * self.hidden_size;
            let end_idx = start_idx + self.hidden_size;
            
            if end_idx > embedding_weights.len() {
                return Err(TransformerError::config_error("嵌入权重维度不匹配"));
            }
            
            embeddings.extend_from_slice(&embedding_weights[start_idx..end_idx]);
        }
        
        Ok(embeddings)
    }
    
    /// 添加位置编码
    fn add_positional_encoding(&self, embeddings: &[f32]) -> Result<Vec<f32>, TransformerError> {
        let pos_encoding = self.positional_encoding.as_ref()
            .ok_or_else(|| TransformerError::config_error("位置编码未初始化"))?;
        
        let seq_len = embeddings.len() / self.hidden_size;
        if seq_len > pos_encoding.len() {
            return Err(TransformerError::InputError("序列长度超过位置编码范围".to_string()));
        }
        
        let mut result = Vec::with_capacity(embeddings.len());
        
        for pos in 0..seq_len {
            let emb_start = pos * self.hidden_size;
            let emb_end = emb_start + self.hidden_size;
            
            for i in 0..self.hidden_size {
                let emb_val = embeddings[emb_start + i];
                let pos_val = pos_encoding[pos][i];
                result.push(emb_val + pos_val);
            }
        }
        
        Ok(result)
    }
    
    /// Transformer层
    fn transformer_layer(&self, input: &[f32], layer_idx: usize) -> Result<Vec<f32>, TransformerError> {
        let mut hidden_states = input.to_vec();
        
        // 自注意力层
        if self.config.use_attention {
            let attn_output = self.self_attention(&hidden_states, layer_idx)?;
            
            if self.config.use_residual {
                hidden_states = self.add_vectors(&hidden_states, &attn_output)?;
            } else {
                hidden_states = attn_output;
            }
            
            if self.config.use_layer_norm {
                hidden_states = self.layer_norm(&hidden_states, &format!("layer_{}_ln1", layer_idx))?;
            }
        }
        
        // 前馈网络
        let ffn_output = self.feed_forward_network(&hidden_states, layer_idx)?;
        
        if self.config.use_residual {
            hidden_states = self.add_vectors(&hidden_states, &ffn_output)?;
        } else {
            hidden_states = ffn_output;
        }
        
        if self.config.use_layer_norm {
            hidden_states = self.layer_norm(&hidden_states, &format!("layer_{}_ln2", layer_idx))?;
        }
        
        Ok(hidden_states)
    }
    
    /// 自注意力机制
    fn self_attention(&self, input: &[f32], layer_idx: usize) -> Result<Vec<f32>, TransformerError> {
        let seq_len = input.len() / self.hidden_size;
        let head_dim = self.hidden_size / self.num_heads;
        
        // 计算QKV
        let qkv_weights = self.layer_weights.get(&format!("layer_{}_self_attn_qkv", layer_idx))
            .ok_or_else(|| TransformerError::config_error("QKV权重未找到"))?;
        
        let qkv = self.linear_transform(input, qkv_weights, self.hidden_size * 3)?;
        
        // 重塑为多头注意力
        let mut attention_outputs = Vec::new();
        
        for head in 0..self.num_heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;
            
            // 提取当前头的Q、K、V
            let mut q = Vec::new();
            let mut k = Vec::new();
            let mut v = Vec::new();
            
            for pos in 0..seq_len {
                let pos_start = pos * self.hidden_size * 3;
                
                // Q
                let q_start = pos_start;
                q.extend_from_slice(&qkv[q_start + head_start..q_start + head_end]);
                
                // K
                let k_start = pos_start + self.hidden_size;
                k.extend_from_slice(&qkv[k_start + head_start..k_start + head_end]);
                
                // V
                let v_start = pos_start + self.hidden_size * 2;
                v.extend_from_slice(&qkv[v_start + head_start..v_start + head_end]);
            }
            
            // 计算注意力分数
            let attention_scores = self.compute_attention_scores(&q, &k, seq_len, head_dim)?;
            
            // 应用softmax
            let attention_probs = self.softmax(&attention_scores)?;
            
            // 计算注意力输出
            let head_output = self.compute_attention_output(&attention_probs, &v, seq_len, head_dim)?;
            
            attention_outputs.push(head_output);
        }
        
        // 合并多头输出
        let mut concatenated = Vec::new();
        for pos in 0..seq_len {
            for head in 0..self.num_heads {
                let head_start = pos * head_dim;
                let head_end = head_start + head_dim;
                concatenated.extend_from_slice(&attention_outputs[head][head_start..head_end]);
            }
        }
        
        // 输出投影
        let output_weights = self.layer_weights.get(&format!("layer_{}_self_attn_output", layer_idx))
            .ok_or_else(|| TransformerError::config_error("输出投影权重未找到"))?;
        
        self.linear_transform(&concatenated, output_weights, self.hidden_size)
    }
    
    /// 计算注意力分数
    fn compute_attention_scores(&self, q: &[f32], k: &[f32], seq_len: usize, head_dim: usize) -> Result<Vec<f32>, TransformerError> {
        let mut scores = Vec::with_capacity(seq_len * seq_len);
        
        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut score = 0.0;
                
                let q_start = i * head_dim;
                let k_start = j * head_dim;
                
                for d in 0..head_dim {
                    score += q[q_start + d] * k[k_start + d];
                }
                
                score /= (head_dim as f32).sqrt();
                scores.push(score);
            }
        }
        
        Ok(scores)
    }
    
    /// Softmax函数
    fn softmax(&self, logits: &[f32]) -> Result<Vec<f32>, TransformerError> {
        if logits.is_empty() {
            return Err(TransformerError::computation_error("输入为空"));
        }
        
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut exp_logits = Vec::with_capacity(logits.len());
        let mut sum_exp = 0.0;
        
        for &logit in logits {
            let exp_val = (logit - max_logit).exp();
            exp_logits.push(exp_val);
            sum_exp += exp_val;
        }
        
        if sum_exp == 0.0 {
            return Err(TransformerError::computation_error("Softmax分母为零"));
        }
        
        let mut probs = Vec::with_capacity(logits.len());
        for exp_val in exp_logits {
            probs.push(exp_val / sum_exp);
        }
        
        Ok(probs)
    }
    
    /// 计算注意力输出
    fn compute_attention_output(&self, attention_probs: &[f32], v: &[f32], seq_len: usize, head_dim: usize) -> Result<Vec<f32>, TransformerError> {
        let mut output = Vec::with_capacity(seq_len * head_dim);
        
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut weighted_sum = 0.0;
                
                for j in 0..seq_len {
                    let prob_idx = i * seq_len + j;
                    let v_idx = j * head_dim + d;
                    
                    weighted_sum += attention_probs[prob_idx] * v[v_idx];
                }
                
                output.push(weighted_sum);
            }
        }
        
        Ok(output)
    }
    
    /// 前馈网络
    fn feed_forward_network(&self, input: &[f32], layer_idx: usize) -> Result<Vec<f32>, TransformerError> {
        // 第一层线性变换
        let ffn_input_weights = self.layer_weights.get(&format!("layer_{}_ffn_input", layer_idx))
            .ok_or_else(|| TransformerError::config_error("前馈网络输入权重未找到"))?;
        
        let intermediate = self.linear_transform(input, ffn_input_weights, self.feed_forward_dim)?;
        
        // GELU激活函数
        let activated = self.gelu(&intermediate)?;
        
        // 第二层线性变换
        let ffn_output_weights = self.layer_weights.get(&format!("layer_{}_ffn_output", layer_idx))
            .ok_or_else(|| TransformerError::config_error("前馈网络输出权重未找到"))?;
        
        self.linear_transform(&activated, ffn_output_weights, self.hidden_size)
    }
    
    /// GELU激活函数
    fn gelu(&self, x: &[f32]) -> Result<Vec<f32>, TransformerError> {
        let mut result = Vec::with_capacity(x.len());
        
        for &val in x {
            let gelu_val = 0.5 * val * (1.0 + (val * 0.797885 + 0.044715 * val.powi(3)).tanh());
            result.push(gelu_val);
        }
        
        Ok(result)
    }
    
    /// 线性变换
    fn linear_transform(&self, input: &[f32], weights: &[f32], output_size: usize) -> Result<Vec<f32>, TransformerError> {
        let input_size = input.len() / output_size;
        
        if weights.len() != input_size * output_size {
            return Err(TransformerError::dimension_mismatch(
                format!("{}", input_size * output_size),
                format!("{}", weights.len())
            ));
        }
        
        let mut output = Vec::with_capacity(output_size);
        
        for i in 0..output_size {
            let mut sum = 0.0;
            
            for j in 0..input_size {
                let weight_idx = j * output_size + i;
                sum += input[j] * weights[weight_idx];
            }
            
            output.push(sum);
        }
        
        Ok(output)
    }
    
    /// 层归一化
    fn layer_norm(&self, input: &[f32], layer_name: &str) -> Result<Vec<f32>, TransformerError> {
        let ln_weights = self.layer_weights.get(layer_name)
            .ok_or_else(|| TransformerError::config_error(format!("层归一化权重未找到: {}", layer_name)))?;
        
        if ln_weights.len() != self.hidden_size * 2 {
            return Err(TransformerError::config_error("层归一化权重维度不匹配"));
        }
        
        let mut normalized = Vec::with_capacity(input.len());
        let seq_len = input.len() / self.hidden_size;
        
        for pos in 0..seq_len {
            let pos_start = pos * self.hidden_size;
            let pos_end = pos_start + self.hidden_size;
            let pos_input = &input[pos_start..pos_end];
            
            // 计算均值
            let mean = pos_input.iter().sum::<f32>() / self.hidden_size as f32;
            
            // 计算方差
            let variance = pos_input.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / self.hidden_size as f32;
            
            // 归一化
            let std_dev = (variance + 1e-12).sqrt();
            
            for (i, &val) in pos_input.iter().enumerate() {
                let normalized_val = (val - mean) / std_dev;
                let weight = ln_weights[i];
                let bias = ln_weights[i + self.hidden_size];
                normalized.push(normalized_val * weight + bias);
            }
        }
        
        Ok(normalized)
    }
    
    /// 向量加法
    fn add_vectors(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, TransformerError> {
        if a.len() != b.len() {
            return Err(TransformerError::dimension_mismatch(
                format!("{}", a.len()),
                format!("{}", b.len())
            ));
        }
        
        let mut result = Vec::with_capacity(a.len());
        for (i, &val_a) in a.iter().enumerate() {
            result.push(val_a + b[i]);
        }
        
        Ok(result)
    }
    
    /// 池化输出
    fn pool_output(&self, hidden_states: &[f32]) -> Result<Vec<f32>, TransformerError> {
        let seq_len = hidden_states.len() / self.hidden_size;
        
        if seq_len == 0 {
            return Err(TransformerError::InputError("序列长度为0".to_string()));
        }
        
        // 使用平均池化
        let mut pooled = Vec::with_capacity(self.hidden_size);
        
        for dim in 0..self.hidden_size {
            let mut sum = 0.0;
            
            for pos in 0..seq_len {
                let idx = pos * self.hidden_size + dim;
                sum += hidden_states[idx];
            }
            
            pooled.push(sum / seq_len as f32);
        }
        
        Ok(pooled)
    }
    
    /// 获取配置
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }
    
    /// 获取分词器
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::TransformerConfig;

    #[test]
    fn test_encoder_creation() {
        let config = TransformerConfig::default();
        let mut tokenizer = Tokenizer::new(config.clone());
        tokenizer.initialize().unwrap();
        
        let encoder_config = EncoderConfig::default();
        let encoder = Encoder::new(encoder_config, tokenizer, &config);
        
        assert_eq!(encoder.vocab_size, config.vocab_size);
        assert_eq!(encoder.hidden_size, config.hidden_size);
    }

    #[test]
    fn test_encoder_encode() {
        let config = TransformerConfig::default();
        let mut tokenizer = Tokenizer::new(config.clone());
        tokenizer.initialize().unwrap();
        
        let encoder_config = EncoderConfig::default();
        let encoder = Encoder::new(encoder_config, tokenizer, &config);
        
        let result = encoder.encode("Hello world");
        assert!(result.is_ok());
        
        let features = result.unwrap();
        assert_eq!(features.len(), config.hidden_size);
    }

    #[test]
    fn test_gelu_activation() {
        let config = TransformerConfig::default();
        let mut tokenizer = Tokenizer::new(config.clone());
        tokenizer.initialize().unwrap();
        
        let encoder_config = EncoderConfig::default();
        let encoder = Encoder::new(encoder_config, tokenizer, &config);
        
        let input = vec![1.0, -1.0, 0.0];
        let result = encoder.gelu(&input).unwrap();
        
        assert_eq!(result.len(), input.len());
        assert!(result[0] > 0.0); // GELU(1.0) > 0
        assert!(result[1] < 0.0); // GELU(-1.0) < 0
        assert_eq!(result[2], 0.0); // GELU(0.0) = 0
    }
} 

/// 句子编码器
/// 专门用于将句子编码为固定维度的向量表示
#[derive(Debug)]
pub struct SentenceEncoder {
    /// 基础编码器
    encoder: Encoder,
    /// 句子池化方法
    pooling_method: PoolingMethod,
    /// 输出维度
    output_dimension: usize,
    /// 是否归一化输出
    normalize_output: bool,
    /// 句子最大长度
    max_sentence_length: usize,
    /// 词汇表
    vocabulary: HashMap<String, usize>,
    /// 句子缓存
    sentence_cache: HashMap<String, Vec<f32>>,
    /// 缓存大小限制
    cache_size_limit: usize,
}

/// 池化方法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingMethod {
    /// 平均池化
    Mean,
    /// 最大池化
    Max,
    /// 注意力池化
    Attention,
    /// CLS token池化
    CLS,
    /// 加权平均池化
    WeightedMean,
}

impl Default for PoolingMethod {
    fn default() -> Self {
        PoolingMethod::Mean
    }
}

impl SentenceEncoder {
    /// 创建新的句子编码器
    pub fn new(
        encoder: Encoder,
        pooling_method: PoolingMethod,
        output_dimension: usize,
        normalize_output: bool,
        max_sentence_length: usize,
    ) -> Self {
        Self {
            encoder,
            pooling_method,
            output_dimension,
            normalize_output,
            max_sentence_length,
            vocabulary: HashMap::new(),
            sentence_cache: HashMap::new(),
            cache_size_limit: 1000,
        }
    }

    /// 从配置创建句子编码器
    pub fn from_config(
        encoder_config: EncoderConfig,
        model_config: &super::config::TransformerConfig,
        pooling_method: PoolingMethod,
        output_dimension: usize,
    ) -> Result<Self, super::error::TransformerError> {
        let tokenizer = super::tokenizer::Tokenizer::new(model_config.clone());
        let encoder = Encoder::new(encoder_config, tokenizer, model_config);
        
        Ok(Self::new(
            encoder,
            pooling_method,
            output_dimension,
            true, // 默认归一化输出
            model_config.max_seq_length,
        ))
    }

    /// 编码句子
    pub fn encode_sentence(&mut self, sentence: &str) -> Result<Vec<f32>, super::error::TransformerError> {
        // 检查缓存
        if let Some(cached) = self.sentence_cache.get(sentence) {
            return Ok(cached.clone());
        }

        // 预处理句子
        let processed_sentence = self.preprocess_sentence(sentence);
        
        // 使用基础编码器编码
        let token_embeddings = self.encoder.encode(&processed_sentence)?;
        
        // 应用池化方法
        let pooled_embedding = self.apply_pooling(&token_embeddings)?;
        
        // 调整维度
        let resized_embedding = self.resize_embedding(&pooled_embedding)?;
        
        // 归一化（如果需要）
        let final_embedding = if self.normalize_output {
            self.normalize_embedding(&resized_embedding)?
        } else {
            resized_embedding
        };

        // 缓存结果
        self.cache_embedding(sentence, &final_embedding);

        Ok(final_embedding)
    }

    /// 批量编码句子
    pub fn encode_sentences(&mut self, sentences: &[String]) -> Result<Vec<Vec<f32>>, super::error::TransformerError> {
        let mut embeddings = Vec::with_capacity(sentences.len());
        
        for sentence in sentences {
            let embedding = self.encode_sentence(sentence)?;
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }

    /// 计算句子相似度
    pub fn compute_similarity(&mut self, sentence1: &str, sentence2: &str) -> Result<f32, super::error::TransformerError> {
        let embedding1 = self.encode_sentence(sentence1)?;
        let embedding2 = self.encode_sentence(sentence2)?;
        
        self.cosine_similarity(&embedding1, &embedding2)
    }

    /// 预处理句子
    fn preprocess_sentence(&self, sentence: &str) -> String {
        let mut processed = sentence.to_string();
        
        // 转换为小写
        processed = processed.to_lowercase();
        
        // 去除多余空格
        processed = processed.split_whitespace().collect::<Vec<_>>().join(" ");
        
        // 截断到最大长度
        if processed.len() > self.max_sentence_length {
            processed.truncate(self.max_sentence_length);
        }
        
        processed
    }

    /// 应用池化方法
    fn apply_pooling(&self, embeddings: &[f32]) -> Result<Vec<f32>, super::error::TransformerError> {
        let hidden_size = self.encoder.hidden_size;
        let seq_len = embeddings.len() / hidden_size;
        
        if seq_len == 0 {
            return Err(super::error::TransformerError::InputError("序列长度为0".to_string()));
        }

        match self.pooling_method {
            PoolingMethod::Mean => self.mean_pooling(embeddings, seq_len, hidden_size),
            PoolingMethod::Max => self.max_pooling(embeddings, seq_len, hidden_size),
            PoolingMethod::Attention => self.attention_pooling(embeddings, seq_len, hidden_size),
            PoolingMethod::CLS => self.cls_pooling(embeddings, hidden_size),
            PoolingMethod::WeightedMean => self.weighted_mean_pooling(embeddings, seq_len, hidden_size),
        }
    }

    /// 平均池化
    fn mean_pooling(&self, embeddings: &[f32], seq_len: usize, hidden_size: usize) -> Result<Vec<f32>, super::error::TransformerError> {
        let mut pooled = Vec::with_capacity(hidden_size);
        
        for dim in 0..hidden_size {
            let mut sum = 0.0;
            
            for pos in 0..seq_len {
                let idx = pos * hidden_size + dim;
                sum += embeddings[idx];
            }
            
            pooled.push(sum / seq_len as f32);
        }
        
        Ok(pooled)
    }

    /// 最大池化
    fn max_pooling(&self, embeddings: &[f32], seq_len: usize, hidden_size: usize) -> Result<Vec<f32>, super::error::TransformerError> {
        let mut pooled = Vec::with_capacity(hidden_size);
        
        for dim in 0..hidden_size {
            let mut max_val = f32::NEG_INFINITY;
            
            for pos in 0..seq_len {
                let idx = pos * hidden_size + dim;
                max_val = max_val.max(embeddings[idx]);
            }
            
            pooled.push(max_val);
        }
        
        Ok(pooled)
    }

    /// 注意力池化
    fn attention_pooling(&self, embeddings: &[f32], seq_len: usize, hidden_size: usize) -> Result<Vec<f32>, super::error::TransformerError> {
        // 简单的注意力机制：使用第一个token作为query
        let query = &embeddings[..hidden_size];
        let mut attention_weights = Vec::with_capacity(seq_len);
        
        for pos in 0..seq_len {
            let key = &embeddings[pos * hidden_size..(pos + 1) * hidden_size];
            let score = self.dot_product(query, key)?;
            attention_weights.push(score);
        }
        
        // 应用softmax
        let attention_weights = self.softmax(&attention_weights)?;
        
        // 加权平均
        let mut pooled = vec![0.0; hidden_size];
        for pos in 0..seq_len {
            let weight = attention_weights[pos];
            let token_embedding = &embeddings[pos * hidden_size..(pos + 1) * hidden_size];
            
            for (i, &val) in token_embedding.iter().enumerate() {
                pooled[i] += weight * val;
            }
        }
        
        Ok(pooled)
    }

    /// CLS池化
    fn cls_pooling(&self, embeddings: &[f32], hidden_size: usize) -> Result<Vec<f32>, super::error::TransformerError> {
        // 使用第一个token的表示
        if embeddings.len() < hidden_size {
            return Err(super::error::TransformerError::InputError("嵌入维度不足".to_string()));
        }
        
        Ok(embeddings[..hidden_size].to_vec())
    }

    /// 加权平均池化
    fn weighted_mean_pooling(&self, embeddings: &[f32], seq_len: usize, hidden_size: usize) -> Result<Vec<f32>, super::error::TransformerError> {
        // 使用位置权重
        let mut weights = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            weights.push(1.0 / (pos + 1) as f32); // 递减权重
        }
        
        // 归一化权重
        let sum_weights: f32 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= sum_weights;
        }
        
        // 加权平均
        let mut pooled = vec![0.0; hidden_size];
        for pos in 0..seq_len {
            let weight = weights[pos];
            let token_embedding = &embeddings[pos * hidden_size..(pos + 1) * hidden_size];
            
            for (i, &val) in token_embedding.iter().enumerate() {
                pooled[i] += weight * val;
            }
        }
        
        Ok(pooled)
    }

    /// 调整嵌入维度
    fn resize_embedding(&self, embedding: &[f32]) -> Result<Vec<f32>, super::error::TransformerError> {
        if embedding.len() == self.output_dimension {
            return Ok(embedding.to_vec());
        }
        
        if embedding.len() > self.output_dimension {
            // 截断
            Ok(embedding[..self.output_dimension].to_vec())
        } else {
            // 填充
            let mut resized = embedding.to_vec();
            resized.resize(self.output_dimension, 0.0);
            Ok(resized)
        }
    }

    /// 归一化嵌入
    fn normalize_embedding(&self, embedding: &[f32]) -> Result<Vec<f32>, super::error::TransformerError> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm < 1e-8 {
            return Ok(embedding.to_vec());
        }
        
        Ok(embedding.iter().map(|&x| x / norm).collect())
    }

    /// 点积计算
    fn dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32, super::error::TransformerError> {
        if a.len() != b.len() {
            return Err(super::error::TransformerError::InputError("向量维度不匹配".to_string()));
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
    }

    /// Softmax函数
    fn softmax(&self, logits: &[f32]) -> Result<Vec<f32>, super::error::TransformerError> {
        if logits.is_empty() {
            return Ok(Vec::new());
        }
        
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp = exp_logits.iter().sum::<f32>();
        
        if sum_exp < 1e-8 {
            return Err(super::error::TransformerError::InputError("Softmax计算错误".to_string()));
        }
        
        Ok(exp_logits.iter().map(|&x| x / sum_exp).collect())
    }

    /// 余弦相似度
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32, super::error::TransformerError> {
        if a.len() != b.len() {
            return Err(super::error::TransformerError::InputError("向量维度不匹配".to_string()));
        }
        
        let dot_product = self.dot_product(a, b)?;
        let norm_a = (a.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let norm_b = (b.iter().map(|x| x * x).sum::<f32>()).sqrt();
        
        if norm_a < 1e-8 || norm_b < 1e-8 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm_a * norm_b))
    }

    /// 缓存嵌入
    fn cache_embedding(&mut self, sentence: &str, embedding: &[f32]) {
        // 如果缓存已满，清除最旧的条目
        if self.sentence_cache.len() >= self.cache_size_limit {
            let oldest_key = self.sentence_cache.keys().next().cloned();
            if let Some(key) = oldest_key {
                self.sentence_cache.remove(&key);
            }
        }
        
        self.sentence_cache.insert(sentence.to_string(), embedding.to_vec());
    }

    /// 获取输出维度
    pub fn output_dimension(&self) -> usize {
        self.output_dimension
    }

    /// 获取池化方法
    pub fn pooling_method(&self) -> PoolingMethod {
        self.pooling_method
    }

    /// 设置池化方法
    pub fn set_pooling_method(&mut self, method: PoolingMethod) {
        self.pooling_method = method;
    }

    /// 设置输出维度
    pub fn set_output_dimension(&mut self, dimension: usize) {
        self.output_dimension = dimension;
    }

    /// 设置是否归一化输出
    pub fn set_normalize_output(&mut self, normalize: bool) {
        self.normalize_output = normalize;
    }

    /// 清除缓存
    pub fn clear_cache(&mut self) {
        self.sentence_cache.clear();
    }

    /// 获取缓存大小
    pub fn cache_size(&self) -> usize {
        self.sentence_cache.len()
    }
}

impl Default for SentenceEncoder {
    fn default() -> Self {
        let encoder_config = EncoderConfig::default();
        let model_config = super::config::TransformerConfig::default();
        let encoder = Encoder::new(encoder_config, super::tokenizer::Tokenizer::new(model_config.clone()), &model_config);
        
        Self::new(
            encoder,
            PoolingMethod::Mean,
            768, // 默认输出维度
            true, // 默认归一化
            512,  // 默认最大句子长度
        )
    }
} 