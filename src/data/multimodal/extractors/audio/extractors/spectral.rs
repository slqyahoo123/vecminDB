//! 频谱特征提取器实现
//!
//! 这个模块包含各种频谱相关的特征提取器，如色度特征、谱质心、
//! 谱衰减、零交叉率和能量特征等。这些特征常用于音乐分析和音频分类。

#[cfg(feature = "multimodal")]
use ndarray::{Array2, Axis};
#[cfg(feature = "multimodal")]
use crate::{Result, Error};
#[cfg(feature = "multimodal")]
use super::{
    apply_fft, 
    compute_power_spectrum, 
    apply_hanning_window, 
    frame_signal
};

/// 色度特征提取器
pub struct ChromaExtractor {
    /// 色度滤波器数量 (通常是12，对应12个半音)
    n_chroma: usize,
    /// FFT窗口大小
    n_fft: usize,
    /// 帧长度
    frame_size: usize,
    /// 帧移步长
    hop_size: usize,
    /// 采样率
    sample_rate: u32,
    /// 是否归一化
    normalize: bool,
    /// 预加重因子
    preemphasis: f32,
}

impl Default for ChromaExtractor {
    fn default() -> Self {
        Self {
            n_chroma: 12,
            n_fft: 2048,
            frame_size: 512,
            hop_size: 256,
            sample_rate: 44100,
            normalize: true,
            preemphasis: 0.97,
        }
    }
}

impl ChromaExtractor {
    /// 创建新的色度特征提取器
    pub fn new(
        n_chroma: usize,
        n_fft: usize,
        frame_size: usize,
        hop_size: usize,
        sample_rate: u32,
        normalize: bool,
        preemphasis: f32,
    ) -> Self {
        Self {
            n_chroma,
            n_fft,
            frame_size,
            hop_size,
            sample_rate,
            normalize,
            preemphasis,
        }
    }
    
    /// 应用预加重滤波
    fn apply_preemphasis(&self, signal: &[f32]) -> Vec<f32> {
        if self.preemphasis <= 0.0 {
            return signal.to_vec();
        }
        
        let mut output = Vec::with_capacity(signal.len());
        output.push(signal[0]);
        
        for i in 1..signal.len() {
            output.push(signal[i] - self.preemphasis * signal[i-1]);
        }
        
        output
    }
    
    /// 创建色度滤波器组
    #[cfg(feature = "multimodal")]
    fn create_chroma_filterbank(&self) -> Array2<f32> {
        let n_bins = self.n_fft / 2 + 1;
        let mut filterbank = Array2::zeros((self.n_chroma, n_bins));
        
        // 计算频率到色度映射
        // 参考频率为A4 = 440 Hz
        let a440 = 440.0;
        let c0 = a440 * 2.0f32.powf(-4.75); // C0是A4以下的9个半音，频率比例为2^(-4.75)
        
        for i in 0..n_bins {
            // 获取当前bin的频率
            let freq = i as f32 * self.sample_rate as f32 / self.n_fft as f32;
            
            // 跳过非常低的频率
            if freq < 20.0 {
                continue;
            }
            
            // 计算相对于C0的音高
            let mut pitch = (12.0 * (freq / c0).log2()).round() as i32;
            
            // 确保pitch是非负的
            while pitch < 0 {
                pitch += 12;
            }
            
            // 映射到0-11的色度范围
            let chroma = (pitch % 12) as usize;
            
            // 更新滤波器组
            filterbank[[chroma, i]] += 1.0;
        }
        
        // 归一化每个色度滤波器
        for i in 0..self.n_chroma {
            let row_sum = filterbank.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_bins {
                    filterbank[[i, j]] /= row_sum;
                }
            }
        }
        
        filterbank
    }
    
    /// 提取色度特征
    #[cfg(feature = "multimodal")]
    pub fn extract(&self, samples: &[f32]) -> Result<Array2<f32>> {
        // 1. 预加重
        let preemphasized = self.apply_preemphasis(samples);
        
        // 2. 分帧
        let frames = frame_signal(&preemphasized, self.frame_size, self.hop_size);
        if frames.is_empty() {
            return Err(Error::data(
                "音频太短，无法提取足够的帧".to_string()
            ));
        }
        
        // 3. 加窗
        let windowed_frames: Vec<Vec<f32>> = frames.iter()
            .map(|frame| apply_hanning_window(frame))
            .collect();
        
        // 4. 对每一帧应用短时傅里叶变换(STFT)
        let stft_frames: Vec<Vec<f32>> = windowed_frames.iter()
            .map(|frame| {
                let fft_result = apply_fft(frame, self.n_fft);
                let power_spectrum = compute_power_spectrum(&fft_result);
                power_spectrum[..self.n_fft/2 + 1].to_vec()
            })
            .collect();
        
        // 5. 获取色度滤波器组
        let chroma_filterbank = self.create_chroma_filterbank();
        
        // 6. 将功率谱应用到色度滤波器组
        let mut chroma_features = Array2::zeros((frames.len(), self.n_chroma));
        
        for (i, spectrum) in stft_frames.iter().enumerate() {
            for j in 0..self.n_chroma {
                let mut energy = 0.0;
                for k in 0..spectrum.len() {
                    energy += spectrum[k] * chroma_filterbank[[j, k]];
                }
                chroma_features[[i, j]] = energy;
            }
        }
        
        // 7. 如果需要，进行归一化
        if self.normalize {
            for mut row in chroma_features.axis_iter_mut(Axis(0)) {
                let sum = row.sum();
                if sum > 1e-10 {
                    for val in row.iter_mut() {
                        *val /= sum;
                    }
                }
            }
        }
        
        Ok(chroma_features)
    }
}

/// 谱质心特征提取器
pub struct SpectralCentroidExtractor {
    /// FFT窗口大小
    n_fft: usize,
    /// 帧长度
    frame_size: usize,
    /// 帧移步长
    hop_size: usize,
    /// 采样率
    sample_rate: u32,
    /// 是否归一化
    normalize: bool,
    /// 预加重因子
    preemphasis: f32,
}

impl Default for SpectralCentroidExtractor {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            frame_size: 512,
            hop_size: 256,
            sample_rate: 44100,
            normalize: true,
            preemphasis: 0.97,
        }
    }
}

impl SpectralCentroidExtractor {
    /// 创建新的谱质心特征提取器
    pub fn new(
        n_fft: usize,
        frame_size: usize,
        hop_size: usize,
        sample_rate: u32,
        normalize: bool,
        preemphasis: f32,
    ) -> Self {
        Self {
            n_fft,
            frame_size,
            hop_size,
            sample_rate,
            normalize,
            preemphasis,
        }
    }
    
    /// 应用预加重滤波
    fn apply_preemphasis(&self, signal: &[f32]) -> Vec<f32> {
        if self.preemphasis <= 0.0 {
            return signal.to_vec();
        }
        
        let mut output = Vec::with_capacity(signal.len());
        output.push(signal[0]);
        
        for i in 1..signal.len() {
            output.push(signal[i] - self.preemphasis * signal[i-1]);
        }
        
        output
    }
    
    /// 提取谱质心特征
    #[cfg(feature = "multimodal")]
    pub fn extract(&self, samples: &[f32]) -> Result<Array2<f32>> {
        // 1. 预加重
        let preemphasized = self.apply_preemphasis(samples);
        
        // 2. 分帧
        let frames = frame_signal(&preemphasized, self.frame_size, self.hop_size);
        if frames.is_empty() {
            return Err(Error::data(
                "音频太短，无法提取足够的帧".to_string()
            ));
        }
        
        // 3. 加窗
        let windowed_frames: Vec<Vec<f32>> = frames.iter()
            .map(|frame| apply_hanning_window(frame))
            .collect();
        
        // 4. 对每一帧应用短时傅里叶变换(STFT)
        let stft_frames: Vec<Vec<f32>> = windowed_frames.iter()
            .map(|frame| {
                let fft_result = apply_fft(frame, self.n_fft);
                let power_spectrum = compute_power_spectrum(&fft_result);
                power_spectrum[..self.n_fft/2 + 1].to_vec()
            })
            .collect();
        
        // 5. 计算谱质心
        let mut centroids = Vec::with_capacity(frames.len());
        
        for spectrum in &stft_frames {
            let mut weighted_sum = 0.0;
            let mut sum = 0.0;
            
            for (i, &power) in spectrum.iter().enumerate() {
                let freq = i as f32 * self.sample_rate as f32 / self.n_fft as f32;
                weighted_sum += freq * power;
                sum += power;
            }
            
            // 避免除以0
            let centroid = if sum > 1e-10 { weighted_sum / sum } else { 0.0 };
            centroids.push(centroid);
        }
        
        // 6. 将结果转换为2D数组
        let mut result = Array2::zeros((frames.len(), 1));
        for (i, &centroid) in centroids.iter().enumerate() {
            result[[i, 0]] = centroid;
        }
        
        // 7. 如果需要，进行归一化
        if self.normalize {
            // 归一化到0-1范围
            let nyquist = self.sample_rate as f32 / 2.0;
            result.mapv_inplace(|x| x / nyquist);
        }
        
        Ok(result)
    }
}

/// 谱衰减特征提取器
pub struct SpectralRolloffExtractor {
    /// FFT窗口大小
    n_fft: usize,
    /// 帧长度
    frame_size: usize,
    /// 帧移步长
    hop_size: usize,
    /// 采样率
    sample_rate: u32,
    /// 是否归一化
    normalize: bool,
    /// 预加重因子
    preemphasis: f32,
    /// 衰减百分比（默认0.85，即85%的能量）
    roll_percent: f32,
}

impl Default for SpectralRolloffExtractor {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            frame_size: 512,
            hop_size: 256,
            sample_rate: 44100,
            normalize: true,
            preemphasis: 0.97,
            roll_percent: 0.85,
        }
    }
}

impl SpectralRolloffExtractor {
    /// 创建新的谱衰减特征提取器
    pub fn new(
        n_fft: usize,
        frame_size: usize,
        hop_size: usize,
        sample_rate: u32,
        normalize: bool,
        preemphasis: f32,
        roll_percent: f32,
    ) -> Self {
        Self {
            n_fft,
            frame_size,
            hop_size,
            sample_rate,
            normalize,
            preemphasis,
            roll_percent: roll_percent.max(0.0).min(1.0), // 确保在0-1范围内
        }
    }
    
    /// 应用预加重滤波
    fn apply_preemphasis(&self, signal: &[f32]) -> Vec<f32> {
        if self.preemphasis <= 0.0 {
            return signal.to_vec();
        }
        
        let mut output = Vec::with_capacity(signal.len());
        output.push(signal[0]);
        
        for i in 1..signal.len() {
            output.push(signal[i] - self.preemphasis * signal[i-1]);
        }
        
        output
    }
    
    /// 提取谱衰减特征
    #[cfg(feature = "multimodal")]
    pub fn extract(&self, samples: &[f32]) -> Result<Array2<f32>> {
        // 1. 预加重
        let preemphasized = self.apply_preemphasis(samples);
        
        // 2. 分帧
        let frames = frame_signal(&preemphasized, self.frame_size, self.hop_size);
        if frames.is_empty() {
            return Err(Error::data(
                "音频太短，无法提取足够的帧".to_string()
            ));
        }
        
        // 3. 加窗
        let windowed_frames: Vec<Vec<f32>> = frames.iter()
            .map(|frame| apply_hanning_window(frame))
            .collect();
        
        // 4. 对每一帧应用短时傅里叶变换(STFT)
        let stft_frames: Vec<Vec<f32>> = windowed_frames.iter()
            .map(|frame| {
                let fft_result = apply_fft(frame, self.n_fft);
                let power_spectrum = compute_power_spectrum(&fft_result);
                power_spectrum[..self.n_fft/2 + 1].to_vec()
            })
            .collect();
        
        // 5. 计算谱衰减
        let mut rolloffs = Vec::with_capacity(frames.len());
        
        for spectrum in &stft_frames {
            // 计算总能量
            let total_energy: f32 = spectrum.iter().sum();
            let target_energy = total_energy * self.roll_percent;
            
            // 查找衰减点
            let mut cumulative_energy = 0.0;
            let mut rolloff_bin = 0;
            
            for (i, &energy) in spectrum.iter().enumerate() {
                cumulative_energy += energy;
                if cumulative_energy >= target_energy {
                    rolloff_bin = i;
                    break;
                }
            }
            
            // 转换为频率
            let rolloff_freq = rolloff_bin as f32 * self.sample_rate as f32 / self.n_fft as f32;
            rolloffs.push(rolloff_freq);
        }
        
        // 6. 将结果转换为2D数组
        let mut result = Array2::zeros((frames.len(), 1));
        for (i, &rolloff) in rolloffs.iter().enumerate() {
            result[[i, 0]] = rolloff;
        }
        
        // 7. 如果需要，进行归一化
        if self.normalize {
            // 归一化到0-1范围
            let nyquist = self.sample_rate as f32 / 2.0;
            result.mapv_inplace(|x| x / nyquist);
        }
        
        Ok(result)
    }
}

/// 零交叉率特征提取器
pub struct ZeroCrossingRateExtractor {
    /// 帧长度
    frame_size: usize,
    /// 帧移步长
    hop_size: usize,
    /// 采样率
    sample_rate: u32,
    /// 是否归一化
    normalize: bool,
}

impl Default for ZeroCrossingRateExtractor {
    fn default() -> Self {
        Self {
            frame_size: 512,
            hop_size: 256,
            sample_rate: 44100,
            normalize: true,
        }
    }
}

impl ZeroCrossingRateExtractor {
    /// 创建新的零交叉率特征提取器
    pub fn new(
        frame_size: usize,
        hop_size: usize,
        sample_rate: u32,
        normalize: bool,
    ) -> Self {
        Self {
            frame_size,
            hop_size,
            sample_rate,
            normalize,
        }
    }
    
    /// 提取零交叉率特征
    #[cfg(feature = "multimodal")]
    pub fn extract(&self, samples: &[f32]) -> Result<Array2<f32>> {
        // 1. 分帧
        let frames = frame_signal(samples, self.frame_size, self.hop_size);
        if frames.is_empty() {
            return Err(Error::data(
                "音频太短，无法提取足够的帧".to_string()
            ));
        }
        
        // 2. 计算每帧的零交叉率
        let mut zcrs = Vec::with_capacity(frames.len());
        
        for frame in &frames {
            let mut crossings = 0;
            
            for i in 1..frame.len() {
                // 检测是否发生过零点
                if (frame[i] >= 0.0 && frame[i-1] < 0.0) || 
                   (frame[i] < 0.0 && frame[i-1] >= 0.0) {
                    crossings += 1;
                }
            }
            
            // 计算ZCR（零交叉率）
            let zcr = crossings as f32 / (frame.len() as f32 - 1.0);
            zcrs.push(zcr);
        }
        
        // 3. 将结果转换为2D数组
        let mut result = Array2::zeros((frames.len(), 1));
        for (i, &zcr) in zcrs.iter().enumerate() {
            result[[i, 0]] = zcr;
        }
        
        // 4. 如果需要，进行归一化
        if self.normalize {
            // 零交叉率已经是一个比率，通常在0到1之间
            // 但是可以进一步标准化为Z分数
            let mean = result.mean().unwrap_or(0.0);
            let std_dev = result.std(0.0);
            
            if std_dev > 1e-10 {
                result.mapv_inplace(|x| (x - mean) / std_dev);
            }
        }
        
        Ok(result)
    }
}

/// 能量特征提取器
pub struct EnergyExtractor {
    /// 帧长度
    frame_size: usize,
    /// 帧移步长
    hop_size: usize,
    /// 是否归一化
    normalize: bool,
    /// 是否使用分贝尺度
    use_db: bool,
    /// 最小分贝值（用于对数变换）
    min_db: f32,
}

impl Default for EnergyExtractor {
    fn default() -> Self {
        Self {
            frame_size: 512,
            hop_size: 256,
            normalize: true,
            use_db: true,
            min_db: -80.0,
        }
    }
}

impl EnergyExtractor {
    /// 创建新的能量特征提取器
    pub fn new(
        frame_size: usize,
        hop_size: usize,
        normalize: bool,
        use_db: bool,
        min_db: f32,
    ) -> Self {
        Self {
            frame_size,
            hop_size,
            normalize,
            use_db,
            min_db,
        }
    }
    
    /// 提取能量特征
    #[cfg(feature = "multimodal")]
    pub fn extract(&self, samples: &[f32]) -> Result<Array2<f32>> {
        // 1. 分帧
        let frames = frame_signal(samples, self.frame_size, self.hop_size);
        if frames.is_empty() {
            return Err(Error::data(
                "音频太短，无法提取足够的帧".to_string()
            ));
        }
        
        // 2. 计算每帧的能量
        let mut energies = Vec::with_capacity(frames.len());
        
        for frame in &frames {
            // 计算能量作为平方和
            let energy = frame.iter().map(|&s| s * s).sum::<f32>();
            
            if self.use_db {
                // 转换为分贝尺度
                let db = if energy > 1e-10 {
                    10.0 * energy.log10()
                } else {
                    self.min_db
                };
                energies.push(db.max(self.min_db));
            } else {
                energies.push(energy);
            }
        }
        
        // 3. 将结果转换为2D数组
        let mut result = Array2::zeros((frames.len(), 1));
        for (i, &energy) in energies.iter().enumerate() {
            result[[i, 0]] = energy;
        }
        
        // 4. 如果需要，进行归一化
        if self.normalize {
            if self.use_db {
                // 分贝尺度归一化：映射到0-1
                let ref_db = if self.min_db < 0.0 { -self.min_db } else { 0.0 };
                result.mapv_inplace(|x| (x + ref_db) / ref_db);
            } else {
                // 线性尺度归一化：映射到0-1
                let max_val = result.fold(0.0f32, |m, &x| m.max(x));
                if max_val > 1e-10 {
                    result.mapv_inplace(|x| x / max_val);
                }
            }
        }
        
        Ok(result)
    }
} 