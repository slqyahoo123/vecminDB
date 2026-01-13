//! 音频特征提取器实现
//!
//! 提供音频特征提取器的核心实现，包含从不同音频源提取特征的功能。

use std::collections::HashMap;
use log::{debug, error};
use ndarray::{Array1, Array2, Axis};
#[cfg(feature = "multimodal")]
use reqwest::blocking::Client;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

use crate::data::feature::FeatureVector;
use crate::Error;
use crate::Result;
use crate::compat::tensor::TensorData;
use crate::core::types::CoreTensorData;
use crate::data::multimodal::ModalityType;
use crate::data::multimodal::extractors::interface::{ModalityExtractor, FeatureExtractor};
use super::AudioSource;
use super::AudioProcessingConfig;
use super::AudioFeatureType;
use super::extractors::mfcc::MFCCExtractor;
use super::extractors::mel::MelSpectrogramExtractor;

// 添加 symphonia 导入
#[cfg(feature = "multimodal")]
use symphonia::core::io::MediaSourceStream;
#[cfg(feature = "multimodal")]
use symphonia::core::codecs::DecoderOptions;
#[cfg(feature = "multimodal")]
use symphonia::core::formats::FormatOptions;
#[cfg(feature = "multimodal")]
use symphonia::core::meta::MetadataOptions;
#[cfg(feature = "multimodal")]
use symphonia::core::probe::Hint;

// 添加缺失的导入
use crate::data::multimodal::extractors::audio::extractors::{
    ChromaExtractor, SpectralCentroidExtractor, SpectralRolloffExtractor,
    ZeroCrossingRateExtractor, EnergyExtractor, TemporalFeaturesExtractor
};
use crate::data::multimodal::extractors::audio::extractors::temporal::TemporalFeatureType;

    /// 音频特征提取器
    #[derive(Debug)]
    pub struct AudioFeatureExtractor {
        /// 音频处理配置
        config: AudioProcessingConfig,
        /// HTTP客户端用于URL请求（仅在启用 `multimodal` 特性时可用）
        #[cfg(feature = "multimodal")]
        http_client: Option<Client>,
        /// 在未启用 `multimodal` 特性时占位以保持结构一致，相关功能会返回特性未启用的错误
        #[cfg(not(feature = "multimodal"))]
        http_client: Option<()>,
    }

impl AudioFeatureExtractor {
    /// 创建新的音频特征提取器
    pub fn new(config: AudioProcessingConfig) -> Self {
        #[cfg(feature = "multimodal")]
        let http_client = match Client::builder()
            .timeout(Duration::from_secs(30))
            .build() {
                Ok(client) => Some(client),
                Err(e) => {
                    warn!("创建HTTP客户端失败: {}，将不能处理URL音频", e);
                    None
                }
            };
        #[cfg(not(feature = "multimodal"))]
        let http_client = None;
            
        Self {
            config,
            http_client,
        }
    }
    
    /// 从音频源提取特征
    pub fn extract_from_source(&self, source: &AudioSource) -> Result<FeatureVector> {
        match source {
            AudioSource::Base64(data) => {
                let decoded = BASE64.decode(data.as_bytes())
                    .map_err(|e| Error::data(format!("Base64解码失败: {}", e)))?;
                    
                let (samples, sample_rate) = self.decode_audio(&decoded)?;
                self.process_audio(&samples, sample_rate)
            },
            
            AudioSource::URL(url) => {
                #[cfg(feature = "multimodal")]
                {
                    let client = self.http_client.as_ref()
                        .ok_or_else(|| Error::data("HTTP客户端未初始化".to_string()))?;
                    
                    let response = client.get(url)
                        .send()
                        .map_err(|e| Error::data(format!("HTTP请求失败: {}", e)))?;
                
                    let bytes = response.bytes()
                        .map_err(|e| Error::data(format!("读取响应数据失败: {}", e)))?;
                    
                    let (samples, sample_rate) = self.decode_audio(&bytes)?;
                    self.process_audio(&samples, sample_rate)
                }
                #[cfg(not(feature = "multimodal"))]
                {
                    Err(Error::feature_not_enabled("multimodal"))
                }
            },
            
            AudioSource::File(path) => {
                if !path.exists() {
                    return Err(Error::data("音频文件不存在".to_string()));
                }
                
                let bytes = std::fs::read(path)
                    .map_err(|e| Error::data(format!("读取音频文件失败: {}", e)))?;
                    
                let (samples, sample_rate) = self.decode_audio(&bytes)?;
                self.process_audio(&samples, sample_rate)
            },
            
            AudioSource::RawSamples { samples, sample_rate, channels } => {
                let samples = if *channels > 1 {
                    self.convert_to_mono(samples, *channels as usize)
                } else {
                    samples.clone()
                };
                
                self.process_audio(&samples, *sample_rate)
            }
        }
    }
    
    /// 解码音频数据
    fn decode_audio(&self, audio_data: &[u8]) -> Result<(Vec<f32>, u32)> {
        #[cfg(feature = "multimodal")]
        {
            // 使用symphonia库解码音频数据
            let cursor = Cursor::new(audio_data);
            let mss = MediaSourceStream::new(Box::new(cursor), Default::default());
            
            // 创建格式探针
            let hint = Hint::new();
            let format_opts = FormatOptions::default();
            let metadata_opts = MetadataOptions::default();
            let decoder_opts = DecoderOptions::default();
            
            // 探测并获取格式读取器
            let probed = symphonia::default::get_probe()
                .format(&hint, mss, &format_opts, &metadata_opts)
                .map_err(|e| Error::data(format!("音频解码失败: {}", e)))?;
                
            let mut format = probed.format;
            
            // 获取默认音轨
            let track = format.default_track()
                .ok_or_else(|| Error::data("没有找到默认音轨".to_string()))?;
                
            let sample_rate = track.codec_params.sample_rate
                .ok_or_else(|| Error::data("无法获取采样率".to_string()))?;
                
            // 获取解码器
            let mut decoder = symphonia::default::get_codecs()
                .make(&track.codec_params, &decoder_opts)
                .map_err(|e| Error::data(format!("创建解码器失败: {}", e)))?;
                
            // 解码所有音频帧并收集样本
            let mut samples = Vec::new();
            let channels = track.codec_params.channels
                .ok_or_else(|| Error::data("无法获取通道数".to_string()))?
                .count();
                
            loop {
                let packet = match format.next_packet() {
                    Ok(packet) => packet,
                    Err(_) => break, // 结束解码
                };
                
                // 如果数据包不属于选定的音轨，跳过
                if packet.track_id() != track.id {
                    continue;
                }
                
                // 解码音频帧
                match decoder.decode(&packet) {
                    Ok(decoded) => {
                        // 根据采样格式，转换为f32样本
                        // decoded 是 AudioBufferRef，直接匹配其类型
                        match decoded {
                            symphonia::core::audio::AudioBufferRef::F32(buf) => {
                                // 使用 planes() 方法访问音频数据，返回 &[&[S]]，可以直接索引访问通道
                                let planes_slice = buf.planes().planes();
                                if !planes_slice.is_empty() {
                                    let channel = planes_slice[0];
                                    samples.extend_from_slice(channel);
                                }
                            },
                            symphonia::core::audio::AudioBufferRef::U8(buf) => {
                                let planes_slice = buf.planes().planes();
                                if !planes_slice.is_empty() {
                                    let channel = planes_slice[0];
                                    for &sample in channel.iter() {
                                        samples.push((sample as f32 / 128.0) - 1.0);
                                    }
                                }
                            },
                            symphonia::core::audio::AudioBufferRef::U16(buf) => {
                                let planes_slice = buf.planes().planes();
                                if !planes_slice.is_empty() {
                                    let channel = planes_slice[0];
                                    for &sample in channel.iter() {
                                        samples.push((sample as f32 / 32768.0) - 1.0);
                                    }
                                }
                            },
                            symphonia::core::audio::AudioBufferRef::U24(buf) => {
                                let planes_slice = buf.planes().planes();
                                if !planes_slice.is_empty() {
                                    let channel = planes_slice[0];
                                    for &sample in channel.iter() {
                                        // u24 转换为 f32: u24 的值范围是 0..=16777215
                                        // u24 类型在 symphonia 中是一个包装类型，需要手动转换
                                        // 使用 unsafe 转换，因为 u24 内部存储为 u32
                                        let sample_u32: u32 = unsafe { std::mem::transmute(sample) };
                                        samples.push((sample_u32 as f32 / 8388608.0) - 1.0);
                                    }
                                }
                            },
                            symphonia::core::audio::AudioBufferRef::U32(buf) => {
                                let planes_slice = buf.planes().planes();
                                if !planes_slice.is_empty() {
                                    let channel = planes_slice[0];
                                    for &sample in channel.iter() {
                                        samples.push((sample as f32 / 2147483648.0) - 1.0);
                                    }
                                }
                            },
                            symphonia::core::audio::AudioBufferRef::S8(buf) => {
                                let planes_slice = buf.planes().planes();
                                if !planes_slice.is_empty() {
                                    let channel = planes_slice[0];
                                    for &sample in channel.iter() {
                                        samples.push(sample as f32 / 128.0);
                                    }
                                }
                            },
                            symphonia::core::audio::AudioBufferRef::S16(buf) => {
                                let planes_slice = buf.planes().planes();
                                if !planes_slice.is_empty() {
                                    let channel = planes_slice[0];
                                    for &sample in channel.iter() {
                                        samples.push(sample as f32 / 32768.0);
                                    }
                                }
                            },
                            symphonia::core::audio::AudioBufferRef::S24(buf) => {
                                let planes_slice = buf.planes().planes();
                                if !planes_slice.is_empty() {
                                    let channel = planes_slice[0];
                                    for &sample in channel.iter() {
                                        // i24 转换为 f32: i24 的值范围是 -8388608..=8388607
                                        // i24 类型在 symphonia 中是一个包装类型，需要手动转换
                                        // 使用 unsafe 转换，因为 i24 内部存储为 i32
                                        let sample_i32: i32 = unsafe { std::mem::transmute(sample) };
                                        samples.push(sample_i32 as f32 / 8388608.0);
                                    }
                                }
                            },
                            symphonia::core::audio::AudioBufferRef::S32(buf) => {
                                let planes_slice = buf.planes().planes();
                                if !planes_slice.is_empty() {
                                    let channel = planes_slice[0];
                                    for &sample in channel.iter() {
                                        samples.push(sample as f32 / 2147483648.0);
                                    }
                                }
                            },
                        }
                    },
                    Err(e) => {
                        debug!("解码帧错误: {}", e);
                        continue;
                    }
                }
            }
            
            // 如果有多个通道，转换为单声道
            if channels > 1 {
                let mono_samples = self.convert_to_mono(&samples, channels);
                return Ok((mono_samples, sample_rate));
            }
            
            Ok((samples, sample_rate))
        }
        #[cfg(not(feature = "multimodal"))]
        {
            Err(Error::feature_not_enabled("multimodal"))
        }
    }
    
    /// 将多通道音频转换为单通道
    fn convert_to_mono(&self, samples: &[f32], channels: usize) -> Vec<f32> {
        if channels == 1 {
            return samples.to_vec();
        }
        
        let frames = samples.len() / channels;
        let mut mono_samples = Vec::with_capacity(frames);
        
        for i in 0..frames {
            let mut sum = 0.0;
            for c in 0..channels {
                sum += samples[i * channels + c];
            }
            mono_samples.push(sum / channels as f32);
        }
        
        mono_samples
    }
    
    /// 处理音频并提取特征
    pub fn process_audio(&self, samples: &[f32], sample_rate: u32) -> Result<FeatureVector> {
        // 调整采样率（如有必要）
        let processed_samples = if sample_rate as usize != self.config.sample_rate {
            debug!("重采样音频从 {} Hz 到 {} Hz", sample_rate, self.config.sample_rate);
            // 简单实现，实际应使用专业的重采样库
            let ratio = self.config.sample_rate as f32 / sample_rate as f32;
            let new_len = (samples.len() as f32 * ratio) as usize;
            let mut resampled = Vec::with_capacity(new_len);
            
            for i in 0..new_len {
                let orig_idx = (i as f32 / ratio) as usize;
                if orig_idx < samples.len() {
                    resampled.push(samples[orig_idx]);
                } else {
                    break;
                }
            }
            resampled
        } else {
            samples.to_vec()
        };
        
        // 使用处理后的样本（不限制长度，由配置中的其他参数控制）
        let limited_samples = processed_samples;
        
        // 提取特征
        let sample_rate_u32 = self.config.sample_rate as u32;
        let features = match self.config.feature_type {
            AudioFeatureType::MFCC => self.extract_mfcc(&limited_samples, sample_rate_u32)?,
            AudioFeatureType::MelSpectrogram => self.extract_mel_spectrogram(&limited_samples, sample_rate_u32)?,
            AudioFeatureType::Chroma => self.extract_chroma(&limited_samples, sample_rate_u32)?,
            AudioFeatureType::SpectralCentroid => self.extract_spectral_centroid(&limited_samples, sample_rate_u32)?,
            AudioFeatureType::SpectralRolloff => self.extract_spectral_rolloff(&limited_samples, sample_rate_u32)?,
            AudioFeatureType::ZeroCrossingRate => self.extract_zero_crossing_rate(&limited_samples, sample_rate_u32)?,
            AudioFeatureType::Energy => self.extract_energy(&limited_samples, sample_rate_u32)?,
            AudioFeatureType::RMSEnergy => self.extract_energy(&limited_samples, sample_rate_u32)?, // 使用Energy提取器
            AudioFeatureType::HarmonicRatio => self.extract_harmonic_noise_ratio(&limited_samples, sample_rate_u32)?,
            AudioFeatureType::Waveform => self.extract_energy(&limited_samples, sample_rate_u32)?, // 使用Energy提取器作为波形特征
            AudioFeatureType::SpectralBandwidth => self.extract_spectral_centroid(&limited_samples, sample_rate_u32)?, // 使用SpectralCentroid提取器
            AudioFeatureType::SpectralFlatness => self.extract_spectral_centroid(&limited_samples, sample_rate_u32)?, // 使用SpectralCentroid提取器
            AudioFeatureType::SpectralContrast => self.extract_spectral_centroid(&limited_samples, sample_rate_u32)?, // 使用SpectralCentroid提取器
            AudioFeatureType::SpectralFlux => self.extract_spectral_centroid(&limited_samples, sample_rate_u32)?, // 使用SpectralCentroid提取器
            AudioFeatureType::SpectralEntropy => self.extract_spectral_centroid(&limited_samples, sample_rate_u32)?, // 使用SpectralCentroid提取器
            AudioFeatureType::Pitch => self.extract_spectral_centroid(&limited_samples, sample_rate_u32)?, // 使用SpectralCentroid提取器
            AudioFeatureType::LPC => self.extract_mfcc(&limited_samples, sample_rate_u32)?, // 使用MFCC提取器
            AudioFeatureType::LPCC => self.extract_mfcc(&limited_samples, sample_rate_u32)?, // 使用MFCC提取器
            AudioFeatureType::PerceptualWeights => self.extract_mel_spectrogram(&limited_samples, sample_rate_u32)?, // 使用MelSpectrogram提取器
            AudioFeatureType::ChromaCQT | AudioFeatureType::ChromaENS | AudioFeatureType::ChromaCENS => self.extract_chroma(&limited_samples, sample_rate_u32)?,
            AudioFeatureType::Custom(ref name) => self.extract_custom_features(&limited_samples, sample_rate_u32, name)?,
        };
        
        // 归一化特征（如有必要）
        let mut normalized_features = features;
        if self.config.normalize {
            self.normalize_features(&mut normalized_features);
        }
        
        // 创建特征向量
        let feature_data = if normalized_features.shape()[0] == 1 {
            // 单行，转换为1D向量
            let flattened: Vec<f32> = normalized_features.iter().copied().collect();
            TensorData::from_vec(vec![self.config.get_feature_dimension()], flattened)?
        } else {
            // 多行，计算平均值
            let mut avg_features = Array1::zeros(normalized_features.shape()[1]);
            for row in normalized_features.axis_iter(Axis(0)) {
                avg_features += &row;
            }
            avg_features /= normalized_features.shape()[0] as f32;
            
            let flattened: Vec<f32> = avg_features.iter().copied().collect();
            TensorData::from_vec(vec![self.config.get_feature_dimension()], flattened)?
        };
        
        // 创建特征向量并添加元数据
        let feature_values = feature_data.to_vec()
            .map_err(|e| Error::data(format!("无法转换特征数据: {}", e)))?;
        let feature_name = format!("audio-{:?}", self.config.feature_type);
        let mut feature_vector = FeatureVector::new(feature_values, &feature_name);
        // 添加特征名称到元数据
        for i in 0..feature_vector.data.len() {
            feature_vector.metadata.insert(format!("feature_{}", i), format!("{}_{}", feature_name, i));
        }
        
        // 添加元数据
        feature_vector.metadata.insert("sample_rate".to_string(), self.config.sample_rate.to_string());
        feature_vector.metadata.insert("feature_type".to_string(), format!("{:?}", self.config.feature_type));
        feature_vector.metadata.insert("duration".to_string(), (limited_samples.len() as f32 / self.config.sample_rate as f32).to_string());
        
        Ok(feature_vector)
    }
    
    /// 提取MFCC特征
    fn extract_mfcc(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let mfcc_extractor = MFCCExtractor::new(
            13, // n_mfcc
            26, // n_mels
            2048, // n_fft
            self.config.frame_length,
            self.config.hop_length,
            true, // include_energy
            self.config.normalize,
            2, // dct_type
            sample_rate,
            0.97, // preemphasis
        );
        
        mfcc_extractor.extract(samples)
    }
    
    /// 提取梅尔谱图特征
    fn extract_mel_spectrogram(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let mel_extractor = MelSpectrogramExtractor::new(
            self.config.get_feature_dimension(),
            2048, // n_fft
            self.config.frame_length,
            self.config.hop_length,
            self.config.normalize,
            sample_rate,
            0.97, // preemphasis
            2.0, // power
            true, // log_mel
            0.0, // f_min
            None, // f_max
        );
        
        mel_extractor.extract(samples)
    }
    
    /// 特征向量归一化
    fn normalize_features(&self, features: &mut Array2<f32>) {
        // L2归一化
        for mut row in features.axis_iter_mut(Axis(0)) {
            let mut norm = 0.0;
            for &val in row.iter() {
                norm += val * val;
            }
            norm = norm.sqrt();
            
            if norm > 1e-10 {
                for val in row.iter_mut() {
                    *val /= norm;
                }
            }
        }
    }
    
    /// 提取色度特征
    fn extract_chroma(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let chroma_extractor = ChromaExtractor::new(
            12, // n_chroma
            2048, // n_fft
            self.config.frame_length,
            self.config.hop_length,
            sample_rate,
            self.config.normalize,
            0.97, // preemphasis
        );
        
        chroma_extractor.extract(samples)
    }
    
    /// 提取谱质心特征
    fn extract_spectral_centroid(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let centroid_extractor = SpectralCentroidExtractor::new(
            2048, // n_fft
            self.config.frame_length,
            self.config.hop_length,
            sample_rate,
            self.config.normalize,
            0.97, // preemphasis
        );
        
        centroid_extractor.extract(samples)
    }
    
    /// 提取谱衰减特征
    fn extract_spectral_rolloff(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let rolloff_extractor = SpectralRolloffExtractor::new(
            2048, // n_fft
            self.config.frame_length,
            self.config.hop_length,
            sample_rate,
            self.config.normalize,
            0.97, // preemphasis
            0.85, // roll_percent
        );
        
        rolloff_extractor.extract(samples)
    }
    
    /// 提取零交叉率特征
    fn extract_zero_crossing_rate(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let zcr_extractor = ZeroCrossingRateExtractor::new(
            self.config.frame_length,
            self.config.hop_length,
            sample_rate,
            self.config.normalize,
        );
        
        zcr_extractor.extract(samples)
    }
    
    /// 提取能量特征
    fn extract_energy(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let energy_extractor = EnergyExtractor::new(
            self.config.frame_length,
            self.config.hop_length,
            self.config.normalize,
            true, // use_db
            -80.0, // min_db
        );
        
        energy_extractor.extract(samples)
    }
    
    /// 提取时域特征
    fn extract_temporal_features(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        let features = vec![
            TemporalFeatureType::Mean,
            TemporalFeatureType::StdDev,
            TemporalFeatureType::RootMeanSquare,
            TemporalFeatureType::ZeroCrossingRate,
            TemporalFeatureType::Kurtosis,
            TemporalFeatureType::Skewness,
            TemporalFeatureType::CrestFactor,
        ];
        
        let temporal_extractor = TemporalFeaturesExtractor::new(
            self.config.frame_length,
            self.config.hop_length,
            sample_rate,
            self.config.normalize,
            features,
        );
        
        temporal_extractor.extract(samples)
    }
    
    /// 提取谐波/噪声比特征
    #[cfg(feature = "multimodal")]
    fn extract_harmonic_noise_ratio(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        use super::extractors::{apply_fft, compute_power_spectrum, apply_hanning_window, frame_signal};
        
        if samples.len() < self.config.frame_length {
            return Err(Error::data("音频太短，无法提取谐波/噪声比特征".to_string()));
        }
        
        // 1. 分帧
        let frames = frame_signal(samples, self.config.frame_length, self.config.hop_length);
        if frames.is_empty() {
            return Err(Error::data("无法从音频中提取足够的帧".to_string()));
        }
        
        // 2. 对每一帧计算谐波/噪声比
        let mut harmonic_ratios = Vec::with_capacity(frames.len());
        
        for frame in &frames {
            // 加窗
            let windowed = apply_hanning_window(frame);
            
            // 计算FFT
            let n_fft = self.config.frame_length.max(2048);
            let fft_result = apply_fft(&windowed, n_fft);
            let power_spectrum = compute_power_spectrum(&fft_result);
            
            // 只使用正频率部分
            let n_bins = n_fft / 2 + 1;
            let power_spectrum = &power_spectrum[..n_bins];
            
            // 找到基频（fundamental frequency）
            // 使用自相关方法找到基频
            let fundamental_freq = self.find_fundamental_frequency(frame, sample_rate);
            
            if fundamental_freq > 0.0 {
                // 计算谐波能量和噪声能量
                let harmonic_energy = self.compute_harmonic_energy(
                    power_spectrum,
                    fundamental_freq,
                    sample_rate,
                    n_fft
                );
                let total_energy: f32 = power_spectrum.iter().sum();
                let noise_energy = total_energy - harmonic_energy;
                
                // 计算谐波/噪声比（使用对数刻度避免除零）
                let ratio = if noise_energy > 1e-10 {
                    (harmonic_energy / noise_energy).ln_1p() // ln(1 + ratio) 避免负值
                } else {
                    10.0 // 如果噪声能量很小，认为主要是谐波
                };
                
                harmonic_ratios.push(ratio);
            } else {
                // 如果找不到基频，使用频谱峰值作为近似
                let max_power: f32 = power_spectrum.iter().fold(0.0_f32, |a, &b| a.max(b));
                let mean_power: f32 = power_spectrum.iter().sum::<f32>() / power_spectrum.len() as f32;
                let ratio = if mean_power > 1e-10 {
                    (max_power / mean_power).ln_1p()
                } else {
                    0.0
                };
                harmonic_ratios.push(ratio);
            }
        }
        
        // 3. 转换为Array2格式
        let dimension = self.config.get_feature_dimension();
        let n_frames = harmonic_ratios.len();
        
        // 如果维度为1，返回每帧的谐波/噪声比
        if dimension == 1 {
            let mut result = Array2::zeros((n_frames, 1));
            for (i, &ratio) in harmonic_ratios.iter().enumerate() {
                result[[i, 0]] = ratio;
            }
            Ok(result)
        } else {
            // 如果维度大于1，使用时间池化
            let mean_ratio = harmonic_ratios.iter().sum::<f32>() / harmonic_ratios.len() as f32;
            let std_ratio = if harmonic_ratios.len() > 1 {
                let variance: f32 = harmonic_ratios.iter()
                    .map(|&r| (r - mean_ratio).powi(2))
                    .sum::<f32>() / (harmonic_ratios.len() - 1) as f32;
                variance.sqrt()
            } else {
                0.0
            };
            
            let mut result = Array2::zeros((1, dimension));
            result[[0, 0]] = mean_ratio;
            if dimension > 1 {
                result[[0, 1]] = std_ratio;
            }
            // 填充剩余维度
            for i in 2..dimension {
                result[[0, i]] = mean_ratio * (i as f32 * 0.1).sin();
            }
            
            Ok(result)
        }
    }
    
    /// 找到基频（使用自相关方法）
    fn find_fundamental_frequency(&self, frame: &[f32], sample_rate: u32) -> f32 {
        if frame.len() < 2 {
            return 0.0;
        }
        
        // 计算自相关
        let max_lag = (sample_rate as usize / 80).min(frame.len() / 2); // 最低80Hz
        let min_lag = (sample_rate as usize / 2000).max(2); // 最高2000Hz
        
        let mut max_corr = 0.0;
        let mut best_lag = 0;
        
        for lag in min_lag..max_lag {
            let mut corr = 0.0;
            for i in 0..(frame.len() - lag) {
                corr += frame[i] * frame[i + lag];
            }
            
            if corr > max_corr {
                max_corr = corr;
                best_lag = lag;
            }
        }
        
        if best_lag > 0 && max_corr > 0.1 {
            sample_rate as f32 / best_lag as f32
        } else {
            0.0
        }
    }
    
    /// 计算谐波能量
    fn compute_harmonic_energy(&self, power_spectrum: &[f32], fundamental_freq: f32, sample_rate: u32, n_fft: usize) -> f32 {
        let mut harmonic_energy = 0.0;
        
        // 计算前10个谐波的能量
        for harmonic in 1..=10 {
            let harmonic_freq = fundamental_freq * harmonic as f32;
            let bin_idx = (harmonic_freq * n_fft as f32 / sample_rate as f32) as usize;
            
            if bin_idx < power_spectrum.len() {
                // 考虑相邻bin的能量（使用简单的窗口）
                let start_bin = bin_idx.saturating_sub(1);
                let end_bin = (bin_idx + 2).min(power_spectrum.len());
                
                for i in start_bin..end_bin {
                    harmonic_energy += power_spectrum[i];
                }
            }
        }
        
        harmonic_energy
    }
    
    /// 提取组合特征
    fn extract_combined_features(&self, samples: &[f32], sample_rate: u32) -> Result<Array2<f32>> {
        // 组合特征是把多种特征合并在一起
        // 这里我们结合MFCC, 谱质心和能量特征
        
        // 提取MFCC特征
        let mfcc_features = self.extract_mfcc(samples, sample_rate)?;
        
        // 提取谱质心特征
        let centroid_features = self.extract_spectral_centroid(samples, sample_rate)?;
        
        // 提取能量特征
        let energy_features = self.extract_energy(samples, sample_rate)?;
        
        // 确定最短帧数
        let n_frames = mfcc_features.shape()[0]
            .min(centroid_features.shape()[0])
            .min(energy_features.shape()[0]);
            
        // 创建结果数组，设置适当的维度
        let mfcc_dim = mfcc_features.shape()[1];
        let combined_dim = mfcc_dim + 1 + 1; // MFCC维度 + 谱质心维度 + 能量维度
        
        // 确保不超过配置中指定的维度
        let output_dim = combined_dim.min(self.config.get_feature_dimension());
        let mut result = Array2::zeros((n_frames, output_dim));
        
        // 合并特征
        for i in 0..n_frames {
            let mut col_idx = 0;
            
            // 复制MFCC特征
            for j in 0..mfcc_dim.min(output_dim) {
                result[[i, col_idx]] = mfcc_features[[i, j]];
                col_idx += 1;
                
                if col_idx >= output_dim {
                    break;
                }
            }
            
            // 添加谱质心特征
            if col_idx < output_dim {
                result[[i, col_idx]] = centroid_features[[i, 0]];
                col_idx += 1;
            }
            
            // 添加能量特征
            if col_idx < output_dim {
                result[[i, col_idx]] = energy_features[[i, 0]];
            }
        }
        
        Ok(result)
    }
    
    /// 提取自定义特征
    fn extract_custom_features(&self, samples: &[f32], sample_rate: u32, name: &str) -> Result<Array2<f32>> {
        // 根据名称调用不同的特征提取器
        match name {
            "harmonic_noise" => self.extract_harmonic_noise_ratio(samples, sample_rate),
            "combined" => self.extract_combined_features(samples, sample_rate),
            "mfcc" => self.extract_mfcc(samples, sample_rate),
            "mel" => self.extract_mel_spectrogram(samples, sample_rate),
            "chroma" => self.extract_chroma(samples, sample_rate),
            "centroid" => self.extract_spectral_centroid(samples, sample_rate),
            "rolloff" => self.extract_spectral_rolloff(samples, sample_rate),
            "zcr" => self.extract_zero_crossing_rate(samples, sample_rate),
            "energy" => self.extract_energy(samples, sample_rate),
            "temporal" => self.extract_temporal_features(samples, sample_rate),
            _ => Err(Error::data(format!("未知的自定义特征类型: {}", name))),
        }
    }

    /// 从原始音频数据中提取特征向量
    pub fn extract_from_raw_data(&self, data: &[u8]) -> Result<FeatureVector> {
        let (samples, sample_rate) = self.decode_audio(data)?;
        self.process_audio(&samples, sample_rate)
    }
}

impl ModalityExtractor for AudioFeatureExtractor {
    fn extract_features(&self, data: &serde_json::Value) -> Result<CoreTensorData> {
        // 从JSON中提取音频数据
        let audio_data = match data {
            serde_json::Value::String(base64_data) => base64_data.clone(),
            serde_json::Value::Object(obj) => {
                if let Some(serde_json::Value::String(base64_data)) = obj.get("data") {
                    base64_data.clone()
                } else {
                    return Err(Error::InvalidArgument("Audio data not found in JSON object".to_string()));
                }
            },
            _ => return Err(Error::InvalidArgument("Invalid audio data format".to_string())),
        };
        
        let source = AudioSource::Base64(audio_data);
        let feature_vector = self.extract_from_source(&source)?;
        
        // 转换为CoreTensorData
        use chrono::Utc;
        use uuid::Uuid;
        let tensor = CoreTensorData {
            id: Uuid::new_v4().to_string(),
            shape: vec![1, feature_vector.data.len()],
            data: feature_vector.data.clone(),
            dtype: "Float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: std::collections::HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        Ok(tensor)
    }
    
    fn get_config(&self) -> Result<serde_json::Value, Error> {
        let config = serde_json::to_value(&self.config)
            .map_err(|e| Error::InvalidArgument(format!("Failed to serialize config: {}", e)))?;
        Ok(config)
    }
    
    fn get_modality_type(&self) -> ModalityType {
        ModalityType::Audio
    }
    
    fn get_dimension(&self) -> usize {
        self.config.get_feature_dimension()
    }
}

impl FeatureExtractor for AudioFeatureExtractor {
    fn extract_features(&self, data: &[u8], metadata: Option<&HashMap<String, String>>) -> Result<Vec<f32>> {
        // 从原始数据提取特征
        let feature_vector = self.extract_from_raw_data(data)?;
        Ok(feature_vector.data.clone())
    }

    fn batch_extract(&self, data_batch: &[Vec<u8>], metadata_batch: Option<&[HashMap<String, String>]>) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(data_batch.len());
        
        for data in data_batch {
            match self.extract_from_raw_data(data) {
                Ok(feature) => results.push(feature.data.clone()),
                Err(e) => {
                    error!("批量提取音频特征时发生错误: {}", e);
                    return Err(e);
                }
            }
        }
        
        Ok(results)
    }
    
    fn get_output_dim(&self) -> usize {
        self.config.get_feature_dimension()
    }
    
    fn get_extractor_type(&self) -> String {
        format!("AudioFeatureExtractor-{:?}", self.config.feature_type)
    }
}

/// 创建默认音频特征提取器
pub fn create_default_audio_extractor() -> AudioFeatureExtractor {
    AudioFeatureExtractor::new(AudioProcessingConfig::default())
} 