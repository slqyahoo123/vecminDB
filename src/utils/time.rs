//! 时间工具模块
//! 
//! 提供时间格式化、时间计算、性能测量等功能。

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};


/// 时间工具
pub struct TimeUtils;

impl TimeUtils {
    /// 获取当前Unix时间戳（秒）
    pub fn unix_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// 获取当前Unix时间戳（毫秒）
    pub fn unix_timestamp_millis() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// 获取当前Unix时间戳（微秒）
    pub fn unix_timestamp_micros() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }

    /// 获取当前Unix时间戳（纳秒）
    pub fn unix_timestamp_nanos() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    /// 从Unix时间戳创建SystemTime
    pub fn from_unix_timestamp(timestamp: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(timestamp)
    }

    /// 从Unix时间戳（毫秒）创建SystemTime
    pub fn from_unix_timestamp_millis(timestamp: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_millis(timestamp)
    }

    /// 格式化持续时间为可读字符串
    pub fn format_duration(duration: Duration) -> String {
        let total_secs = duration.as_secs();
        let days = total_secs / 86400;
        let hours = (total_secs % 86400) / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        let millis = duration.subsec_millis();

        if days > 0 {
            format!("{}d {}h {}m {}s", days, hours, minutes, seconds)
        } else if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else if seconds > 0 {
            format!("{}.{}s", seconds, millis / 100)
        } else {
            format!("{}ms", millis)
        }
    }

    /// 解析持续时间字符串（简单版本）
    pub fn parse_duration(s: &str) -> Option<Duration> {
        if s.ends_with("ms") {
            let num_str = &s[..s.len() - 2];
            num_str.parse::<u64>().ok().map(Duration::from_millis)
        } else if s.ends_with("s") {
            let num_str = &s[..s.len() - 1];
            num_str.parse::<u64>().ok().map(Duration::from_secs)
        } else if s.ends_with("m") {
            let num_str = &s[..s.len() - 1];
            num_str.parse::<u64>().ok().map(|n| Duration::from_secs(n * 60))
        } else if s.ends_with("h") {
            let num_str = &s[..s.len() - 1];
            num_str.parse::<u64>().ok().map(|n| Duration::from_secs(n * 3600))
        } else {
            s.parse::<u64>().ok().map(Duration::from_secs)
        }
    }

    /// 计算两个时间点之间的差值
    pub fn time_diff(start: SystemTime, end: SystemTime) -> Duration {
        end.duration_since(start).unwrap_or_default()
    }

    /// 检查时间是否在指定范围内
    pub fn is_within_range(
        time: SystemTime,
        start: SystemTime,
        end: SystemTime,
    ) -> bool {
        time >= start && time <= end
    }

    /// 获取今天的开始时间（简化版本）
    pub fn today_start() -> SystemTime {
        let now = Self::unix_timestamp();
        let day_start = (now / 86400) * 86400;
        Self::from_unix_timestamp(day_start)
    }

    /// 获取明天的开始时间
    pub fn tomorrow_start() -> SystemTime {
        Self::today_start() + Duration::from_secs(86400)
    }
}

/// 性能计时器
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    /// 创建新的计时器
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
        }
    }

    /// 重启计时器
    pub fn restart(&mut self) {
        self.start = Instant::now();
    }

    /// 获取已经过的时间
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// 获取已经过的毫秒数
    pub fn elapsed_millis(&self) -> u64 {
        self.elapsed().as_millis() as u64
    }

    /// 获取已经过的微秒数
    pub fn elapsed_micros(&self) -> u64 {
        self.elapsed().as_micros() as u64
    }

    /// 停止计时器并返回持续时间
    pub fn stop(self) -> Duration {
        self.elapsed()
    }

    /// 停止计时器并打印结果
    pub fn stop_and_print(self) {
        let duration = self.elapsed();
        println!("{}: {}", self.name, TimeUtils::format_duration(duration));
    }

    /// 记录检查点
    pub fn checkpoint(&self, label: &str) {
        let duration = self.elapsed();
        println!("{} - {}: {}", self.name, label, TimeUtils::format_duration(duration));
    }
}

/// 时间窗口
pub struct TimeWindow {
    start: SystemTime,
    duration: Duration,
}

impl TimeWindow {
    /// 创建新的时间窗口
    pub fn new(start: SystemTime, duration: Duration) -> Self {
        Self { start, duration }
    }

    /// 创建从现在开始的时间窗口
    pub fn from_now(duration: Duration) -> Self {
        Self {
            start: SystemTime::now(),
            duration,
        }
    }

    /// 获取窗口结束时间
    pub fn end(&self) -> SystemTime {
        self.start + self.duration
    }

    /// 检查给定时间是否在窗口内
    pub fn contains(&self, time: SystemTime) -> bool {
        time >= self.start && time <= self.end()
    }

    /// 检查窗口是否已过期
    pub fn is_expired(&self) -> bool {
        SystemTime::now() > self.end()
    }

    /// 获取剩余时间
    pub fn remaining(&self) -> Option<Duration> {
        let now = SystemTime::now();
        let end = self.end();
        
        if now < end {
            end.duration_since(now).ok()
        } else {
            None
        }
    }

    /// 获取已过去的时间
    pub fn elapsed(&self) -> Duration {
        SystemTime::now()
            .duration_since(self.start)
            .unwrap_or_default()
    }

    /// 获取窗口进度（0.0 到 1.0）
    pub fn progress(&self) -> f64 {
        let elapsed = self.elapsed();
        let total = self.duration;
        
        if total.is_zero() {
            1.0
        } else {
            (elapsed.as_secs_f64() / total.as_secs_f64()).min(1.0)
        }
    }
}

/// 频率限制器
pub struct RateLimiter {
    last_check: Instant,
    interval: Duration,
}

impl RateLimiter {
    /// 创建新的频率限制器
    pub fn new(interval: Duration) -> Self {
        Self {
            last_check: Instant::now(),
            interval,
        }
    }

    /// 检查是否可以执行操作
    pub fn check(&mut self) -> bool {
        let now = Instant::now();
        
        if now.duration_since(self.last_check) >= self.interval {
            self.last_check = now;
            true
        } else {
            false
        }
    }

    /// 等待直到可以执行操作
    pub fn wait(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_check);
        
        if elapsed < self.interval {
            let wait_time = self.interval - elapsed;
            std::thread::sleep(wait_time);
        }
        
        self.last_check = Instant::now();
    }
}

/// 超时检查器
pub struct Timeout {
    start: Instant,
    duration: Duration,
}

impl Timeout {
    /// 创建新的超时检查器
    pub fn new(duration: Duration) -> Self {
        Self {
            start: Instant::now(),
            duration,
        }
    }

    /// 检查是否已超时
    pub fn is_expired(&self) -> bool {
        self.start.elapsed() >= self.duration
    }

    /// 获取剩余时间
    pub fn remaining(&self) -> Duration {
        let elapsed = self.start.elapsed();
        if elapsed >= self.duration {
            Duration::from_secs(0)
        } else {
            self.duration - elapsed
        }
    }

    /// 重置超时计时器
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

/// 延迟执行器
pub struct DelayedExecutor<F> {
    delay: Duration,
    callback: Option<F>,
    start_time: Option<Instant>,
}

impl<F> DelayedExecutor<F>
where
    F: FnOnce(),
{
    /// 创建新的延迟执行器
    pub fn new(delay: Duration, callback: F) -> Self {
        Self {
            delay,
            callback: Some(callback),
            start_time: None,
        }
    }

    /// 开始延迟计时
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// 检查是否应该执行回调
    pub fn check_and_execute(&mut self) -> bool {
        if let Some(start_time) = self.start_time {
            if start_time.elapsed() >= self.delay {
                if let Some(callback) = self.callback.take() {
                    callback();
                    return true;
                }
            }
        }
        false
    }

    /// 取消延迟执行
    pub fn cancel(&mut self) {
        self.callback = None;
        self.start_time = None;
    }
}

/// 时间格式化器
pub struct TimeFormatter;

impl TimeFormatter {
    /// 格式化为ISO 8601格式（简化版本）
    pub fn to_iso8601(time: SystemTime) -> String {
        let timestamp = time
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        // 简化版本，实际应该使用chrono库
        format!("1970-01-01T00:00:{}Z", timestamp)
    }

    /// 格式化为RFC 3339格式（简化版本）
    pub fn to_rfc3339(time: SystemTime) -> String {
        Self::to_iso8601(time) // 简化实现
    }

    /// 格式化为自定义格式
    pub fn format_custom(time: SystemTime, _format: &str) -> String {
        // 简化实现，实际应该支持格式字符串
        Self::to_iso8601(time)
    }
} 