mod dummy;
mod wasm;
mod process;
mod docker;

pub use dummy::DummySandbox;
pub use wasm::WasmSandbox;
pub use process::ProcessSandbox;
pub use docker::DockerSandbox;

use crate::Result;
use crate::algorithm::executor::config::{SandboxType, ExecutorConfig};
use crate::algorithm::executor::sandbox::interface::Sandbox;
use std::future::Future;
use std::pin::Pin;

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// 创建指定类型的沙箱实例
pub async fn create_sandbox(
    sandbox_type: SandboxType,
    config: &ExecutorConfig,
) -> Result<Box<dyn Sandbox>> {
    match sandbox_type {
        SandboxType::Wasm => {
            // 创建WASM沙箱
            let sandbox = WasmSandbox::new(config).await?;
            Ok(Box::new(sandbox))
        },
        SandboxType::Docker => {
            // 创建Docker沙箱
            let sandbox = DockerSandbox::new(config).await?;
            Ok(Box::new(sandbox))
        },
        SandboxType::Process => {
            // 创建进程沙箱
            let sandbox = ProcessSandbox::new(config).await?;
            Ok(Box::new(sandbox))
        },
        SandboxType::LocalProcess => {
            // 创建本地进程沙箱（直接在当前进程中执行）
            let sandbox = ProcessSandbox::new_local(config).await?;
            Ok(Box::new(sandbox))
        },
        SandboxType::IsolatedProcess => {
            // 创建隔离进程沙箱
            let sandbox = ProcessSandbox::new_isolated(config).await?;
            Ok(Box::new(sandbox))
        },
    }
} 