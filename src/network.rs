//! Network module stub
//!
//! This is a minimal stub for network-related types.

use serde::{Deserialize, Serialize};
use crate::Result;

/// Node role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    Master,
    Worker,
    Coordinator,
}

/// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub role: NodeRole,
    pub address: String,
}

/// Network manager stub
#[derive(Debug, Clone)]
pub struct NetworkManager;

impl NetworkManager {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }
}

impl Default for NetworkManager {
    fn default() -> Self {
        Self
    }
}

