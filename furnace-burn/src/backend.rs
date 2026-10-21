use burn::tensor::Device;
use serde::{Deserialize, Serialize};
use std::fmt;

use furnace::error::{ModelError, Result};

/// Compute backends that can be selected at runtime.
///
/// Since Burn 0.22 the backend is chosen through a runtime [`Device`] rather than a
/// type parameter, so a single binary can serve on any backend compiled into it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// Pure-Rust CPU backend (burn-flex). Always available.
    #[default]
    Cpu,
    #[cfg(feature = "wgpu")]
    Wgpu,
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl BackendKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => "wgpu",
            #[cfg(feature = "metal")]
            BackendKind::Metal => "metal",
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => "cuda",
        }
    }

    /// Parse a backend name. Fails if the backend was not compiled into this binary.
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_lowercase().as_str() {
            "cpu" | "flex" => Ok(BackendKind::Cpu),
            #[cfg(feature = "wgpu")]
            "wgpu" | "gpu" => Ok(BackendKind::Wgpu),
            #[cfg(feature = "metal")]
            "metal" => Ok(BackendKind::Metal),
            #[cfg(feature = "cuda")]
            "cuda" => Ok(BackendKind::Cuda),
            _ => Err(ModelError::InvalidArgument {
                arg: "backend".to_string(),
                value: name.to_string(),
                reason: format!(
                    "backend not available in this build. Available backends: {}",
                    Self::available_names().join(", ")
                ),
            }
            .into()),
        }
    }

    /// Backends compiled into this binary.
    pub fn available() -> Vec<BackendKind> {
        vec![
            BackendKind::Cpu,
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu,
            #[cfg(feature = "metal")]
            BackendKind::Metal,
            #[cfg(feature = "cuda")]
            BackendKind::Cuda,
        ]
    }

    pub fn available_names() -> Vec<&'static str> {
        Self::available().iter().map(|b| b.as_str()).collect()
    }

    /// Create the Burn device for this backend.
    pub fn device(&self) -> Device {
        match self {
            BackendKind::Cpu => Device::flex(),
            #[cfg(feature = "wgpu")]
            BackendKind::Wgpu => Device::wgpu(Default::default()),
            #[cfg(feature = "metal")]
            BackendKind::Metal => Device::metal(Default::default()),
            #[cfg(feature = "cuda")]
            BackendKind::Cuda => Device::cuda(0),
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_available() {
        assert_eq!(BackendKind::from_name("cpu").unwrap(), BackendKind::Cpu);
        assert_eq!(BackendKind::from_name("CPU").unwrap(), BackendKind::Cpu);
        assert!(BackendKind::available().contains(&BackendKind::Cpu));
    }

    #[test]
    fn unknown_backend_is_rejected() {
        assert!(BackendKind::from_name("tpu").is_err());
    }
}
