//! Loading Burnpack (`.bpk`) weight files.
//!
//! The file is memory-mapped read-only and its pages are handed to Burn as shared bytes. The
//! CPU backend (burn-flex) keeps tensor data in that buffer instead of copying it, so the
//! weights are backed by the page cache and do not count toward the process's private memory.
//! If mapping fails (for example on a filesystem without mmap support), the file is read into
//! memory instead.

use std::fs::File;
use std::path::Path;

use burn::tensor::{AllocationProperty, Bytes};
use burn_store::{BurnpackStore, ModuleSnapshot};
use memmap2::Mmap;
use tracing::warn;

use crate::error::{ModelError, Result};

/// Load Burnpack weights from `path` into `model`, reporting errors instead of panicking
/// (release builds use `panic = "abort"`).
pub fn load_burnpack<M: ModuleSnapshot>(model: &mut M, path: &Path) -> Result<()> {
    let failed = |e: Box<dyn std::error::Error + Send + Sync>| ModelError::LoadFailed {
        path: path.to_path_buf(),
        source: e,
    };
    // Opened here, so a missing or unreadable file is reported as such, not as a failed mapping
    let file = File::open(path).map_err(|e| failed(Box::new(e)))?;
    let mut store = match map_file(&file) {
        Ok(bytes) => BurnpackStore::from_bytes(Some(bytes)),
        Err(e) => {
            warn!(
                "Could not memory-map {}: {e}; reading it into memory instead",
                path.display()
            );
            BurnpackStore::from_file(path).auto_extension(false)
        }
    };
    model
        .load_from(&mut store)
        .map_err(|e| failed(Box::new(e)))?;
    Ok(())
}

/// Map `file` read-only and wrap the mapping as shared, zero-copy tensor bytes.
fn map_file(file: &File) -> std::io::Result<Bytes> {
    // SAFETY: `Mmap::map` is unsafe because the mapped memory changes if another process
    // writes to the file, which breaks the immutability `&[u8]` promises, and on Unix,
    // truncating it makes later reads fault with SIGBUS. Weight files are read-only deployment
    // artifacts: the server never writes them, and they must not be edited in place while it
    // runs. Replace them by renaming a new file over the old one, which leaves the existing
    // mapping intact. Windows also refuses to truncate or delete a file while it is mapped.
    // The Burnpack reader checks every tensor's offsets against the mapped length before
    // creating views, so a malformed file is reported as an error, not read out of bounds.
    let mmap = unsafe { Mmap::map(file)? };
    // The mapping owns its pages and outlives `file`; `Bytes::from_owner` keeps it alive for
    // as long as any tensor references it.
    let shared = bytes::Bytes::from_owner(mmap);
    Ok(Bytes::from_shared(shared, AllocationProperty::File))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::{Linear, LinearConfig};
    use burn::tensor::Device;

    fn linear() -> Linear {
        LinearConfig::new(4, 3).init(&Device::default())
    }

    #[test]
    fn round_trips_through_mmap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("linear.bpk");
        let original = linear();
        let mut store = BurnpackStore::from_file(&path).auto_extension(false);
        original.save_into(&mut store).unwrap();

        let mut loaded = linear();
        load_burnpack(&mut loaded, &path).unwrap();
        assert_eq!(
            loaded.weight.val().into_data(),
            original.weight.val().into_data()
        );
        // The mapping stays alive through the tensors after the store is gone.
        drop(store);
        let _ = loaded.weight.val().sum().into_data();
    }

    #[test]
    fn missing_file_is_an_error() {
        let mut model = linear();
        let err = load_burnpack(&mut model, Path::new("does-not-exist.bpk")).unwrap_err();
        assert!(err.to_string().contains("does-not-exist.bpk"), "{err}");
        // Reported as the failed open, before any attempt to map or read it
        let crate::error::FurnaceError::Model(ModelError::LoadFailed { source, .. }) = err else {
            panic!("{err:?}");
        };
        assert!(
            source.downcast_ref::<std::io::Error>().is_some(),
            "{source:?}"
        );
    }

    #[test]
    fn malformed_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.bpk");
        std::fs::write(&path, b"not a burnpack file").unwrap();
        let mut model = linear();
        assert!(load_burnpack(&mut model, &path).is_err());
    }
}
