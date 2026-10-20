//! A process's memory, in MB (10^6 bytes), from the operating system's counters

/// One reading of a process's memory; zeros where the platform has no such counter
#[derive(Debug, Clone, Copy, Default)]
pub struct Memory {
    /// Memory private to the process now: Windows' private bytes (commit charge), Linux's
    /// anonymous resident memory (`RssAnon`)
    pub private_mb: f64,
    /// The most private memory the process has had: Windows' peak commit charge. Linux has
    /// no such counter, so this is `private_mb` there: read it after the work, not during
    pub peak_private_mb: f64,
    /// The most physical memory the process has had, shared pages included: Windows' peak
    /// working set, Linux's `VmHWM`
    pub peak_working_set_mb: f64,
}

/// The memory of process `pid` now; zeros if it cannot be read (no such process, no access,
/// or a platform other than Windows and Linux)
#[cfg(windows)]
pub fn of_process(pid: u32) -> Memory {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_VM_READ,
    };

    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    // SAFETY: plain Win32 calls on a handle opened and closed here. The counters struct is
    // plain data, so all zeros is a valid value, and GetProcessMemoryInfo writes at most `cb`
    // bytes of it; it is read only when that call succeeds.
    let counters = unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid);
        if handle.is_null() {
            return Memory::default();
        }
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        counters.cb = size;
        let ok = GetProcessMemoryInfo(
            handle,
            (&mut counters as *mut PROCESS_MEMORY_COUNTERS_EX).cast::<PROCESS_MEMORY_COUNTERS>(),
            size,
        );
        CloseHandle(handle);
        if ok == 0 {
            return Memory::default();
        }
        counters
    };
    Memory {
        private_mb: counters.PrivateUsage as f64 / 1e6,
        peak_private_mb: counters.PeakPagefileUsage as f64 / 1e6,
        peak_working_set_mb: counters.PeakWorkingSetSize as f64 / 1e6,
    }
}

/// The memory of process `pid` now, from `/proc/PID/status`; zeros if it cannot be read
#[cfg(target_os = "linux")]
pub fn of_process(pid: u32) -> Memory {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status")).unwrap_or_default();
    let mb = |field: &str| {
        status
            .lines()
            .find_map(|l| l.strip_prefix(field))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|kb| kb.parse::<f64>().ok())
            .map_or(0.0, |kb| kb * 1024.0 / 1e6)
    };
    let private_mb = mb("RssAnon:");
    Memory {
        private_mb,
        peak_private_mb: private_mb,
        peak_working_set_mb: mb("VmHWM:"),
    }
}

/// No counters read on this platform
#[cfg(not(any(windows, target_os = "linux")))]
pub fn of_process(_pid: u32) -> Memory {
    Memory::default()
}
