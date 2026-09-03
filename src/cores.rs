//! Keep the GPU-driving thread from being throttled.
//!
//! The engine issues sixteen small kernels per frame and spends most of its
//! time waiting on the GPU. Windows reads that as a background workload and,
//! on a CPU with efficiency cores, moves it there; the launch loop then runs
//! about three times slower (measured 0.55x realtime against 1.5x). Opting the
//! process out of power throttling is enough to undo that — no core topology
//! is inspected and nothing is pinned, so the call is a no-op wherever the
//! scheduler was not doing this in the first place.

/// Tell the OS this process wants execution speed over power saving. Windows
/// only; elsewhere it does nothing.
pub fn prefer_performance_cores() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation,
            PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            PROCESS_POWER_THROTTLING_STATE,
        };
        let mut state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0,
        };
        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            &mut state as *mut _ as *mut _,
            std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        );
    }
}
