pub use burn;
pub use error::TtsError;

pub type VulkanBackend = burn::backend::Vulkan<half::f16, i32>;
pub type VulkanEngine = engine::Engine<VulkanBackend>;

pub use burn::backend::wgpu::{MemoryConfiguration, WgpuDevice};

/// Where cubecl keeps compiled SPIR-V and autotune results between runs.
///
/// Without a cache every launch recompiles and re-tunes every kernel, which
/// measured 3.5 minutes on the 1.7B model; with one, later launches warm up in
/// under a second. The setting is process-global and write-once in cubecl, so
/// the library only touches it when asked.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum KernelCache {
    /// Leave cubecl's configuration alone: whatever `cubecl.toml`, `burn.toml`
    /// or the environment of the host process say. cubecl's own default keeps
    /// no compilation cache.
    #[default]
    Inherit,
    /// `<root>/vulkan` and `<root>/autotune` under the per-user local data
    /// directory (`%LOCALAPPDATA%` on Windows, `~/.config` on Linux).
    Global,
    /// `<dir>/vulkan` and `<dir>/autotune`.
    Dir(std::path::PathBuf),
}

/// How the engine is attached to the GPU and the process.
///
/// `Default` changes nothing outside the engine: no global cubecl
/// configuration, no memory-pool override, no process power policy. That is
/// the right default for a library, and it also means no kernel cache unless
/// the host configured one. [`EngineOptions::app`] is what a standalone
/// application wants and what [`load_vulkan`] uses.
#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub device: WgpuDevice,
    /// `Some` initialises the device with this pool before loading; `None`
    /// leaves initialisation to burn, which lets a host that already created
    /// the device keep it.
    pub memory: Option<MemoryConfiguration>,
    pub kernel_cache: KernelCache,
    /// Opt the process out of power throttling on Windows. The generation loop
    /// waits on the GPU most of the time, which the scheduler reads as
    /// background work and moves to efficiency cores, where it ran three times
    /// slower. Process-wide; off by default.
    pub power_hint: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            device: WgpuDevice::default(),
            memory: None,
            kernel_cache: KernelCache::Inherit,
            power_hint: false,
        }
    }
}

impl EngineOptions {
    /// Global kernel cache, exclusive-pages memory pool (exact allocations that
    /// are released, instead of slabs that are kept — about 1 GB less resident
    /// on this model), and the power hint.
    pub fn app() -> Self {
        Self {
            device: WgpuDevice::default(),
            memory: Some(MemoryConfiguration::ExclusivePages),
            kernel_cache: KernelCache::Global,
            power_hint: true,
        }
    }

    pub fn device(mut self, device: WgpuDevice) -> Self {
        self.device = device;
        self
    }

    pub fn memory(mut self, memory: MemoryConfiguration) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn kernel_cache(mut self, cache: KernelCache) -> Self {
        self.kernel_cache = cache;
        self
    }

    pub fn power_hint(mut self, on: bool) -> Self {
        self.power_hint = on;
        self
    }
}

fn configure_kernel_cache(cache: &KernelCache) -> Result<(), TtsError> {
    use cubecl::config::cache::CacheConfig;
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    use std::sync::Mutex;

    static APPLIED: Mutex<Option<KernelCache>> = Mutex::new(None);
    let mut applied = APPLIED.lock().unwrap_or_else(|e| e.into_inner());
    match &*applied {
        Some(prev) if prev == cache => return Ok(()),
        Some(prev) => {
            return Err(TtsError::InvalidConfig(format!(
                "kernel cache already configured as {prev:?} in this process"
            )))
        }
        None => {}
    }
    let to = |c: &KernelCache| match c {
        KernelCache::Inherit => unreachable!(),
        KernelCache::Global => CacheConfig::Global,
        KernelCache::Dir(p) => CacheConfig::File(p.clone()),
    };
    let mut cfg = CubeClRuntimeConfig::default();
    cfg.compilation.cache = Some(to(cache));
    cfg.autotune.cache = to(cache);
    // cubecl panics if its configuration was already read, which any earlier
    // GPU work in the process will have done.
    std::panic::catch_unwind(|| CubeClRuntimeConfig::set(cfg)).map_err(|_| {
        TtsError::InvalidConfig(
            "cubecl configuration was already initialised; set the kernel cache before any GPU work"
                .into(),
        )
    })?;
    *applied = Some(cache.clone());
    Ok(())
}

fn init_device(device: &WgpuDevice, memory: MemoryConfiguration) -> Result<(), TtsError> {
    use burn::backend::wgpu::graphics::Vulkan as VulkanApi;
    use burn::backend::wgpu::{init_setup, RuntimeOptions};
    use std::collections::HashSet;
    use std::sync::Mutex;

    static DONE: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut done = DONE.lock().unwrap_or_else(|e| e.into_inner());
    let set = done.get_or_insert_with(HashSet::new);
    let key = format!("{device:?}");
    if set.contains(&key) {
        return Ok(());
    }
    if matches!(device, WgpuDevice::Existing(_)) {
        return Err(TtsError::InvalidConfig(
            "an existing wgpu device cannot be given a memory configuration; leave `memory` unset"
                .into(),
        ));
    }
    let opts = RuntimeOptions {
        memory_config: memory,
        ..Default::default()
    };
    std::panic::catch_unwind(|| init_setup::<VulkanApi>(device, opts)).map_err(|_| {
        TtsError::Gpu(
            "no usable Vulkan device, or the device was already initialised by the host".into(),
        )
    })?;
    set.insert(key);
    Ok(())
}

/// Load the engine with explicit options.
pub fn load_vulkan_with(model_dir: &str, opts: EngineOptions) -> Result<VulkanEngine, TtsError> {
    engine::Engine::<VulkanBackend>::check_model_dir(model_dir)?;
    if opts.kernel_cache != KernelCache::Inherit {
        configure_kernel_cache(&opts.kernel_cache)?;
    }
    if opts.power_hint {
        cores::prefer_performance_cores();
    }
    if let Some(memory) = opts.memory {
        init_device(&opts.device, memory)?;
    }
    engine::Engine::load(model_dir, opts.device)
}

/// Load the engine the way a standalone application wants it:
/// [`EngineOptions::app`]. This configures a process-global kernel cache and,
/// on Windows, the process power policy; a library embedding the engine in a
/// larger program should call [`load_vulkan_with`] and choose.
pub fn load_vulkan(model_dir: &str) -> Result<VulkanEngine, TtsError> {
    load_vulkan_with(model_dir, EngineOptions::app())
}

pub mod audio;
pub mod code_predictor;
pub mod cores;
pub mod decoder;
pub mod deesser;
pub mod encoder;
pub mod engine;
pub mod error;
pub mod lang;
pub mod lowpass;
pub mod mel;
pub mod pipeline;
pub mod postproc;
pub mod resample;
pub mod sampling;
pub mod speaker;
pub mod talker;
pub mod tokenizer;
pub mod weights;
