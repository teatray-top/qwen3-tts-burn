use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum TtsError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    ModelFileMissing {
        path: PathBuf,
    },
    BadSafetensors {
        path: PathBuf,
        detail: String,
    },
    MissingTensor {
        name: String,
    },
    TensorShape {
        name: String,
        expected: String,
        got: Vec<usize>,
    },
    TensorDtype {
        name: String,
        dtype: String,
    },
    Tokenizer(String),
    Gpu(String),
    BadWav {
        path: PathBuf,
        detail: String,
    },
    UnsupportedWav {
        path: PathBuf,
        detail: String,
    },
    BadSampleRate {
        path: PathBuf,
        rate: u32,
    },
    ReferenceTooShort {
        samples_ms: f64,
        min_ms: f64,
    },
    ReferenceTooLong {
        frames: usize,
        max: usize,
    },
    EmptyText,
    EmptyReferenceText,
    InvalidPrompt(String),
    InvalidFrames(String),
    InvalidConfig(String),
    Numeric(String),
}

// `resample` reports a bad rate without a file to name, so an empty path prints nothing.
fn at(path: &Path) -> String {
    if path.as_os_str().is_empty() {
        String::new()
    } else {
        format!("{}: ", path.display())
    }
}

impl fmt::Display for TtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{}{source}", at(path)),
            Self::ModelFileMissing { path } => write!(f, "{}model file not found", at(path)),
            Self::BadSafetensors { path, detail } => {
                write!(f, "{}not a valid safetensors file ({detail})", at(path))
            }
            Self::MissingTensor { name } => write!(f, "tensor {name} is missing from the weights"),
            Self::TensorShape {
                name,
                expected,
                got,
            } => {
                write!(f, "tensor {name} has shape {got:?}, expected {expected}")
            }
            Self::TensorDtype { name, dtype } => {
                write!(f, "tensor {name} has unsupported dtype {dtype}")
            }
            Self::Tokenizer(detail) => write!(f, "tokenizer: {detail}"),
            Self::Gpu(detail) => write!(f, "gpu: {detail}"),
            Self::BadWav { path, detail } => write!(f, "{}bad wav file ({detail})", at(path)),
            Self::UnsupportedWav { path, detail } => {
                write!(f, "{}unsupported wav format ({detail})", at(path))
            }
            Self::BadSampleRate { path, rate } => {
                write!(f, "{}sample rate {rate} Hz is not usable", at(path))
            }
            Self::ReferenceTooShort { samples_ms, min_ms } => write!(
                f,
                "reference audio is too short: {samples_ms:.1} ms, at least {min_ms:.1} ms needed"
            ),
            Self::ReferenceTooLong { frames, max } => {
                write!(
                    f,
                    "reference audio is too long: {frames} frames, at most {max} allowed"
                )
            }
            Self::EmptyText => write!(f, "text is empty"),
            Self::EmptyReferenceText => write!(f, "reference text is empty"),
            Self::InvalidPrompt(detail) => write!(f, "invalid prompt: {detail}"),
            Self::InvalidFrames(detail) => write!(f, "invalid frames: {detail}"),
            Self::InvalidConfig(detail) => write!(f, "invalid config: {detail}"),
            Self::Numeric(detail) => write!(f, "numeric error: {detail}"),
        }
    }
}

impl std::error::Error for TtsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<TtsError> for String {
    fn from(e: TtsError) -> String {
        e.to_string()
    }
}
