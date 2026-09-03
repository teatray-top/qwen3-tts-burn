# qwen3-tts-burn

A Vulkan port of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for people
without CUDA.

It is built natively on [burn](https://burn.dev)'s wgpu backend: any GPU with
a Vulkan driver will run it, and the build needs nothing beyond `cargo build`.
No CUDA toolkit, no Python, no native libraries.

The model is reimplemented in Rust and loads the official `safetensors` weights
directly.

## Features

- Talker, code predictor, and codec decoder for the 12 Hz tokenizer.
- Voice cloning from a reference clip, by speaker vector (ECAPA) or by
  in-context learning from a clip and its transcript.
- The model's ten languages, or `Auto` as the official default.
- Streaming synthesis for low-latency output.
- The official sampling configuration by default; the relay app's heuristics
  and audio filters as explicit, documented switches.

The supported checkpoint is Qwen3-TTS-12Hz-1.7B-Base on the 12 Hz tokenizer;
"Model coverage" below says what the others would need.

## Install

```toml
[dependencies]
qwen3-tts-burn = { git = "https://github.com/teatray-top/qwen3-tts-burn" }
```

## Get the weights

Not bundled. Download from Hugging Face and lay them out like this:

```
<model-dir>/
  model.safetensors            # Qwen/Qwen3-TTS-12Hz-1.7B-Base
  config.json
  vocab.json                   # text tokenizer, from the same repository
  merges.txt
  tokenizer_config.json
  speech_tokenizer/
    model.safetensors          # Qwen/Qwen3-TTS-Tokenizer-12Hz
    config.json
```

## Synthesize from Rust

```rust
let engine = qwen3_tts_burn::load_vulkan("path/to/model")?;
engine.warmup()?;

use qwen3_tts_burn::engine::PostProcess;
use qwen3_tts_burn::lang::Language;

let prompt = engine.build_clone_prompt(
    "reference.wav",                   // speaker vector
    "reference.wav",                   // in-context example, may be another file
    "what the reference clip says",    // its transcript, verbatim
    Language::English,                 // language of the text to be spoken
)?;

// The model as published: the official sampling config, the text as given,
// the official stopping rule, the decoder's output untouched.
let cfg = qwen3_tts_burn::sampling::SamplerCfg::default();
let audio = engine.synthesize("the line to speak", &prompt, cfg, 400)?;
qwen3_tts_burn::audio::write_wav_24k("out.wav", &audio)?;   // 24 kHz mono

// The relay app's tuning, every switch of which is documented on the type:
// a colder sampler with greedy acoustic codes, a comma in place of the final
// punctuation, end-of-speech held until the text is spoken, an attention bias
// toward the reference, and trims, a low-pass and a de-esser on the audio.
let audio = engine.synthesize_with(
    "the line to speak", &prompt, SamplerCfg::app(), 400, PostProcess::app_default(),
)?;
```

Every function returns `Result<_, TtsError>`; a missing model file, a WAV the
loader cannot read, a reference clip too short for the speaker encoder, an
empty line or a hand-built prompt with the wrong shape come back as an error
with the path or the reason in it, not as a panic.

`load_vulkan` is the standalone-application loader: it turns on a per-user
kernel cache, uses an exclusive-pages memory pool and, on Windows, opts the
process out of power throttling — all of them process-wide settings. A program
that embeds the engine should call `load_vulkan_with(dir, EngineOptions::default())`,
which touches nothing outside the engine, and turn on what it wants:

```rust
use qwen3_tts_burn::{EngineOptions, KernelCache};
let engine = qwen3_tts_burn::load_vulkan_with(
    "path/to/model",
    EngineOptions::default().kernel_cache(KernelCache::Dir("cache".into())),
)?;
```

## Synthesize from the command line

```
cargo run --release --example synthesize -- \
    <model-dir> <reference.wav> "<reference transcript>" "<text>" [lang] [out.wav]     [--post] [--profile official|app|greedy] [--xvector] [--seed N] [--temp T]
```

Without flags this is the official configuration. `--post` adds the relay
app's heuristics and filters, `--profile app` its sampler, `--xvector` drops
the in-context example and clones from the speaker vector alone.

## Defaults and switches

The defaults are the published model. `SamplerCfg::default()` is the model's
`generation_config.json`: talker at temperature 0.9, top-k 50, no nucleus cut,
repetition penalty 1.05 over the utterance, and the fifteen acoustic codes
sampled at the same temperature and top-k. `Language::default()` is `Auto`.
`synthesize` feeds the text as given and stops at the first end-of-speech
token. The prompt — role header, speaker vector, language token, in-context
reference — is built as the official implementation builds it, and generated
frames are decoded with the reference frames in front.

Two named profiles turn that into what the relay app this grew out of uses,
and each switch says what it costs on the type that carries it:

| | |
| --- | --- |
| `SamplerCfg::app()` | colder (0.7), nucleus-cut (0.9), penalty 1.1, redraws a token seen in the last ten, argmax on the acoustic codes |
| `PostProcess::app_default()` | comma ending, held end-of-speech, attention bias toward the reference, trims, 10.5 kHz low-pass, de-esser |

`synthesize_streaming` and `synthesize_speak` take a `PostProcess` too, so the
streaming and one-shot paths run the published model on `PostProcess::none()`.
The two rules that exist to cover a held end-of-speech token follow the switch
that creates the need: the silent-tail cutoff applies with `hold_eos`, and the
end fade with `trailing_trim`, since it is the trim that cuts mid-waveform.

Fed the official frames on the official inputs, the talker's f16 logits pick
the same top token as the official f32 logits at all 87 steps, cosine 0.9999 or
better. `examples/parity.rs` runs the check against the data in `eval/parity/`;
BENCHMARKS.md has the tables and the method.

## Notes

The transcript given to `build_clone_prompt` must match the reference audio.
The model sees that pair as an example before generating, so a mismatch
degrades the output.

`warmup()` compiles and tunes the kernels up front. The first launch on a
machine takes about three and a half minutes end to end (kernel compilation,
then autotuning for the prompt and the first sentence); the results are cached
on disk and every later launch warms up in under a second. Skipping it moves
that cost into the first sentence. This is the cost of the Vulkan path: there
is no vendor compiler cache to lean on, so a short-lived or serverless process
should ship the cache directory with it.

Weights run in `f16`. The 1.7B model holds about 3.7 GB of VRAM after
loading and about 4.8 GB while speaking, measured per process on Windows;
the numbers and how they were taken are in `BENCHMARKS.md`.

A tensor shape the runtime has not seen before triggers autotune, so widely
varying text lengths cost more than steady ones.

## License

MIT. Model weights are licensed separately by their authors.

## Numbers

RTX 5070 Ti, 1.7B Base with an in-context reference, warm process, ten lines
per language. Full tables, method, raw rows and the comparison with other
implementations are in [BENCHMARKS.md](BENCHMARKS.md).

| | |
| --- | --- |
| speed, app profile | 43 ms per generated frame plus 2.7 ms per reference frame: 1.7–1.8x realtime for a 3–16 s line from the 7.6 s reference, 1.84x asymptotically (the function and the fit are in BENCHMARKS.md; `examples/fit_speed.rs` reproduces it from the committed grid) |
| speed, official profile | about 1.3x: sampling the fifteen acoustic codes reads their logits back to the host each frame |
| first audio, streaming | 405–430 ms |
| WER / CER, 10 lines each | 0.9 % English, 1.7 % Korean; in these lines every miss is a spelling or numeral form in the transcript |
| speaker similarity to the reference | 0.90 (the reference to itself: 0.85–0.90) |
| VRAM, process | 3.7 GB after load, 4.8 GB while speaking |
| first launch on a machine | 3.5 min of kernel compilation and tuning, then under a second |

On the same card, llama.cpp's CUDA build measured 4.55x and the other Vulkan
implementation 1.10x. On other cards, faster-qwen3-tts publishes 4.22x for this
model on an RTX 4090 and 1.83x on an RTX 4060 with CUDA graphs.

## Model coverage

The only supported model is Qwen3-TTS-12Hz-1.7B-Base with the 12 Hz tokenizer:
voice cloning from a reference clip, with or without the in-context example, in
the model's ten languages (`lang.rs`: Chinese, English, German, Italian,
Portuguese, Spanish, Japanese, Korean, French, Russian).

What each of the other checkpoints would need, from reading the official
implementation against this one:

- **0.6B Base** — the code predictor's `small_to_mtp_projection` is absent from
  that checkpoint (it is an identity there) and the loader requires it; make it
  optional and the rest is the same shapes. Small.
- **CustomVoice** — a preset speaker is a row of the talker's codec embedding
  rather than an ECAPA vector, plus an optional instruct prefix. There is no
  API for it here; the checkpoint's speaker table is not read and the prompt
  is not built. Small to add.
- **VoiceDesign** — the instruct prefix with no speaker slot. Small, shares the
  CustomVoice work.
- **25 Hz tokenizer** — a different codec (flow-matching DiT decoder and a
  BigVGAN vocoder, one code per frame), so a second codec port; and its
  weights are not publicly downloadable at the time of writing. Large.

## Tests

`cargo test --release` runs CPU-only tests of the signal-processing and text
helpers; no model, no GPU. The golden test pins the model's output:

```
QTB_MODEL_DIR=path/to/model cargo test --release --test golden -- --ignored --test-threads=1
QTB_MODEL_DIR=path/to/model cargo test --release --test streaming -- --ignored
```

The first pins the codec frames the sampler draws for a fixed seed, once for
the official profile and once for the app profile (`tests/golden/*.frames`),
and checks that bad input comes back as an error. The second checks that the
streaming path hands back its first chunk well before the utterance is done.
The pinned frames hold for a given GPU and driver; on other hardware, delete
them and let the test rewrite them. CI builds, runs the CPU tests, and holds
the tree to `cargo fmt` and `clippy -D warnings`; it has no GPU.

## Samples

**[Listen](https://teatray-top.github.io/qwen3-tts-burn/)** — a page with the
clips in players (GitHub does not play audio inside a README).

| | |
| --- | --- |
| [English](samples/sample_en.wav?raw=1) | The opening of this README, generated one paragraph at a time from the 7.6 s reference. `samples/sample_en.txt` is the spoken form it was fed: acronyms spelled out, "W G P U", "safe tensors". |
| [Korean](samples/sample_ko.wav?raw=1) | The same opening translated into Korean, one paragraph at a time, from a 7.7 s KSS reference (CC BY-NC-SA; see `samples/LICENSES.md`). `samples/sample_ko.txt` is its spoken form: "큐원 쓰리 티티에스", "더블유지피유". |
| [English reference](samples/reference_en.wav?raw=1) · [Korean reference](samples/reference_ko_kss.wav?raw=1) | The only inputs besides the text. |

The English reference is from
[People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech)
(`clean` split, CC BY), chosen for signal-to-noise ratio, length and plain
content; the Korean
demo uses a clip from [KSS](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset),
a professional single-speaker corpus that is CC BY-NC-SA, so the three
KSS-derived files carry that licence rather than MIT (`samples/LICENSES.md`).
The Korean benchmark numbers use a CC BY clip from
[Zeroth-Korean](https://huggingface.co/datasets/Bingsu/zeroth-korean) instead
(`samples/reference_ko.wav`, chosen the same way plus a transcript check by
speech recognition), so that they can be reproduced without the restriction.
Long passages flatten the delivery and single sentences make the seams
audible, so the samples are generated one paragraph at a time (`eval_batch`
with `keep-tail`, seed 77) and joined with 500 ms of silence.

Speed and memory figures, and how they were measured, are in
[BENCHMARKS.md](BENCHMARKS.md).
