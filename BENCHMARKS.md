# Benchmarks

RTX 5070 Ti (16 GB), Windows 11. Every implementation generates once to compile
kernels, capture graphs or fill caches; only the runs after that are timed.
Speed is audio seconds produced per second of wall clock, so higher is faster.

The engine rows come from the examples under `examples/`; the other
implementations were run from their own command lines with the settings stated
in each section.

## Quality

Ten short lines per language (3–5 s each, `eval/lines_*.txt`), 1.7B Base with
an in-context reference, seed 77, the app's post-processing. Transcribed with
faster-whisper `medium` (int8, CPU); speaker similarity is the cosine between
resemblyzer embeddings of the output and of the reference clip.

| | English | Korean |
| --- | --- | --- |
| WER / CER | 0.9 % WER | 1.7 % CER |
| speaker similarity, mean (min) | 0.903 (0.854) | 0.886 (0.858) |
| same speaker, two halves of the reference | 0.901 | 0.912 |
| different speaker (the other language's reference) | 0.501 | 0.501 |

Every counted error is the recogniser's spelling, not the speech: the one
English miss is "neighbours" written as "neighbors", and the Korean misses are
spoken numerals written as digits ("일곱 시" as "7시", "두 개" as "2개"). The
output is as close to the reference speaker as the reference is to itself.

With the language left to the model (`Language::Auto`, the official
default, which sends the three-token "no think" prefix instead of a language
id) the same ten lines score 0.9 % WER / similarity 0.907 in English and
2.4 % CER / 0.881 in Korean — one more Korean line has a numeral written as a
digit by the recogniser. Naming the language is the safer setting for Korean.

The references are the two clips in `samples/`: People's Speech for English
and Zeroth-Korean for Korean, both CC BY. An earlier version of this table used
a KSS clip for Korean; KSS is CC BY-NC-SA and cannot ship here, so those rows
were retired and re-measured with the Zeroth clip. Sample size is what it
looks like: ten lines per language on one GPU, three timed runs where a time
is quoted. Speed is not in this table because it is a function of reference
and output length, given below.

For scale, the official evaluation reports 1.24 % WER (Seed-TTS test-en) and
0.775 speaker similarity for the same 1.7B Base model, on different test sets
and a different similarity model, so the two are not directly comparable.

## Speed as a function

A speed figure for this engine is a function of two lengths, so here is the
function rather than a number. Per generated frame the talker runs one step
and the code predictor fifteen; before that the in-context reference is
prefilled; afterwards the codec decodes reference and output together, padded
to a power of two. Fitted on a grid of five reference lengths and four output
lengths in both languages (`examples/speed_grid.rs`; raw rows in
`eval/results/speed_grid_*.tsv`; `examples/fit_speed.rs` does the least-squares
fit and wrote `eval/results/fit_speed.txt`; app profile, one warm-up per
prompt):

```
t_gen(R, N) = 2.73 ms · R + 43.4 ms · N          median error 6 % over 40 rows
t_dec(R, N) = 0.24 ms · pow2(R + N)               30 / 70 / 145 ms at 128 / 256 / 512
speed       = 80 ms · N / (t_gen + t_dec)          audio seconds per wall second
```

R is the reference clip in codec frames (12.5 per second), N the generated
frames. The intercept fitted to −0.14 s, i.e. there is no fixed per-utterance
cost worth naming; the reference is paid for once per utterance at 2.7 ms a
frame, and the decoder is a rounding error. The per-frame cost does not depend
on the KV-cache bucket (42.8 / 43.4 / 43.3 ms in the 448 / 1024 / 2048
buckets), which says the talker step is bound by kernel launches, not by
attention width: sixteen small launches a frame on a GPU that sits mostly idle.

| reference | line | audio | predicted | speed |
| --- | --- | --- | --- | --- |
| 2 s (25 fr) | 40 fr | 3.2 s | 1.69 s | 1.90x |
| 7.6 s (95 fr) | 44 fr | 3.5 s | 2.08 s | 1.69x |
| 7.6 s (95 fr) | 100 fr | 8.0 s | 4.51 s | 1.77x |
| 7.6 s (95 fr) | 200 fr | 16.0 s | 8.91 s | 1.80x |
| 15 s (188 fr) | 60 fr | 4.8 s | 3.03 s | 1.58x |

The asymptote is 80 / 43.4 = 1.84x. Anything quoted as "1.6x" or "1.9x"
elsewhere in this file is a point on this curve.

## Time to first audio

`examples/ttfa.rs`, streaming path, warm process. The lead buffer is the
relay app's guard against playback overtaking generation.

| | English (3.4 s line, 95-frame reference) | Korean (4.5 s line, 87-frame reference) |
| --- | --- | --- |
| first chunk, no lead buffer | 406–420 ms | 568 ms |
| first chunk, app lead buffer | 417–430 ms | 574 ms |

The English app-lead figure was 6.5 s — the whole utterance — before this
commit. The lead is sized from the generation rate, and that rate was measured
from process start, so the prefill was folded into it and the onset looked
several times slower than realtime. It is now measured from the first frame.

## Parity with the official implementation

Two measurements, both in `examples/parity.rs`, both on the shipped English
reference and the sentence "The quick brown fox jumps over the lazy dog.".
The official side is the `qwen_tts` package on CUDA, once in bf16 and once in
f32 (the checkpoint is bf16, so f32 is the same weights upcast):
`create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)`,
then `model.generate(..., languages=["English"], non_streaming_mode=False,
do_sample=False, subtalker_dosample=False, repetition_penalty=1.05,
max_new_tokens=400)`, with the talker's raw logits recorded at every step.
Its reference codes and speaker vector were saved once and fed back to both
dtypes, so everything under `eval/parity/` — frames, logits, inputs — comes
from the same inputs, and the engine can be given exactly those.

**Logits on the same history.** The engine is fed the official frame sequence
(teacher forcing, official inputs) and its talker logits are compared with the
official logits at each step:

| official inputs, official frames fed back | steps | top-1 agrees | cosine mean (min) |
| --- | --- | --- | --- |
| engine f16 vs official f32, along the f32 path | 48 | 48 | 1.0000 (1.0000) |
| engine f16 vs official bf16, along the bf16 path | 39 | 39 | 1.0000 (0.9999) |
| official bf16 vs official f32, step 0 | 1 | 1 | 1.0000 |
| engine f16 on its own reference codes and speaker vector vs official f32, step 0 | 1 | 1 | 0.9919 |

The talker — the 28-layer model that picks the semantic token — reproduces the
official logits: the same top token at all 87 steps, cosine 0.9999 or better.
The last row is what the two encoders add: the engine's own reference codes
and speaker vector differ slightly from the official ones, and that moves the
first step's logits by more than the whole talker does. The example exits
non-zero if, at the first step, the engine's f16 logits sit further from the
official f32 logits than the official bf16 logits do.

**Greedy frames**, the same thing measured the naive way — each side runs its
own greedy loop and the sequences are compared:

| | frames | first difference | semantic column agrees | whole frames agree |
| --- | --- | --- | --- | --- |
| official bf16 vs official f32 | 38 vs 47 | frame 0 | 10.5 % | 0.0 % |
| engine f16, own reference codes and speaker vector, vs official bf16 | 40 vs 38 | frame 0 | 10.5 % | 0.0 % |
| engine f16, own inputs, vs official f32 | 40 vs 47 | frame 0 | 5.0 % | 0.0 % |
| engine f16, official reference codes and speaker vector, vs official bf16 | 38 vs 38 | frame 0 | 15.8 % | 0.0 % |
| engine f16, official inputs, vs official f32 | 38 vs 47 | frame 1 | 23.7 % | 2.6 % |

The first row sets the scale. The official code disagrees with itself between
two dtypes from the first frame on, even though its first-step logits agree to
four decimals: a frame also carries fifteen acoustic codes from the code
predictor's own greedy chain, and once one code differs every later frame is
conditioned on a different history. Frame agreement therefore measures how
early the first near-tie lands, not how close two implementations are, which
is why the logit table above is the one the gate reads. The code predictor is
not compared at the logit level; that is the one component the same-history
check does not cover. The engine's frame sequences are written next to the
official ones on every run.

## Cold start

Kernel cache (`%LOCALAPPDATA%\vulkan`) and autotune cache
(`%LOCALAPPDATA%\autotune`) removed, then one English line:

| | first launch | next launch |
| --- | --- | --- |
| load | 6.3 s | 6.4 s |
| warmup | 160.8 s | 0.6 s |
| first prompt | 28.4 s | 0.4 s |
| first sentence (3.4 s audio) | 12.4 s | 2.0 s |
| total | 209 s | 9.4 s |

There is no vendor shader cache to fall back on, so a process that cannot keep
those two directories between runs pays three and a half minutes every time.

## Published numbers elsewhere

What other Qwen3-TTS implementations state in their own READMEs, as of
2026-09-03, with each project's definition converted to audio seconds per
wall-clock second where the definition is unambiguous. Hardware differs in
every row, so this places the Vulkan path rather than ranks it.

| Project | Backend | Hardware | Model | Speed | First audio | Memory |
| --- | --- | --- | --- | --- | --- | --- |
| this engine | burn/wgpu Vulkan, f16 | RTX 5070 Ti | 1.7B Base, ICL | see the function below | 406–574 ms | 4.8 GB process |
| [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) | PyTorch, CUDA graphs | RTX 4090 / 4060 / T4 | 1.7B Base, ICL | 4.22x / 1.83x / 0.93x | 174 / 460 / 1096 ms | not published |
| [cgisky1980/Qwen3-TTS-Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust) | llama.cpp Vulkan + ONNX | RTX 2080 Ti | Q5_K_M, ~2.2 s clips | 1.67x Vulkan, 1.81x CUDA | "as low as 300 ms" | 0.7–1.5 GB |
| [TrevorS/qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) | Candle CUDA bf16 + FA2 | DGX Spark (GB10) | 1.7B Base | 1.54x | ~580 ms | 767 MB (unspecified) |
| [alfonsodg/concurrent-faster-qwen3-server](https://github.com/alfonsodg/concurrent-faster-qwen3-server) | Rust + CUDA, batched | L40S | 1.7B Base, clone | 1.61x single stream | ~230–325 ms | 5.2 GB |
| [luka-loehr/qwen3-tts-native](https://github.com/luka-loehr/qwen3-tts-native) | native Rust + CUDA | DGX Spark | 1.7B VoiceDesign | 1.25x single stream | 94 ms p95 | 5.68 GB |
| [gabriele-mastrapasqua/qwen3-tts](https://github.com/gabriele-mastrapasqua/qwen3-tts) | pure C, CUDA / Metal | RTX 4060-class / M4 GPU | 1.7B quant-mixed / int4 | 2.27x / 2.44x | 517 ms (M2 Pro) | not published |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | C++ ggml, CUDA | RTX 5090 | "qwen3 tts", variant unstated | 2.56x | not published | not published |
| [second-state/qwen3_tts_rs](https://github.com/second-state/qwen3_tts_rs) | MLX | Apple M4 | 1.7B CustomVoice | 0.31–0.34x | not published | not published |
| [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | PyTorch | unstated | 1.7B Base | not published | "as low as 97 ms" | not published |

Projects named in reviews that publish no Qwen3-TTS numbers at all:
[Crane](https://github.com/lucasjinreal/Crane) (Candle; its speed table is
Qwen2.5-500M tokens/s), danielclough/qwen3-tts-rs, HeiSir2014/qwen3-tts-candle.
CPU-only ports exist ([franken_tts](https://github.com/Dicklesworthstone/franken_tts)
1.4–1.6x on an M4 Pro for 0.6B, darkautism/qwen3-tts ~0.35x on an RK3588) and
are not in the table because they answer a different question.

Two cautions on reading the table. "RTF" means the opposite thing in different
READMEs — wall/audio (lower is better) for cgisky, TrevorS, second-state and
gabriele-mastrapasqua; audio/wall (higher is better) for faster-qwen3-tts — and
the table converts all of them to audio/wall. And cgisky's README does not name
its model; the talker GGUF it downloads is 2.84 GB in f16, which is the 1.7B
family, not 0.6B.

The only rows on a Vulkan device are this engine and cgisky1980's build, and the
measured head-to-head between those two is in the next section. The CUDA
figures above come from other cards; the same-card comparison is the next
table, where the two CUDA implementations run 1.7 to 2.7 times faster than this
engine.

## Against other implementations

Same sentence, same reference clip, voice cloning in every case.

| Implementation | Backend | Speed | Peak process VRAM |
| --- | --- | --- | --- |
| qwen3-tts-burn, app profile (greedy acoustic codes) | Vulkan throughout | 1.66–1.75x | 4.8 GB |
| qwen3-tts-burn, official profile (sampled acoustic codes) | Vulkan throughout | 1.27x | 4.8 GB |
| [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts), in-context clone | PyTorch, CUDA graphs | 2.89–3.13x | — |
| llama.cpp b10762 `llama-tts`, `--tts-speaker-file` | CUDA | 4.47–4.55x | 8.9 GB |

The two engine rows differ by one thing: the official profile samples the
fifteen acoustic codes, and this engine draws each of them on the host, which
costs a GPU→CPU readback per code — fifteen per frame on a loop that is
already launch-bound. Sampling on the device would remove that; the app
profile takes the argmax on the device and pays no readback. Ranges are
min–max over three warm runs of one 6.6 s line, on one GPU.

Both of the other two are CUDA-only, so this is not a fair contest — CUDA on an
NVIDIA card is expected to win, and neither of them runs on anything else. The
llama.cpp row clones from the same reference clip through
`--tts-speaker-file` (speaker vector only; its CLI takes no transcript), and
the engine is shown both with the in-context example and speaker-vector only
so that row has a like-for-like partner.

## Against the other Vulkan implementation

[cgisky1980/Qwen3-TTS-Rust](https://github.com/cgisky1980/Qwen3-TTS-Rust) is the
like-for-like comparison: the other project that reaches a Vulkan device. It
wraps llama.cpp for the talker and predictor and runs the codec decoder on ONNX
Runtime, so only part of its pipeline is on the GPU.

Matched as closely as the two allow — same sentence, speaker vector only with no
in-context prompt on either side, both processes on the performance cores, three
timed runs after a warm-up:

| | Speed | Peak process VRAM |
| --- | --- | --- |
| qwen3-tts-burn, Vulkan throughout | 1.76x (1.76–1.76) | 5657 MB |
| cgisky1980 v0.1.6, llama.cpp Vulkan + ONNX CPU | 1.10x (1.08–1.11) | 1594 MB |

Both processes were held on the CPU's performance cores for this table; see
the next section for why that matters and what the engine now does about it.

The other build uses a q5_k_m talker against this engine's f16, which is most of
the memory difference.

Three things cost time before that table was trustworthy, all worth knowing:

- Windows ships its own `onnxruntime.dll` (1.17.1) in `system32`, and it is
  found before the one the release bundles. ONNX Runtime then loads, but the
  codec decoder produces noise rather than speech — in every language, which is
  easy to misread as the model failing. Setting `ORT_DYLIB_PATH` to the bundled
  DLL fixes it. With it set, that build reads English, Japanese and Chinese back
  exactly.
- The v0.1.6 release works; the repository head at the time of writing (0.1.7)
  does not — it runs to the step limit and emits unrelated speech. Its
  `lang_id` is also fixed at 2055 there, with the comment "hardcoded for now or
  parameterize later" (`src/tts/engine.rs:328`); the release does not have that
  limitation.
- Its decoder is on the CPU, so it is even more exposed to core placement than
  this engine; unpinned and at idle priority its timings swung by 4x.

One quality difference showed up while checking that both builds actually say
what they are given. Transcribed with the same recogniser, this engine returns
"마지막 글자까지 잘 들리나요?" in full, while v0.1.6 drops the final syllable
and returns "마지막 글자까지 잘 들리나". Both handle a declarative ending
correctly. That build uses no reference clip, so it is not the cause described
in `postproc::speech_bounds`.

llama.cpp b10762's Vulkan backend cannot run this model at all — it aborts in
`ggml-vulkan.cpp:12105` on a `GET_ROWS` assertion, with bf16 and with Q8_0
alike. cgisky1980's build pins llama.cpp vB7885, where it works, so this is a
regression rather than a standing limitation.

## Core placement on hybrid CPUs

Throughout the measurements above, identical runs came out at either ~0.55x or
~1.6x realtime with nothing in between, and process priority made no
difference. The cause is where Windows puts the thread. The engine spends most
of its time waiting on the GPU (utilisation sits around 20% at full clocks),
which the scheduler reads as background work; on a CPU with efficiency cores
it moves the thread there, and the sixteen kernel launches per frame then run
about three times slower. Measured on an i7-13700K, same binary and sentence:

| Placement | Speed |
| --- | --- |
| pinned to performance cores | 1.64 / 1.36 / 1.64x |
| pinned to efficiency cores | 0.54 / 0.56 / 0.56x |
| left to the scheduler | 0.55 / 0.57 / 0.57x |

`load_vulkan` (and `EngineOptions::power_hint`) opts the process out of
power throttling (`SetProcessInformation(ProcessPowerThrottling)` with
execution-speed throttling disabled). That single call, with no core pinning
and no topology detection, brings the scheduler's default back to 1.56–1.64x;
with it in place the process priority class no longer matters either (normal,
below-normal and idle all measured 1.61–1.75x on the same line). It is Windows
only; the equivalent hint on Linux is utilisation clamping through
`sched_setattr`, which is not implemented and could not be measured here. The
same risk exists on big.LITTLE ARM under an energy-aware scheduler.

This is a mitigation. The condition it mitigates — a launch loop that leaves the
GPU 80% idle — is the engine's own overhead, and reducing it is what would make
placement irrelevant on every platform.

## What leaving CUDA costs

llama.cpp b10762 running Qwen3-1.7B — the same shape as this model's talker
(28 layers, hidden 2048, GQA 16/8, head_dim 128, FFN 6144) — on both backends:

| Qwen3-1.7B | CUDA | Vulkan | Vulkan / CUDA |
| --- | --- | --- | --- |
| f16, tg128 (batch-1 decode) | 175.6 t/s | 173.1 t/s | 98.6% |
| f16, pp512 (prompt) | 18920.3 t/s | 17289.7 t/s | 91.4% |
| Q8_0, tg128 | 310.1 t/s | 286.5 t/s | 92.4% |
| Q8_0, pp512 | 24946.2 t/s | 17026.7 t/s | 68.3% |

This measures one large step per token. A TTS frame is one talker step plus 15
code-predictor steps, so per-launch overhead weighs far more here than these
numbers suggest, and the gap above should not be read as the gap for this
workload.

## Memory

Process dedicated VRAM by stage, exact-pid perf counter, 1.7B Base with a
7.7 s ICL reference (`examples/mem_probe.rs`):

| Stage | Before | After | Change |
| --- | --- | --- | --- |
| after load | 4333 MB | 3652 MB | -681 MB |
| after warmup | 7661 MB | 4899 MB | -2762 MB |
| after prompt | 7301 MB | 3849 MB | -3452 MB |
| after synthesis (steady) | 5489 MB | 4771 MB | -718 MB |

Weights are 3.86 GB bf16 on disk and load as f16, so 3652 MB after load is
within 200 MB of the weights themselves. What changed, in order of effect:

- The talker's text embedding table (151936 x 2048, 594 MiB) is the largest
  tensor in the model and is only ever read a few rows at a time. It now stays
  in host memory and the rows for each utterance are gathered on the CPU
  (`weights.rs: HostTable`, `talker.rs: embed_text`).
- The speaker encoder and speech-tokenizer encoder are only needed while a
  prompt is built. They are loaded from the mapped weight files on demand and
  dropped afterwards (`engine.rs: build_clone_prompt`), which also holds the
  after-prompt figure below the after-warmup one instead of above it.
- Warmup no longer exercises the decoder's 512-frame bucket. Nothing in the
  app one-shot-decodes more than about 90 frames, and that bucket was the
  largest transient in the process — its pages stayed resident afterwards.
- After warmup, after each prompt and after each synthesis the engine hands
  free pool pages back to the driver (`Engine::trim_memory`, which is
  `B::sync` followed by `B::memory_cleanup`). cubecl's `ExclusivePages`
  allocator otherwise keeps every page it has ever needed.
- The safetensors header is parsed once per file instead of once per tensor;
  load time went from 7.7 s to 6.5–7.1 s.

Output is bit-identical to before on every sentence checked, and warm
throughput is unchanged (1.56x on the long Korean line, against 1.52–1.64x
before).

Weight quantisation was measured and rejected for now. cubecl 0.10 does run
quantised matmul on Vulkan, but on this batch-1 decode shape it costs 2–5x
the time of f16 (`examples/quant_probe.rs`: Q8S block-128 at 0.34x the f16
speed for a 0.66% relative error, Q4S block-32 at 0.58x for 9.6%). Halving the
weights is not worth tripling the frame time.

## This engine on its own

1.7B Base, in-context prompt; load, warm-up and prompt times from
`examples/ttfa.rs`, peak VRAM from the process's GPU memory counter:

| | load | warmup | prompt | peak process VRAM |
| --- | --- | --- | --- | --- |
| English, warm | 6.5–7.1 s | 0.4–0.6 s | 0.1 s | 4.8 GB |
| Korean, warm | 6.5–7.1 s | 0.4–0.6 s | 0.1–0.2 s | 4.8 GB |

Speed is the function above. Two things swing a single timing and were the
cause of the spread in earlier versions of this table: kernel autotuning
landing inside the measurement when there is no warm-up run, and the core
placement described under "Core placement on hybrid CPUs".

VRAM is the Windows GPU performance counter for the process
(`\GPU Process Memory(pid_<pid>_*)\Dedicated Usage`, with the underscore — a
bare `pid_<pid>*` also matches longer pids), not `nvidia-smi`: a Vulkan
process does not appear in the compute-app list, and GPU-wide usage moves with
whatever else is running.
