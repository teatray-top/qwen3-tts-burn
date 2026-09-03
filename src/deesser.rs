use crate::lowpass::ButterworthLp;

/// Splits the signal at this frequency; everything above is the sibilant band.
const SPLIT_HZ: f64 = 4500.0;
/// A sibilant is detected when the high-band envelope exceeds this multiple of
/// its slow running average.
const THRESH_RATIO: f32 = 1.6;
/// dB of high-band reduction per octave the envelope is over threshold.
const RATIO_DB: f32 = 6.0;

/// Streaming de-esser: dynamically attenuates the >4.5 kHz band only when it
/// spikes (fricatives/affricates — the "지글거림"), leaving vowels/formants
/// untouched. Complementary split (`hi = x - lp(x)`), a fast-attack/slow-release
/// envelope on `hi`, and a slow running average as the (level-adaptive) threshold,
/// so it works on a continuous stream without a global percentile. All state is
/// held here, so chunks processed in order stay seamless.
pub struct Deesser {
    split: ButterworthLp,
    env: f32,
    avg: f32,
    atk: f32,
    rel: f32,
    avg_c: f32,
    max_red_db: f32,
}

impl Deesser {
    pub fn new(sample_rate_hz: f64, max_reduction_db: f32) -> Self {
        let coef = |secs: f64| (-1.0 / (sample_rate_hz * secs)).exp() as f32;
        Self {
            split: ButterworthLp::new(SPLIT_HZ, sample_rate_hz, 4),
            env: 0.0,
            avg: 1e-3,
            atk: coef(0.002),
            rel: coef(0.040),
            avg_c: coef(0.200),
            max_red_db: max_reduction_db,
        }
    }

    pub fn process_buffer(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            let x = *s;
            let lo = self.split.process1(x);
            let hi = x - lo;
            let e = hi.abs();
            let coef = if e > self.env { self.atk } else { self.rel };
            self.env = coef * self.env + (1.0 - coef) * e;
            self.avg = self.avg_c * self.avg + (1.0 - self.avg_c) * self.env;
            let over = (self.env / (self.avg * THRESH_RATIO + 1e-9)).max(1.0);
            let red_db = (RATIO_DB * over.log2()).min(self.max_red_db);
            let gain = 10f32.powf(-red_db / 20.0);
            *s = lo + hi * gain;
        }
    }
}
