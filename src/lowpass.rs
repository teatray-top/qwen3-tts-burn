/// Butterworth low-pass (RBJ biquad cascade), ported from the relay app's
/// calibrated vocoder-noise filter (10.5 kHz, order 6 for clone output).
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    fn process(&mut self, x0: f64) -> f64 {
        let y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;
        y0
    }
}

pub struct ButterworthLp {
    stages: Vec<Biquad>,
}

impl ButterworthLp {
    pub fn new(cutoff_hz: f64, sample_rate_hz: f64, order: usize) -> Self {
        let order = order.max(2) & !1;
        let stages = (0..order / 2)
            .map(|k| {
                let theta = (2 * k + 1) as f64 * std::f64::consts::PI / (2.0 * order as f64);
                let q = 1.0 / (2.0 * theta.cos());
                let w0 = 2.0 * std::f64::consts::PI * cutoff_hz / sample_rate_hz;
                let alpha = w0.sin() / (2.0 * q);
                let cos_w0 = w0.cos();
                let a0 = 1.0 + alpha;
                Biquad {
                    b0: (1.0 - cos_w0) / 2.0 / a0,
                    b1: (1.0 - cos_w0) / a0,
                    b2: (1.0 - cos_w0) / 2.0 / a0,
                    a1: -2.0 * cos_w0 / a0,
                    a2: (1.0 - alpha) / a0,
                    x1: 0.0,
                    x2: 0.0,
                    y1: 0.0,
                    y2: 0.0,
                }
            })
            .collect();
        Self { stages }
    }

    pub fn process_buffer(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process1(*s);
        }
    }

    /// Filter one sample (keeps state). Used by the de-esser's band split.
    pub fn process1(&mut self, x: f32) -> f32 {
        let mut v = x as f64;
        for st in self.stages.iter_mut() {
            v = st.process(v);
        }
        v as f32
    }
}
