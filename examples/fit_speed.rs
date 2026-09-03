//! Fit the generation-time model to the speed grid.
//!
//! ```text
//! cargo run --release --example fit_speed [-- <grid.tsv>...]
//! ```
//!
//! Without arguments it reads `eval/results/speed_grid_en.tsv` and
//! `speed_grid_ko.tsv`, the output of `examples/speed_grid.rs`. The structure
//! is fixed by the architecture, only the coefficients are fitted:
//!
//! ```text
//! t_gen(R, N) = c0 + c1 * R + tau * N     prefill, then talker + code predictor per frame
//! t_dec(T)    = d0 + d1 * T               codec decoder, T = pow2(R + N), at least 32
//! realtime    = 0.08 * N / (t_gen + t_dec)
//! ```
//!
//! R = reference frames, N = generated frames. tau is fitted in the
//! 1024-position KV bucket, where every ordinary sentence lands; the first use
//! of another bucket or decoder size inside a process autotunes kernels and is
//! excluded (the grid does not warm those separately).

use std::path::PathBuf;

struct Row {
    r: f64,
    n: f64,
    pad: f64,
    bucket: usize,
    gen: f64,
    dec: f64,
}

fn bucket(r: usize, max_frames: usize) -> usize {
    let n = r + 24 + max_frames;
    if n <= 448 {
        448
    } else if n <= 1024 {
        1024
    } else {
        2048
    }
}

fn load(path: &PathBuf) -> Vec<Row> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split('\t').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("{}: no column {name}", path.display()))
    };
    let (ir, imax, inn, ipad, igen, idec) = (
        col("ref_frames"),
        col("max_frames"),
        col("frames"),
        col("padded"),
        col("gen_s"),
        col("decode_s"),
    );
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            let f = |i: usize| -> f64 { c[i].parse().expect("number") };
            let r = f(ir) as usize;
            Row {
                r: r as f64,
                n: f(inn),
                pad: f(ipad),
                bucket: bucket(r, f(imax) as usize),
                gen: f(igen),
                dec: f(idec),
            }
        })
        .collect()
}

/// Least squares by the normal equations; the systems here are 2x2 and 3x3.
fn lstsq(rows: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, f64, f64) {
    let k = rows[0].len();
    let mut ata = vec![vec![0.0; k]; k];
    let mut atb = vec![0.0; k];
    for (a, &b) in rows.iter().zip(y) {
        for i in 0..k {
            atb[i] += a[i] * b;
            for j in 0..k {
                ata[i][j] += a[i] * a[j];
            }
        }
    }
    for i in 0..k {
        let p = (i..k)
            .max_by(|&a, &b| ata[a][i].abs().total_cmp(&ata[b][i].abs()))
            .unwrap();
        ata.swap(i, p);
        atb.swap(i, p);
        let row_i = ata[i].clone();
        for j in (i + 1)..k {
            let f = ata[j][i] / row_i[i];
            for (c, v) in ata[j].iter_mut().enumerate().skip(i) {
                *v -= f * row_i[c];
            }
            atb[j] -= f * atb[i];
        }
    }
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let s: f64 = ((i + 1)..k).map(|j| ata[i][j] * x[j]).sum();
        x[i] = (atb[i] - s) / ata[i][i];
    }
    let mut errs: Vec<f64> = rows
        .iter()
        .zip(y)
        .map(|(a, &b)| {
            let pred: f64 = a.iter().zip(&x).map(|(p, q)| p * q).sum();
            (pred - b).abs() / b.max(1e-9)
        })
        .collect();
    errs.sort_by(|a, b| a.total_cmp(b));
    let median = errs[errs.len() / 2];
    let max = *errs.last().unwrap();
    (x, median, max)
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        paths = ["speed_grid_en.tsv", "speed_grid_ko.tsv"]
            .iter()
            .map(|f| root.join("eval").join("results").join(f))
            .collect();
    }
    let rows: Vec<Row> = paths.iter().flat_map(load).collect();
    println!("{} rows", rows.len());

    let warm: Vec<&Row> = rows.iter().filter(|r| r.bucket == 1024).collect();
    let (x, med, mx) = lstsq(
        &warm.iter().map(|r| vec![1.0, r.r, r.n]).collect::<Vec<_>>(),
        &warm.iter().map(|r| r.gen).collect::<Vec<_>>(),
    );
    let (c0, c1, tau) = (x[0], x[1], x[2]);
    println!(
        "t_gen = {c0:.3} s + {:.2} ms * R + {:.2} ms * N   (1024 bucket, {} rows; median |err| {:.1}%, max {:.1}%)",
        c1 * 1000.0,
        tau * 1000.0,
        warm.len(),
        med * 100.0,
        mx * 100.0
    );
    println!(
        "   per-frame cost {:.1} ms against an 80 ms frame -> {:.2}x realtime asymptotically",
        tau * 1000.0,
        0.08 / tau
    );
    for m in [448usize, 2048] {
        let mut per: Vec<f64> = rows
            .iter()
            .filter(|r| r.bucket == m)
            .map(|r| (r.gen - c0 - c1 * r.r) / r.n)
            .collect();
        if !per.is_empty() {
            per.sort_by(|a, b| a.total_cmp(b));
            println!(
                "   bucket {m}: apparent per-frame {:.1} ms over {} rows, first use in the process included (autotune inside the timing)",
                per[per.len() / 2] * 1000.0,
                per.len()
            );
        }
    }

    let dec: Vec<&Row> = rows.iter().filter(|r| r.dec < 0.5).collect();
    let (xd, medd, mxd) = lstsq(
        &dec.iter().map(|r| vec![1.0, r.pad]).collect::<Vec<_>>(),
        &dec.iter().map(|r| r.dec).collect::<Vec<_>>(),
    );
    println!(
        "t_dec = {:.0} ms + {:.3} ms * T   (T = pow2(R+N); {} rows, median |err| {:.1}%, max {:.1}%)",
        xd[0] * 1000.0,
        xd[1] * 1000.0,
        dec.len(),
        medd * 100.0,
        mxd * 100.0
    );
    let mut sizes: Vec<usize> = dec.iter().map(|r| r.pad as usize).collect();
    sizes.sort_unstable();
    sizes.dedup();
    for t in sizes {
        let mut ds: Vec<f64> = dec
            .iter()
            .filter(|r| r.pad as usize == t)
            .map(|r| r.dec)
            .collect();
        ds.sort_by(|a, b| a.total_cmp(b));
        println!(
            "   T={t:4}: measured {:4.0} ms (n={})",
            ds[ds.len() / 2] * 1000.0,
            ds.len()
        );
    }
    let skipped = rows.len() - dec.len();
    if skipped > 0 {
        println!(
            "   excluded {skipped} decode rows above 0.5 s (first use of a decoder size: autotune)"
        );
    }

    let predict = |r: usize, n: usize| -> f64 {
        let t = ((r + n).next_power_of_two()).max(32) as f64;
        c0 + c1 * r as f64 + tau * n as f64 + xd[0] + xd[1] * t
    };
    println!("\nworked examples (in-context reference, app profile):");
    for (r, n) in [
        (25usize, 40usize),
        (66, 44),
        (95, 44),
        (87, 59),
        (95, 100),
        (95, 200),
        (188, 60),
    ] {
        let t = predict(r, n);
        println!(
            "   R={r:3} N={n:3} ({:4.1}s audio): {t:5.2} s -> {:4.2}x realtime",
            0.08 * n as f64,
            0.08 * n as f64 / t
        );
    }
    let (lo, hi) = (37usize, 200usize);
    let speeds: Vec<f64> = (lo..=hi)
        .map(|n| 0.08 * n as f64 / predict(95, n))
        .collect();
    let min = speeds.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = speeds.iter().cloned().fold(0.0, f64::max);
    println!(
        "lines of {:.0}-{:.0} s from a 95-frame (7.6 s) reference: {min:.2}x to {max:.2}x realtime",
        0.08 * lo as f64,
        0.08 * hi as f64
    );
}
