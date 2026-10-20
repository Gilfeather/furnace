//! Expectations on the Numenta Anomaly Benchmark (NAB): 58 series with labelled anomaly
//! windows, scored with NAB's own method and compared with its published detectors.
//!
//! The detector is an `Expectations` neuron per series with a scripted brain. Each row is a
//! JSON signal `{"value": v, "delta": v - previous}`. On the first row the brain installs a
//! `range` on each field, with `learn` letting the periphery widen it, and an `every` at the
//! series' sampling interval. During NAB's probationary period (the first 15%) it answers every
//! violation `normal`, so the ranges learn the envelope of the start of the series; at its end it
//! narrows what may still be learned to half the envelope's width on either side. After that a
//! violation pages, unless the brain paged within a third of the probationary period before, in which case it
//! answers `normal` and the range learns (within its bounds). A page is an anomaly score of 1 on
//! that row; every other row scores 0.
//!
//! The scorer is NAB's (`nab/sweeper.py`, `nab/runner.py`): scaled-sigmoid weights in and after
//! the windows, the probationary rows left out, one threshold for the whole corpus chosen to
//! maximise the score (here by an exact sweep), normalised as `100 * (score - null) / (perfect -
//! null)` for the standard, reward-low-FP and reward-low-FN profiles. It is checked against the
//! published `numenta` detector's outputs, whose published scores it must reproduce.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use furnace::nerve::expect::{ExpectConfig, Expectations};
use furnace::nerve::{Command, Feature, Payload, Signal};
use furnace_bench_common::fail;
use serde::Serialize;
use serde_json::Value;

const PROFILES: [(&str, f64, f64, f64); 3] = [
    // name, tp, fp, fn
    ("standard", 1.0, 0.11, 1.0),
    ("reward_low_FP_rate", 1.0, 0.22, 1.0),
    ("reward_low_FN_rate", 1.0, 0.11, 2.0),
];

/// Seconds since 1970 of `YYYY-MM-DD HH:MM:SS[.ffffff]`
fn seconds(timestamp: &str) -> i64 {
    let n = |s: &str| {
        s.parse::<i64>()
            .unwrap_or_else(|_| fail(&format!("timestamp {timestamp}")))
    };
    let (date, time) = timestamp
        .split_once(' ')
        .unwrap_or_else(|| fail(&format!("timestamp {timestamp}")));
    let mut d = date.split('-');
    let (y, m, day) = (
        n(d.next().unwrap_or("")),
        n(d.next().unwrap_or("")),
        n(d.next().unwrap_or("")),
    );
    let time = time.split('.').next().unwrap_or("");
    let mut t = time.split(':');
    let (hh, mm, ss) = (
        n(t.next().unwrap_or("")),
        n(t.next().unwrap_or("")),
        n(t.next().unwrap_or("")),
    );
    // Howard Hinnant's days_from_civil
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    days * 86_400 + hh * 3600 + mm * 60 + ss
}

struct Series {
    name: String,
    times: Vec<i64>,
    values: Vec<f64>,
    windows: Vec<(i64, i64)>,
}

fn read_csv(path: &Path, column: &str) -> (Vec<i64>, Vec<f64>) {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| fail(&format!("cannot read {}: {e}", path.display())));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
    let col = header
        .iter()
        .position(|h| *h == column)
        .unwrap_or_else(|| fail(&format!("no column {column} in {}", path.display())));
    let (mut times, mut values) = (Vec::new(), Vec::new());
    for line in lines.filter(|l| !l.is_empty()) {
        let cells: Vec<&str> = line.split(',').collect();
        times.push(seconds(cells[0]));
        values.push(cells[col].parse().unwrap_or(0.0));
    }
    (times, values)
}

fn corpus(dir: &Path) -> Vec<Series> {
    let labels: BTreeMap<String, Vec<Vec<String>>> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("labels/combined_windows.json"))
            .unwrap_or_else(|e| fail(&format!("labels: {e}"))),
    )
    .unwrap_or_else(|e| fail(&format!("labels: {e}")));
    labels
        .into_iter()
        .map(|(name, windows)| {
            let (times, values) = read_csv(&dir.join("data").join(&name), "value");
            Series {
                windows: windows
                    .iter()
                    .map(|w| (seconds(&w[0]), seconds(&w[1])))
                    .collect(),
                name,
                times,
                values,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// NAB's scorer

fn scaled_sigmoid(position: f64) -> f64 {
    if position > 3.0 {
        -1.0
    } else {
        2.0 / (1.0 + (5.0 * position).exp()) - 1.0
    }
}

fn probation(rows: usize) -> usize {
    ((0.15 * rows as f64).floor()).min(0.15 * 5000.0) as usize
}

/// A scorable row: its anomaly score, its weighted score, and its window (global index)
struct Point {
    score: f64,
    sweep: f64,
    window: Option<usize>,
}

/// `calcSweepScore`: the scorable points of one series, windows numbered from `first_window`
fn sweep(series: &Series, scores: &[f64], tp: f64, fp: f64, first_window: usize) -> Vec<Point> {
    let max_tp = scaled_sigmoid(-1.0);
    let probationary = probation(series.times.len());
    let mut windows = series.windows.iter().enumerate().peekable();
    let mut current: Option<(usize, usize, f64)> = None; // (window, right index, width)
    let mut previous: Option<(usize, f64)> = None; // (right index, width)
    let index_of = |t: i64| series.times.iter().position(|&x| x == t);
    let mut points = Vec::with_capacity(series.times.len());
    for (i, (&t, &score)) in series.times.iter().zip(scores).enumerate() {
        if let Some(&(k, &(start, end))) = windows.peek()
            && t == start
        {
            windows.next();
            let right = index_of(end).unwrap_or(series.times.len() - 1);
            let left = index_of(start).unwrap_or(i);
            current = Some((first_window + k, right, (right - left + 1) as f64));
        }
        let weighted = match current {
            Some((_, right, width)) => {
                let position = -((right as f64 - i as f64 + 1.0) / width);
                scaled_sigmoid(position) * tp / max_tp
            }
            None => {
                let unweighted = match previous {
                    None => -1.0,
                    Some((right, width)) => {
                        scaled_sigmoid((i as f64 - right as f64).abs() / (width - 1.0))
                    }
                };
                unweighted * fp
            }
        };
        if i >= probationary {
            points.push(Point {
                score,
                sweep: weighted,
                window: current.map(|c| c.0),
            });
        }
        if let Some((_, right, width)) = current
            && i == right
        {
            previous = Some((right, width));
            current = None;
        }
    }
    points
}

/// `calcScoreByThreshold`, keeping the best threshold's score
fn best_score(mut points: Vec<Point>, fn_weight: f64, windows: usize) -> (f64, f64) {
    points.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut parts: Vec<Option<f64>> = vec![None; windows];
    for p in &points {
        if let Some(w) = p.window {
            parts[w] = Some(-fn_weight);
        }
    }
    let mut fp = 0.0;
    let total = |parts: &[Option<f64>], fp: f64| parts.iter().flatten().sum::<f64>() + fp;
    let mut best = (total(&parts, fp), 1.1);
    let mut threshold = 1.1;
    for p in &points {
        if p.score != threshold {
            let score = total(&parts, fp);
            if score > best.0 {
                best = (score, threshold);
            }
            threshold = p.score;
        }
        match p.window {
            None => fp += p.sweep,
            Some(w) => parts[w] = Some(parts[w].unwrap_or(-fn_weight).max(p.sweep)),
        }
    }
    let score = total(&parts, fp);
    if score > best.0 {
        best = (score, threshold);
    }
    best
}

/// Normalised NAB scores per profile of a detector's per-row scores
fn score(series: &[Series], scores: &[Vec<f64>]) -> BTreeMap<String, f64> {
    let windows: usize = series.iter().map(|s| s.windows.len()).sum();
    PROFILES
        .iter()
        .map(|&(name, tp, fp, fn_weight)| {
            let points = |all: &[Vec<f64>]| {
                let mut first = 0;
                let mut points = Vec::new();
                for (s, scores) in series.iter().zip(all) {
                    points.extend(sweep(s, scores, tp, fp, first));
                    first += s.windows.len();
                }
                points
            };
            let (raw, _) = best_score(points(scores), fn_weight, windows);
            let zeros: Vec<Vec<f64>> = series.iter().map(|s| vec![0.0; s.times.len()]).collect();
            let (null, _) = best_score(points(&zeros), fn_weight, windows);
            let perfect = windows as f64 * tp;
            (name.to_string(), 100.0 * (raw - null) / (perfect - null))
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The detector

struct Detected {
    scores: Vec<f64>,
    brain_calls: u64,
    pages: u64,
    absences: u64,
}

fn interval(times: &[i64]) -> u64 {
    let mut gaps: Vec<i64> = times
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|g| *g > 0)
        .collect();
    gaps.sort_unstable();
    gaps.get(gaps.len() / 2).copied().unwrap_or(300) as u64
}

fn detect(series: &Series) -> Detected {
    let n = series.times.len();
    let probationary = probation(n);
    // Only what NAB gives a detector: the probationary period, not the series' length
    let cooldown = (probationary / 3).max(1);
    let expectations = Expectations::new(
        "nab-expect",
        ExpectConfig::new()
            .effectors(["page"])
            .ttl(Duration::from_secs(365 * 86_400)),
    );
    let expect = expectations.effector("expect");
    let epoch = expectations.epoch();
    let t0 = series.times[0];
    let at = |t: i64| epoch + Duration::from_secs((t - t0).max(0) as u64);
    let period_ms = interval(&series.times) * 1000;
    let wide = [-1e300, 1e300];
    let mut out = Detected {
        scores: vec![0.0; n],
        brain_calls: 0,
        pages: 0,
        absences: 0,
    };
    let mut last_page: Option<usize> = None;
    let mut previous = series.values[0];
    for i in 0..n {
        let now: Instant = at(series.times[i]);
        for absence in expectations.due(now) {
            if absence.payload.as_command().is_none() {
                out.brain_calls += 1;
                out.absences += 1;
            }
        }
        if i == probationary {
            // The end of the probationary period: what may still be learned is narrowed
            for id in ["value", "delta"] {
                if let Some(mut e) = expectations.get(id)
                    && let (Some(lo), Some(hi)) = (e.param("min"), e.param("max"))
                {
                    let w = (hi - lo).max(f64::EPSILON);
                    e.learn.insert("min".into(), [lo - 0.5 * w, lo]);
                    e.learn.insert("max".into(), [hi, hi + 0.5 * w]);
                    let replace = serde_json::json!({"replace": e});
                    let _ = expect.act_at(&Command::new("expect", replace), &absence_free(), now);
                }
            }
        }
        let value = series.values[i];
        let signal = Signal::new(
            Feature::NONE,
            Payload::json(serde_json::json!({"value": value, "delta": value - previous})),
        );
        previous = value;
        for fired in expectations.react(signal, now) {
            if fired.payload.as_command().is_some() {
                continue;
            }
            out.brain_calls += 1;
            let surprise = fired
                .payload
                .to_json()
                .and_then(|v| v.get("surprise").cloned())
                .unwrap_or_default();
            let template = surprise["observation"]["template"].clone();
            let commands: Vec<Value> = match surprise["reason"].as_str().unwrap_or_default() {
                "unexpected" => vec![serde_json::json!({"add": [
                    {"id": "value", "kind": "range", "field": "value", "min": value, "max": value,
                     "selector": {"template": template},
                     "learn": {"min": wide, "max": wide}},
                    {"id": "delta", "kind": "range", "field": "delta", "min": 0.0, "max": 0.0,
                     "selector": {"template": template},
                     "learn": {"min": wide, "max": wide}},
                    {"id": "sampling", "kind": "every", "period_ms": period_ms,
                     "tolerance_ms": period_ms, "selector": {"template": template}},
                ]})],
                "violation" => {
                    let paging = i >= probationary && last_page.is_none_or(|p| i - p > cooldown);
                    if paging {
                        last_page = Some(i);
                        out.pages += 1;
                        out.scores[i] = 1.0;
                        Vec::new()
                    } else {
                        surprise["broken"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .map(|b| {
                                serde_json::json!({"verdict": {"id": b["expectation"], "normal": true}})
                            })
                            .collect()
                    }
                }
                _ => Vec::new(),
            };
            for args in commands {
                let _ = expect.act_at(&Command::new("expect", args), &fired, now);
            }
        }
    }
    out
}

/// A signal for a command that answers no surprise
fn absence_free() -> Signal {
    Signal::new(Feature::NONE, Payload::None)
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub series: usize,
    pub rows: usize,
    pub windows: usize,
    /// The scorer reproduces the published detectors' scores (from their outputs)
    pub check: BTreeMap<String, BTreeMap<String, f64>>,
    pub published: BTreeMap<String, BTreeMap<String, f64>>,
    pub expectations: BTreeMap<String, f64>,
    pub brain_calls: u64,
    pub pages: u64,
    pub absences: u64,
    pub seconds: f64,
}

pub fn bench(dir: &str) -> Report {
    let dir = Path::new(dir).join("NAB");
    let series = corpus(&dir);
    let rows: usize = series.iter().map(|s| s.times.len()).sum();
    let windows: usize = series.iter().map(|s| s.windows.len()).sum();
    eprintln!(
        "NAB: {} series, {rows} rows, {windows} windows",
        series.len()
    );
    let mut check = BTreeMap::new();
    for detector in ["numenta", "null"] {
        let scores: Vec<Vec<f64>> = series
            .iter()
            .map(|s| {
                let (group, file) = s.name.split_once('/').unwrap_or(("", &s.name));
                let path = dir
                    .join("results")
                    .join(detector)
                    .join(group)
                    .join(format!("{detector}_{file}"));
                read_csv(&path, "anomaly_score").1
            })
            .collect();
        check.insert(detector.to_string(), score(&series, &scores));
    }
    let published: BTreeMap<String, BTreeMap<String, f64>> = serde_json::from_str(
        r#"{
        "numenta": {"standard": 70.1, "reward_low_FP_rate": 63.1, "reward_low_FN_rate": 74.3},
        "ARTime": {"standard": 74.9, "reward_low_FP_rate": 65.2, "reward_low_FN_rate": 80.4},
        "windowedGaussian": {"standard": 39.6, "reward_low_FP_rate": 20.9, "reward_low_FN_rate": 47.4},
        "skyline": {"standard": 35.7, "reward_low_FP_rate": 27.1, "reward_low_FN_rate": 44.5},
        "random": {"standard": 16.8, "reward_low_FP_rate": 5.8, "reward_low_FN_rate": 25.9},
        "null": {"standard": 0.0, "reward_low_FP_rate": 0.0, "reward_low_FN_rate": 0.0}
    }"#,
    )
    .unwrap_or_default();
    let started = Instant::now();
    let detected: Vec<Detected> = series.iter().map(detect).collect();
    let seconds = started.elapsed().as_secs_f64();
    let scores: Vec<Vec<f64>> = detected.iter().map(|d| d.scores.clone()).collect();
    Report {
        series: series.len(),
        rows,
        windows,
        check,
        published,
        expectations: score(&series, &scores),
        brain_calls: detected.iter().map(|d| d.brain_calls).sum(),
        pages: detected.iter().map(|d| d.pages).sum(),
        absences: detected.iter().map(|d| d.absences).sum(),
        seconds,
    }
}

pub fn print(report: &Report) {
    println!(
        "NAB: {} series, {} rows, {} windows",
        report.series, report.rows, report.windows
    );
    let row = |name: &str, s: &BTreeMap<String, f64>| {
        println!(
            "{name:<34} {:>8.1} {:>8.1} {:>8.1}",
            s.get("standard").copied().unwrap_or(f64::NAN),
            s.get("reward_low_FP_rate").copied().unwrap_or(f64::NAN),
            s.get("reward_low_FN_rate").copied().unwrap_or(f64::NAN)
        );
    };
    println!(
        "{:<34} {:>8} {:>8} {:>8}",
        "", "standard", "low FP", "low FN"
    );
    for (name, s) in &report.check {
        row(&format!("{name} (rescored here)"), s);
    }
    for (name, s) in &report.published {
        row(&format!("{name} (published)"), s);
    }
    row("expectations", &report.expectations);
    println!(
        "brain calls {} for {} rows ({:.2}%), pages {}, absences {}, {:.1} s",
        report.brain_calls,
        report.rows,
        100.0 * report.brain_calls as f64 / report.rows as f64,
        report.pages,
        report.absences,
        report.seconds
    );
}
