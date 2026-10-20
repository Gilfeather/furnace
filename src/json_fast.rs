//! A fast path for the two JSON bodies `POST /predict` almost always gets,
//! `{"input": [numbers]}` and `{"inputs": [[numbers], ...]}`.
//!
//! A ResNet-18 image is 150,528 numbers. serde_json parses each through its general machinery
//! into an `f64`; here the numbers are found with a byte scan, parsed straight to `f32` by
//! `fast-float2`, and, for large arrays, in parallel chunks. Anything else (other keys, other
//! value types, whitespace or number forms this scanner does not take, numbers past `f32`'s
//! range) returns `None`, and the caller falls back to serde_json, which then also produces the
//! error messages.

use rayon::prelude::*;

/// The numbers of a `{"input": [...]}` or `{"inputs": [[...], ...]}` body
#[derive(Debug, PartialEq)]
pub enum FastInput {
    Single(Vec<f32>),
    Batch(Vec<Vec<f32>>),
}

/// Arrays at least this long are parsed in parallel chunks
const PARALLEL_MIN_BYTES: usize = 64 * 1024;

/// Parse `body` if it is one of the two shapes, else `None`
pub fn parse(body: &[u8]) -> Option<FastInput> {
    let mut s = Scanner { b: body, i: 0 };
    s.ws();
    s.expect(b'{')?;
    s.ws();
    let batch = if s.literal(b"\"input\"") {
        false
    } else if s.literal(b"\"inputs\"") {
        true
    } else {
        return None;
    };
    s.ws();
    s.expect(b':')?;
    s.ws();
    let input = if batch {
        s.expect(b'[')?;
        let mut arrays = Vec::new();
        s.ws();
        if !s.peek_is(b']') {
            loop {
                arrays.push(s.array_span()?);
                s.ws();
                if s.eat(b',') {
                    s.ws();
                    continue;
                }
                break;
            }
        }
        s.expect(b']')?;
        let arrays: Option<Vec<Vec<f32>>> = arrays
            .par_iter()
            .map(|&(start, end)| numbers(&body[start..end]))
            .collect();
        FastInput::Batch(arrays?)
    } else {
        let (start, end) = s.array_span()?;
        FastInput::Single(numbers(&body[start..end])?)
    };
    s.ws();
    s.expect(b'}')?;
    s.ws();
    (s.i == body.len()).then_some(input)
}

struct Scanner<'a> {
    b: &'a [u8],
    i: usize,
}

impl Scanner<'_> {
    fn ws(&mut self) {
        while let Some(b' ' | b'\n' | b'\r' | b'\t') = self.b.get(self.i) {
            self.i += 1;
        }
    }

    fn peek_is(&self, c: u8) -> bool {
        self.b.get(self.i) == Some(&c)
    }

    fn eat(&mut self, c: u8) -> bool {
        let hit = self.peek_is(c);
        if hit {
            self.i += 1;
        }
        hit
    }

    fn expect(&mut self, c: u8) -> Option<()> {
        self.eat(c).then_some(())
    }

    fn literal(&mut self, text: &[u8]) -> bool {
        let hit = self.b[self.i..].starts_with(text);
        if hit {
            self.i += text.len();
        }
        hit
    }

    /// The span between the brackets of a flat array of numbers, which may not contain
    /// brackets, braces or quotes
    fn array_span(&mut self) -> Option<(usize, usize)> {
        self.expect(b'[')?;
        let start = self.i;
        let len = self.b[start..]
            .iter()
            .position(|&c| matches!(c, b']' | b'[' | b'{' | b'}' | b'"'))?;
        let end = start + len;
        self.i = end;
        self.expect(b']')?;
        Some((start, end))
    }
}

/// The comma-separated numbers between an array's brackets
fn numbers(span: &[u8]) -> Option<Vec<f32>> {
    if span.len() < PARALLEL_MIN_BYTES {
        let mut out = Vec::with_capacity(span.len() / 8);
        return parse_list(span, &mut out, true).then_some(out);
    }
    // Cut at commas into one chunk per thread or so, parse them in parallel, then join
    let threads = rayon::current_num_threads().max(1);
    let target = span.len().div_ceil(threads * 2).max(PARALLEL_MIN_BYTES / 4);
    let mut cuts = vec![0];
    let mut at = target;
    while at < span.len() {
        let comma = at + span[at..].iter().position(|&c| c == b',')?;
        cuts.push(comma + 1);
        at = comma + 1 + target;
    }
    cuts.push(span.len() + 1);
    let chunks: Option<Vec<Vec<f32>>> = cuts
        .windows(2)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|w| {
            let (start, end) = (w[0], (w[1] - 1).min(span.len()));
            let mut out = Vec::with_capacity((end - start) / 8);
            parse_list(&span[start..end], &mut out, false).then_some(out)
        })
        .collect();
    Some(chunks?.concat())
}

/// Parse `number (, number)*` (or nothing, when `empty_ok`) into `out`
fn parse_list(text: &[u8], out: &mut Vec<f32>, empty_ok: bool) -> bool {
    let mut i = 0;
    let skip_ws = |i: &mut usize| {
        while let Some(b' ' | b'\n' | b'\r' | b'\t') = text.get(*i) {
            *i += 1;
        }
    };
    skip_ws(&mut i);
    if i == text.len() {
        return empty_ok;
    }
    loop {
        let Some(len) = json_number_len(&text[i..]) else {
            return false;
        };
        // A number past f32's range comes back infinite: leave it to serde_json and the model's
        // checks, so this path never accepts what the fallback would answer otherwise
        match fast_float2::parse::<f32, _>(&text[i..i + len]) {
            Ok(value) if value.is_finite() => out.push(value),
            _ => return false,
        }
        i += len;
        skip_ws(&mut i);
        if i == text.len() {
            return true;
        }
        if text[i] != b',' {
            return false;
        }
        i += 1;
        skip_ws(&mut i);
    }
}

/// Length of the JSON number at the start of `s` (`-?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?`)
fn json_number_len(s: &[u8]) -> Option<usize> {
    let mut i = 0;
    if s.first() == Some(&b'-') {
        i += 1;
    }
    let digits = |i: &mut usize| {
        let start = *i;
        while s.get(*i).is_some_and(u8::is_ascii_digit) {
            *i += 1;
        }
        *i - start
    };
    match s.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => {
            digits(&mut i);
        }
        _ => return None,
    }
    if s.get(i) == Some(&b'.') {
        i += 1;
        if digits(&mut i) == 0 {
            return None;
        }
    }
    if let Some(b'e' | b'E') = s.get(i) {
        i += 1;
        if let Some(b'+' | b'-') = s.get(i) {
            i += 1;
        }
        if digits(&mut i) == 0 {
            return None;
        }
    }
    Some(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize, Debug, PartialEq)]
    #[serde(untagged)]
    enum Serde {
        Single { input: Vec<f32> },
        Batch { inputs: Vec<Vec<f32>> },
    }

    fn agree(body: &str) {
        let fast = parse(body.as_bytes());
        let serde: Option<Serde> = serde_json::from_str(body).ok();
        match (fast, serde) {
            (Some(FastInput::Single(a)), Some(Serde::Single { input: b })) => close(&a, &b, body),
            (Some(FastInput::Batch(a)), Some(Serde::Batch { inputs: b })) => {
                assert_eq!(a.len(), b.len(), "{body}");
                for (x, y) in a.iter().zip(&b) {
                    close(x, y, body);
                }
            }
            // Falling back is always allowed; accepting what serde rejects is not
            (None, _) => {}
            (fast, serde) => panic!("{body}: fast {fast:?}, serde {serde:?}"),
        }
    }

    /// Same values; serde rounds through f64, so allow one f32 ulp
    fn close(a: &[f32], b: &[f32], body: &str) {
        assert_eq!(a.len(), b.len(), "{body}");
        for (x, y) in a.iter().zip(b) {
            let ulps = (x.to_bits() as i64 - y.to_bits() as i64).abs();
            assert!(x == y || ulps <= 1, "{body}: {x} vs {y}");
        }
    }

    #[test]
    fn common_shapes() {
        assert_eq!(
            parse(b"{\"input\": [1, 2.5, -3e2]}"),
            Some(FastInput::Single(vec![1.0, 2.5, -300.0]))
        );
        assert_eq!(
            parse(b" {\"inputs\":[[1,2],[3, 4]] } "),
            Some(FastInput::Batch(vec![vec![1.0, 2.0], vec![3.0, 4.0]]))
        );
        assert_eq!(parse(b"{\"input\": []}"), Some(FastInput::Single(vec![])));
        assert_eq!(parse(b"{\"inputs\": []}"), Some(FastInput::Batch(vec![])));
    }

    #[test]
    fn falls_back_on_anything_else() {
        for body in [
            "{\"input\": [1, 2], \"extra\": 1}",
            "{\"inputs\": [1, 2]}",
            "{\"input\": [\"1\"]}",
            "{\"input\": [NaN]}",
            "{\"input\": [1.]}",
            "{\"input\": [.5]}",
            "{\"input\": [01]}",
            "{\"input\": [1,]}",
            "{\"input\": [,1]}",
            "{\"input\": [1 2]}",
            "{\"input\": [1]",
            "{\"input\": [1]} x",
            "{\"input\": [1e39]}",
            "{\"input\": [-1e39]}",
            "{\"input\": [1e400]}",
            "{\"inputs\": [[1], [1e400]]}",
            "{\"other\": [1]}",
            "[1, 2]",
            "",
        ] {
            assert_eq!(parse(body.as_bytes()), None, "{body}");
            agree(body);
        }
    }

    #[test]
    fn matches_serde_on_many_numbers() {
        let mut state = 12345u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mut values = Vec::new();
        for _ in 0..200_000 {
            let r = next();
            let v = match r % 5 {
                0 => format!("{}", (r >> 11) as f64 / (1u64 << 53) as f64),
                1 => format!("{:e}", ((r >> 20) as f64 - 1e6) * 1.7),
                2 => format!("{}", (r % 1000) as i64 - 500),
                3 => format!("-0.{:08}", r % 100_000_000),
                _ => format!("{}", ((r >> 8) as f32) / 3.0),
            };
            values.push(v);
        }
        // Large enough for the parallel path, with irregular whitespace
        let single = format!("{{ \"input\" : [ {} ] }}", values.join(" ,\n "));
        assert!(single.len() > 4 * PARALLEL_MIN_BYTES);
        assert!(parse(single.as_bytes()).is_some());
        agree(&single);
        let batch = format!(
            "{{\"inputs\": [[{}], [{}]]}}",
            values[..1000].join(","),
            values[1000..90_000].join(",")
        );
        assert!(parse(batch.as_bytes()).is_some());
        agree(&batch);
    }
}
