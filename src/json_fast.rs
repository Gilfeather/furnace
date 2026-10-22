//! A fast path for the two JSON bodies `POST /predict` almost always gets,
//! `{"input": [numbers]}` and `{"inputs": [[numbers], ...]}`.
//!
//! A ResNet-18 image is 150,528 numbers. serde_json parses each through its general machinery
//! into an `f64`; here the numbers are found with a byte scan, parsed straight to `f32` by the
//! standard library, and, for large arrays, in parallel chunks on scoped threads. Anything else (other keys, other
//! value types, whitespace or number forms this scanner does not take, numbers past `f32`'s
//! range) returns `None`, and the caller falls back to serde_json, which then also produces the
//! error messages.

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
        let spans: Vec<&[u8]> = arrays
            .iter()
            .map(|&(start, end)| &body[start..end])
            .collect();
        FastInput::Batch(numbers(&spans)?)
    } else {
        let (start, end) = s.array_span()?;
        FastInput::Single(numbers(&[&body[start..end]])?.pop()?)
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

/// The comma-separated numbers between the brackets of each array span. Many bytes in all are
/// cut into about one run per CPU and parsed on scoped threads.
fn numbers(spans: &[&[u8]]) -> Option<Vec<Vec<f32>>> {
    let total: usize = spans.iter().map(|span| span.len()).sum();
    if total < PARALLEL_MIN_BYTES {
        return spans.iter().map(|span| piece(span, true)).collect();
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let target = total.div_ceil(threads).max(PARALLEL_MIN_BYTES / 4);

    // Arrays longer than `target` are cut after commas into pieces of about `target` bytes
    let mut pieces: Vec<(usize, &[u8], bool)> = Vec::new();
    for (array, span) in spans.iter().enumerate() {
        if span.len() <= target {
            pieces.push((array, span, true));
            continue;
        }
        let mut start = 0;
        let mut at = target;
        while at < span.len() {
            let comma = at + span[at..].iter().position(|&c| c == b',')?;
            pieces.push((array, &span[start..comma], false));
            start = comma + 1;
            at = start + target;
        }
        pieces.push((array, &span[start..], false));
    }

    // Consecutive pieces of about `target` bytes in all make one thread's run
    let mut runs: Vec<&[(usize, &[u8], bool)]> = Vec::new();
    let (mut first, mut bytes) = (0, 0);
    for (i, &(_, text, _)) in pieces.iter().enumerate() {
        bytes += text.len();
        if bytes >= target {
            runs.push(&pieces[first..=i]);
            (first, bytes) = (i + 1, 0);
        }
    }
    if first < pieces.len() {
        runs.push(&pieces[first..]);
    }
    let run = |run: &[(usize, &[u8], bool)]| -> Option<Vec<Vec<f32>>> {
        run.iter()
            .map(|&(_, text, whole)| piece(text, whole))
            .collect()
    };
    let (mine, others) = runs.split_first()?;
    let parsed = std::thread::scope(|scope| {
        let others: Vec<_> = others
            .iter()
            .map(|&other| scope.spawn(move || run(other)))
            .collect();
        let mut parsed = run(mine)?;
        for other in others {
            parsed.extend(
                other
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))?,
            );
        }
        Some(parsed)
    })?;

    // Join each array's pieces back together, in order, each array grown once to its length
    let mut lens = vec![0; spans.len()];
    for (&(array, _, _), numbers) in pieces.iter().zip(&parsed) {
        lens[array] += numbers.len();
    }
    let mut arrays = vec![Vec::new(); spans.len()];
    for (&(array, _, _), numbers) in pieces.iter().zip(parsed) {
        if arrays[array].is_empty() {
            arrays[array] = numbers;
            let rest = lens[array] - arrays[array].len();
            arrays[array].reserve_exact(rest);
        } else {
            arrays[array].extend(numbers);
        }
    }
    Some(arrays)
}

/// The numbers of a whole array, which may be empty, or of a piece cut after a comma
fn piece(text: &[u8], whole: bool) -> Option<Vec<f32>> {
    let mut out = Vec::with_capacity(text.len() / 8);
    parse_list(text, &mut out, whole).then_some(out)
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
        // `json_number_len` took ASCII digits, signs, `.` and `e` only, so this is valid UTF-8
        let number = std::str::from_utf8(&text[i..i + len]).ok();
        match number.map(str::parse::<f32>) {
            Some(Ok(value)) if value.is_finite() => out.push(value),
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
        // Many short arrays, an empty one among them, past the parallel threshold in all
        let mut rows: Vec<String> = values[..100_000]
            .chunks(50)
            .map(|row| format!("[{}]", row.join(",")))
            .collect();
        rows.insert(7, "[ ]".to_string());
        let rows = format!("{{\"inputs\": [{}]}}", rows.join(", "));
        assert!(rows.len() > 4 * PARALLEL_MIN_BYTES);
        assert!(matches!(parse(rows.as_bytes()), Some(FastInput::Batch(b)) if b.len() == 2001));
        agree(&rows);
    }

    mod properties {
        use super::*;
        use proptest::prelude::*;

        /// A JSON number this parser takes, written one of several ways
        fn number() -> impl Strategy<Value = String> {
            prop_oneof![
                any::<i32>().prop_map(|n| n.to_string()),
                (-1e6f64..1e6).prop_map(|x| x.to_string()),
                (-1e30f32..1e30).prop_map(|x| format!("{x:e}")),
                (-1e30f32..1e30).prop_map(|x| format!("{x:E}")),
                (0u32..100_000_000, any::<bool>())
                    .prop_map(|(n, neg)| format!("{}0.{n:08}", if neg { "-" } else { "" })),
                (1u32..1000, -30i32..30).prop_map(|(m, e)| format!("{m}e{e:+}")),
                Just("-0".to_string()),
                Just("0".to_string()),
            ]
        }

        /// JSON whitespace, usually none
        fn ws() -> impl Strategy<Value = String> {
            prop::collection::vec(prop::sample::select(vec![" ", "\n", "\r", "\t"]), 0..3)
                .prop_map(|parts| parts.concat())
        }

        fn array(len: usize) -> impl Strategy<Value = String> {
            prop::collection::vec((ws(), number(), ws()), 0..len).prop_map(|items| {
                let items: Vec<String> = items
                    .into_iter()
                    .map(|(before, n, after)| format!("{before}{n}{after}"))
                    .collect();
                format!("[{}]", items.join(","))
            })
        }

        /// `{"input": [...]}` or `{"inputs": [[...], ...]}`, with whitespace around the parts
        fn body() -> impl Strategy<Value = String> {
            let single = (ws(), ws(), array(40))
                .prop_map(|(a, b, array)| format!("{a}{{{b}\"input\"{a}:{b}{array}{a}}}{b}"));
            let batch = (ws(), prop::collection::vec(array(12), 0..6)).prop_map(|(w, arrays)| {
                format!(
                    "{{\"inputs\"{w}:{w}[{w}{}{w}]}}",
                    arrays.join(&format!("{w},{w}"))
                )
            });
            prop_oneof![single, batch]
        }

        /// A byte of the kind JSON is made of, to break a body with
        fn json_byte() -> impl Strategy<Value = u8> {
            prop_oneof![
                prop::sample::select(b"[]{},:\" -+.eE0123456789 \n".to_vec()),
                any::<u8>(),
            ]
        }

        /// `body` broken at random places: bytes removed, replaced or inserted, or cut short
        fn broken() -> impl Strategy<Value = Vec<u8>> {
            (
                body(),
                prop::collection::vec((any::<prop::sample::Index>(), 0u8..3, json_byte()), 1..4),
                any::<Option<prop::sample::Index>>(),
            )
                .prop_map(|(body, edits, cut)| {
                    let mut bytes = body.into_bytes();
                    for (at, kind, byte) in edits {
                        let i = at.index(bytes.len() + 1);
                        match kind {
                            0 if i < bytes.len() => {
                                bytes.remove(i);
                            }
                            1 if i < bytes.len() => bytes[i] = byte,
                            _ => bytes.insert(i, byte),
                        }
                    }
                    if let Some(cut) = cut {
                        bytes.truncate(cut.index(bytes.len() + 1));
                    }
                    bytes
                })
        }

        /// The fast path took `body`: serde_json reads the same numbers from it
        fn agree_bytes(body: &[u8]) {
            if parse(body).is_some() {
                let text = std::str::from_utf8(body).expect("an accepted body is UTF-8");
                agree(text);
            }
        }

        proptest! {
            #[test]
            fn valid_bodies_parse_as_serde_json_does(body in body()) {
                // Every number generated fits an f32, so the fast path takes them all
                prop_assert!(parse(body.as_bytes()).is_some(), "{}", body);
                agree(&body);
            }

            #[test]
            fn broken_bodies_never_panic_and_never_disagree(body in broken()) {
                agree_bytes(&body);
            }

            #[test]
            fn any_bytes_never_panic(body in prop::collection::vec(json_byte(), 0..64)) {
                agree_bytes(&body);
            }
        }

        proptest! {
            // Bodies past the parallel threshold, each well past it: at most 16 cases;
            // PROPTEST_CASES lowers it
            #![proptest_config(ProptestConfig::with_cases(
                ProptestConfig::default().cases.min(16)
            ))]

            #[test]
            fn large_bodies_cut_into_pieces_parse_as_serde_json_does(
                values in prop::collection::vec(number(), 1..64),
                rows in 1usize..6,
                edit in any::<Option<(prop::sample::Index, u8)>>(),
            ) {
                // About 160 KB: the values repeated, split into `rows` arrays
                let row_bytes: usize = values.iter().map(|v| v.len() + 2).sum();
                let per_row = (160 * 1024 / rows) / row_bytes + 1;
                let row = vec![values.join(", "); per_row].join(",\n");
                let body = if rows == 1 {
                    format!("{{\"input\": [{row}]}}")
                } else {
                    format!("{{\"inputs\": [{}]}}", vec![format!("[{row}]"); rows].join(","))
                };
                prop_assert!(body.len() > PARALLEL_MIN_BYTES);
                let mut bytes = body.into_bytes();
                match edit {
                    None => {
                        prop_assert!(parse(&bytes).is_some());
                    }
                    // One byte replaced somewhere, maybe a piece's cut point
                    Some((at, byte)) => {
                        let i = at.index(bytes.len());
                        bytes[i] = byte;
                    }
                }
                agree_bytes(&bytes);
            }
        }
    }
}
