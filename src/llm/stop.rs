//! Stop strings over streamed text: a match may span any number of pieces, so the end of the
//! text that could still become a match is held back until the next piece decides it.

/// Finds stop strings in text that arrives in pieces
#[derive(Debug, Clone, Default)]
pub struct StopStrings {
    stops: Vec<String>,
    /// Text not emitted yet: the end of what arrived, while it is the start of a stop string
    held: String,
}

impl StopStrings {
    /// Empty stop strings are ignored
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            held: String::new(),
        }
    }

    /// Add the next piece. Returns the text that is safe to emit, and whether a stop string was
    /// found; the text then ends where the match began, and nothing after it counts.
    pub fn push(&mut self, piece: String) -> (String, bool) {
        if self.stops.is_empty() {
            return (piece, false);
        }
        self.held.push_str(&piece);
        // What was held matched nothing, so the earliest match ends in the new piece
        if let Some(start) = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()))
            .min()
        {
            self.held.truncate(start);
            return (std::mem::take(&mut self.held), true);
        }
        // Hold back the longest end of the text that begins some stop string
        let keep_from = self
            .held
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| {
                let tail = &self.held[i..];
                self.stops.iter().any(|stop| stop.starts_with(tail))
            })
            .unwrap_or(self.held.len());
        let held = self.held.split_off(keep_from);
        (std::mem::replace(&mut self.held, held), false)
    }

    /// The text held back, at the end of the output (it turned out not to be a stop string)
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (Vec<String>, bool, String) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut matcher = StopStrings::new(&stops);
        let mut out = Vec::new();
        for piece in pieces {
            let (text, hit) = matcher.push(piece.to_string());
            out.push(text);
            if hit {
                return (out, true, String::new());
            }
        }
        (out, false, matcher.finish())
    }

    #[test]
    fn without_stop_strings_text_passes_through() {
        let (out, hit, rest) = run(&[], &["Hel", "lo"]);
        assert_eq!(out, ["Hel", "lo"]);
        assert!(!hit);
        assert_eq!(rest, "");
    }

    #[test]
    fn a_match_inside_one_piece_cuts_the_text() {
        let (out, hit, _) = run(&["STOP"], &["abcSTOPdef"]);
        assert_eq!(out, ["abc"]);
        assert!(hit);
    }

    #[test]
    fn a_match_across_pieces_is_found_and_never_emitted() {
        let (out, hit, _) = run(&["</answer>"], &["42 <", "/ans", "wer", "> more"]);
        assert_eq!(out.concat(), "42 ");
        assert!(hit);
        // The `<` was held back as soon as it could start the stop string
        assert_eq!(out[0], "42 ");
    }

    #[test]
    fn a_partial_match_that_breaks_is_released() {
        let (out, hit, rest) = run(&["\n\n"], &["a\n", "b\n", "c"]);
        assert!(!hit);
        assert_eq!(out, ["a", "\nb", "\nc"]);
        assert_eq!(rest, "");
    }

    #[test]
    fn held_text_comes_out_at_the_end() {
        let (out, hit, rest) = run(&["###"], &["x#", "#"]);
        assert!(!hit);
        assert_eq!(out.concat(), "x");
        assert_eq!(rest, "##");
    }

    #[test]
    fn the_earliest_of_several_stop_strings_wins() {
        let (out, hit, _) = run(&["cd", "b"], &["a", "bcd"]);
        assert!(hit);
        assert_eq!(out.concat(), "a");
    }

    #[test]
    fn multibyte_characters_are_never_split() {
        let (out, hit, _) = run(&["é!"], &["café", "!"]);
        assert!(hit);
        assert_eq!(out.concat(), "caf");
        let (out, hit, rest) = run(&["éa"], &["café", "b"]);
        assert!(!hit);
        assert_eq!(out.concat() + &rest, "caféb");
    }
}
