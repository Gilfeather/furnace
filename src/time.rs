//! Timestamps as the server reports them, in RFC 3339 and UTC: `2026-10-21T05:00:00.123+00:00`
//! in responses and logs, `2026-10-21T05:00:00.123Z` for [`ModelStats`](crate::model::ModelStats)
//! in JSON. The fraction has as many digits as it needs of 0, 3, 6 or 9.

use std::fmt::Write as _;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The current time, e.g. `2026-10-21T05:00:00.123456+00:00`
pub fn rfc3339_now() -> String {
    rfc3339(SystemTime::now(), "+00:00")
}

/// `time` in RFC 3339, UTC, with `zone` (`+00:00` or `Z`) after the seconds. A time before 1970
/// is written as 1970.
fn rfc3339(time: SystemTime, zone: &str) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since.as_secs();
    let (year, month, day) = civil_from_days((secs / 86_400) as i64);
    let of_day = secs % 86_400;
    let mut out = format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}",
        of_day / 3600,
        of_day / 60 % 60,
        of_day % 60
    );
    let nanos = since.subsec_nanos();
    let _ = match nanos {
        0 => Ok(()),
        n if n % 1_000_000 == 0 => write!(out, ".{:03}", n / 1_000_000),
        n if n % 1_000 == 0 => write!(out, ".{:06}", n / 1_000),
        n => write!(out, ".{n:09}"),
    };
    out.push_str(zone);
    out
}

/// Parse an RFC 3339 timestamp (`2026-10-21T05:00:00.5Z`, `...+09:00`, `t` and `z` too)
fn parse_rfc3339(text: &str) -> Option<SystemTime> {
    let bytes = text.as_bytes();
    let number = |range: std::ops::Range<usize>| -> Option<u64> {
        let digits = text.get(range)?;
        digits
            .bytes()
            .all(|b| b.is_ascii_digit())
            .then(|| digits.parse().ok())?
    };
    if bytes.len() < 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || !matches!(bytes[10], b'T' | b't' | b' ')
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return None;
    }
    let (year, month, day) = (number(0..4)?, number(5..7)?, number(8..10)?);
    let (hour, minute, second) = (number(11..13)?, number(14..16)?, number(17..19)?);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 {
        return None;
    }
    // A leap second (:60) counts as the last second of its minute
    let second = second.min(59);
    let mut rest = &text[19..];
    let mut nanos = 0u32;
    if let Some(fraction) = rest.strip_prefix('.') {
        let digits = fraction.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return None;
        }
        for (i, b) in fraction
            .bytes()
            .take(9)
            .take_while(u8::is_ascii_digit)
            .enumerate()
        {
            nanos += u32::from(b - b'0') * 10u32.pow(8 - i as u32);
        }
        rest = &fraction[digits..];
    }
    let offset_secs: i64 = match rest.as_bytes() {
        [b'Z' | b'z'] => 0,
        [sign @ (b'+' | b'-'), _, _, b':', _, _] => {
            let hours = number(text.len() - 5..text.len() - 3)? as i64;
            let minutes = number(text.len() - 2..text.len())? as i64;
            let offset = hours * 3600 + minutes * 60;
            if *sign == b'+' { offset } else { -offset }
        }
        _ => return None,
    };
    let days = days_from_civil(year as i64, month as u32, day as u32);
    let secs = days * 86_400 + (hour * 3600 + minute * 60 + second) as i64 - offset_secs;
    let secs = u64::try_from(secs).ok()?;
    Some(UNIX_EPOCH + Duration::new(secs, nanos))
}

/// The year, month and day `days` after 1970-01-01, in the proleptic Gregorian calendar
/// (Howard Hinnant's `civil_from_days`)
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * shifted_month + 2) / 5 + 1) as u32;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Days from 1970-01-01 to the given date (Howard Hinnant's `days_from_civil`)
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let shifted_month = i64::from((month + 9) % 12);
    let day_of_year = (153 * shifted_month + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// `#[serde(with = "crate::time::serde_utc")]` for an `Option<SystemTime>`: RFC 3339 with `Z`
pub(crate) mod serde_utc {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::SystemTime;

    pub fn serialize<S: Serializer>(time: &Option<SystemTime>, s: S) -> Result<S::Ok, S::Error> {
        match time {
            Some(time) => s.serialize_some(&super::rfc3339(*time, "Z")),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<SystemTime>, D::Error> {
        Option::<String>::deserialize(d)?
            .map(|text| {
                super::parse_rfc3339(&text).ok_or_else(|| {
                    serde::de::Error::custom(format!("not an RFC 3339 timestamp: {text}"))
                })
            })
            .transpose()
    }
}

/// `time` as [`rfc3339_now`] writes the current time
pub(crate) fn rfc3339_utc(time: SystemTime) -> String {
    rfc3339(time, "+00:00")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(secs: u64, nanos: u32) -> SystemTime {
        UNIX_EPOCH + Duration::new(secs, nanos)
    }

    #[test]
    fn formats_like_rfc3339() {
        // Nanoseconds in multiples of 100: Windows' clock has no finer steps
        assert_eq!(rfc3339(at(0, 0), "+00:00"), "1970-01-01T00:00:00+00:00");
        // 2026-10-21T05:06:07Z
        let t = 1_792_559_167;
        assert_eq!(rfc3339(at(t, 0), "Z"), "2026-10-21T05:06:07Z");
        assert_eq!(rfc3339(at(t, 120_000_000), "Z"), "2026-10-21T05:06:07.120Z");
        assert_eq!(
            rfc3339(at(t, 120_400_000), "Z"),
            "2026-10-21T05:06:07.120400Z"
        );
        assert_eq!(
            rfc3339(at(t, 120_400_100), "Z"),
            "2026-10-21T05:06:07.120400100Z"
        );
        // A leap day and the last second of a century's leap year
        assert_eq!(rfc3339(at(951_782_400, 0), "Z"), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(at(978_307_199, 0), "Z"), "2000-12-31T23:59:59Z");
        assert_eq!(rfc3339(at(4_107_542_400, 0), "Z"), "2100-03-01T00:00:00Z");
    }

    #[test]
    fn parses_what_it_writes_and_other_offsets() {
        for (secs, nanos) in [
            (0, 0),
            (1_792_559_167, 120_400_100),
            (951_782_400, 5_000_000),
        ] {
            let time = at(secs, nanos);
            assert_eq!(parse_rfc3339(&rfc3339(time, "Z")), Some(time));
            assert_eq!(parse_rfc3339(&rfc3339(time, "+00:00")), Some(time));
        }
        assert_eq!(
            parse_rfc3339("2026-10-21T14:06:07.5+09:00"),
            Some(at(1_792_559_167, 500_000_000))
        );
        assert_eq!(
            parse_rfc3339("2026-10-20T23:06:07-06:00"),
            Some(at(1_792_559_167, 0))
        );
        for bad in [
            "",
            "2026-10-21",
            "2026-10-21T05:06:07",
            "2026-13-01T00:00:00Z",
            "2026-10-21T05:06:07.Z",
            "2026-10-21T05:06:07+0900",
        ] {
            assert_eq!(parse_rfc3339(bad), None, "{bad}");
        }
    }

    #[test]
    fn calendar_round_trips() {
        for days in [-719_468, -1, 0, 1, 10_957, 11_016, 20_747, 47_541] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days, "{y}-{m}-{d}");
        }
    }
}
