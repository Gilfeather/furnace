/// Normalized form used to match `--model-name` against registry names:
/// case-insensitive, ignoring `-`, `_` and `.` (so `resnet-18` finds `resnet18`).
///
/// Included by src/models/mod.rs and by build/spec.rs, which drops models whose names collide
/// after normalization, so the two always agree.
fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|c| !matches!(c, '-' | '_' | '.'))
        .flat_map(char::to_lowercase)
        .collect()
}
