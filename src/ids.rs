use sha1::{Digest, Sha1};

/// Build a deterministic, compact ID from semantic parts.
///
/// This is intended for graph node IDs and replayable selection jitter. It is
/// not a security primitive.
pub fn stable_id<'a>(prefix: &str, parts: impl IntoIterator<Item = &'a str>) -> String {
    let mut hasher = Sha1::new();
    hasher.update(prefix.as_bytes());
    hasher.update([0]);
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update([0]);
    }
    let digest = hasher.finalize();
    format!("{prefix}:{}", hex_prefix(&digest, 16))
}

pub(crate) fn stable_unit(seed: u64, label: &str) -> f64 {
    let mut hasher = Sha1::new();
    hasher.update(seed.to_be_bytes());
    hasher.update(label.as_bytes());
    let digest = hasher.finalize();

    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    let value = u64::from_be_bytes(bytes);

    // Keep the value in (0, 1) so log/softmax sampling cannot hit ln(0).
    ((value as f64) + 1.0) / ((u64::MAX as f64) + 2.0)
}

fn hex_prefix(bytes: &[u8], chars: usize) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(chars);
    for byte in bytes {
        if out.len() >= chars {
            break;
        }
        out.push(HEX[(byte >> 4) as usize] as char);
        if out.len() >= chars {
            break;
        }
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_id_is_repeatable() {
        let a = stable_id("exp", ["root", "try smaller model"]);
        let b = stable_id("exp", ["root", "try smaller model"]);
        assert_eq!(a, b);
        assert!(a.starts_with("exp:"));
    }

    #[test]
    fn stable_unit_is_in_open_unit_interval() {
        let value = stable_unit(42, "exp_0001");
        assert!(value > 0.0);
        assert!(value < 1.0);
    }
}
