#![cfg(feature = "std")]

use std::{
    fs,
    io::Read as _,
    path::{Path, PathBuf},
};

use flate2::read::GzDecoder;
use serde::de::DeserializeOwned;

pub fn fixture_path(relative_path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(relative_path)
}

pub fn load_fixture_json<T: DeserializeOwned>(relative_path: &str) -> T {
    let path = fixture_path(relative_path);
    let json = if matches!(path.extension().and_then(|extension| extension.to_str()), Some("gz")) {
        let bytes =
            fs::read(&path).unwrap_or_else(|_| panic!("failed to read fixture {}", path.display()));
        let mut json = String::new();
        GzDecoder::new(bytes.as_slice())
            .read_to_string(&mut json)
            .unwrap_or_else(|_| panic!("failed to decompress fixture {}", path.display()));
        json
    } else {
        fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("failed to read fixture {}", path.display()))
    };
    serde_json::from_str(&json)
        .unwrap_or_else(|_| panic!("`{}` must contain valid JSON", path.display()))
}
