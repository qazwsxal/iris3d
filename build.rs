use std::path::PathBuf;

/// Root of the proto tree, shared with `buf` — see `buf.yaml` at the repository
/// root.
const PROTO_ROOT: &str = "proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed={PROTO_ROOT}/");

    // Discover the tree rather than listing files, so that adding a `.proto`
    // compiles it into the app automatically. `buf` already discovers the whole
    // tree; a hand-maintained list here would silently drift out of step with
    // the generated client libraries.
    let protos: Vec<PathBuf> =
        glob::glob(&format!("{PROTO_ROOT}/**/*.proto"))?.collect::<Result<_, _>>()?;

    if protos.is_empty() {
        return Err(format!("no .proto files found under {PROTO_ROOT}").into());
    }

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&protos, &[PathBuf::from(PROTO_ROOT)])?;

    Ok(())
}
