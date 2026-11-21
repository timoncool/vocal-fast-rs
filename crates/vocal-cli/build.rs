use std::env;

fn main() {
    if env::var_os("CARGO_FEATURE_CPU_GRAPH").is_none()
        && env::var_os("CARGO_FEATURE_TENSORRT_RTX").is_none()
    {
        return;
    }
    let runtime_dir = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => "@executable_path",
        Ok("linux") => "$ORIGIN",
        _ => return,
    };
    println!("cargo:rustc-link-arg-bin=vocal-fast=-Wl,-rpath,{runtime_dir}");
}
