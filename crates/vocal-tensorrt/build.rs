mod build_support;

use std::{
    env, fs,
    hash::{Hash, Hasher},
    path::PathBuf,
    process::Command,
};

fn main() {
    let windows = target_os_is_windows();
    for variable in [
        "TENSORRT_RTX_ROOT",
        "TRT_RTX_ROOT",
        "TENSORRT_ROOT",
        "TRT_ROOT",
        "CUDA_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    let r = env::var_os("TENSORRT_RTX_ROOT")
        .or_else(|| env::var_os("TRT_RTX_ROOT"))
        .or_else(|| env::var_os("TENSORRT_ROOT"))
        .or_else(|| env::var_os("TRT_ROOT"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if windows {
                PathBuf::from(r"C:\Program Files\NVIDIA Corporation\TensorRT-RTX\1.6.1.120")
            } else {
                PathBuf::from("/opt/TensorRT-RTX")
            }
        });
    let c = env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            if windows {
                PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4")
            } else {
                PathBuf::from("/usr/local/cuda")
            }
        });
    // Native graphs are statically linked: a nonexistent DLL's timestamp
    // cannot invalidate plans after graph fixes. Hash builder sources instead.
    let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
    for file in [
        "native/bridge.h",
        "native/common.cpp",
        "native/onnx.cpp",
        "native/hubert.cpp",
        "native/rmvpe.cpp",
        "native/frontend.cpp",
        "native/plugin.cpp",
        "native/serialization.cpp",
        "native/graph/layers.h",
        "native/graph/layers.cpp",
        "native/graph/encoder.cpp",
        "native/graph/flow.cpp",
        "native/graph/decoder.cpp",
        "native/graph/build.cpp",
        "src/abi.rs",
        "src/builder.rs",
        "src/acoustic.rs",
        "src/graph.rs",
    ] {
        println!("cargo:rerun-if-changed={file}");
        file.hash(&mut fingerprint);
        fs::read(file)
            .expect("builder source exists")
            .hash(&mut fingerprint);
    }
    let (runtime, parser, runtime_dir) = if windows {
        (
            "tensorrt_rtx_1_6",
            "tensorrt_onnxparser_rtx_1_6",
            r.join("bin"),
        )
    } else {
        ("tensorrt_rtx", "tensorrt_onnxparser_rtx", r.join("lib"))
    };
    let library = |name: &str| {
        runtime_dir.join(if windows {
            format!("{name}.dll")
        } else {
            format!("lib{name}.so")
        })
    };
    for file in [
        r.join("include/NvInferVersion.h"),
        library(runtime),
        library(parser),
    ] {
        println!("cargo:rerun-if-changed={}", file.display());
        if file.extension().is_some_and(|ext| ext == "h") {
            fs::read(&file).ok().hash(&mut fingerprint);
        } else if let Ok(metadata) = fs::metadata(file) {
            metadata.len().hash(&mut fingerprint);
            metadata.modified().ok().hash(&mut fingerprint);
        }
    }
    println!(
        "cargo:rustc-env=VOCAL_TRT_BUILDER_FINGERPRINT={:016x}",
        fingerprint.finish()
    );
    for file in [
        "native/bridge.h",
        "native/nvtx.h",
        "native/common.cpp",
        "native/cuda.cpp",
        "native/runtime.cpp",
        "native/onnx.cpp",
        "native/hubert.cpp",
        "native/rmvpe.cpp",
        "native/frontend.cpp",
        "native/serialization.cpp",
        "native/plugin.cpp",
        "native/retrieval.cu",
        "native/retrieval_protection.cu",
        "native/pitch_matching.cu",
        "native/retrieval_block.cuh",
        "native/retrieval_kernels.cuh",
        "native/cufft_frontend.cu",
        "native/graph/layers.h",
        "native/graph/layers.cpp",
        "native/graph/encoder.cpp",
        "native/graph/flow.cpp",
        "native/graph/decoder.cpp",
        "native/graph/build.cpp",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }
    println!("cargo:rustc-link-search=native={}", r.join("lib").display());
    println!(
        "cargo:rustc-link-search=native={}",
        c.join(if windows { "lib/x64" } else { "lib64" }).display()
    );
    println!("cargo:rustc-link-lib=dylib={runtime}");
    println!("cargo:rustc-link-lib=dylib={parser}");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cufft");
    println!("cargo:rerun-if-env-changed=VOCAL_TRT_COPY_DLLS");
    println!("cargo:rerun-if-env-changed=VOCAL_TRT_DLL_DIR");
    if windows && build_support::enabled() {
        let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
        build_support::deploy(&r, &c, &out_dir);
    }
    cc::Build::new()
        .cpp(true)
        .files([
            "native/common.cpp",
            "native/cuda.cpp",
            "native/runtime.cpp",
            "native/onnx.cpp",
            "native/hubert.cpp",
            "native/rmvpe.cpp",
            "native/frontend.cpp",
            "native/serialization.cpp",
            "native/plugin.cpp",
            "native/graph/layers.cpp",
            "native/graph/encoder.cpp",
            "native/graph/flow.cpp",
            "native/graph/decoder.cpp",
            "native/graph/build.cpp",
        ])
        .include(r.join("include"))
        .include(c.join("include"))
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("vocal_tensorrt_native");

    // cuFFT is a CUDA translation unit rather than a TensorRT plugin. Compile
    // it with nvcc into a small static library so the Rust crate only links a
    // stable C ABI. The plan and scratch allocations are created at runtime,
    // before any CUDA graph capture begins.
    let nvcc = c.join(if windows { "bin/nvcc.exe" } else { "bin/nvcc" });
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    let cufft_lib = out_dir.join(if windows {
        "vocal_tensorrt_cufft.lib"
    } else {
        "libvocal_tensorrt_cufft.a"
    });
    let mut compile = Command::new(&nvcc);
    if windows {
        let cl = find_msvc_cl().unwrap_or_else(|| {
            panic!("nvcc requires cl.exe; set CXX or run cargo from a VS developer shell")
        });
        compile
            .arg("-ccbin")
            .arg(cl)
            .args(["-Xcompiler=/EHsc", "-Xcompiler=/MD"]);
    } else {
        let compiler = cc::Build::new().cpp(true).get_compiler();
        compile
            .arg("-ccbin")
            .arg(compiler.path())
            .arg("-Xcompiler=-fPIC");
    }
    println!("cargo:rerun-if-env-changed=VOCAL_CUDA_ARCHS");
    let archs = env::var("VOCAL_CUDA_ARCHS").unwrap_or_else(|_| "75;80;86;89;90;120".into());
    for arch in archs.split(';') {
        assert!(
            arch.len() >= 2 && arch.bytes().all(|c| c.is_ascii_digit()),
            "invalid CUDA architecture {arch}"
        );
        compile.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
    let status = compile
        .arg("-lib")
        .arg("-std=c++17")
        .arg("-I")
        .arg("native")
        .arg("-I")
        .arg(r.join("include"))
        .arg("-I")
        .arg(c.join("include"))
        .arg("native/cufft_frontend.cu")
        .arg("native/retrieval.cu")
        .arg("native/retrieval_protection.cu")
        .arg("native/pitch_matching.cu")
        .arg("-o")
        .arg(&cufft_lib)
        .status()
        .unwrap_or_else(|error| panic!("failed to launch nvcc {}: {error}", nvcc.display()));
    if !status.success() {
        panic!("nvcc failed to build cuFFT frontend ({status})");
    }
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=vocal_tensorrt_cufft");
}

fn target_os_is_windows() -> bool {
    env::var("CARGO_CFG_TARGET_OS").is_ok_and(|value| value.eq_ignore_ascii_case("windows"))
}

fn find_msvc_cl() -> Option<PathBuf> {
    // Reuse cc-rs discovery, including CXX overrides and VS installations
    // nested under Community/Enterprise on hosted runners. nvcc must use the
    // same host compiler as the C++ bridge, rather than a guessed VS version.
    let compiler = cc::Build::new().cpp(true).get_compiler();
    let path = compiler.path();
    (compiler.is_like_msvc()
        && path
            .file_stem()
            .is_some_and(|name| name.eq_ignore_ascii_case("cl"))
        && path.is_file())
    .then(|| path.to_path_buf())
}
