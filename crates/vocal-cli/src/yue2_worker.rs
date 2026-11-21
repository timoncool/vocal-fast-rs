use std::{
    fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use serde_json::json;
#[cfg(not(feature = "tensorrt-rtx"))]
use vocal_cpu::CpuConfig as BackendConfig;
#[cfg(not(feature = "tensorrt-rtx"))]
use vocal_engine::offline::CpuOfflinePipeline as Pipeline;
#[cfg(feature = "tensorrt-rtx")]
use vocal_engine::offline::TensorRtOfflinePipeline as Pipeline;
use vocal_engine::offline::{AlignedOfflineConfig, OfflineModelSources, run_aligned_offline};
use vocal_model::{RetrievalIndex, Tensor, load_rvc_pth_auto, write_cache};
#[cfg(feature = "tensorrt-rtx")]
use vocal_tensorrt::TensorRtConfig as BackendConfig;

use crate::CliResult;

const PROTOCOL: u32 = 1;

#[derive(Deserialize)]
struct Request {
    vocal: PathBuf,
    voice: PathBuf,
    index: Option<PathBuf>,
    voice_sha256: String,
    index_sha256: Option<String>,
    output: PathBuf,
    semitone_shift: i32,
    retrieval_blend: f64,
}

pub fn run(models: &Path, cache: &Path) -> CliResult<()> {
    let [hubert, rmvpe] = ["hubert.vfrm", "rmvpe.vfrm"].map(|name| models.join(name));
    for model in [&hubert, &rmvpe] {
        if !model.is_file() {
            return Err(format!("missing acoustic model {}", model.display()).into());
        }
    }
    #[cfg(feature = "tensorrt-rtx")]
    let (backend, backend_config, builder) = {
        vocal_tensorrt::CudaStream::new()?.synchronize()?;
        let mut trt = BackendConfig::from_env()?;
        trt.build.workspace_bytes = 2 << 30;
        trt.build.rmvpe_fft_cufft = true;
        trt.build.rmvpe_fft_radix2 = false;
        (
            "tensorrt-rtx",
            trt,
            json!(vocal_tensorrt::BUILDER_FINGERPRINT),
        )
    };
    #[cfg(not(feature = "tensorrt-rtx"))]
    let (backend, backend_config, builder) = {
        let cpu = BackendConfig::default();
        cpu.validate().map_err(|error| error.to_string())?;
        ("cpu-graph", cpu, json!(vocal_cpu::operator_version()))
    };
    fs::create_dir_all(cache)?;
    let mut worker = Worker {
        hubert,
        rmvpe,
        cache,
        backend_config,
        pipeline: None,
    };
    let mut output = io::stdout().lock();
    writeln!(
        output,
        "{}",
        json!({
            "protocol": PROTOCOL,
            "backend": backend,
            "version": env!("CARGO_PKG_VERSION"),
            "builder": builder,
        })
    )?;
    output.flush()?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let result = serde_json::from_str(&line)
            .map_err(Into::into)
            .and_then(|request| worker.convert(request));
        match result {
            Ok(details) => writeln!(output, "{}", json!({"ok": true, "details": details}))?,
            Err(error) => writeln!(
                output,
                "{}",
                json!({"ok": false, "error": error.to_string()})
            )?,
        }
        output.flush()?;
    }
    Ok(())
}

struct Worker<'a> {
    hubert: PathBuf,
    rmvpe: PathBuf,
    cache: &'a Path,
    backend_config: BackendConfig,
    pipeline: Option<(String, Pipeline)>,
}

impl Worker<'_> {
    fn convert(&mut self, request: Request) -> CliResult<serde_json::Value> {
        if !(-24..=24).contains(&request.semitone_shift)
            || !(0.0..=1.0).contains(&request.retrieval_blend)
        {
            return Err("invalid RVC conversion controls".into());
        }
        if request.index.is_some() != request.index_sha256.is_some()
            || !std::iter::once(&request.voice_sha256)
                .chain(request.index_sha256.iter())
                .all(|sha| sha.len() == 64 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err("invalid RVC voice or index hash".into());
        }
        let mut id = request.voice_sha256.to_ascii_lowercase();
        if let Some(index) = &request.index_sha256 {
            id.push('-');
            id.push_str(&index.to_ascii_lowercase());
        }
        if self.pipeline.as_ref().is_none_or(|(key, _)| key != &id) {
            self.pipeline = None;
            let synth = self.cache.join(format!("{id}.vfrm"));
            if !synth.is_file() {
                import_voice(&request, &synth)?;
            }
            let sources = OfflineModelSources {
                synth,
                hubert: self.hubert.clone(),
                rmvpe: self.rmvpe.clone(),
            };
            #[cfg(feature = "tensorrt-rtx")]
            let pipeline =
                Pipeline::new(sources, self.cache.to_owned(), self.backend_config.clone())?;
            #[cfg(not(feature = "tensorrt-rtx"))]
            let pipeline = Pipeline::new(sources, self.backend_config.clone())?;
            self.pipeline = Some((id, pipeline));
        }
        let mut config = AlignedOfflineConfig::default();
        config.conversion.pitch_semitones = request.semitone_shift as f32;
        config.conversion.auto_pitch.enabled = false;
        config.conversion.index_ratio = if request.index.is_some() {
            request.retrieval_blend as f32
        } else {
            0.
        };
        config.conversion.protect = 0.33;
        config.conversion.rms_mix_rate = 0.25;
        let samples = vocal_audio::read_wav(&request.vocal)?.mono_16k();
        let pipeline = &mut self.pipeline.as_mut().expect("voice prepared").1;
        let (wave, details) = run_aligned_offline(pipeline, &samples, &config)?;
        if wave.is_empty() || wave.iter().any(|sample| !sample.is_finite()) {
            return Err("native RVC returned invalid audio".into());
        }
        vocal_audio::write_wav_pcm16(&request.output, details.output_sample_rate, &wave)?;
        Ok(json!({
            "offline": details,
            "retrieval_blend": config.conversion.index_ratio,
        }))
    }
}

fn import_voice(request: &Request, output: &Path) -> CliResult<()> {
    let name = request
        .voice
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("RVC voice");
    let mut package = load_rvc_pth_auto(&request.voice, name, "yue2-installed-voice")?;
    if let Some(index) = &request.index {
        if fs::metadata(index)?.len() > 2 * 1024 * 1024 * 1024 {
            return Err("retrieval index exceeds 2 GiB".into());
        }
        package
            .tensors
            .push(Tensor::byte_tensor("aux.retrieval.faiss", fs::read(index)?));
        package.manifest.index_file = index.file_name().map(|name| name.to_string_lossy().into());
        RetrievalIndex::from_package(&package)?;
    }
    package.validate_rvc_synth()?;
    let temp = output.with_extension(format!("{}.tmp.vfrm", std::process::id()));
    let result = write_cache(&temp, &package)
        .map_err(Into::into)
        .and_then(|()| {
            if output.is_file() {
                return Ok(());
            }
            fs::rename(&temp, output).map_err(Into::into)
        });
    let _ = fs::remove_file(temp);
    result
}
