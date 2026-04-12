use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Context, Result};
use ffmpeg_next as ffmpeg;
use log::{debug, trace, warn};
use opus_decoder::OpusDecoder as PureOpusDecoder;
use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};

const OGG_OPUS_PROBE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub(crate) enum AudioSource {
    Path(PathBuf, Option<String>),
    Bytes(Vec<u8>, Option<String>),
}

impl AudioSource {
    pub(crate) fn log_label(&self) -> String {
        match self {
            Self::Path(path, Some(path_hint)) => {
                format!("path={} path_hint={path_hint}", path.display())
            }
            Self::Path(path, None) => format!("path={}", path.display()),
            Self::Bytes(bytes, Some(path_hint)) => {
                format!("bytes={} path_hint={path_hint}", bytes.len())
            }
            Self::Bytes(bytes, None) => format!("bytes={}", bytes.len()),
        }
    }
}

pub(crate) fn decode_audio(
    source: AudioSource,
    fallback_sample_rate: u32,
    ffmpeg_fallback: bool,
) -> Result<(Vec<f32>, u32)> {
    let source_label = source.log_label();
    debug!(
        "decoding audio source={} fallback_sample_rate={} ffmpeg_fallback={}",
        source_label, fallback_sample_rate, ffmpeg_fallback
    );
    if source_looks_like_ogg_opus(&source)? {
        debug!(
            "decoding Ogg/Opus source={} with opus-decoder before Symphonia",
            source_label
        );
        return decode_audio_with_opus_decoder_or_ffmpeg(
            source,
            fallback_sample_rate,
            ffmpeg_fallback,
            None,
        );
    }
    match decode_audio_symphonia(source.clone()) {
        Ok(decoded) => {
            debug!(
                "decoded audio with symphonia source={} samples={} sample_rate={}",
                source_label,
                decoded.0.len(),
                decoded.1
            );
            Ok(decoded)
        }
        Err(symphonia_error) => {
            if source_looks_like_ogg_opus(&source)? {
                return decode_audio_with_opus_decoder_or_ffmpeg(
                    source,
                    fallback_sample_rate,
                    ffmpeg_fallback,
                    Some(symphonia_error),
                );
            }
            if ffmpeg_fallback {
                warn!(
                    "symphonia decode failed for {}; falling back to FFmpeg libraries: {symphonia_error:#}",
                    source_label
                );
                decode_audio_ffmpeg(source, fallback_sample_rate).with_context(|| {
                    format!("symphonia decode failed: {symphonia_error:#}; FFmpeg fallback failed")
                })
            } else {
                Err(symphonia_error)
            }
        }
    }
}

fn decode_audio_symphonia(source: AudioSource) -> Result<(Vec<f32>, u32)> {
    let (mss, extension) = match source {
        AudioSource::Path(path, path_hint) => {
            trace!("opening audio file with symphonia path={}", path.display());
            let extension = path_hint
                .as_deref()
                .and_then(|path| Path::new(path).extension())
                .or_else(|| path.extension())
                .and_then(|value| value.to_str())
                .map(str::to_owned);
            let file = File::open(&path)
                .with_context(|| format!("failed to open audio file {}", path.display()))?;
            (
                MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default()),
                extension,
            )
        }
        AudioSource::Bytes(bytes, path_hint) => {
            let extension = path_hint
                .as_deref()
                .and_then(|path| Path::new(path).extension())
                .and_then(|value| value.to_str())
                .map(str::to_owned);
            (
                MediaSourceStream::new(
                    Box::new(Cursor::new(bytes)),
                    MediaSourceStreamOptions::default(),
                ),
                extension,
            )
        }
    };

    let mut hint = Hint::new();
    if let Some(extension) = extension.as_deref() {
        hint.with_extension(extension);
    }
    let probed = get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow!("audio container has no default track"))?;
    if track.codec_params.codec == CODEC_TYPE_NULL {
        bail!("unsupported null audio codec");
    }
    let track_id = track.id;
    debug!(
        "symphonia selected track id={} codec={:?} sample_rate={:?}",
        track_id, track.codec_params.codec, track.codec_params.sample_rate
    );
    let mut decoder = get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let mut mono = Vec::new();
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(16_000);

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(error))
                if error.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::ResetRequired) => {
                bail!("decoder reset is not supported for this audio stream");
            }
            Err(error) => return Err(error.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(message)) => {
                trace!("symphonia skipped packet decode error: {message}");
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        append_mono_samples(decoded, &mut mono, &mut sample_rate);
    }

    if mono.is_empty() {
        bail!("decoded audio stream is empty");
    }
    Ok((mono, sample_rate))
}

fn decode_audio_with_opus_decoder_or_ffmpeg(
    source: AudioSource,
    fallback_sample_rate: u32,
    ffmpeg_fallback: bool,
    symphonia_error: Option<anyhow::Error>,
) -> Result<(Vec<f32>, u32)> {
    let source_label = source.log_label();
    match decode_audio_opus_decoder(source.clone()) {
        Ok(decoded) => {
            debug!(
                "decoded audio with opus-decoder source={} samples={} sample_rate={}",
                source_label,
                decoded.0.len(),
                decoded.1
            );
            Ok(decoded)
        }
        Err(opus_error) if ffmpeg_fallback => {
            if let Some(symphonia_error) = symphonia_error {
                warn!(
                    "symphonia decode failed for {}; opus-decoder fallback also failed; falling back to FFmpeg libraries: symphonia={symphonia_error:#}; opus-decoder={opus_error:#}",
                    source_label
                );
                decode_audio_ffmpeg(source, fallback_sample_rate).with_context(|| {
                    format!(
                        "symphonia decode failed: {symphonia_error:#}; opus-decoder fallback failed: {opus_error:#}; FFmpeg fallback failed"
                    )
                })
            } else {
                warn!(
                    "opus-decoder fallback failed for {}; falling back to FFmpeg libraries: {opus_error:#}",
                    source_label
                );
                decode_audio_ffmpeg(source, fallback_sample_rate).with_context(|| {
                    format!("opus-decoder fallback failed: {opus_error:#}; FFmpeg fallback failed")
                })
            }
        }
        Err(opus_error) => {
            if let Some(symphonia_error) = symphonia_error {
                Err(opus_error).with_context(|| {
                    format!(
                        "symphonia decode failed: {symphonia_error:#}; opus-decoder fallback failed"
                    )
                })
            } else {
                Err(opus_error).context("opus-decoder fallback failed")
            }
        }
    }
}

fn decode_audio_opus_decoder(source: AudioSource) -> Result<(Vec<f32>, u32)> {
    debug!(
        "decoding audio with opus-decoder source={}",
        source.log_label()
    );
    let bytes = source_bytes(&source)?;
    let stream = parse_ogg_opus_stream(&bytes)?;
    let mut decoder = PureOpusDecoder::new(48_000, stream.channels)
        .context("failed to create opus-decoder decoder")?;
    let mut pcm = vec![0.0; decoder.max_frame_size_per_channel() * stream.channels];
    let mut mono = Vec::new();
    let mut skip = stream.pre_skip;
    for packet in stream.audio_packets {
        let samples_per_channel = decoder
            .decode_float(&packet, &mut pcm, false)
            .context("opus-decoder failed to decode packet")?;
        append_interleaved_mono_samples(
            &pcm[..samples_per_channel * stream.channels],
            stream.channels,
            &mut skip,
            &mut mono,
        );
    }
    if let Some(final_granule) = stream.final_granule {
        if let Ok(target_len) = usize::try_from(final_granule) {
            mono.truncate(target_len);
        }
    }
    if mono.is_empty() {
        bail!("opus-decoder decoded audio stream is empty");
    }
    Ok((mono, 48_000))
}

fn source_bytes(source: &AudioSource) -> Result<Vec<u8>> {
    match source {
        AudioSource::Path(path, _) => {
            fs::read(path).with_context(|| format!("failed to read audio file {}", path.display()))
        }
        AudioSource::Bytes(bytes, _) => Ok(bytes.clone()),
    }
}

fn source_looks_like_ogg_opus(source: &AudioSource) -> Result<bool> {
    if !source_may_be_ogg_opus(source) {
        return Ok(false);
    }
    let Some(probe) = source_probe_bytes(source)? else {
        return Ok(false);
    };
    Ok(contains_subsequence(&probe, b"OggS") && contains_subsequence(&probe, b"OpusHead"))
}

fn source_may_be_ogg_opus(source: &AudioSource) -> bool {
    source_extension(source).is_some_and(|extension| {
        ["opus", "ogg", "oga", "bin"]
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    })
}

fn source_extension(source: &AudioSource) -> Option<&str> {
    match source {
        AudioSource::Path(path, path_hint) => path_hint
            .as_deref()
            .and_then(path_extension)
            .or_else(|| path.extension().and_then(|value| value.to_str())),
        AudioSource::Bytes(_, path_hint) => path_hint.as_deref().and_then(path_extension),
    }
}

fn source_probe_bytes(source: &AudioSource) -> Result<Option<Vec<u8>>> {
    match source {
        AudioSource::Path(path, _) => {
            let mut file = match File::open(path) {
                Ok(file) => file,
                Err(_) => return Ok(None),
            };
            let mut bytes = vec![0; OGG_OPUS_PROBE_BYTES];
            let len = file
                .read(&mut bytes)
                .with_context(|| format!("failed to probe audio file {}", path.display()))?;
            bytes.truncate(len);
            Ok(Some(bytes))
        }
        AudioSource::Bytes(bytes, _) => Ok(Some(
            bytes.iter().take(OGG_OPUS_PROBE_BYTES).copied().collect(),
        )),
    }
}

fn path_extension(path: &str) -> Option<&str> {
    Path::new(path).extension().and_then(|value| value.to_str())
}

fn contains_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

struct OggOpusStream {
    channels: usize,
    pre_skip: usize,
    final_granule: Option<u64>,
    audio_packets: Vec<Vec<u8>>,
}

fn parse_ogg_opus_stream(bytes: &[u8]) -> Result<OggOpusStream> {
    let mut offset = 0;
    let mut packets = Vec::new();
    let mut current_packet = Vec::new();
    let mut final_granule = None;

    while offset < bytes.len() {
        let header = bytes
            .get(offset..offset + 27)
            .ok_or_else(|| anyhow!("truncated Ogg page header"))?;
        if &header[..4] != b"OggS" {
            bail!("invalid Ogg page capture pattern");
        }
        if header[4] != 0 {
            bail!("unsupported Ogg bitstream version {}", header[4]);
        }

        let continued_packet = header[5] & 0x01 != 0;
        if !continued_packet && !current_packet.is_empty() {
            current_packet.clear();
        }

        let granule = i64::from_le_bytes(header[6..14].try_into().expect("slice length checked"));
        if granule >= 0 {
            final_granule = Some(granule as u64);
        }

        let segment_count = usize::from(header[26]);
        let lacing_start = offset + 27;
        let lacing = bytes
            .get(lacing_start..lacing_start + segment_count)
            .ok_or_else(|| anyhow!("truncated Ogg lacing table"))?;
        let payload_len = lacing
            .iter()
            .try_fold(0usize, |sum, value| sum.checked_add(usize::from(*value)))
            .ok_or_else(|| anyhow!("Ogg page payload length overflow"))?;
        let payload_start = lacing_start + segment_count;
        let payload_end = payload_start
            .checked_add(payload_len)
            .ok_or_else(|| anyhow!("Ogg page payload offset overflow"))?;
        let payload = bytes
            .get(payload_start..payload_end)
            .ok_or_else(|| anyhow!("truncated Ogg page payload"))?;

        let mut payload_offset = 0;
        for segment_len in lacing.iter().copied().map(usize::from) {
            let segment_end = payload_offset + segment_len;
            current_packet.extend_from_slice(&payload[payload_offset..segment_end]);
            payload_offset = segment_end;
            if segment_len < 255 {
                packets.push(std::mem::take(&mut current_packet));
            }
        }

        offset = payload_end;
    }

    let opus_head_index = packets
        .iter()
        .position(|packet| packet.starts_with(b"OpusHead"))
        .ok_or_else(|| anyhow!("Ogg stream has no OpusHead packet"))?;
    let (channels, pre_skip) = parse_opus_head(&packets[opus_head_index])?;
    let audio_packets = packets
        .into_iter()
        .skip(opus_head_index + 1)
        .filter(|packet| {
            !packet.is_empty()
                && !packet.starts_with(b"OpusHead")
                && !packet.starts_with(b"OpusTags")
        })
        .collect::<Vec<_>>();
    if audio_packets.is_empty() {
        bail!("Ogg/Opus stream has no audio packets");
    }

    Ok(OggOpusStream {
        channels,
        pre_skip,
        final_granule,
        audio_packets,
    })
}

fn parse_opus_head(packet: &[u8]) -> Result<(usize, usize)> {
    if packet.len() < 19 || !packet.starts_with(b"OpusHead") {
        bail!("invalid OpusHead packet");
    }
    let channels = usize::from(packet[9]);
    if !(1..=2).contains(&channels) {
        bail!("opus-decoder fallback only supports mono or stereo Opus, got {channels} channels");
    }
    let pre_skip = usize::from(u16::from_le_bytes([packet[10], packet[11]]));
    Ok((channels, pre_skip))
}

fn append_interleaved_mono_samples(
    samples: &[f32],
    channels: usize,
    skip: &mut usize,
    output: &mut Vec<f32>,
) {
    for frame in samples.chunks(channels) {
        if *skip > 0 {
            *skip -= 1;
            continue;
        }
        let sum: f32 = frame.iter().copied().sum();
        output.push(sum / channels as f32);
    }
}

fn decode_audio_ffmpeg(source: AudioSource, sample_rate: u32) -> Result<(Vec<f32>, u32)> {
    debug!(
        "decoding audio with FFmpeg libraries source={} output_sample_rate={}",
        source.log_label(),
        sample_rate
    );
    match source {
        AudioSource::Path(path, _) => decode_audio_ffmpeg_path(&path, sample_rate),
        AudioSource::Bytes(bytes, path_hint) => {
            decode_audio_ffmpeg_bytes(bytes, path_hint, sample_rate)
        }
    }
}

fn decode_audio_ffmpeg_bytes(
    bytes: Vec<u8>,
    path_hint: Option<String>,
    sample_rate: u32,
) -> Result<(Vec<f32>, u32)> {
    let mut builder = tempfile::Builder::new();
    builder.prefix("asr_features_audio_");
    let suffix;
    if let Some(extension) = path_hint
        .as_deref()
        .and_then(|path| Path::new(path).extension())
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
    {
        suffix = format!(".{extension}");
        builder.suffix(&suffix);
    }
    let mut file = builder
        .tempfile()
        .context("failed to create temporary audio file for FFmpeg decode")?;
    file.write_all(&bytes)
        .context("failed to write audio bytes to temporary file for FFmpeg decode")?;
    file.flush()
        .context("failed to flush temporary audio file for FFmpeg decode")?;
    decode_audio_ffmpeg_path(file.path(), sample_rate)
}

fn init_ffmpeg() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    match INIT.get_or_init(|| ffmpeg::init().map_err(|error| error.to_string())) {
        Ok(()) => Ok(()),
        Err(error) => bail!("failed to initialize FFmpeg libraries: {error}"),
    }
}

fn decode_audio_ffmpeg_path(path: &Path, sample_rate: u32) -> Result<(Vec<f32>, u32)> {
    init_ffmpeg()?;
    let mut input = ffmpeg::format::input(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let stream = input
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or_else(|| anyhow!("FFmpeg found no audio stream in {}", path.display()))?;
    let stream_index = stream.index();
    let context = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
        .context("failed to create FFmpeg decoder context")?;
    let mut decoder = context
        .decoder()
        .audio()
        .context("failed to open FFmpeg audio decoder")?;
    let mut samples = Vec::new();
    let mut resampler = None;

    for (packet_stream, packet) in input.packets() {
        if packet_stream.index() == stream_index {
            decoder
                .send_packet(&packet)
                .context("failed to send packet to FFmpeg decoder")?;
            receive_ffmpeg_frames(&mut decoder, &mut resampler, sample_rate, &mut samples)
                .context("failed to receive FFmpeg decoded frame")?;
        }
    }
    decoder
        .send_eof()
        .context("failed to flush FFmpeg decoder")?;
    receive_ffmpeg_frames(&mut decoder, &mut resampler, sample_rate, &mut samples)
        .context("failed to receive final FFmpeg decoded frame")?;
    flush_ffmpeg_resampler(&mut resampler, &mut samples)
        .context("failed to flush FFmpeg resampler")?;

    if samples.is_empty() {
        bail!("FFmpeg decoded audio stream is empty");
    }
    debug!(
        "decoded audio with FFmpeg libraries samples={} sample_rate={}",
        samples.len(),
        sample_rate
    );
    Ok((samples, sample_rate))
}

struct ActiveFfmpegResampler {
    context: ffmpeg::software::resampling::Context,
    format: ffmpeg::format::Sample,
    channel_layout: ffmpeg::ChannelLayout,
    rate: u32,
}

fn receive_ffmpeg_frames(
    decoder: &mut ffmpeg::decoder::Audio,
    resampler: &mut Option<ActiveFfmpegResampler>,
    output_sample_rate: u32,
    samples: &mut Vec<f32>,
) -> Result<()> {
    let mut decoded = ffmpeg::frame::Audio::empty();
    while decoder.receive_frame(&mut decoded).is_ok() {
        if decoded.channel_layout().is_empty() {
            decoded.set_channel_layout(ffmpeg::ChannelLayout::default(i32::from(
                decoded.channels(),
            )));
        }
        ensure_ffmpeg_resampler(resampler, &decoded, output_sample_rate, samples)?;
        let mut resampled = ffmpeg::frame::Audio::empty();
        resampler
            .as_mut()
            .expect("resampler is initialized")
            .context
            .run(&decoded, &mut resampled)
            .context("FFmpeg resampling failed")?;
        append_ffmpeg_samples(&resampled, samples)?;
    }
    Ok(())
}

fn ensure_ffmpeg_resampler(
    resampler: &mut Option<ActiveFfmpegResampler>,
    frame: &ffmpeg::frame::Audio,
    output_sample_rate: u32,
    samples: &mut Vec<f32>,
) -> Result<()> {
    let input_layout = frame_channel_layout(frame);
    let input_format = frame.format();
    let input_rate = frame.rate();
    let needs_rebuild = resampler.as_ref().is_none_or(|active| {
        active.format != input_format
            || active.channel_layout != input_layout
            || active.rate != input_rate
    });
    if !needs_rebuild {
        return Ok(());
    }
    flush_ffmpeg_resampler(resampler, samples)?;
    let context = ffmpeg::software::resampling::Context::get(
        input_format,
        input_layout,
        input_rate,
        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
        ffmpeg::ChannelLayout::MONO,
        output_sample_rate,
    )
    .context("failed to create FFmpeg audio resampler")?;
    *resampler = Some(ActiveFfmpegResampler {
        context,
        format: input_format,
        channel_layout: input_layout,
        rate: input_rate,
    });
    Ok(())
}

fn frame_channel_layout(frame: &ffmpeg::frame::Audio) -> ffmpeg::ChannelLayout {
    if frame.channel_layout().is_empty() {
        ffmpeg::ChannelLayout::default(i32::from(frame.channels()))
    } else {
        frame.channel_layout()
    }
}

fn flush_ffmpeg_resampler(
    resampler: &mut Option<ActiveFfmpegResampler>,
    samples: &mut Vec<f32>,
) -> Result<()> {
    let Some(active) = resampler.as_mut() else {
        return Ok(());
    };
    loop {
        let output = *active.context.output();
        let mut resampled = ffmpeg::frame::Audio::new(output.format, 8192, output.channel_layout);
        if active
            .context
            .flush(&mut resampled)
            .context("FFmpeg resampler flush failed")?
            .is_none()
        {
            break;
        }
        append_ffmpeg_samples(&resampled, samples)?;
    }
    Ok(())
}

fn append_ffmpeg_samples(frame: &ffmpeg::frame::Audio, output: &mut Vec<f32>) -> Result<()> {
    if frame.samples() == 0 {
        return Ok(());
    }
    if !frame.is_packed()
        || frame.format() != ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed)
    {
        bail!(
            "FFmpeg resampler produced unsupported sample format {:?}",
            frame.format()
        );
    }
    output.extend_from_slice(frame.plane::<f32>(0));
    Ok(())
}

fn append_mono_samples(decoded: AudioBufferRef<'_>, output: &mut Vec<f32>, sample_rate: &mut u32) {
    let spec = *decoded.spec();
    *sample_rate = spec.rate;
    let channels = spec.channels.count().max(1);
    let mut sample_buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
    sample_buffer.copy_interleaved_ref(decoded);
    for frame in sample_buffer.samples().chunks(channels) {
        let sum: f32 = frame.iter().copied().sum();
        output.push(sum / channels as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    #[test]
    fn ffmpeg_library_decoder_reads_wav_file() {
        let mut file = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        write_test_wav(&mut file, 16_000, 1_600).unwrap();

        let (samples, sample_rate) = decode_audio_ffmpeg_path(file.path(), 16_000).unwrap();

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 1_600);
        assert!(samples.iter().any(|sample| sample.abs() > 1e-4));
    }

    #[test]
    fn opus_decoder_reads_generated_ogg_opus() {
        let Some((_output_dir, opus_path)) = generate_test_ogg_opus() else {
            return;
        };

        let (samples, sample_rate) =
            decode_audio_opus_decoder(AudioSource::Path(opus_path, Some("audio.opus".to_string())))
                .unwrap();

        assert_eq!(sample_rate, 48_000);
        assert!(!samples.is_empty());
        assert!(samples.iter().any(|sample| sample.abs() > 1e-4));
    }

    #[test]
    fn decode_audio_uses_opus_decoder_for_tagless_ogg_opus() {
        let Some((_output_dir, opus_path)) = generate_test_ogg_opus() else {
            return;
        };
        let bytes = std::fs::read(opus_path).unwrap();
        let tagless = strip_ogg_pages_containing(&bytes, b"OpusTags");
        let source = AudioSource::Bytes(tagless, Some("audio.opus".to_string()));

        assert!(source_looks_like_ogg_opus(&source).unwrap());
        let (samples, sample_rate) = decode_audio(source, 16_000, false).unwrap();

        assert_eq!(sample_rate, 48_000);
        assert!(!samples.is_empty());
        assert!(samples.iter().any(|sample| sample.abs() > 1e-4));
    }

    #[test]
    fn ogg_opus_byte_source_prefers_opus_decoder() {
        let source = AudioSource::Bytes(
            b"OggS..........OpusHead..........audio-packet".to_vec(),
            Some("audio.opus".to_string()),
        );

        assert!(source_looks_like_ogg_opus(&source).unwrap());
    }

    #[test]
    fn tagged_ogg_opus_byte_source_prefers_opus_decoder() {
        let source = AudioSource::Bytes(
            b"OggS..........OpusHead..........OpusTags..........audio-packet".to_vec(),
            Some("audio.opus".to_string()),
        );

        assert!(source_looks_like_ogg_opus(&source).unwrap());
    }

    fn generate_test_ogg_opus() -> Option<(tempfile::TempDir, PathBuf)> {
        if !command_succeeds(Command::new("ffmpeg").arg("-version")) {
            return None;
        }
        let mut wav = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
        write_test_wav(&mut wav, 48_000, 4_800).unwrap();
        let output_dir = tempfile::tempdir().unwrap();
        let opus_path = output_dir.path().join("audio.opus");
        let status = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(wav.path())
            .args(["-c:a", "libopus", "-ar", "48000", "-ac", "1"])
            .arg(&opus_path)
            .status()
            .unwrap();
        status.success().then_some((output_dir, opus_path))
    }

    fn strip_ogg_pages_containing(bytes: &[u8], needle: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        let mut offset = 0;
        while offset < bytes.len() {
            let segment_count = usize::from(bytes[offset + 26]);
            let lacing_start = offset + 27;
            let payload_start = lacing_start + segment_count;
            let payload_len: usize = bytes[lacing_start..payload_start]
                .iter()
                .map(|value| usize::from(*value))
                .sum();
            let page_end = payload_start + payload_len;
            let page = &bytes[offset..page_end];
            if !contains_subsequence(page, needle) {
                output.extend_from_slice(page);
            }
            offset = page_end;
        }
        output
    }

    fn command_succeeds(command: &mut Command) -> bool {
        command
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    fn write_test_wav(
        file: &mut tempfile::NamedTempFile,
        sample_rate: u32,
        frames: u32,
    ) -> Result<()> {
        let channels = 1u16;
        let bits_per_sample = 16u16;
        let bytes_per_sample = bits_per_sample / 8;
        let block_align = channels * bytes_per_sample;
        let byte_rate = sample_rate * u32::from(block_align);
        let data_len = frames * u32::from(block_align);
        file.write_all(b"RIFF")?;
        file.write_all(&(36 + data_len).to_le_bytes())?;
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;
        file.write_all(&channels.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&byte_rate.to_le_bytes())?;
        file.write_all(&block_align.to_le_bytes())?;
        file.write_all(&bits_per_sample.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&data_len.to_le_bytes())?;
        for index in 0..frames {
            let sample = (12_000.0 * (index as f32 * 0.03).sin()) as i16;
            file.write_all(&sample.to_le_bytes())?;
        }
        file.flush()?;
        Ok(())
    }
}
