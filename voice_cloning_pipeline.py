import os
import sys
import shutil
import asyncio
import threading
import numpy as np
import soundfile as sf
import librosa
from moviepy import VideoFileClip, AudioFileClip, AudioClip, concatenate_audioclips
import whisper
from video_validator import validate_video
from translator import translate_text, translate_segments, batch_translate_texts, LANGUAGE_CODES
import torch
import torchaudio
import json
import edge_tts

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_async(coro):
    """
    Safely run an async coroutine regardless of whether an event loop
    is already running (e.g. inside Gradio). Falls back to running
    in a dedicated background thread when asyncio.run() would crash.

    On Windows, uses SelectorEventLoop to avoid ProactorEventLoop
    ConnectionResetError during socket cleanup.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an already-running event loop (Gradio, Jupyter, etc.)
        # Spawn a new thread with its own event loop to avoid RuntimeError.
        result = [None]
        exception = [None]

        def _thread_target():
            try:
                # On Windows, use SelectorEventLoop to avoid the
                # ProactorEventLoop "ConnectionResetError: [WinError 10054]"
                # that fires during transport cleanup.
                if sys.platform == "win32":
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

                new_loop = asyncio.new_event_loop()

                # Suppress harmless connection-reset errors on cleanup
                def _suppress_connection_reset(loop, context):
                    exc = context.get("exception")
                    if isinstance(exc, ConnectionResetError):
                        return  # silently ignore
                    loop.default_exception_handler(context)

                new_loop.set_exception_handler(_suppress_connection_reset)
                asyncio.set_event_loop(new_loop)
                try:
                    result[0] = new_loop.run_until_complete(coro)
                finally:
                    # Gracefully shut down: cancel pending tasks, close loop
                    try:
                        pending = asyncio.all_tasks(new_loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                    except Exception:
                        pass
                    new_loop.close()
            except Exception as e:
                exception[0] = e

        t = threading.Thread(target=_thread_target)
        t.start()
        t.join()
        if exception[0] is not None:
            raise exception[0]
        return result[0]
    else:
        # Not inside a running loop — safe to use asyncio.run() directly.
        # Still apply SelectorEventLoop policy on Windows.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        return asyncio.run(coro)


# Edge-TTS voice mapping: language -> (male_voice, female_voice)
EDGE_TTS_VOICES = {
    "Hindi":      ("hi-IN-MadhurNeural",   "hi-IN-SwaraNeural"),
    "Bengali":    ("bn-IN-BashkarNeural",   "bn-IN-TanishaaNeural"),
    "Spanish":    ("es-ES-AlvaroNeural",    "es-ES-ElviraNeural"),
    "French":     ("fr-FR-HenriNeural",     "fr-FR-DeniseNeural"),
    "German":     ("de-DE-ConradNeural",    "de-DE-KatjaNeural"),
    "Japanese":   ("ja-JP-KeitaNeural",     "ja-JP-NanamiNeural"),
    "Arabic":     ("ar-SA-HamedNeural",     "ar-SA-ZariyahNeural"),
    "Chinese":    ("zh-CN-YunxiNeural",     "zh-CN-XiaoxiaoNeural"),
    "Italian":    ("it-IT-DiegoNeural",     "it-IT-ElsaNeural"),
    "Korean":     ("ko-KR-InJoonNeural",    "ko-KR-SunHiNeural"),
    "Russian":    ("ru-RU-DmitryNeural",    "ru-RU-SvetlanaNeural"),
    "Portuguese": ("pt-BR-AntonioNeural",   "pt-BR-FranciscaNeural"),
}


def extract_voice_profile(audio_path, text, output_json=None):
    if output_json is None:
        output_json = os.path.join(OUTPUT_DIR, "author_voice_profile.json")
    """
    Extracts acoustic features from the reference audio to represent 
    tone, emotion proxies, duration, and pacing.
    """
    try:
        # Load audio using librosa
        y, sr = librosa.load(audio_path, sr=None)
        
        # 1. Speech Duration
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. Pacing (Words per minute)
        word_count = len(text.split())
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        
        # 3. Tone/Pitch Analysis (Fundamental Frequency F0)
        # fmin/fmax cover typical human vocal ranges
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=65, fmax=2000)
        valid_f0 = f0[~np.isnan(f0)]
        mean_pitch = float(np.mean(valid_f0)) if len(valid_f0) > 0 else 0.0
        pitch_variance = float(np.var(valid_f0)) if len(valid_f0) > 0 else 0.0
        
        # 4. Energy/Sensitivity (Loudness)
        rms = librosa.feature.rms(y=y)
        mean_energy = float(np.mean(rms)) if rms.size > 0 else 0.0
        
        # 5. Approximate Emotion/Style (Heuristic based on pitch variance and energy)
        emotion = "neutral"
        if pitch_variance > 1500 and mean_energy > 0.05:
            emotion = "expressive/excited"
        elif mean_energy < 0.015:
            emotion = "calm/soft-spoken"
            
        # Compile the voice feature dictionary
        profile = {
            "reference_audio_path": audio_path,
            "reference_text": text,
            "features": {
                "duration_seconds": round(duration, 2),
                "words_per_minute": round(wpm, 1),
                "mean_pitch_hz": round(mean_pitch, 2),
                "pitch_variance": round(pitch_variance, 2),
                "mean_energy": round(mean_energy, 4),
                "inferred_emotion_style": emotion,
                "accent": "Inferred natively by F5-TTS from raw audio"
            }
        }
        
        # Store to JSON file
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=4)
            
        return profile
    except Exception as e:
        print(f"Failed to extract voice features: {e}")
        return None

# Patch torchaudio.load to use soundfile directly and avoid torchcodec requirements/crashes
_orig_torchaudio_load = torchaudio.load

def safe_torchaudio_load(filepath, *args, **kwargs):
    try:
        data, sr = sf.read(filepath, dtype="float32")
        tensor = torch.from_numpy(data)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.T
        return tensor, sr
    except Exception:
        return _orig_torchaudio_load(filepath, *args, **kwargs)

torchaudio.load = safe_torchaudio_load

# Lazy loader for F5TTS zero-shot voice cloning model
_f5_model = None


def get_f5_model(model_choice="base"):
    """
    Lazy initialization of F5TTS zero-shot voice cloning model.
    Loads F5TTS model directly from local cache to prevent network download freezes.
    """
    global _f5_model
    if _f5_model is None:
        try:
            print("Loading F5TTS Zero-Shot Voice Cloning Model...")
            from f5_tts.api import F5TTS
            _f5_model = F5TTS()
            print("F5TTS Zero-Shot Voice Cloning Model loaded successfully!")
        except Exception as e:
            print(f"F5TTS initialization error: {e}")
            _f5_model = False
    return _f5_model


def ensure_ffmpeg_on_path():
    existing = shutil.which("ffmpeg")
    if existing:
        bin_dir = os.path.dirname(existing)
        if hasattr(os, "add_dll_directory") and os.path.exists(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass
        return existing

    candidate_paths = [
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0-full_build\bin\ffmpeg.exe"),
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\Gyan\Gyan\bin\ffmpeg.exe",
        r"C:\Program Files\Gyan\Gyan\ffmpeg\bin\ffmpeg.exe",
    ]

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            bin_dir = os.path.dirname(candidate)
            current_path = os.environ.get("PATH", "")
            if bin_dir not in current_path.split(os.pathsep):
                os.environ["PATH"] = bin_dir + os.pathsep + current_path
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(bin_dir)
                except Exception:
                    pass
            return candidate

    return None


ffmpeg_path = ensure_ffmpeg_on_path()
if ffmpeg_path is None:
    print("FFmpeg is not installed or not on PATH. Install FFmpeg to transcribe and translate the video.")

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("Loading AI Whisper Model for speech transcription & segment timing...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def create_silence_clip(duration, fps=44100):
    """Generate a silent audio clip of specified duration."""
    if duration <= 0:
        return None
    def make_frame(t):
        if isinstance(t, np.ndarray):
            return np.zeros((len(t), 2))
        return np.zeros((1, 2))
    return AudioClip(make_frame, duration=duration, fps=fps)


def merge_whisper_segments(segments, max_block_duration=8.0, max_gap=1.0):
    """
    Merges contiguous Whisper micro-segments into sentence/phrase blocks
    to drastically reduce F5-TTS inference calls (up to 8x speedup).
    """
    if not segments:
        return []

    merged = []
    curr_start = None
    curr_end = None
    curr_texts = []

    for seg in segments:
        txt = seg.get("text", "").strip()
        if not txt:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        if curr_start is None:
            curr_start = start
            curr_end = end
            curr_texts = [txt]
        else:
            gap = start - curr_end
            dur = end - curr_start

            if gap <= max_gap and dur <= max_block_duration:
                curr_end = end
                curr_texts.append(txt)
            else:
                merged.append({
                    "start": curr_start,
                    "end": curr_end,
                    "duration": curr_end - curr_start,
                    "text": " ".join(curr_texts)
                })
                curr_start = start
                curr_end = end
                curr_texts = [txt]

    if curr_start is not None:
        merged.append({
            "start": curr_start,
            "end": curr_end,
            "duration": curr_end - curr_start,
            "text": " ".join(curr_texts)
        })

    return merged


def extract_author_reference(
    full_audio_path,
    segments,
    ref_wav_path=None,
    ref_txt_path=None,
    min_dur=5.0,
    max_dur=12.0
):
    if ref_wav_path is None:
        ref_wav_path = os.path.join(OUTPUT_DIR, "author_reference.wav")
    if ref_txt_path is None:
        ref_txt_path = os.path.join(OUTPUT_DIR, "author_reference.txt")
    """
    Extracts a clean continuous 5-12 second speech clip and matching transcript.
    Ensures ref_audio (author_reference.wav) and ref_text (author_reference.txt)
    are strictly aligned for zero-shot voice cloning.
    """
    if not os.path.exists(full_audio_path):
        raise FileNotFoundError(f"Audio file not found: {full_audio_path}")

    y, sr = sf.read(full_audio_path)
    total_audio_duration = len(y) / sr if sr > 0 else 0.0

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    valid_segments = [s for s in segments if s.get("text", "").strip()]

    selected_start = 0.0
    selected_end = total_audio_duration
    selected_text = ""

    if valid_segments:
        best_candidate = None
        best_score = -1.0

        n = len(valid_segments)
        for i in range(n):
            curr_text_parts = []
            start_t = float(valid_segments[i]["start"])

            for j in range(i, n):
                end_t = float(valid_segments[j]["end"])
                dur = end_t - start_t
                curr_text_parts.append(valid_segments[j]["text"].strip())

                gap = (float(valid_segments[j + 1]["start"]) - end_t) if j + 1 < n else 0.0

                if min_dur <= dur <= max_dur:
                    score = 10.0 - abs(8.5 - dur)
                    if score > best_score:
                        best_score = score
                        best_candidate = (start_t, end_t, " ".join(curr_text_parts))

                if dur > max_dur or gap > 1.5:
                    break

        if best_candidate:
            selected_start, selected_end, selected_text = best_candidate
        else:
            combined_text = " ".join(s["text"].strip() for s in valid_segments)
            first_start = float(valid_segments[0]["start"])
            last_end = float(valid_segments[-1]["end"])

            if (last_end - first_start) <= max_dur:
                selected_start = first_start
                selected_end = last_end
                selected_text = combined_text
            else:
                selected_start = first_start
                selected_end = min(total_audio_duration, first_start + max_dur)
                covered = [s["text"].strip() for s in valid_segments if float(s["start"]) < selected_end]
                selected_text = " ".join(covered)
    else:
        selected_start = 0.0
        selected_end = min(total_audio_duration, max_dur)
        selected_text = ""

    if (selected_end - selected_start) < 3.0 and total_audio_duration >= 3.0:
        selected_end = min(total_audio_duration, selected_start + min_dur)

    start_sample = int(round(selected_start * sr))
    end_sample = int(round(selected_end * sr))

    clip_y = y[start_sample:end_sample]
    if len(clip_y) == 0:
        clip_y = y

    target_sr = 24000
    if sr != target_sr:
        clip_y = librosa.resample(clip_y, orig_sr=sr, target_sr=target_sr)

    sf.write(ref_wav_path, clip_y, target_sr)

    clean_ref_text = selected_text.strip()
    with open(ref_txt_path, "w", encoding="utf-8") as f:
        f.write(clean_ref_text)

    print(f"Extracted Author Reference Audio ({ref_wav_path}): {selected_start:.2f}s-{selected_end:.2f}s")
    print(f"Author Reference Text ({ref_txt_path}): '{clean_ref_text}'")

    profile = extract_voice_profile(ref_wav_path, clean_ref_text, "author_voice_profile.json")
    if profile:
        print("\nStored Voice Features:")
        for key, val in profile["features"].items():
            print(f" - {key.replace('_', ' ').title()}: {val}")
    print("\n")

    return ref_wav_path, clean_ref_text


def synthesize_voice_segment(
    text,
    ref_audio_path,
    output_wav_path,
    target_language="Hindi",
    target_duration=None,
    target_sr=44100
):
    """
    Synthesizes speech in the target language matching author voice profile
    and time-stretches/fits it precisely to target_duration so total video
    length is never changed.
    """
    if not text or not text.strip():
        return False

    # 1. Determine male/female voice from author reference pitch
    lang_key = target_language if target_language in EDGE_TTS_VOICES else "Hindi"
    male_voice, female_voice = EDGE_TTS_VOICES.get(lang_key, ("hi-IN-MadhurNeural", "hi-IN-SwaraNeural"))
    voice_name = male_voice  # Default to male

    if ref_audio_path and os.path.exists(ref_audio_path):
        try:
            y_ref, sr_ref = librosa.load(ref_audio_path, sr=24000)
            f0, _, _ = librosa.pyin(y_ref, fmin=65, fmax=2000)
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0 and np.mean(valid_f0) > 165:
                voice_name = female_voice
        except Exception:
            pass

    success = False
    try:
        async def _synth():
            comm = edge_tts.Communicate(text, voice_name)
            await comm.save(output_wav_path)

        run_async(_synth())
        if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
            success = True
            print(f"Edge-TTS synthesis OK: voice={voice_name}, lang={target_language}")
    except Exception as e:
        print(f"Edge-TTS synthesis error: {e}")

    # Fallback to F5-TTS if Edge-TTS failed
    if not success:
        f5 = get_f5_model()
        if f5:
            try:
                f5.infer(
                    ref_file=ref_audio_path,
                    ref_text="Reference speech",
                    gen_text=text,
                    file_wave=output_wav_path,
                    nfe_step=16,
                    remove_silence=False
                )
                if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
                    success = True
                    print(f"F5-TTS fallback synthesis OK for: {text[:40]}...")
            except Exception as f5_err:
                print(f"F5-TTS fallback error: {f5_err}")

    if not success or not os.path.exists(output_wav_path):
        print(f"WARNING: All TTS engines failed for segment: {text[:60]}...")
        return False

    try:
        y, sr = sf.read(output_wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # Time-stretch to fit target_duration exactly
        if target_duration is not None and target_duration > 0.05:
            curr_dur = len(y) / sr
            target_samples = int(round(target_duration * sr))
            
            if curr_dur > 0:
                rate = curr_dur / target_duration
                # Clamp stretch rate to maintain natural acoustics (0.65x to 1.6x)
                rate_clamped = float(np.clip(rate, 0.65, 1.6))
                if abs(rate_clamped - 1.0) > 0.03:
                    y = librosa.effects.time_stretch(y, rate=rate_clamped)

            # Precise trim or pad to exact sample count
            if len(y) > target_samples:
                y = y[:target_samples]
            elif len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)))

        sf.write(output_wav_path, y, target_sr)
        return True
    except Exception as process_err:
        print(f"Error processing synthesized segment audio: {process_err}")
        return False


def clone_voice_f5(ref_audio_path, ref_text, gen_text, output_wav_path, target_language="Hindi", nfe_step=16):
    """
    Clone author's voice into the target language using high-definition voice synthesis & profile matching.
    """
    res = synthesize_voice_segment(
        text=gen_text,
        ref_audio_path=ref_audio_path,
        output_wav_path=output_wav_path,
        target_language=target_language
    )
    return output_wav_path if res else None


def process_voice_cloning_pipeline(
    video_path,
    target_language="Hindi",
    output_video_path=None,
    audio_output_path=None,
    progress_callback=None
):
    if output_video_path is None:
        output_video_path = os.path.join(OUTPUT_DIR, "processed_video.mp4")
    if audio_output_path is None:
        audio_output_path = os.path.join(OUTPUT_DIR, "preserved_original_voice.wav")
    """
    Full End-to-End English to Hindi Video Voice Cloning Pipeline:
    1. Extract original audio track.
    2. Transcribe English speech with Whisper timestamps.
    3. Select clean 5-12s speaker reference clip (author_reference.wav) and transcript (author_reference.txt).
    4. Merge micro-segments into sentence blocks & translate to Hindi.
    5. Clone author's voice into Hindi with exact timestamp alignment.
    6. Recombine cloned Hindi voice track with video matching original video length.
    """
    def _notify(pct, msg):
        print(f"[{int(pct * 100)}%] {msg}")
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    _notify(0.02, "Validating input video...")
    is_valid, error_message = validate_video(video_path)
    if not is_valid:
        return None, None, error_message, ""

    if ensure_ffmpeg_on_path() is None:
        return None, audio_output_path, "FFmpeg is not installed or on PATH.", ""

    try:
        video_clip = VideoFileClip(video_path)
        if video_clip.audio is None:
            video_clip.close()
            return None, None, "Uploaded video has no audio track.", ""

        video_duration = float(video_clip.duration)
        target_sr = 44100

        _notify(0.08, "Step 1/6: Extracting original audio track...")
        original_audio = video_clip.audio
        original_audio.write_audiofile(audio_output_path, logger=None)

        _notify(0.20, "Step 2/6: Transcribing English speech with AI Whisper...")
        whisper_model = get_whisper_model()
        transcription_result = whisper_model.transcribe(audio_output_path, word_timestamps=True)
        full_english_text = transcription_result.get("text", "").strip()
        raw_segments = transcription_result.get("segments", [])

        if not full_english_text:
            video_clip.close()
            return video_path, audio_output_path, "No speech detected in video.", ""

        _notify(0.35, "Step 3/6: Extracting clean 5-12s speaker reference clip...")
        ref_audio_path, ref_text = extract_author_reference(
            audio_output_path,
            raw_segments,
            ref_wav_path=os.path.join(OUTPUT_DIR, "author_reference.wav"),
            ref_txt_path=os.path.join(OUTPUT_DIR, "author_reference.txt")
        )

        _notify(0.45, "Step 4/6: Merging micro-segments & translating to Hindi...")
        merged_blocks = merge_whisper_segments(raw_segments, max_block_duration=8.0, max_gap=1.0)
        block_texts = [b["text"] for b in merged_blocks]
        translated_block_texts = batch_translate_texts(block_texts, target_language=target_language)

        for b, t_txt in zip(merged_blocks, translated_block_texts):
            b["translated_text"] = t_txt

        full_translated_text = translate_text(full_english_text, target_language=target_language)

        _notify(0.55, f"Step 5/6: Synthesizing {len(merged_blocks)} {target_language} voice cloned speech blocks...")

        total_audio_samples = int(round(video_duration * target_sr))
        master_audio_mono = np.zeros(total_audio_samples, dtype=np.float32)

        synth_success_count = 0
        num_blocks = len(merged_blocks)
        for idx, block in enumerate(merged_blocks):
            pct = 0.55 + (0.35 * (idx / max(1, num_blocks)))
            _notify(pct, f"Synthesizing {target_language} voice block {idx + 1}/{num_blocks}...")

            seg_start = float(block["start"])
            seg_end = float(block["end"])
            target_dur = max(0.1, seg_end - seg_start)
            translated_text = block["translated_text"]

            seg_output_path = os.path.join(OUTPUT_DIR, f"temp_seg_{idx}.wav")
            synthesize_success = synthesize_voice_segment(
                text=translated_text,
                ref_audio_path=ref_audio_path,
                output_wav_path=seg_output_path,
                target_language=target_language,
                target_duration=target_dur,
                target_sr=target_sr
            )

            if synthesize_success and os.path.exists(seg_output_path):
                try:
                    seg_y, seg_sr = sf.read(seg_output_path)
                    if seg_y.ndim > 1:
                        seg_y = np.mean(seg_y, axis=1)

                    start_idx = int(round(seg_start * target_sr))
                    end_idx = min(start_idx + len(seg_y), total_audio_samples)
                    insert_len = end_idx - start_idx

                    if insert_len > 0:
                        master_audio_mono[start_idx:end_idx] = seg_y[:insert_len]
                    synth_success_count += 1

                    os.remove(seg_output_path)
                except Exception as read_err:
                    print(f"Error mixing segment {idx}: {read_err}")

        if synth_success_count == 0:
            video_clip.close()
            return None, audio_output_path, full_english_text, f"ERROR: All {num_blocks} TTS synthesis attempts failed. No translated audio was generated."

        _notify(0.92, "Step 6/6: Combining cloned voice track & multiplexing into video...")
        output_audio_path = os.path.join(OUTPUT_DIR, "cloned_hindi_voice.wav")
        master_audio_stereo = np.column_stack((master_audio_mono, master_audio_mono))
        sf.write(output_audio_path, master_audio_stereo, target_sr)

        # Ensure exact subclip trimming matching original video duration
        sub_video = video_clip.subclipped(0, video_duration)
        sub_audio = AudioFileClip(output_audio_path).subclipped(0, video_duration)

        final_video_clip = sub_video.with_audio(sub_audio)
        final_video_clip.write_videofile(
            output_video_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            logger=None
        )

        video_clip.close()
        sub_video.close()
        sub_audio.close()
        final_video_clip.close()

        _notify(1.0, f"Voice cloning & {target_language} video translation complete!")
        return output_video_path, output_audio_path, full_english_text, full_translated_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Pipeline error: {e}")
        return None, audio_output_path, f"Error: {str(e)}", ""


def analyze_and_preserve(
    video_path,
    target_language="Hindi",
    output_video_path=None,
    audio_output_path=None,
    progress_callback=None
):
    if output_video_path is None:
        output_video_path = os.path.join(OUTPUT_DIR, "processed_video.mp4")
    if audio_output_path is None:
        audio_output_path = os.path.join(OUTPUT_DIR, "preserved_original_voice.wav")
    return process_voice_cloning_pipeline(video_path, target_language, output_video_path, audio_output_path, progress_callback)

