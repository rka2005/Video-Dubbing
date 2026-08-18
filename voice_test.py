import os
import sys
import asyncio
import numpy as np
import soundfile as sf
import librosa
from moviepy import VideoFileClip, ColorClip, AudioFileClip
import whisper
import edge_tts

from voice_cloning_pipeline import ensure_ffmpeg_on_path
ensure_ffmpeg_on_path()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torchaudio

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

# Global reference model
_f5_model_instance = None


def get_f5_model(model_choice="base"):
    """
    Lazy loader for F5TTS zero-shot voice cloning model.
    Loads F5TTS model directly from local cache to prevent network download freezes.
    """
    global _f5_model_instance
    if _f5_model_instance is not None:
        return _f5_model_instance

    print("Initializing F5-TTS Zero-Shot Voice Cloning Model...")
    from f5_tts.api import F5TTS

    try:
        print("Loading cached F5TTS zero-shot model (SWivid/F5-TTS)...")
        _f5_model_instance = F5TTS()
        print("F5TTS model loaded successfully!")
    except Exception as base_err:
        print(f"Error initializing F5TTS model: {base_err}")
        _f5_model_instance = None

    return _f5_model_instance


def generate_speech_test_video(video_path="test_spoken.mp4", duration=6):
    """
    Generates a synthetic test video with spoken English audio using Edge-TTS
    if no test video with speech is present.
    """
    print(f"Generating synthetic English spoken test video: {video_path}...")
    sample_text = "Hello everyone, welcome to this video translation test. We are verifying author voice cloning."
    temp_wav = "temp_spoken_eng.wav"

    async def _make_speech():
        communicate = edge_tts.Communicate(sample_text, "en-US-GuyNeural")
        await communicate.save(temp_wav)

    asyncio.run(_make_speech())

    if os.path.exists(temp_wav):
        audio_clip = AudioFileClip(temp_wav)
        dur = max(duration, audio_clip.duration)
        color_clip = ColorClip(size=(320, 240), color=(20, 60, 100), duration=dur)
        video_clip = color_clip.with_audio(audio_clip)
        video_clip.write_videofile(video_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
        color_clip.close()
        video_clip.close()
        audio_clip.close()
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        return video_path
    return None


def extract_author_reference(
    full_audio_path,
    segments,
    ref_wav_path="author_reference.wav",
    ref_txt_path="author_reference.txt",
    min_dur=5.0,
    max_dur=12.0
):
    """
    Extracts a clean continuous 5-12 second speech clip and matching transcript.
    Ensures ref_audio (author_reference.wav) and ref_text (author_reference.txt)
    are strictly aligned.
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

    print("\n--- AUTHOR REFERENCE EXTRACTION RESULT ---")
    print(f"Reference Audio File: {ref_wav_path}")
    print(f"Time Window: {selected_start:.2f}s to {selected_end:.2f}s (Duration: {(selected_end - selected_start):.2f}s)")
    print(f"Matching Transcript ({ref_txt_path}): '{clean_ref_text}'")
    print("-------------------------------------------\n")

    return ref_wav_path, clean_ref_text


def run_stage_1_voice_test(video_path=None):
    """
    Executes isolated Stage 1 pipeline:
    1. Extract audio from video.
    2. Transcribe speech using Whisper.
    3. Crop clean 5-12 sec reference clip (author_reference.wav + author_reference.txt).
    4. Synthesize Hindi test audio using Hindi F5-TTS model.
    """
    if not video_path:
        video_path = sys.argv[1] if len(sys.argv) > 1 else ("test_spoken.mp4" if os.path.exists("test_spoken.mp4") else ("test.mp4" if os.path.exists("test.mp4") else None))

    if not video_path or not os.path.exists(video_path):
        print("Target video not found. Generating synthetic speech test video...")
        video_path = generate_speech_test_video("test_spoken.mp4")

    print(f"=== STAGE 1 ISOLATED VOICE CLONING TEST ===")
    print(f"Input Video: {video_path}")

    # 1. Extract audio track
    audio_extracted = "test_extracted_audio.wav"
    vclip = VideoFileClip(video_path)
    if vclip.audio is None:
        vclip.close()
        print("Video has no audio track! Generating spoken test video...")
        video_path = generate_speech_test_video("test_spoken.mp4")
        vclip = VideoFileClip(video_path)

    vclip.audio.write_audiofile(audio_extracted, logger=None)
    vclip.close()

    # 2. Transcribe with Whisper
    print("Running Whisper speech detection & timestamp analysis...")
    whisper_model = whisper.load_model("base")
    transcription = whisper_model.transcribe(audio_extracted, word_timestamps=True)
    segments = transcription.get("segments", [])
    full_text = transcription.get("text", "").strip()

    if not full_text:
        print("No speech detected in video audio! Regenerating spoken test video...")
        video_path = generate_speech_test_video("test_spoken.mp4")
        vclip = VideoFileClip(video_path)
        vclip.audio.write_audiofile(audio_extracted, logger=None)
        vclip.close()
        transcription = whisper_model.transcribe(audio_extracted, word_timestamps=True)
        segments = transcription.get("segments", [])
        full_text = transcription.get("text", "").strip()

    print(f"Full Detected Transcription: '{full_text}'")

    # 3. Extract 5-12 sec clean author reference audio & matching text
    ref_audio_path, ref_text = extract_author_reference(
        audio_extracted,
        segments,
        ref_wav_path="author_reference.wav",
        ref_txt_path="author_reference.txt"
    )

    # 4. Load F5-TTS Hindi Model
    f5 = get_f5_model(model_choice="hindi")
    if not f5:
        print("Failed to initialize F5-TTS model.")
        return False

    # 5. Synthesize Test Hindi Speech
    test_hindi_text = "नमस्ते दोस्तों! यह लेखक की आवाज़ में हिंदी वॉइस क्लोनिंग का परीक्षण है।"
    test_output_wav = "test_hindi.wav"

    print(f"Synthesizing Hindi test clip: '{test_hindi_text}'...")
    print(f"Using ref_audio: '{ref_audio_path}'")
    print(f"Using ref_text:  '{ref_text}'")

    f5.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=test_hindi_text,
        file_wave=test_output_wav,
        remove_silence=False
    )

    if os.path.exists(test_output_wav) and os.path.getsize(test_output_wav) > 0:
        info = sf.info(test_output_wav)
        print("\n=== STAGE 1 VOICE TEST SUCCESSFUL! ===")
        print(f"1. Author Reference Audio: {os.path.abspath(ref_audio_path)}")
        print(f"2. Author Reference Text:  {os.path.abspath('author_reference.txt')}")
        print(f"3. Cloned Hindi Audio:     {os.path.abspath(test_output_wav)} (Duration: {info.duration:.2f}s, SR: {info.samplerate}Hz)")
        print("You can now listen to author_reference.wav and test_hindi.wav to verify voice identity preservation.")
        return True
    else:
        print("ERROR: Test Hindi output WAV was not generated or is empty.")
        return False


if __name__ == "__main__":
    run_stage_1_voice_test()
