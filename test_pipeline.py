import os
import sys
import numpy as np
from moviepy import ColorClip, AudioClip, VideoFileClip
from voice_cloning_pipeline import (
    analyze_and_preserve,
    process_voice_cloning_pipeline,
    clone_voice_f5,
    extract_author_reference,
    get_f5_model
)
from translator import translate_text, translate_segments

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_sample_video(filename="sample_test_video.mp4", duration=4):
    """Generate a synthetic test video with audio tone for testing pipeline."""
    print("Generating synthetic sample video for testing...")
    clip = ColorClip(size=(320, 240), color=(30, 80, 140), duration=duration)

    def make_frame(t):
        return np.sin(2 * np.pi * 350 * t)

    audio = AudioClip(make_frame, duration=duration, fps=44100)
    video_with_audio = clip.with_audio(audio)
    video_with_audio.write_videofile(filename, fps=24, codec="libx264", audio_codec="aac", logger=None)
    clip.close()
    video_with_audio.close()
    return filename

def test_voice_cloning_pipeline():
    print("=== STARTING VOICE CLONING PIPELINE VERIFICATION TEST ===")
    test_video = "test_spoken.mp4" if os.path.exists("test_spoken.mp4") else ("test.mp4" if os.path.exists("test.mp4") else "sample_test_video.mp4")
    if not os.path.exists(test_video):
        generate_sample_video(test_video)
    print(f"Using reference video: {test_video}")

    sample_segments = [
        {"start": 0.0, "end": 2.0, "text": "Welcome to our speech translation system."},
        {"start": 2.5, "end": 4.0, "text": "It preserves the author's original voice identity."}
    ]
    translated = translate_segments(sample_segments, "Hindi")
    print("Translated Segments:")
    for seg in translated:
        print(f"  [{seg['start']}s - {seg['end']}s] {seg['orig_text']} -> {seg['hindi_text']}")
    assert len(translated) == 2, "Segment translation failed!"

    audio_full = "test_extracted.wav"
    v = VideoFileClip(test_video)
    if v.audio:
        v.audio.write_audiofile(audio_full, logger=None)
    v.close()

    ref_wav, ref_txt = extract_author_reference(
        audio_full,
        sample_segments,
        ref_wav_path="test_author_ref.wav",
        ref_txt_path="test_author_ref.txt"
    )
    print(f"Extracted Ref WAV: {ref_wav}, Text: '{ref_txt}'")
    assert os.path.exists(ref_wav) and os.path.getsize(ref_wav) > 0, "Reference WAV creation failed!"
    assert os.path.exists("test_author_ref.txt"), "Reference text file missing!"

    out_wav = "test_cloned_hindi_pipeline.wav"
    cloned_path = clone_voice_f5(ref_wav, ref_txt, "नमस्ते दुनिया", out_wav)
    print(f"Cloned audio path: {cloned_path}")
    assert cloned_path and os.path.exists(cloned_path), "Voice cloning synthesis failed!"
    assert os.path.getsize(cloned_path) > 0, "Cloned audio file is empty!"

    print("=== ALL VOICE CLONING UNIT TESTS PASSED! ===")


if __name__ == "__main__":
    test_voice_cloning_pipeline()
