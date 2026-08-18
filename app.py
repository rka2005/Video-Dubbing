import os
import gradio as gr
from voice_cloning_pipeline import analyze_and_preserve, EDGE_TTS_VOICES
from translator import LANGUAGE_CODES

SUPPORTED_TARGET_LANGUAGES = [
    lang for lang in LANGUAGE_CODES
    if lang != "English" and lang in EDGE_TTS_VOICES
]

def process_video(video_path, target_language="Hindi", progress=gr.Progress()):
    if not video_path:
        return None, None, "Please upload an English video file.", ""

    def update_progress(pct, desc):
        progress(pct, desc=desc)

    output_video_path, audio_output_path, analysis_text, translation_text = analyze_and_preserve(
        video_path=video_path,
        target_language=target_language,
        progress_callback=update_progress
    )
    return output_video_path, audio_output_path, analysis_text, translation_text


with gr.Blocks(title="Author Voice Cloning Video Translator") as demo:
    gr.Markdown("# 🎙️ Author Voice Cloning & Video Translator")
    gr.Markdown(
        "Upload an English video and select a target language. This tool performs **zero-shot voice cloning** "
        "(cloning the author's exact vocal identity, gender, accent, emotion, speaking style, pauses, "
        "and segment timing) to generate a translated audio track and merge it back onto the video!"
    )

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload English Video")

            target_language = gr.Dropdown(
                choices=SUPPORTED_TARGET_LANGUAGES,
                value="Hindi",
                label="Target Language (Zero-Shot Voice Cloning)",
                interactive=True,
            )
            analyze_btn = gr.Button(
                "🎙️ Clone Author Voice & Translate Video",
                variant="primary"
            )

        with gr.Column():
            video_output = gr.Video(label="🎬 Output Video (Cloned Author Voice)")
            audio_output = gr.Audio(label="🔊 Cloned Translated Audio Track")
            text_output = gr.Textbox(label="📜 Original English Transcription", lines=4)
            translated_output = gr.Textbox(label="🌐 Translated Text", lines=4)

    analyze_btn.click(
        fn=process_video,
        inputs=[video_input, target_language],
        outputs=[video_output, audio_output, text_output, translated_output],
    )


if __name__ == "__main__":
    print("Starting application... Open the local link in your browser to upload a video.")
    demo.launch()
