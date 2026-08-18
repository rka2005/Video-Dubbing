import os
from pathlib import Path

def validate_video(video_path):
    if not video_path:
        return False, "No video uploaded."

    if not os.path.exists(video_path):
        return False, "Video file was not found."

    return True, None


def get_safe_output_path(file_name, extension):
    base_name = Path(file_name).stem
    return str(Path(f"{base_name}_{extension}"))
