# pipeline/upscale.py
# Stage 7: 4x resolution upscaling using Real-ESRGAN
# Also runs RIFE frame interpolation to smooth motion
# Final quality polish before output

import torch
import numpy as np
import cv2
from PIL import Image

def load_upscaler():
    """
    Load Real-ESRGAN 4x upscaler
    Downloads once, cached to Drive after that
    """
    from huggingface_hub import hf_hub_download
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Download Real-ESRGAN weights
    model_path = hf_hub_download(
        repo_id="ai-forever/Real-ESRGAN",
        filename="RealESRGAN_x4plus.pth",
        local_dir="./checkpoints/realesrgan"
    )

    # Build model
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=512,          # process in tiles to save VRAM
        tile_pad=10,
        pre_pad=0,
        half=True if device == "cuda" else False,
        device=device
    )

    print("✓ Real-ESRGAN 4x upscaler loaded")
    return upscaler, device


def load_rife_interpolator():
    """
    Load RIFE frame interpolator
    Doubles frame rate for smoother motion (24fps → 48fps etc)
    """
    from huggingface_hub import snapshot_download
    import sys

    rife_path = snapshot_download(
        repo_id="hzwer/ECCV2022-RIFE",
        local_dir="./checkpoints/rife"
    )
    sys.path.append(rife_path)

    from model.RIFE_HDv3 import Model

    model = Model()
    model.load_model(f"{rife_path}/train_log", -1)
    model.eval()
    model.device()

    print("✓ RIFE frame interpolator loaded")
    return model


def upscale_frame(upscaler, frame_rgb):
    """
    Upscale a single frame 4x using Real-ESRGAN
    Input:  e.g. 512x912 frame
    Output: e.g. 2048x3648 frame
    """
    try:
        # Real-ESRGAN expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        upscaled_bgr, _ = upscaler.enhance(frame_bgr, outscale=4)
        upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        return upscaled_rgb
    except Exception as e:
        print(f"⚠ Upscale failed on frame: {e}")
        # Fallback: simple bicubic upscale
        h, w = frame_rgb.shape[:2]
        return cv2.resize(
            frame_rgb, (w * 4, h * 4),
            interpolation=cv2.INTER_CUBIC
        )


def upscale_sequence(upscaler, frames, target_size=None):
    """
    Upscale all frames in sequence
    target_size: optional (width, height) to resize after upscale
    e.g. (1080, 1920) for standard vertical TikTok format
    """
    result_frames = []
    total = len(frames)

    print(f"Upscaling {total} frames at 4x...")

    for i, frame in enumerate(frames):
        upscaled = upscale_frame(upscaler, frame)

        # Resize to target if specified
        if target_size is not None:
            upscaled = cv2.resize(
                upscaled,
                target_size,
                interpolation=cv2.INTER_LANCZOS4
            )

        result_frames.append(upscaled)

        if i % 5 == 0:
            print(f"  Frame {i}/{total} upscaled")

    print(f"✓ Upscaling complete — output size: {result_frames[0].shape}")
    return result_frames


def interpolate_frames(rife_model, frame1_rgb, frame2_rgb):
    """
    Generate an intermediate frame between two frames using RIFE
    Makes motion look smoother by doubling frame count
    """
    def to_tensor(frame):
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).cuda()

    with torch.no_grad():
        img0 = to_tensor(frame1_rgb)
        img1 = to_tensor(frame2_rgb)
        mid = rife_model.inference(img0, img1)
        mid = mid[0].cpu().numpy().transpose(1, 2, 0)
        mid = (mid * 255).astype(np.uint8)

    return mid


def interpolate_sequence(rife_model, frames):
    """
    Double the frame count using RIFE interpolation
    24fps video becomes 48fps, 30fps becomes 60fps
    Makes the swap look much more fluid on TikTok
    """
    result_frames = []
    total = len(frames)

    print(f"Interpolating {total} frames to {total * 2} frames...")

    for i in range(total - 1):
        result_frames.append(frames[i])

        # Generate intermediate frame
        mid_frame = interpolate_frames(rife_model, frames[i], frames[i + 1])
        result_frames.append(mid_frame)

        if i % 10 == 0:
            print(f"  Interpolated {i}/{total - 1} pairs")

    # Add last frame
    result_frames.append(frames[-1])

    print(f"✓ Interpolation complete — {len(result_frames)} frames")
    return result_frames


def save_final_video(
    frames,
    output_path="final_output.mp4",
    fps=30,
    crf=18
):
    """
    Save final high quality video
    crf: quality factor — 18 = very high, 23 = standard
    Lower crf = better quality, larger file
    Uses H.264 for maximum compatibility with TikTok/Instagram
    """
    h, w = frames[0].shape[:2]

    # Write to temp file first
    temp_path = output_path.replace(".mp4", "_temp.mp4")

    out = cv2.VideoWriter(
        temp_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()

    # Re-encode with ffmpeg for proper H.264 + audio support
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_path,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("⚠ ffmpeg not found — saving without re-encode")
        import shutil
        shutil.move(temp_path, output_path)
    else:
        import os
        os.remove(temp_path)
        print(f"✓ Final video saved: {output_path}")

    return output_path


def process_video_upscale(
    temporal_frames,
    original_fps=30,
    target_resolution=(1080, 1920),
    do_interpolation=True,
    rife_model=None
):
    """
    Main function — upscales and optionally interpolates final video

    temporal_frames: frames from temporal.py
    original_fps: fps of source video
    target_resolution: (width, height) — (1080, 1920) = TikTok vertical
    do_interpolation: True = double frame rate with RIFE
    """
    upscaler, device = load_upscaler()

    # Step 1: Upscale all frames
    print("=== Upscaling frames ===")
    upscaled_frames = upscale_sequence(
        upscaler,
        temporal_frames,
        target_size=target_resolution
    )

    # Step 2: Frame interpolation (optional — needs extra VRAM)
    output_fps = original_fps
    if do_interpolation and rife_model is not None:
        print("=== Interpolating frames ===")
        try:
            upscaled_frames = interpolate_sequence(rife_model, upscaled_frames)
            output_fps = original_fps * 2
            print(f"✓ Frame rate doubled: {original_fps} → {output_fps} fps")
        except torch.cuda.OutOfMemoryError:
            print("⚠ Not enough VRAM for interpolation — skipping")
            torch.cuda.empty_cache()

    return upscaled_frames, output_fps


# ── Test block ──
if __name__ == "__main__":
    import sys
    sys.path.append(".")

    if len(sys.argv) < 2:
        print("Usage: python upscale.py video.mp4")
        sys.exit(1)

    # Quick standalone test — upscale any video
    video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    print(f"Loaded {len(frames)} frames at {fps} fps")

    upscaled, out_fps = process_video_upscale(
        frames,
        original_fps=fps,
        do_interpolation=False  # skip RIFE for quick test
    )

    save_final_video(upscaled, "upscale_test.mp4", out_fps)
    print("Done! Check upscale_test.mp4")