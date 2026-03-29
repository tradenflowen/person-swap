# app/app.py
# HuggingFace Spaces Gradio interface
# This is what users see and interact with
# Wraps the entire 7-stage pipeline in a simple upload UI

import gradio as gr
import torch
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline"))

# ── Cache directory — persists across sessions ─────────
CACHE_DIR = os.environ.get("HF_HOME", "./hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# ── Global model cache — load once, reuse ──────────────
# Models stay loaded between requests on HuggingFace Spaces
# This makes second+ requests much faster
models = {}


def load_all_models():
    """
    Load all pipeline models into memory
    Called once on startup — takes 5-10 mins first time
    After that models stay cached in CACHE_DIR
    """
    global models

    if models:
        print("✓ Models already loaded")
        return

    print("Loading all pipeline models — this takes a few minutes first time...")

    from segment import load_sam2
    from pose import load_dwpose
    from bodyswap import load_bodyswap_pipeline
    from faceswap import load_faceswap_pipeline
    from hair import load_hair_pipeline, load_skin_enhancer
    from temporal import load_temporal_pipeline
    from upscale import load_upscaler

    models["sam2"] = load_sam2()
    models["dwpose"] = load_dwpose()
    models["bodyswap"] = load_bodyswap_pipeline()
    models["face_analyzer"], models["faceswap"] = load_faceswap_pipeline()
    models["hairfast"], models["device"] = load_hair_pipeline()
    models["skin_enhancer"] = load_skin_enhancer()
    models["raft"], models["propainter"], models["device"] = load_temporal_pipeline()
    models["upscaler"], _ = load_upscaler()

    print("✓ All models loaded and ready")


def process_swap(
    source_video,
    reference_image,
    max_frames,
    output_resolution,
    use_simple_temporal,
    skip_faceswap,
    progress=gr.Progress()
):
    """
    Main processing function called when user clicks Submit
    Runs the full 7-stage pipeline and returns output video
    """
    if source_video is None:
        return None, "❌ Please upload a source video"
    if reference_image is None:
        return None, "❌ Please upload a reference image"

    # Create temp directory for this job
    job_dir = tempfile.mkdtemp()
    output_path = os.path.join(job_dir, "output.mp4")

    try:
        # Parse resolution
        res_map = {
            "1080x1920 (TikTok/Reels vertical)": (1080, 1920),
            "1920x1080 (YouTube horizontal)":    (1920, 1080),
            "1080x1080 (Instagram square)":      (1080, 1080),
            "720x1280 (HD vertical)":            (720, 1280),
        }
        target_res = res_map.get(output_resolution, (1080, 1920))

        progress(0.05, "Loading models...")
        load_all_models()

        # Run pipeline
        progress(0.10, "Stage 1: Segmenting body...")
        from segment import segment_video
        frames, masks, fps = segment_video(
            source_video,
            max_frames=int(max_frames)
        )

        progress(0.20, "Stage 2: Estimating pose...")
        from pose import get_pose_sequence, get_body_bbox
        pose_images, keypoints_seq = get_pose_sequence(
            models["dwpose"], frames
        )
        face_bboxes = [get_body_bbox(k) for k in keypoints_seq]

        progress(0.35, "Stage 3: Transferring body appearance...")
        from bodyswap import process_video_bodyswap
        body_frames = process_video_bodyswap(
            frames, masks, pose_images,
            reference_image,
            models["bodyswap"]
        )

        progress(0.50, "Stage 4: Swapping face identity...")
        if not skip_faceswap:
            from faceswap import process_video_faceswap
            face_frames = process_video_faceswap(
                body_frames,
                reference_image,
                models["face_analyzer"],
                models["faceswap"]
            )
        else:
            face_frames = body_frames

        progress(0.65, "Stage 5: Transferring hair and skin...")
        from hair import process_video_hair
        hair_frames = process_video_hair(
            face_frames,
            reference_image,
            models["hairfast"],
            models["skin_enhancer"],
            face_bboxes=face_bboxes
        )

        progress(0.78, "Stage 6: Applying temporal consistency...")
        from temporal import process_video_temporal
        final_frames = process_video_temporal(
            hair_frames, masks,
            models["raft"],
            models["propainter"],
            models["device"],
            use_simple_fallback=use_simple_temporal
        )

        progress(0.88, "Stage 7: Upscaling to 4K...")
        from upscale import process_video_upscale, save_final_video
        upscaled_frames, output_fps = process_video_upscale(
            final_frames,
            original_fps=fps,
            target_resolution=target_res,
            do_interpolation=False,
            rife_model=None
        )

        progress(0.95, "Saving final video...")
        save_final_video(upscaled_frames, output_path, output_fps)

        progress(1.0, "✅ Done!")

        # Read stats
        frame_count = len(upscaled_frames)
        duration = frame_count / output_fps

        status = (
            f"✅ Complete!\n"
            f"Frames: {frame_count}\n"
            f"Duration: {duration:.1f}s\n"
            f"FPS: {output_fps:.1f}\n"
            f"Resolution: {target_res[0]}x{target_res[1]}"
        )

        return output_path, status

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, (
            "❌ Out of GPU memory.\n"
            "Try: fewer frames, enable Simple Temporal, or enable Skip Face Swap"
        )
    except Exception as e:
        return None, f"❌ Error: {str(e)}"
    finally:
        # Clean up temp files but keep output
        pass


# ── Gradio UI ──────────────────────────────────────────
with gr.Blocks(
    title="Person Swap AI",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🎭 Person Swap AI
    **Full body person swap — face, body, hair, skin — better than CapCut**
    
    Upload a source video and a reference photo of the target person.
    The AI will replace the person in the video completely.
    """)

    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Inputs")

            source_video = gr.Video(
                label="Source Video (person to replace)",
                sources=["upload"]
            )
            reference_image = gr.Image(
                label="Reference Photo (target person — full body preferred)",
                type="filepath",
                sources=["upload"]
            )

            gr.Markdown("### ⚙️ Settings")

            max_frames = gr.Slider(
                minimum=10,
                maximum=300,
                value=60,
                step=10,
                label="Max frames to process (fewer = faster)"
            )
            output_resolution = gr.Dropdown(
                choices=[
                    "1080x1920 (TikTok/Reels vertical)",
                    "1920x1080 (YouTube horizontal)",
                    "1080x1080 (Instagram square)",
                    "720x1280 (HD vertical)",
                ],
                value="1080x1920 (TikTok/Reels vertical)",
                label="Output resolution"
            )

            with gr.Row():
                use_simple_temporal = gr.Checkbox(
                    label="Simple temporal (saves VRAM)",
                    value=False
                )
                skip_faceswap = gr.Checkbox(
                    label="Skip face swap (saves VRAM)",
                    value=False
                )

            submit_btn = gr.Button(
                "🚀 Start Person Swap",
                variant="primary",
                size="lg"
            )

        # Right column — output
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Output")

            output_video = gr.Video(
                label="Swapped Video",
                interactive=False
            )
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                lines=6
            )

    gr.Markdown("""
    ### 💡 Tips for best results
    - Use a **short clip** (2-5 seconds) for faster processing
    - Reference photo should be **clear, front-facing, full body**
    - If you get memory errors, enable **Simple temporal** and/or **Skip face swap**
    - First run takes longer — models download and cache automatically
    
    ### ⏱️ Expected processing time
    | Frames | Approx time |
    |--------|-------------|
    | 30     | ~8 mins     |
    | 60     | ~15 mins    |
    | 120    | ~28 mins    |
    """)

    # Wire up the button
    submit_btn.click(
        fn=process_swap,
        inputs=[
            source_video,
            reference_image,
            max_frames,
            output_resolution,
            use_simple_temporal,
            skip_faceswap
        ],
        outputs=[output_video, status_text]
    )

# ── Launch ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        share=True,          # creates public URL
        show_error=True,
        max_threads=1        # one job at a time on free GPU
    )