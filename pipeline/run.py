# pipeline/run.py
# Master pipeline — connects all 7 stages end to end
# This is the single entry point for the full person swap
# Usage: python run.py source_video.mp4 reference_image.jpg output.mp4

import torch
import gc
import os
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.dirname(__file__))


def free_vram():
    """Free GPU memory between stages — critical on T4/free GPUs"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_stage(n, name):
    print(f"\n{'='*50}")
    print(f"  STAGE {n}: {name}")
    print(f"{'='*50}")


def run_pipeline(
    source_video_path,
    reference_image_path,
    output_path="output.mp4",
    max_frames=120,
    target_resolution=(1080, 1920),
    do_interpolation=False,
    use_simple_temporal=False,
    skip_faceswap=False
):
    """
    Full person swap pipeline — runs all 7 stages in sequence

    source_video_path:    video of person to be replaced
    reference_image_path: photo of target person (full body preferred)
    output_path:          where to save the result
    max_frames:           limit frames for testing (None = full video)
    target_resolution:    output size (width, height)
    do_interpolation:     double frame rate with RIFE (needs more VRAM)
    use_simple_temporal:  use lightweight temporal smoothing (saves VRAM)
    skip_faceswap:        skip InstantID if VRAM is very tight
    """
    start_time = time.time()

    # Validate inputs
    if not os.path.exists(source_video_path):
        raise FileNotFoundError(f"Source video not found: {source_video_path}")
    if not os.path.exists(reference_image_path):
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")

    print(f"\n🎬 Person Swap Pipeline Starting")
    print(f"   Source:     {source_video_path}")
    print(f"   Reference:  {reference_image_path}")
    print(f"   Output:     {output_path}")
    print(f"   Max frames: {max_frames}")
    print(f"   Resolution: {target_resolution}")

    # ── Stage 1: Segmentation ──────────────────────────
    print_stage(1, "Body Segmentation (SAM 2)")
    from segment import load_sam2, extract_frames, segment_video

    frames, masks, fps = segment_video(
        source_video_path,
        max_frames=max_frames
    )
    print(f"   Extracted {len(frames)} frames at {fps:.1f} fps")
    free_vram()

    # ── Stage 2: Pose Estimation ───────────────────────
    print_stage(2, "Pose Estimation (DWPose)")
    from pose import load_dwpose, get_pose_sequence

    detector = load_dwpose()
    pose_images, keypoints_seq = get_pose_sequence(detector, frames)
    del detector
    free_vram()

    # Extract face bboxes from keypoints for hair stage
    from pose import get_body_bbox
    face_bboxes = [get_body_bbox(kpts) for kpts in keypoints_seq]

    # ── Stage 3: Body Swap ─────────────────────────────
    print_stage(3, "Full Body Appearance Transfer (OOTDiffusion)")
    from bodyswap import load_bodyswap_pipeline, process_video_bodyswap

    body_pipe = load_bodyswap_pipeline()
    body_frames = process_video_bodyswap(
        frames, masks, pose_images,
        reference_image_path, body_pipe
    )
    del body_pipe
    free_vram()

    # ── Stage 4: Face Swap ─────────────────────────────
    if not skip_faceswap:
        print_stage(4, "Face Identity Swap (InstantID)")
        from faceswap import (
            load_faceswap_pipeline,
            process_video_faceswap
        )

        face_analyzer, face_pipe = load_faceswap_pipeline()
        face_frames = process_video_faceswap(
            body_frames,
            reference_image_path,
            face_analyzer,
            face_pipe
        )
        del face_pipe
        free_vram()
    else:
        print_stage(4, "Face Swap SKIPPED (skip_faceswap=True)")
        face_frames = body_frames

    # ── Stage 5: Hair + Skin ───────────────────────────
    print_stage(5, "Hair Transfer + Skin Enhancement (HairFastGAN + GPEN)")
    from hair import (
        load_hair_pipeline,
        load_skin_enhancer,
        process_video_hair
    )

    hair_fast, device = load_hair_pipeline()
    enhancer = load_skin_enhancer()
    hair_frames = process_video_hair(
        face_frames,
        reference_image_path,
        hair_fast, enhancer,
        face_bboxes=face_bboxes
    )
    del hair_fast, enhancer
    free_vram()

    # ── Stage 6: Temporal Consistency ─────────────────
    print_stage(6, "Temporal Consistency (ProPainter + RAFT)")
    from temporal import load_temporal_pipeline, process_video_temporal

    if use_simple_temporal:
        raft_model, propainter, device = None, None, "cuda"
    else:
        raft_model, propainter, device = load_temporal_pipeline()

    final_frames = process_video_temporal(
        hair_frames, masks,
        raft_model, propainter, device,
        use_simple_fallback=use_simple_temporal
    )
    del raft_model, propainter
    free_vram()

    # ── Stage 7: Upscale + Interpolation ──────────────
    print_stage(7, "Upscaling + Frame Interpolation (Real-ESRGAN + RIFE)")
    from upscale import (
        load_upscaler,
        load_rife_interpolator,
        process_video_upscale,
        save_final_video
    )

    rife_model = None
    if do_interpolation:
        try:
            rife_model = load_rife_interpolator()
        except Exception as e:
            print(f"⚠ RIFE load failed: {e} — skipping interpolation")

    upscaled_frames, output_fps = process_video_upscale(
        final_frames,
        original_fps=fps,
        target_resolution=target_resolution,
        do_interpolation=do_interpolation,
        rife_model=rife_model
    )

    # ── Save output ────────────────────────────────────
    print_stage("✓", "Saving Final Video")
    save_final_video(upscaled_frames, output_path, output_fps)

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*50}")
    print(f"  ✅ PIPELINE COMPLETE")
    print(f"  Output: {output_path}")
    print(f"  Time:   {minutes}m {seconds}s")
    print(f"  Frames: {len(upscaled_frames)} @ {output_fps:.1f} fps")
    print(f"{'='*50}\n")

    return output_path


# ── CLI entry point ────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full body person swap pipeline"
    )
    parser.add_argument(
        "source_video",
        help="Path to source video (person to replace)"
    )
    parser.add_argument(
        "reference_image",
        help="Path to target person reference photo"
    )
    parser.add_argument(
        "--output", "-o",
        default="output.mp4",
        help="Output video path (default: output.mp4)"
    )
    parser.add_argument(
        "--max-frames", "-n",
        type=int,
        default=120,
        help="Max frames to process (default: 120, None=full video)"
    )
    parser.add_argument(
        "--resolution", "-r",
        default="1080x1920",
        help="Output resolution WxH (default: 1080x1920 for TikTok)"
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Double frame rate with RIFE (needs more VRAM)"
    )
    parser.add_argument(
        "--simple-temporal",
        action="store_true",
        help="Use lightweight temporal smoothing (saves VRAM)"
    )
    parser.add_argument(
        "--skip-faceswap",
        action="store_true",
        help="Skip InstantID face swap (saves VRAM)"
    )

    args = parser.parse_args()

    # Parse resolution
    w, h = map(int, args.resolution.split("x"))

    run_pipeline(
        source_video_path=args.source_video,
        reference_image_path=args.reference_image,
        output_path=args.output,
        max_frames=args.max_frames,
        target_resolution=(w, h),
        do_interpolation=args.interpolate,
        use_simple_temporal=args.simple_temporal,
        skip_faceswap=args.skip_faceswap
    )