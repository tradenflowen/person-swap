# pipeline/hair.py
# Stage 5: Hair and skin transfer using HairFastGAN
# Takes swapped body frame + target reference
# Returns frame with target person's hair style, color and skin tone matched
# This is the stage that fixes the "wig effect" seen in other apps

import torch
import numpy as np
import cv2
from PIL import Image

def load_hair_pipeline():
    """
    Load HairFastGAN model
    Downloads once, cached to Drive after that
    """
    from huggingface_hub import snapshot_download
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Download HairFastGAN weights
    model_path = snapshot_download(
        repo_id="AIRI-Institute/HairFastGAN",
        local_dir="./checkpoints/hairfastgan"
    )

    # HairFastGAN is already in sys.path from bootstrap
    # Import from correct module structure
    from models.HairFast import HairFast
    from options.train_options import TrainOptions
    
    parser = TrainOptions()
    args = parser.parse()
    
    # Override args for inference
    args.device = device
    args.latent_path = f"{model_path}/latent_code_base.npy"
    args.ckpt = f"{model_path}/ckpt/base.pt"

    hair_fast = HairFast(args)
    hair_fast.model.eval()
    
    print("✓ HairFastGAN loaded")
    return hair_fast, device

def load_skin_enhancer():
    """
    Prepare GPEN for face enhancement
    GPEN runs as a CLI tool, not importable Python class
    """
    import subprocess
    
    gpen_path = "/kaggle/working/external/GPEN"
    checkpoint_path = "./checkpoints/gpen/GPEN-BFR-512.pth"
    
    if not os.path.exists(checkpoint_path):
        print("⚠ GPEN checkpoint not found - skipping skin enhancement")
        return None
    
    print("✓ GPEN ready (will run via CLI)")
    return {
        "gpen_path": gpen_path,
        "checkpoint": checkpoint_path
    }

def segment_hair_region(frame_rgb, face_bbox):
    """
    Roughly segment the hair region above the face
    Used to apply hair transfer only to the right area
    
    face_bbox: dict with x1,y1,x2,y2 from faceswap.py
    """
    h, w = frame_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if face_bbox is None:
        return mask

    x1 = face_bbox["x1"]
    y1 = face_bbox["y1"]
    x2 = face_bbox["x2"]
    y2 = face_bbox["y2"]

    # Hair region: above the face, same width + padding
    hair_top = max(0, y1 - (y2 - y1))
    hair_bottom = int(y1 + (y2 - y1) * 0.3)
    hair_left = max(0, x1 - 20)
    hair_right = min(w, x2 + 20)

    mask[hair_top:hair_bottom, hair_left:hair_right] = 255

    # Soften mask
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    return mask


def transfer_hair(
    hair_fast,
    source_frame_rgb,
    target_reference_rgb,
    align=True
):
    """
    Transfer hair style and color from target to source frame
    
    align: True = align faces before transfer (more accurate)
    """
    source_pil = Image.fromarray(source_frame_rgb).convert("RGB")
    target_pil = Image.fromarray(target_reference_rgb).convert("RGB")

    # HairFastGAN expects: face image, hair reference image
    try:
        result = hair_fast.swap(
            source_pil,   # face to apply hair to
            target_pil,   # hair style reference
            target_pil,   # color reference (same as style)
            align=align
        )
        return np.array(result)
    except Exception as e:
        print(f"⚠ Hair transfer failed on this frame: {e}")
        return source_frame_rgb


def enhance_skin(enhancer_config, frame_rgb):
    """
    Run GPEN skin enhancement via CLI
    """
    if enhancer_config is None:
        return frame_rgb
    
    try:
        import subprocess
        import tempfile
        
        # Save frame to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
            input_path = tmp_in.name
            Image.fromarray(frame_rgb).save(input_path)
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
            output_path = tmp_out.name
        
        # Run GPEN CLI
        cmd = [
            "python",
            f"{enhancer_config['gpen_path']}/face_enhancement.py",
            "--model", "GPEN-BFR-512",
            "--in_size", "512",
            "--channel_multiplier", "2",
            "--narrow", "1",
            "--indir", os.path.dirname(input_path),
            "--outdir", os.path.dirname(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Load result
        enhanced_rgb = np.array(Image.open(output_path).convert("RGB"))
        
        # Cleanup
        os.unlink(input_path)
        os.unlink(output_path)
        
        return enhanced_rgb
        
    except Exception as e:
        print(f"⚠ GPEN enhancement failed: {e}")
        return frame_rgb

def blend_hair_result(
    swapped_frame,
    hair_result,
    hair_mask,
    blend_strength=0.85
):
    """
    Blend hair transfer result back into swapped frame
    Only replaces the hair region, keeps everything else intact
    
    blend_strength: 0.8-0.95 recommended
    """
    h, w = swapped_frame.shape[:2]

    # Resize hair result if needed
    if hair_result.shape[:2] != (h, w):
        hair_result = cv2.resize(hair_result, (w, h))

    # Apply blend strength to mask
    soft_mask = (hair_mask[:, :, np.newaxis] / 255.0) * blend_strength

    blended = (
        hair_result * soft_mask +
        swapped_frame * (1 - soft_mask)
    ).astype(np.uint8)

    return blended


def process_video_hair(
    swapped_frames,
    reference_image_path,
    hair_fast,
    enhancer,
    face_bboxes=None
):
    """
    Main function — applies hair transfer and skin enhancement
    across all video frames

    swapped_frames: frames output from bodyswap.py
    reference_image_path: same target person reference photo
    hair_fast: loaded HairFastGAN model
    enhancer: loaded GPEN model
    face_bboxes: list of face bounding boxes per frame (optional)
    """
    print("Loading target reference for hair...")
    target_rgb = np.array(
        Image.open(reference_image_path).convert("RGB")
    )

    result_frames = []
    total = len(swapped_frames)

    print(f"Applying hair transfer + skin enhancement to {total} frames...")

    for i, frame in enumerate(swapped_frames):
        # Get face bbox for this frame if available
        bbox = face_bboxes[i] if face_bboxes else None

        # Step 1: Transfer hair
        hair_result = transfer_hair(hair_fast, frame, target_rgb)

        # Step 2: Get hair mask for this frame
        hair_mask = segment_hair_region(frame, bbox)

        # Step 3: Blend hair into frame
        blended = blend_hair_result(frame, hair_result, hair_mask)

        # Step 4: Enhance skin texture
        enhanced = enhance_skin(enhancer, blended)

        result_frames.append(enhanced)

        if i % 10 == 0:
            print(f"  Frame {i}/{total} done")

    print(f"✓ Hair + skin processing complete — {len(result_frames)} frames")
    return result_frames


def save_hair_preview(result_frames, output_path="hair_preview.mp4", fps=30):
    """Save hair transfer result video"""
    h, w = result_frames[0].shape[:2]
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )
    for frame in result_frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()
    print(f"✓ Hair preview saved to {output_path}")


# ── Test block ──
if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from segment import extract_frames
    from pose import load_dwpose, get_pose_sequence
    from bodyswap import (
        load_bodyswap_pipeline,
        process_video_bodyswap
    )
    from segment import segment_video

    if len(sys.argv) < 3:
        print("Usage: python hair.py video.mp4 reference.jpg")
        sys.exit(1)

    video_path = sys.argv[1]
    reference_path = sys.argv[2]

    # Run previous stages
    print("=== Stage 1: Segmentation ===")
    frames, masks, fps = segment_video(video_path)

    print("=== Stage 2: Pose ===")
    detector = load_dwpose()
    pose_images, _ = get_pose_sequence(detector, frames)

    print("=== Stage 3: Body Swap ===")
    body_pipe = load_bodyswap_pipeline()
    swapped_frames = process_video_bodyswap(
        frames, masks, pose_images,
        reference_path, body_pipe
    )

    print("=== Stage 4: Hair + Skin ===")
    hair_fast, device = load_hair_pipeline()
    enhancer = load_skin_enhancer()
    result_frames = process_video_hair(
        swapped_frames, reference_path,
        hair_fast, enhancer
    )

    save_hair_preview(result_frames, "hair_preview.mp4", fps)
    print("Done! Check hair_preview.mp4")