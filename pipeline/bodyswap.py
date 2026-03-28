# pipeline/bodyswap.py
# Stage 4: Full body appearance transfer using OOTDiffusion
# Takes source frame + target person reference image
# Returns frame with target person's full body appearance applied
# This handles clothes, body shape, skin tone across the whole body

import torch
import numpy as np
import cv2
from PIL import Image

def load_bodyswap_pipeline():
    """
    Load OOTDiffusion pipeline
    Downloads once, cached to Drive after that
    """
    from diffusers import UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer
    from huggingface_hub import snapshot_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Download OOTDiffusion weights
    model_path = snapshot_download(
        repo_id="levihsu/OOTDiffusion",
        local_dir="./checkpoints/ootdiffusion"
    )

    # Load pipeline components
    from pipelines.OOTDiffusion import OOTDiffusionPipeline

    pipe = OOTDiffusionPipeline(
        model_path=model_path,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    print("✓ OOTDiffusion pipeline loaded")
    return pipe


def prepare_body_inputs(
    source_frame_rgb,
    source_mask,
    pose_image,
    target_reference_rgb
):
    """
    Prepare all inputs needed for body appearance transfer

    source_frame_rgb: original video frame
    source_mask: person mask from segment.py (binary 0/255)
    pose_image: skeleton image from pose.py
    target_reference_rgb: full body photo of target person
    """
    h, w = source_frame_rgb.shape[:2]

    # Resize all inputs to model's expected size
    model_size = (768, 1024)  # width, height

    source_pil = Image.fromarray(source_frame_rgb).resize(model_size)
    mask_pil = Image.fromarray(source_mask).resize(model_size)
    pose_pil = Image.fromarray(pose_image).resize(model_size)
    target_pil = Image.fromarray(target_reference_rgb).resize(model_size)

    return source_pil, mask_pil, pose_pil, target_pil, (w, h)


def transfer_body_appearance(
    pipe,
    source_pil,
    mask_pil,
    pose_pil,
    target_pil,
    original_size,
    num_steps=20,
    guidance_scale=2.0,
    seed=42
):
    """
    Core body appearance transfer
    Applies target person's full appearance to source body pose

    guidance_scale: 1.5-2.5 works best for body swap
    num_steps: 20 is fast, 30 is higher quality
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Run OOTDiffusion
    result = pipe(
        image=source_pil,
        mask_image=mask_pil,
        pose_image=pose_pil,
        reference_image=target_pil,
        prompt="a person, photorealistic, high quality",
        negative_prompt=(
            "blurry, distorted, bad anatomy, "
            "extra limbs, missing limbs, low quality"
        ),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    # Resize back to original video dimensions
    result_resized = result.resize(original_size)
    return np.array(result_resized)


def apply_body_mask_blend(
    original_frame,
    swapped_body,
    mask,
    edge_blur=21
):
    """
    Blend swapped body back into original frame
    Only replaces pixels inside the person mask
    Softens edges for seamless result
    """
    # Soften mask edges
    soft_mask = cv2.GaussianBlur(mask, (edge_blur, edge_blur), 0)
    soft_mask_3ch = soft_mask[:, :, np.newaxis] / 255.0

    # Resize swapped body to match original frame if needed
    h, w = original_frame.shape[:2]
    if swapped_body.shape[:2] != (h, w):
        swapped_body = cv2.resize(swapped_body, (w, h))

    # Blend: inside mask = swapped body, outside = original background
    blended = (
        swapped_body * soft_mask_3ch +
        original_frame * (1 - soft_mask_3ch)
    ).astype(np.uint8)

    return blended


def process_video_bodyswap(
    frames,
    masks,
    pose_images,
    reference_image_path,
    pipe,
    batch_size=4
):
    """
    Main function — applies full body swap across all video frames

    frames: RGB frames from segment.py
    masks: body masks from segment.py
    pose_images: skeleton images from pose.py
    reference_image_path: full body photo of target person
    batch_size: process N frames at once for speed
    """
    # Load target reference image
    print("Loading target reference image...")
    target_rgb = np.array(
        Image.open(reference_image_path).convert("RGB")
    )

    result_frames = []
    total = len(frames)

    print(f"Transferring body appearance across {total} frames...")

    for i in range(0, total, batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_masks = masks[i:i + batch_size]
        batch_poses = pose_images[i:i + batch_size]

        for j, (frame, mask, pose) in enumerate(
            zip(batch_frames, batch_masks, batch_poses)
        ):
            # Prepare inputs
            source_pil, mask_pil, pose_pil, target_pil, orig_size = \
                prepare_body_inputs(frame, mask, pose, target_rgb)

            # Transfer appearance
            swapped = transfer_body_appearance(
                pipe,
                source_pil,
                mask_pil,
                pose_pil,
                target_pil,
                orig_size
            )

            # Blend back into original
            blended = apply_body_mask_blend(frame, swapped, mask)
            result_frames.append(blended)

        print(f"  Processed {min(i + batch_size, total)}/{total} frames")

    print(f"✓ Body swap complete — {len(result_frames)} frames processed")
    return result_frames


def save_bodyswap_preview(result_frames, output_path="bodyswap_preview.mp4", fps=30):
    """Save body swap result video"""
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
    print(f"✓ Body swap preview saved to {output_path}")


# ── Test block ──
if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from segment import extract_frames, segment_video
    from pose import load_dwpose, get_pose_sequence

    if len(sys.argv) < 3:
        print("Usage: python bodyswap.py video.mp4 reference_fullbody.jpg")
        sys.exit(1)

    video_path = sys.argv[1]
    reference_path = sys.argv[2]

    # Run previous stages first
    print("=== Stage 1: Segmentation ===")
    frames, masks, fps = segment_video(video_path)

    print("=== Stage 2: Pose Estimation ===")
    detector = load_dwpose()
    pose_images, _ = get_pose_sequence(detector, frames)

    print("=== Stage 3: Body Swap ===")
    pipe = load_bodyswap_pipeline()
    result_frames = process_video_bodyswap(
        frames, masks, pose_images,
        reference_path, pipe
    )

    save_bodyswap_preview(result_frames, "bodyswap_preview.mp4", fps)
    print("Done! Check bodyswap_preview.mp4")