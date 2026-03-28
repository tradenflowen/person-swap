# pipeline/faceswap.py
# Stage 3: Face identity swap using InstantID + IP-Adapter
# Takes source frame + target person reference image
# Returns frame with target person's face identity applied

import torch
import numpy as np
import cv2
from PIL import Image

def load_faceswap_pipeline():
    """
    Load InstantID + IP-Adapter on SDXL
    Downloads once, cached to Drive after that
    """
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import load_image
    import insightface
    from insightface.app import FaceAnalysis
    from huggingface_hub import hf_hub_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load face analyzer — detects and embeds face identity
    face_analyzer = FaceAnalysis(
        name="antelopev2",
        root="./checkpoints",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    # Download InstantID face adapter weights
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir="./checkpoints/instantid"
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/config.json",
        local_dir="./checkpoints/instantid"
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir="./checkpoints/instantid"
    )

    # Load SDXL base pipeline
    from diffusers import ControlNetModel
    from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

    controlnet = ControlNetModel.from_pretrained(
        "./checkpoints/instantid/ControlNetModel",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid("./checkpoints/instantid/ip-adapter.bin")

    print("✓ InstantID pipeline loaded")
    return face_analyzer, pipe


def extract_face_embedding(face_analyzer, image_rgb):
    """
    Extract the identity embedding from a reference face image
    This embedding captures WHO the person is — used to transfer identity
    """
    faces = face_analyzer.get(image_rgb)

    if len(faces) == 0:
        print("⚠ No face detected in reference image")
        return None

    # Use the largest detected face
    face = sorted(faces, key=lambda x: x.bbox[2] - x.bbox[0])[-1]
    print(f"✓ Face detected — confidence: {face.det_score:.2f}")
    return face


def get_face_keypoints(face):
    """Extract facial landmark keypoints from detected face"""
    if face is None:
        return None
    from insightface.utils import face_align
    return face.kps


def swap_face_in_frame(
    face_analyzer,
    pipe,
    source_frame_rgb,
    target_face_embedding,
    target_kps,
    strength=0.7,
    guidance_scale=5.0
):
    """
    Apply target identity onto a single source frame
    
    strength: how strongly to apply the identity (0.5-0.9 recommended)
    guidance_scale: how closely to follow the identity (5-7 recommended)
    """
    source_pil = Image.fromarray(source_frame_rgb)

    # Detect face in source frame to get position
    source_faces = face_analyzer.get(source_frame_rgb)
    if len(source_faces) == 0:
        print("⚠ No face in source frame — skipping swap")
        return source_frame_rgb

    source_face = sorted(
        source_faces,
        key=lambda x: x.bbox[2] - x.bbox[0]
    )[-1]

    # Run InstantID — applies target identity to source pose/position
    result = pipe(
        prompt="a person, photorealistic, high quality, natural lighting",
        negative_prompt="blurry, distorted, cartoon, painting, low quality",
        image_embeds=target_face_embedding.normed_embedding,
        image=source_pil,
        controlnet_conditioning_scale=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=20,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]

    return np.array(result)


def blend_face_back(
    original_frame,
    swapped_frame,
    face,
    blend_radius=15
):
    """
    Blend the swapped face back into the original frame
    Uses a soft mask around the face boundary for seamless blending
    """
    h, w = original_frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Get face bounding box
    x1, y1, x2, y2 = [int(v) for v in face.bbox]

    # Add padding around face
    pad = 30
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    # Draw filled ellipse for natural face shape mask
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)

    # Soften mask edges for smooth blending
    mask = cv2.GaussianBlur(mask, (blend_radius * 2 + 1, blend_radius * 2 + 1), blend_radius)
    mask_3ch = mask[:, :, np.newaxis] / 255.0

    # Blend original and swapped frames using the mask
    blended = (
        swapped_frame * mask_3ch +
        original_frame * (1 - mask_3ch)
    ).astype(np.uint8)

    return blended


def process_video_faceswap(
    frames,
    reference_image_path,
    face_analyzer,
    pipe
):
    """
    Main function — applies face swap across all video frames
    
    frames: list of RGB numpy arrays from segment.py
    reference_image_path: path to target person reference photo
    """
    # Load and analyze reference face
    print("Analyzing reference face...")
    ref_img = np.array(Image.open(reference_image_path).convert("RGB"))
    target_face = extract_face_embedding(face_analyzer, ref_img)

    if target_face is None:
        raise ValueError("Could not detect face in reference image. Use a clear front-facing photo.")

    target_kps = get_face_keypoints(target_face)

    # Process each frame
    print(f"Swapping faces across {len(frames)} frames...")
    result_frames = []

    for i, frame in enumerate(frames):
        swapped = swap_face_in_frame(
            face_analyzer, pipe,
            frame,
            target_face,
            target_kps
        )
        blended = blend_face_back(frame, swapped, target_face)
        result_frames.append(blended)

        if i % 5 == 0:
            print(f"  Frame {i}/{len(frames)} done")

    print(f"✓ Face swap complete — {len(result_frames)} frames processed")
    return result_frames


# ── Test block ──
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from segment import extract_frames, save_masked_preview

    if len(sys.argv) < 3:
        print("Usage: python faceswap.py video.mp4 reference_face.jpg")
        sys.exit(1)

    video_path = sys.argv[1]
    reference_path = sys.argv[2]

    frames, fps = extract_frames(video_path, max_frames=20)
    face_analyzer, pipe = load_faceswap_pipeline()
    result_frames = process_video_faceswap(
        frames, reference_path,
        face_analyzer, pipe
    )

    # Save result
    h, w = result_frames[0].shape[:2]
    out = cv2.VideoWriter(
        "faceswap_preview.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )
    for frame in result_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print("Done! Check faceswap_preview.mp4")