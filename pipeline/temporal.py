# pipeline/temporal.py
# Stage 6: Temporal consistency using ProPainter + RAFT optical flow
# Fixes flickering between frames — the biggest quality gap vs consumer apps
# Without this, even perfect per-frame swaps look like a slideshow

import torch
import numpy as np
import cv2
from PIL import Image

def load_temporal_pipeline():
    """
    Load RAFT optical flow + ProPainter models
    Downloads once, cached to Drive after that
    """
    from huggingface_hub import snapshot_download
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Download RAFT optical flow model
    raft_path = snapshot_download(
        repo_id="Tsaiyue/RAFT",
        local_dir="./checkpoints/raft"
    )

    # Download ProPainter
    propainter_path = snapshot_download(
        repo_id="sczhou/ProPainter",
        local_dir="./checkpoints/propainter"
    )

    sys.path.append(raft_path)
    sys.path.append(propainter_path)

    # Load RAFT
    from raft import RAFT
    from utils.utils import InputPadder

    class RAFTArgs:
        model = f"{raft_path}/models/raft-things.pth"
        small = False
        mixed_precision = False
        alternate_corr = False

    raft_model = torch.nn.DataParallel(RAFT(RAFTArgs()))
    raft_model.load_state_dict(torch.load(RAFTArgs.model))
    raft_model = raft_model.module
    raft_model.to(device)
    raft_model.eval()

    # Load ProPainter
    from model.propainter import InpaintGenerator

    propainter = InpaintGenerator()
    propainter_weights = torch.load(
        f"{propainter_path}/weights/ProPainter.pth",
        map_location=device
    )
    propainter.load_state_dict(propainter_weights)
    propainter.to(device)
    propainter.eval()

    print("✓ RAFT optical flow loaded")
    print("✓ ProPainter loaded")
    return raft_model, propainter, device


def compute_optical_flow(raft_model, frame1_rgb, frame2_rgb, device):
    """
    Compute optical flow between two consecutive frames
    Flow tells us how each pixel moves from frame1 to frame2
    Used to warp appearance consistently across frames
    """
    def frame_to_tensor(frame):
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        return tensor[None].to(device)

    with torch.no_grad():
        img1 = frame_to_tensor(frame1_rgb)
        img2 = frame_to_tensor(frame2_rgb)

        # Pad to multiple of 8 (RAFT requirement)
        from utils.utils import InputPadder
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)

        # Compute flow (12 iterations = good quality/speed balance)
        _, flow = raft_model(img1, img2, iters=12, test_mode=True)
        flow = padder.unpad(flow)

    return flow[0].cpu().numpy()  # shape: (2, H, W)


def warp_frame_with_flow(frame_rgb, flow):
    """
    Warp a frame using optical flow
    Moves each pixel to where it will be in the next frame
    Creates smooth transitions between swap results
    """
    h, w = frame_rgb.shape[:2]

    # Build coordinate grid
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Add flow to coordinates
    flow_x = flow[0]  # horizontal movement
    flow_y = flow[1]  # vertical movement

    new_x = x_coords + flow_x
    new_y = y_coords + flow_y

    # Remap frame pixels to new positions
    warped = cv2.remap(
        frame_rgb,
        new_x, new_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return warped


def compute_flow_sequence(raft_model, frames, device):
    """
    Compute optical flow for all consecutive frame pairs
    Returns forward and backward flows
    """
    flows_forward = []
    flows_backward = []
    total = len(frames)

    print(f"Computing optical flow for {total - 1} frame pairs...")

    for i in range(total - 1):
        # Forward flow: frame i → frame i+1
        flow_fwd = compute_optical_flow(
            raft_model, frames[i], frames[i + 1], device
        )
        # Backward flow: frame i+1 → frame i
        flow_bwd = compute_optical_flow(
            raft_model, frames[i + 1], frames[i], device
        )

        flows_forward.append(flow_fwd)
        flows_backward.append(flow_bwd)

        if i % 10 == 0:
            print(f"  Flow {i}/{total - 1} computed")

    print("✓ Optical flow computation complete")
    return flows_forward, flows_backward


def apply_propainter(
    propainter,
    swapped_frames,
    masks,
    flows_forward,
    flows_backward,
    device,
    window_size=10
):
    """
    Run ProPainter to enforce temporal consistency
    Uses optical flow to propagate appearance across frames
    Eliminates flickering by ensuring smooth transitions

    window_size: number of frames processed together
    Larger = more consistent but slower
    """
    total = len(swapped_frames)
    result_frames = []

    print(f"Applying ProPainter consistency to {total} frames...")
    print(f"Window size: {window_size} frames")

    def to_tensor(frame):
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device)

    def to_mask_tensor(mask):
        t = torch.from_numpy(mask).float() / 255.0
        return t.unsqueeze(0).unsqueeze(0).to(device)

    def flow_to_tensor(flow):
        t = torch.from_numpy(flow).float()
        return t.unsqueeze(0).to(device)

    # Process in sliding windows for memory efficiency
    for start in range(0, total, window_size // 2):
        end = min(start + window_size, total)
        window_frames = swapped_frames[start:end]
        window_masks = masks[start:end]

        # Get flows for this window
        w_flows_fwd = flows_forward[start:end - 1]
        w_flows_bwd = flows_backward[start:end - 1]

        # Convert to tensors
        frame_tensors = torch.cat([to_tensor(f) for f in window_frames])
        mask_tensors = torch.cat([to_mask_tensor(m) for m in window_masks])

        if len(w_flows_fwd) > 0:
            fwd_tensors = torch.cat(
                [flow_to_tensor(f) for f in w_flows_fwd]
            )
            bwd_tensors = torch.cat(
                [flow_to_tensor(f) for f in w_flows_bwd]
            )
        else:
            # Single frame window edge case
            fwd_tensors = None
            bwd_tensors = None

        with torch.no_grad():
            try:
                if fwd_tensors is not None:
                    output = propainter(
                        frame_tensors,
                        mask_tensors,
                        fwd_tensors,
                        bwd_tensors
                    )
                else:
                    output = frame_tensors

                # Convert output tensors back to numpy frames
                for j in range(len(window_frames)):
                    frame_out = output[j].cpu().numpy()
                    frame_out = (
                        frame_out.transpose(1, 2, 0) * 255
                    ).astype(np.uint8)

                    # Only add non-overlapping frames to avoid duplicates
                    if start == 0 or j >= window_size // 4:
                        result_frames.append(frame_out)

            except Exception as e:
                print(f"⚠ ProPainter failed on window {start}-{end}: {e}")
                # Fall back to original frames for this window
                for j, frame in enumerate(window_frames):
                    if start == 0 or j >= window_size // 4:
                        result_frames.append(frame)

        print(f"  Window {start}-{end} processed ({len(result_frames)}/{total})")

    # Trim to exact frame count
    result_frames = result_frames[:total]

    print(f"✓ Temporal consistency complete — {len(result_frames)} frames")
    return result_frames


def simple_temporal_smooth(frames, alpha=0.6):
    """
    Lightweight fallback if ProPainter runs out of VRAM
    Simple exponential moving average between frames
    Not as good as ProPainter but much better than nothing
    Eliminates high-frequency flicker at low cost

    alpha: 0.5-0.7 recommended (lower = smoother but more blur)
    """
    print("Applying simple temporal smoothing (fallback mode)...")
    result = [frames[0]]

    for i in range(1, len(frames)):
        # Blend current frame with previous result
        smoothed = cv2.addWeighted(
            frames[i], alpha,
            result[i - 1], 1 - alpha,
            0
        )
        result.append(smoothed)

    print(f"✓ Simple smoothing complete — {len(result)} frames")
    return result


def process_video_temporal(
    swapped_frames,
    original_masks,
    raft_model,
    propainter,
    device,
    use_simple_fallback=False
):
    """
    Main function — applies temporal consistency to swapped video

    swapped_frames: frames from hair.py
    original_masks: body masks from segment.py
    use_simple_fallback: True = use simple smoothing (saves VRAM)
    """
    if use_simple_fallback or propainter is None:
        return simple_temporal_smooth(swapped_frames)

    try:
        # Full ProPainter pipeline
        print("Computing optical flow...")
        flows_fwd, flows_bwd = compute_flow_sequence(
            raft_model, swapped_frames, device
        )

        print("Applying ProPainter...")
        result = apply_propainter(
            propainter,
            swapped_frames,
            original_masks,
            flows_fwd,
            flows_bwd,
            device
        )
        return result

    except torch.cuda.OutOfMemoryError:
        print("⚠ VRAM too low for ProPainter — switching to simple smoothing")
        torch.cuda.empty_cache()
        return simple_temporal_smooth(swapped_frames)


def save_temporal_preview(
    result_frames,
    output_path="temporal_preview.mp4",
    fps=30
):
    """Save temporally consistent result video"""
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
    print(f"✓ Temporal preview saved to {output_path}")


# ── Test block ──
if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from segment import segment_video
    from pose import load_dwpose, get_pose_sequence
    from bodyswap import load_bodyswap_pipeline, process_video_bodyswap
    from hair import load_hair_pipeline, load_skin_enhancer, process_video_hair

    if len(sys.argv) < 3:
        print("Usage: python temporal.py video.mp4 reference.jpg")
        sys.exit(1)

    video_path = sys.argv[1]
    reference_path = sys.argv[2]

    print("=== Stage 1: Segmentation ===")
    frames, masks, fps = segment_video(video_path)

    print("=== Stage 2: Pose ===")
    detector = load_dwpose()
    pose_images, _ = get_pose_sequence(detector, frames)

    print("=== Stage 3: Body Swap ===")
    body_pipe = load_bodyswap_pipeline()
    body_frames = process_video_bodyswap(
        frames, masks, pose_images,
        reference_path, body_pipe
    )

    print("=== Stage 4: Hair + Skin ===")
    hair_fast, device = load_hair_pipeline()
    enhancer = load_skin_enhancer()
    hair_frames = process_video_hair(
        body_frames, reference_path,
        hair_fast, enhancer
    )

    print("=== Stage 5: Temporal Consistency ===")
    raft_model, propainter, device = load_temporal_pipeline()
    result_frames = process_video_temporal(
        hair_frames, masks,
        raft_model, propainter, device
    )

    save_temporal_preview(result_frames, "temporal_preview.mp4", fps)
    print("Done! Check temporal_preview.mp4")