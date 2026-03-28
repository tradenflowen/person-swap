# pipeline/segment.py
# Stage 1: Body segmentation using SAM 2
# Takes a video frame, returns a mask of the person

import torch
import numpy as np
import cv2
from PIL import Image

def load_sam2():
    """Load SAM 2 model — downloads once, cached after that"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Use the smaller/faster model first for testing
    model = build_sam2(
        "sam2_hiera_small.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    )
    predictor = SAM2ImagePredictor(model)
    print("✓ SAM 2 loaded")
    return predictor


def get_person_mask(predictor, frame_rgb):
    """
    Given a single RGB frame (numpy array),
    returns a binary mask of the person (255 = person, 0 = background)
    """
    predictor.set_image(frame_rgb)

    # Auto-point in the center of the frame as starting hint
    h, w = frame_rgb.shape[:2]
    center_point = np.array([[w // 2, h // 2]])
    center_label = np.array([1])  # 1 = foreground

    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True
    )

    # Pick the mask with highest confidence score
    best_mask = masks[np.argmax(scores)]

    # Convert boolean mask to uint8 (0 or 255)
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    return mask_uint8


def extract_frames(video_path, max_frames=50):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {total} frames at {fps:.1f} fps")

    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        count += 1

    cap.release()
    print(f"✓ Extracted {len(frames)} frames")
    return frames, fps


def segment_video(video_path):
    """
    Main function — takes video path, returns list of masks per frame
    """
    print("Loading SAM 2...")
    predictor = load_sam2()

    print("Extracting frames...")
    frames, fps = extract_frames(video_path)

    print("Generating masks...")
    masks = []
    for i, frame in enumerate(frames):
        mask = get_person_mask(predictor, frame)
        masks.append(mask)
        if i % 10 == 0:
            print(f"  Frame {i}/{len(frames)} done")

    print(f"✓ Segmentation complete — {len(masks)} masks generated")
    return frames, masks, fps


def save_masked_preview(frames, masks, output_path="preview.mp4", fps=30):
    """Save a preview video showing the mask overlay — for testing"""
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    for frame, mask in zip(frames, masks):
        # Green overlay on detected person area
        overlay = frame.copy()
        overlay[mask > 0] = (overlay[mask > 0] * 0.7 + 
                             np.array([0, 255, 0]) * 0.3).astype(np.uint8)
        bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        out.write(bgr)

    out.release()
    print(f"✓ Preview saved to {output_path}")


# ── Test block — only runs when you execute this file directly ──
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python segment.py path/to/video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    frames, masks, fps = segment_video(video_path)
    save_masked_preview(frames, masks, "segmentation_preview.mp4", fps)
    print("Done! Check segmentation_preview.mp4")