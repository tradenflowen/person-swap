# pipeline/pose.py
# Stage 2: Body pose estimation using DWPose
# Takes a frame, returns 133 body keypoints (full body including hands and face)

import torch
import numpy as np
import cv2
from PIL import Image

def load_dwpose():
    """Load DWPose model — downloads once, cached after that"""
    from controlnet_aux import DWposeDetector

    detector = DWposeDetector()
    detector = detector.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("✓ DWPose loaded")
    return detector


def get_pose_keypoints(detector, frame_rgb):
    """
    Given a single RGB frame (numpy array),
    returns:
    - pose_image: visual skeleton drawn on black background
    - keypoints: raw keypoint coordinates dict
    """
    pil_image = Image.fromarray(frame_rgb)

    # DWPose returns a rendered skeleton image + keypoints
    pose_image, keypoints = detector(
        pil_image,
        detect_resolution=512,
        image_resolution=frame_rgb.shape[1],
        output_type="both"
    )

    return np.array(pose_image), keypoints


def get_pose_sequence(detector, frames):
    """
    Process a list of frames and return pose for each frame
    Used to track body motion across video
    """
    pose_images = []
    keypoints_sequence = []

    print("Estimating poses...")
    for i, frame in enumerate(frames):
        pose_img, kpts = get_pose_keypoints(detector, frame)
        pose_images.append(pose_img)
        keypoints_sequence.append(kpts)

        if i % 10 == 0:
            print(f"  Frame {i}/{len(frames)} done")

    print(f"✓ Pose estimation complete — {len(pose_images)} frames processed")
    return pose_images, keypoints_sequence


def save_pose_preview(pose_images, output_path="pose_preview.mp4", fps=30):
    """Save skeleton visualization video — for testing"""
    h, w = pose_images[0].shape[:2]
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    for pose_img in pose_images:
        bgr = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
        out.write(bgr)

    out.release()
    print(f"✓ Pose preview saved to {output_path}")


def get_body_bbox(keypoints):
    """
    Extract bounding box of the full body from keypoints
    Useful for cropping the person region precisely
    """
    all_points = []

    # Collect all valid keypoint coordinates
    if keypoints and "bodies" in keypoints:
        for body in keypoints["bodies"]["candidate"]:
            if body[0] > 0 and body[1] > 0:  # skip missing points
                all_points.append(body[:2])

    if not all_points:
        return None

    all_points = np.array(all_points)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    # Add padding around the body
    padding = 20
    return {
        "x1": max(0, int(x_min - padding)),
        "y1": max(0, int(y_min - padding)),
        "x2": int(x_max + padding),
        "y2": int(y_max + padding)
    }


def check_pose_quality(keypoints):
    """
    Returns True if pose is clear enough to process
    Rejects frames where person is too occluded or far away
    """
    if not keypoints or "bodies" not in keypoints:
        return False

    candidates = keypoints["bodies"]["candidate"]
    valid_points = sum(1 for p in candidates if p[0] > 0 and p[1] > 0)

    # Need at least 10 visible keypoints for reliable swap
    return valid_points >= 10


# ── Test block ──
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pose.py path/to/video.mp4")
        sys.exit(1)

    # Import extract_frames from segment.py
    sys.path.append(".")
    from segment import extract_frames

    video_path = sys.argv[1]
    frames, fps = extract_frames(video_path)

    detector = load_dwpose()
    pose_images, keypoints_seq = get_pose_sequence(detector, frames)
    save_pose_preview(pose_images, "pose_preview.mp4", fps)

    # Print quality check on first frame
    quality = check_pose_quality(keypoints_seq[0])
    print(f"Pose quality check on frame 0: {'✓ Good' if quality else '✗ Poor'}")
    print("Done! Check pose_preview.mp4")