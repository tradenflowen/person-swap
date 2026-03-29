# pipeline/pose.py
# Stage 2: Body pose estimation using MediaPipe
# Updated to use new MediaPipe Tasks API (0.10.x compatible)

import cv2
import numpy as np
import urllib.request
import os
from PIL import Image

def load_dwpose():
    """
    Load MediaPipe Pose — replaces DWPose for Python 3.12 compatibility
    Downloads model once, reuses after that
    """
    import mediapipe as mp

    model_path = "./checkpoints/pose_landmarker.task"
    os.makedirs("./checkpoints", exist_ok=True)

    if not os.path.exists(model_path):
        print("Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
            model_path
        )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1
    )

    detector = PoseLandmarker.create_from_options(options)
    print("✓ MediaPipe Pose loaded")
    return detector


def get_pose_keypoints(detector, frame_rgb):
    """
    Given a single RGB frame returns pose image and keypoints
    """
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    h, w = frame_rgb.shape[:2]
    skeleton = np.zeros((h, w, 3), dtype=np.uint8)

    if not result.pose_landmarks:
        return skeleton, None

    keypoints = result.pose_landmarks[0]

    # Draw landmarks
    for landmark in keypoints:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(skeleton, (x, y), 6, (0, 255, 0), -1)

    # Draw connections
    connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
    for conn in connections:
        start = keypoints[conn.start]
        end = keypoints[conn.end]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(skeleton, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return skeleton, keypoints


def get_pose_sequence(detector, frames):
    """Process list of frames and return pose for each"""
    pose_images = []
    keypoints_sequence = []

    print(f"Estimating poses for {len(frames)} frames...")
    for i, frame in enumerate(frames):
        pose_img, kpts = get_pose_keypoints(detector, frame)
        pose_images.append(pose_img)
        keypoints_sequence.append(kpts)
        if i % 10 == 0:
            print(f"  Frame {i}/{len(frames)} done")

    print(f"✓ Pose estimation complete")
    return pose_images, keypoints_sequence


def get_body_bbox(keypoints):
    """Extract bounding box from keypoints"""
    if keypoints is None:
        return None

    import mediapipe as mp
    h, w = 1080, 1920  # default, overridden when frame available

    valid = [(kp.x, kp.y) for kp in keypoints if kp.visibility > 0.5]
    if not valid:
        return None

    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]

    return {
        "x1": int(min(xs) * w),
        "y1": int(min(ys) * h),
        "x2": int(max(xs) * w),
        "y2": int(max(ys) * h)
    }


def check_pose_quality(keypoints):
    """Returns True if pose is clear enough"""
    if keypoints is None:
        return False
    visible = sum(1 for kp in keypoints if kp.visibility > 0.5)
    return visible >= 10


def save_pose_preview(pose_images, output_path="pose_preview.mp4", fps=30):
    """Save skeleton visualization video"""
    h, w = pose_images[0].shape[:2]
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )
    for pose_img in pose_images:
        bgr = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()
    print(f"✓ Pose preview saved to {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from segment import extract_frames

    if len(sys.argv) < 2:
        print("Usage: python pose.py path/to/video.mp4")
        sys.exit(1)

    frames, fps = extract_frames(sys.argv[1])
    detector = load_dwpose()
    pose_images, keypoints_seq = get_pose_sequence(detector, frames)
    save_pose_preview(pose_images, "pose_preview.mp4", fps)
    quality = check_pose_quality(keypoints_seq[0])
    print(f"Quality check frame 0: {'✓ Good' if quality else '✗ Poor'}")