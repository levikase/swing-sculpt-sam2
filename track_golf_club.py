#!/usr/bin/env python3
"""
Automatically detect and track the golf club head through a golf swing video using SAM 2.

This script automatically identifies the golf club head by analyzing motion patterns
and tracks it through the entire video, drawing the trajectory.

Usage:
    python track_golf_club.py <video_path> [--output_dir OUTPUT_DIR] [--checkpoint CHECKPOINT]
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def get_device():
    """Select the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Use bfloat16 for faster inference
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # Enable TF32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS."
        )
    else:
        device = torch.device("cpu")
    return device


def show_mask(mask, ax, obj_id=None, random_color=False):
    """Visualize a mask on a matplotlib axis."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def get_mask_center(mask):
    """Get the center point of a mask."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.squeeze()
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return None
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    return (center_x, center_y)


def get_mask_bbox(mask):
    """Get bounding box of a mask."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.squeeze()
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0:
        return None
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
    return (x_min, y_min, x_max, y_max)


def detect_fast_moving_object(video_path, num_frames_to_analyze=10):
    """
    Detect the golf club head by finding the fastest-moving object in early frames.
    
    Returns the center point and bounding box of the detected object.
    """
    print("Analyzing motion to detect golf club head...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Read first few frames
    frames = []
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read video frames")
    
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    frame_count = 1
    
    while frame_count < num_frames_to_analyze and ret:
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            frame_count += 1
    
    cap.release()
    
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for motion analysis")
    
    # Calculate optical flow to find moving objects
    # Use Farneback optical flow for dense motion estimation
    flow = cv2.calcOpticalFlowFarneback(
        frames[0], frames[-1], None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Calculate magnitude of motion
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    
    # Threshold to find fast-moving regions
    # Golf club head should be one of the fastest-moving objects
    motion_threshold = np.percentile(magnitude, 95)  # Top 5% of motion
    
    # Create binary mask of fast-moving regions
    motion_mask = (magnitude > motion_threshold).astype(np.uint8)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        motion_mask, connectivity=8
    )
    
    if num_labels < 2:  # Only background
        print("Warning: Could not detect fast-moving objects. Using center of frame.")
        h, w = frames[0].shape
        return (w // 2, h // 2), (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    
    # Find the component with highest average motion magnitude
    best_component = None
    best_motion_score = 0
    
    for i in range(1, num_labels):  # Skip background (label 0)
        component_mask = (labels == i).astype(np.uint8)
        component_area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter by size - golf club head should be relatively small
        # Typically 0.1% to 2% of frame area
        h, w = frames[0].shape
        frame_area = h * w
        min_area = frame_area * 0.001  # 0.1%
        max_area = frame_area * 0.02  # 2%
        
        if component_area < min_area or component_area > max_area:
            continue
        
        # Calculate average motion magnitude in this component
        avg_motion = np.mean(magnitude[component_mask > 0])
        
        if avg_motion > best_motion_score:
            best_motion_score = avg_motion
            best_component = i
    
    if best_component is None:
        print("Warning: No suitable fast-moving object found. Using largest motion region.")
        # Use the largest component
        component_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        best_component = np.argmax(component_areas) + 1
    
    # Get center and bounding box of best component
    center = centroids[best_component]
    bbox = (
        stats[best_component, cv2.CC_STAT_LEFT],
        stats[best_component, cv2.CC_STAT_TOP],
        stats[best_component, cv2.CC_STAT_LEFT] + stats[best_component, cv2.CC_STAT_WIDTH],
        stats[best_component, cv2.CC_STAT_TOP] + stats[best_component, cv2.CC_STAT_HEIGHT],
    )
    
    print(f"Detected fast-moving object at center: ({int(center[0])}, {int(center[1])})")
    print(f"Bounding box: {bbox}")
    
    return (int(center[0]), int(center[1])), bbox


def detect_club_head_with_sam(video_path, image_predictor, num_frames_to_analyze=5):
    """
    Alternative method: Use SAM 2's automatic mask generation on early frames
    and select the object with highest motion.
    """
    print("Using SAM 2 to detect golf club head...")
    
    cap = cv2.VideoCapture(video_path)
    frames_rgb = []
    
    for _ in range(num_frames_to_analyze):
        ret, frame = cap.read()
        if not ret:
            break
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    
    if len(frames_rgb) < 2:
        raise ValueError("Need at least 2 frames")
    
    # Use first frame for detection
    first_frame = frames_rgb[0]
    h, w = first_frame.shape[:2]
    
    # Set image for predictor
    image_predictor.set_image(first_frame)
    
    # Generate masks using a grid of points
    # Sample points across the image
    points_per_side = 20
    step = max(h, w) // points_per_side
    
    best_mask = None
    best_motion = 0
    best_center = None
    best_bbox = None
    
    print("Scanning frame for objects...")
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            points = np.array([[x, y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            
            masks, scores, _ = image_predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
            
            # Use the best mask for this point
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            
            # Check if this mask represents a fast-moving object
            # by comparing with second frame
            if len(frames_rgb) > 1:
                # Get center of mask
                center = get_mask_center(mask)
                if center is None:
                    continue
                
                # Check motion at this location in second frame
                cx, cy = center
                # Sample a small region around center
                y1, y2 = max(0, cy - 10), min(h, cy + 10)
                x1, x2 = max(0, cx - 10), min(w, cx + 10)
                
                if y2 > y1 and x2 > x1:
                    region1 = frames_rgb[0][y1:y2, x1:x2].astype(np.float32)
                    region2 = frames_rgb[1][y1:y2, x1:x2].astype(np.float32)
                    
                    # Calculate motion (frame difference)
                    motion = np.mean(np.abs(region1 - region2))
                    
                    # Filter by size
                    mask_area = np.sum(mask > 0)
                    frame_area = h * w
                    min_area = frame_area * 0.001
                    max_area = frame_area * 0.02
                    if min_area <= mask_area <= max_area:
                        if motion > best_motion:
                            best_motion = motion
                            best_mask = mask
                            best_center = center
                            best_bbox = get_mask_bbox(mask)
    
    if best_center is None:
        # Fallback to motion detection
        print("SAM detection failed, falling back to motion detection...")
        return detect_fast_moving_object(video_path, num_frames_to_analyze)
    
    print(f"Detected golf club head at: {best_center}")
    return best_center, best_bbox


def extract_video_frames(video_path, output_dir=None):
    """
    Extract video frames to a directory of JPEG files.
    This allows SAM 2 to load videos without requiring decord.
    """
    import tempfile
    
    if output_dir is None:
        # Create temporary directory
        output_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        is_temp = True
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_dir = str(output_dir)
        is_temp = False
    
    output_path = Path(output_dir)
    print(f"Extracting frames from video to: {output_dir}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        print(f"Video reports ~{total_frames} frames (may be approximate for some codecs).")

    frame_count = 0
    last_log_time = time.time()
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Extracting frames", unit="frame")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil.save(output_path / f"{frame_count:05d}.jpg", quality=95)
            frame_count += 1
            pbar.update(1)

            # Extra periodic log in case tqdm isn't rendering in the user's terminal.
            now = time.time()
            if now - last_log_time > 5:
                if total_frames > 0:
                    pct = 100.0 * frame_count / max(total_frames, 1)
                    print(f"  extracted {frame_count}/{total_frames} frames ({pct:.1f}%)...")
                else:
                    print(f"  extracted {frame_count} frames...")
                last_log_time = now
    finally:
        pbar.close()
    
    cap.release()
    print(f"Extracted {frame_count} frames")
    
    return output_dir, is_temp


def track_golf_club(
    video_path,
    checkpoint_path=None,
    model_cfg=None,
    output_dir=None,
    device=None,
    use_sam_detection=False,
    async_loading_frames=False,
):
    """Automatically detect and track the golf club head through a video."""
    import tempfile
    import shutil
    
    if device is None:
        device = get_device()

    print(f"Using device: {device}")

    # Extract frames if needed (to avoid decord requirement)
    # Check if video_path is already a directory
    if os.path.isdir(video_path):
        frames_dir = video_path
        is_temp_frames = False
        original_video_path = None
    else:
        # Extract frames to temporary directory
        frames_dir, is_temp_frames = extract_video_frames(video_path)
        original_video_path = video_path
    
    # Set default checkpoint and config
    if checkpoint_path is None:
        checkpoint_path = "checkpoints/sam2.1_hiera_large.pt"
    if not os.path.exists(checkpoint_path):
        # Try other checkpoints
        for ckpt in [
            "checkpoints/sam2.1_hiera_base_plus.pt",
            "checkpoints/sam2.1_hiera_small.pt",
            "checkpoints/sam2.1_hiera_tiny.pt",
        ]:
            if os.path.exists(ckpt):
                checkpoint_path = ckpt
                break
        else:
            raise FileNotFoundError(
                f"Could not find SAM 2 checkpoint. Please download one to the checkpoints/ directory.\n"
                f"Run: cd checkpoints && bash download_ckpts.sh"
            )

    if model_cfg is None:
        # Infer config from checkpoint name
        if "tiny" in checkpoint_path:
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif "small" in checkpoint_path:
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "base_plus" in checkpoint_path or "b+" in checkpoint_path.lower():
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:  # default to large
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print(f"Loading model from {checkpoint_path} with config {model_cfg}")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)

    # Auto-detect golf club head.
    # Use the original video file when available (OpenCV supports more formats than SAM2's mp4 loader),
    # otherwise fall back to the frames directory.
    detection_video = original_video_path if original_video_path else frames_dir
    
    if use_sam_detection:
        # Create image predictor for detection
        image_predictor = SAM2ImagePredictor(predictor.sam_model)
        center, bbox = detect_club_head_with_sam(detection_video, image_predictor)
    else:
        center, bbox = detect_fast_moving_object(detection_video)

    # Initialize inference state (use frames directory)
    print(f"\nLoading video frames from: {frames_dir}")
    print(
        "Initializing SAM2 inference state (frame loading + first-frame feature warmup)... "
        "this can take a while on MPS."
    )
    t_init0 = time.time()
    inference_state = predictor.init_state(
        video_path=frames_dir, async_loading_frames=async_loading_frames
    )
    print(f"Init state done in {time.time() - t_init0:.1f}s")
    num_frames = inference_state["num_frames"]
    print(f"Video has {num_frames} frames")

    # Add prompt to first frame using detected center point
    ann_frame_idx = 0
    ann_obj_id = 1

    print(f"Using detected point: {center}")
    points = np.array([center], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)  # positive click
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Propagate through video
    print("\nTracking golf club head through video...")
    video_segments = {}
    trajectory = []  # Store center points for trajectory

    last_log_time = time.time()
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        mask = (out_mask_logits[0] > 0.0).cpu().numpy()
        video_segments[out_frame_idx] = {
            out_obj_id: mask for out_obj_id in out_obj_ids
        }
        
        # Calculate center point for trajectory
        center = get_mask_center(mask)
        if center is not None:
            trajectory.append((out_frame_idx, center[0], center[1]))

        # Extra periodic log in case tqdm isn't rendering (sam2 uses tqdm internally).
        now = time.time()
        if now - last_log_time > 5:
            pct = 100.0 * (out_frame_idx + 1) / max(num_frames, 1)
            print(f"  tracking: frame {out_frame_idx+1}/{num_frames} ({pct:.1f}%)...")
            last_log_time = now

    print(f"Tracking complete! Tracked {len(trajectory)} frames")

    # Save results
    if output_dir is None:
        output_dir = Path(video_path).stem + "_tracking_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save trajectory as CSV
    trajectory_file = output_dir / "trajectory.csv"
    with open(trajectory_file, "w") as f:
        f.write("frame,x,y\n")
        for frame_idx, x, y in trajectory:
            f.write(f"{frame_idx},{x},{y}\n")
    print(f"Saved trajectory to {trajectory_file}")

    # Create visualization with trajectory overlay
    print("Creating visualization...")
    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(exist_ok=True)

    # Load video frames for visualization
    # If the input was a folder of frames, we can't open it with OpenCV.
    # In that case, we'll read the corresponding extracted JPEGs instead.
    cap = None
    if original_video_path is not None:
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video for visualization: {original_video_path}")
            cap = None
    
    # Visualize every Nth frame (adjust stride as needed)
    stride = max(1, num_frames // 50)  # Show ~50 frames

    vis_pbar = tqdm(range(0, num_frames, stride), desc="Rendering visualization", unit="frame")
    for frame_idx in vis_pbar:
        if cap is not None:
            # Seek to the desired frame index (important when stride > 1).
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_path = Path(frames_dir) / f"{frame_idx:05d}.jpg"
            if not frame_path.exists():
                break
            frame_rgb = np.array(Image.open(frame_path).convert("RGB"))
        h, w = frame_rgb.shape[:2]

        if frame_idx in video_segments:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(frame_rgb)
            
            # Draw mask
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                # Resize mask to match frame size
                mask_resized = cv2.resize(
                    out_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
                show_mask(mask_resized, ax, obj_id=out_obj_id)

            # Draw trajectory up to this frame
            if trajectory:
                traj_points = [
                    (x, y)
                    for f, x, y in trajectory
                    if f <= frame_idx
                ]
                if len(traj_points) > 1:
                    traj_x, traj_y = zip(*traj_points)
                    ax.plot(traj_x, traj_y, "r-", linewidth=3, alpha=0.8, label="Trajectory", zorder=10)
                    # Draw current position
                    if traj_points:
                        ax.scatter(traj_x[-1], traj_y[-1], c="red", s=150, marker="o", 
                                 edgecolors="white", linewidths=2, zorder=11, label="Current Position")

            ax.set_title(f"Frame {frame_idx} - Golf Club Head Tracking", fontsize=14, fontweight="bold")
            ax.axis("off")
            if trajectory and len(traj_points) > 1:
                ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(vis_dir / f"frame_{frame_idx:05d}.png", dpi=150, bbox_inches="tight")
            plt.close()

    if cap is not None:
        cap.release()
    print(f"Saved visualization frames to {vis_dir}")

    # Create trajectory plot
    if trajectory:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        traj_x, traj_y = zip(*[(x, y) for _, x, y in trajectory])
        
        # Color trajectory by time (gradient)
        points = ax.scatter(traj_x, traj_y, c=range(len(traj_x)), cmap="viridis", 
                           s=50, alpha=0.6, zorder=5)
        ax.plot(traj_x, traj_y, "b-", linewidth=2, alpha=0.5, zorder=4)
        
        # Mark start and end
        ax.scatter(traj_x[0], traj_y[0], c="green", s=300, marker="o", 
                  edgecolors="white", linewidths=3, label="Start", zorder=6)
        ax.scatter(traj_x[-1], traj_y[-1], c="red", s=300, marker="o", 
                  edgecolors="white", linewidths=3, label="End", zorder=6)
        
        ax.set_xlabel("X Position (pixels)", fontsize=12)
        ax.set_ylabel("Y Position (pixels)", fontsize=12)
        ax.set_title("Golf Club Head Trajectory", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Invert Y axis to match image coordinates
        plt.colorbar(points, ax=ax, label="Frame Number")
        plt.tight_layout()
        plt.savefig(output_dir / "trajectory_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved trajectory plot to {output_dir / 'trajectory_plot.png'}")

    print(f"\n✓ All results saved to: {output_dir}")
    
    # Clean up temporary frames directory if we created it
    if is_temp_frames:
        print(f"Cleaning up temporary frames directory...")
        shutil.rmtree(frames_dir)
    
    return video_segments, trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Automatically detect and track golf club head through a video using SAM 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video_path", type=str, help="Path to the golf swing video")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SAM 2 checkpoint (default: auto-detect from checkpoints/)",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default=None,
        help="Path to model config file (default: auto-detect from checkpoint)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: <video_name>_tracking_results)",
    )
    parser.add_argument(
        "--use_sam_detection",
        action="store_true",
        help="Use SAM 2 automatic mask generation for detection (slower but more accurate)",
    )
    parser.add_argument(
        "--async_loading_frames",
        action="store_true",
        help="Load JPEG frames asynchronously to get to tracking sooner (useful on slow disks / MPS).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    try:
        track_golf_club(
            video_path=args.video_path,
            checkpoint_path=args.checkpoint,
            model_cfg=args.model_cfg,
            output_dir=args.output_dir,
            use_sam_detection=args.use_sam_detection,
            async_loading_frames=args.async_loading_frames,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
