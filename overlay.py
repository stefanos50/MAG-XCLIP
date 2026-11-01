import os
import math
from pathlib import Path
import torch
import torch.nn as nn
from transformers import XCLIPProcessor, XCLIPModel
import cv2
import numpy as np
from PIL import Image
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
import argparse

parser = argparse.ArgumentParser(description="Save videos with violence probability overlays (no evaluation).")
parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--normal-dir", default="./movies_dataset/noFights")
parser.add_argument("--violence-dir", default="./movies_dataset/fights")
parser.add_argument("--model-path", default="./checkpoints/movies.pth")
parser.add_argument("--window-size", type=int, choices=[8], default=8,
                    help="This X-CLIP checkpoint uses 8-frame clips.")
parser.add_argument("--input-dir", default=None,
                    help="Process this directory instead of the two dataset directories.")
parser.add_argument("--output-dir", default="./annotated_videos")
parser.add_argument("--border-width", type=int, default=10)
parser.add_argument("--fallback-fps", type=float, default=25.0)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
if args.border_width < 1:
    parser.error("--border-width must be positive")
if not math.isfinite(args.fallback_fps) or args.fallback_fps <= 0:
    parser.error("--fallback-fps must be finite and positive")

device = args.device
normal_dir = args.normal_dir
violence_dir = args.violence_dir
model_path = args.model_path
window_size = args.window_size

print(f"Using device: {device}")

# -----------------------------
# 1. Optical Flow
# -----------------------------
flow_model = ptlflow.get_model('neuflow2', ckpt_path='sintel').to(device)
flow_model.eval()

@torch.no_grad()
def run_flow(frames):
    """Return normalized tensor [2,H,W] optical flow"""
    io_adapter = IOAdapter(flow_model, frames[0].shape[:2])
    inputs = io_adapter.prepare_inputs(frames)
    inputs['images'] = inputs['images'].to(device)
    preds = flow_model(inputs)
    flow = preds['flows'][0, 0].detach().cpu()
    flow = flow / (flow.abs().max() + 1e-6)  # normalize
    return flow

# -----------------------------
# 2. X-CLIP (frozen)
# -----------------------------
processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32").to(device)
xclip_model.eval()
for p in xclip_model.parameters():
    p.requires_grad = False

@torch.no_grad()
def extract_clip_features(frames):
    """Extract [D] feature vector for 8 RGB PIL frames"""
    inputs = processor(videos=frames, return_tensors="pt").to(device)
    feats = xclip_model.get_video_features(**inputs)
    return feats.squeeze(0).cpu()

# -----------------------------
# 3. Motion-aware Fusion Head
# -----------------------------
def compute_flow_stats(flows, motion_thresh=0.1):
    """
    flows: torch.Tensor [T,2,H,W] (CPU or GPU)
    Returns: torch.Tensor shape [7] ->
        [mean_mag, std_mag, max_mag, coverage_ratio,
         temporal_consistency, directional_consistency, global_motion_ratio]
    
    Metrics:
      - mean_mag: average motion magnitude
      - std_mag: std of magnitudes
      - max_mag: max motion magnitude
      - coverage_ratio: fraction of pixels with motion > motion_thresh
      - temporal_consistency: smoothness of motion magnitude over time (lower = more stable)
      - directional_consistency: variation of motion direction (lower = more consistent)
      - global_motion_ratio: strength of global (camera-like) motion vs. residual (scene) motion
    """
    flows = flows.float()  # [T,2,H,W]
    T, _, H, W = flows.shape

    # --- Basic motion stats ---
    magnitude = torch.sqrt(flows[:, 0]**2 + flows[:, 1]**2)  # [T,H,W]
    mean_mag = magnitude.mean()
    std_mag = magnitude.std()
    max_mag = magnitude.max()
    motion_mask = (magnitude > motion_thresh).float()
    motion_coverage = motion_mask.mean()  # [scalar]

    # --- Temporal consistency (option 2) ---
    # Mean absolute change between consecutive frames
    if T > 1:
        diff = torch.abs(magnitude[1:] - magnitude[:-1])  # [T-1,H,W]
        temporal_consistency = 1.0 / (diff.mean() + 1e-6)  # higher = more consistent
    else:
        temporal_consistency = torch.tensor(0.0, device=flows.device)

    # --- Directional consistency (option 3) ---
    # Normalize flow vectors to unit direction
    norm = torch.norm(flows, dim=1, keepdim=True) + 1e-6
    unit_flow = flows / norm  # [T,2,H,W]
    dir_var = unit_flow.var(dim=0).mean()  # variance over time
    directional_consistency = 1.0 / (dir_var + 1e-6)  # higher = more consistent

    # --- Global motion ratio (option 4) ---
    mean_flow_vec = flows.mean(dim=[2, 3])  # [T,2]
    residuals = flows - mean_flow_vec.view(T, 2, 1, 1)
    global_motion_ratio = mean_flow_vec.norm(dim=1).mean() / (residuals.norm(dim=1).mean() + 1e-6)

    # --- Combine all ---
    stats = torch.stack([
        mean_mag,
        std_mag,
        max_mag,
        motion_coverage,
        temporal_consistency,
        directional_consistency,
        global_motion_ratio
    ])
    return stats


def compute_motion_energy_map(flows, out_size=(8, 8)):
    """
    flows: torch.Tensor [T,2,H,W]
    Returns: torch.Tensor flattened small map, shape [out_size[0]*out_size[1]]
    """
    flows = flows.float()
    magnitude = torch.sqrt(flows[:, 0]**2 + flows[:, 1]**2)  # [T,H,W]
    motion_energy = magnitude.mean(dim=0, keepdim=True)  # [1,H,W]
    motion_energy = motion_energy.unsqueeze(0)  # [1,1,H,W]
    motion_energy_small = nn.functional.adaptive_avg_pool2d(motion_energy, out_size)  # [1,1,h,w]
    motion_energy_flat = motion_energy_small.view(-1)  # [(h*w)]
    return motion_energy_flat  # 1D tensor


# ----------------- New motion-aware fusion head -----------------
class FlowFusionHead(nn.Module):
    def __init__(self, clip_feat_dim=512, flow_dim=2, num_classes=2,
                 motion_map_size=(8, 8), motion_proj_dim=32):
        super().__init__()
        self.clip_feat_dim = clip_feat_dim
        self.motion_map_size = motion_map_size
        motion_map_elems = motion_map_size[0] * motion_map_size[1]
        self.motion_vec_dim = 7 + motion_map_elems  # 4 stats + flattened map

        # project flow (spatial) to clip feature dim
        self.flow_proj = nn.Sequential(
            nn.Conv2d(flow_dim, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, clip_feat_dim)
        )

        # project motion vec to compact context
        self.motion_proj = nn.Sequential(
            nn.Linear(self.motion_vec_dim, motion_proj_dim),
            nn.ReLU()
        )

        # gate takes [clip_feat | flow_feat | motion_context]
        gate_in = clip_feat_dim + clip_feat_dim + motion_proj_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, clip_feat_dim),
            #nn.ReLU(), #comment
            #nn.Linear(clip_feat_dim, clip_feat_dim), #comment
            nn.Sigmoid()
        )

        # final classifier takes fused clip-sized representation
        self.classifier = nn.Sequential(
            nn.Linear(clip_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, clip_feats, flows):
        """
        clip_feats: [B, D] tensor (on same device as returned)
        flows: [B, T, 2, H, W] tensor (might be cpu or cuda depending on pipeline)
        """
        device = clip_feats.device
        B = flows.size(0)
        flow_feats = []
        motion_contexts = []

        for i in range(B):
            # mean flow across time for spatial projection
            mean_flow = flows[i].mean(dim=0, keepdim=True).to(device)  # [1,2,H,W]
            flow_feat = self.flow_proj(mean_flow)  # [1,clip_feat_dim]
            flow_feat = flow_feat.squeeze(0)
            flow_feats.append(flow_feat)

            # compute motion stats + motion energy map
            stats = compute_flow_stats(flows[i].to(device))  # [4]
            energy_map = compute_motion_energy_map(flows[i].to(device), out_size=self.motion_map_size)  # [h*w]
            motion_vec = torch.cat([stats, energy_map.to(device)], dim=0)  # [motion_vec_dim]
            motion_ctx = self.motion_proj(motion_vec)  # [motion_proj_dim]
            motion_contexts.append(motion_ctx)

        flow_feats = torch.stack(flow_feats, dim=0)            # [B, D]
        motion_contexts = torch.stack(motion_contexts, dim=0)  # [B, motion_proj_dim]
        clip_feats = clip_feats.to(device)

        gate_input = torch.cat([clip_feats, flow_feats, motion_contexts], dim=-1)  # [B, gate_in]
        gate = self.gate(gate_input)  # [B, clip_feat_dim] in (0,1)

        fused = gate * clip_feats + (1.0 - gate) * flow_feats  # [B, clip_feat_dim]
        return self.classifier(fused)


# Instantiate the new fusion head
fusion_head = FlowFusionHead().to(device)
fusion_head.load_state_dict(torch.load(model_path, map_location=device))
fusion_head.eval()
print(f"✅ Loaded motion-aware model from {model_path}")

# -----------------------------
# 4. Inference and annotated video export (no labels or evaluation)
# -----------------------------
@torch.no_grad()
def predict_probability(frames):
    """One violence probability per clip; repeat the last frame for short tails."""
    clip_frames = [
        Image.fromarray(cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB))
        for frame in frames
    ]
    clip_frames += [clip_frames[-1]] * (window_size - len(clip_frames))
    flows = torch.stack([
        run_flow([np.array(clip_frames[i]), np.array(clip_frames[i + 1])])
        for i in range(window_size - 1)
    ]).unsqueeze(0).to(device)
    clip_feat = extract_clip_features(clip_frames).unsqueeze(0).to(device)
    probability = torch.softmax(fusion_head(clip_feat, flows), dim=-1)[0, 1].item()
    if not math.isfinite(probability):
        raise RuntimeError("Model returned a non-finite probability")
    return probability


def probability_color(probability):
    """OpenCV BGR: 0 = green, 0.5 = yellow, 1 = red."""
    p = max(0.0, min(1.0, float(probability)))
    return (0, round(255 * min(1.0, 2.0 * (1.0 - p))),
            round(255 * min(1.0, 2.0 * p)))


def draw_overlay(frame, probability):
    frame = frame.copy()
    height, width = frame.shape[:2]
    color = probability_color(probability)
    border = min(args.border_width, max(1, min(height, width) // 10))
    frame[:border, :] = color
    frame[-border:, :] = color
    frame[:, :border] = color
    frame[:, -border:] = color

    label = f"Violence probability: {probability:.1%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(1.0, width / 640.0)
    thickness = max(1, round(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    margin = border + 6
    available = max(1, width - 2 * margin - 12)
    if text_width > available:
        scale *= available / text_width
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(frame, (margin, margin),
                  (margin + text_width + 12, margin + text_height + baseline + 12),
                  (0, 0, 0), -1)
    cv2.putText(frame, label, (margin + 6, margin + text_height + 6),
                font, scale, color, thickness, cv2.LINE_AA)
    return frame


def save_annotated_video(input_path, output_path):
    """Stream at most one clip at a time. OpenCV exports video without audio."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing output: {output_path} (use --overwrite to replace)")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".partial.mp4")
    cap = cv2.VideoCapture(str(input_path))
    writer = None
    written = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = args.fallback_fps
            print(f"Warning: invalid source FPS; using {fps:g}: {input_path}")
        while True:
            frames = []
            for _ in range(window_size):
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            if writer is None:
                height, width = frames[0].shape[:2]
                # MP4 codecs commonly require even dimensions; pad at most one pixel.
                output_size = (width + width % 2, height + height % 2)
                writer = cv2.VideoWriter(str(temporary_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create video: {temporary_path}")
            probability = predict_probability(frames)
            for frame in frames:
                height, width = frame.shape[:2]
                frame = cv2.copyMakeBorder(frame, 0, output_size[1] - height,
                                           0, output_size[0] - width, cv2.BORDER_REPLICATE)
                writer.write(draw_overlay(frame, probability))
                written += 1
            print(f"{input_path.name} | Frames: {written} | Violence: {probability:.1%}")
        if written == 0:
            raise RuntimeError(f"No readable frames: {input_path}")
        writer.release()
        writer = None
        os.replace(temporary_path, output_path)
        print(f"Saved: {output_path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if temporary_path.exists():
            temporary_path.unlink()


def main():
    output_root = Path(args.output_dir).expanduser().resolve()
    sources = [(args.input_dir, "")] if args.input_dir else [
        (normal_dir, "noFights"), (violence_dir, "fights")
    ]
    extensions = {".mp4", ".avi", ".mkv", ".mpg", ".mpeg", ".mov", ".m4v", ".webm"}
    jobs = []
    for folder, subfolder in sources:
        source = Path(folder).expanduser().resolve()
        destination = output_root / subfolder
        if not source.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {source}")
        if destination == source:
            raise ValueError("Choose an output directory different from the input directory")
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                # Keep the source extension in the name to avoid same-stem collisions.
                jobs.append((path, destination / f"{path.name}_annotated.mp4"))
    if not jobs:
        print("No supported videos found.")
        return
    failed = 0
    for input_path, output_path in jobs:
        try:
            save_annotated_video(input_path, output_path)
        except Exception as error:
            failed += 1
            print(f"Failed: {input_path}: {error}")
    if failed:
        raise SystemExit(f"{failed} video(s) failed; see messages above.")


if __name__ == "__main__":
    main()
