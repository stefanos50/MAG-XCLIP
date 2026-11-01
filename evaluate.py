import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, average_precision_score
from transformers import XCLIPProcessor, XCLIPModel
import cv2
import numpy as np
from PIL import Image
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--normal-dir", default="./movies_dataset/noFights")
parser.add_argument("--violence-dir", default="./movies_dataset/fights")
parser.add_argument("--model-path", default="./checkpoints/scvd.pth")
parser.add_argument("--window-size", type=int, default=8)
args = parser.parse_args()

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
# 4. Video utils
# -----------------------------
def read_video_frames(path, size=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

def split_chunks(frames, n=8):
    return [frames[i:i+n] for i in range(0, len(frames), n) if len(frames[i:i+n]) == n]

def collect_videos(folder, label):
    exts = ('.mp4', '.avi', '.mkv', '.mpg')
    paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    return paths, [label]*len(paths)

# -----------------------------
# 5. Evaluation
# -----------------------------
def evaluate_clip_level():
    normal_videos, normal_labels = collect_videos(normal_dir, 0)
    violence_videos, violence_labels = collect_videos(violence_dir, 1)

    all_videos = normal_videos + violence_videos
    all_labels = normal_labels + violence_labels

    y_true, y_pred, y_prob = [], [], []

    for path, label in zip(all_videos, all_labels):
        frames = read_video_frames(path)
        chunks = split_chunks(frames, window_size)
        if len(chunks) == 0:
            continue

        print(f"Processing {os.path.basename(path)} ({len(chunks)} clips)")

        for idx, clip_frames in enumerate(chunks):
            flows = []
            for i in range(len(clip_frames)-1):
                f1 = np.array(clip_frames[i])
                f2 = np.array(clip_frames[i+1])
                flows.append(run_flow([f1, f2]))
            flows = torch.stack(flows).unsqueeze(0).to(device)

            clip_feat = extract_clip_features(clip_frames).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = fusion_head(clip_feat, flows)
                probs = torch.softmax(logits, dim=-1)
                pred = probs.argmax(dim=-1).item()
                prob_violence = probs[0,1].item()  # probability of violence

                y_true.append(label)
                y_pred.append(pred)
                y_prob.append(prob_violence)

                print(f"Clip {idx+1}/{len(chunks)} | True: {label} | Pred: {pred} | Prob_violence: {prob_violence:.4f}")
    
    np.save("preds.npy", y_prob)
    np.save("true_labels.npy", y_true)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ap = average_precision_score(y_true, y_prob)

    print("\n Evaluation Results:")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"AP: {ap:.4f}")

    return acc, roc_auc, pr_auc

if __name__ == "__main__":
    evaluate_clip_level()

