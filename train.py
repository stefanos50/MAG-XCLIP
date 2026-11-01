import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
from transformers import XCLIPProcessor, XCLIPModel
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train-file", default="./splits/WVD/train.txt")
parser.add_argument("--valid-file", default="./splits/WVD/test.txt")
parser.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--window-size", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--num-epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--lambda-adv", type=float, default=0.1)
parser.add_argument("--checkpoint-dir", default="./checkpoints")
parser.add_argument(
    "--best-metric", choices=["acc", "roc_auc", "pr_auc"], default="pr_auc"
)
args = parser.parse_args()

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)
device = args.device

device = device if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# config
window_size = args.window_size
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
lambda_adv = args.lambda_adv

# Optical Flow (unchanged)
flow_model = ptlflow.get_model('neuflow2', ckpt_path='sintel').to(device)
flow_model.eval()

@torch.no_grad()
def run_flow(frames):
    """Return tensor [2,H,W] optical flow between 2 RGB numpy frames normalized"""
    io_adapter = IOAdapter(flow_model, frames[0].shape[:2])
    inputs = io_adapter.prepare_inputs(frames)
    inputs['images'] = inputs['images'].to(device)
    preds = flow_model(inputs)
    flow = preds['flows'][0, 0].detach().cpu()
    flow = flow / (flow.abs().max() + 1e-6)  # normalized
    return flow

# X-CLIP (unchanged)
processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
xclip_model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32").to(device)
xclip_model.eval()
for p in xclip_model.parameters():
    p.requires_grad = False

@torch.no_grad()
def extract_clip_features(frames):
    inputs = processor(videos=frames, return_tensors="pt").to(device)
    feats = xclip_model.get_video_features(**inputs)
    return feats.squeeze(0).cpu()


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


class CrowdDataset(Dataset):
    def __init__(self, video_paths, labels, window_size=8, random_chunk=False):
        self.video_paths = video_paths
        self.labels = labels
        self.window_size = window_size
        self.random_chunk = random_chunk 

        self.video_chunks = []
        for vid_idx, path in enumerate(video_paths):
            cap = cv2.VideoCapture(path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            n_chunks = n_frames // window_size
            for i in range(n_chunks):
                self.video_chunks.append((vid_idx, i * window_size))

    def __len__(self):
        if self.random_chunk:
            return len(self.video_paths)
        else:
            return len(self.video_chunks)

    def read_video_frames(self, path, start_idx=0, n=8, size=(224, 224)):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _ in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        # If fewer than n frames left, pad with last
        if len(frames) < n and len(frames) > 0:
            frames += [frames[-1]] * (n - len(frames))
        return frames

    def __getitem__(self, idx):
        if self.random_chunk:
            vid_idx = idx
            path = self.video_paths[vid_idx]
            label = self.labels[vid_idx]
            cap = cv2.VideoCapture(path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n_frames <= self.window_size:
                start_idx = 0
            else:
                start_idx = random.randint(0, n_frames - self.window_size)
        else:
            vid_idx, start_idx = self.video_chunks[idx]
            path = self.video_paths[vid_idx]
            label = self.labels[vid_idx]

        # Read frames on the fly
        frames = self.read_video_frames(path, start_idx, self.window_size)

        # Compute flow sequence (T-1, 2, H, W)
        flows = []
        for i in range(len(frames) - 1):
            f1 = np.array(frames[i])
            f2 = np.array(frames[i + 1])
            flow = run_flow([f1, f2])
            flows.append(flow)
        flows = torch.stack(flows)  # [T-1, 2, H, W]

        # X-CLIP feature extraction (no grad)
        clip_feat = extract_clip_features(frames)
        return clip_feat, flows, torch.tensor(label, dtype=torch.long)



def compute_flow_stats(flows, motion_thresh=0.1):
    """
    flows: torch.Tensor [T,2,H,W] (CPU or GPU)
    Returns: torch.Tensor shape [7] ->
        [mean_mag, std_mag, max_mag, coverage_ratio,
         temporal_consistency, directional_consistency, global_motion_ratio]
    
    Statistics:
      - mean_mag: average motion magnitude (How strong is the motion on average?)
      - std_mag: std of magnitudes (How much variation in motion strength exists?)
      - max_mag: max motion magnitude (What’s the fastest motion we see?)
      - coverage_ratio: fraction of pixels with motion > motion_thresh (How much of the frame actually moves?)
      - temporal_consistency: smoothness of motion magnitude over time (Does motion change smoothly over time?) - Artifacts?
      - directional_consistency: variation of motion direction (lower = more consistent) (Do things move in similar directions over time?) Usufull in evacuation?
      - global_motion_ratio: strength of global (camera-like) motion vs. residual (scene) motion (How much of the motion looks like a camera movement?)
    """
    flows = flows.float()  # [T,2,H,W]
    T, _, H, W = flows.shape

    magnitude = torch.sqrt(flows[:, 0]**2 + flows[:, 1]**2)  # [T,H,W]
    mean_mag = magnitude.mean()
    std_mag = magnitude.std()
    max_mag = magnitude.max()
    motion_mask = (magnitude > motion_thresh).float()
    motion_coverage = motion_mask.mean()  # [scalar]

    if T > 1:
        diff = torch.abs(magnitude[1:] - magnitude[:-1])  # [T-1,H,W]
        temporal_consistency = 1.0 / (diff.mean() + 1e-6)  # higher = more consistent
    else:
        temporal_consistency = torch.tensor(0.0, device=flows.device)

    norm = torch.norm(flows, dim=1, keepdim=True) + 1e-6
    unit_flow = flows / norm  # [T,2,H,W]
    dir_var = unit_flow.var(dim=0).mean()
    directional_consistency = 1.0 / (dir_var + 1e-6)

    mean_flow_vec = flows.mean(dim=[2, 3])
    residuals = flows - mean_flow_vec.view(T, 2, 1, 1)
    global_motion_ratio = mean_flow_vec.norm(dim=1).mean() / (residuals.norm(dim=1).mean() + 1e-6)

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


#Creates a heatmap where motion happens
def compute_motion_energy_map(flows, out_size=(8, 8)):
    """
    flows: torch.Tensor [T,2,H,W]
    Returns: torch.Tensor flattened small map, shape [out_size[0]*out_size[1]]
    """
    flows = flows.float()
    magnitude = torch.sqrt(flows[:, 0]**2 + flows[:, 1]**2)  # Compute motion magnitude at each pixel and time step.
    motion_energy = magnitude.mean(dim=0, keepdim=True)  # Average across time → get the mean motion energy per spatial location.
    motion_energy = motion_energy.unsqueeze(0)  # [1,1,H,W]
    motion_energy_small = nn.functional.adaptive_avg_pool2d(motion_energy, out_size)  #Downsample the motion map to a fixed grid (e.g., 8×8). Resolution indipendent for linear layer.
    motion_energy_flat = motion_energy_small.view(-1)  # flat to 1d vector to feed to linear layer
    return motion_energy_flat  # 1D tensor



class FlowFusionHead(nn.Module):
    def __init__(self, clip_feat_dim=512, flow_dim=2, num_classes=2,
                 motion_map_size=(8, 8), motion_proj_dim=32):
        super().__init__()
        self.clip_feat_dim = clip_feat_dim
        self.motion_map_size = motion_map_size
        motion_map_elems = motion_map_size[0] * motion_map_size[1]
        self.motion_vec_dim = 7 + motion_map_elems  # 4 stats + flattened map

        self.flow_proj = nn.Sequential(
            nn.Conv2d(flow_dim, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, clip_feat_dim)
        )

        self.motion_proj = nn.Sequential(
            nn.Linear(self.motion_vec_dim, motion_proj_dim),
            nn.ReLU()
        )

        gate_in = clip_feat_dim + clip_feat_dim + motion_proj_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, clip_feat_dim),
            #nn.ReLU(), #comment
            #nn.Linear(clip_feat_dim, clip_feat_dim), #comment
            nn.Sigmoid()
        )


        self.classifier = nn.Sequential(
            nn.Linear(clip_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, clip_feats, flows):
        device = clip_feats.device
        B = flows.size(0)
        flow_feats = []
        motion_contexts = []

        for i in range(B):
            mean_flow = flows[i].mean(dim=0, keepdim=True).to(device)
            flow_feat = self.flow_proj(mean_flow)  # [1,clip_feat_dim]
            flow_feat = flow_feat.squeeze(0)
            flow_feats.append(flow_feat)

            stats = compute_flow_stats(flows[i].to(device))
            energy_map = compute_motion_energy_map(flows[i].to(device), out_size=self.motion_map_size)
            motion_vec = torch.cat([stats, energy_map.to(device)], dim=0) 
            motion_ctx = self.motion_proj(motion_vec)
            motion_contexts.append(motion_ctx)

        flow_feats = torch.stack(flow_feats, dim=0)            # [B, D]
        motion_contexts = torch.stack(motion_contexts, dim=0)  # [B, motion_proj_dim]
        clip_feats = clip_feats.to(device)

        gate_input = torch.cat([clip_feats, flow_feats, motion_contexts], dim=-1)
        gate = self.gate(gate_input)

        fused = gate * clip_feats + (1.0 - gate) * flow_feats
        return self.classifier(fused)


class Discriminator(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def load_split(file_path):
    video_paths, labels = [], []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on the last space
            path, label = line.rsplit(' ', 1)
            video_paths.append(path)
            labels.append(int(label))
    return video_paths, labels



train_videos, train_labels = load_split(args.train_file)
valid_videos, valid_labels = load_split(args.valid_file)

train_ds = CrowdDataset(train_videos, train_labels)
valid_ds = CrowdDataset(valid_videos, valid_labels)


train_ds = CrowdDataset(train_videos, train_labels, window_size=window_size)
valid_ds = CrowdDataset(valid_videos, valid_labels, window_size=window_size)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

print(f"Loaded: {len(train_ds)} train, {len(valid_ds)} validation samples.")


fusion_head = FlowFusionHead().to(device)
discriminator = Discriminator().to(device)

# Adversarial Discriminative Learning
optimizer_G = torch.optim.AdamW(fusion_head.parameters(), lr=lr, weight_decay=1e-4)
optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=lr*0.5, weight_decay=1e-4)
scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs)
criterion_cls = nn.CrossEntropyLoss()
criterion_adv = nn.BCELoss()

os.makedirs(args.checkpoint_dir, exist_ok=True)
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc

# ---------------- CONFIG ----------------
best_metric_mode = args.best_metric 
best_metric = 0.0
best_model_path = None

# ---------------- TRAINING ----------------
for epoch in range(num_epochs):
    fusion_head.train()
    discriminator.train()
    total_loss, total_adv_loss = 0.0, 0.0

    for clip_feats, flows, labels in train_loader:
        clip_feats, flows, labels = clip_feats.to(device), flows.to(device), labels.to(device)

        # ===== Generator (Fusion Head) =====
        preds = fusion_head(clip_feats, flows)
        cls_loss = criterion_cls(preds, labels)

        # Adversarial loss for generator
        fake_scores = discriminator(torch.softmax(preds.detach(), dim=-1))
        valid = torch.ones_like(fake_scores)
        adv_loss_G = criterion_adv(fake_scores, valid)

        loss_G = cls_loss + lambda_adv * adv_loss_G
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        total_loss += cls_loss.item() * clip_feats.size(0)

        discriminator.zero_grad()
        real_input = torch.nn.functional.one_hot(labels, num_classes=2).float()
        real_scores = discriminator(real_input)
        fake_input = torch.softmax(preds.detach(), dim=-1)
        fake_scores = discriminator(fake_input)

        valid = torch.ones_like(real_scores)
        fake = torch.zeros_like(fake_scores)
        loss_real = criterion_adv(real_scores, valid)
        loss_fake = criterion_adv(fake_scores, fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()
        total_adv_loss += loss_D.item() * clip_feats.size(0)

    scheduler_G.step()
    avg_loss = total_loss / len(train_loader.dataset)
    avg_adv_loss = total_adv_loss / len(train_loader.dataset)
    print(f"\nEpoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Adv Loss: {avg_adv_loss:.4f}")

    fusion_head.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for clip_feats, flows, labels in valid_loader:
            preds = fusion_head(clip_feats.to(device), flows.to(device))
            probs = torch.softmax(preds, dim=-1)
            pred = probs.argmax(dim=-1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())  # probability for violence class

    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    print(f"Validation | Acc: {acc*100:.2f}% | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    # --- Select which metric to track for best model ---
    if best_metric_mode == "acc":
        current_metric = acc
    elif best_metric_mode == "roc_auc":
        current_metric = roc_auc
    elif best_metric_mode == "pr_auc":
        current_metric = pr_auc
    else:
        raise ValueError("Invalid best_metric_mode. Choose from 'acc', 'roc_auc', or 'pr_auc'.")

    # --- Save best model ---
    if current_metric > best_metric:
        best_metric = current_metric
        best_model_path = os.path.join(
            args.checkpoint_dir, f"fusion_head_best_{best_metric_mode}.pth"
        )
        torch.save(fusion_head.state_dict(), best_model_path)
        print(f" Saved new best model with {best_metric_mode} = {best_metric:.4f}")

print(f"\nTraining complete. Best model saved at: {best_model_path} ({best_metric_mode} = {best_metric:.4f})")

