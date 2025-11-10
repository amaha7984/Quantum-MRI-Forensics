import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from PIL import ImageFile

import pennylane as qml
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


N_QUBITS = 4          # keeping small
Q_DEPTH = 2           # keeping shallow
MEAN = (0.2447, 0.2446, 0.2447)
STD = (0.1892, 0.1892, 0.1891)

# =========================================================
# Quantum device selection (GPU if possible)
# =========================================================

USE_QGPU = False
dev = None

if torch.cuda.is_available():
    try:
        # Requires pennylane-lightning-gpu (or gpu-enabled lightning)
        dev = qml.device("lightning.gpu", wires=N_QUBITS)
        USE_QGPU = True
        print("Using PennyLane lightning.gpu backend for quantum simulation.")
    except Exception as e:
        print(f"Could not initialize lightning.gpu ({e}), falling back to default.qubit (CPU).")
        dev = qml.device("default.qubit", wires=N_QUBITS)
        USE_QGPU = False
else:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    USE_QGPU = False
    print("CUDA not available; using default.qubit (CPU).")

DEVICE = torch.device("cuda:0") if USE_QGPU else torch.device("cpu")
print(f"Using DEVICE for model: {DEVICE}")

# =========================================================
# Data
# =========================================================

def train_transforms(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(torch.tensor(MEAN), torch.tensor(STD)),
        ]
    )


def val_transforms(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(torch.tensor(MEAN), torch.tensor(STD)),
        ]
    )


def make_dataloaders(train_path, val_path, batch_size):
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=train_transforms(224),
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=val_path,
        transform=val_transforms(224),
    )

    if len(train_dataset.classes) != 2:
        print(f"WARNING: Expected 2 classes for binary classification, "
              f"found {len(train_dataset.classes)}: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return train_loader, val_loader


# =========================================================
# Quantum Layer
# =========================================================

class QuantumCircuitLayer(nn.Module):
    """
    Multi-qubit variational quantum layer.

    Input:  [B, N_QUBITS]  (float32 on DEVICE)
    Output: [B, N_QUBITS]  (float32 on DEVICE)
    """

    def __init__(self, n_qubits=N_QUBITS, depth=Q_DEPTH):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth

        # Trainable parameters: [depth, n_qubits, 2] for RY and RZ
        w_init = 0.01 * torch.randn(depth, n_qubits, 2, dtype=torch.float32)
        self.weights = nn.Parameter(w_init)

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(x, w):
            """
            x: [n_qubits], w: [depth, n_qubits, 2]
            Returns: [n_qubits]
            """
            # Ensure we're working with float32 tensors
            x_ = x.to(torch.float32)
            w_ = w.to(torch.float32)

            # Encode inputs
            for i in range(n_qubits):
                qml.RY(x_[i] * np.pi, wires=i)

            # Variational layers with ring entanglement
            for d in range(depth):
                # Ring of CNOTs
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

                # Local rotations
                for i in range(n_qubits):
                    qml.RY(w_[d, i, 0], wires=i)
                    qml.RZ(w_[d, i, 1], wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_qubits] on DEVICE
        Returns: [B, n_qubits] on DEVICE
        """
        assert x.device == DEVICE, f"QuantumCircuitLayer expects {DEVICE}, got {x.device}"
        bsz, feat_dim = x.shape
        assert feat_dim == self.n_qubits, f"Expected {self.n_qubits} features, got {feat_dim}"

        outputs = []

        # Strategy:
        # - If using GPU backend: call QNode with CUDA tensors.
        # - If using CPU backend: move inputs & weights to CPU for the QNode, then back to DEVICE.
        for i in range(bsz):
            if USE_QGPU:
                x_in = x[i]                      # stays on cuda
                w_in = self.weights              # stays on cuda
            else:
                x_in = x[i].to("cpu")
                w_in = self.weights.to("cpu")

            q_out = self.qnode(x_in, w_in)

            # Normalize to float32 tensor on DEVICE
            if isinstance(q_out, (list, tuple)):
                elems = []
                for elem in q_out:
                    if isinstance(elem, torch.Tensor):
                        elems.append(elem.to(dtype=torch.float32))
                    else:
                        elems.append(torch.tensor(elem, dtype=torch.float32))
                q_out = torch.stack(elems, dim=0)
            elif isinstance(q_out, torch.Tensor):
                q_out = q_out.to(dtype=torch.float32)
            else:
                q_out = torch.tensor(q_out, dtype=torch.float32)

            q_out = q_out.to(DEVICE)
            outputs.append(q_out)

        return torch.stack(outputs, dim=0)  # [B, n_qubits]


# =========================================================
# Hybrid QNN Classifier
# =========================================================

class HybridQNNBinaryClassifier(nn.Module):
    """
    224x224x3 image
      -> classical encoder
      -> N_QUBITS features in [-1,1]
      -> quantum layer
      -> linear head -> 1 logit
    """

    def __init__(self, n_qubits=N_QUBITS, depth=Q_DEPTH):
        super().__init__()
        self.n_qubits = n_qubits

        self.pool = nn.AdaptiveAvgPool2d((32, 32))  # [B,3,32,32] -> 3072

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_qubits),
            nn.Tanh(),  # keep in [-1,1]
        )

        self.quantum = QuantumCircuitLayer(n_qubits=n_qubits, depth=depth)

        self.head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),  # single logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(DEVICE, dtype=torch.float32)
        x = self.pool(x)
        x = self.encoder(x)               # [B, n_qubits] on DEVICE
        q = self.quantum(x)               # [B, n_qubits] on DEVICE
        logits = self.head(q)             # [B, 1] on DEVICE
        return logits


# =========================================================
# Training / Evaluation
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE).float().unsqueeze(1)  # [B,1]

        optimizer.zero_grad()

        outputs = model(images)               # [B,1]
        loss = criterion(outputs, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, acc


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)




def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum QNN Binary Classifier (Single-process, optional GPU, no DDP)"
    )
    parser.add_argument("train_path", type=str, help="Path to training dataset (ImageFolder with 2 classes)")
    parser.add_argument("val_path", type=str, help="Path to validation dataset (ImageFolder with 2 classes)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size. For QNN, 4–16 is realistic. 256 is NOT.")
    parser.add_argument("--total_epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--save_path", type=str, default="./saved_models/qnn_model.pth",
                        help="Path to save best model (by val accuracy)")

    args = parser.parse_args()

    print(f"Train path: {args.train_path}")
    print(f"Val path:   {args.val_path}")

    if args.batch_size > 32:
        print(f"WARNING: batch_size={args.batch_size} is very large for a quantum simulator.")
        print("For a fair + runnable experiment, use --batch_size 4, 8, or 16 instead.")

    train_loader, val_loader = make_dataloaders(
        args.train_path,
        args.val_path,
        batch_size=args.batch_size,
    )

    model = HybridQNNBinaryClassifier(n_qubits=N_QUBITS, depth=Q_DEPTH).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    best_val_acc = 0.0

    for epoch in range(args.total_epochs):
        print(f"\nEpoch {epoch + 1}/{args.total_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, args.save_path)
            print(f"✓ Saved new best model with Val Acc = {best_val_acc:.2f}%")

    print(f"\nBest Val Acc: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()
