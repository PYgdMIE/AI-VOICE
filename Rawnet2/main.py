import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from data_utils import (
    ASVspoof2019EvalDataset,
    ASVspoof2019TrainDataset,
    ASVspoof2021EvalDataset,
    AudioFolderDataset,
    CSVAudioDataset,
    discover_ljspeech_audio,
    discover_audio_files,
    load_asvspoof_2019_protocol,
    load_asvspoof_2021_eval_list,
    load_csv_audio_manifest,
)
from metrics_utils import (
    attach_truth_to_row,
    compute_metrics_from_rows,
    load_cm_trial_metadata,
    save_metrics,
)
from model import RawNet

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    from tensorboardX import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or score the RawNet2 anti-spoofing model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train on ASVspoof 2019 LA.")
    add_common_model_args(train_parser)
    train_parser.add_argument("--data-root", type=Path, required=True)
    train_parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    train_parser.add_argument("--run-name", type=str, default=None)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--num-epochs", type=int, default=100)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--train-subset", type=int, default=None)
    train_parser.add_argument("--dev-subset", type=int, default=None)
    train_parser.add_argument("--max-train-batches", type=int, default=None)
    train_parser.add_argument("--max-dev-batches", type=int, default=None)
    train_parser.add_argument("--log-interval", type=int, default=10)
    train_parser.add_argument("--ljspeech-root", type=Path, default=None)
    train_parser.add_argument("--ljspeech-limit", type=int, default=None)
    train_parser.add_argument("--resume", action="store_true", default=False)

    dev_parser = subparsers.add_parser(
        "score-dev", help="Score the ASVspoof 2019 LA development set."
    )
    add_common_model_args(dev_parser)
    dev_parser.add_argument("--data-root", type=Path, required=True)
    dev_parser.add_argument("--output-path", type=Path, required=True)
    dev_parser.add_argument("--batch-size", type=int, default=32)
    dev_parser.add_argument("--num-workers", type=int, default=0)
    dev_parser.add_argument("--subset", type=int, default=None)
    dev_parser.add_argument("--resume", action="store_true", default=False)

    eval2021_parser = subparsers.add_parser(
        "score-2021", help="Score the ASVspoof 2021 LA evaluation set."
    )
    add_common_model_args(eval2021_parser)
    eval2021_parser.add_argument("--eval-root", type=Path, required=True)
    eval2021_parser.add_argument("--output-path", type=Path, required=True)
    eval2021_parser.add_argument("--protocol-path", type=Path, default=None)
    eval2021_parser.add_argument("--batch-size", type=int, default=64)
    eval2021_parser.add_argument("--num-workers", type=int, default=0)
    eval2021_parser.add_argument("--subset", type=int, default=None)
    eval2021_parser.add_argument("--resume", action="store_true", default=False)

    folder_parser = subparsers.add_parser(
        "score-dir", help="Score a local folder of .wav or .flac files."
    )
    add_common_model_args(folder_parser)
    folder_parser.add_argument("--input-dir", type=Path, required=True)
    folder_parser.add_argument("--output-path", type=Path, required=True)
    folder_parser.add_argument("--batch-size", type=int, default=64)
    folder_parser.add_argument("--num-workers", type=int, default=0)
    folder_parser.add_argument("--subset", type=int, default=None)
    folder_parser.add_argument("--resume", action="store_true", default=False)

    csv_parser = subparsers.add_parser(
        "score-csv",
        help="Score utterances listed in a CSV file and write a scored CSV.",
    )
    add_common_model_args(csv_parser)
    csv_parser.add_argument("--input-csv", type=Path, required=True)
    csv_parser.add_argument("--audio-root", type=Path, required=True)
    csv_parser.add_argument("--output-path", type=Path, required=True)
    csv_parser.add_argument("--id-column", type=str, default="utterance_id")
    csv_parser.add_argument("--batch-size", type=int, default=64)
    csv_parser.add_argument("--num-workers", type=int, default=0)
    csv_parser.add_argument("--subset", type=int, default=None)
    csv_parser.add_argument("--cm-key-path", type=Path, default=None)
    csv_parser.add_argument("--metrics-output", type=Path, default=None)
    csv_parser.add_argument("--log-interval", type=int, default=10)
    csv_parser.add_argument("--progress-csv", type=Path, default=None)
    csv_parser.add_argument("--progress-plot", type=Path, default=None)
    csv_parser.add_argument("--resume", action="store_true", default=False)

    return parser.parse_args()


def add_common_model_args(parser):
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("model_config_RawNet2.yaml"),
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def build_model(config_path, device, checkpoint_path=None):
    config = load_model_config(config_path)
    model = RawNet(config["model"], device.type).to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(extract_model_state(checkpoint))
    return model


def build_run_name(args):
    if args.run_name:
        return args.run_name
    return f"rawnet2_bs{args.batch_size}_lr{args.lr}_ep{args.num_epochs}"


def maybe_slice(items, limit):
    if limit is None:
        return items
    return items[:limit]


def load_training_state(state_path, device):
    checkpoint = torch.load(state_path, map_location=device)
    required_keys = {"model_state_dict", "optimizer_state_dict", "epoch"}
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint.keys()):
        raise ValueError(f"{state_path} is not a valid training-state checkpoint.")
    return checkpoint


def save_training_state(
    state_path,
    epoch,
    model,
    optimizer,
    best_dev_accuracy,
    history,
    run_name,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dev_accuracy": best_dev_accuracy,
        "history": history,
        "run_name": run_name,
    }
    torch.save(checkpoint, state_path)


def read_existing_text_keys(output_path):
    output_path = Path(output_path)
    if not output_path.exists():
        return set()
    keys = set()
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if parts:
                keys.add(parts[0])
    return keys


def read_existing_csv_keys(output_path, key_column):
    output_path = Path(output_path)
    if not output_path.exists():
        return set()
    with open(output_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or key_column not in reader.fieldnames:
            return set()
        return {
            (row.get(key_column) or "").strip()
            for row in reader
            if (row.get(key_column) or "").strip()
        }


def read_existing_scored_csv_rows(output_path):
    output_path = Path(output_path)
    if not output_path.exists():
        return []
    with open(output_path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_training_visuals(history, run_root):
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    history_csv_path = run_root / "history.csv"
    with open(history_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_accuracy", "dev_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)

    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    train_accuracy = [entry["train_accuracy"] for entry in history]
    dev_accuracy = [entry["dev_accuracy"] for entry in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(epochs, train_loss, color="#1f77b4", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accuracy, label="Train Accuracy", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, dev_accuracy, label="Dev Accuracy", color="#d62728", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.savefig(run_root / "training_curves.png", dpi=160)
    plt.close(fig)


def format_duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def write_progress_visuals(progress_points, csv_path=None, plot_path=None, title="Progress"):
    if not progress_points:
        return

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "batch_index",
                    "processed_items",
                    "total_items",
                    "percent_complete",
                    "elapsed_seconds",
                    "items_per_second",
                ],
            )
            writer.writeheader()
            writer.writerows(progress_points)

    if plot_path is None:
        return

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    elapsed_minutes = [point["elapsed_seconds"] / 60.0 for point in progress_points]
    percent_complete = [point["percent_complete"] for point in progress_points]
    processed_items = [point["processed_items"] for point in progress_points]
    total_items = progress_points[-1]["total_items"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(elapsed_minutes, percent_complete, color="#1f77b4", linewidth=2)
    axes[0].set_title(f"{title} Completion")
    axes[0].set_xlabel("Elapsed Time (minutes)")
    axes[0].set_ylabel("Complete (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(elapsed_minutes, processed_items, color="#2ca02c", linewidth=2)
    axes[1].set_title("Processed Items")
    axes[1].set_xlabel("Elapsed Time (minutes)")
    axes[1].set_ylabel("Items")
    axes[1].set_ylim(0, max(total_items, 1))
    axes[1].grid(True, alpha=0.3)

    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def train_one_epoch(
    data_loader,
    model,
    optimizer,
    device,
    epoch,
    log_interval=10,
    max_batches=None,
):
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([0.1, 0.9], dtype=torch.float32, device=device)
    )
    model.train()
    running_loss = 0.0
    num_correct = 0
    num_total = 0
    total_batches = len(data_loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    for batch_index, (batch_x, batch_y) in enumerate(data_loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device=device, dtype=torch.int64)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_x.size(0)
        running_loss += loss.item() * batch_size
        num_total += batch_size
        num_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        if (
            batch_index == 1
            or batch_index % max(log_interval, 1) == 0
            or batch_index == total_batches
        ):
            avg_loss = running_loss / max(num_total, 1)
            avg_acc = 100.0 * num_correct / max(num_total, 1)
            print(
                f"[Train] epoch {epoch:03d} "
                f"batch {batch_index:04d}/{total_batches:04d} "
                f"loss={loss.item():.5f} avg_loss={avg_loss:.5f} avg_acc={avg_acc:.2f}",
                flush=True,
            )

    return running_loss / max(num_total, 1), (100.0 * num_correct / max(num_total, 1))


@torch.no_grad()
def evaluate_accuracy(data_loader, model, device, max_batches=None):
    model.eval()
    num_correct = 0
    num_total = 0

    for batch_index, (batch_x, batch_y) in enumerate(data_loader, start=1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device=device, dtype=torch.int64)
        logits = model(batch_x)
        num_total += batch_x.size(0)
        num_correct += (logits.argmax(dim=1) == batch_y).sum().item()

    return 100.0 * num_correct / max(num_total, 1)


@torch.no_grad()
def score_asvspoof2019_dev(data_loader, model, device, output_path, append=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    model.eval()
    with open(output_path, mode, encoding="utf-8") as handle:
        for batch_x, batch_meta in data_loader:
            batch_x = batch_x.to(device)
            bonafide_scores = model(batch_x)[:, 1].detach().cpu().numpy().tolist()
            for utt_id, attack_id, label, score in zip(
                batch_meta["utt_id"],
                batch_meta["attack_id"],
                batch_meta["label"],
                bonafide_scores,
            ):
                handle.write(f"{utt_id} {attack_id} {label} {score}\n")
            handle.flush()


@torch.no_grad()
def score_asvspoof2021_eval(data_loader, model, device, output_path, append=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    model.eval()
    with open(output_path, mode, encoding="utf-8") as handle:
        for batch_x, batch_ids in data_loader:
            batch_x = batch_x.to(device)
            bonafide_scores = model(batch_x)[:, 1].detach().cpu().numpy().tolist()
            for utt_id, score in zip(batch_ids, bonafide_scores):
                handle.write(f"{utt_id} {score}\n")
            handle.flush()


@torch.no_grad()
def score_audio_folder(data_loader, model, device, output_path, append=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    model.eval()
    with open(output_path, mode, encoding="utf-8", newline="") as handle:
        if not append:
            handle.write("file,bonafide_score,spoof_probability,bonafide_probability,prediction\n")
        for batch_x, batch_paths in data_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            bonafide_scores = logits[:, 1].detach().cpu().numpy()
            for path_str, bonafide_score, prob_pair in zip(
                batch_paths, bonafide_scores, probs
            ):
                prob_spoof = float(prob_pair[0])
                prob_bonafide = float(prob_pair[1])
                prediction = "bonafide" if prob_bonafide >= prob_spoof else "spoof"
                handle.write(
                    f"{Path(path_str).as_posix()},{bonafide_score},{prob_spoof},"
                    f"{prob_bonafide},{prediction}\n"
                )
            handle.flush()


@torch.no_grad()
def score_csv_manifest(
    data_loader,
    model,
    device,
    output_path,
    truth_index=None,
    id_column="utterance_id",
    append=False,
    log_interval=10,
    progress_csv_path=None,
    progress_plot_path=None,
    completed_offset=0,
    total_items=None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    model.eval()
    write_header = (not append) or (append and (not output_path.exists() or output_path.stat().st_size == 0))
    total_items = int(total_items if total_items is not None else len(data_loader.dataset))
    total_batches = len(data_loader)
    processed_now = 0
    progress_points = []
    start_time = time.time()

    with open(output_path, mode, encoding="utf-8-sig", newline="") as handle:
        writer = None
        for batch_index, (batch_x, batch_meta) in enumerate(data_loader, start=1):
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            bonafide_scores = logits[:, 1].detach().cpu().numpy()

            batch_size = len(bonafide_scores)
            for row_index in range(batch_size):
                row = {key: batch_meta[key][row_index] for key in batch_meta}
                prob_spoof = float(probs[row_index][0])
                prob_bonafide = float(probs[row_index][1])
                row["bonafide_score"] = float(bonafide_scores[row_index])
                row["spoof_probability"] = prob_spoof
                row["bonafide_probability"] = prob_bonafide
                row["prediction"] = (
                    "bonafide" if prob_bonafide >= prob_spoof else "spoof"
                )
                if truth_index is not None:
                    row = attach_truth_to_row(row, truth_index, id_column=id_column)

                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                    if write_header:
                        writer.writeheader()
                writer.writerow(row)
            handle.flush()
            processed_now += batch_size

            if (
                batch_index == 1
                or batch_index % max(log_interval, 1) == 0
                or batch_index == total_batches
            ):
                completed_items = min(completed_offset + processed_now, total_items)
                elapsed_seconds = time.time() - start_time
                items_per_second = processed_now / max(elapsed_seconds, 1e-9)
                remaining_items = max(total_items - completed_items, 0)
                eta_seconds = (
                    remaining_items / items_per_second if items_per_second > 0 else None
                )
                percent_complete = 100.0 * completed_items / max(total_items, 1)
                print(
                    f"[ScoreCSV] batch {batch_index:04d}/{total_batches:04d} "
                    f"rows {completed_items:05d}/{total_items:05d} "
                    f"pct={percent_complete:6.2f} "
                    f"elapsed={format_duration(elapsed_seconds)} "
                    f"eta={format_duration(eta_seconds)}",
                    flush=True,
                )
                progress_points.append(
                    {
                        "batch_index": batch_index,
                        "processed_items": completed_items,
                        "total_items": total_items,
                        "percent_complete": percent_complete,
                        "elapsed_seconds": float(elapsed_seconds),
                        "items_per_second": float(items_per_second),
                    }
                )
                write_progress_visuals(
                    progress_points,
                    csv_path=progress_csv_path,
                    plot_path=progress_plot_path,
                    title="Score CSV",
                )


def run_train(args):
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    data_root = args.data_root.resolve()
    train_protocol = (
        data_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    )
    dev_protocol = (
        data_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    )
    train_root = data_root / "ASVspoof2019_LA_train"
    dev_root = data_root / "ASVspoof2019_LA_dev"

    train_trials = maybe_slice(
        load_asvspoof_2019_protocol(train_protocol), args.train_subset
    )
    dev_trials = maybe_slice(load_asvspoof_2019_protocol(dev_protocol), args.dev_subset)

    ljspeech_audio_paths = []
    if args.ljspeech_root is not None:
        ljspeech_audio_paths = maybe_slice(
            discover_ljspeech_audio(args.ljspeech_root.resolve()),
            args.ljspeech_limit,
        )
        if not ljspeech_audio_paths:
            raise ValueError(
                f"No LJSpeech audio files were found in {args.ljspeech_root.resolve()}."
            )

    train_dataset = ASVspoof2019TrainDataset(
        train_trials,
        train_root,
        supplemental_bonafide_paths=ljspeech_audio_paths,
    )
    dev_dataset = ASVspoof2019TrainDataset(dev_trials, dev_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    run_name = build_run_name(args)
    run_root = args.output_root.resolve() / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "train_state.pt"
    writer = SummaryWriter(log_dir=str(run_root / "tensorboard"))

    model = build_model(args.model_config, device, args.model_path)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_dev_accuracy = 0.0
    history = []
    start_epoch = 1

    if args.resume and state_path.exists():
        checkpoint = load_training_state(state_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_dev_accuracy = float(checkpoint.get("best_dev_accuracy", 0.0))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming training from epoch {start_epoch} using {state_path}")

    print(f"Device: {device}")
    print(f"Training files: {len(train_dataset)}")
    print(f"Validation files: {len(dev_dataset)}")
    print(f"Run directory: {run_root}")
    print(
        f"Train label sources | asvspoof={len(train_trials)} | "
        f"ljspeech_bonafide={len(ljspeech_audio_paths)}"
    )
    if history:
        write_training_visuals(history, run_root)

    if start_epoch > args.num_epochs:
        print(
            f"Training already reached epoch {start_epoch - 1}, "
            f"which is >= requested num_epochs={args.num_epochs}. Nothing to do."
        )
        print(f"Training curves: {run_root / 'training_curves.png'}")
        writer.close()
        return

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            train_loader,
            model,
            optimizer,
            device,
            epoch=epoch,
            log_interval=args.log_interval,
            max_batches=args.max_train_batches,
        )
        dev_acc = evaluate_accuracy(
            dev_loader, model, device, max_batches=args.max_dev_batches
        )
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("accuracy/train", train_acc, epoch)
        writer.add_scalar("accuracy/dev", dev_acc, epoch)

        epoch_state = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "dev_accuracy": dev_acc,
        }
        history.append(epoch_state)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} "
            f"| train_acc={train_acc:.2f} | dev_acc={dev_acc:.2f}",
            flush=True,
        )
        write_training_visuals(history, run_root)

        if dev_acc >= best_dev_accuracy:
            best_dev_accuracy = dev_acc
            torch.save(model.state_dict(), run_root / "best.pth")
            print(f"Saved new best checkpoint to {run_root / 'best.pth'}")

        torch.save(model.state_dict(), run_root / "last.pth")
        save_training_state(
            state_path=state_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_dev_accuracy=best_dev_accuracy,
            history=history,
            run_name=run_name,
        )

    writer.close()
    summary = {
        "device": str(device),
        "run_name": run_name,
        "best_dev_accuracy": best_dev_accuracy,
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "ljspeech_examples": len(ljspeech_audio_paths),
        "ljspeech_root": str(args.ljspeech_root.resolve()) if args.ljspeech_root else None,
        "history": history,
    }
    with open(run_root / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Training curves: {run_root / 'training_curves.png'}")


def run_score_dev(args):
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    data_root = args.data_root.resolve()
    protocol = (
        data_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    )
    dev_root = data_root / "ASVspoof2019_LA_dev"
    trials = maybe_slice(load_asvspoof_2019_protocol(protocol), args.subset)
    output_path = args.output_path.resolve()
    append = False

    if args.resume and output_path.exists():
        processed_ids = read_existing_text_keys(output_path)
        trials = [trial for trial in trials if trial.utt_id not in processed_ids]
        append = True
        print(f"Resuming dev scoring: skipping {len(processed_ids)} existing rows.")

    if not trials:
        print(f"All requested dev utterances are already present in {output_path}")
        return

    dataset = ASVspoof2019EvalDataset(trials, dev_root)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    model = build_model(args.model_config, device, args.model_path)
    score_asvspoof2019_dev(data_loader, model, device, output_path, append=append)
    print(f"Saved ASVspoof 2019 dev scores to {output_path}")


def run_score_2021(args):
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    eval_root = args.eval_root.resolve()
    protocol_path = (
        args.protocol_path.resolve()
        if args.protocol_path
        else eval_root / "ASVspoof2021.LA.cm.eval.trl.txt"
    )
    utt_ids = maybe_slice(load_asvspoof_2021_eval_list(protocol_path), args.subset)
    output_path = args.output_path.resolve()
    append = False

    if args.resume and output_path.exists():
        processed_ids = read_existing_text_keys(output_path)
        utt_ids = [utt_id for utt_id in utt_ids if utt_id not in processed_ids]
        append = True
        print(f"Resuming 2021 scoring: skipping {len(processed_ids)} existing rows.")

    if not utt_ids:
        print(f"All requested 2021 utterances are already present in {output_path}")
        return

    dataset = ASVspoof2021EvalDataset(utt_ids, eval_root)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    model = build_model(args.model_config, device, args.model_path)
    score_asvspoof2021_eval(data_loader, model, device, output_path, append=append)
    print(f"Saved ASVspoof 2021 evaluation scores to {output_path}")


def run_score_dir(args):
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    audio_files = maybe_slice(discover_audio_files(args.input_dir.resolve()), args.subset)
    output_path = args.output_path.resolve()
    append = False

    if args.resume and output_path.exists():
        processed_files = read_existing_csv_keys(output_path, "file")
        audio_files = [
            audio_path
            for audio_path in audio_files
            if Path(audio_path).as_posix() not in processed_files
        ]
        append = True
        print(f"Resuming folder scoring: skipping {len(processed_files)} existing rows.")

    if not audio_files:
        if output_path.exists():
            print(f"All requested folder items are already present in {output_path}")
            return
        raise FileNotFoundError(f"No .wav or .flac files were found in {args.input_dir}")

    dataset = AudioFolderDataset(audio_files)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    model = build_model(args.model_config, device, args.model_path)
    score_audio_folder(data_loader, model, device, output_path, append=append)
    print(f"Saved folder scores to {output_path}")


def run_score_csv(args):
    set_random_seed(args.seed)
    device = resolve_device(args.device)
    rows = maybe_slice(
        load_csv_audio_manifest(args.input_csv.resolve(), id_column=args.id_column),
        args.subset,
    )
    if not rows:
        raise ValueError(f"No rows with '{args.id_column}' were found in {args.input_csv}")
    total_requested_rows = len(rows)

    output_path = args.output_path.resolve()
    append = False
    completed_offset = 0
    if args.resume and output_path.exists():
        processed_ids = read_existing_csv_keys(output_path, args.id_column)
        rows = [row for row in rows if row[args.id_column] not in processed_ids]
        completed_offset = total_requested_rows - len(rows)
        append = True
        print(f"Resuming CSV scoring: skipping {len(processed_ids)} existing rows.")

    truth_index = None
    if args.cm_key_path is not None:
        truth_index = load_cm_trial_metadata(args.cm_key_path.resolve())

    if rows:
        dataset = CSVAudioDataset(
            rows=rows,
            audio_root=args.audio_root.resolve(),
            id_column=args.id_column,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        model = build_model(args.model_config, device, args.model_path)
        score_csv_manifest(
            data_loader,
            model,
            device,
            output_path,
            truth_index=truth_index,
            id_column=args.id_column,
            append=append,
            log_interval=args.log_interval,
            progress_csv_path=args.progress_csv.resolve() if args.progress_csv else None,
            progress_plot_path=args.progress_plot.resolve() if args.progress_plot else None,
            completed_offset=completed_offset,
            total_items=total_requested_rows,
        )
        print(f"Saved scored CSV to {output_path}")
    elif output_path.exists():
        print(f"All requested CSV rows are already present in {output_path}")
    else:
        raise ValueError("No rows available to score after resume filtering.")

    if truth_index is not None:
        scored_rows = read_existing_scored_csv_rows(output_path)
        metrics = compute_metrics_from_rows(scored_rows)
        if args.metrics_output is not None:
            save_metrics(metrics, args.metrics_output.resolve())
            print(f"Saved metrics JSON to {args.metrics_output.resolve()}")

        overall = metrics.get("overall", {})
        if overall:
            eer_value = overall.get("eer_percent")
            eer_text = "n/a" if eer_value is None else f"{eer_value:.6f}"
            print(
                "Overall metrics | "
                f"rows={overall.get('num_rows')} | "
                f"accuracy={overall.get('accuracy', 0.0):.4f} | "
                f"balanced_accuracy={overall.get('balanced_accuracy', 0.0):.4f} | "
                f"eer={eer_text}"
            )


def main():
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "score-dev":
        if args.model_path is None:
            raise ValueError("--model-path is required for score-dev")
        run_score_dev(args)
    elif args.command == "score-2021":
        if args.model_path is None:
            raise ValueError("--model-path is required for score-2021")
        run_score_2021(args)
    elif args.command == "score-dir":
        if args.model_path is None:
            raise ValueError("--model-path is required for score-dir")
        run_score_dir(args)
    elif args.command == "score-csv":
        if args.model_path is None:
            raise ValueError("--model-path is required for score-csv")
        run_score_csv(args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
