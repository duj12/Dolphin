###
# Author: Kai Li
# Date: 2025-09-28
# LastEditors: Kai Li
# LastEditTime: 2026-04-13
###

import os
import argparse
import warnings

import torch
import torchaudio
import yaml
import numpy as np

warnings.filterwarnings("ignore")

import look2hear.models
import look2hear.datas
from look2hear.metrics import ALLMetricsTracker

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def main(config):
    # ---- paths ----
    exp_dir = os.path.join(
        os.getcwd(),
        "Experiments",
        "checkpoint",
        config["exp"]["exp_name"],
    )
    model_path = os.path.join(exp_dir, "best_model.pth")
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ---- device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- model ----
    audionet_name = config["audionet"]["audionet_name"]
    audionet_cfg = dict(config["audionet"]["audionet_config"])
    audionet_cfg["is_train"] = False

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = getattr(look2hear.models, audionet_name)(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **audionet_cfg,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # ---- data ----
    datamodule = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"],
    )
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    # ---- metrics ----
    metrics = ALLMetricsTracker(
        save_file=os.path.join(results_dir, "metrics.csv"),
    )

    # ---- progress bar ----
    progress = Progress(
        TextColumn("[bold blue]Evaluating", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    # ---- evaluation loop ----
    n_src = config["datamodule"]["data_config"].get("n_src", 1)
    sample_rate = config["datamodule"]["data_config"]["sample_rate"]

    with torch.no_grad(), progress:
        for idx in progress.track(range(len(test_set))):
            if n_src == 1:
                mix, source, mouth, key = test_set[idx]
                mix = mix.to(device)
                source = source.to(device)
                mouth = mouth.unsqueeze(0).unsqueeze(0).to(device)

                est_source = model(mix.unsqueeze(0), mouth)
                est_source = est_source.squeeze(0)

                metrics(
                    mix=mix,
                    clean=source.unsqueeze(0),
                    estimate=est_source,
                    key=key,
                )

                # save separated audio
                save_dir = os.path.join(results_dir, "wavs")
                os.makedirs(save_dir, exist_ok=True)
                torchaudio.save(
                    os.path.join(save_dir, key),
                    est_source.cpu(),
                    sample_rate,
                )
            else:
                mix, sources, mouths, key = test_set[idx]
                mix = mix.to(device)
                sources = sources.to(device)
                mouths = mouths.unsqueeze(0).to(device)

                est_sources = model(mix.unsqueeze(0), mouths)
                est_sources = est_sources.squeeze(0)

                metrics(
                    mix=mix,
                    clean=sources.unsqueeze(0),
                    estimate=est_sources,
                    key=key,
                )

                # save separated audio per speaker
                for spk_idx in range(est_sources.shape[0]):
                    save_dir = os.path.join(results_dir, f"wavs/s{spk_idx + 1}")
                    os.makedirs(save_dir, exist_ok=True)
                    torchaudio.save(
                        os.path.join(save_dir, key),
                        est_sources[spk_idx : spk_idx + 1].cpu(),
                        sample_rate,
                    )

    metrics.final()

    # ---- print summary ----
    mean_metrics = metrics.get_mean()
    std_metrics = metrics.get_std()
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k in mean_metrics:
        print(f"  {k:>10s}: {mean_metrics[k]:.4f} +/- {std_metrics[k]:.4f}")
    print("=" * 60)
    print(f"Results saved to {results_dir}/metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dolphin AVSS model")
    parser.add_argument(
        "--conf_dir",
        default="Experiments/checkpoint/LRS2-final-Dolphin-1gpu-batch4/conf.yml",
        help="Path to the experiment config (conf.yml saved during training).",
    )
    args = parser.parse_args()

    with open(args.conf_dir, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    main(config)
