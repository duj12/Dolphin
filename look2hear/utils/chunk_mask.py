"""
Chunk mask generation utilities for streaming-aware training.

Generates two types of masks from chunk parameters (chunk_size, history_len, future_len):
1. Feature-level binary mask: zeros out invisible positions to prevent conv leakage
2. Attention-level mask: restricts self-attention to the visible window

All chunk sizes are specified at waveform level (samples at 16kHz) and internally
converted to feature level by dividing by audio_encoder_stride.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import random
import torch
import torch.nn.functional as F


@dataclass
class ChunkMaskConfig:
    chunk_size: int        # in waveform samples
    history_len: int       # in waveform samples
    future_len: int        # in waveform samples
    audio_encoder_stride: int = 4
    num_separator_stages: int = 4
    chunk_start: int = -1  # in waveform samples; -1 means random


def _waveform_to_feat(samples: int, stride: int) -> int:
    return samples // stride


def generate_chunk_masks(
    T_feat: int,
    device: torch.device,
    config: ChunkMaskConfig,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Generate all masks needed for one forward pass.

    Args:
        T_feat: temporal length at feature level (after audio encoder, before separator)
        device: torch device
        config: chunk mask parameters

    Returns:
        feat_mask: [1, 1, T_feat] float, 1=visible 0=masked
        attn_mask: [1, 1, T_attn, T_attn] bool, True=blocked
        feat_masks_per_stage: list of [1, 1, T_i] for each encoder stage resolution
    """
    stride = config.audio_encoder_stride
    stages = config.num_separator_stages

    chunk_size_feat = _waveform_to_feat(config.chunk_size, stride)
    history_feat = _waveform_to_feat(config.history_len, stride)
    future_feat = _waveform_to_feat(config.future_len, stride)

    # Determine chunk start position in feature frames
    if config.chunk_start < 0:
        # Random start, ensure visible window fits or clips gracefully
        chunk_start_feat = random.randint(0, max(0, T_feat - chunk_size_feat))
    else:
        chunk_start_feat = _waveform_to_feat(config.chunk_start, stride)

    chunk_end_feat = min(chunk_start_feat + chunk_size_feat, T_feat)

    # Visible window boundaries (clipped to sequence)
    vis_start = max(0, chunk_start_feat - history_feat)
    vis_end = min(T_feat, chunk_end_feat + future_feat)

    # 1) Feature mask at full resolution
    feat_mask = torch.zeros(1, 1, T_feat, device=device)
    feat_mask[:, :, vis_start:vis_end] = 1.0

    # 2) Attention mask at pos_k resolution (used by DU_MHSA)
    #    All DU_MHSA layers downsample to pos_k_len = T_feat / 2^stages
    T_attn = T_feat // (2 ** stages)
    if T_attn < 1:
        T_attn = 1

    vis_start_attn = max(0, vis_start // (2 ** stages))
    vis_end_attn = min(T_attn, (vis_end + (2 ** stages) - 1) // (2 ** stages))

    # True = masked (cannot attend)
    attn_mask = torch.ones(1, 1, T_attn, T_attn, dtype=torch.bool, device=device)
    attn_mask[:, :, vis_start_attn:vis_end_attn, vis_start_attn:vis_end_attn] = False

    # 3) Feature masks per encoder stage resolution
    #    Stage i operates at T_feat / 2^i (after DownConv halving)
    #    We provide (stages) masks: stage 0 at T_feat, stage 1 at T_feat/2, etc.
    feat_masks_per_stage = []
    for i in range(stages):
        T_i = T_feat // (2 ** i)
        if T_i < 1:
            T_i = 1
        vis_start_i = max(0, vis_start // (2 ** i))
        vis_end_i = min(T_i, (vis_end + (2 ** i) - 1) // (2 ** i))
        mask_i = torch.zeros(1, 1, T_i, device=device)
        mask_i[:, :, vis_start_i:vis_end_i] = 1.0
        feat_masks_per_stage.append(mask_i)

    return feat_mask, attn_mask, feat_masks_per_stage


def sample_chunk_config(
    streaming_config: dict,
    audio_encoder_stride: int = 4,
    num_separator_stages: int = 4,
    T_feat: Optional[int] = None,
) -> ChunkMaskConfig:
    """Randomly sample chunk parameters from configured ranges.

    Args:
        streaming_config: the "streaming" section of the training config
        audio_encoder_stride: audio encoder stride (default 4)
        num_separator_stages: number of separator stages (default 4)
        T_feat: if provided, ensures visible window doesn't exceed sequence length

    Returns:
        ChunkMaskConfig with randomly sampled parameters
    """
    chunk_range = streaming_config.get("chunk_size_range", [3200, 8000])
    hist_range = streaming_config.get("history_len_range", [1600, 6400])
    fut_range = streaming_config.get("future_len_range", [0, 3200])

    chunk_size = random.randint(chunk_range[0], chunk_range[1])
    history_len = random.randint(hist_range[0], hist_range[1])
    future_len = random.randint(fut_range[0], fut_range[1])

    # If T_feat is known, ensure the total visible window fits
    if T_feat is not None:
        stride = audio_encoder_stride
        max_visible = T_feat * stride  # convert feat samples back to waveform
        total = chunk_size + history_len + future_len
        if total > max_visible:
            # shrink proportionally
            scale = max_visible / total
            chunk_size = int(chunk_size * scale)
            history_len = int(history_len * scale)
            future_len = int(future_len * scale)
            chunk_size = max(chunk_size, stride)  # at least 1 feature frame

    return ChunkMaskConfig(
        chunk_size=chunk_size,
        history_len=history_len,
        future_len=future_len,
        audio_encoder_stride=audio_encoder_stride,
        num_separator_stages=num_separator_stages,
        chunk_start=-1,  # random
    )
