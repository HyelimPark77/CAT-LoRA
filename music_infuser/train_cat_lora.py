#!/usr/bin/env python3
"""CAT-LoRA training entrypoint for MusicInfuser.

This file intentionally leaves ``music_infuser/train.py`` unchanged. It reuses
MusicInfuser's dataset/model-loading path and adds counterfactual
audio-temporal losses on precomputed Wav2Vec audio features.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import time
from contextlib import contextmanager, nullcontext
from glob import glob
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import click
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from safetensors.torch import save_file
from torch import Tensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from tqdm import tqdm

import genmo.mochi_preview.dit.joint_model.audio_adapter as audio
import genmo.mochi_preview.dit.joint_model.lora as lora

from dataset import MusicInfuserDataset
from genmo.mochi_preview.pipelines import DitModelFactory, cast_dit, compute_packed_indices, load_to_cpu
from genmo.mochi_preview.vae.latent_dist import LatentDistribution
from genmo.mochi_preview.vae.vae_stats import vae_latents_to_dit_latents
from cat_lora.audio_feature_perturbations import make_feature_counterfactual
from cat_lora.torch_losses import (
    corr_loss,
    delta_response_loss,
    latent_motion_curve,
    motion_smoothness_loss,
    peak_kl_loss,
    resample_curve,
    silence_suppression_loss,
    temporal_mean_preserve_loss,
)


torch._dynamo.config.cache_size_limit = 32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.use_deterministic_algorithms(False)


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def map_to_device(x, device: torch.device):
    if isinstance(x, dict):
        return {k: map_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [map_to_device(y, device) for y in x]
    if isinstance(x, tuple):
        return tuple(map_to_device(y, device) for y in x)
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x


@contextmanager
def timer(description: str = "Task", enabled: bool = True):
    if enabled:
        start = time.perf_counter()
    try:
        yield
    finally:
        if enabled:
            elapsed = time.perf_counter() - start
            print(f"{description} took {elapsed:.4f} seconds")


def infinite_dl(dl):
    epoch_idx = 0
    while True:
        epoch_idx += 1
        for batch in dl:
            yield epoch_idx, batch


def get_cosine_annealing_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def estimate_clean_latent(z_sigma: Tensor, sigma: Tensor, pred_ut_dit: Tensor) -> Tensor:
    z_sigma_dit = vae_latents_to_dit_latents(z_sigma.float())
    sigma_view = sigma[:, None, None, None, None].to(device=pred_ut_dit.device, dtype=pred_ut_dit.dtype)
    return z_sigma_dit + sigma_view * pred_ut_dit.float()


def build_model(cfg: DictConfig, checkpoint_path: Path, device_id: int):
    audio_mode = getattr(cfg, "audio_mode", "cross_attn")
    audio_cross_attn_layers = getattr(
        cfg,
        "audio_cross_attn_layers",
        [6, 7, 8, 9, 10, 21, 34, 35, 36, 38, 39, 43, 44, 45, 46, 47],
    )
    is_audio = audio_mode is not None
    is_lora = "lora" in cfg.model.type
    is_full = "full" in cfg.model.type
    trainable_scope = str(cfg.model.get("trainable_scope", "audio_lora" if is_audio else "lora_only"))
    pretrained_adapter_path = cfg.model.get("pretrained_adapter_path")

    patch_model_fns = []
    model_kwargs = {}
    if is_lora:
        if trainable_scope == "audio_lora":
            def mark_lora_params(m):
                audio.mark_audio_and_lora_as_trainable(m, bias="none")
                return m
        elif trainable_scope == "lora_only":
            def mark_lora_params(m):
                lora.mark_only_lora_as_trainable(m, bias="none")
                return m
        else:
            raise ValueError(f"Unsupported trainable_scope for LoRA model: {trainable_scope}")
        patch_model_fns.append(mark_lora_params)
        model_kwargs = dict(**cfg.model.kwargs)
        for k, v in model_kwargs.items():
            if isinstance(v, ListConfig):
                model_kwargs[k] = list(v)
    elif is_audio and not is_full:
        def mark_lora_params(m):
            audio.mark_only_audio_as_trainable(m, bias="none")
            return m
        patch_model_fns.append(mark_lora_params)

    if cfg.training.get("model_dtype"):
        assert cfg.training.model_dtype == "bf16", "Only bf16 is supported"
        patch_model_fns.append(lambda m: cast_dit(m, torch.bfloat16))

    model = DitModelFactory(
        model_path=str(checkpoint_path),
        lora_path=str(pretrained_adapter_path) if pretrained_adapter_path else None,
        model_dtype="bf16",
        attention_mode=cfg.attention_mode,
        audio_mode=audio_mode,
        audio_cross_attn_layers=audio_cross_attn_layers,
    ).get_model(
        local_rank=0,
        device_id=device_id,
        model_kwargs=model_kwargs,
        patch_model_fns=patch_model_fns,
        world_size=1,
        strict_load=not is_lora and not is_audio,
        fast_init=not is_lora and not is_audio,
    ).train()
    return model, model_kwargs, is_lora, is_full, is_audio, trainable_scope


def get_train_loader(cfg: DictConfig):
    train_vids = list(sorted(glob(f"{cfg.train_data_dir}/*.mp4")))
    train_vids = [v for v in train_vids if not v.endswith(".recon.mp4")]
    assert train_vids, f"No training data found in {cfg.train_data_dir}"
    if cfg.single_video_mode:
        train_vids = train_vids[:1]

    train_vids_2 = list(sorted(glob(f"{cfg.train_data_dir_2}/*.mp4")))
    train_vids_2 = [v for v in train_vids_2 if not v.endswith(".recon.mp4")]

    dataset = MusicInfuserDataset(
        train_vids,
        train_vids_2,
        cfg.train_data_dir_2_ratio,
        basic_prompt_ratio=getattr(cfg, "basic_prompt_ratio", 1.0),
        repeat=1_000 if cfg.single_video_mode else 1,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=int(cfg.training.get("num_workers", 4)),
        shuffle=True,
        pin_memory=True,
    )


def save_adapter_checkpoint(
    *,
    model,
    save_path: Path,
    model_kwargs: dict,
    is_lora: bool,
    is_full: bool,
    is_audio: bool,
    trainable_scope: str,
) -> None:
    if is_lora:
        model_sd = lora.lora_state_dict(model, bias="none")
    elif is_full:
        model_sd, _optimizer_sd = get_state_dict(
            model, [], options=StateDictOptions(cpu_offload=True, full_state_dict=True)
        )
    else:
        model_sd = {}

    if is_audio and trainable_scope in {"audio_lora", "audio_only"}:
        model_sd.update(model.audio_projection.get_state_dict())
        if model.audio_cross_attn_blocks:
            audio_ca_sd = model.audio_cross_attn_blocks.state_dict()
            model_sd.update({f"audio_cross_attn_blocks.{k}": v for k, v in audio_ca_sd.items()})

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(model_sd, save_path, metadata=dict(kwargs=json.dumps(model_kwargs)))


@click.command()
@click.option("--config-path", type=click.Path(exists=True), required=True)
def main(config_path: str) -> None:
    cfg = cast(DictConfig, OmegaConf.load(config_path))
    set_seed(int(getattr(cfg, "seed", 42)))

    device_id = 0
    device = torch.device("cuda:0")
    checkpoint_path = Path(cfg.init_checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint file not found: {checkpoint_path}"
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pattern = r"model_(\d+)\.(lora|checkpoint|adapter)\.(safetensors|pt)"
    match = re.search(pattern, str(checkpoint_path))
    start_step_num = int(match.group(1)) if match else 0

    print(f"CAT-LoRA training from checkpoint={checkpoint_path}, start_step={start_step_num}")
    print(f"MOCHI_T5_MODEL={os.environ.get('MOCHI_T5_MODEL', '<default google/t5-v1_1-xxl>')}")

    train_dl_iter = infinite_dl(get_train_loader(cfg))
    model, model_kwargs, is_lora, is_full, is_audio, trainable_scope = build_model(cfg, checkpoint_path, device_id)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {trainable_param_count:,} / {total_param_count:,}")
    print(f"Trainable scope: {trainable_scope}")
    optimizer = torch.optim.AdamW(trainable_params, **cfg.optimizer)
    scheduler = get_cosine_annealing_lr_scheduler(
        optimizer,
        warmup_steps=int(cfg.training.warmup_steps),
        total_steps=int(cfg.training.num_steps),
    )

    beta_beta = getattr(cfg, "beta_beta", 3)
    beta_half = getattr(cfg, "beta_half", 200)
    cat_cfg = cfg.get("cat_lora", {})
    loss_w = cat_cfg.get("loss_weights", {})
    use_counterfactual = bool(cat_cfg.get("enabled", True))
    dry_run_no_backward = bool(cfg.training.get("dry_run_no_backward", False))

    pbar = tqdm(range(start_step_num, int(cfg.training.num_steps)), total=int(cfg.training.num_steps), initial=start_step_num)
    for step in pbar:
        with torch.no_grad(), timer("load_batch", enabled=False):
            epoch_idx, batch = next(train_dl_iter)
            latent, embed, audio_embed = cast(Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]], batch)
            embed, audio_embed = map_to_device((embed, audio_embed), device)
            embed = cast(Dict[str, Any], embed)
            audio_embed = cast(Dict[str, Tensor], audio_embed)

            ldist = LatentDistribution(latent["mean"], latent["logvar"])
            z = ldist.sample().to(device)
            eps = torch.randn_like(z)
            if beta_beta is None:
                sigma = torch.rand(z.shape[:1], device=device, dtype=torch.float32)
            else:
                beta = 1 + (beta_beta - 1) * math.exp(-(step / beta_half) * math.log(2))
                sigma = torch.distributions.Beta(torch.ones(z.shape[:1], device=device), beta).sample()

            if random.random() < cfg.training.caption_dropout:
                embed["y_mask"][0].zero_()
                embed["y_feat"][0].zero_()

            num_latent_toks = np.prod(z.shape[-3:])
            indices = compute_packed_indices(device, cast(Tensor, embed["y_mask"][0]), int(num_latent_toks))
            sigma_bcthw = sigma[:, None, None, None, None]
            z_sigma = (1 - sigma_bcthw) * z + sigma_bcthw * eps
            ut = z - eps

        forward_context = torch.no_grad() if dry_run_no_backward else nullcontext()
        with forward_context, torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(
                x=z_sigma,
                sigma=sigma,
                packed_indices=indices,
                audio_feat=audio_embed["audio_embeddings"],
                **embed,
                num_ff_checkpoint=cfg.training.num_ff_checkpoint,
                num_qkv_checkpoint=cfg.training.num_qkv_checkpoint,
            )
            assert preds.shape == z.shape

        ut_dit_space = vae_latents_to_dit_latents(ut.float())
        diffusion_loss = F.mse_loss(preds.float(), ut_dit_space)
        total_loss = diffusion_loss
        log_kwargs = {
            "train/loss_diffusion": diffusion_loss.item(),
            "train/epoch": epoch_idx,
            "train/lr": scheduler.get_last_lr()[0],
        }

        if use_counterfactual:
            cf = make_feature_counterfactual(
                audio_embed["audio_embeddings"],
                local_silence_prob=float(cat_cfg.get("local_silence_prob", 0.35)),
                shift_prob=float(cat_cfg.get("shift_prob", 0.30)),
                tempo_prob=float(cat_cfg.get("tempo_prob", 0.25)),
                global_silence_prob=float(cat_cfg.get("global_silence_prob", 0.10)),
            )
            forward_context = torch.no_grad() if dry_run_no_backward else nullcontext()
            with forward_context, torch.autocast("cuda", dtype=torch.bfloat16):
                cf_preds = model(
                    x=z_sigma,
                    sigma=sigma,
                    packed_indices=indices,
                    audio_feat=cf.audio_embeddings,
                    **embed,
                    num_ff_checkpoint=cfg.training.num_ff_checkpoint,
                    num_qkv_checkpoint=cfg.training.num_qkv_checkpoint,
                )

            orig_clean = estimate_clean_latent(z_sigma, sigma, preds)
            cf_clean = estimate_clean_latent(z_sigma, sigma, cf_preds)
            orig_motion = latent_motion_curve(orig_clean)
            cf_motion = latent_motion_curve(cf_clean)
            orig_curve = resample_curve(cf.original_curve.to(device), orig_motion.shape[-1])
            cf_curve = resample_curve(cf.counterfactual_curve.to(device), cf_motion.shape[-1])
            silence_mask = resample_curve(cf.silence_mask.to(device), cf_motion.shape[-1])

            align_loss = corr_loss(orig_curve, orig_motion)
            peak_loss = peak_kl_loss(orig_curve, orig_motion)
            delta_loss = delta_response_loss(orig_curve, cf_curve, orig_motion, cf_motion)
            silence_loss = silence_suppression_loss(orig_motion, cf_motion, silence_mask)
            preserve_loss = temporal_mean_preserve_loss(cf_clean, orig_clean)
            smooth_loss = motion_smoothness_loss(cf_motion)

            total_loss = total_loss + float(loss_w.get("audio_motion_corr", 0.10)) * align_loss
            total_loss = total_loss + float(loss_w.get("peak_distribution", 0.05)) * peak_loss
            total_loss = total_loss + float(loss_w.get("counterfactual_delta", 0.20)) * delta_loss
            total_loss = total_loss + float(loss_w.get("silence_suppression", 0.10)) * silence_loss
            total_loss = total_loss + float(loss_w.get("temporal_mean_preserve", 0.50)) * preserve_loss
            total_loss = total_loss + float(loss_w.get("motion_smoothness", 0.01)) * smooth_loss

            log_kwargs.update(
                {
                    "train/cf_kind": cf.kind,
                    "train/loss_align": align_loss.item(),
                    "train/loss_peak": peak_loss.item(),
                    "train/loss_delta": delta_loss.item(),
                    "train/loss_silence": silence_loss.item(),
                    "train/loss_preserve": preserve_loss.item(),
                    "train/loss_smooth": smooth_loss.item(),
                }
            )

        if dry_run_no_backward:
            print("Dry run complete:", {k: v for k, v in log_kwargs.items() if k != "train/cf_kind"})
            return

        total_loss.backward()
        if cfg.training.get("grad_clip"):
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.training.grad_clip))
            log_kwargs["train/gnorm"] = gnorm.item()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        clear_cuda_cache()

        log_kwargs["train/loss_total"] = total_loss.item()
        pbar.set_postfix(**log_kwargs)

        if cfg.training.save_interval and step > 0 and (step + 1) % cfg.training.save_interval == 0:
            save_path = checkpoint_dir / f"model_{step+1}.cat_lora.safetensors"
            with timer("save_adapter_checkpoint"):
                save_adapter_checkpoint(
                    model=model,
                    save_path=save_path,
                    model_kwargs=model_kwargs,
                    is_lora=is_lora,
                    is_full=is_full,
                    is_audio=is_audio,
                    trainable_scope=trainable_scope,
                )


if __name__ == "__main__":
    main()
