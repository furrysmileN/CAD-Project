from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from cad_data_gen.qwen3vl_point_finetune import (
    FinetuneConfig,
    QwenPointAdapter,
    apply_lora_if_available,
    build_early_concat_inputs,
    build_finetune_splits,
    build_point_tensor,
    forward_fused_language_model,
    generate_fixed_samples,
    prepare_sample,
    save_finetune_config,
    save_generation_records,
    save_jsonl_records,
    save_reference_samples,
    summarize_train_log,
    tokenize_training_sample,
    write_eval_summary,
)
from cad_data_gen.qwen3vl_point_finetune.dataset import write_issues_jsonl
from cad_data_gen.qwen3vl_point_finetune.modeling import iter_trainable_parameters
from cad_data_gen.qwen3vl_point_finetune.modeling import count_trainable_parameters


ANSWER_FIELD_NAMES = ("object_name", "shape_summary", "notable_features", "modeling_history")


class JsonlLogger:
    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.file = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.file = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        if self.file is None:
            return
        self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class DistributedContext:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.enabled = self.world_size > 1
        self.is_main = self.rank == 0

    def setup(self) -> None:
        if not self.enabled or dist.is_initialized():
            return
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend=backend)

    def barrier(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.barrier()

    def cleanup(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "enabled": self.enabled,
            "is_main": self.is_main,
        }


def print_status(status_label: str, status_text: str, /, **fields: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
    suffix = f" | {payload}" if payload else ""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{status_label}] {status_text}{suffix}", flush=True)


def run_training(config: FinetuneConfig) -> dict[str, Any]:
    dist_ctx = DistributedContext()
    dist_ctx.setup()
    set_seed(config.seed)
    if config.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print_status("startup", "已禁用 cuDNN，CUDA 卷积将使用非 cuDNN 后端")
    output_dir = Path(config.output_dir)
    if dist_ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    print_status(
        "startup",
        "训练任务启动",
        output_dir=output_dir,
        stage=config.stage,
        fusion_mode=config.fusion_mode,
        point_ablation=config.point_ablation,
        max_steps=config.max_steps,
    )
    if dist_ctx.is_main:
        save_finetune_config(config, output_dir / "finetune_config.yaml")
        write_run_command_snapshot(output_dir, config)
    dist_ctx.barrier()

ize)logger = JsonlLogger(output_dir / f"train_log_rank{dist_ctx.rank}.jsonl" if not dist_ctx.is_main else output_dir / "train_log.jsonl", enabled=
dist_ctx.is_main)
    started = time.time()
    try:print_status("data", "开始加载 manifest 并构建训练划分", manifest=co
nfig.resolved_manifest_path)
        splits = build_finetune_splits(config)print_status("data", "数据划分完成", rank=dist_ctx.rank, **splits.s
ummary())
        if dist_ctx.is_main:write_issues_jsonl(splits.issues, output_dir / "data_issues.j
sonl")
            if splits.issues:print_status("data", "发现坏样本，详情已写入 data_issues.jso
nl", issues=len(splits.issues), path=output_dir / "data_issues.jsonl")
            save_reference_samples(splits.sanity, output_dir)
        dist_ctx.barrier()print_status("model", "开始加载 Qwen3-VL processor 和 model", mode
l_path=config.model_path, dtype=config.dtype)
        processor, model = load_qwen_and_processor(config)print_status("model", "Qwen3-VL 加载完成，开始应用 LoRA/冻结策略", en
able_lora=config.enable_lora, freeze_qwen=config.freeze_qwen)
        model = apply_lora_if_available(model, config)
        adapter = QwenPointAdapter(model, config)
        if config.resume_adapter_path:
            adapter.load_adapter(config.resume_adapter_path)print_status("model", "已加载点云 adapter checkpoint", path=co
nfig.resume_adapter_path)
        device = resolve_device(config)print_status("device", "解析训练设备完成，开始迁移模型", device=device, cuda_available=torch.cuda.is_available(), cuda_device_count=torch.cud
a.device_count())
        model.to(device)
        adapter.to(device)print_status("device", "模型和点云 adapter 已迁移到设备", memory=col
lect_memory(device))
        model.train()
        adapter.train()trainable_parameters = _unique_trainable_parameters(adapter, mode
l)optimizer = AdamW(trainable_parameters, lr=config.learning_rate,
weight_decay=config.weight_decay)
        print_status(
            "optimizer",
            "优化器初始化完成",
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            trainable_parameter_tensors=len(trainable_parameters),
        )
        state = {
            "stage": config.stage,
            "dataset": splits.summary(),
            "trainable": {
                "adapter": adapter.trainable_summary().to_dict(),
                "qwen_or_lora": count_trainable_parameters(model),
            },
            "device": str(device),
            "pointcloud": pointcloud_config_summary(config),
            "environment": collect_environment(config),
            "distributed": dist_ctx.to_dict(),
            "started_at": started,
        }
        if dist_ctx.is_main:
            write_json(output_dir / "run_state.json", state)
        logger.write({"type": "start", **state})print_status("state", "运行状态已写出", path=output_dir / "run_stat
e.json", rank=dist_ctx.rank)
        all_train_samples = select_stage_samples(splits.train, config)train_samples = shard_samples_for_rank(all_train_samples, dist_ct
x)train_sanity_samples = all_train_samples[: min(config.sanity_coun
t, len(all_train_samples))]heldout_sanity_samples = (splits.val or splits.sanity)[: min(conf
ig.sanity_count, len(splits.val or splits.sanity))]
        print_status(
            "train",
            "训练样本选择完成",
            selected=len(train_samples),
            selected_global=len(all_train_samples),
            original_train=len(splits.train),
            stage=config.stage,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
        )
        if dist_ctx.is_main and train_sanity_samples:save_reference_samples(train_sanity_samples, output_dir, pref
ix="train_sanity")
        if dist_ctx.is_main and heldout_sanity_samples:save_reference_samples(heldout_sanity_samples, output_dir, pr
efix="heldout_sanity")if dist_ctx.is_main and splits.sanity and config.generate_every
> 0:
ples=min(config.sanity_count, len(splits.sanity)))
            generate_and_save_sanity(
                splits.sanity,
                processor,
                model,
                adapter,
                config,
                device,
                output_dir,
                filename="generations_before.jsonl",
                logger=logger,
                step=0,
            )
        if not train_samples:raise ValueError(f"no training samples available on rank {dist_ctx.rank}; selected_global={len(all_train_samples)} world_size={dist_ct
x.world_size}")max_steps = 1 if config.stage == "single_batch" else int(config.m
ax_steps)print_status("train", "进入训练循环", max_steps=max_steps, gradient_accumulation_steps=config.gradient_accumulation_steps, log_every=config.
log_every)
        global_step = 0
        ema_loss: float | None = None
        optimizer.zero_grad(set_to_none=True)progress = tqdm(total=max_steps, desc=f"qwen-point-finetune:{conf
ig.stage}:rank{dist_ctx.rank}", disable=not dist_ctx.is_main)
        while global_step < max_steps:
            for sample_index, sample in enumerate(train_samples):
                next_step = global_step + 1
                step_started = time.time()replacement_sample = train_samples[(sample_index + 1) % len(train_samples)] if config.point_ablation == "replace" and len(train_sa
mples) > 1 else Noneprint_status("step", "开始处理样本", step=next_step, sample_id=sample.sample_id, replacement_sample_id=getattr(replacement_sample,
"sample_id", None), rank=dist_ctx.rank)
                try:loss, batch_record = run_one_step(sample, processor,
model, adapter, config, device, replacement_sample=replacement_sample)print_status("step", "前向传播完成", step=next_step, sample_id=sample.sample_id, loss=float(loss.detach().cpu()), rank=dist_ctx.
rank)scaled_loss = loss / max(1, config.gradient_accumulat
ion_steps)
                    scaled_loss_value = float(scaled_loss.detach().cpu())
                    scaled_loss.backward()
                    average_gradients(trainable_parameters, dist_ctx)print_status("step", "反向传播完成", step=next_step, sa
mple_id=sample.sample_id, rank=dist_ctx.rank)
                except Exception as exc:print_status("error", "训练 step 失败", step=next_step, sample_id=sample.sample_id, error_type=type(exc).__name__, message=str
(exc), rank=dist_ctx.rank)
                    print(traceback.format_exc(), flush=True)
                    raiseif (global_step + 1) % config.gradient_accumulation_step
s == 0 or config.stage == "single_batch":grad_norm = torch.nn.utils.clip_grad_norm_(trainable_
parameters, config.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)print_status("optimizer", "参数更新完成", step=next_step, grad_norm=float(grad_norm.detach().cpu()) if torch.isfinite(grad_nor
m) else None, rank=dist_ctx.rank)
                else:
                    grad_norm = torch.tensor(float("nan"))print_status("optimizer", "累积梯度中，暂不更新参数", ste
p=next_step, rank=dist_ctx.rank)
                global_step += 1
                raw_loss_value = float(loss.detach().cpu())ema_loss = raw_loss_value if ema_loss is None else 0.9 *
ema_loss + 0.1 * raw_loss_value
                progress.update(1)if global_step % config.log_every == 0 or global_step ==
1:
                    logger.write(
                        {
                            "type": "train_step",
                            "step": global_step,
                            "loss": raw_loss_value,
                            "loss_raw": raw_loss_value,
                            "loss_scaled": scaled_loss_value,
                            "ema_loss": ema_loss,"gradient_accumulation_steps": config.gradien
t_accumulation_steps,"grad_norm": float(grad_norm.detach().cpu())
if torch.isfinite(grad_norm) else None,
                            "lr": optimizer.param_groups[0]["lr"],
                            "memory": collect_memory(device),
                            **batch_record,
                        }
                    )print_status("log", "训练 step 日志已写入", step=global_step, elapsed_seconds=round(time.time() - step_started, 3), memory=colle
ct_memory(device), rank=dist_ctx.rank)if global_step % config.save_every == 0 or global_step =
= max_steps or config.stage == "single_batch":
                    if dist_ctx.is_main:print_status("checkpoint", "开始保存 checkpoint",
step=global_step)save_checkpoint(output_dir, adapter, optimizer, c
onfig, global_step, logger)print_status("checkpoint", "checkpoint 保存完成", step=global_step, checkpoint_dir=output_dir / f"checkpoint-{global_step:0
6d}")
                    dist_ctx.barrier()if dist_ctx.is_main and config.generate_every > 0 and splits.sanity and (global_step % config.generate_every == 0 or global_step =
= max_steps):print_status("generation", "开始训练中 sanity generation", step=global_step, samples=min(config.sanity_count, len(splits.sanit
y)))
                    generate_and_save_sanity(
                        splits.sanity,
                        processor,
                        model,
                        adapter,
                        config,
                        device,
                        output_dir,filename=f"generations_step_{global_step:06d}.jso
nl",
                        logger=logger,
                        step=global_step,
                    )
                if global_step >= max_steps:
                    break
        progress.close()
        if not dist_ctx.is_main:
            result = {
                "status": "ok",
                "steps": global_step,
                "elapsed_seconds": time.time() - started,
                "output_dir": str(output_dir),
                "rank": dist_ctx.rank,
            }print_status("finished", "分布式 worker 训练正常结束", steps=glo
bal_step, rank=dist_ctx.rank)
            return resultprint_status("summary", "训练循环结束，开始汇总 train_log", steps=gl
obal_step)
        eval_summary = summarize_train_log(output_dir / "train_log.json
l")if splits.val:
            print_status("validation", "开始验证集 loss/NLL 评估", samples=
len(splits.val))
            val_summary = evaluate_and_save_loss_metrics(
                splits.val,
                processor,
                model,
                adapter,
                config,
                device,
                output_dir,
                logger=logger,
                prefix="val",
            )
            eval_summary.update(
                {
                    "val_loss": val_summary.get("val_loss"),
                    "val_token_nll": val_summary.get("val_token_nll"),"val_perplexity": val_summary.get("val_perplexity"),
                    "val_field_losses": val_summary.get("field_losses",
{}),
                    "val_summary_path": "val_summary.json",
                    "val_records_path": "val_loss_records.jsonl",
                }
            )if train_sanity_samples:
            print_status("generation", "开始训练后 train sanity generatio
n", samples=len(train_sanity_samples))
            generate_and_save_sanity(
                train_sanity_samples,
                processor,
                model,
                adapter,
                config,
                device,
                output_dir,
                filename="train_sanity_16.jsonl",
                logger=logger,
                step=global_step,
            )if heldout_sanity_samples:
            print_status("generation", "开始训练后 held-out sanity generat
ion", samples=len(heldout_sanity_samples))
            generate_and_save_sanity(
                heldout_sanity_samples,
                processor,
                model,
                adapter,
                config,
                device,
                output_dir,
                filename="heldout_sanity_16.jsonl",
                logger=logger,
                step=global_step,
            )if splits.sanity:
            print_status("generation", "开始训练后 sanity generation", sam
ples=min(config.sanity_count, len(splits.sanity)))
            generate_and_save_sanity(
                splits.sanity,
                processor,
                model,
                adapter,
                config,
                device,
                output_dir,
                filename="generations_after.jsonl",
                logger=logger,
                step=global_step,
            )
        write_eval_summary(output_dir, eval_summary)
        result = {
            "status": "ok",
            "steps": global_step,
            "elapsed_seconds": time.time() - started,
            "output_dir": str(output_dir),
            "eval_summary": eval_summary,
        }logger.write({"type": "finished", **result})
        print_status("finished", "训练任务正常结束", steps=global_step, ela
psed_seconds=round(result["elapsed_seconds"], 3), output_dir=output_dir)
        return resultexcept Exception as exc:
        logger.write({"type": "failed", "error_type": type(exc).__name__, "message": str(exc)})
        print_status("failed", "训练任务异常退出", error_type=type(exc).__n
ame__, message=str(exc), output_dir=output_dir, rank=dist_ctx.rank)
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        logger.close()dist_ctx.cleanup()
        print_status("cleanup", "训练日志文件已关闭", path=output_dir / "tra
in_log.jsonl", rank=dist_ctx.rank)


def run_one_step(
    sample: Any,
    processor: Any,
    model: torch.nn.Module,
    adapter: QwenPointAdapter,
    config: FinetuneConfig,
    device: torch.device,
    *,
    replacement_sample: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    print_status(
        "sample",
        "开始预处理样本",
        sample_id=sample.sample_id,
        fusion_mode=config.fusion_mode,point_ablation=config.point_ablation,
        replacement_sample_id=getattr(replacement_sample, "sample_id", No
ne),)
    prepared = prepare_sample(sample, config, replacement_sample=replacement_sample)print_status("sample", "样本预处理完成", sample_id=sample.sample_id, im
age_count=len(prepared.images), point_tokens_shape=list(prepared.point_tokens.shape))
    print_status("tokenize", "开始 processor/tokenizer 编码", sample_id=sa
mple.sample_id)tokenized = tokenize_training_sample(prepared, processor, config)print_status("tokenize", "编码完成，开始迁移 tensor 到设备", sample_id=sa
mple.sample_id, input_shape=list(tokenized.input_ids.shape), labels_shape
=list(tokenized.labels.shape))
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)
    labels = tokenized.labels.to(device)
    if config.fusion_mode == "image_text":
        print_status("forward", "使用 Qwen 原生视觉前向", sample_id=sample.
sample_id)model_inputs = move_model_inputs(tokenized.model_inputs, device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mas
k, labels=labels, use_cache=False, **model_inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        effective_labels = labels
        effective_attention_mask = attention_mask
        fused_summary: dict[str, Any] = {
            "fusion_mode": "image_text",
            "uses_qwen_native_visual_forward": True,
            "model_input_keys": sorted(model_inputs.keys()),
        }else:
        print_status("fusion", "开始构建点云 tensor 和融合输入", sample_id=s
ample.sample_id)
        point_tensor = build_point_tensor(prepared, device=device)
        fused = build_early_concat_inputs(
            qwen_model=model,
            adapter=adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_tokens=point_tensor,
            labels=labels,
            insert_after=tokenized.insert_after,
            config=config,)
        print_status("forward", "开始融合语言模型前向", sample_id=sample.sam
ple_id, fused_inputs_shape=list(fused.inputs_embeds.shape))
        outputs = forward_fused_language_model(model, fused)loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        effective_labels = fused.labels if fused.labels is not None else
labels
        effective_attention_mask = fused.attention_mask
        fused_summary = fused.shape_summary()if not torch.isfinite(loss):
        raise FloatingPointError(f"loss is not finite for sample {sample.
sample_id}: {loss}")
    return loss, {"sample_id": sample.sample_id,
        "pointcloud_fidelity": summarize_pointcloud_fidelity(prepared, co
nfig),
        "loss_detail": build_loss_detail(
            outputs=outputs,
            loss=loss,
            labels=effective_labels,
            attention_mask=effective_attention_mask,
            answer_start=tokenized.answer_start,original_labels=labels,
            field_token_spans=build_answer_field_token_spans(tokenized, processor, effective_labels),
            field_char_spans=parse_answer_field_char_spans(tokenized.answ
er_text),
        ),
        "tokenized": tokenized.shape_summary(),
        "prepared": prepared.shape_summary(),
        "fused": fused_summary,
    }


def build_loss_detail(
    *,
    outputs: Any,
    loss: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_start: int,
    original_labels: torch.Tensor | None = None,
    field_token_spans: dict[str, tuple[int, int]] | None = None,
    field_char_spans: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    valid_labels = labels != -100
    prompt_valid = valid_labels[:, :answer_start]
    answer_valid = valid_labels[:, answer_start:]
    supervised_token_count = int(valid_labels.sum().item())
    total_token_count = int(labels.numel())
    detail: dict[str, Any] = {
        "model_loss": float(loss.detach().cpu()),
        "batch_size": int(labels.shape[0]),
        "total_token_count": total_token_count,
        "attention_token_count": int(attention_mask.sum().item()),
        "supervised_token_count": supervised_token_count,"tokens_per_loss": supervised_token_count,
        "ignored_token_count": total_token_count - supervised_token_count,
        "label_valid_ratio": supervised_token_count / max(total_token_cou
nt, 1),
        "answer_start": int(answer_start),
        "prompt_supervised_token_count": int(prompt_valid.sum().item()),"answer_supervised_token_count": int(answer_valid.sum().item()),
        "per_sample_supervised_token_count": [int(item) for item in valid
_labels.sum(dim=1).detach().cpu().tolist()],}
    if original_labels is not None and original_labels.shape != labels.sh
ape:original_valid = original_labels != -100
        original_supervised_token_count = int(original_valid.sum().item
())original_total_token_count = int(original_labels.numel())
        original_ignored_token_count = original_total_token_count - origi
nal_supervised_token_count
        detail.update(
            {"original_total_token_count": original_total_token_count,
                "original_supervised_token_count": original_supervised_token_count,

l_token_count,
                "inserted_ignored_token_count": (total_token_count - supe
rvised_token_count) - original_ignored_token_count,
            })
    logits = getattr(outputs, "logits", None)
    if logits is None or supervised_token_count == 0 or logits.shape[:2]
!= labels.shape:
        return detail
    if logits.shape[1] <= 1:
        return detail
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_valid = shift_labels != -100
    if not bool(shift_valid.any().item()):
        return detail
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]).float(),
        shift_labels.clamp_min(0).view(-1),
        reduction="none",
    ).view_as(shift_labels)supervised_losses = token_losses[shift_valid]
    per_sample_loss: list[float | None] = []
    for sample_losses, sample_valid in zip(token_losses, shift_valid, strict=False):
        if bool(sample_valid.any().item()):
            per_sample_loss.append(float(sample_losses[sample_valid].mean
().detach().cpu()))
        else:
            per_sample_loss.append(None)
    ce_mean = float(supervised_losses.mean().detach().cpu())detail.update(
        {
            "shifted_supervised_token_count": int(shift_valid.sum().item()),
            "per_token_cross_entropy_mean": ce_mean,"per_token_cross_entropy_sum": float(supervised_losses.sum().
detach().cpu()),"per_token_cross_entropy_min": float(supervised_losses.min().
detach().cpu()),
            "per_token_cross_entropy_max": float(supervised_losses.max().
detach().cpu()),
            "per_sample_loss": per_sample_loss,
            "loss_ce_mean_delta": float(loss.detach().cpu()) - ce_mean,
        })
    if field_token_spans:
        detail["field_losses"] = summarize_field_losses(token_losses, shift_valid, field_token_spans)
    if field_char_spans:
        detail["field_char_spans"] = {field: [int(start), int(end)] for f
ield, (start, end) in field_char_spans.items()}
    return detail
def parse_answer_field_char_spans(answer_text: str) -> dict[str, tuple[in
t, int]]:
    pattern = re.compile(rf"(?m)^({'|'.join(re.escape(field) for field i
n ANSWER_FIELD_NAMES)})\s*:\s*")
    matches = list(pattern.finditer(answer_text))
    spans: dict[str, tuple[int, int]] = {}
    for index, match in enumerate(matches):field = match.group(1)
        value_start = match.end()value_end = matches[index + 1].start() if index + 1 < len(matche
s) else len(answer_text)
        while value_end > value_start and answer_text[value_end - 1].issp
ace():
            value_end -= 1
        spans[field] = (value_start, value_end)
    return spans

def build_answer_field_token_spans(tokenized: Any, processor: Any, label
s: torch.Tensor) -> dict[str, tuple[int, int]]:
    char_spans = parse_answer_field_char_spans(tokenized.answer_text)if not char_spans:
        return {}
    inserted_token_count = max(0, int(labels.shape[1]) - int(tokenized.la
bels.shape[1]))spans: dict[str, tuple[int, int]] = {}
    for field, (char_start, char_end) in char_spans.items():answer_start_offset = count_text_tokens(processor, tokenized.answ
er_text[:char_start])
        answer_end_offset = count_text_tokens(processor, tokenized.answer_text[:char_end])
        start = int(tokenized.answer_start) + answer_start_offset
        end = int(tokenized.answer_start) + max(answer_end_offset, answer
_start_offset + 1)
        if inserted_token_count and start >= int(tokenized.insert_after):
            start += inserted_token_count
            end += inserted_token_count
        start = max(0, min(start, int(labels.shape[1])))
        end = max(start, min(end, int(labels.shape[1])))
        if end > start:
            spans[field] = (start, end)
    return spans


def count_text_tokens(processor: Any, text: str) -> int:tokenizer = getattr(processor, "tokenizer", processor)
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_tensor
s="pt")
    except TypeError:
        try:
