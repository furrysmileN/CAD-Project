import argparse
import copy
import inspect
import json
import logging
import math
import os
import random
import shutil
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor, Trainer, TrainingArguments, TrainerCallback, set_seed

from qwen3vl_data import BenchCADIndexedDataset, build_pointcloud_adapter, collate_qwen3vl

try:
    import yaml
except Exception:
    yaml = None


LOGGER = logging.getLogger(__name__)


class JsonlFileLogCallback(TrainerCallback):
    def __init__(self, log_file: str):
        self.log_file = log_file

    def on_init_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        payload = {
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        for key, value in (logs or {}).items():
            if isinstance(value, (int, float)):
                payload[key] = float(value)
            else:
                payload[key] = value

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class NaNLossStopCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if isinstance(loss, (int, float)) and (math.isnan(loss) or math.isinf(loss)):
            LOGGER.error("检测到异常 loss（NaN/Inf），训练即将停止。")
            control.should_training_stop = True


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        if "." in text:
            return float(text)
        return int(text)
    except Exception:
        return text


def _set_nested(cfg: Dict[str, Any], key_path: str, value: Any):
    keys = key_path.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _load_config(path: str) -> Dict[str, Any]:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"配置文件不存在: {abs_path}")

    if abs_path.endswith((".yml", ".yaml")):
        if yaml is None:
            raise ImportError("当前环境缺少 PyYAML，请先安装 pyyaml")
        with open(abs_path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
    elif abs_path.endswith(".json"):
        with open(abs_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        raise ValueError(f"仅支持 .yaml/.yml/.json 配置文件: {abs_path}")

    if not isinstance(obj, dict):
        raise ValueError("配置文件顶层必须是字典对象")
    return obj


def _save_config(path: str, cfg: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith((".yaml", ".yml")) and yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)


def _resolve_path(path_value: Optional[str], base_dir: str) -> Optional[str]:
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def _ensure_dir_writable(path_value: str):
    abs_path = os.path.abspath(path_value)
    os.makedirs(abs_path, exist_ok=True)
    test_file = os.path.join(abs_path, ".write_test")
    try:
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def _torch_dtype_from_name(name: str):
    key = (name or "bfloat16").lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "auto": "auto",
    }
    if key not in mapping:
        raise ValueError(f"不支持的 dtype: {name}")
    return mapping[key]


def _first_existing_glob(directory: str, patterns: List[str]) -> Optional[str]:
    import glob

    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        matches = [m for m in matches if os.path.isfile(m)]
        if matches:
            return matches[0]
    return None


def _link_or_copy(src: str, dst: str):
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.basename(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _normalize_local_model_files(model_path: str) -> List[str]:
    """Make browser-downloaded Qwen directories loadable by transformers.

    Some local model folders contain names such as `config (3).json`. Transformers
    only looks for canonical names, so we create lightweight symlinks when needed.
    """
    if not os.path.isdir(model_path):
        return []

    aliases = {
        "config.json": ["config*.json"],
        "tokenizer.json": ["tokenizer*.json"],
        "tokenizer_config.json": ["tokenizer_config*.json"],
        "generation_config.json": ["generation_config*.json"],
    }
    created = []
    for canonical, patterns in aliases.items():
        dst = os.path.join(model_path, canonical)
        if os.path.exists(dst):
            continue
        src = _first_existing_glob(model_path, patterns)
        if src is None or os.path.basename(src) == canonical:
            continue
        _link_or_copy(src, dst)
        created.append(canonical)
    return created


def _check_model_weights(model_path: str):
    weight_patterns = [
        "*.safetensors",
        "*.bin",
        "model*.index.json",
        "pytorch_model*.index.json",
    ]
    if _first_existing_glob(model_path, weight_patterns) is None:
        raise FileNotFoundError(
            "本地模型目录缺少权重文件（*.safetensors / *.bin / index.json）。"
            f" 当前路径: {model_path}。请重新下载完整的 Qwen3-VL-2B-Instruct 权重后再训练。"
        )


def _infer_model_type(model_path: str, trust_remote_code: bool) -> str:
    _normalize_local_model_files(model_path)
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model_type = getattr(cfg, "model_type", "")
        if model_type:
            return str(model_type)
    except Exception as exc:
        LOGGER.warning("AutoConfig 解析 model_type 失败，将回退读取 config.json。错误: %s", exc)

    local_config_path = os.path.join(model_path, "config.json")
    if os.path.exists(local_config_path):
        try:
            with open(local_config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            model_type = raw.get("model_type", "")
            if model_type:
                return str(model_type)
        except Exception as exc:
            LOGGER.warning("读取本地 config.json 失败: %s", exc)

    return ""


def _load_vision_text_model(
    model_path: str,
    model_kwargs: Dict[str, Any],
    point_encoder_cfg: Optional[Dict[str, Any]] = None,
):
    import transformers

    _check_model_weights(model_path)
    model_type = _infer_model_type(
        model_path,
        trust_remote_code=bool(model_kwargs.get("trust_remote_code", False)),
    )

    if point_encoder_cfg and point_encoder_cfg.get("type") == "point_transformer_v3":
        if model_type != "qwen3_vl":
            raise ValueError(f"PointTransformerV3 encoder 仅支持 qwen3_vl，当前 model_type={model_type}")
        from cadrille_qwen3_ptv3 import (
            Qwen3VLWithPointTransformerV3,
            initialize_point_encoder_weights,
            reset_point_transformer_v3_luts,
        )

        local_kwargs = copy.deepcopy(model_kwargs)
        local_kwargs["point_encoder_config"] = point_encoder_cfg
        try:
            LOGGER.info("使用 Qwen3VLWithPointTransformerV3 加载模型。")
            model = Qwen3VLWithPointTransformerV3.from_pretrained(model_path, **local_kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if local_kwargs.get("attn_implementation") == "flash_attention_2" and (
                "flash_attn" in msg or "flashattention2" in msg
            ):
                LOGGER.warning("检测到 flash-attn 不可用，降级为 attn_implementation='eager' 后重试。")
                local_kwargs["attn_implementation"] = "eager"
                model = Qwen3VLWithPointTransformerV3.from_pretrained(model_path, **local_kwargs)
            else:
                raise

        if hasattr(model, "point_encoder"):
            model.point_encoder.float()
            initialize_point_encoder_weights(model.point_encoder)
        reset_point_transformer_v3_luts()
        return model

    if model_type == "qwen3_vl":
        class_names = [
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ]
    elif model_type == "qwen2_vl":
        class_names = [
            "Qwen2VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
        ]
    else:
        class_names = [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "Qwen3VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
        ]

    last_error = None
    attempts: List[str] = []
    for cls_name in class_names:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            attempts.append(f"{cls_name}:class_not_found")
            continue

        local_kwargs = copy.deepcopy(model_kwargs)
        try:
            LOGGER.info("尝试使用 %s 加载模型（model_type=%s）", cls_name, model_type)
            return cls.from_pretrained(model_path, **local_kwargs)
        except Exception as exc:
            last_error = exc
            attempts.append(f"{cls_name}:{type(exc).__name__}")
            msg = str(exc).lower()

            if local_kwargs.get("attn_implementation") == "flash_attention_2" and (
                "flash_attn" in msg or "flashattention2" in msg
            ):
                LOGGER.warning("检测到 flash-attn 不可用，降级为 attn_implementation='eager' 后重试。")
                local_kwargs["attn_implementation"] = "eager"
                try:
                    return cls.from_pretrained(model_path, **local_kwargs)
                except Exception as exc2:
                    last_error = exc2
                    attempts.append(f"{cls_name}_fallback_eager:{type(exc2).__name__}")

    raise RuntimeError(
        "无法加载视觉语言模型，请检查 transformers 版本与模型兼容性。"
        f" model_type={model_type}, 尝试序列={class_names}, 尝试结果={attempts}, 最后错误: {last_error}"
    )


def _maybe_apply_lora(model, cfg: Dict[str, Any]):
    train_mode = cfg["training"].get("train_mode", "lora").lower()
    if train_mode != "lora":
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise ImportError("启用 LoRA 训练需要安装 peft") from exc

    lora_cfg = cfg.get("lora", {})
    modules_to_save = lora_cfg.get("modules_to_save")
    if modules_to_save is None and cfg.get("point_encoder", {}).get("type") == "point_transformer_v3":
        modules_to_save = ["point_encoder"]
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        modules_to_save=modules_to_save,
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        ),
    )
    model = get_peft_model(model, peft_cfg)
    if cfg.get("point_encoder", {}).get("type") == "point_transformer_v3":
        for name, param in model.named_parameters():
            if "point_encoder" in name:
                param.requires_grad_(True)
        LOGGER.info("已在 LoRA 模式下保持 point_encoder 参数可训练。")
    model.print_trainable_parameters()
    return model


def _patch_accelerator_unwrap_model_for_compat(accelerator):
    unwrap_fn = getattr(accelerator, "unwrap_model", None)
    if unwrap_fn is None:
        return

    try:
        params = inspect.signature(unwrap_fn).parameters
    except Exception:
        return

    if "keep_torch_compile" in params:
        return

    LOGGER.warning("检测到旧版 accelerate.unwrap_model 签名，应用 keep_torch_compile 兼容补丁。")

    def _compat_unwrap_model(model, *args, **kwargs):
        kwargs.pop("keep_torch_compile", None)
        return unwrap_fn(model, *args, **kwargs)

    accelerator.unwrap_model = _compat_unwrap_model


def _count_parameters(model) -> Dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"total": int(total), "trainable": int(trainable)}


def _enable_input_require_grads_for_checkpointing(model):
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
            LOGGER.info("已通过 enable_input_require_grads 开启输入梯度。")
            return
        except Exception as exc:
            LOGGER.warning("enable_input_require_grads 调用失败，尝试 hook 方式。错误: %s", exc)

    emb = None
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
        except Exception:
            emb = None
    if emb is None and hasattr(model, "base_model") and hasattr(model.base_model, "get_input_embeddings"):
        try:
            emb = model.base_model.get_input_embeddings()
        except Exception:
            emb = None
    if emb is None:
        LOGGER.warning("未找到输入 embedding 层，无法安装输入梯度 hook。")
        return

    def _make_inputs_require_grad(module, inputs, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
        elif isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    item.requires_grad_(True)

    emb.register_forward_hook(_make_inputs_require_grad)
    LOGGER.info("已安装输入 embedding 前向 hook 以开启输入梯度。")


def _build_run_dir(cfg: Dict[str, Any]) -> str:
    root = os.path.abspath(cfg["paths"]["output_root"])
    run_name = cfg["experiment"].get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("qwen3vl_%Y%m%d_%H%M%S")
        cfg["experiment"]["run_name"] = run_name
    run_dir = os.path.join(root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _build_datasets(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    adapter = build_pointcloud_adapter(
        name=data_cfg.get("pointcloud_adapter", "serialize"),
        num_points=int(data_cfg.get("pointcloud_num_points", 256)),
        precision=int(data_cfg.get("pointcloud_precision", 4)),
    )

    train_ds = BenchCADIndexedDataset(
        index_path=data_cfg["index_path"],
        split=data_cfg.get("train_split", "train"),
        prompt=data_cfg.get("prompt", "Generate cadquery code"),
        pointcloud_adapter=adapter,
        max_images=int(data_cfg.get("max_images", 4)),
    )
    if len(train_ds) == 0:
        raise ValueError(f"训练集为空，请检查 index_path 和 split: {data_cfg['index_path']}")

    eval_ds = None
    val_split = data_cfg.get("val_split")
    if val_split:
        candidate = BenchCADIndexedDataset(
            index_path=data_cfg["index_path"],
            split=val_split,
            prompt=data_cfg.get("prompt", "Generate cadquery code"),
            pointcloud_adapter=adapter,
            max_images=int(data_cfg.get("max_images", 4)),
        )
        if len(candidate) > 0:
            eval_ds = candidate

    return train_ds, eval_ds


def _setup_logging(run_dir: str, level: str):
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    log_file = os.path.join(run_dir, "logs", "train.log")

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def _validate_and_normalize_config(cfg: Dict[str, Any], cfg_path: str) -> Dict[str, Any]:
    required_sections = ["paths", "data", "model", "training", "experiment"]
    for section in required_sections:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"配置缺少必要字段: {section}")

    base_dir = os.path.dirname(os.path.abspath(cfg_path))
    cfg["paths"]["output_root"] = _resolve_path(cfg["paths"].get("output_root", "./work_dirs"), base_dir)
    cfg["data"]["index_path"] = _resolve_path(cfg["data"].get("index_path"), base_dir)
    cfg["model"]["model_path"] = _resolve_path(cfg["model"].get("model_path"), base_dir)

    if not cfg["data"].get("index_path") or not os.path.exists(cfg["data"]["index_path"]):
        raise FileNotFoundError(
            f"索引文件不存在: {cfg['data'].get('index_path')}。请先运行 build_index.py 生成索引。"
        )
    if not cfg["model"].get("model_path") or not os.path.exists(cfg["model"]["model_path"]):
        raise FileNotFoundError(f"模型路径不存在: {cfg['model'].get('model_path')}")

    created = _normalize_local_model_files(cfg["model"]["model_path"])
    if created:
        LOGGER.info("已为本地模型目录补齐标准文件名: %s", ", ".join(created))

    _ensure_dir_writable(cfg["paths"]["output_root"])

    resume_ckpt = cfg["training"].get("resume_from_checkpoint")
    if resume_ckpt:
        resume_ckpt = _resolve_path(resume_ckpt, base_dir)
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"resume_from_checkpoint 路径不存在: {resume_ckpt}")
        cfg["training"]["resume_from_checkpoint"] = resume_ckpt

    point_encoder_cfg = cfg.get("point_encoder")
    if isinstance(point_encoder_cfg, dict) and point_encoder_cfg.get("type") == "point_transformer_v3":
        point_encoder_cfg.setdefault("num_points", int(cfg["data"].get("pointcloud_num_points", 256)))
        point_encoder_cfg.setdefault("repo_path", "/root/autodl-tmp")
        cfg["point_encoder"] = point_encoder_cfg

    return cfg


def _training_args_from_config(run_dir: str, training_cfg: Dict[str, Any], use_eval: bool) -> TrainingArguments:
    precision = training_cfg.get("precision", "bf16").lower()
    eval_strategy = training_cfg.get("evaluation_strategy", "steps") if use_eval else "no"
    args_kwargs = {
        "output_dir": os.path.join(run_dir, "checkpoints"),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "dataloader_num_workers": int(training_cfg.get("dataloader_num_workers", 4)),
        "dataloader_pin_memory": bool(training_cfg.get("dataloader_pin_memory", True)),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-5)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "warmup_steps": int(training_cfg.get("warmup_steps", 0)),
        "max_steps": int(training_cfg.get("max_steps", -1)),
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 1.0)),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        "save_strategy": training_cfg.get("save_strategy", "steps"),
        "save_steps": int(training_cfg.get("save_steps", 200)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 3)),
        "remove_unused_columns": False,
        "bf16": precision in {"bf16", "bfloat16"},
        "fp16": precision in {"fp16", "float16"},
        "report_to": [],
        "ddp_find_unused_parameters": training_cfg.get("ddp_find_unused_parameters", None),
        "load_best_model_at_end": bool(training_cfg.get("load_best_model_at_end", False)),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", None),
        "greater_is_better": training_cfg.get("greater_is_better", None),
    }

    if use_eval:
        args_kwargs["eval_steps"] = int(training_cfg.get("eval_steps", 200))
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        args_kwargs["eval_strategy"] = eval_strategy
    else:
        args_kwargs["evaluation_strategy"] = eval_strategy

    valid_params = inspect.signature(TrainingArguments.__init__).parameters
    args_kwargs = {k: v for k, v in args_kwargs.items() if k in valid_params and v is not None}
    return TrainingArguments(**args_kwargs)


def _trainer_kwargs(model, args, train_ds, eval_ds, collate_fn, processor):
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collate_fn,
    }
    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = processor
    elif "tokenizer" in params:
        kwargs["tokenizer"] = getattr(processor, "tokenizer", processor)
    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL-2B-Instruct BenchCAD 训练入口")
    parser.add_argument("--config", type=str, default="configs-qwn3_train.yaml", help="配置文件路径")
    parser.add_argument("--set", action="append", default=[], help="通过 key=value 覆盖配置")
    parser.add_argument("--model-path", type=str, default=None, help="覆盖 model.model_path")
    parser.add_argument("--index-path", type=str, default=None, help="覆盖 data.index_path")
    parser.add_argument("--output-root", type=str, default=None, help="覆盖 paths.output_root")
    parser.add_argument("--run-name", type=str, default=None, help="覆盖 experiment.run_name")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="覆盖 training.resume_from_checkpoint")
    parser.add_argument("--check-only", action="store_true", help="只检查配置、数据和 processor，不加载模型权重")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_config(args.config)

    if args.model_path:
        _set_nested(cfg, "model.model_path", args.model_path)
    if args.index_path:
        _set_nested(cfg, "data.index_path", args.index_path)
    if args.output_root:
        _set_nested(cfg, "paths.output_root", args.output_root)
    if args.run_name:
        _set_nested(cfg, "experiment.run_name", args.run_name)
    if args.resume_from_checkpoint:
        _set_nested(cfg, "training.resume_from_checkpoint", args.resume_from_checkpoint)

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set 参数格式错误（应为 key=value）: {item}")
        key, raw_value = item.split("=", 1)
        _set_nested(cfg, key.strip(), _parse_scalar(raw_value))

    return _validate_and_normalize_config(cfg, args.config)


def train(cfg: Dict[str, Any], check_only: bool = False):
    run_dir = _build_run_dir(cfg)
    _setup_logging(run_dir=run_dir, level=cfg["experiment"].get("log_level", "INFO"))
    LOGGER.info("训练目录: %s", run_dir)

    snapshot_path = os.path.join(run_dir, "config_resolved.yaml")
    _save_config(snapshot_path, cfg)
    LOGGER.info("配置快照已写入: %s", snapshot_path)

    seed = int(cfg["experiment"].get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    LOGGER.info("开始构建数据集...")
    train_ds, eval_ds = _build_datasets(cfg)
    LOGGER.info("训练样本数: %d", len(train_ds))
    LOGGER.info("验证样本数: %d", len(eval_ds) if eval_ds is not None else 0)

    model_cfg = cfg["model"]
    LOGGER.info("加载 processor: %s", model_cfg["model_path"])
    processor = AutoProcessor.from_pretrained(
        model_cfg["model_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        min_pixels=model_cfg.get("min_pixels", None),
        max_pixels=model_cfg.get("max_pixels", None),
        padding_side=model_cfg.get("padding_side", "left"),
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = model_cfg.get("padding_side", "left")
    elif hasattr(processor, "padding_side"):
        processor.padding_side = model_cfg.get("padding_side", "left")

    collate_fn = partial(collate_qwen3vl, processor=processor, train_mode=True)
    sample_batch = collate_fn([train_ds[0]])
    LOGGER.info("单样本 collate 成功，batch keys=%s", sorted(sample_batch.keys()))

    if check_only:
        LOGGER.info("check-only 完成：已验证配置、数据集、processor 和单样本 collate。")
        return

    model_kwargs = {
        "torch_dtype": _torch_dtype_from_name(model_cfg.get("dtype", "bfloat16")),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    attn_impl = model_cfg.get("attn_implementation", "auto")
    if attn_impl and str(attn_impl).lower() != "auto":
        model_kwargs["attn_implementation"] = attn_impl

    LOGGER.info("加载模型权重...")
    model = _load_vision_text_model(
        model_cfg["model_path"],
        model_kwargs,
        point_encoder_cfg=cfg.get("point_encoder"),
    )
    LOGGER.info("模型加载完成，训练模式: %s", cfg["training"].get("train_mode", "lora"))
    model = _maybe_apply_lora(model, cfg)

    param_stat = _count_parameters(model)
    LOGGER.info("参数统计 | total=%d | trainable=%d", param_stat["total"], param_stat["trainable"])
    if param_stat["trainable"] <= 0:
        raise RuntimeError("当前模型可训练参数为 0，无法进行训练。请检查 LoRA 注入和 train_mode 配置。")

    if bool(cfg["training"].get("gradient_checkpointing", False)):
        _enable_input_require_grads_for_checkpointing(model)
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        LOGGER.info("已启用 gradient checkpointing。")

    args = _training_args_from_config(run_dir, cfg["training"], use_eval=eval_ds is not None)
    callbacks = [
        JsonlFileLogCallback(os.path.join(run_dir, "logs", "metrics.jsonl")),
        NaNLossStopCallback(),
    ]
    trainer = Trainer(
        **_trainer_kwargs(model, args, train_ds, eval_ds, collate_fn, processor),
        callbacks=callbacks,
    )
    _patch_accelerator_unwrap_model_for_compat(trainer.accelerator)

    resume_ckpt = cfg["training"].get("resume_from_checkpoint")
    LOGGER.info("开始训练%s", f"，从 checkpoint 恢复: {resume_ckpt}" if resume_ckpt else "")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = os.path.join(run_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    LOGGER.info("训练完成，最终模型已保存到: %s", final_dir)


def main():
    args = parse_args()
    cfg = build_config(args)
    train(cfg, check_only=args.check_only)


if __name__ == "__main__":
    main()
