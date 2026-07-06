import os
import sys
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast


def _cache_seq_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    try:
        return int(past_key_values[0][0].shape[-2])
    except Exception:
        return 0


def reset_point_transformer_v3_luts():
    try:
        from PointTransformerV3.serialization import z_order

        z_order._key_lut = z_order.KeyLUT()
    except Exception:
        pass


def initialize_point_encoder_weights(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, (nn.BatchNorm1d, nn.LayerNorm)):
            if getattr(submodule, "weight", None) is not None:
                nn.init.ones_(submodule.weight)
            if getattr(submodule, "bias", None) is not None:
                nn.init.zeros_(submodule.bias)
        elif submodule.__class__.__module__.startswith("spconv"):
            weight = getattr(submodule, "weight", None)
            if isinstance(weight, torch.Tensor):
                nn.init.kaiming_uniform_(weight, a=5**0.5)
            bias = getattr(submodule, "bias", None)
            if isinstance(bias, torch.Tensor):
                nn.init.zeros_(bias)


class PointTransformerV3PrefixEncoder(nn.Module):
    def __init__(self, hidden_size: int, cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        cfg = dict(cfg or {})
        repo_path = cfg.get("repo_path", "/root/autodl-tmp")
        ptv3_parent = os.path.abspath(repo_path)
        if ptv3_parent not in sys.path:
            sys.path.insert(0, ptv3_parent)

        try:
            from PointTransformerV3.model import PointTransformerV3
        except Exception as exc:
            raise ImportError(
                "启用 PointTransformerV3 点云 encoder 需要安装并可导入 "
                "spconv.pytorch、torch_scatter、timm，并确保 /root/autodl-tmp/PointTransformerV3 存在。"
            ) from exc

        self.num_points = int(cfg.get("num_points", 256))
        self.grid_size = float(cfg.get("grid_size", 0.02))
        self.output_channels = int(cfg.get("output_channels", 32))

        patch_size = int(cfg.get("patch_size", 128))
        enc_channels = tuple(cfg.get("enc_channels", [32, 64, 128]))
        enc_depths = tuple(cfg.get("enc_depths", [1, 1, 1]))
        enc_num_head = tuple(cfg.get("enc_num_head", [2, 4, 8]))
        stride = tuple(cfg.get("stride", [2, 2]))
        dec_channels = tuple(cfg.get("dec_channels", [self.output_channels, 64]))
        dec_depths = tuple(cfg.get("dec_depths", [1, 1]))
        dec_num_head = tuple(cfg.get("dec_num_head", [2, 4]))

        self.backbone = PointTransformerV3(
            in_channels=3,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=tuple([patch_size] * len(enc_channels)),
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=tuple([patch_size] * len(dec_channels)),
            drop_path=float(cfg.get("drop_path", 0.1)),
            enable_flash=bool(cfg.get("enable_flash", False)),
            upcast_attention=bool(cfg.get("upcast_attention", True)),
            upcast_softmax=bool(cfg.get("upcast_softmax", True)),
            cls_mode=False,
        )
        self.proj = nn.Linear(dec_channels[0], hidden_size)
        reset_point_transformer_v3_luts()

    def forward(self, point_clouds: torch.Tensor) -> torch.Tensor:
        if point_clouds.ndim != 3 or point_clouds.shape[-1] != 3:
            raise ValueError(f"point_clouds 需要形状 [B,N,3]，实际: {tuple(point_clouds.shape)}")

        batch_size, num_points, _ = point_clouds.shape
        points = point_clouds.float().contiguous()
        flat_points = points.view(batch_size * num_points, 3)
        offset = torch.arange(
            1,
            batch_size + 1,
            device=point_clouds.device,
            dtype=torch.long,
        ) * num_points
        data_dict = {
            "feat": flat_points,
            "coord": flat_points,
            "grid_size": self.grid_size,
            "offset": offset,
        }
        encoded = self.backbone(data_dict).feat
        encoded = encoded.view(batch_size, num_points, -1)
        return self.proj(encoded)


class Qwen3VLWithPointTransformerV3(Qwen3VLForConditionalGeneration):
    def __init__(self, config, point_encoder_config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("无法从 Qwen3-VL config 中解析 hidden_size")

        point_encoder_config = dict(point_encoder_config or {})
        if "num_points" not in point_encoder_config:
            point_encoder_config["num_points"] = 256
        self.config.point_encoder_config = point_encoder_config
        self.point_encoder = PointTransformerV3PrefixEncoder(
            hidden_size=int(hidden_size),
            cfg=point_encoder_config,
        )

    def _inject_point_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        point_clouds: Optional[torch.Tensor],
        point_token_mask: Optional[torch.Tensor],
        past_key_values=None,
    ) -> torch.Tensor:
        if point_clouds is None or _cache_seq_len(past_key_values) > 0:
            return inputs_embeds
        if point_token_mask is None:
            raise ValueError("使用 point_clouds 时必须提供 point_token_mask")

        point_token_mask = point_token_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        point_embeds = self.point_encoder(point_clouds.to(inputs_embeds.device))
        point_embeds = point_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        expected = int(point_embeds.shape[1])
        actual_per_row = point_token_mask.sum(dim=1)
        if not torch.all(actual_per_row == expected):
            raise ValueError(
                "point_token_mask 中每个样本的 True 数量必须等于 point_clouds 点数。"
                f" expected={expected}, actual={actual_per_row.tolist()}"
            )

        updated = inputs_embeds.clone()
        updated[point_token_mask] = point_embeds.reshape(-1, point_embeds.shape[-1])
        return updated

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        point_clouds: Optional[torch.Tensor] = None,
        point_token_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if point_clouds is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if input_ids is None:
            raise ValueError("PointTransformerV3 prefix 注入需要 input_ids 来定位视觉和点云 token")
        if inputs_embeds is not None:
            raise ValueError("PointTransformerV3 prefix 注入暂不支持外部传入 inputs_embeds")

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = self._inject_point_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            point_clouds=point_clouds,
            point_token_mask=point_token_mask,
            past_key_values=past_key_values,
        )

        vision_kwargs = dict(kwargs)
        vision_kwargs.pop("return_dict", None)
        image_mask = None
        video_mask = None
        if pixel_values is not None:
            image_outputs = self.model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **vision_kwargs
            )
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            deepstack_image_embeds = image_outputs.deepstack_features

        if pixel_values_videos is not None:
            video_outputs = self.model.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **vision_kwargs
            )
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            deepstack_video_embeds = video_outputs.deepstack_features

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.model.rope_deltas,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if "point_clouds" in kwargs:
            model_inputs["point_clouds"] = kwargs["point_clouds"]
        if "point_token_mask" in kwargs:
            model_inputs["point_token_mask"] = kwargs["point_token_mask"]
        return model_inputs
