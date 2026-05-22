from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
try:
    from transformers.utils.generic import ModelOutput, can_return_tuple, check_model_inputs
    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import ModelOutput
    from transformers.processing_utils import Unpack
    from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig
except ImportError as e:
    import transformers
    raise ImportError(f"transformers=={transformers.__version__}: {e}")

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
from .token_selector import CoverageTokenSelector
from .concept_selector import ConceptSelector
from .longnet import LongNet
from .breastgpt import BreastGPTOutput
def debug(x):
    # print(x)
    pass

########################################### Config ########################################################
class BreastGPT2Config(PretrainedConfig):
    model_type = "breastgpt2"
    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfig,
        "text_config": Qwen2_5_VLTextConfig,
    }
 
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        do_select=True,                # ← 新增
        image_token_id=151655,
        histo_token_id=151669,
        feature_dim=768,
        image_token_num=64,
        video_token_num=128,
        histo_token_num=128,
        long_net_heads=8,
        long_net_layers=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
 
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
 
        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)
 
        self.do_select = do_select
        self.image_token_id = image_token_id
        self.histo_token_id = histo_token_id
        self.feature_dim = feature_dim
        self.image_token_num = image_token_num
        self.histo_token_num = histo_token_num
        self.video_token_num = video_token_num
        self.long_net_heads = long_net_heads
        self.long_net_layers = long_net_layers
 
 
# ═══════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════
 
class BreastGPT2(Qwen2_5_VLForConditionalGeneration):
    config_class = BreastGPT2Config
    base_model_prefix = "breastgpt2"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False
 
    def __init__(self, config):
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(config)
        self.model = Qwen2_5_VLModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
 
        self.slide = LongNet(
            embed_dim=config.feature_dim,
            llm_hidden_size=config.text_config.hidden_size,
            num_heads=config.long_net_heads,
            num_layers=config.long_net_layers,
        )
        # ── 对齐 BreastGPT: 传入 use_st ──
        self.visual_selector = CoverageTokenSelector(use_st=self.config.do_select)
        self.deepstack_selector = CoverageTokenSelector(use_st=self.config.do_select)
        self.rope_deltas = self.model.rope_deltas
        self.post_init()
 
    def _init_weights(self, module):
        """对齐 BreastGPT: TokenSelector 打分器零初始化"""
        super()._init_weights(module)
        if "TokenSelector" in module.__class__.__name__:
            if hasattr(module, "scorer") and isinstance(module.scorer[-1], nn.Linear):
                module.scorer[-1].weight.data.zero_()
                module.scorer[-1].bias.data.zero_()
 
    # ═══════════════════════════════════════════════════════════════
    # get_rope_index — 对齐 BreastGPT: 加入 histo_grid_thw 支持
    # ═══════════════════════════════════════════════════════════════
 
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        histo_grid_thw: Optional[torch.LongTensor] = None,          # ← 新增
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        histo_token_id = self.config.histo_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
 
        has_visual = (
            image_grid_thw is not None
            or video_grid_thw is not None
            or histo_grid_thw is not None
        )
 
        if input_ids is not None and has_visual:
            total_input_ids = input_ids
            if attention_mask is not None:
                attention_mask = attention_mask == 1
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1],
                dtype=input_ids.dtype, device=input_ids.device,
            )
            image_index, video_index, histo_index = 0, 0, 0
 
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i]]
 
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
 
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()
                histo_nums = (vision_tokens == histo_token_id).sum().item()
 
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_histo = (
                    image_nums, video_nums, histo_nums,
                )
 
                for _ in range(image_nums + video_nums + histo_nums):
                    ed_image = (
                        input_tokens.index(image_token_id, st)
                        if image_token_id in input_tokens and remain_images > 0
                        else len(input_tokens) + 1
                    )
                    ed_video = (
                        input_tokens.index(video_token_id, st)
                        if video_token_id in input_tokens and remain_videos > 0
                        else len(input_tokens) + 1
                    )
                    ed_histo = (
                        input_tokens.index(histo_token_id, st)
                        if histo_token_id in input_tokens and remain_histo > 0
                        else len(input_tokens) + 1
                    )
 
                    min_ed = min(ed_image, ed_video, ed_histo)
 
                    if min_ed == ed_image:
                        t, h, w = image_grid_thw[image_index]
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    elif min_ed == ed_video:
                        t, h, w = video_grid_thw[video_index]
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    elif min_ed == ed_histo:
                        t, h, w = histo_grid_thw[histo_index]
                        second_per_grid_t = 0
                        histo_index += 1
                        remain_histo -= 1
                        ed = ed_histo
                    else:
                        raise NotImplementedError(
                            f"{image_grid_thw} {video_grid_thw} {histo_grid_thw}"
                        )
 
                    text_len = ed - st
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
 
                    # ── 对齐 BreastGPT: 用 actual_len 数连续视觉 token ──
                    vis_token_ids = {image_token_id, video_token_id, histo_token_id}
                    actual_len = 0
                    while (
                        (ed + actual_len) < len(input_tokens)
                        and input_tokens[ed + actual_len] in vis_token_ids
                    ):
                        actual_len += 1
 
                    t_index = torch.arange(actual_len)
                    h_index = torch.arange(actual_len)
                    w_index = torch.arange(actual_len)
 
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + actual_len
 
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )
 
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                if attention_mask is not None:
                    position_ids[..., i, attention_mask[i]] = llm_positions.to(
                        position_ids.device
                    )
                else:
                    position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
 
            mrope_position_deltas = (
                torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)
            )
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
                    attention_mask.device
                )
                max_position_ids = (
                    position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                )
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas
 

    def _fix_visual_position_ids(
        self,
        position_ids,        # [3, B, L_compressed]
        image_mask_1d,       # [B, L_compressed]
        video_mask_1d,
        histo_mask_1d,
        image_idxs,          # list[Tensor[K_i]]
        video_idxs,
        histo_idxs,
        image_grid_thw,
        video_grid_thw,
        histo_grid_thw,
    ) -> torch.LongTensor:
        merge_size = self.config.vision_config.spatial_merge_size
        position_ids = position_ids.clone()
 
        def _replace(modal_mask, idxs, grid_thw, is_histo=False):
            if idxs is None or grid_thw is None:
                return
            split_sizes = [idx.shape[0] for idx in idxs]
            b_all, s_all = torch.where(modal_mask)
            s_splits = torch.split(s_all, split_sizes)
            b_splits = torch.split(b_all, split_sizes)
 
            for i, (s_split, b_split) in enumerate(zip(s_splits, b_splits)):
                b = b_split[0].item()
                base = position_ids[0, b, s_split[0]].item()
 
                if is_histo:
                    H_prime, W_prime = 1, grid_thw[i, 2].item()
                else:
                    T, H, W = grid_thw[i]
                    H_prime = H.item() // merge_size
                    W_prime = W.item() // merge_size
 
                idx = idxs[i].long()
                t_c = idx // (H_prime * W_prime)
                h_c = (idx % (H_prime * W_prime)) // W_prime
                w_c = idx % W_prime
 
                position_ids[0, b, s_split] = base + t_c
                position_ids[1, b, s_split] = base + h_c
                position_ids[2, b, s_split] = base + w_c
 
        _replace(image_mask_1d, image_idxs, image_grid_thw)
        _replace(video_mask_1d, video_idxs, video_grid_thw)
        _replace(histo_mask_1d, histo_idxs, histo_grid_thw, is_histo=True)
 
        return position_ids
 
    # ═══════════════════════════════════════════════════════════════
    # get_placeholder_mask — 对齐 BreastGPT: 加入 expand 参数
    # ═══════════════════════════════════════════════════════════════
 
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
        histo_features: Optional[torch.FloatTensor] = None,
        expand=True,
    ):
        if input_ids is None:
            embed_fn = self.get_input_embeddings()
            img_tok = torch.tensor(
                self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device
            )
            special_image_mask = (inputs_embeds == embed_fn(img_tok)).all(-1)
            vid_tok = torch.tensor(
                self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device
            )
            special_video_mask = (inputs_embeds == embed_fn(vid_tok)).all(-1)
            his_tok = torch.tensor(
                self.config.histo_token_id, dtype=torch.long, device=inputs_embeds.device
            )
            special_histo_mask = (inputs_embeds == embed_fn(his_tok)).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_histo_mask = input_ids == self.config.histo_token_id
 
        if expand:
            n_image_tokens = special_image_mask.sum()
            special_image_mask = (
                special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            if (
                image_features is not None
                and inputs_embeds[special_image_mask].numel() != image_features.numel()
            ):
                raise ValueError(
                    f"Image features and tokens do not match: "
                    f"tokens: {n_image_tokens}, features {image_features.shape[0]}"
                )
 
            n_video_tokens = special_video_mask.sum()
            special_video_mask = (
                special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            if (
                video_features is not None
                and inputs_embeds[special_video_mask].numel() != video_features.numel()
            ):
                raise ValueError(
                    f"Video features and tokens do not match: "
                    f"tokens: {n_video_tokens}, features {video_features.shape[0]}"
                )
 
            n_histo_tokens = special_histo_mask.sum()
            special_histo_mask = (
                special_histo_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            if (
                histo_features is not None
                and inputs_embeds[special_histo_mask].numel() != histo_features.numel()
            ):
                raise ValueError(
                    f"Histo features and tokens do not match: "
                    f"tokens: {n_histo_tokens}, features {histo_features.shape[0]}"
                )
 
        return special_image_mask, special_video_mask, special_histo_mask
 
    # ═══════════════════════════════════════════════════════════════
    # visual_token_merge — 对齐 BreastGPT: 返回 selected_idxs
    # ═══════════════════════════════════════════════════════════════
 
    def visual_token_merge(
        self,
        visual_embeds_tuple: tuple[torch.FloatTensor],
        grid_thw: torch.LongTensor,
        token_num,
    ):
        merged_visuals = []
        selected_idxs = []
        loss = 0
        for feat in visual_embeds_tuple:
            compressed, selected_idx, loss_logits = self.visual_selector(
                feat.unsqueeze(0), token_num
            )
            loss += loss_logits
            merged_visuals.append(compressed.squeeze(0))
            selected_idxs.append(selected_idx.squeeze(0))
 
        final_visual_embeds = torch.cat(merged_visuals, dim=0)
        return final_visual_embeds, selected_idxs, loss
 
    # ═══════════════════════════════════════════════════════════════
    # get_histo_features — 对齐 BreastGPT: 返回 histo_idxs
    # ═══════════════════════════════════════════════════════════════
 
    def get_histo_features(self, input_features: list[torch.FloatTensor]):
        loss = 0
        merged_features = []
        histo_idxs = []
        for feat in input_features:
            feat = feat.unsqueeze(0).to(self.device)
            feat = self.slide(feat)
            feat, histo_idx, _loss = self.visual_selector(feat, self.config.histo_token_num)
            loss += _loss
            merged_features.append(feat.squeeze(0))
            histo_idxs.append(histo_idx.squeeze(0))
        return merged_features, loss, histo_idxs
 
    def load_ori_state_dict(self, mllm_path):
        original_model = self.from_pretrained(mllm_path)
        original_state_dict = original_model.state_dict()
        self.load_state_dict(original_state_dict, strict=False)
        del original_model
 
    @property
    def language_model(self):
        return self.model.language_model
 
    # ═══════════════════════════════════════════════════════════════
    # forward — 对齐 BreastGPT: dummy input + position fix
    # ═══════════════════════════════════════════════════════════════
 
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        input_features: Optional[list[torch.FloatTensor]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        histo_grid_thw: Optional[torch.LongTensor] = None,          # ← 新增
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BreastGPTOutput]:
 
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
 
        debug(f"[{time.time()}] START TRAINING DATA: {input_ids.shape}")
 
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
 
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
 
        image_mask = None
        video_mask = None
        histo_mask = None
 
        dummy_input = True
        image_idxs = video_idxs = histo_idxs = None
 
        # ════════════════════════════════════════════════
        # 1. Image
        # ════════════════════════════════════════════════
        if pixel_values is not None:
            image_embeds_tuple = self.get_image_features(pixel_values, image_grid_thw)
            debug(f"[{time.time()}] GETED IMAGE TOKEN: {pixel_values.shape} -> {image_embeds_tuple[0].shape}")
 
            image_embeds, image_idxs, loss = self.visual_token_merge(
                image_embeds_tuple, image_grid_thw, self.config.image_token_num
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
 
            image_mask, _, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            debug(f"[{time.time()}] IMAGE TOKEN PROCESSED: {image_embeds.shape}")
 
        elif dummy_input:
            # ── 对齐 BreastGPT: dummy image 保证梯度覆盖 ──
            dummy_image_grid_thw = torch.tensor(
                [[inputs_embeds.shape[0], 16, 8]],
                dtype=torch.long, device=inputs_embeds.device,
            )
            total_patches = dummy_image_grid_thw[0].prod().item()
            dim_pixels = (
                self.config.vision_config.in_channels
                * self.config.vision_config.temporal_patch_size
                * self.config.vision_config.patch_size ** 2
            )
            dummy_pixel_values = torch.zeros(
                total_patches, dim_pixels,
                dtype=torch.bfloat16, device=inputs_embeds.device,
            )
            image_embeds_tuple = self.get_image_features(
                dummy_pixel_values, dummy_image_grid_thw
            )
            image_embeds, _, _ = self.visual_token_merge(
                image_embeds_tuple, dummy_image_grid_thw, self.config.image_token_num
            )
            inputs_embeds = inputs_embeds + image_embeds.mean() * 0.0
            debug(f"IMAGE DUMMY LOSS: {image_embeds.mean()}")
 
        # ════════════════════════════════════════════════
        # 2. Video
        # ════════════════════════════════════════════════
        if pixel_values_videos is not None:
            debug(f"[{time.time()}] START PROCESSING Video TOKEN: {pixel_values_videos.shape}")
            video_embeds_tuple = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds, video_idxs, loss = self.visual_token_merge(
                video_embeds_tuple, video_grid_thw, self.config.video_token_num
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
 
            _, video_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            debug(f"[{time.time()}] VIDEO TOKEN PROCESSED: {video_embeds.shape}")
 
        # ════════════════════════════════════════════════
        # 3. Histo
        # ════════════════════════════════════════════════
        if input_features is not None:
            histo_embeds_list, _loss, histo_idxs = self.get_histo_features(input_features)
            histo_embeds = torch.cat(histo_embeds_list, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, _, histo_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, histo_features=histo_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(histo_mask, histo_embeds)
            debug(f"[{time.time()}] Histo TOKEN PROCESSED: {histo_embeds.shape}")
 
        elif dummy_input:
            # ── 对齐 BreastGPT: dummy histo 保证梯度覆盖 ──
            dummy_input_features = [
                torch.zeros(
                    (self.config.histo_token_num, self.config.feature_dim),
                    dtype=inputs_embeds.dtype, device=inputs_embeds.device,
                )
            ] * inputs_embeds.shape[0]
            histo_embeds_list, _, _ = self.get_histo_features(dummy_input_features)
            for his in histo_embeds_list:
                inputs_embeds = inputs_embeds + his.mean() * 0.0
            debug(f"HISTO DUMMY LOSS: {histo_embeds_list[0].mean()}")
 
        # ════════════════════════════════════════════════
        # 4. Position IDs (with histo_grid_thw)
        # ════════════════════════════════════════════════
        if position_ids is None:
            prefill_compiled = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_normal = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if prefill_compiled or prefill_normal or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    histo_grid_thw=histo_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros(
                        (batch_size, seq_length), device=inputs_embeds.device
                    )
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids = position_ids + delta.to(position_ids.device)
 
        # ── 对齐 BreastGPT: 用 selector idx 修正视觉段 position_ids ──
        image_mask_1d, video_mask_1d, histo_mask_1d = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, expand=False
        )
        position_ids = self._fix_visual_position_ids(
            position_ids,
            image_mask_1d, video_mask_1d, histo_mask_1d,
            image_idxs, video_idxs, histo_idxs,
            image_grid_thw, video_grid_thw, histo_grid_thw,
        )
 
        # ════════════════════════════════════════════════
        # 5. LLM forward
        # ════════════════════════════════════════════════
        debug(f"[{time.time()}] MEDIA FEATURE PROCESSED: {inputs_embeds.shape}")
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        debug(f"[{time.time()}] LANGUAGE MODEL PROCESSED")
 
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
 
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )
        return BreastGPTOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )