from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
try:
    from transformers.utils.generic import ModelOutput, check_model_inputs, TransformersKwargs
    from transformers.utils.import_utils import is_torchdynamo_compiling
    from transformers.cache_utils import Cache
    from transformers.processing_utils import Unpack
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
except ImportError as e:
    import transformers
    raise ImportError(f"transformers=={transformers.__version__}: {e}")

from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel
from .token_selector import CoverageTokenSelector
from .concept_selector import ConceptSelector
from .longnet import LongNet

def debug(x):
    # print(x)
    pass

# transformers==4.57.6
############################################################### Config ########################################################
class BreastGPTConfig(PretrainedConfig):
    """Configuration class for BreastGPT model."""

    model_type = 'breastgpt'
    sub_configs = {'vision_config': Qwen3VLVisionConfig, 'text_config': Qwen3VLTextConfig}

    def __init__(self,
                 text_config=None,
                 vision_config=None,
                 do_select=True,
                 image_token_id=151655,
                 histo_token_id=151669,
                 feature_dim=768,
                 image_token_num=64,
                 histo_token_num=128,
                 video_token_num=128,
                 long_net_heads=8,      # 新增：LongNet 的注意力头数
                 long_net_layers=3,     # 新增：LongNet 的层数
                 selector='CoverageTokenSelector',
                 selector_use_txt=False,
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs['vision_config'](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs['vision_config']()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs['text_config'](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs['text_config'](**kwargs)
        self.do_select = do_select
        self.image_token_id = image_token_id
        self.histo_token_id = histo_token_id
        self.feature_dim = feature_dim
        self.image_token_num = image_token_num
        self.histo_token_num = histo_token_num
        self.video_token_num = video_token_num
        self.long_net_heads = long_net_heads
        self.long_net_layers = long_net_layers
        self.selector = selector
        self.selector_use_txt = selector_use_txt


def dummy_loss_for_deepspeed(model):
    # 在 forward 的最后，loss 计算逻辑处
    dummy_loss = 0.0

    # 策略：直接遍历全模型参数，而不仅仅是自定义模块
    for name, p in model.named_parameters():
        if p.requires_grad:
            # 这里用 p.view(-1)[0] 或 p.mean() 都可以
            # 目的是建立计算图连接，0.0 确保不影响真实梯度
            dummy_loss += 0.0 * p.sum()
    return dummy_loss

@dataclass
class BreastGPTOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

############################################################### MODEL ########################################################

class BreastGPT(Qwen3VLForConditionalGeneration):
    config_class = BreastGPTConfig
    base_model_prefix = 'breastgpt'
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def _init_selector(self, D):
        if self.config.selector == 'ConceptSelector':
            return ConceptSelector(
                hidden_dim=D, num_heads=8,
                num_learnable_concepts=32,
            )
        elif self.config.selector == 'CoverageTokenSelector':
            return CoverageTokenSelector(
                use_st=self.config.do_select,
                use_txt=self.config.selector_use_txt
            )
        raise KeyError(f'SELECTOR NOT EXISTS: {self.config.selector}')

    def __init__(self, config):
        super(Qwen3VLForConditionalGeneration, self).__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.slide = LongNet(
            embed_dim=config.feature_dim,
            llm_hidden_size=config.text_config.hidden_size,
            num_heads=config.long_net_heads,
            num_layers=config.long_net_layers,
            num_deepstack = len(config.vision_config.deepstack_visual_indexes)
        )
        # Override config from env vars if set
        import os
        if "BREASTGPT_SELECTOR" in os.environ:
            config.selector = os.environ["BREASTGPT_SELECTOR"]
        if "BREASTGPT_USE_TXT" in os.environ:
            config.selector_use_txt = os.environ["BREASTGPT_USE_TXT"].lower() == "true"
        if "BREASTGPT_USE_ST" in os.environ:
            config.use_st = os.environ["BREASTGPT_USE_ST"].lower() == "true"
        if "SELECT_IMAGE_NUM" in os.environ:
            self.config.image_token_num = int(os.environ["SELECT_IMAGE_NUM"])
        if "SELECT_HISTO_NUM" in os.environ:
            self.config.histo_token_num = int(os.environ["SELECT_HISTO_NUM"])
        if "SELECT_VIDEO_NUM" in os.environ:
            self.config.video_token_num = int(os.environ["SELECT_VIDEO_NUM"])
            
        self.visual_selector = self._init_selector(config.text_config.hidden_size)
        self.deepstack_selector = self._init_selector(config.text_config.hidden_size)
        self.rope_deltas = self.model.rope_deltas
        self.post_init()
        print(f"BREAST GPT SELECTOR: {self.config.selector} | USE TXT: {self.config.selector_use_txt} ｜ SELECT_IMAGE_NUM: {self.config.image_token_num} | SELECT_HISTO_NUM: {self.config.histo_token_num} | SELECT_VIDEO_NUM: {self.config.video_token_num}")
    
    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        histo_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        histo_token_id = self.config.histo_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None or histo_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            # Initialize all three indices
            image_index, video_index, histo_index = 0, 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                
                image_nums = (vision_tokens == image_token_id).sum().item()
                video_nums = (vision_tokens == video_token_id).sum().item()
                histo_nums = (vision_tokens == histo_token_id).sum().item()
                
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos, remain_histo = image_nums, video_nums, histo_nums
                
                # Include histo_nums in the loop range
                for _ in range(image_nums + video_nums + histo_nums):
                    ed_image = input_tokens.index(image_token_id, st) if image_token_id in input_tokens and remain_images > 0 else len(input_tokens) + 1
                    ed_video = input_tokens.index(video_token_id, st) if video_token_id in input_tokens and remain_videos > 0 else len(input_tokens) + 1
                    ed_histo = input_tokens.index(histo_token_id, st) if histo_token_id in input_tokens and remain_histo > 0 else len(input_tokens) + 1
                    
                    # Determine which modality token appears next in the sequence
                    min_ed = min(ed_image, ed_video, ed_histo)
                    
                    if min_ed == ed_image:
                        t, h, w = image_grid_thw[image_index]
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size
                        )
                        is_1d_histo = False # 标记为普通 3D 逻辑
                        
                    elif min_ed == ed_video:
                        t, h, w = video_grid_thw[video_index]
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(), h.item() // spatial_merge_size, w.item() // spatial_merge_size
                        )
                        is_1d_histo = False # 标记为普通 3D 逻辑
                        
                    elif min_ed == ed_histo:
                        t, h, w = histo_grid_thw[histo_index]
                        histo_index += 1
                        remain_histo -= 1
                        ed = ed_histo
                        
                        # 【核心魔改】：histo 直接走 1D 通道！
                        # 绕过 spatial_merge_size 的除法，w 直接就是真实的 Token 数量
                        llm_grid_t, llm_grid_h, llm_grid_w = 1, 1, w.item()
                        is_1d_histo = True # 标记为 1D 逻辑
                    else:
                        raise NotImplementedError(f'{image_grid_thw} {video_grid_thw} {histo_grid_thw}')

                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # # 【核心魔改】：根据是否是 1D 分支生成坐标
                    # if is_1d_histo:
                    #     # 像文本一样：(i, i, i)，拉满 3 个通道的位置编码能力
                    #     token_len = llm_grid_w
                    #     t_index = torch.arange(token_len)
                    #     h_index = torch.arange(token_len)
                    #     w_index = torch.arange(token_len)
                    # else:
                    #     # 原版 3D/2D 模式
                    #     t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    #     h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    #     w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                        
                    # llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    # st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                    # 数 ed 位置之后实际有多少个连续视觉 token
                    vis_token_ids = {image_token_id, video_token_id, histo_token_id}
                    actual_len = 0
                    while (ed + actual_len) < len(input_tokens) and input_tokens[ed + actual_len] in vis_token_ids:
                        actual_len += 1

                    # 生成 actual_len 个顺序坐标（_fix_visual_position_ids 之后会替换成真实 2D 坐标）
                    t_index = torch.arange(actual_len)
                    h_index = torch.arange(actual_len)
                    w_index = torch.arange(actual_len)

                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + actual_len   # ← 替换原来的 st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
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

    def _modality_batch_map(self, input_ids, token_id):
        """每个 modality item 对应哪个 batch sample."""
        batch_map = []
        for b in range(input_ids.shape[0]):
            mask = (input_ids[b] == token_id)
            # 数连续段数
            n = (mask[1:] & ~mask[:-1]).sum().item() + (1 if mask[0] else 0)
            batch_map.extend([b] * n)
        return batch_map

    def get_histo_features(self, text_concepts, input_features: list[torch.FloatTensor], batch_map, return_intermediate=False):
        loss = 0
        merged_features = []
        histo_idxs = []
        deepstack_per_sample = []   # List[List[Tensor]]  shape: [n_histo][n_layers][K, D]

        for i, feat in enumerate(input_features):
            b = batch_map[i]
            feat = feat.unsqueeze(0).to(self.device)  # [1, N, D]

            # ── 新增：拿中间层特征 ──────────────────────────
            if not return_intermediate:
                feat = self.slide(feat, return_intermediate=False)
                intermediates = None
            else:
                feat, intermediates = self.slide(feat, return_intermediate=True)
                sample_ds = []
                for inter in intermediates:
                    # inter: [1, N_i, D_slide]，需要投影到 LLM hidden size
                    compressed, _, ds_loss = self.deepstack_selector(inter, self.config.histo_token_num, text_feats=text_concepts[b:b+1])
                    loss += ds_loss
                    sample_ds.append(compressed.squeeze(0))      # [K, D_llm]
                deepstack_per_sample.append(sample_ds)
            # 顶层特征 → visual_selector 压缩
            feat, histo_idx, _loss = self.visual_selector(feat, self.config.histo_token_num, text_feats=text_concepts[b:b+1])
            loss += _loss
            merged_features.append(feat.squeeze(0))          # [K, D]
            histo_idxs.append(histo_idx.squeeze(0))

            
        if return_intermediate:
            # 把 n_histo 个样本的 deepstack 按层合并
            # deepstack_histo_embeds: List[Tensor[sum_K, D_llm]]，和 image 的格式一致
            n_layers = len(deepstack_per_sample[0])
            deepstack_histo_embeds = [
                torch.cat([deepstack_per_sample[i][l] for i in range(len(input_features))], dim=0)
                for l in range(n_layers)
            ]
        else:
            deepstack_histo_embeds = None

        return merged_features, loss, histo_idxs, deepstack_histo_embeds


    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
        histo_features: Optional[torch.FloatTensor] = None,
        expand = True,
    ):
        # 纯粹的提取 1D 掩码 [Batch, SeqLen]
        if input_ids is None:
            embed_fn = self.get_input_embeddings()
            img_tok = torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            special_image_mask = (inputs_embeds == embed_fn(img_tok)).all(-1)
            vid_tok = torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            special_video_mask = (inputs_embeds == embed_fn(vid_tok)).all(-1)
            his_tok = torch.tensor(self.config.histo_token_id, dtype=torch.long, device=inputs_embeds.device)
            special_histo_mask = (inputs_embeds == embed_fn(his_tok)).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id
            special_histo_mask = input_ids == self.config.histo_token_id
        if expand:
            # --- 校验 Image ---
            n_image_tokens = special_image_mask.sum()
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
                raise ValueError(f"Image features and tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}")

            # --- 校验 Video ---
            n_video_tokens = special_video_mask.sum()
            special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
                raise ValueError(f"Videos features and tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}")

            # --- 校验 Histo ---
            n_histo_tokens = special_histo_mask.sum()
            special_histo_mask = special_histo_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            if histo_features is not None and inputs_embeds[special_histo_mask].numel() != histo_features.numel():
                raise ValueError(f"Histo features and tokens do not match: tokens: {n_histo_tokens}, features {histo_features.shape[0]}")

        return special_image_mask, special_video_mask, special_histo_mask


    def visual_token_merge(
        self, 
        text_concepts,
        visual_embeds_tuple: tuple[torch.FloatTensor], 
        deepstack_embeds_list: list[torch.FloatTensor],
        grid_thw: torch.LongTensor,
        token_num,
        batch_map,
    ):
        # ==========================================
        # 宏观流: 处理深层全局特征
        # ==========================================
        merged_visuals = []
        selected_idxs = []
        loss = 0
        for i, feat in enumerate(visual_embeds_tuple):
            b = batch_map[i]
            compressed, selected_idx, loss_logits = self.visual_selector(feat.unsqueeze(0), token_num, text_feats=text_concepts[b:b+1])
            loss += loss_logits
            merged_visuals.append(compressed.squeeze(0))
            selected_idxs.append(selected_idx.squeeze(0))
            
        final_visual_embeds = torch.cat(merged_visuals, dim=0) 
        # ==========================================
        # 微观流：TokenSelector 处理浅层/中层 DeepStack 特征
        # ==========================================
        final_deepstack = []
        all_concept_indices = [] # 收集所有图像、所有层的索引
        
        if deepstack_embeds_list is not None and len(deepstack_embeds_list) > 0:
            spatial_merge_size = self.visual.spatial_merge_size
            split_sizes = (grid_thw.prod(-1) // spatial_merge_size**2).tolist()
            
            for layer_feat in deepstack_embeds_list:
                layer_feat_splits = torch.split(layer_feat, split_sizes)
                selected_splits = []
                layer_indices = []
                
                for i, feat in enumerate(layer_feat_splits):
                    # 【修正点】加上 Batch 维度 [N_i, D] -> [1, N_i, D]
                    b = batch_map[i]
                    feat_batch = feat.unsqueeze(0)
                    selected, indices, loss_logits= self.deepstack_selector(feat_batch, token_num, text_feats=text_concepts[b:b+1])
                    loss += loss_logits
                    selected_splits.append(selected.squeeze(0))
                    layer_indices.append(indices.squeeze(0)) 
                    
                final_deepstack.append(torch.cat(selected_splits, dim=0))
                all_concept_indices.append(layer_indices)
        del deepstack_embeds_list
        return final_visual_embeds, selected_idxs, final_deepstack, all_concept_indices, loss
    

    def load_ori_state_dict(self, mllm_path):
        # load original model state dict
        original_model = self.from_pretrained(mllm_path)

        # get original model state dict
        original_state_dict = original_model.state_dict()

        # load weights
        self.load_state_dict(original_state_dict, strict=False)
        del original_model  # release memory

    @property
    def language_model(self):
        return self.model.language_model

    def _fix_visual_position_ids(
        self,
        position_ids,       # [3, B, L_compressed]  长度已对，视觉段坐标是 1D 顺序值
        image_mask_1d,      # [B, L_compressed]
        video_mask_1d,
        histo_mask_1d,
        image_idxs,         # list[Tensor[K_i]]  selector 选的位置（在原始 grid 里的坐标）
        video_idxs,
        histo_idxs,
        image_grid_thw,     # 原始完整 grid
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
                # 当前视觉段的起始 offset（三通道相同）
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

    def _extract_text_concepts(
        self,
        input_ids: torch.LongTensor,        # (B, SeqLen)
        inputs_embeds: torch.FloatTensor,    # (B, SeqLen, D)
    ) -> torch.FloatTensor:
        """
        Extract pure text token embeddings as concept queries for selectors.
        Excludes all visual placeholder tokens (image/video/histo).

        Returns:
            text_embeds: (B, m, D) where m = max text token count in batch,
                        zero-padded for shorter sequences.
        """
        visual_ids = {
            self.config.image_token_id,
            self.config.video_token_id,
            self.config.histo_token_id,
        }
        # also exclude vision_start / vision_end bracket tokens
        if hasattr(self.config, 'vision_start_token_id'):
            visual_ids.add(self.config.vision_start_token_id)
        if hasattr(self.config, 'vision_end_token_id'):
            visual_ids.add(self.config.vision_end_token_id)

        B, L, D = inputs_embeds.shape

        # (B, L) bool mask: True = text token
        text_mask = torch.ones(B, L, dtype=torch.bool, device=input_ids.device)
        for vid in visual_ids:
            text_mask &= (input_ids != vid)

        # gather per-sample, pad to max length
        max_m = text_mask.sum(dim=1).max().item()
        text_embeds = inputs_embeds.new_zeros(B, max_m, D)

        for b in range(B):
            idx = text_mask[b].nonzero(as_tuple=True)[0]
            text_embeds[b, :idx.shape[0]] = inputs_embeds[b, idx]

        return text_embeds.detach()
    
    @check_model_inputs
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
        histo_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, BreastGPTOutput]:
        """
        input_ids / attention_mask / labels 由 collator 按 K 个占位符预处理好。
        position_ids 按原始 grid_thw 计算（保留空间信息），之后用 selector idx shrink。

        与 oriforward 的差异：
          1. visual_token_merge 返回的 selected_idxs 被保留
          2. position_ids 计算后调用 _shrink_position_ids 修正视觉段坐标
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        text_concepts = self._extract_text_concepts(input_ids, inputs_embeds)
        img_bmap = self._modality_batch_map(input_ids, self.config.image_token_id)
        vid_bmap = self._modality_batch_map(input_ids, self.config.video_token_id)
        histo_bmap = self._modality_batch_map(input_ids, self.config.histo_token_id)
        aux_loss = 0
        image_mask = None
        video_mask = None
        histo_mask = None

        dummy_input = True
        image_idxs = video_idxs = histo_idxs = None
        deepstack_image_embeds = deepstack_video_embeds = deepstack_histo_embeds = []

        if dummy_input and pixel_values is None and pixel_values_videos is None:
            dummy_img_bmap = [0] 
            dummy_image_grid_thw = torch.tensor([[inputs_embeds.shape[0], 16, 8]], dtype=torch.long, device=inputs_embeds.device)
            total_patches = dummy_image_grid_thw[0].prod().item()
            # pixel_values 的形状受 patch_embed 影响，这里粗略模拟其输入
            dim_pixels = self.config.vision_config.in_channels * self.config.vision_config.temporal_patch_size * self.config.vision_config.patch_size ** 2
            dummy_pixel_values = torch.zeros(total_patches, dim_pixels, dtype=torch.bfloat16, device=inputs_embeds.device)
            
            image_embeds_tuple, deepstack_image_embeds = self.get_image_features(dummy_pixel_values, dummy_image_grid_thw)
            image_embeds, _, deepstack_image_embeds, _, _ = self.visual_token_merge(
                text_concepts, image_embeds_tuple, deepstack_image_embeds, dummy_image_grid_thw, self.config.image_token_num, dummy_img_bmap
            )
            inputs_embeds = inputs_embeds + image_embeds.mean() * 0.0  
            debug(f'IMAGE DUMMY LOSS: {image_embeds.mean()}')
            if deepstack_image_embeds is not None and len(deepstack_image_embeds) > 0:
                for deepstack_image_embed in deepstack_image_embeds:
                    inputs_embeds = inputs_embeds + deepstack_image_embed.mean() * 0.0  
        else:
            # ── Image ─────────────────────────────────────────────────────────────
            if pixel_values is not None:
                image_embeds_tuple, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds, image_idxs, deepstack_image_embeds, _, loss = self.visual_token_merge(
                    text_concepts, image_embeds_tuple, deepstack_image_embeds, image_grid_thw, self.config.image_token_num,img_bmap
                )
                aux_loss += loss
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # ── Video ─────────────────────────────────────────────────────────────
            if pixel_values_videos is not None:
                video_embeds_tuple, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds, video_idxs, deepstack_video_embeds, _, loss = self.visual_token_merge(
                    text_concepts, video_embeds_tuple, deepstack_video_embeds, video_grid_thw, self.config.video_token_num,vid_bmap
                )
                aux_loss += loss
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # ── Histo ─────────────────────────────────────────────────────────────
        if input_features is not None and len(histo_bmap) > 0:
            input_features = [f.to(torch.bfloat16) for f in input_features]
            histo_embeds_list, loss, histo_idxs, deepstack_histo_embeds = self.get_histo_features(text_concepts, input_features, histo_bmap, return_intermediate=True)
            aux_loss += loss
            histo_embeds = torch.cat(histo_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, histo_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, histo_features=histo_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(histo_mask, histo_embeds)
        elif dummy_input:
            B = inputs_embeds.shape[0]
            dummy_input_features = [torch.zeros((self.config.histo_token_num, self.config.feature_dim), 
                                                dtype=inputs_embeds.dtype, device=inputs_embeds.device)] * B
            dummy_histo_bmap = list(range(B))  # ← 每个 feature 对应一个 batch sample
            histo_embeds_list, _, _, deepstack_histo_embeds = self.get_histo_features(
                text_concepts, dummy_input_features, dummy_histo_bmap, return_intermediate=True)
            # 强制绑定 Histo Tensor
            for his in histo_embeds_list:
                inputs_embeds = inputs_embeds + his.mean() * 0.0 
            for deepstack_histo_embed in deepstack_histo_embeds:
                inputs_embeds = inputs_embeds + deepstack_histo_embed.mean() * 0.0 
            debug(f'HISTO BUMMY LOSS: {histo_embeds_list[0].mean()}')
        # ── DeepStack masks ──────────────────────────
        visual_pos_masks = None
        deepstack_visual_embeds = None

        if image_mask is not None and video_mask is not None and histo_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            histo_mask = histo_mask[..., 0]
            visual_pos_masks = image_mask | video_mask | histo_mask
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            histo_mask_joint = histo_mask[visual_pos_masks]
            deepstack_visual_embeds = []
            for img_e, vid_e, hit_e in zip(deepstack_image_embeds, deepstack_video_embeds, deepstack_histo_embeds):
                joint = img_e.new_zeros(visual_pos_masks.sum(), img_e.shape[-1]).to(img_e.device)
                joint[image_mask_joint] = img_e
                joint[video_mask_joint] = vid_e
                joint[histo_mask_joint] = hit_e
                deepstack_visual_embeds.append(joint)

        elif image_mask is not None and video_mask is not None:  # ← elif
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            deepstack_visual_embeds = []
            for img_e, vid_e in zip(deepstack_image_embeds, deepstack_video_embeds):
                joint = img_e.new_zeros(visual_pos_masks.sum(), img_e.shape[-1]).to(img_e.device)
                joint[image_mask_joint] = img_e
                joint[video_mask_joint] = vid_e
                deepstack_visual_embeds.append(joint)

        elif image_mask is not None and histo_mask is not None:  # ← 补充两两组合
            image_mask = image_mask[..., 0]
            histo_mask = histo_mask[..., 0]
            visual_pos_masks = image_mask | histo_mask
            image_mask_joint = image_mask[visual_pos_masks]
            histo_mask_joint = histo_mask[visual_pos_masks]
            deepstack_visual_embeds = []
            for img_e, hit_e in zip(deepstack_image_embeds, deepstack_histo_embeds):
                joint = img_e.new_zeros(visual_pos_masks.sum(), img_e.shape[-1]).to(img_e.device)
                joint[image_mask_joint] = img_e
                joint[histo_mask_joint] = hit_e
                deepstack_visual_embeds.append(joint)

        elif video_mask is not None and histo_mask is not None:  # ← 补充两两组合
            video_mask = video_mask[..., 0]
            histo_mask = histo_mask[..., 0]
            visual_pos_masks = video_mask | histo_mask
            video_mask_joint = video_mask[visual_pos_masks]
            histo_mask_joint = histo_mask[visual_pos_masks]
            deepstack_visual_embeds = []
            for vid_e, hit_e in zip(deepstack_video_embeds, deepstack_histo_embeds):
                joint = vid_e.new_zeros(visual_pos_masks.sum(), vid_e.shape[-1]).to(vid_e.device)
                joint[video_mask_joint] = vid_e
                joint[histo_mask_joint] = hit_e
                deepstack_visual_embeds.append(joint)

        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds

        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        elif histo_mask is not None:
            visual_pos_masks = histo_mask[..., 0]
            deepstack_visual_embeds = deepstack_histo_embeds

        # ── position_ids：计算后用 idx shrink 视觉段为真实空间坐标 ─────────────
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()
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
                    input_ids, image_grid_thw, video_grid_thw, histo_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)

        # 2D mask（collator 已压缩，expand=False）
        image_mask_1d, video_mask_1d, histo_mask_1d = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, expand=False
        )
        position_ids = self._fix_visual_position_ids(
            position_ids,
            image_mask_1d, video_mask_1d, histo_mask_1d,
            image_idxs, video_idxs, histo_idxs,
            image_grid_thw, video_grid_thw, histo_grid_thw,
        )

        # ── LLM ──────────────────────────────────────────────────────────────

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )
            loss = loss + 0.01 * aux_loss

        return BreastGPTOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
        )