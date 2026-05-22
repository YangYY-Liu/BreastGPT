import argparse
import os
from typing import Any, Tuple, cast

import torch
from .breastgpt import BreastGPT, BreastGPTConfig, BreastGPT2Config, BreastGPT2


def build_model(ori_config, mllm_path, version=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create model
    print('\n=== create model ===')
    if version == 2:
        config = BreastGPT2Config()
        model = BreastGPT2(config).to(device).half()
    else:
        config = BreastGPTConfig.from_json_file(ori_config)
        model = BreastGPT(config).to(device).half()

    print('\n=== load original model config ===')
    # model.load_ori_state_dict(mllm_path)
    model.train()
    
    # A. 模拟全切片图像特征 (WSI Features)
    # 2. 构造 Dummy Inputs
    batch_size = 1
    
    # a. 构造 Text / Input IDs (包含文字与精准数量的占位符)
    # 模拟 Prompt: "诊断意见：[64个图片占位符] 组织学特征：[64个病理占位符]"
    text_prefix = torch.randint(0, 1000, (batch_size, 10)).to(device)
    img_placeholders = torch.full((batch_size, config.visual_token_num), config.image_token_id, dtype=torch.long).to(device)
    text_mid = torch.randint(0, 1000, (batch_size, 5)).to(device)
    histo_placeholders = torch.full((batch_size, config.histo_token_num), config.histo_token_id, dtype=torch.long).to(device)
    text_suffix = torch.randint(0, 1000, (batch_size, 10)).to(device)
    
    input_ids = torch.cat([text_prefix, img_placeholders, text_mid, histo_placeholders, text_suffix], dim=1)


    # b. 构造病理 WSI 超长特征 (Histo)
    # 模拟一张切片提取出了 15,000 个特征向量
    seq_len_histo = 15000 
    input_features = [torch.randn(seq_len_histo, config.feature_dim, dtype=torch.half).to(device)]

    # c. 构造常规影像特征 (Image)
    # Qwen-VL 视觉编码器接收的是 flatten 后的 patch 序列
    img_grid_thw = torch.tensor([[1, 28, 28]], dtype=torch.long).to(device) # 1帧, 28x28网格
    total_patches = img_grid_thw[0].prod().item()
    # pixel_values 的形状受 patch_embed 影响，这里粗略模拟其输入
    dim_pixels = config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size ** 2
    pixel_values = torch.randn(total_patches, dim_pixels, dtype=torch.bfloat16).to(device)

    print(f"\n=== 输入张量信息 ===")
    print(f"Input IDs 形状: {input_ids.shape}")
    print(f"病理特征 (Histo): 1 张切片, {seq_len_histo} 个 Token")
    print(f"影像特征 (Image): {total_patches} 个 Patches {pixel_values.shape}")

    # C. 构造 Attention Mask
    attention_mask = torch.ones_like(input_ids).to(device)
    # 3. 运行前向传播测试
    outputs = model(
        input_ids=input_ids,
        input_features=input_features,
        pixel_values=pixel_values,
        image_grid_thw=img_grid_thw,
        attention_mask=attention_mask,
        return_dict=True
    )

    print(f"输出 Hidden State 形状: {outputs.hidden_states.shape}")
    

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Forward 显存峰值: {max_mem:.2f} MB")

    # 4. 执行 Backward Pass 验证梯度流 (验证 STE 是否生效)
    print("\n=== 开始 Backward Pass (验证梯度连通性) ===")
    # 取最后一个 token 的 hidden state 算一个伪造的 Loss
    loss = outputs.hidden_states[:, -1, :].mean()
    loss.backward()

def hack_tokenizer_json(model_dir):
    import json
    tokenizer_file = os.path.join(model_dir, "tokenizer.json")
    
    if not os.path.exists(tokenizer_file):
        print("没找到 tokenizer.json！")
        return
        
    print("1. 正在读取核心 tokenizer.json...")
    with open(tokenizer_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    import pdb; pdb.set_trace()
    # 2. 注入到 added_tokens 列表 (告诉 Rust 引擎这是一个特殊词)
    added_tokens = data.get("added_tokens", [])
    if not any(t.get("content") == "<|histo_pad|>" for t in added_tokens):
        added_tokens.append({
            "id": 151669,
            "content": "<|histo_pad|>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        })
        data["added_tokens"] = added_tokens
        print("   ✅ 成功注入 added_tokens 列表！")

    # 4. 写回文件
    with open(tokenizer_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("🎉 tokenizer.json 终极物理篡改完成！")



def tokenizer():
    from transformers import AutoTokenizer

    # 加载你刚刚保存的新目录
    tokenizer = AutoTokenizer.from_pretrained("/nas/yangye.ly/breastGPT/ms-swift/architectures/saved_models/breastgpt_4B", trust_remote_code=True)

    # 打印新 Token 的 ID
    print("Histo Pad ID:", tokenizer.convert_tokens_to_ids("<|histo_pad|>"),  tokenizer.convert_tokens_to_ids("<|image_pad|>"))

    # 测试分词：看看它是不是被当成了一个整体，而不是被切碎的普通字符串
    print("Encode Test:", tokenizer.encode("这里有一个病理特征 <|histo_pad|> 测试"))

# if __name__ == '__main__':
#     model = build_model('/nas/yangye.ly/breastGPT/ms-swift/architectures/config_breastgpt_4B.json', 
#                         '/nas/yangye.ly/breastGPT/modelscope/models/Qwen/Qwen3-VL-4B-Instruct')

if __name__ == '__main__':
    # hack_tokenizer_json("/nas/yangye.ly/breastGPT/ms-swift/architectures/saved_models/breastgpt_4B")
    # tokenizer()
    build_model(
        '/nas/yangye.ly/breastGPT/ms-swift/architectures/config_breastgpt_4B.json',
        '/nas/yangye.ly/breastGPT/modelscope/models/Qwen/Qwen3-VL-4B-Instruct'
    )