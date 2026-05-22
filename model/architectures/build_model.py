import argparse
import os
import re
from typing import Any, Tuple, cast

import torch
from transformers import AutoProcessor, AutoTokenizer
from tokenizers import AddedToken
from .breastgpt import BreastGPT, BreastGPTConfig
from .breastgpt2 import BreastGPT2, BreastGPT2Config


def hack_tokenizer_json(model_dir):
    import json
    tokenizer_file = os.path.join(model_dir, "tokenizer.json")
    
    if not os.path.exists(tokenizer_file):
        print("没找到 tokenizer.json！")
        return
        
    print("1. 正在读取核心 tokenizer.json...")
    with open(tokenizer_file, "r", encoding="utf-8") as f:
        data = json.load(f)
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

def add_hist_token(tokenizer, model):
    # ================= 优雅扩充核心逻辑 =================
    
    # 1. 向 Tokenizer 注册新的特殊占位符
    new_token = AddedToken("<|histo_pad|>", special=True, lstrip=False, rstrip=False, normalized=False)
    # 常规添加
    tokenizer.add_special_tokens({"additional_special_tokens": [new_token]})
    new_id = tokenizer.convert_tokens_to_ids("<|histo_pad|>")
    # 🔥 核心优雅操作：在内存中强行绑定到解码字典，绕过 Qwen 的序列化 Bug
    if not hasattr(tokenizer, "added_tokens_decoder"):
        tokenizer.added_tokens_decoder = {}
    # 将新 ID 和其对应的对象强行关联
    tokenizer.added_tokens_decoder[new_id] = new_token
    print(f"   ✅ 成功分配专属 ID: {new_id}")

    # 2. 扩充模型的 Embedding 矩阵大小
    model.resize_token_embeddings(len(tokenizer))
    
    # 3. 🔥 高级技巧：平滑初始化 (Smooth Initialization) 🔥
    # 获取新旧 Token 的 ID
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"新 Token ID: {new_id}, 参考 Token ID: {image_pad_id}")
    
    # 将 <|image_pad|> 的嵌入权重克隆给 <|histo_pad|>，防止初始训练 Loss 爆炸
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        embeddings[new_id] = embeddings[image_pad_id].clone()
        
        # 如果模型有独立的输出层（lm_head），最好也同步克隆
        if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
            output_embeddings = model.get_output_embeddings().weight
            output_embeddings[new_id] = output_embeddings[image_pad_id].clone()
            
    print("✅ 权重平滑初始化完成！")
    return tokenizer, model

def build_model(ori_config, mllm_path, save_model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BreastGPTConfig.from_json_file(ori_config)

    print('\n=== load tokenizer ===')
    tokenizer = AutoTokenizer.from_pretrained(mllm_path)
    
    print('\n=== load processor ===')
    processor = AutoProcessor.from_pretrained(mllm_path)
    # create model
    print('\n=== create model ===')
    model = BreastGPT(config)
    print(model)
    print('\n=== load original model config ===')
    model.load_ori_state_dict(mllm_path)

    tokenizer, model = add_hist_token(tokenizer, model)
    # save tokenizer and processor
    os.makedirs(save_model_path, exist_ok=True)
    tokenizer.save_pretrained(save_model_path)
    processor.save_pretrained(save_model_path)
    model.save_pretrained(save_model_path)
    print(f'Model has been saved to {save_model_path}')
    hack_tokenizer_json(save_model_path)
    return save_model_path

def build(config):
    savename = config.replace('config_', '').replace('.json', '')
    s = int(re.search(r'(\d+)B', savename).group(1))

    build_model(f'model/architectures/{config}', 
                f'modelscope/models/Qwen/Qwen3-VL-{s}B-Instruct', 
                f'model/architectures/saved_models/{savename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_breastgpt_8B.json')
    args = parser.parse_args()
    build(args.config)

    
