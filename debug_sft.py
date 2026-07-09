# debug_sft.py
import json
import numpy as np
from train import SFTDataGenerator, ModelConfig, load_vocab

def debug_sft_generator():
    print("=" * 60)
    print("🔍 深度调试 SFTDataGenerator")
    print("=" * 60)
    
    # 加载词汇表
    vocab = load_vocab("vocab.json")
    print(f"词汇表大小: {len(vocab)}")
    
    # 配置
    config = ModelConfig(seq_len=256, batch_size=2)
    
    # 创建生成器
    gen = SFTDataGenerator("sft_train.jsonl", vocab, config, max_samples=20)
    
    print(f"\n样本数: {len(gen.samples)}")
    
    # 检查每个样本的 prompt_len 和实际内容
    print("\n📊 样本详情:")
    for i, sample in enumerate(gen.samples[:5]):
        x = sample['x']
        y = sample['y']
        prompt_len = sample['prompt_len']
        
        print(f"\n样本 {i+1}:")
        print(f"  x 长度: {len(x)}")
        print(f"  y 长度: {len(y)}")
        print(f"  prompt_len: {prompt_len}")
        print(f"  x 前20个token: {x[:20]}")
        print(f"  y 前20个token: {y[:20]}")
        print(f"  x 后20个token: {x[-20:]}")
        print(f"  y 后20个token: {y[-20:]}")
        
        # 检查 response 部分（prompt_len 之后）
        if len(x) > prompt_len:
            response_part = x[prompt_len:prompt_len+20]
            print(f"  response部分 (前20个): {response_part}")
            print(f"  response长度: {len(x) - prompt_len}")
        else:
            print(f"  ⚠️ 没有response部分!")
    
    # 检查生成的 batch
    print("\n" + "=" * 60)
    print("📊 检查生成的 batch")
    print("=" * 60)
    
    for batch_idx, (x, y, mask) in enumerate(gen()):
        if batch_idx >= 1:
            break
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  x shape: {x.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  mask shape: {mask.shape}")
        print(f"  mask sum: {np.sum(mask)}")
        
        for i in range(min(2, x.shape[0])):
            print(f"\n  样本 {i+1}:")
            print(f"    mask 前20个: {mask[i][:20]}")
            print(f"    mask 中段 (100-120): {mask[i][100:120]}")
            print(f"    mask 后20个: {mask[i][-20:]}")
            print(f"    x 前20个token: {x[i][:20]}")
            print(f"    y 前20个token: {y[i][:20]}")
            
            # 找出 mask 中第一个 1 的位置
            mask_ones = np.where(mask[i] == 1.0)[0]
            if len(mask_ones) > 0:
                print(f"    ✅ 第一个 mask=1 的位置: {mask_ones[0]}")
                print(f"    mask=1 的数量: {len(mask_ones)}")
            else:
                print(f"    ❌ 没有 mask=1 的位置!")

if __name__ == "__main__":
    debug_sft_generator()
