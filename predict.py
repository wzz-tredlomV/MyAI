# predict.py
import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
import sys
import os
import time
from typing import List, Optional, Tuple

# 导入训练代码中的自定义类
from train import (
    LiteratureTransformer, ModelConfig, RotaryEmbedding,
    CustomLayerNorm, CustomMultiHeadAttention, CustomFFN,
    CustomTransformerBlock, load_model_keras, load_vocab
)

class TextGenerator:
    """文本生成器类，支持多种生成策略"""
    
    def __init__(self, model_path: str, vocab_path: str, config_path: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            model_path: 模型路径（包含model.keras的文件夹）
            vocab_path: 词汇表路径
            config_path: 配置文件路径（可选）
        """
        # 设置混合精度
        keras.mixed_precision.set_global_policy("float32")
        
        # 加载词汇表
        self.vocab = load_vocab(vocab_path)
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
        
        # 获取特殊token ID
        self.pad_id = self.vocab.get('<|pad|>', 0)
        self.unk_id = self.vocab.get('<|unk|>', 3)
        self.bos_id = self.vocab.get('<|bos|>', 1)
        self.eos_id = self.vocab.get('<|eos|>', 2)
        self.user_id = self.vocab.get('<|user|>', 4)
        self.bot_id = self.vocab.get('<|bot|>', 5)
        
        # 加载模型
        self.model = self._load_model(model_path, config_path)
        
        # 获取模型配置
        if hasattr(self.model, 'config'):
            self.config = self.model.config
        else:
            # 如果模型没有config属性，从配置文件加载
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                self.config = ModelConfig.from_dict(config_dict)
            else:
                # 使用默认配置
                self.config = ModelConfig(
                    vocab_size=len(self.vocab),
                    seq_len=256,
                    embed_dim=384,
                    num_heads=6,
                    num_layers=6
                )
        
        print(f"✓ 生成器初始化完成")
        print(f"  - 词汇表大小: {len(self.vocab)}")
        print(f"  - 最大序列长度: {self.config.seq_len}")
        print(f"  - 模型层数: {self.config.num_layers}")
    
    def _load_model(self, model_path: str, config_path: Optional[str] = None):
        """加载模型"""
        try:
            # 尝试直接加载.keras文件
            keras_path = os.path.join(model_path, "model.keras")
            if os.path.exists(keras_path):
                model = load_model_keras(model_path)
                print(f"✓ 从 {keras_path} 加载模型")
                return model
            
            # 尝试从SavedModel加载
            savedmodel_path = os.path.join(model_path, "savedmodel")
            if os.path.exists(savedmodel_path):
                model = tf.saved_model.load(savedmodel_path)
                print(f"✓ 从 {savedmodel_path} 加载模型")
                return model
            
            # 尝试从config重建
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                config = ModelConfig.from_dict(config_dict)
                model = LiteratureTransformer(config)
                # 尝试加载权重
                weights_path = os.path.join(model_path, "weights.h5")
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                    print(f"✓ 从 {weights_path} 加载权重")
                    return model
            
            raise FileNotFoundError(f"无法在 {model_path} 找到模型文件")
            
        except Exception as e:
            print(f"✗ 加载模型失败: {e}")
            raise
    
    def encode_text(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        将文本编码为token IDs
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊token
        
        Returns:
            token IDs列表
        """
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)
        
        for ch in text:
            token_id = self.vocab.get(ch, self.unk_id)
            # 确保token ID在有效范围内
            if token_id >= len(self.vocab):
                token_id = self.unk_id
            tokens.append(token_id)
        
        return tokens
    
    def decode_tokens(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        将token IDs解码为文本
        
        Args:
            tokens: token IDs列表
            skip_special_tokens: 是否跳过特殊token
        
        Returns:
            解码后的文本
        """
        special_tokens = {self.pad_id, self.bos_id, self.eos_id, self.user_id, self.bot_id}
        chars = []
        for token in tokens:
            if skip_special_tokens and token in special_tokens:
                continue
            # 确保token在词汇表范围内
            if token < len(self.vocab):
                chars.append(self.idx_to_char.get(token, '<UNK>'))
            else:
                chars.append('<UNK>')
        return ''.join(chars)
    
    def generate_greedy(
        self, 
        prompt: str, 
        max_length: int = 100, 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[List[int]] = None
    ) -> Tuple[str, float]:
        """
        贪婪搜索生成文本
        
        Args:
            prompt: 提示文本
            max_length: 最大生成长度
            temperature: 温度参数（>0）
            top_k: Top-K采样
            top_p: Top-P采样（核采样）
            stop_tokens: 停止token列表
        
        Returns:
            (生成的文本, 生成时间)
        """
        start_time = time.time()
        
        # 编码输入
        input_ids = self.encode_text(prompt)
        input_ids = np.array([input_ids], dtype=np.int32)
        
        generated = input_ids[0].tolist()
        
        # 设置停止token
        if stop_tokens is None:
            stop_tokens = [self.eos_id]
        
        for _ in range(max_length):
            # 如果序列太长，截断
            if len(generated) >= self.config.seq_len:
                break
            
            # 准备输入
            current_input = np.array([generated], dtype=np.int32)
            
            # 前向传播
            logits = self.model(current_input, training=False)
            
            # 获取最后一个token的logits
            last_logits = logits[0, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                last_logits = last_logits / temperature
            
            # 采样策略
            if top_k is not None and top_k > 0:
                # Top-K采样
                indices_to_remove = last_logits < tf.math.top_k(last_logits, top_k)[0][..., -1, None]
                last_logits = tf.where(indices_to_remove, -float('inf'), last_logits)
            
            if top_p is not None and top_p < 1.0:
                # Top-P采样（核采样）
                sorted_logits = tf.sort(last_logits, direction='DESCENDING')
                sorted_probs = tf.nn.softmax(sorted_logits)
                cumulative_probs = tf.cumsum(sorted_probs)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = tf.concat(
                    [tf.zeros_like(sorted_indices_to_remove[:1]), sorted_indices_to_remove[:-1]], 
                    axis=-1
                )
                indices_to_remove = sorted_indices_to_remove
                last_logits = tf.where(indices_to_remove, -float('inf'), last_logits)
            
            # 采样下一个token
            probs = tf.nn.softmax(last_logits)
            next_token = tf.random.categorical(tf.math.log(probs)[None, :], 1)[0, 0].numpy()
            
            # 确保生成的token在有效范围内
            if next_token >= len(self.vocab):
                next_token = self.unk_id
            
            # 检查停止条件
            if next_token in stop_tokens:
                break
            
            generated.append(next_token)
        
        # 解码生成的文本
        generated_text = self.decode_tokens(generated)
        
        elapsed_time = time.time() - start_time
        return generated_text, elapsed_time
    
    def generate_beam_search(
        self, 
        prompt: str, 
        max_length: int = 100, 
        beam_width: int = 3,
        temperature: float = 1.0,
        stop_tokens: Optional[List[int]] = None
    ) -> Tuple[str, float]:
        """
        束搜索生成文本
        
        Args:
            prompt: 提示文本
            max_length: 最大生成长度
            beam_width: 束宽度
            temperature: 温度参数
            stop_tokens: 停止token列表
        
        Returns:
            (生成的文本, 生成时间)
        """
        start_time = time.time()
        
        # 编码输入
        input_ids = self.encode_text(prompt)
        
        # 初始化beam
        beam = [(input_ids, 0.0)]  # (tokens, score)
        
        if stop_tokens is None:
            stop_tokens = [self.eos_id]
        
        for _ in range(max_length):
            candidates = []
            
            for tokens, score in beam:
                # 如果已经结束，直接保留
                if tokens[-1] in stop_tokens:
                    candidates.append((tokens, score))
                    continue
                
                # 准备输入
                current_input = np.array([tokens], dtype=np.int32)
                
                # 前向传播
                logits = self.model(current_input, training=False)
                
                # 获取最后一个token的logits
                last_logits = logits[0, -1, :]
                
                # 应用温度
                if temperature != 1.0:
                    last_logits = last_logits / temperature
                
                # 计算概率
                probs = tf.nn.softmax(last_logits).numpy()
                
                # 选择top-k个候选
                top_k_indices = np.argsort(probs)[-beam_width:][::-1]
                
                for idx in top_k_indices:
                    # 确保token在有效范围内
                    if idx >= len(self.vocab):
                        idx = self.unk_id
                    new_tokens = tokens + [idx]
                    new_score = score + np.log(probs[idx] + 1e-10)
                    candidates.append((new_tokens, new_score))
            
            # 选择得分最高的beam_width个候选
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]
            
            # 如果所有beam都结束了，提前终止
            if all(tokens[-1] in stop_tokens for tokens, _ in beam):
                break
        
        # 选择得分最高的序列
        best_tokens, best_score = beam[0]
        
        # 解码生成的文本
        generated_text = self.decode_tokens(best_tokens)
        
        elapsed_time = time.time() - start_time
        return generated_text, elapsed_time
    
    def generate(
        self, 
        prompt: str, 
        max_length: int = 100, 
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        beam_width: Optional[int] = None,
        return_time: bool = False
    ) -> str:
        """
        通用的生成接口
        
        Args:
            prompt: 提示文本
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K采样参数
            top_p: Top-P采样参数
            beam_width: 如果指定，使用束搜索
            return_time: 是否返回生成时间
        
        Returns:
            生成的文本，如果return_time为True，返回(文本, 时间)
        """
        if beam_width is not None and beam_width > 1:
            text, elapsed = self.generate_beam_search(
                prompt, max_length, beam_width, temperature
            )
        else:
            text, elapsed = self.generate_greedy(
                prompt, max_length, temperature, top_k, top_p
            )
        
        if return_time:
            return text, elapsed
        return text
    
    def chat(
        self, 
        prompt: str, 
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        聊天接口，支持系统提示词
        
        Args:
            prompt: 用户输入
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K采样
            top_p: Top-P采样
            system_prompt: 系统提示词
        
        Returns:
            模型回复
        """
        if system_prompt:
            full_prompt = f"系统: {system_prompt}\n用户: {prompt}\n助手: "
        else:
            full_prompt = f"用户: {prompt}\n助手: "
        
        response, _ = self.generate_greedy(
            full_prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=[self.eos_id, self.user_id]
        )
        
        # 提取助手回复
        if "助手: " in response:
            response = response.split("助手: ")[-1]
        
        return response.strip()
    
    def generate_batch(
        self, 
        prompts: List[str], 
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        batch_size: int = 8
    ) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 提示列表
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K采样
            top_p: Top-P采样
            batch_size: 批次大小
        
        Returns:
            生成的文本列表
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                text, _ = self.generate_greedy(
                    prompt, max_length, temperature, top_k, top_p
                )
                batch_results.append(text)
            
            results.extend(batch_results)
        
        return results

def interactive_mode(generator: TextGenerator):
    """交互模式"""
    print("\n" + "="*50)
    print("进入交互模式 (输入 'quit' 退出, 'help' 查看帮助)")
    print("="*50)
    print("\n支持的特殊命令:")
    print("  /temp <value>  - 设置温度 (0.1-2.0)")
    print("  /topk <value>  - 设置Top-K")
    print("  /topp <value>  - 设置Top-P")
    print("  /len <value>   - 设置最大生成长度")
    print("  /beam <value>  - 设置束宽度 (0=禁用束搜索)")
    
    # 默认参数
    temp = 0.8
    topk = 50
    topp = 0.9
    max_len = 100
    beam_width = 0
    
    while True:
        try:
            user_input = input("\n>>> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if user_input.lower() == 'help':
                print("\n可用命令:")
                print("  /temp 0.8    - 设置温度")
                print("  /topk 50     - 设置Top-K")
                print("  /topp 0.9    - 设置Top-P")
                print("  /len 100     - 设置最大长度")
                print("  /beam 3      - 设置束宽度")
                print("  直接输入文本 - 开始生成")
                continue
            
            # 解析命令
            if user_input.startswith('/'):
                parts = user_input.split()
                if len(parts) != 2:
                    print("格式: /命令 值")
                    continue
                
                cmd, value = parts[0].lower(), parts[1]
                try:
                    if cmd == '/temp':
                        temp = float(value)
                        print(f"温度设置为: {temp}")
                    elif cmd == '/topk':
                        topk = int(value)
                        print(f"Top-K设置为: {topk}")
                    elif cmd == '/topp':
                        topp = float(value)
                        print(f"Top-P设置为: {topp}")
                    elif cmd == '/len':
                        max_len = int(value)
                        print(f"最大长度设置为: {max_len}")
                    elif cmd == '/beam':
                        beam_width = int(value)
                        print(f"束宽度设置为: {beam_width}")
                    else:
                        print(f"未知命令: {cmd}")
                except ValueError:
                    print("无效的值")
                continue
            
            # 生成文本
            print("\n生成中...")
            start_time = time.time()
            
            if beam_width > 1:
                text, elapsed = generator.generate_beam_search(
                    user_input,
                    max_length=max_len,
                    beam_width=beam_width,
                    temperature=temp
                )
            else:
                text, elapsed = generator.generate_greedy(
                    user_input,
                    max_length=max_len,
                    temperature=temp,
                    top_k=topk,
                    top_p=topp
                )
            
            print(f"\n生成结果 ({elapsed:.2f}s):")
            print("-" * 50)
            print(text)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n中断生成")
            continue
        except Exception as e:
            print(f"错误: {e}")
            continue

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="文学语言模型文本生成")
    parser.add_argument("--model_path", type=str, default="saved_model/rl_final",
                       help="模型路径")
    parser.add_argument("--vocab_path", type=str, default="vocab.json",
                       help="词汇表路径")
    parser.add_argument("--config_path", type=str, default="saved_model/config.json",
                       help="配置文件路径")
    parser.add_argument("--prompt", type=str, help="提示文本")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K采样")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P采样")
    parser.add_argument("--beam_width", type=int, default=0, help="束宽度 (0=禁用)")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--batch", type=str, help="批量生成文件路径")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        print("可用的模型路径:")
        for path in ["saved_model/pretrain_final", "saved_model/sft_final", "saved_model/rl_final"]:
            if os.path.exists(path):
                print(f"  - {path}")
        sys.exit(1)
    
    if not os.path.exists(args.vocab_path):
        print(f"错误: 词汇表不存在: {args.vocab_path}")
        sys.exit(1)
    
    # 初始化生成器
    print("正在加载模型...")
    generator = TextGenerator(
        args.model_path,
        args.vocab_path,
        args.config_path
    )
    
    # 交互模式
    if args.interactive:
        interactive_mode(generator)
        return
    
    # 批量生成
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"错误: 批量文件不存在: {args.batch}")
            sys.exit(1)
        
        with open(args.batch, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"批量生成 {len(prompts)} 个提示...")
        results = generator.generate_batch(
            prompts,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # 保存结果
        output_file = "generated_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for prompt, result in zip(prompts, results):
                f.write(f"提示: {prompt}\n")
                f.write(f"生成: {result}\n")
                f.write("-" * 50 + "\n")
        
        print(f"结果已保存到: {output_file}")
        return
    
    # 单次生成
    if not args.prompt:
        print("请提供 --prompt 参数或使用 --interactive 交互模式")
        print("示例: python predict.py --prompt \"床前明月光\"")
        sys.exit(1)
    
    print(f"提示: {args.prompt}")
    print(f"参数: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    
    try:
        if args.beam_width > 1:
            text, elapsed = generator.generate_beam_search(
                args.prompt,
                max_length=args.max_length,
                beam_width=args.beam_width,
                temperature=args.temperature
            )
            print(f"束搜索 (宽度={args.beam_width})")
        else:
            text, elapsed = generator.generate_greedy(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            print("贪婪采样")
        
        print(f"\n生成结果 ({elapsed:.2f}s):")
        print("-" * 50)
        print(text)
        print("-" * 50)
    except Exception as e:
        print(f"生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()