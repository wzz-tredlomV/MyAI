"""
predict.py
适配 train.py 的推理脚本
支持字符级 vocab.json 和 BPE tokenizer.json
支持贪婪/Top-K/Top-P/Beam Search 采样
"""
import tensorflow as tf
from tensorflow import keras
import json
import numpy as np
import sys
import os
import time
from typing import List, Optional, Tuple

# ============================================================
# 导入模型定义
# ============================================================
try:
    from train import (
        LiteratureTransformer, ModelConfig, RotaryEmbedding,
        CustomLayerNorm, CustomMultiHeadAttention, CustomFFN,
        CustomTransformerBlock, load_vocab, load_model_keras
    )
    MODEL_SOURCE = "train"
except ImportError as e:
    print(f"无法导入训练模块: {e}")
    print("请确保 train.py 在同一目录下")
    sys.exit(1)


# ============================================================
# 文本生成器
# ============================================================
class TextGenerator:
    """文本生成器，支持贪婪/Top-K/Top-P/Beam Search"""

    def __init__(self, model_path: str, vocab_path: str, config_path: Optional[str] = None):
        keras.mixed_precision.set_global_policy("float32")

        # ✅ 加载词汇表（自动检测 BPE / 字符级）
        self.vocab = load_vocab(vocab_path)
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # 特殊 token ID
        self.pad_id = self.vocab.get('<|pad|>', 0)
        self.unk_id = self.vocab.get('<|unk|>', 3)
        self.bos_id = self.vocab.get('<|bos|>', 1)
        self.eos_id = self.vocab.get('<|eos|>', 2)
        self.user_id = self.vocab.get('<|user|>', 4)
        self.bot_id = self.vocab.get('<|bot|>', 5)

        # 加载模型
        self.model = self._load_model(model_path, config_path)

        # 获取配置
        if hasattr(self.model, 'config') and self.model.config is not None:
            self.config = self.model.config
        else:
            cfg = None
            candidates = []
            if config_path:
                candidates.append(config_path)
            candidates.extend([
                os.path.join(model_path, "config.json"),
                os.path.join(model_path, "best_model", "config.json"),
            ])
            for cp in candidates:
                if os.path.exists(cp):
                    with open(cp, 'r', encoding='utf-8') as f:
                        cfg = ModelConfig.from_dict(json.load(f))
                    break
            self.config = cfg or ModelConfig(
                vocab_size=self.vocab_size, seq_len=256,
                embed_dim=384, num_heads=6, num_layers=6
            )

        print(f"✓ 生成器初始化完成")
        print(f"  - 词汇表大小: {self.vocab_size}")
        print(f"  - 最大序列长度: {self.config.seq_len}")
        print(f"  - 模型来源: {MODEL_SOURCE}")

    def _load_model(self, model_path: str, config_path: Optional[str] = None):
        """加载模型，支持多种格式和路径结构"""

        # 1. 直接 .keras 文件
        direct_keras = os.path.join(model_path, "model.keras")
        if os.path.exists(direct_keras):
            try:
                model = keras.models.load_model(direct_keras)
                print(f"✓ 从 {direct_keras} 加载模型")
                return model
            except Exception as e:
                print(f"  ⚠️ 直接加载 .keras 失败: {e}")

        # 2. best_model 子目录
        best_keras = os.path.join(model_path, "best_model", "model.keras")
        if os.path.exists(best_keras):
            try:
                model = load_model_keras(model_path)
                print(f"✓ 从 {best_keras} 加载模型")
                return model
            except Exception as e:
                print(f"  ⚠️ best_model/.keras 加载失败: {e}")

        # 3. SavedModel
        savedmodel_path = os.path.join(model_path, "savedmodel")
        if os.path.exists(savedmodel_path):
            try:
                model = tf.saved_model.load(savedmodel_path)
                print(f"✓ 从 {savedmodel_path} 加载模型")
                return model
            except Exception as e:
                print(f"  ⚠️ SavedModel 加载失败: {e}")

        # 4. config + weights 重建
        cfg_path = None
        if config_path and os.path.exists(config_path):
            cfg_path = config_path
        else:
            for p in [os.path.join(model_path, "config.json"),
                      os.path.join(model_path, "best_model", "config.json")]:
                if os.path.exists(p):
                    cfg_path = p
                    break

        if cfg_path:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                config = ModelConfig.from_dict(json.load(f))
            model = LiteratureTransformer(config)
            dummy = tf.constant([[self.bos_id]], dtype=tf.int32)
            _ = model(dummy, training=False)

            for wp in [os.path.join(model_path, "weights.h5"),
                       os.path.join(model_path, "best_model", "weights.h5")]:
                if os.path.exists(wp):
                    model.load_weights(wp)
                    print(f"✓ 从 {wp} 加载权重")
                    return model

        raise FileNotFoundError(
            f"无法在 {model_path} 找到可用模型文件。\n"
            f"请确保目录下存在以下任一文件：\n"
            f"  - model.keras\n"
            f"  - best_model/model.keras\n"
            f"  - savedmodel/\n"
            f"  - config.json + weights.h5"
        )

    # ========================================================
    # 编码 / 解码（✅ 适配 BPE 和字符级）
    # ========================================================
    def encode_chars(self, text: str) -> List[int]:
        """编码文本，兼容 BPE 和字符级"""
        return self.vocab.encode(text)

    def decode_tokens(self, tokens: List[int], skip_special: bool = True) -> str:
        """解码 token 列表，兼容 BPE 和字符级"""
        return self.vocab.decode(tokens, skip_special_tokens=skip_special)

    def build_chat_ids(self, prompt: str, system_prompt: Optional[str] = None) -> List[int]:
        """
        构建与训练时格式完全一致的输入序列：
        [bos, <|user|>, prompt..., <|bot|>]
        """
        ids = [self.bos_id, self.user_id]
        if system_prompt:
            ids.extend(self.encode_chars(system_prompt))
            ids.append(self.bot_id)
        ids.extend(self.encode_chars(prompt))
        ids.append(self.bot_id)
        return ids

    # ========================================================
    # 核心采样函数
    # ========================================================
    def _sample_next_token(self, logits: tf.Tensor,
                           temperature: float = 1.0,
                           top_k: Optional[int] = None,
                           top_p: Optional[float] = None) -> int:
        """从 logits 中采样下一个 token"""
        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        if top_k is not None and top_k > 0:
            top_k = min(top_k, self.vocab_size)
            kth_val = tf.math.top_k(logits, top_k)[0][..., -1]
            logits = tf.where(logits < kth_val, -1e9, logits)

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits = tf.sort(logits, direction='DESCENDING')
            sorted_probs = tf.nn.softmax(sorted_logits)
            cumulative_probs = tf.cumsum(sorted_probs)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove = tf.concat([
                tf.zeros_like(sorted_indices_to_remove[:1], dtype=tf.bool),
                sorted_indices_to_remove[:-1]
            ], axis=-1)

            sorted_indices = tf.argsort(logits, direction='DESCENDING')
            indices_to_remove = tf.scatter_nd(
                sorted_indices[..., None],
                tf.cast(sorted_indices_to_remove, tf.int32),
                [self.vocab_size]
            )
            indices_to_remove = tf.cast(indices_to_remove, tf.bool)
            logits = tf.where(indices_to_remove, -1e9, logits)

        next_token = tf.random.categorical(logits[None, :], 1)[0, 0].numpy()
        return int(next_token)

    # ========================================================
    # 生成接口
    # ========================================================
    def generate(self,
                 prompt_ids: List[int],
                 max_length: int = 100,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 stop_ids: Optional[List[int]] = None,
                 echo: bool = False) -> Tuple[str, List[int], float]:
        """自回归生成"""
        if stop_ids is None:
            stop_ids = [self.eos_id]

        start_time = time.time()
        generated = list(prompt_ids)

        for _ in range(max_length):
            if len(generated) >= self.config.seq_len:
                break

            input_ids = np.array([generated[-self.config.seq_len:]], dtype=np.int32)
            logits = self.model(input_ids, training=False)
            last_logits = logits[0, -1, :self.vocab_size]

            next_token = self._sample_next_token(
                last_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            if next_token in stop_ids:
                break

            generated.append(next_token)

        output_ids = generated if echo else generated[len(prompt_ids):]
        text = self.decode_tokens(output_ids)
        elapsed = time.time() - start_time
        return text, output_ids, elapsed

    def chat(self,
             prompt: str,
             max_length: int = 200,
             temperature: float = 0.8,
             top_k: int = 50,
             top_p: float = 0.9,
             system_prompt: Optional[str] = None) -> str:
        """聊天接口"""
        prompt_ids = self.build_chat_ids(prompt, system_prompt=system_prompt)
        text, _, _ = self.generate(
            prompt_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_ids=[self.eos_id, self.user_id],
            echo=False
        )
        return text.strip()

    def generate_from_text(self,
                           prompt: str,
                           max_length: int = 100,
                           temperature: float = 0.8,
                           top_k: int = 50,
                           top_p: float = 0.9) -> Tuple[str, float]:
        """从纯文本提示生成"""
        prompt_ids = [self.bos_id] + self.encode_chars(prompt)
        text, _, elapsed = self.generate(
            prompt_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_ids=[self.eos_id]
        )
        return text, elapsed

    # ========================================================
    # Beam Search
    # ========================================================
    def generate_beam(self,
                      prompt_ids: List[int],
                      max_length: int = 100,
                      beam_width: int = 3,
                      temperature: float = 1.0,
                      stop_ids: Optional[List[int]] = None) -> Tuple[str, float]:
        """束搜索生成"""
        if stop_ids is None:
            stop_ids = [self.eos_id]

        start_time = time.time()
        beams = [(list(prompt_ids), 0.0)]
        completed = []

        for _ in range(max_length):
            candidates = []
            all_done = True

            for tokens, score in beams:
                if tokens[-1] in stop_ids:
                    completed.append((tokens, score))
                    continue
                all_done = False

                input_ids = np.array([tokens[-self.config.seq_len:]], dtype=np.int32)
                logits = self.model(input_ids, training=False)
                last_logits = logits[0, -1, :self.vocab_size]

                if temperature != 1.0:
                    last_logits = last_logits / temperature

                log_probs = tf.nn.log_softmax(last_logits).numpy()
                top_indices = np.argsort(log_probs)[-beam_width:][::-1]

                for idx in top_indices:
                    new_tokens = tokens + [int(idx)]
                    new_score = score + log_probs[idx]
                    candidates.append((new_tokens, new_score))

            if all_done:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

        if completed:
            best = max(completed, key=lambda x: x[1] / len(x[0]))
        else:
            best = max(beams, key=lambda x: x[1])

        output_ids = best[0][len(prompt_ids):]
        text = self.decode_tokens(output_ids)
        return text, time.time() - start_time


# ============================================================
# 交互模式
# ============================================================
def interactive_mode(generator: TextGenerator):
    print("\n" + "=" * 50)
    print("🤖 交互模式 (输入 'quit' 退出)")
    print("=" * 50)
    print("命令:")
    print("  /temp <0.1~2.0>  设置温度")
    print("  /topk <int>      设置 Top-K (0=禁用)")
    print("  /topp <0.0~1.0>  设置 Top-P (1.0=禁用)")
    print("  /len <int>       设置最大长度")
    print("  /beam <int>      设置束宽度 (0=贪婪)")
    print("  /system <text>   设置系统提示词")
    print("  /clear           清空系统提示词")
    print("-" * 50)

    settings = {
        'temp': 0.8,
        'topk': 50,
        'topp': 0.9,
        'len': 150,
        'beam': 0,
        'system': None
    }

    while True:
        try:
            user_input = input("\n>>> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ('quit', 'exit', 'q'):
                print("再见！")
                break

            if user_input.startswith('/'):
                parts = user_input.split(None, 1)
                cmd = parts[0].lower()
                val = parts[1] if len(parts) > 1 else ''

                if cmd == '/temp':
                    settings['temp'] = float(val)
                    print(f"温度: {settings['temp']}")
                elif cmd == '/topk':
                    settings['topk'] = int(val)
                    print(f"Top-K: {settings['topk']}")
                elif cmd == '/topp':
                    settings['topp'] = float(val)
                    print(f"Top-P: {settings['topp']}")
                elif cmd == '/len':
                    settings['len'] = int(val)
                    print(f"最大长度: {settings['len']}")
                elif cmd == '/beam':
                    settings['beam'] = int(val)
                    print(f"束宽度: {settings['beam']}")
                elif cmd == '/system':
                    settings['system'] = val
                    print(f"系统提示: {val}")
                elif cmd == '/clear':
                    settings['system'] = None
                    print("系统提示已清空")
                else:
                    print(f"未知命令: {cmd}")
                continue

            print("生成中...")
            if settings['beam'] > 1:
                prompt_ids = generator.build_chat_ids(user_input, settings['system'])
                text, elapsed = generator.generate_beam(
                    prompt_ids,
                    max_length=settings['len'],
                    beam_width=settings['beam'],
                    temperature=settings['temp']
                )
            else:
                text = generator.chat(
                    user_input,
                    max_length=settings['len'],
                    temperature=settings['temp'],
                    top_k=settings['topk'],
                    top_p=settings['topp'],
                    system_prompt=settings['system']
                )
                elapsed = 0.0

            print(f"\n💬 回复 ({elapsed:.2f}s):")
            print("-" * 50)
            print(text)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n已中断")
            continue
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            continue


# ============================================================
# 命令行入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="文学语言模型文本生成（支持 BPE/字符级）")
    parser.add_argument("--model_path", type=str, default="output",
                        help="模型目录路径")
    parser.add_argument("--vocab_path", type=str, default="tokenizer.json",
                        help="词汇表路径（tokenizer.json 或 vocab.json）")
    parser.add_argument("--config_path", type=str, default=None,
                        help="配置文件路径（可选）")
    parser.add_argument("--prompt", type=str, help="单次生成提示文本")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P")
    parser.add_argument("--beam_width", type=int, default=0, help="束宽度")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--system", type=str, default=None, help="系统提示词")
    args = parser.parse_args()

    # 自动检测词表
    if not os.path.exists(args.vocab_path):
        alt = args.vocab_path.replace("tokenizer.json", "vocab.json")
        if os.path.exists(alt):
            args.vocab_path = alt

    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        print("可用路径:")
        for p in ["output", "saved_model/rl_final", "saved_model/sft_final", "saved_model/pretrain_final"]:
            if os.path.exists(p):
                print(f"  ✓ {p}")
        sys.exit(1)

    if not os.path.exists(args.vocab_path):
        print(f"❌ 词汇表不存在: {args.vocab_path}")
        sys.exit(1)

    print("正在加载模型...")
    generator = TextGenerator(args.model_path, args.vocab_path, args.config_path)

    if args.interactive:
        interactive_mode(generator)
        return

    if not args.prompt:
        print("请提供 --prompt 或使用 --interactive 交互模式")
        print('示例: python predict.py --prompt "你好"')
        sys.exit(1)

    print(f"提示: {args.prompt}")
    print(f"参数: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")

    try:
        if args.beam_width > 1:
            prompt_ids = generator.build_chat_ids(args.prompt, args.system)
            text, elapsed = generator.generate_beam(
                prompt_ids,
                max_length=args.max_length,
                beam_width=args.beam_width,
                temperature=args.temperature
            )
            print(f"\n[束搜索 宽度={args.beam_width}] ({elapsed:.2f}s)")
        else:
            text, elapsed = generator.generate_from_text(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            print(f"\n[贪婪/采样] ({elapsed:.2f}s)")

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
