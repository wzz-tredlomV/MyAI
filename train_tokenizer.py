"""
train_tokenizer.py
训练 Byte-level BPE Tokenizer
功能：
  1. 自动检测文件编码 (GBK/ANSI/Big5/Latin-1/UTF-8 等) 并转为 UTF-8 保存
  2. 根据语料大小自动推荐 vocab_size
  3. 原生支持中英文混合
"""
import os
import glob
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.normalizers import NFKC

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|user|>", "<|bot|>"]


def auto_decode(fp):
    """尝试多种编码读取文件，返回 (内容, 编码名)"""
    for enc in ['utf-8', 'gbk', 'gb2312', 'cp936', 'big5', 'latin-1', 'cp1252']:
        try:
            with open(fp, 'r', encoding=enc) as f:
                return f.read(), enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None, None


def convert_corpus_to_utf8(input_dir, output_dir=None, inplace=False):
    """
    将语料目录下所有 .txt 转为 UTF-8 并保存
    
    Returns:
        str: 转换后的目录路径
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not files:
        raise ValueError(f"在 {input_dir} 中未找到 .txt 文件")

    if inplace:
        target_dir = input_dir
        print(f"⚠️  直接覆盖原文件模式")
    else:
        target_dir = output_dir or (input_dir + "_utf8")
        os.makedirs(target_dir, exist_ok=True)

    total_size = 0
    converted = 0
    already_ok = 0
    failed = 0

    for fp in files:
        fname = os.path.basename(fp)
        fsize = os.path.getsize(fp)
        total_size += fsize

        content, enc = auto_decode(fp)
        if content is None:
            print(f"  ❌ {fname}: 无法识别编码，跳过")
            failed += 1
            continue

        out_path = os.path.join(target_dir, fname)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)

        if enc == 'utf-8':
            already_ok += 1
        else:
            print(f"  🔄 {fname}: {enc:8s} -> UTF-8 ({fsize/1024:.0f} KB)")
            converted += 1

    print(f"\n{'='*50}")
    print(f"📊 转码完成: {len(files)} 个文件")
    print(f"   已是 UTF-8: {already_ok} | 已转换: {converted} | 失败: {failed}")
    print(f"   总大小: {total_size/1024/1024:.1f} MB")
    print(f"   输出目录: {target_dir}/")
    print(f"{'='*50}\n")

    return target_dir, total_size


def auto_vocab_size(total_bytes):
    """
    根据语料总大小自动推荐 vocab_size
    
    经验规则：
      - < 10 MB    ->  6000
      - 10-50 MB   ->  8000
      - 50-200 MB  ->  12000
      - 200-500 MB ->  16000
      - 500 MB-1 GB->  24000
      - > 1 GB     ->  32000 (上限，避免嵌入层过大)
    """
    total_mb = total_bytes / (1024 * 1024)

    if total_mb < 10:
        size = 6000
    elif total_mb < 50:
        size = 8000
    elif total_mb < 200:
        size = 12000
    elif total_mb < 500:
        size = 16000
    elif total_mb < 1024:
        size = 24000
    else:
        size = 32000

    # 向上取整到最近的 1000，方便记忆
    size = ((size + 999) // 1000) * 1000
    return size


def train_bpe(corpus_dir, output_path="tokenizer.json", vocab_size=None, inplace=False):
    # 1. 自动转 UTF-8
    train_dir, total_size = convert_corpus_to_utf8(corpus_dir, inplace=inplace)

    # 2. 自动计算 vocab_size
    if vocab_size is None or vocab_size <= 0:
        vocab_size = auto_vocab_size(total_size)
        print(f"🧠 自动调整词汇表大小: {vocab_size} (基于语料 {total_size/1024/1024:.1f} MB)")
    else:
        print(f"🧠 用户指定词汇表大小: {vocab_size}")

    # 3. 准备 tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.normalizer = NFKC()

    files = sorted(glob.glob(os.path.join(train_dir, "*.txt")))
    print(f"📂 使用 {len(files)} 个 UTF-8 文件训练 BPE...")
    print(f"🚀 开始训练 (vocab_size={vocab_size})...\n")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train(files, trainer)
    tokenizer.save(output_path)

    vocab = tokenizer.get_vocab()
    actual_size = tokenizer.get_vocab_size()
    print(f"\n✅ Tokenizer 已保存到: {output_path}")
    print(f"   实际词汇表大小: {actual_size}")

    print("\n📋 特殊 token 检查:")
    for tok in SPECIAL_TOKENS:
        if tok in vocab:
            print(f"   ✓ {tok:12s} -> id {vocab[tok]}")
        else:
            print(f"   ✗ {tok:12s} -> 缺失!")

    # 4. 快速测试（中英文混合）
    print("\n🧪 快速测试:")
    tests = [
        "Hello world!",
        "你好，世界！",
        "混合中英文: Hello 你好",
        "中华人民共和国",
        "Transformer架构在NLP领域非常流行。",
        "🎉 Emoji测试: 中文+English+123",
    ]
    all_pass = True
    for text in tests:
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc.ids)
        ok = (dec == text)
        if not ok:
            all_pass = False
        mark = "✅" if ok else "❌"
        print(f"   {mark} Token数: {len(enc.ids):2d} | 原文: {text}")
        print(f"      解码: {dec}")

    if all_pass:
        print("\n🎉 所有测试通过！Tokenizer 工作正常。")
    else:
        print("\n⚠️  部分测试未通过，请检查语料质量。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="训练 BPE Tokenizer（自动编码检测 + 自动词汇表大小）"
    )
    parser.add_argument("--corpus_dir", type=str, default="corpus",
                        help="预训练语料目录（支持混合编码）")
    parser.add_argument("--output", type=str, default="tokenizer.json",
                        help="输出文件名")
    parser.add_argument("--vocab_size", type=int, default=0,
                        help="词汇表大小（0=自动根据语料大小计算）")
    parser.add_argument("--inplace", action="store_true",
                        help="直接覆盖原文件为 UTF-8（默认保存到 corpus_utf8/）")
    args = parser.parse_args()

    train_bpe(
        args.corpus_dir,
        args.output,
        vocab_size=args.vocab_size,
        inplace=args.inplace
    )
