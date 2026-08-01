"""
pretrain.py - 预训练脚本（改进版）
改进点：
  1. 数据生成器：随机 stride + 随机起始偏移，增加多样性
  2. 验证集划分：按文档级别划分，避免内容泄漏
  3. 损失函数：添加 Label Smoothing
  4. 验证评估：计算 Perplexity + Token-level Accuracy
  5. 优化器：beta_2=0.98, epsilon=1e-6
  6. 训练日志：更详细的指标输出
"""
import tensorflow as tf
from tensorflow import keras
import json
import os
import sys
import glob
import shutil
import time
import warnings
import random
from datetime import datetime
import numpy as np

from config import ModelConfig, WarmupCosineDecay, AdaptiveLRManager
from models import LiteratureTransformer

warnings.filterwarnings("ignore", category=SyntaxWarning)


# ============================================================
# 环境设置
# ============================================================
def setup_environment():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"  GPU 内存设置失败: {e}")
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


# ============================================================
# 数据加载（流式）—— 改进版
# ============================================================
def file_generator(file_list, vocab, seq_len, is_training=True):
    """
    逐文件读取、编码、滑动窗口生成样本
    改进：
      - 训练时随机 stride（seq_len//4 ~ seq_len//2）
      - 训练时随机起始偏移（0~min(100, len)）
      - 验证时固定 stride = seq_len // 8
    """
    unk_id = vocab.get('<|unk|>', 3) if hasattr(vocab, 'get') else 3

    # 训练时随机化，验证时固定
    if is_training:
        stride = random.randint(seq_len // 4, seq_len // 2)
    else:
        stride = seq_len // 8

    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if hasattr(vocab, 'encode') and callable(vocab.encode):
            ids = vocab.encode(text)
        else:
            ids = [vocab.get(ch, unk_id) for ch in text]

        # 训练时随机起始偏移
        if is_training:
            start_offset = random.randint(0, min(100, max(len(ids) - seq_len - 1, 0)))
        else:
            start_offset = 0

        for i in range(start_offset, len(ids) - seq_len, stride):
            segment = ids[i:i+seq_len+1]
            if len(segment) < seq_len+1:
                continue
            x = segment[:-1]
            y = segment[1:]
            yield np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def create_pretrain_dataset(file_list, vocab, config, is_training=True):
    """创建 tf.data.Dataset 用于预训练"""
    dataset = tf.data.Dataset.from_generator(
        lambda: file_generator(file_list, vocab, config.seq_len, is_training=is_training),
        output_types=(tf.int32, tf.int32),
        output_shapes=((config.seq_len,), (config.seq_len,))
    )
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    if is_training:
        dataset = dataset.shuffle(buffer_size=2000).repeat()  # buffer 从 1000 提升到 2000
    else:
        dataset = dataset.repeat(1)
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_vocab(path):
    """加载词汇表"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"词表文件不存在: {path}")
    from tokenizer_wrapper import TokenizerWrapper
    return TokenizerWrapper(path)


# ============================================================
# 保存/加载
# ============================================================
def save_best_model_only(model, save_dir, is_best=False):
    if not is_best:
        return
    best_dir = os.path.join(save_dir, "best_model")
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir)
    os.makedirs(best_dir, exist_ok=True)
    try:
        model.save(os.path.join(best_dir, "model.keras"))
        with open(os.path.join(best_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(model.config.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"  ✓ 最佳模型已保存到 {best_dir}")
    except Exception as e:
        print(f"  ⚠️ 保存失败: {e}")


def save_checkpoint(model, optimizer, epoch, step, save_dir):
    ckpt_dir = os.path.join(save_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_path = os.path.join(ckpt_dir, "latest")
    checkpoint.write(checkpoint_path)
    state = {"epoch": epoch, "step": step}
    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
        json.dump(state, f)


def load_checkpoint_if_exists(model, optimizer, save_dir):
    ckpt_dir = os.path.join(save_dir, "checkpoint")
    state_path = os.path.join(ckpt_dir, "training_state.json")
    if not os.path.exists(state_path):
        return None, 0, 0
    with open(state_path, "r") as f:
        state = json.load(f)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest:
        checkpoint.restore(latest)
        print(f"  ✅ 恢复训练状态: Epoch {state['epoch']}, Step {state['step']}")
        return state, state["epoch"], state["step"]
    return None, 0, 0


# ============================================================
# 训练函数（改进版）
# ============================================================
def check_gradients_nan_inf(grads):
    for g in grads:
        if g is None:
            continue
        if tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
            return True
    return False


def create_optimizer(lr_schedule, config):
    """改进：beta_2=0.98, epsilon=1e-6"""
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        beta_1=0.9,
        beta_2=0.98,          # 从 0.95 提升，更稳定的二阶矩估计
        epsilon=1e-6,         # 防止除零
        clipnorm=1.0
    )


# ✅ 新增：计算 Perplexity 和 Token-level Accuracy
def evaluate_metrics(model, val_dataset, loss_fn, config):
    """
    评估验证集指标：
      - Val Loss（按有效 token 加权）
      - Perplexity
      - Token-level Accuracy（忽略 padding）
    """
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    total_valid = 0
    steps = 0

    for x, y in val_dataset:
        logits = model(x, training=False)

        # 计算 loss（按有效 token 加权）
        mask = tf.cast(tf.not_equal(y, 0), tf.float32)

        # 使用与训练相同的 label smoothing 逻辑
        if config.label_smoothing > 0:
            vocab_size = tf.shape(logits)[-1]
            y_one_hot = tf.one_hot(y, depth=vocab_size)
            y_smooth = (1.0 - config.label_smoothing) * y_one_hot + config.label_smoothing / tf.cast(vocab_size, tf.float32)
            per_token_loss = tf.keras.losses.categorical_crossentropy(y_smooth, logits, from_logits=True)
        else:
            per_token_loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)

        # 忽略 padding
        per_token_loss = per_token_loss * mask
        batch_loss = tf.reduce_sum(per_token_loss)
        batch_tokens = tf.reduce_sum(mask)

        total_loss += float(batch_loss)
        total_tokens += float(batch_tokens)

        # Token-level Accuracy
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        correct = tf.cast(tf.equal(predictions, y), tf.float32) * mask
        correct_tokens += float(tf.reduce_sum(correct))
        total_valid += float(batch_tokens)

        steps += 1
        if steps >= config.val_max_steps:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    accuracy = correct_tokens / max(total_valid, 1)

    return avg_loss, perplexity, accuracy


def pretrain(model, train_dataset, val_dataset, vocab_size, config, save_dir="output/pretrain"):
    print("\n🚀 开始预训练")
    if config.enable_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("  ✅ 已启用 mixed_float16")

    total_steps = config.pretrain_epochs * config.steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)

    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=config.pretrain_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        alpha=config.min_lr_ratio
    )
    optimizer = create_optimizer(lr_schedule, config)
    lr_manager = AdaptiveLRManager(optimizer, lr_schedule, config, total_steps, config.pretrain_lr)

    _, start_epoch, global_step = load_checkpoint_if_exists(model, optimizer, save_dir)

    # ✅ 改进：Label Smoothing 手动实现（SparseCategoricalCrossentropy 不支持 label_smoothing）
    base_loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        ignore_class=0
    )

    def loss_fn(y_true, y_pred):
        """带 Label Smoothing 的稀疏交叉熵损失"""
        if config.label_smoothing > 0:
            # 获取 vocab_size
            vocab_size = tf.shape(y_pred)[-1]
            # 创建 one-hot 标签
            y_one_hot = tf.one_hot(y_true, depth=vocab_size)
            # 应用 label smoothing: (1 - smoothing) * one_hot + smoothing / vocab_size
            y_smooth = (1.0 - config.label_smoothing) * y_one_hot + config.label_smoothing / tf.cast(vocab_size, tf.float32)
            # 使用 categorical_crossentropy（from_logits=True）
            loss = tf.keras.losses.categorical_crossentropy(y_smooth, y_pred, from_logits=True)
            # 忽略 padding (class 0)
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            loss = loss * mask
            return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)
        else:
            return base_loss_fn(y_true, y_pred)

    best_val_loss = float('inf')
    patience_counter = 0
    writer = tf.summary.create_file_writer(os.path.join(config.log_dir, "pretrain"))

    for epoch in range(start_epoch, config.pretrain_epochs):
        epoch_start = time.time()
        accum_loss = 0.0
        train_steps = 0

        print(f"\n📊 Epoch {epoch+1}/{config.pretrain_epochs}")

        for step, (x, y) in enumerate(train_dataset):
            if step >= config.steps_per_epoch:
                break
            global_step += 1

            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = loss_fn(y, logits)
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps

            grads = tape.gradient(loss, model.trainable_variables)
            if check_gradients_nan_inf(grads):
                if lr_manager.on_nan_detected():
                    print("\n⏹️ NaN 容忍耗尽，停止训练")
                    return
                continue

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            accum_loss += float(loss)
            train_steps += 1
            lr_manager.on_step_end(loss)

            # 每 500 步打印一次进度
            if (step + 1) % 500 == 0:
                current_avg = accum_loss / train_steps
                current_lr = lr_manager.get_current_lr(global_step)
                print(f"  Step {step+1}/{config.steps_per_epoch} | Avg Loss: {current_avg:.4f} | LR: {current_lr:.2e}")

        avg_train_loss = accum_loss / max(train_steps, 1)

        # ✅ 改进：使用 evaluate_metrics 计算完整指标
        print("  📊 验证中...")
        val_loss, perplexity, accuracy = evaluate_metrics(model, val_dataset, loss_fn, config)
        current_lr = lr_manager.get_current_lr(global_step)

        with writer.as_default():
            tf.summary.scalar('train_loss', avg_train_loss, step=global_step)
            tf.summary.scalar('val_loss', val_loss, step=global_step)
            tf.summary.scalar('perplexity', perplexity, step=global_step)
            tf.summary.scalar('token_accuracy', accuracy, step=global_step)
            tf.summary.scalar('learning_rate', current_lr, step=global_step)

        epoch_time = time.time() - epoch_start
        # ✅ 改进：更详细的日志输出
        print(f"  ✅ Epoch {epoch+1} 完成 | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | PPL: {perplexity:.2f} | Acc: {accuracy:.4f} | "
              f"LR: {current_lr:.2e} | 耗时: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_best_model_only(model, save_dir, is_best=True)
            print(f"  ⭐ 新的最佳模型! (Val Loss: {val_loss:.4f}, PPL: {perplexity:.2f})")
        else:
            patience_counter += 1
            gap = val_loss - avg_train_loss
            print(f"  ⚠️ Val Loss 未提升 ({patience_counter}/{config.early_stop_patience}) | 过拟合差距: {gap:.4f}")

        # 早停判断
        should_stop = lr_manager.on_epoch_end(val_loss)
        if should_stop or patience_counter >= config.early_stop_patience:
            print(f"⏹️ 训练终止 (早停: {patience_counter}/{config.early_stop_patience})")
            break

        save_checkpoint(model, optimizer, epoch, global_step, save_dir)
        print(f"  💾 Checkpoint 已保存")


# ============================================================
# 自动配置（优化版）
# ============================================================
def auto_config_pretrain(config, corpus_bytes):
    """根据预训练语料总字节数自动调整"""
    stride = config.seq_len // 8
    approx_chars = corpus_bytes // 2
    total_samples = max(approx_chars // stride, 1)
    steps = total_samples // config.batch_size
    config.steps_per_epoch = min(max(steps, 500), 10000)

    # 固定为 12 个 epoch
    config.pretrain_epochs = 12

    # ✅ 改进：学习率已改为 5e-5（在 main 中设置）
    if config.steps_per_epoch >= 8000:
        config.pretrain_epochs = max(config.pretrain_epochs, 10)
    elif config.steps_per_epoch >= 5000:
        config.pretrain_epochs = max(config.pretrain_epochs, 12)

    total_tokens = config.steps_per_epoch * config.batch_size * config.seq_len * config.pretrain_epochs
    print(f"📊 预训练自动配置: corpus={corpus_bytes/1e6:.1f}MB, "
          f"steps_per_epoch={config.steps_per_epoch}, epochs={config.pretrain_epochs}, "
          f"总训练tokens≈{total_tokens/1e6:.0f}M")
    print(f"📊 学习率: {config.pretrain_lr:.2e} | Dropout: {config.dropout} | Weight Decay: {config.weight_decay}")
    print(f"📊 Label Smoothing: {config.label_smoothing} | Stochastic Depth: {config.stochastic_depth_rate}")


# ============================================================
# ✅ 改进：按文档级别划分训练/验证集
# ============================================================
def split_train_val_files(all_files, train_ratio=0.95, min_val_files=5):
    """
    按文档级别划分训练/验证集，避免内容泄漏。
    确保验证集至少有 min_val_files 个文件。
    """
    n = len(all_files)
    if n < min_val_files + 1:
        # 文件太少，无法划分，全部用于训练，验证也用训练集（会过拟合警告）
        print(f"⚠️ 文件数太少 ({n})，无法划分验证集")
        return all_files, all_files[-1:] if n > 0 else []

    # 随机打乱文件列表
    shuffled = all_files.copy()
    random.shuffle(shuffled)

    # 计算验证集大小（至少 min_val_files 个）
    val_size = max(min_val_files, int(n * (1 - train_ratio)))
    val_size = min(val_size, n // 5)  # 验证集不超过 20%

    train_files = shuffled[:-val_size]
    val_files = shuffled[-val_size:]

    return train_files, val_files


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("🤖 Literature Transformer 预训练（改进版）")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  TensorFlow: {tf.__version__}")
    print(f"🖥️  GPU: {tf.config.list_physical_devices('GPU')}")
    print("=" * 60)

    setup_environment()
    config = ModelConfig()

    # ✅ 改进：调整超参数
    config.pretrain_lr = 5e-5           # 从 1e-4 降低到 5e-5
    config.dropout = 0.25               # 从 0.15 提升到 0.25
    config.weight_decay = 0.1           # 从 0.01 提升到 0.1
    config.early_stop_patience = 10
    config.label_smoothing = 0.1        # 新增
    config.stochastic_depth_rate = 0.1  # 新增：10% 概率丢弃中间层
    config.val_max_steps = 500          # 验证步数

    base_dir = "/kaggle/working/MyAI"

    # 加载词表
    vocab_path = os.path.join(base_dir, "tokenizer.json")
    if not os.path.exists(vocab_path):
        vocab_path = os.path.join(base_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"❌ 词表不存在")
        sys.exit(1)

    vocab = load_vocab(vocab_path)
    config.vocab_size = len(vocab)
    print(f"📊 词表大小: {len(vocab)}")

    # BPE 优化
    if config.vocab_size > 15000:
        print("🧠 检测到 BPE 大词表，自动优化配置")
        if config.seq_len < 512:
            config.seq_len = 512

    # 获取预训练文件
    corpus_dir = os.path.join(base_dir, "corpus")
    if os.path.exists(corpus_dir):
        all_files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
        print(f"\n📂 发现 {len(all_files)} 个预训练文本文件")
    else:
        print(f"\n⚠️ 未找到 corpus 目录")
        sys.exit(1)

    total_bytes = sum(os.path.getsize(f) for f in all_files)
    print(f"📊 预训练语料总大小: {total_bytes/1e6:.1f}MB")

    auto_config_pretrain(config, total_bytes)

    # ✅ 改进：使用文档级别划分
    train_files, val_files = split_train_val_files(all_files, train_ratio=0.95, min_val_files=5)

    print(f"📂 训练文件: {len(train_files)} 个, 验证文件: {len(val_files)} 个")
    print(f"   验证文件列表: {[os.path.basename(f) for f in val_files[:3]]}{'...' if len(val_files) > 3 else ''}")

    # 创建 Dataset
    print("\n📊 创建训练数据集...")
    train_dataset = create_pretrain_dataset(train_files, vocab, config, is_training=True)
    val_dataset = create_pretrain_dataset(val_files, vocab, config, is_training=False)

    # 创建模型
    print("\n🏗️  创建模型...")
    model = LiteratureTransformer(config)
    dummy = tf.zeros((1, config.seq_len), dtype=tf.int32)
    _ = model(dummy)
    model.summary()
    print(f"📊 模型参数量: {model.count_params():,}")

    # 预训练
    try:
        pretrain(model, train_dataset, val_dataset, config.vocab_size, config,
                 save_dir=os.path.join(base_dir, "output", "pretrain"))
    except Exception as e:
        print(f"\n❌ 预训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ 预训练完成！")


if __name__ == "__main__":
    main()
