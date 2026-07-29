"""
pretrain.py - 预训练脚本（精简版，无进度条）
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
# 数据加载（流式）
# ============================================================
def file_generator(file_list, vocab, seq_len):
    """逐文件读取、编码、滑动窗口生成样本"""
    unk_id = vocab.get('<|unk|>', 3) if hasattr(vocab, 'get') else 3
    stride = seq_len // 8

    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if hasattr(vocab, 'encode') and callable(vocab.encode):
            ids = vocab.encode(text)
        else:
            ids = [vocab.get(ch, unk_id) for ch in text]

        for i in range(0, len(ids) - seq_len, stride):
            segment = ids[i:i+seq_len+1]
            if len(segment) < seq_len+1:
                continue
            x = segment[:-1]
            y = segment[1:]
            yield np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def create_pretrain_dataset(file_list, vocab, config, is_training=True):
    """创建 tf.data.Dataset 用于预训练"""
    dataset = tf.data.Dataset.from_generator(
        lambda: file_generator(file_list, vocab, config.seq_len),
        output_types=(tf.int32, tf.int32),
        output_shapes=((config.seq_len,), (config.seq_len,))
    )
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000).repeat()
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
# 训练函数（无进度条）
# ============================================================
def check_gradients_nan_inf(grads):
    for g in grads:
        if g is None:
            continue
        if tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
            return True
    return False


def create_optimizer(lr_schedule, config):
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.weight_decay,
        beta_1=0.9,
        beta_2=0.95,
        clipnorm=1.0
    )


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

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
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

        # 验证
        print("  📊 验证中...")
        val_loss = 0.0
        val_steps = 0
        for x, y in val_dataset:
            logits = model(x, training=False)
            val_loss += float(loss_fn(y, logits))
            val_steps += 1
            if val_steps >= 100:  # 从 50 增加到 100，更稳定
                break
        val_loss = val_loss / max(val_steps, 1)
        current_lr = lr_manager.get_current_lr(global_step)

        with writer.as_default():
            tf.summary.scalar('train_loss', avg_train_loss, step=global_step)
            tf.summary.scalar('val_loss', val_loss, step=global_step)
            tf.summary.scalar('learning_rate', current_lr, step=global_step)

        epoch_time = time.time() - epoch_start
        print(f"  ✅ Epoch {epoch+1} 完成 | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e} | 耗时: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_best_model_only(model, save_dir, is_best=True)
            print(f"  ⭐ 新的最佳模型! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠️ Val Loss 未提升 ({patience_counter}/{config.early_stop_patience})")

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

    # ✅ 固定为 12 个 epoch，让模型充分收敛
    config.pretrain_epochs = 12

    # ✅ 学习率降低到 1e-4，更稳定
    config.pretrain_lr = 1e-4

    if config.steps_per_epoch >= 8000:
        config.pretrain_epochs = max(config.pretrain_epochs, 10)
    elif config.steps_per_epoch >= 5000:
        config.pretrain_epochs = max(config.pretrain_epochs, 12)

    total_tokens = config.steps_per_epoch * config.batch_size * config.seq_len * config.pretrain_epochs
    print(f"📊 预训练自动配置: corpus={corpus_bytes/1e6:.1f}MB, "
          f"steps_per_epoch={config.steps_per_epoch}, epochs={config.pretrain_epochs}, "
          f"总训练tokens≈{total_tokens/1e6:.0f}M")
    print(f"📊 学习率: {config.pretrain_lr:.2e} (已降低)")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("🤖 Literature Transformer 预训练")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  TensorFlow: {tf.__version__}")
    print(f"🖥️  GPU: {tf.config.list_physical_devices('GPU')}")
    print("=" * 60)

    setup_environment()
    config = ModelConfig()
    
    # ✅ 调整超参数
    config.pretrain_lr = 1e-4
    config.dropout = 0.15
    config.early_stop_patience = 10
    
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

    # ✅ 关键修复：随机打乱文件列表，确保验证集分布与训练集一致
    random.shuffle(all_files)
    print(f"✅ 文件列表已随机打乱")

    split_idx = max(1, int(len(all_files) * 0.95))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    if not val_files:
        val_files = train_files[-1:]

    print(f"📂 训练文件: {len(train_files)} 个, 验证文件: {len(val_files)} 个")

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