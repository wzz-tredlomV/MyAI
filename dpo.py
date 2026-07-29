"""
dpo.py - DPO 偏好对齐训练脚本
"""
import tensorflow as tf
from tensorflow import keras
import json
import os
import sys
import time
import warnings
from datetime import datetime
import numpy as np

from config import ModelConfig, WarmupCosineDecay, AdaptiveLRManager

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


# ============================================================
# 工具函数
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


def save_best_model_only(model, save_dir, is_best=False):
    if not is_best:
        return
    import shutil
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


def load_model_keras(load_dir):
    keras.mixed_precision.set_global_policy("float32")
    keras_path = os.path.join(load_dir, "best_model", "model.keras")
    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"找不到模型文件: {keras_path}")
    model = keras.models.load_model(keras_path)
    print(f"  模型已从 {keras_path} 加载")
    return model


def load_vocab(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"词表文件不存在: {path}")
    from tokenizer_wrapper import TokenizerWrapper
    return TokenizerWrapper(path)


def count_jsonl_lines(path):
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


# ============================================================
# DPO 数据生成器
# ============================================================
class DPODataGenerator:
    def __init__(self, jsonl_path, vocab, config: ModelConfig, max_pairs=None, pairs=None, infinite=False):
        self.vocab = vocab
        self.config = config
        self.pad_id = vocab.get('<|pad|>', 0)
        self.unk_id = vocab.get('<|unk|>', 3)
        self.bos_id = vocab.get('<|bos|>', 1)
        self.eos_id = vocab.get('<|eos|>', 2)
        self.user_id = vocab.get('<|user|>', 4)
        self.bot_id = vocab.get('<|bot|>', 5)
        self.infinite = infinite
        if pairs is not None:
            self.pairs = pairs
        else:
            self.pairs = self._load_data(jsonl_path, max_pairs)

    def _load_data(self, path, max_pairs=None):
        pairs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    chosen = self._encode(item['prompt'], item['chosen'])
                    rejected = self._encode(item['prompt'], item['rejected'])
                    if len(chosen) > 0 and len(rejected) > 0:
                        pairs.append((chosen, rejected))
                    if max_pairs and len(pairs) >= max_pairs:
                        break
        return pairs

    def _encode(self, prompt, response):
        ids = [self.bos_id, self.user_id]
        ids.extend(self.vocab.encode(prompt))
        ids.append(self.bot_id)
        ids.extend(self.vocab.encode(response))
        ids.append(self.eos_id)
        return ids

    def _pad_or_truncate(self, ids):
        if len(ids) < self.config.seq_len:
            return ids + [self.pad_id] * (self.config.seq_len - len(ids))
        return ids[:self.config.seq_len]

    def __call__(self):
        while True:
            indices = list(range(len(self.pairs)))
            np.random.shuffle(indices)
            batch_cx, batch_cy, batch_rx, batch_ry = [], [], [], []
            count = 0
            for idx in indices:
                chosen, rejected = self.pairs[idx]
                if len(chosen) < 2 or len(rejected) < 2:
                    continue
                cx = self._pad_or_truncate(chosen[:-1])
                cy = self._pad_or_truncate(chosen[1:])
                rx = self._pad_or_truncate(rejected[:-1])
                ry = self._pad_or_truncate(rejected[1:])
                batch_cx.append(cx)
                batch_cy.append(cy)
                batch_rx.append(rx)
                batch_ry.append(ry)
                count += 1
                if count == self.config.batch_size:
                    yield (np.array(batch_cx, dtype=np.int32),
                           np.array(batch_cy, dtype=np.int32),
                           np.array(batch_rx, dtype=np.int32),
                           np.array(batch_ry, dtype=np.int32))
                    batch_cx, batch_cy, batch_rx, batch_ry = [], [], [], []
                    count = 0
            if not self.infinite:
                break


def auto_config_dpo(config, num_pairs):
    steps = max(num_pairs // config.batch_size, 50)
    config.steps_per_epoch = min(steps, 2000)
    if config.steps_per_epoch >= 1000:
        config.rl_epochs = 5
    else:
        config.rl_epochs = 8
    print(f"📊 DPO 自动配置: pairs={num_pairs}, steps_per_epoch={config.steps_per_epoch}, epochs={config.rl_epochs}")


# ============================================================
# DPO 训练函数
# ============================================================
def dpo_train(model, ref_model, train_gen, config, save_dir="output/rl"):
    print("\n🚀 开始 DPO 训练")
    if config.enable_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    total_steps = config.rl_epochs * config.steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)

    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=config.rl_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        alpha=config.min_lr_ratio
    )
    optimizer = create_optimizer(lr_schedule, config)
    lr_manager = AdaptiveLRManager(optimizer, lr_schedule, config, total_steps, config.rl_lr)
    _, start_epoch, global_step = load_checkpoint_if_exists(model, optimizer, save_dir)

    beta = 0.1
    best_reward_margin = -float('inf')
    patience_counter = 0
    writer = tf.summary.create_file_writer(os.path.join(config.log_dir, "rl"))

    def compute_logps(logits, labels):
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot = tf.one_hot(labels, depth=config.vocab_size)
        per_token_logps = tf.reduce_sum(log_probs * one_hot, axis=-1)
        mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        return tf.reduce_sum(per_token_logps * mask, axis=-1) / tf.maximum(tf.reduce_sum(mask, axis=-1), 1.0)

    for epoch in range(start_epoch, config.rl_epochs):
        epoch_start = time.time()
        accum_loss = 0.0
        train_steps = 0

        for step, (cx, cy, rx, ry) in enumerate(train_gen()):
            if step >= config.steps_per_epoch:
                break
            global_step += 1

            with tf.GradientTape() as tape:
                c_logits = model(cx, training=True)
                r_logits = model(rx, training=True)
                c_logps = compute_logps(c_logits, cy)
                r_logps = compute_logps(r_logits, ry)

                ref_c_logits = tf.stop_gradient(ref_model(cx, training=False))
                ref_r_logits = tf.stop_gradient(ref_model(rx, training=False))
                ref_c_logps = compute_logps(ref_c_logits, cy)
                ref_r_logps = compute_logps(ref_r_logits, ry)

                pi_ratio = c_logps - r_logps
                ref_ratio = ref_c_logps - ref_r_logps
                logits_dpo = beta * (pi_ratio - ref_ratio)
                loss = tf.reduce_mean(tf.nn.softplus(-logits_dpo))

                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps

            grads = tape.gradient(loss, model.trainable_variables)
            if check_gradients_nan_inf(grads):
                if lr_manager.on_nan_detected():
                    return
                continue

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            accum_loss += float(loss)
            train_steps += 1
            lr_manager.on_step_end(loss)

        avg_loss = accum_loss / max(train_steps, 1)
        current_lr = lr_manager.get_current_lr(global_step)

        margin_sum = 0.0
        margin_steps = 0
        for cx, cy, rx, ry in train_gen():
            c_logits = model(cx, training=False)
            r_logits = model(rx, training=False)
            c_lp = compute_logps(c_logits, cy)
            r_lp = compute_logps(r_logits, ry)
            margin_sum += float(tf.reduce_mean(c_lp - r_lp))
            margin_steps += 1
            if margin_steps >= 20:
                break
        avg_margin = margin_sum / max(margin_steps, 1)

        with writer.as_default():
            tf.summary.scalar('dpo_loss', avg_loss, step=global_step)
            tf.summary.scalar('reward_margin', avg_margin, step=global_step)
            tf.summary.scalar('learning_rate', current_lr, step=global_step)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config.rl_epochs} | Loss: {avg_loss:.4f} | Margin: {avg_margin:.4f} | LR: {current_lr:.2e} | {epoch_time:.1f}s")

        if avg_margin > best_reward_margin:
            best_reward_margin = avg_margin
            patience_counter = 0
            save_best_model_only(model, save_dir, is_best=True)
        else:
            patience_counter += 1

        should_stop = lr_manager.on_epoch_end(avg_loss)
        if should_stop or patience_counter >= config.early_stop_patience:
            print(f"⏹️ DPO 终止")
            break

        save_checkpoint(model, optimizer, epoch, global_step, save_dir)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("🤖 Literature Transformer DPO 偏好对齐")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  TensorFlow: {tf.__version__}")
    print(f"🖥️  GPU: {tf.config.list_physical_devices('GPU')}")
    print("=" * 60)

    setup_environment()
    config = ModelConfig()
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

    # 加载 SFT 模型作为初始模型和参考模型
    print("\n📂 加载 SFT 模型...")
    from train import LiteratureTransformer  # 临时从原文件导入
    sft_dir = os.path.join(base_dir, "output", "sft")
    model = load_model_keras(sft_dir)
    model.config = config

    ref_model = load_model_keras(sft_dir)
    ref_model.config = config
    # 冻结参考模型
    ref_model.trainable = False

    # DPO 数据
    rl_train_path = os.path.join(base_dir, "rl_train.jsonl")
    rl_val_path = os.path.join(base_dir, "rl_val.jsonl")

    if not os.path.exists(rl_train_path):
        print("❌ DPO 训练数据不存在")
        sys.exit(1)

    num_rl_pairs = count_jsonl_lines(rl_train_path)
    auto_config_dpo(config, num_rl_pairs)

    rl_train_gen = DPODataGenerator(rl_train_path, vocab, config, infinite=True)
    rl_val_gen = DPODataGenerator(rl_val_path, vocab, config, infinite=False) if os.path.exists(rl_val_path) else None

    # DPO 训练
    try:
        dpo_train(model, ref_model, rl_train_gen, config,
                  save_dir=os.path.join(base_dir, "output", "rl"))
    except Exception as e:
        print(f"\n❌ DPO 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ DPO 完成！")


if __name__ == "__main__":
    main()