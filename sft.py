"""
sft.py - 监督微调脚本（带详细进度条）
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
from tqdm import tqdm

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


# ============================================================
# SFT 数据生成器
# ============================================================
class SFTDataGenerator:
    def __init__(self, jsonl_path, vocab, config: ModelConfig, max_samples=None, samples=None, infinite=False):
        self.vocab = vocab
        self.config = config
        self.pad_id = vocab.get('<|pad|>', 0)
        self.unk_id = vocab.get('<|unk|>', 3)
        self.bos_id = vocab.get('<|bos|>', 1)
        self.eos_id = vocab.get('<|eos|>', 2)
        self.user_id = vocab.get('<|user|>', 4)
        self.bot_id = vocab.get('<|bot|>', 5)
        self.infinite = infinite

        if samples is not None:
            self.samples = samples
            self.num_samples = len(samples)
        else:
            self._analyze_data(jsonl_path)
            self.samples = self._load_data(jsonl_path, max_samples)

        print(f"  ✅ 加载了 {len(self.samples)} 条样本")
        if self.samples:
            print(f"  ✅ 样本平均长度: {np.mean([len(s['x']) for s in self.samples]):.0f}")

    def _analyze_data(self, jsonl_path, max_samples=500):
        print(f"\n📊 分析数据: {jsonl_path}")
        prompt_lens = []
        response_lens = []
        total_lens = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                if line.strip():
                    item = json.loads(line)
                    prompt_len = len(item.get('prompt', ''))
                    response_len = len(item.get('response', ''))
                    prompt_lens.append(prompt_len)
                    response_lens.append(response_len)
                    total_lens.append(prompt_len + response_len)
        if prompt_lens:
            p95_total = int(np.percentile(total_lens, 95))
            p90_prompt = int(np.percentile(prompt_lens, 90))
            self.seq_len = min(max(p95_total + 10, 128), 512)
            self.seq_len = (self.seq_len // 8) * 8
            self.max_prompt_len = max(self.seq_len - 20, 100)
            self.max_prompt_len = min(self.max_prompt_len, p90_prompt + 20)
            if self.seq_len != self.config.seq_len:
                print(f"  ✅ 自动调整 seq_len: {self.config.seq_len} -> {self.seq_len}")
                self.config.seq_len = self.seq_len
            print(f"  ✅ 调整后: seq_len={self.seq_len}, max_prompt_len={self.max_prompt_len}")

    def _encode(self, prompt, response):
        prompt_ids = [self.bos_id, self.user_id]
        prompt_ids.extend(self.vocab.encode(prompt))
        prompt_ids.append(self.bot_id)
        response_ids = self.vocab.encode(response)
        response_ids.append(self.eos_id)
        return prompt_ids, response_ids

    def _load_data(self, path, max_samples=None):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompt_ids, response_ids = self._encode(item['prompt'], item['response'])
                    if len(prompt_ids) > self.max_prompt_len:
                        prompt_ids = prompt_ids[:self.max_prompt_len]
                    full_ids = prompt_ids + response_ids
                    if len(full_ids) > self.config.seq_len + 1:
                        full_ids = full_ids[:self.config.seq_len + 1]
                    if len(full_ids) >= 2:
                        x = full_ids[:-1]
                        y = full_ids[1:]
                        prompt_len = len(prompt_ids)
                        samples.append({'x': x, 'y': y, 'prompt_len': prompt_len})
                    if max_samples and len(samples) >= max_samples:
                        break
        return samples

    def __len__(self):
        return len(self.samples) // self.config.batch_size

    def __call__(self):
        while True:
            indices = list(range(len(self.samples)))
            np.random.shuffle(indices)
            batch_x, batch_y, batch_mask = [], [], []
            count = 0
            for idx in indices:
                sample = self.samples[idx]
                x = sample['x']
                y = sample['y']
                prompt_len = sample['prompt_len']
                loss_mask = [0.0] * prompt_len + [1.0] * (len(x) - prompt_len)
                if len(x) < self.config.seq_len:
                    pad_len = self.config.seq_len - len(x)
                    x = x + [self.pad_id] * pad_len
                    y = y + [self.pad_id] * pad_len
                    loss_mask = loss_mask + [0.0] * pad_len
                else:
                    x = x[:self.config.seq_len]
                    y = y[:self.config.seq_len]
                    loss_mask = loss_mask[:self.config.seq_len]
                batch_x.append(x)
                batch_y.append(y)
                batch_mask.append(loss_mask)
                count += 1
                if count == self.config.batch_size:
                    yield (np.array(batch_x, dtype=np.int32),
                           np.array(batch_y, dtype=np.int32),
                           np.array(batch_mask, dtype=np.float32))
                    batch_x, batch_y, batch_mask = [], [], []
                    count = 0
            if not self.infinite:
                break


def count_jsonl_lines(path):
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def auto_config_sft(config, num_samples):
    steps = max(num_samples // config.batch_size, 100)
    config.steps_per_epoch = min(steps, 3000)
    if config.steps_per_epoch >= 2000:
        config.sft_epochs = 5
    elif config.steps_per_epoch >= 1000:
        config.sft_epochs = 8
    else:
        config.sft_epochs = 12
    print(f"📊 SFT 自动配置: samples={num_samples}, steps_per_epoch={config.steps_per_epoch}, epochs={config.sft_epochs}")


# ============================================================
# SFT 训练函数（带进度条）
# ============================================================
def sft_train(model, train_gen, val_gen, config, save_dir="output/sft"):
    print("\n🚀 开始监督微调 (SFT)")
    if config.enable_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    total_steps = config.sft_epochs * config.steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)

    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=config.sft_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        alpha=config.min_lr_ratio
    )
    optimizer = create_optimizer(lr_schedule, config)
    lr_manager = AdaptiveLRManager(optimizer, lr_schedule, config, total_steps, config.sft_lr)
    _, start_epoch, global_step = load_checkpoint_if_exists(model, optimizer, save_dir)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=0)
    best_val_loss = float('inf')
    patience_counter = 0
    writer = tf.summary.create_file_writer(os.path.join(config.log_dir, "sft"))

    for epoch in range(start_epoch, config.sft_epochs):
        epoch_start = time.time()
        accum_loss = 0.0
        train_steps = 0

        # 训练进度条
        pbar = tqdm(total=config.steps_per_epoch, desc=f"Epoch {epoch+1}/{config.sft_epochs}", 
                    unit="step", ncols=80, leave=False)

        for step, (x, y, mask) in enumerate(train_gen()):
            if step >= config.steps_per_epoch:
                break
            global_step += 1

            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = loss_fn(y, logits, sample_weight=mask)
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps

            grads = tape.gradient(loss, model.trainable_variables)
            if check_gradients_nan_inf(grads):
                if lr_manager.on_nan_detected():
                    print("\n⏹️ NaN 容忍耗尽，停止训练")
                    pbar.close()
                    return
                continue

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            accum_loss += float(loss)
            train_steps += 1
            lr_manager.on_step_end(loss)

            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{float(loss):.4f}",
                "avg": f"{accum_loss/train_steps:.4f}",
                "lr": f"{lr_manager.get_current_lr(global_step):.2e}"
            })

        pbar.close()

        # 验证
        print("  📊 验证中...", end="")
        val_loss = 0.0
        val_steps = 0
        if val_gen is not None:
            val_pbar = tqdm(total=50, desc="  Val", unit="batch", ncols=80, leave=False)
            for x, y, mask in val_gen():
                logits = model(x, training=False)
                val_loss += float(loss_fn(y, logits, sample_weight=mask))
                val_steps += 1
                val_pbar.update(1)
                if val_steps >= 50:
                    break
            val_pbar.close()
        val_loss = val_loss / max(val_steps, 1)
        print(f" ✓ Val Loss: {val_loss:.4f}")

        avg_train_loss = accum_loss / max(train_steps, 1)
        current_lr = lr_manager.get_current_lr(global_step)

        with writer.as_default():
            tf.summary.scalar('train_loss', avg_train_loss, step=global_step)
            tf.summary.scalar('val_loss', val_loss, step=global_step)
            tf.summary.scalar('learning_rate', current_lr, step=global_step)

        epoch_time = time.time() - epoch_start
        print(f"  ⏱️  Epoch 耗时: {epoch_time:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_best_model_only(model, save_dir, is_best=True)
            print(f"  ⭐ 新的最佳模型! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠️ Val Loss 未提升 ({patience_counter}/{config.early_stop_patience})")

        should_stop = lr_manager.on_epoch_end(val_loss)
        if should_stop or patience_counter >= config.early_stop_patience:
            print(f"⏹️ SFT 终止")
            break

        save_checkpoint(model, optimizer, epoch, global_step, save_dir)
        print(f"  💾 Checkpoint 已保存")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("🤖 Literature Transformer SFT 微调")
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

    # 加载预训练模型
    print("\n📂 加载预训练模型...")
    pretrain_dir = os.path.join(base_dir, "output", "pretrain")
    model = load_model_keras(pretrain_dir)
    model.config = config

    # SFT 数据
    sft_train_path = os.path.join(base_dir, "sft_train.jsonl")
    sft_val_path = os.path.join(base_dir, "sft_val.jsonl")

    if not os.path.exists(sft_train_path):
        print("❌ SFT 训练数据不存在")
        sys.exit(1)

    num_sft_samples = count_jsonl_lines(sft_train_path)
    auto_config_sft(config, num_sft_samples)

    print("\n📊 加载训练数据...")
    sft_train_gen = SFTDataGenerator(sft_train_path, vocab, config, infinite=True)
    sft_val_gen = SFTDataGenerator(sft_val_path, vocab, config, infinite=False) if os.path.exists(sft_val_path) else None

    # SFT 训练
    try:
        sft_train(model, sft_train_gen, sft_val_gen, config,
                  save_dir=os.path.join(base_dir, "output", "sft"))
    except Exception as e:
        print(f"\n❌ SFT 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ SFT 完成！")


if __name__ == "__main__":
    main()