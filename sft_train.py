import tensorflow as tf
from tensorflow import keras
import json
import os
import numpy as np
from tqdm import tqdm
import shutil
from pretrain import (
    ModelConfig, LiteratureTransformer, load_model_keras, 
    save_model_dual_format, save_best_model, load_vocab, build_model
)

# ============================================================
# SFT 数据生成器
# ============================================================

class SFTDataGenerator:
    def __init__(self, jsonl_path, vocab, config: ModelConfig, max_samples=None):
        self.vocab = vocab
        self.config = config
        self.pad_id = vocab.get('<|pad|>', 0)
        self.unk_id = vocab.get('<|unk|>', 3)
        self.bos_id = vocab.get('<|bos|>', 1)
        self.eos_id = vocab.get('<|eos|>', 2)
        self.user_id = vocab.get('<|user|>', 4)
        self.bot_id = vocab.get('<|bot|>', 5)
        self.samples = self._load_data(jsonl_path, max_samples)

    def _load_data(self, path, max_samples=None):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompt_ids, response_ids, full_ids = self._encode(item['prompt'], item['response'])
                    if len(full_ids) >= 2:
                        samples.append({'x': full_ids[:-1], 'y': full_ids[1:], 'prompt_len': len(prompt_ids)})
                    if max_samples and len(samples) >= max_samples:
                        break
        return samples

    def _encode(self, prompt, response):
        prompt_ids = [self.bos_id, self.user_id]
        for ch in prompt:
            prompt_ids.append(self.vocab.get(ch, self.unk_id))
        prompt_ids.append(self.bot_id)
        response_ids = []
        for ch in response:
            response_ids.append(self.vocab.get(ch, self.unk_id))
        response_ids.append(self.eos_id)
        return prompt_ids, response_ids, prompt_ids + response_ids

    def __len__(self):
        return len(self.samples) // self.config.batch_size

    def __call__(self):
        batch_x, batch_y, batch_mask = [], [], []
        count = 0
        for sample in self.samples:
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
                yield np.array(batch_x, dtype=np.int32), np.array(batch_y, dtype=np.int32), np.array(batch_mask, dtype=np.float32)
                batch_x, batch_y, batch_mask = [], [], []
                count = 0


# ============================================================
# SFT 评估函数
# ============================================================

def evaluate_sft(model, val_dataset, sft_loss_fn, max_val_steps=50):
    val_losses = []
    step = 0
    for x, y, mask in val_dataset:
        if step >= max_val_steps:
            break
        logits = model(x, training=False)
        loss = sft_loss_fn(y, logits, mask)
        val_losses.append(loss.numpy())
        step += 1
    return np.mean(val_losses) if val_losses else float('inf')


# ============================================================
# SFT 训练主函数
# ============================================================

def sft_train(model, train_jsonl_path, val_jsonl_path, vocab, config: ModelConfig):
    try:
        print(f"\n📊 SFT训练配置: lr={config.sft_lr}, epochs={config.sft_epochs}, clipnorm=0.5")
        
        train_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config)
        if val_jsonl_path and os.path.exists(val_jsonl_path):
            val_data_gen = SFTDataGenerator(val_jsonl_path, vocab, config)
        else:
            all_samples = SFTDataGenerator(train_jsonl_path, vocab, config).samples
            split_idx = int(len(all_samples) * 0.8)
            train_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config, max_samples=split_idx)
            val_data_gen = SFTDataGenerator(train_jsonl_path, vocab, config)

        def sft_loss(y_true, y_pred, loss_mask):
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
            loss = loss * loss_mask
            total_mask = tf.reduce_sum(loss_mask) + 1e-8
            return tf.reduce_sum(loss) / total_mask

        optimizer = keras.optimizers.AdamW(
            learning_rate=config.sft_lr,
            weight_decay=config.weight_decay,
            global_clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        train_dataset = tf.data.Dataset.from_generator(
            train_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.float32)
            )
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.int32),
                tf.TensorSpec(shape=(config.batch_size, config.seq_len), dtype=tf.float32)
            )
        )
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        total_steps_estimate = len(train_data_gen)
        total_steps = total_steps_estimate * config.sft_epochs

        best_val_loss = float('inf')
        best_epoch = 0

        with tqdm(total=total_steps, desc="SFT总进度", unit="step", position=0, leave=True) as total_pbar:
            for epoch in range(config.sft_epochs):
                epoch_pbar = tqdm(
                    total=total_steps_estimate,
                    desc=f"SFT Epoch {epoch+1}/{config.sft_epochs}",
                    unit="step",
                    position=1,
                    leave=False
                )
                epoch_losses = []
                step = 0
                for x, y, mask in train_dataset:
                    with tf.GradientTape() as tape:
                        logits = model(x, training=True)
                        loss = sft_loss(y, logits, mask)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    loss_val = loss.numpy()
                    epoch_losses.append(loss_val)
                    step += 1

                    epoch_pbar.update(1)
                    epoch_pbar.set_postfix(loss=f"{loss_val:.4f}")
                    total_pbar.update(1)
                    total_pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{loss_val:.4f}")

                epoch_pbar.close()
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0

                print(f"\n  正在验证 SFT Epoch {epoch+1}...")
                val_loss = evaluate_sft(model, val_dataset, sft_loss)
                print(f"  [验证] Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    save_best_model(model, "saved_model/sft_best", is_best=True)
                    print(f"  ★ 新的最佳模型！Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                else:
                    print(f"  当前最佳: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")

                print(f"\n[SFT Epoch {epoch+1}/{config.sft_epochs}] 训练Loss: {avg_loss:.4f}")

        print(f"\nSFT完成！最佳模型来自 Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
        if os.path.exists("saved_model/sft_best"):
            if os.path.exists("saved_model/sft_final"):
                shutil.rmtree("saved_model/sft_final")
            shutil.copytree("saved_model/sft_best", "saved_model/sft_final")
            print("最佳模型已保存为 sft_final")

    except KeyboardInterrupt:
        print("\n训练被用户中断，保存当前模型...")
        save_model_dual_format(model, "saved_model/sft_interrupted")
        raise
    except Exception as e:
        print(f"训练出错: {e}")
        save_model_dual_format(model, "saved_model/sft_error")
        raise


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("float32")
    VOCAB_PATH = "vocab.json"
    SFT_DATA_PATH = "sft_train.jsonl"
    SFT_VAL_PATH = "sft_val.jsonl"

    if not os.path.exists(VOCAB_PATH):
        print(f"找不到词汇表: {VOCAB_PATH}")
        exit()
    if not os.path.exists(SFT_DATA_PATH):
        print(f"找不到SFT数据: {SFT_DATA_PATH}")
        exit()

    vocab = load_vocab(VOCAB_PATH)
    config = ModelConfig(
        vocab_size=len(vocab),
        embed_dim=384,
        num_heads=6,
        num_layers=6,
        max_len=512,
        seq_len=256,
        batch_size=4,
        pretrain_epochs=5,
        sft_epochs=20,
        rl_epochs=15,
        steps_per_epoch=300
    )
    print(f"词汇表大小: {config.vocab_size}")
    print(f"📊 SFT配置: {config.sft_epochs}轮, 预计耗时~4小时")

    print("\n" + "="*50)
    print("加载预训练模型...")
    print("="*50)
    model = load_model_keras("saved_model/pretrain_final")
    
    print("\n" + "="*50)
    print("开始 SFT 微调")
    print("="*50)
    val_path = SFT_VAL_PATH if os.path.exists(SFT_VAL_PATH) else None
    sft_train(model, SFT_DATA_PATH, val_path, vocab, config)

    print("\n" + "="*50)
    print("SFT训练完成！")
    print("模型保存在: saved_model/sft_final/")
    print("="*50)