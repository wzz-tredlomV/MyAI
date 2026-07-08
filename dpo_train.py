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
# DPO 数据生成器
# ============================================================

class DPODataGenerator:
    def __init__(self, jsonl_path, vocab, config: ModelConfig, max_pairs=None):
        self.vocab = vocab
        self.config = config
        self.pad_id = vocab.get('<|pad|>', 0)
        self.unk_id = vocab.get('<|unk|>', 3)
        self.bos_id = vocab.get('<|bos|>', 1)
        self.eos_id = vocab.get('<|eos|>', 2)
        self.user_id = vocab.get('<|user|>', 4)
        self.bot_id = vocab.get('<|bot|>', 5)
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
        for ch in prompt:
            ids.append(self.vocab.get(ch, self.unk_id))
        ids.append(self.bot_id)
        for ch in response:
            ids.append(self.vocab.get(ch, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def __call__(self):
        for chosen, rejected in self.pairs:
            max_len = max(len(chosen), len(rejected))
            chosen = chosen + [self.pad_id] * (max_len - len(chosen))
            rejected = rejected + [self.pad_id] * (max_len - len(rejected))
            yield np.array(chosen, dtype=np.int32), np.array(rejected, dtype=np.int32)


# ============================================================
# DPO 评估函数
# ============================================================

def compute_logprob(logits, tokens, mask):
    token_neg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tokens, logits=logits)
    token_logprob = -token_neg_loss
    return tf.reduce_sum(token_logprob * mask) / tf.reduce_sum(mask)


def evaluate_dpo(model, ref_model, val_dataset, beta=0.1, max_val_steps=30):
    @tf.function
    def eval_step(chosen, rejected):
        policy_chosen_logits = model(chosen, training=False)
        policy_rejected_logits = model(rejected, training=False)
        ref_chosen_logits = ref_model(chosen, training=False)
        ref_rejected_logits = ref_model(rejected, training=False)
        chosen_mask = tf.cast(chosen != 0, tf.float32)
        rejected_mask = tf.cast(rejected != 0, tf.float32)
        policy_chosen_logprob = compute_logprob(policy_chosen_logits, chosen, chosen_mask)
        policy_rejected_logprob = compute_logprob(policy_rejected_logits, rejected, rejected_mask)
        ref_chosen_logprob = compute_logprob(ref_chosen_logits, chosen, chosen_mask)
        ref_rejected_logprob = compute_logprob(ref_rejected_logits, rejected, rejected_mask)
        policy_ratio = policy_chosen_logprob - policy_rejected_logprob
        ref_ratio = ref_chosen_logprob - ref_rejected_logprob
        loss = -tf.math.log(tf.sigmoid(beta * (policy_ratio - ref_ratio)))
        acc = tf.cast(policy_ratio > ref_ratio, tf.float32)
        return loss, acc

    val_losses = []
    val_accs = []
    step = 0
    for chosen, rejected in val_dataset:
        if step >= max_val_steps:
            break
        loss, acc = eval_step(chosen, rejected)
        val_losses.append(loss.numpy())
        val_accs.append(acc.numpy())
        step += 1

    return np.mean(val_losses) if val_losses else float('inf'), np.mean(val_accs) if val_accs else 0.0


# ============================================================
# DPO 训练主函数
# ============================================================

def dpo_train(model, train_jsonl_path, val_jsonl_path, vocab, config: ModelConfig, beta=0.1):
    try:
        print(f"\n📊 DPO训练配置: lr={config.rl_lr}, epochs={config.rl_epochs}, clipnorm=0.5")
        
        ref_model = build_model(config)
        ref_model.set_weights(model.get_weights())

        @tf.function
        def ref_call(x):
            return ref_model(x, training=False)

        train_data_gen = DPODataGenerator(train_jsonl_path, vocab, config)
        if val_jsonl_path and os.path.exists(val_jsonl_path):
            val_data_gen = DPODataGenerator(val_jsonl_path, vocab, config)
        else:
            all_pairs = DPODataGenerator(train_jsonl_path, vocab, config).pairs
            split_idx = int(len(all_pairs) * 0.8)
            train_data_gen = DPODataGenerator(train_jsonl_path, vocab, config, max_pairs=split_idx)
            val_data_gen = DPODataGenerator(train_jsonl_path, vocab, config)

        optimizer = keras.optimizers.AdamW(
            learning_rate=config.rl_lr,
            weight_decay=config.weight_decay,
            global_clipnorm=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        train_dataset = tf.data.Dataset.from_generator(
            train_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        train_dataset = train_dataset.padded_batch(
            config.batch_size,
            padded_shapes=([None], [None]),
            padding_values=(vocab.get('<|pad|>', 0), vocab.get('<|pad|>', 0))
        )
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_generator(
            val_data_gen,
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
            )
        )
        val_dataset = val_dataset.padded_batch(
            config.batch_size,
            padded_shapes=([None], [None]),
            padding_values=(vocab.get('<|pad|>', 0), vocab.get('<|pad|>', 0))
        )
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

        total_pairs = len(train_data_gen.pairs)
        steps_per_epoch = (total_pairs + config.batch_size - 1) // config.batch_size
        total_steps = steps_per_epoch * config.rl_epochs

        @tf.function
        def dpo_step(chosen, rejected):
            with tf.GradientTape() as tape:
                policy_chosen_logits = model(chosen, training=True)
                policy_rejected_logits = model(rejected, training=True)
                ref_chosen_logits = ref_call(chosen)
                ref_rejected_logits = ref_call(rejected)
                chosen_mask = tf.cast(chosen != 0, tf.float32)
                rejected_mask = tf.cast(rejected != 0, tf.float32)
                policy_chosen_logprob = compute_logprob(policy_chosen_logits, chosen, chosen_mask)
                policy_rejected_logprob = compute_logprob(policy_rejected_logits, rejected, rejected_mask)
                ref_chosen_logprob = compute_logprob(ref_chosen_logits, chosen, chosen_mask)
                ref_rejected_logprob = compute_logprob(ref_rejected_logits, rejected, rejected_mask)
                policy_ratio = policy_chosen_logprob - policy_rejected_logprob
                ref_ratio = ref_chosen_logprob - ref_rejected_logprob
                loss = -tf.math.log(tf.sigmoid(beta * (policy_ratio - ref_ratio)))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return loss

        best_val_loss = float('inf')
        best_epoch = 0

        with tqdm(total=total_steps, desc="DPO总进度", unit="step", position=0, leave=True) as total_pbar:
            for epoch in range(config.rl_epochs):
                epoch_pbar = tqdm(
                    total=steps_per_epoch,
                    desc=f"DPO Epoch {epoch+1}/{config.rl_epochs}",
                    unit="step",
                    position=1,
                    leave=False
                )
                epoch_losses = []
                step = 0
                for chosen, rejected in train_dataset:
                    loss = dpo_step(chosen, rejected)
                    loss_val = loss.numpy()
                    epoch_losses.append(loss_val)
                    step += 1

                    epoch_pbar.update(1)
                    epoch_pbar.set_postfix(loss=f"{loss_val:.4f}")
                    total_pbar.update(1)
                    total_pbar.set_postfix(epoch=f"{epoch+1}", loss=f"{loss_val:.4f}")

                epoch_pbar.close()
                avg_loss = np.mean(epoch_losses) if epoch_losses else 0

                print(f"\n  正在验证 DPO Epoch {epoch+1}...")
                val_loss, val_acc = evaluate_dpo(model, ref_model, val_dataset, beta)
                print(f"  [验证] Loss: {val_loss:.4f}, 准确率: {val_acc:.2%}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    save_best_model(model, "saved_model/rl_best", is_best=True)
                    print(f"  ★ 新的最佳模型！Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
                else:
                    print(f"  当前最佳: Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")

                print(f"\n[DPO Epoch {epoch+1}/{config.rl_epochs}] 训练Loss: {avg_loss:.4f}")

        print(f"\nDPO完成！最佳模型来自 Epoch {best_epoch}, Val Loss: {best_val_loss:.4f}")
        if os.path.exists("saved_model/rl_best"):
            if os.path.exists("saved_model/rl_final"):
                shutil.rmtree("saved_model/rl_final")
            shutil.copytree("saved_model/rl_best", "saved_model/rl_final")
            print("最佳模型已保存为 rl_final")

    except KeyboardInterrupt:
        print("\n训练被用户中断，保存当前模型...")
        save_model_dual_format(model, "saved_model/rl_interrupted")
        raise
    except Exception as e:
        print(f"训练出错: {e}")
        save_model_dual_format(model, "saved_model/rl_error")
        raise


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    keras.mixed_precision.set_global_policy("float32")
    VOCAB_PATH = "vocab.json"
    RL_DATA_PATH = "rl_train.jsonl"
    RL_VAL_PATH = "rl_val.jsonl"

    if not os.path.exists(VOCAB_PATH):
        print(f"找不到词汇表: {VOCAB_PATH}")
        exit()
    if not os.path.exists(RL_DATA_PATH):
        print(f"找不到RL数据: {RL_DATA_PATH}")
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
        sft_epochs=10,
        rl_epochs=10,
        steps_per_epoch=300
    )
    print(f"词汇表大小: {config.vocab_size}")
    print(f"📊 DPO配置: {config.rl_epochs}轮, 预计耗时~1.5小时")

    print("\n" + "="*50)
    print("加载SFT模型...")
    print("="*50)
    model = load_model_keras("saved_model/sft_final")
    
    print("\n" + "="*50)
    print("开始 DPO 强化学习")
    print("="*50)
    val_path = RL_VAL_PATH if os.path.exists(RL_VAL_PATH) else None
    dpo_train(model, RL_DATA_PATH, val_path, vocab, config)

    print("\n" + "="*50)
    print("DPO训练完成！")
    print("模型保存在: saved_model/rl_final/")
    print("="*50)