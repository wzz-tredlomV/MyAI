# train.py
import os
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from data_utils import (
    BPETokenizer, ConversationDataset, collate_fn, load_conversations,
    PAD_IDX, SOS_IDX, EOS_IDX
)
from model import TransformerChat, create_loss_function

# ---------- 配置 ----------
CONFIG = {
    'data_path': 'data/lccc_base_conversations.json',
    'tokenizer_path': 'tokenizer/bpe_tokenizer.json',
    'save_dir': 'checkpoints',
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'label_smoothing': 0.1,
    'batch_size': 32,
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'clip': 1.0,
    'max_len': 128,
    'val_split': 0.1,
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(CONFIG['save_dir'], exist_ok=True)

# ---------- 数据准备 ----------
print("加载对话数据...")
convs = load_conversations(CONFIG['data_path'])
print(f"总对话段数: {len(convs)}")

# 训练或加载分词器
if os.path.exists(CONFIG['tokenizer_path']):
    tokenizer = BPETokenizer(CONFIG['tokenizer_path'])
    print("加载已有分词器。")
else:
    tokenizer = BPETokenizer()
    all_texts = []
    for conv in convs:
        for u, b in conv:
            all_texts.append(u)
            all_texts.append(b)
    tokenizer.train(all_texts, vocab_size=8000)
    tokenizer.save(CONFIG['tokenizer_path'])
    print("训练并保存分词器。")
vocab_size = tokenizer.vocab_size
print(f"词汇量: {vocab_size}")

dataset = ConversationDataset(convs, tokenizer, max_len=CONFIG['max_len'])
val_len = int(len(dataset) * CONFIG['val_split'])
train_len = len(dataset) - val_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                          collate_fn=lambda b: collate_fn(b, PAD_IDX), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                        collate_fn=lambda b: collate_fn(b, PAD_IDX), num_workers=4, pin_memory=True)

# ---------- 模型、优化器、损失 ----------
model = TransformerChat(
    vocab_size=vocab_size,
    d_model=CONFIG['d_model'],
    nhead=CONFIG['nhead'],
    num_encoder_layers=CONFIG['num_encoder_layers'],
    num_decoder_layers=CONFIG['num_decoder_layers'],
    dim_feedforward=CONFIG['dim_feedforward'],
    dropout=CONFIG['dropout'],
    pad_idx=PAD_IDX
).to(device)

criterion = create_loss_function(label_smoothing=CONFIG['label_smoothing'], ignore_index=PAD_IDX)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
scaler = GradScaler()

# ---------- 训练函数 ----------
def train_epoch(model, loader, optimizer, criterion, device, scaler, clip):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_pad_mask = (src == PAD_IDX)
        tgt_pad_mask = (tgt_input == PAD_IDX)
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        with autocast():
            output = model(src, tgt_input,
                           src_padding_mask=src_pad_mask,
                           tgt_mask=tgt_mask,
                           tgt_padding_mask=tgt_pad_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_pad_mask = (src == PAD_IDX)
        tgt_pad_mask = (tgt_input == PAD_IDX)
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        with autocast():
            output = model(src, tgt_input,
                           src_padding_mask=src_pad_mask,
                           tgt_mask=tgt_mask,
                           tgt_padding_mask=tgt_pad_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, ppl

# ---------- 训练循环 ----------
best_ppl = float('inf')
for epoch in range(1, CONFIG['epochs']+1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, CONFIG['clip'])
    val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

    if val_ppl < best_ppl:
        best_ppl = val_ppl
        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'best_model.pt'))
        print(f"  保存最佳模型 (PPL: {val_ppl:.2f})")

print("训练完成！")
