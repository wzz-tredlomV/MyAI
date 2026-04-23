# model.py
import math
import torch
import torch.nn as nn
from data_utils import PAD_IDX  # 从数据处理模块导入填充索引

class PositionalEncoding(nn.Module):
    """标准正弦位置编码，支持 dropout"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerChat(nn.Module):
    """带编码器-解码器结构的 Transformer 聊天模型"""
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, pad_idx=PAD_IDX):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """
        生成目标序列的因果掩码（自回归用）。
        返回 bool 张量，True 表示需要遮挡的注意力位置。
        """
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=next(self.parameters()).device),
            diagonal=1
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_padding_mask=None, tgt_padding_mask=None):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.embedding(tgt) * math.sqrt(self.d_model))

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_mask=None
        )
        return self.fc_out(output)

    def encode(self, src, src_mask=None, src_padding_mask=None):
        """仅编码器部分，用于推理时获取记忆张量"""
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb, mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None):
        """仅解码器部分，用于逐步生成"""
        tgt_emb = self.pos_decoder(self.embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer.decoder(tgt_emb, memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_padding_mask)
        return self.fc_out(output)


def create_loss_function(label_smoothing=0.1, ignore_index=PAD_IDX):
    """返回带标签平滑的交叉熵损失函数"""
    return nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing)