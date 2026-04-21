# data_utils.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import torch.nn.utils.rnn as rnn_utils

# 特殊标记索引
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>", "<user>", "<bot>"]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, USER_IDX, BOT_IDX = range(6)

class BPETokenizer:
    def __init__(self, tokenizer_path=None):
        if tokenizer_path:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            self.tokenizer.decoder = decoders.ByteLevel()
            self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    def train(self, texts, vocab_size=8000):
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

    def save(self, path):
        self.tokenizer.save(path)

    def encode(self, text, add_sos=False, add_eos=False, max_len=None):
        ids = self.tokenizer.encode(text).ids
        if add_sos:
            ids = [SOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        if max_len:
            ids = ids[:max_len]
        return ids

    def decode(self, ids, skip_special=True):
        if skip_special:
            ids = [i for i in ids if i not in [PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, USER_IDX, BOT_IDX]]
        return self.tokenizer.decode(ids).strip()

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

class ConversationDataset(Dataset):
    """
    将多轮对话转换为模型输入/输出对，格式：
    src:  "<user> 用户1 <bot> 机器人1 <user> 用户2 <bot> 机器人2 ... <user> 当前用户输入"
    tgt:  "<sos> 目标机器人回复 <eos>"
    """
    def __init__(self, conversations, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        for conv in conversations:
            for i in range(len(conv)):
                history_parts = []
                for j in range(i):
                    history_parts.append(f"<user> {conv[j][0]}")
                    history_parts.append(f"<bot> {conv[j][1]}")
                history_parts.append(f"<user> {conv[i][0]}")
                src_text = " ".join(history_parts)
                tgt_text = conv[i][1]
                self.samples.append((src_text, tgt_text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        src_ids = self.tokenizer.encode(src, add_sos=False, add_eos=False, max_len=self.max_len)
        tgt_ids = self.tokenizer.encode(tgt, add_sos=True, add_eos=True, max_len=self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch, pad_idx=PAD_IDX):
    src_batch, tgt_batch = zip(*batch)
    src_batch = rnn_utils.pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    tgt_batch = rnn_utils.pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=True)
    return src_batch, tgt_batch

def load_conversations(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
