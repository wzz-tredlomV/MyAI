"""
tokenizer_wrapper.py
兼容层：统一封装字符级 vocab.json 和 BPE tokenizer.json
对外提供一致的 encode / decode / get / len 接口
"""
import json
import os


class TokenizerWrapper:
    """同时支持字符级 (vocab.json) 和 BPE (tokenizer.json)"""

    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self.is_bpe = False
        self.tokenizer = None
        self.vocab = {}
        self.idx_to_char = {}

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"词表文件不存在: {vocab_path}")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # HF Tokenizer 格式含有 'model' 和 'version' 字段
        if isinstance(raw, dict) and 'model' in raw and 'version' in raw:
            self._init_bpe(vocab_path)
        else:
            self._init_char_level(raw)

    def _init_char_level(self, data: dict):
        self.vocab = data
        self.idx_to_char = {v: k for k, v in data.items()}
        self._setup_special_ids()
        self.vocab_size = len(self.vocab)
        print(f"  检测到字符级词表，大小: {self.vocab_size}")

    def _init_bpe(self, path: str):
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "BPE 模式需要安装 tokenizers 库。\n"
                "请运行: pip install tokenizers"
            )
        self.tokenizer = Tokenizer.from_file(path)
        self.is_bpe = True
        self.vocab = self.tokenizer.get_vocab()
        self.idx_to_char = {v: k for k, v in self.vocab.items()}
        self._setup_special_ids()
        self.vocab_size = len(self.vocab)
        print(f"  检测到 BPE 词表，大小: {self.vocab_size}")

    def _setup_special_ids(self):
        self.pad_id = self.vocab.get('<|pad|>', 0)
        self.unk_id = self.vocab.get('<|unk|>', 3)
        self.bos_id = self.vocab.get('<|bos|>', 1)
        self.eos_id = self.vocab.get('<|eos|>', 2)
        self.user_id = self.vocab.get('<|user|>', 4)
        self.bot_id = self.vocab.get('<|bot|>', 5)

    # ----- 兼容 dict 接口 -----
    def get(self, key, default=None):
        return self.vocab.get(key, default)

    def __getitem__(self, key):
        return self.vocab[key]

    def __contains__(self, key):
        return key in self.vocab

    def __len__(self):
        return self.vocab_size

    def items(self):
        return self.vocab.items()

    def keys(self):
        return self.vocab.keys()

    def values(self):
        return self.vocab.values()

    # ----- 核心编解码 -----
    def encode(self, text: str) -> list:
        """返回 token ID 列表（不含特殊 token）"""
        if self.is_bpe:
            return self.tokenizer.encode(text, add_special_tokens=False).ids
        else:
            return [self.vocab.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: list, skip_special_tokens: bool = True) -> str:
        """ID 列表转文本"""
        if self.is_bpe:
            if skip_special_tokens:
                special_ids = {self.pad_id, self.bos_id, self.eos_id,
                               self.user_id, self.bot_id}
                ids = [i for i in ids if i not in special_ids]
            return self.tokenizer.decode(ids, skip_special_tokens=False)
        else:
            special_ids = {self.pad_id, self.bos_id, self.eos_id,
                           self.user_id, self.bot_id}
            chars = []
            for tid in ids:
                if skip_special_tokens and tid in special_ids:
                    continue
                if 0 <= tid < self.vocab_size:
                    chars.append(self.idx_to_char.get(tid, ''))
            return ''.join(chars)
