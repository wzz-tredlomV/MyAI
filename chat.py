# chat.py
import torch
import torch.nn.functional as F
from data_utils import BPETokenizer, PAD_IDX, SOS_IDX, EOS_IDX, USER_IDX, BOT_IDX
from model import TransformerChat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path, tokenizer_path, config):
    tokenizer = BPETokenizer(tokenizer_path)
    model = TransformerChat(
        vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        pad_idx=PAD_IDX
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, tokenizer

def repetition_penalty(logits, generated_ids, penalty=1.2):
    """对已生成的 token 施加惩罚，减少重复"""
    if penalty == 1.0 or not generated_ids:
        return logits
    for token_id in set(generated_ids):
        logits[token_id] /= penalty
    return logits

def top_k_top_p_filtering(logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
    """先 top-k 再 top-p 过滤，用于采样"""
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def beam_search_generate(model, tokenizer, src_ids, beam_width=5, max_len=60,
                         length_penalty=1.0, repetition_pen=1.2, device=device):
    model.eval()
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_pad_mask = (src_tensor == PAD_IDX)

    with torch.no_grad():
        memory = model.encode(src_tensor, src_padding_mask=src_pad_mask)

    # 每个 beam: (序列id列表, 累积log概率, 是否结束)
    beams = [([SOS_IDX], 0.0, False)]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for seq, score, done in beams:
            if done:
                completed.append((seq, score))
                continue

            tgt_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            tgt_pad_mask = (tgt_tensor == PAD_IDX)
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

            output = model.decode(tgt_tensor, memory, tgt_mask=tgt_mask,
                                  tgt_padding_mask=tgt_pad_mask)
            logits = output[0, -1, :]  # (vocab_size)
            # 重复惩罚
            logits = repetition_penalty(logits, seq, penalty=repetition_pen)
            log_probs = torch.log_softmax(logits, dim=-1)

            topk_probs, topk_ids = torch.topk(log_probs, beam_width * 2)  # 多取一些候选

            for prob, idx in zip(topk_probs, topk_ids):
                new_seq = seq + [idx.item()]
                # 长度惩罚： (5+len)^alpha / (5+1)^alpha ，常见公式
                lp = ((5.0 + len(new_seq)) ** length_penalty) / ((5.0 + 1) ** length_penalty)
                new_score = (score + prob.item()) / lp
                done = (idx.item() == EOS_IDX)
                new_beams.append((new_seq, new_score, done))

        if not new_beams:
            break
        # 保留 beam_width 个最优
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    completed.extend(beams)
    # 按归一化分数排序
    completed.sort(key=lambda x: x[1], reverse=True)
    best_seq, best_score, _ = completed[0]
    return best_seq

def sample_generate(model, tokenizer, src_ids, max_len=60, temperature=0.8, top_k=50, top_p=0.9,
                    repetition_pen=1.2, device=device):
    """带温度、top-k/top-p 的采样生成（更具多样性）"""
    model.eval()
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_pad_mask = (src_tensor == PAD_IDX)

    with torch.no_grad():
        memory = model.encode(src_tensor, src_padding_mask=src_pad_mask)

    generated = [SOS_IDX]
    for _ in range(max_len):
        tgt_tensor = torch.tensor([generated], dtype=torch.long).to(device)
        tgt_pad_mask = (tgt_tensor == PAD_IDX)
        tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

        output = model.decode(tgt_tensor, memory, tgt_mask=tgt_mask,
                              tgt_padding_mask=tgt_pad_mask)
        logits = output[0, -1, :] / temperature
        logits = repetition_penalty(logits, generated, penalty=repetition_pen)
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        if next_token == EOS_IDX:
            break
    return generated

def generate_response(model, tokenizer, history_text, decode_mode='beam', **kwargs):
    src_ids = tokenizer.encode(history_text, add_sos=False, add_eos=False)
    if decode_mode == 'beam':
        gen_ids = beam_search_generate(model, tokenizer, src_ids, **kwargs)
    else:
        gen_ids = sample_generate(model, tokenizer, src_ids, **kwargs)
    return tokenizer.decode(gen_ids, skip_special=True)

# ---------- 运行交互 ----------
if __name__ == "__main__":
    CONFIG = {
        'd_model': 512, 'nhead': 8, 'num_encoder_layers': 6,
        'num_decoder_layers': 6, 'dim_feedforward': 2048, 'dropout': 0.1,
    }
    model, tokenizer = load_model('checkpoints/best_model.pt', 'tokenizer/bpe_tokenizer.json', CONFIG)

    print("\n===== 多轮对话 (输入 'quit' 退出) =====")
    print("解码模式: beam (稳定) / sample (多样), 输入 'mode beam' 或 'mode sample' 切换")
    history = []
    mode = 'beam'
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'quit':
            break
        if user_input.startswith('mode '):
            mode = user_input.split()[1]
            print(f"切换至 {mode} 模式")
            continue

        # 构建历史
        parts = []
        for u, b in history:
            parts.append(f"<user> {u}")
            parts.append(f"<bot> {b}")
        parts.append(f"<user> {user_input}")
        full_history = " ".join(parts)

        if mode == 'beam':
            response = generate_response(model, tokenizer, full_history,
                                         decode_mode='beam', beam_width=5,
                                         length_penalty=0.6, repetition_pen=1.2)
        else:
            response = generate_response(model, tokenizer, full_history,
                                         decode_mode='sample', temperature=0.9,
                                         top_k=50, top_p=0.9, repetition_pen=1.2)
        print(f"Bot: {response}")
        history.append((user_input, response))
        if len(history) > 5:
            history = history[-5:]
