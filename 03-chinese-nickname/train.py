import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import json
import os
from config import *

# 设置随机种子
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"🖥️  使用设备: {DEVICE}")
print()

# ========== 数据准备 ==========
print("📊 加载数据...")

# 首先读取所有数据来构建词表
with open(DATA_PATH, encoding="utf-8") as f:
    all_lines = [l.strip() for l in f if l.strip()]
all_text = "\n".join(all_lines)

print(f"   - 总数据行数: {len(all_lines)} 行")
print(f"   - 总数据字符数: {len(all_text)} 个")

# 构建字符词典（使用所有数据）
chars = sorted(list(set(all_text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print(f"📝 字符词典大小: {vocab_size} 个不同字符")
print()

# 保存词典
with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
    json.dump({'stoi': stoi, 'itos': {int(k): v for k, v in itos.items()}}, f, ensure_ascii=False, indent=2)

# 编码解码函数
def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return "".join([itos[i] for i in l])

# 根据DATA_LIMIT限制训练数据
if DATA_LIMIT:
    training_lines = all_lines[:DATA_LIMIT]
    training_text = "\n".join(training_lines)
    print(f"   - 训练数据行数: {len(training_lines)} 行 (限制为 {DATA_LIMIT})")
    print(f"   - 训练数据字符数: {len(training_text)} 个")
else:
    training_text = all_text
    print(f"   - 使用全部数据进行训练")

print()

data = encode(training_text)

# ========== 模型定义 ==========
class NicknameRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size

    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

# ========== 检查是否续训 ==========
start_epoch = 0
if os.path.exists(MODEL_PATH):
    print(f"📂 发现已有模型: {MODEL_PATH}")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

        # 检查模型结构是否匹配
        saved_hidden_size = checkpoint.get('hidden_size', HIDDEN_SIZE)
        if saved_hidden_size != HIDDEN_SIZE:
            print(f"⚠️  警告: 已保存模型的 HIDDEN_SIZE={saved_hidden_size}, 但配置文件为 {HIDDEN_SIZE}")
            print(f"   无法续训,将创建新模型 (旧模型备份为 {MODEL_PATH}.bak)")
            os.rename(MODEL_PATH, f"{MODEL_PATH}.bak")
            model = NicknameRNN(vocab_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        else:
            # 续训
            model = NicknameRNN(vocab_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # 尝试加载优化器状态(如果存在)
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("⚠️  优化器状态加载失败,使用新优化器")

            start_epoch = checkpoint.get('epoch', 0)
            print(f"✅ 续训模式: 从第 {start_epoch} 轮继续")
    except Exception as e:
        print(f"⚠️  模型加载失败: {e}")
        print(f"   将创建新模型")
        model = NicknameRNN(vocab_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
else:
    print(f"🆕 创建新模型")
    model = NicknameRNN(vocab_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()

print("🏗️  模型结构:")
print(model)
print()
print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters())} 个")
print()

# ========== 训练阶段 ==========
print("🚀 开始训练...")
if start_epoch > 0:
    print(f"   - 续训轮数: {EPOCHS} (总轮数将达到 {start_epoch + EPOCHS})")
else:
    print(f"   - 训练轮数: {EPOCHS}")
print(f"   - 序列长度: {SEQ_LEN}")
print(f"   - 隐藏层大小: {HIDDEN_SIZE}")
print(f"   - 学习率: {LR}")
print("=" * 60)

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
else:
    history = {'loss': []}

for epoch in range(start_epoch, start_epoch + EPOCHS):
    total_loss = 0
    num_batches = 0

    # 计算总批次数用于进度显示
    total_steps = (len(data) - SEQ_LEN) // SEQ_LEN

    for i in trange(0, len(data) - SEQ_LEN, SEQ_LEN, desc=f"Epoch {epoch+1}/{start_epoch + EPOCHS}", leave=False):
        x = torch.tensor([data[i:i+SEQ_LEN]], dtype=torch.long, device=DEVICE)
        y = torch.tensor([data[i+1:i+SEQ_LEN+1]], dtype=torch.long, device=DEVICE)

        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    history['loss'].append(avg_loss)

    print(f"Epoch [{epoch+1:2d}/{start_epoch + EPOCHS}] | Loss: {avg_loss:.4f}")

print("=" * 60)
print()

# 保存模型
torch.save({
    'epoch': start_epoch + EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab_size': vocab_size,
    'hidden_size': HIDDEN_SIZE,
    'stoi': stoi,
    'itos': itos
}, MODEL_PATH)

print(f"✅ 模型已保存到 {MODEL_PATH}")
if start_epoch > 0:
    print(f"   总训练轮数: {start_epoch + EPOCHS}")

# 保存训练历史
with open(HISTORY_PATH, 'w') as f:
    json.dump(history, f)

print(f"✅ 训练历史已保存到 {HISTORY_PATH}")