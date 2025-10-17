import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
from config import *

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

# ========== 加载模型 ==========
print(f"📂 加载模型: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

vocab_size = checkpoint['vocab_size']
hidden_size = checkpoint['hidden_size']
stoi = checkpoint['stoi']
itos = checkpoint['itos']

model = NicknameRNN(vocab_size, hidden_size=hidden_size).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ 模型加载成功")
print(f"   - 词典大小: {vocab_size}")
print(f"   - 隐藏层大小: {hidden_size}")
print()

# ========== 生成函数 ==========
def generate(model, start="小", max_len=MAX_GEN_LEN, temperature=TEMPERATURE):
    """
    生成昵称

    参数:
        model: 训练好的模型
        start: 起始字符
        max_len: 最大生成长度
        temperature: 温度参数，越高越随机

    返回:
        生成的昵称字符串
    """
    model.eval()

    # 编码起始字符
    input_ids = torch.tensor([[stoi.get(ch, 0) for ch in start]], dtype=torch.long, device=DEVICE)
    hidden = None
    result = list(start)

    with torch.no_grad():
        for _ in range(max_len):
            output, hidden = model(input_ids, hidden)
            logits = output[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1).squeeze()
            idx = torch.multinomial(probs, 1).item()
            ch = itos[idx]

            result.append(ch)

            # 遇到换行符停止
            if ch == "\n":
                break

            input_ids = torch.tensor([[idx]], dtype=torch.long, device=DEVICE)

    return "".join(result).strip()

# ========== 生成示例 ==========
print("✨ 示例生成昵称：")
print("=" * 60)

# 常见起始字
start_chars = ["小", "喵", "星", "梦", "孤", "柚", "风", "糖", "呆", "软"]

for _ in range(15):
    start = random.choice(start_chars)
    nickname = generate(model, start=start)
    print(f"👉 {nickname}")

print("=" * 60)
print()

# ========== 自定义生成 ==========
print("💡 提示: 输入 'q' 或 'quit' 退出")
print()

while True:
    try:
        user_input = input("请输入起始字符（默认随机）: ").strip()

        if user_input.lower() in ['q', 'quit', 'exit']:
            print("👋 再见！")
            break

        if not user_input:
            user_input = random.choice(start_chars)

        # 生成多个样本
        print(f"\n以 '{user_input}' 开头的昵称：")
        for i in range(5):
            nickname = generate(model, start=user_input)
            print(f"  {i+1}. {nickname}")
        print()

    except KeyboardInterrupt:
        print("\n👋 再见！")
        break
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print()