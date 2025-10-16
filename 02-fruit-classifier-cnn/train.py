import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import json

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")
print()

# 1. 读取数据
data = pd.read_csv("data.csv")
print(f"📊 数据集大小: {len(data)} 条记录")
print(f"   - 水果样本: {sum(data['label'] == 1)} 条")
print(f"   - 非水果样本: {sum(data['label'] == 0)} 条")
print()

# 2. 构建字符词典
def build_char_vocab(words, max_chars=100):
    """构建字符到索引的映射"""
    chars = set()
    for word in words:
        chars.update(word)

    # 特殊标记
    char_to_idx = {'<PAD>': 0, '<UNK>': 1}

    # 添加所有字符
    for char in sorted(chars):
        if len(char_to_idx) >= max_chars:
            break
        char_to_idx[char] = len(char_to_idx)

    return char_to_idx

# 3. 文本编码
def encode_text(text, char_to_idx, max_len=10):
    """将文本转换为索引序列"""
    encoded = []
    for char in text[:max_len]:
        encoded.append(char_to_idx.get(char, char_to_idx['<UNK>']))

    # 填充到固定长度
    while len(encoded) < max_len:
        encoded.append(char_to_idx['<PAD>'])

    return encoded

# 4. 自定义数据集
class FruitDataset(Dataset):
    def __init__(self, texts, labels, char_to_idx, max_len=10):
        self.texts = texts
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoded = encode_text(text, self.char_to_idx, self.max_len)

        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.float32)

# 5. CNN 模型定义
class CharCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, num_filters=64, kernel_sizes=[2, 3, 4], dropout=0.5):
        super(CharCNN, self).__init__()

        # 字符嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 多个卷积层（不同的 kernel size）
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=num_filters,
                     kernel_size=k)
            for k in kernel_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]

        # 应用多个卷积核
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # [batch_size, num_filters, seq_len - kernel_size + 1]
            pooled = torch.max_pool1d(conv_out, conv_out.shape[2])  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]

        # 拼接所有卷积输出
        cat_output = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(kernel_sizes)]

        # Dropout 和全连接
        dropped = self.dropout(cat_output)
        logits = self.fc(dropped)  # [batch_size, 1]

        return logits.squeeze(1)  # [batch_size]

# 6. 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for texts, labels in dataloader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

# 7. 验证函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels

# 8. 数据划分
X_train_text, X_val_text, y_train, y_val = train_test_split(
    data["word"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

print(f"📈 数据划分:")
print(f"   - 训练集: {len(X_train_text)} 条")
print(f"   - 验证集: {len(X_val_text)} 条")
print()

# 9. 构建字符词典
char_to_idx = build_char_vocab(data["word"])
print(f"📝 字符词典大小: {len(char_to_idx)} 个字符")
print(f"   包括特殊标记: <PAD>, <UNK>")
print()

# 保存词典
with open('char_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(char_to_idx, f, ensure_ascii=False, indent=2)

# 10. 创建数据集和数据加载器
max_len = max(len(word) for word in data["word"])
print(f"🔢 最大词语长度: {max_len} 个字符")
print()

train_dataset = FruitDataset(X_train_text, y_train, char_to_idx, max_len)
val_dataset = FruitDataset(X_val_text, y_val, char_to_idx, max_len)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 11. 创建模型
model = CharCNN(
    vocab_size=len(char_to_idx),
    embedding_dim=32,
    num_filters=64,
    kernel_sizes=[2, 3, 4],
    dropout=0.5
).to(device)

print("🏗️  模型结构:")
print(model)
print()
print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters())} 个")
print()

# 12. 定义损失函数和优化器
# 使用正样本权重处理类别不平衡
pos_weight = torch.tensor([sum(data['label'] == 0) / sum(data['label'] == 1)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 13. 训练模型
num_epochs = 50
best_val_accuracy = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("🚀 开始训练...")
print("=" * 60)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 保存最佳模型
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_acc,
            'char_to_idx': char_to_idx,
            'max_len': max_len
        }, 'best_model.pth')

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

print("=" * 60)
print()

# 14. 加载最佳模型进行最终评估
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

_, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

print("=" * 60)
print("📊 最终模型评估结果")
print("=" * 60)
print(f"✅ 最佳验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
print()

# 15. 详细分类报告
print("📋 验证集详细报告:")
print(classification_report(val_labels, val_preds, target_names=["非水果", "水果"], zero_division=0))

# 16. 混淆矩阵
print("🔢 验证集混淆矩阵:")
cm = confusion_matrix(val_labels, val_preds)

print()
print("┌" + "─" * 50 + "┐")
print("│" + " " * 18 + "预测结果" + " " * 18 + "│")
print("├" + "─" * 15 + "┬" + "─" * 16 + "┬" + "─" * 16 + "┤")
print(f"│{'实际类别':^13}│{'非水果':^14}│{'水果':^14}│")
print("├" + "─" * 15 + "┼" + "─" * 16 + "┼" + "─" * 16 + "┤")
print(f"│ 非水果        │{cm[0][0]:^16d}│{cm[0][1]:^16d}│")
print(f"│ 水果          │{cm[1][0]:^16d}│{cm[1][1]:^16d}│")
print("└" + "─" * 15 + "┴" + "─" * 16 + "┴" + "─" * 16 + "┘")
print()

print("📌 说明:")
print(f"   ✅ 预测正确: {cm[0][0] + cm[1][1]} 个 ({(cm[0][0] + cm[1][1])/len(val_labels)*100:.1f}%)")
print(f"   ❌ 预测错误: {cm[0][1] + cm[1][0]} 个 ({(cm[0][1] + cm[1][0])/len(val_labels)*100:.1f}%)")
print(f"      - 把水果误判为非水果: {cm[1][0]} 个 (漏判)")
print(f"      - 把非水果误判为水果: {cm[0][1]} 个 (误判)")
print()

# 17. 保存训练历史
with open('history.json', 'w') as f:
    json.dump(history, f)

print("✅ 模型训练完成并已保存！")
print(f"   - 模型文件: best_model.pth")
print(f"   - 字符词典: char_vocab.json")
print(f"   - 训练历史: history.json")
