import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from itertools import product

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")
print()

# 1. 读取数据
data = pd.read_csv("data.csv")
print(f"📊 数据集: {len(data)} 条 ({sum(data['label'] == 1)} 水果 + {sum(data['label'] == 0)} 非水果)")
print()

# 2. 构建字符词典
def build_char_vocab(words, max_chars=100):
    chars = set()
    for word in words:
        chars.update(word)

    char_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for char in sorted(chars):
        if len(char_to_idx) >= max_chars:
            break
        char_to_idx[char] = len(char_to_idx)

    return char_to_idx

# 3. 文本编码
def encode_text(text, char_to_idx, max_len=10):
    encoded = []
    for char in text[:max_len]:
        encoded.append(char_to_idx.get(char, char_to_idx['<UNK>']))

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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            pooled = torch.max_pool1d(conv_out, conv_out.shape[2])
            conv_outputs.append(pooled.squeeze(2))

        cat_output = torch.cat(conv_outputs, dim=1)
        dropped = self.dropout(cat_output)
        logits = self.fc(dropped)
        return logits.squeeze(1)

# 6. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=10):
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # 训练
        model.train()
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)

        # 早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_acc, all_preds, all_labels

# 7. 数据划分
X_train_text, X_val_text, y_train, y_val = train_test_split(
    data["word"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
)

char_to_idx = build_char_vocab(data["word"])
max_len = max(len(word) for word in data["word"])

train_dataset = FruitDataset(X_train_text, y_train, char_to_idx, max_len)
val_dataset = FruitDataset(X_val_text, y_val, char_to_idx, max_len)

# 8. 超参数搜索空间
hyperparams = {
    'embedding_dim': [16, 32, 64],
    'num_filters': [32, 64, 128],
    'dropout': [0.3, 0.5, 0.7],
    'learning_rate': [0.0005, 0.001, 0.002],
    'batch_size': [16, 32, 64]
}

print("🔍 开始超参数搜索...")
print(f"   搜索空间大小: {np.prod([len(v) for v in hyperparams.values()])} 组参数")
print()

# 9. 网格搜索（采样前20组避免时间过长）
results = []
param_combinations = list(product(
    hyperparams['embedding_dim'],
    hyperparams['num_filters'],
    hyperparams['dropout'],
    hyperparams['learning_rate'],
    hyperparams['batch_size']
))

# 随机采样20组参数
np.random.shuffle(param_combinations)
param_combinations = param_combinations[:20]

for idx, (emb_dim, n_filters, drop, lr, batch_sz) in enumerate(param_combinations, 1):
    print(f"[{idx}/20] Testing: emb={emb_dim}, filters={n_filters}, drop={drop}, lr={lr}, batch={batch_sz}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)

    # 创建模型
    model = CharCNN(
        vocab_size=len(char_to_idx),
        embedding_dim=emb_dim,
        num_filters=n_filters,
        kernel_sizes=[2, 3, 4],
        dropout=drop
    ).to(device)

    # 损失函数和优化器
    pos_weight = torch.tensor([sum(data['label'] == 0) / sum(data['label'] == 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    val_acc, preds, labels = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=30, patience=10)

    # 计算详细指标
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    result = {
        'embedding_dim': emb_dim,
        'num_filters': n_filters,
        'dropout': drop,
        'learning_rate': lr,
        'batch_size': batch_sz,
        'val_accuracy': val_acc,
        'non_fruit_precision': precision[0],
        'non_fruit_recall': recall[0],
        'fruit_precision': precision[1],
        'fruit_recall': recall[1]
    }
    results.append(result)

    print(f"   ✅ Val Acc: {val_acc:.4f} | Non-fruit P/R: {precision[0]:.2f}/{recall[0]:.2f} | Fruit P/R: {precision[1]:.2f}/{recall[1]:.2f}")
    print()

# 10. 保存结果
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('val_accuracy', ascending=False)
results_df.to_csv('tuning_results.csv', index=False)

print("=" * 80)
print("🏆 Top 5 最佳参数组合:")
print("=" * 80)
for idx, row in results_df.head(5).iterrows():
    print(f"\n{results_df.index.get_loc(idx) + 1}. Val Acc: {row['val_accuracy']:.4f} ({row['val_accuracy']*100:.2f}%)")
    print(f"   - embedding_dim={int(row['embedding_dim'])}, num_filters={int(row['num_filters'])}, dropout={row['dropout']}")
    print(f"   - learning_rate={row['learning_rate']}, batch_size={int(row['batch_size'])}")
    print(f"   - Non-fruit: P={row['non_fruit_precision']:.2f} R={row['non_fruit_recall']:.2f}")
    print(f"   - Fruit: P={row['fruit_precision']:.2f} R={row['fruit_recall']:.2f}")

print("\n✅ 超参数搜索完成！结果已保存到 tuning_results.csv")
