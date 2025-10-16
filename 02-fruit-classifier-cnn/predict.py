import torch
import torch.nn as nn
import json

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN 模型定义（需要和训练时保持一致）
class CharCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, num_filters=64, kernel_sizes=[2, 3, 4], dropout=0.5):
        super(CharCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=num_filters,
                     kernel_size=k)
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

# 文本编码函数
def encode_text(text, char_to_idx, max_len):
    encoded = []
    for char in text[:max_len]:
        encoded.append(char_to_idx.get(char, char_to_idx['<UNK>']))

    while len(encoded) < max_len:
        encoded.append(char_to_idx['<PAD>'])

    return encoded

# 加载模型和词典
def load_model(model_path='best_model.pth', vocab_path='char_vocab.json'):
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    char_to_idx = checkpoint['char_to_idx']
    max_len = checkpoint['max_len']

    # 创建模型
    model = CharCNN(
        vocab_size=len(char_to_idx),
        embedding_dim=32,
        num_filters=64,
        kernel_sizes=[2, 3, 4],
        dropout=0.5
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, char_to_idx, max_len

# 预测函数
def predict(text, model, char_to_idx, max_len):
    encoded = encode_text(text, char_to_idx, max_len)
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    is_fruit = prob > 0.5
    return is_fruit, prob

# 主程序
if __name__ == "__main__":
    print("🍎 水果分类器 (字符级 CNN 版)")
    print("=" * 50)
    print()

    # 加载模型
    print("📦 加载模型...")
    model, char_to_idx, max_len = load_model()
    print(f"✅ 模型加载成功！")
    print(f"   - 字符词典大小: {len(char_to_idx)}")
    print(f"   - 最大序列长度: {max_len}")
    print()

    # 测试样例
    test_words = [
        "苹果", "香蕉", "橙子", "西瓜", "草莓",
        "电脑", "手机", "桌子", "椅子", "书本",
        "火龙果", "猕猴桃", "榴莲", "山竹", "芒果"
    ]

    print("🧪 测试结果:")
    print("-" * 50)

    for word in test_words:
        is_fruit, prob = predict(word, model, char_to_idx, max_len)
        result = "✅ 是水果" if is_fruit else "❌ 不是水果"
        print(f"{word:10s} | {result} | 置信度: {prob:.4f}")

    print("-" * 50)
    print()

    # 交互式预测
    print("💡 你也可以输入词语进行预测（输入 'q' 退出）:")
    while True:
        user_input = input("\n请输入词语: ").strip()

        if user_input.lower() == 'q':
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        is_fruit, prob = predict(user_input, model, char_to_idx, max_len)
        result = "✅ 是水果" if is_fruit else "❌ 不是水果"
        print(f"预测结果: {result} (置信度: {prob:.4f})")
