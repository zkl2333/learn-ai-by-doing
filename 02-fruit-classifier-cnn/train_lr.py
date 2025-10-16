import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. 读取数据
data = pd.read_csv("data.csv")
print(f"📊 数据集大小: {len(data)} 条记录")
print(f"   - 水果样本: {sum(data['label'] == 1)} 条")
print(f"   - 非水果样本: {sum(data['label'] == 0)} 条")
print()

# 2. 随机划分训练集和验证集 (80% 训练, 20% 验证)
X_train_text, X_val_text, y_train, y_val = train_test_split(
    data["word"],
    data["label"],
    test_size=0.2,      # 20% 用于验证
    random_state=42,    # 设置随机种子，保证结果可复现
    stratify=data["label"]  # 分层采样，保持标签比例
)

print(f"📈 数据划分:")
print(f"   - 训练集: {len(X_train_text)} 条")
print(f"   - 验证集: {len(X_val_text)} 条")
print()

# 3. 特征提取（按字符分解，增加 n-gram 特征）
vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(1, 2),  # 使用 1-gram 和 2-gram 字符组合
    max_features=500     # 限制特征数量防止过拟合
)
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

# 4. 训练逻辑回归模型（使用类别权重平衡）
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # 自动平衡类别权重
    C=0.5,                    # 增强正则化，降低 C 值防止过拟合
    random_state=42,
    solver='liblinear'        # 适合小数据集的求解器
)
model.fit(X_train, y_train)

# 5. 计算准确率
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_accuracy = accuracy_score(y_train, train_pred)
val_accuracy = accuracy_score(y_val, val_pred)

print("=" * 50)
print("📊 模型评估结果")
print("=" * 50)
print(f"✅ 训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"✅ 验证集准确率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print()

# 6. 显示详细的分类报告
print("📋 验证集详细报告:")
print(classification_report(y_val, val_pred, target_names=["非水果", "水果"], zero_division=0))

# 7. 混淆矩阵
print("🔢 验证集混淆矩阵:")
cm = confusion_matrix(y_val, val_pred)

# 美化输出
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

# 添加说明
print("📌 说明:")
print(f"   ✅ 预测正确: {cm[0][0] + cm[1][1]} 个 ({(cm[0][0] + cm[1][1])/len(y_val)*100:.1f}%)")
print(f"   ❌ 预测错误: {cm[0][1] + cm[1][0]} 个 ({(cm[0][1] + cm[1][0])/len(y_val)*100:.1f}%)")
print(f"      - 把水果误判为非水果: {cm[1][0]} 个 (漏判)")
print(f"      - 把非水果误判为水果: {cm[0][1]} 个 (误判)")
print()

# 8. 保存模型和特征提取器
joblib.dump(model, "model_lr.joblib")
joblib.dump(vectorizer, "vectorizer_lr.joblib")

print("✅ 逻辑回归模型训练完成并已保存。")
