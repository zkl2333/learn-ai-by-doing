import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取训练历史
with open('history.json', 'r') as f:
    history = json.load(f)

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
axes[0].plot(history['train_loss'], label='训练损失', linewidth=2)
axes[0].plot(history['val_loss'], label='验证损失', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 准确率曲线
axes[1].plot(history['train_acc'], label='训练准确率', linewidth=2)
axes[1].plot(history['val_acc'], label='验证准确率', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print("✅ 训练曲线已保存到 training_curves.png")

# 显示最佳结果
best_epoch = history['val_acc'].index(max(history['val_acc']))
print()
print("=" * 60)
print("📊 训练总结")
print("=" * 60)
print(f"✅ 最佳 Epoch: {best_epoch + 1}")
print(f"   - 训练准确率: {history['train_acc'][best_epoch]:.4f}")
print(f"   - 验证准确率: {history['val_acc'][best_epoch]:.4f}")
print(f"   - 训练损失: {history['train_loss'][best_epoch]:.4f}")
print(f"   - 验证损失: {history['val_loss'][best_epoch]:.4f}")
print()

# 分析过拟合情况
train_val_gap = history['train_acc'][best_epoch] - history['val_acc'][best_epoch]
if train_val_gap > 0.15:
    print("⚠️  模型可能存在过拟合（训练准确率明显高于验证准确率）")
elif train_val_gap < 0.05:
    print("✅ 模型泛化良好（训练和验证准确率接近）")
else:
    print("ℹ️  模型略有过拟合，属于正常范围")

plt.show()
