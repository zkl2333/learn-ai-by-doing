import json
import matplotlib.pyplot as plt
import matplotlib
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 配置 ==========
HISTORY_PATH = "history.json"
OUTPUT_PATH = "training_curve.png"

# ========== 加载训练历史 ==========
if not os.path.exists(HISTORY_PATH):
    print(f"❌ 未找到训练历史文件: {HISTORY_PATH}")
    print(f"   请先运行 train.py 进行训练")
    exit(1)

print(f"📂 加载训练历史: {HISTORY_PATH}")
with open(HISTORY_PATH, 'r') as f:
    history = json.load(f)

epochs = list(range(1, len(history['loss']) + 1))
loss = history['loss']

print(f"✅ 加载成功")
print(f"   - 训练轮数: {len(epochs)}")
print(f"   - 最终 Loss: {loss[-1]:.4f}")
print()

# ========== 绘制训练曲线 ==========
print("🎨 生成训练曲线...")

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'b-', linewidth=2, label='训练 Loss')
plt.xlabel('训练轮数 (Epoch)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('RNN 昵称生成器训练曲线', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# 添加统计信息
stats_text = f"最小 Loss: {min(loss):.4f}\n最终 Loss: {loss[-1]:.4f}"
plt.text(0.02, 0.98, stats_text,
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='top',
         fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✅ 训练曲线已保存到 {OUTPUT_PATH}")
print()

# ========== 显示图像 ==========
try:
    plt.show()
except Exception as e:
    print(f"⚠️  无法显示图像: {e}")
    print(f"   请直接查看 {OUTPUT_PATH}")