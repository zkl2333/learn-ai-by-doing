import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib import rcParams
from wordcloud import WordCloud

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 自动查找中文字体
import os
import sys

def find_chinese_font():
    """自动查找系统中的中文字体"""
    if sys.platform.startswith('win'):
        # Windows 字体路径
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',      # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',    # 黑体
            'C:/Windows/Fonts/simsun.ttc',    # 宋体
            'C:/Windows/Fonts/simkai.ttf',    # 楷体
        ]
    elif sys.platform == 'darwin':
        # macOS 字体路径
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
    else:
        # Linux 字体路径
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path

    print("⚠️  警告：未找到中文字体，词云可能无法正常显示中文")
    return None

FONT_PATH = find_chinese_font()

print("📊 加载逻辑回归模型和特征提取器...")
if FONT_PATH:
    print(f"✅ 找到中文字体: {FONT_PATH}")
model = joblib.load("model_lr.joblib")
vectorizer = joblib.load("vectorizer_lr.joblib")

# 获取特征名称和权重
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# 创建特征权重 DataFrame
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'weight': coefficients,
    'abs_weight': np.abs(coefficients)
})

# 按绝对值排序
feature_importance = feature_importance.sort_values('abs_weight', ascending=False)

print(f"\n✅ 已加载 {len(feature_names)} 个特征")
print(f"   - 正权重特征（倾向水果）: {sum(coefficients > 0)} 个")
print(f"   - 负权重特征（倾向非水果）: {sum(coefficients < 0)} 个")
print()

# ============================================
# 1. 显示 Top 特征
# ============================================
print("=" * 60)
print("🔝 最能区分水果和非水果的特征 (Top 20)")
print("=" * 60)
print()

# Top 10 水果特征
top_fruit = feature_importance[feature_importance['weight'] > 0].head(10)
print("🍎 Top 10 水果特征（权重越高越像水果）:")
for idx, (_, row) in enumerate(top_fruit.iterrows(), 1):
    print(f"   {idx:2d}. '{row['feature']:4s}' → +{row['weight']:6.3f}")
print()

# Top 10 非水果特征
top_non_fruit = feature_importance[feature_importance['weight'] < 0].head(10)
print("🚫 Top 10 非水果特征（权重越低越不像水果）:")
for idx, (_, row) in enumerate(top_non_fruit.iterrows(), 1):
    print(f"   {idx:2d}. '{row['feature']:4s}' → {row['weight']:6.3f}")
print()

# ============================================
# 2. 绘制特征权重条形图
# ============================================
print("📈 生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('水果分类器特征可视化', fontsize=20, fontweight='bold')

# 2.1 Top 15 最重要特征（按绝对值）
ax1 = axes[0, 0]
top_features = feature_importance.head(15)
colors = ['#FF6B6B' if w > 0 else '#4ECDC4' for w in top_features['weight']]
bars = ax1.barh(range(len(top_features)), top_features['weight'], color=colors)
ax1.set_yticks(range(len(top_features)))
ax1.set_yticklabels(top_features['feature'])
ax1.set_xlabel('权重系数', fontsize=12)
ax1.set_title('Top 15 最重要特征', fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(axis='x', alpha=0.3)
ax1.invert_yaxis()

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', label='水果特征（正权重）'),
    Patch(facecolor='#4ECDC4', label='非水果特征（负权重）')
]
ax1.legend(handles=legend_elements, loc='lower right')

# 2.2 Top 10 水果特征
ax2 = axes[0, 1]
top_fruit_10 = feature_importance[feature_importance['weight'] > 0].head(10)
bars = ax2.barh(range(len(top_fruit_10)), top_fruit_10['weight'], color='#FF6B6B')
ax2.set_yticks(range(len(top_fruit_10)))
ax2.set_yticklabels(top_fruit_10['feature'])
ax2.set_xlabel('权重系数', fontsize=12)
ax2.set_title('Top 10 水果特征', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# 2.3 Top 10 非水果特征
ax3 = axes[1, 0]
top_non_fruit_10 = feature_importance[feature_importance['weight'] < 0].head(10)
bars = ax3.barh(range(len(top_non_fruit_10)), top_non_fruit_10['weight'], color='#4ECDC4')
ax3.set_yticks(range(len(top_non_fruit_10)))
ax3.set_yticklabels(top_non_fruit_10['feature'])
ax3.set_xlabel('权重系数', fontsize=12)
ax3.set_title('Top 10 非水果特征', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
ax3.invert_yaxis()

# 2.4 特征权重分布直方图
ax4 = axes[1, 1]
ax4.hist(coefficients, bins=50, color='#95E1D3', edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零点')
ax4.set_xlabel('权重系数', fontsize=12)
ax4.set_ylabel('特征数量', fontsize=12)
ax4.set_title('特征权重分布', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 添加统计信息
mean_weight = np.mean(coefficients)
std_weight = np.std(coefficients)
ax4.text(0.02, 0.98,
         f'均值: {mean_weight:.4f}\n标准差: {std_weight:.4f}',
         transform=ax4.transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=10)

plt.tight_layout()
plt.savefig('feature_visualization_lr.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存为 feature_visualization_lr.png")
print()

# ============================================
# 2.5 生成词云图
# ============================================
print("☁️  生成词云图...")

# 创建词云图
fig_cloud, axes_cloud = plt.subplots(1, 2, figsize=(20, 8))
fig_cloud.suptitle('特征词云 - 水果 vs 非水果', fontsize=20, fontweight='bold')

# 水果特征词云
fruit_features = feature_importance[feature_importance['weight'] > 0]
fruit_freq = dict(zip(fruit_features['feature'], fruit_features['weight']))

wordcloud_fruit = WordCloud(
    font_path=FONT_PATH,
    width=800,
    height=600,
    background_color='white',
    colormap='Reds',
    relative_scaling=0.5,
    min_font_size=10
).generate_from_frequencies(fruit_freq)

axes_cloud[0].imshow(wordcloud_fruit, interpolation='bilinear')
axes_cloud[0].axis('off')
axes_cloud[0].set_title('水果特征（正权重）', fontsize=16, fontweight='bold', pad=20, color='#D63031')

# 非水果特征词云
non_fruit_features = feature_importance[feature_importance['weight'] < 0]
non_fruit_freq = dict(zip(non_fruit_features['feature'], np.abs(non_fruit_features['weight'])))

wordcloud_non_fruit = WordCloud(
    font_path=FONT_PATH,
    width=800,
    height=600,
    background_color='white',
    colormap='Blues',
    relative_scaling=0.5,
    min_font_size=10
).generate_from_frequencies(non_fruit_freq)

axes_cloud[1].imshow(wordcloud_non_fruit, interpolation='bilinear')
axes_cloud[1].axis('off')
axes_cloud[1].set_title('非水果特征（负权重）', fontsize=16, fontweight='bold', pad=20, color='#0984E3')

plt.tight_layout()
plt.savefig('wordcloud_lr.png', dpi=300, bbox_inches='tight')
print("✅ 词云图已保存为 wordcloud_lr.png")
print()

# ============================================
# 3. 分析特定字符的影响
# ============================================
print("=" * 60)
print("🔍 特定字符分析")
print("=" * 60)
print()

# 单字符特征
single_char_features = feature_importance[feature_importance['feature'].str.len() == 1]
print(f"📝 单字符特征统计（共 {len(single_char_features)} 个）:")

# 最有水果特征的字符
top_fruit_chars = single_char_features[single_char_features['weight'] > 0].head(10)
if len(top_fruit_chars) > 0:
    print("\n   🍎 最像水果的字符:")
    for idx, (_, row) in enumerate(top_fruit_chars.iterrows(), 1):
        print(f"      {idx}. '{row['feature']}' → +{row['weight']:.3f}")

# 最有非水果特征的字符
top_non_fruit_chars = single_char_features[single_char_features['weight'] < 0].head(10)
if len(top_non_fruit_chars) > 0:
    print("\n   🚫 最不像水果的字符:")
    for idx, (_, row) in enumerate(top_non_fruit_chars.iterrows(), 1):
        print(f"      {idx}. '{row['feature']}' → {row['weight']:.3f}")
print()

# 双字符特征
double_char_features = feature_importance[feature_importance['feature'].str.len() == 2]
print(f"📝 双字符特征统计（共 {len(double_char_features)} 个）:")

top_fruit_bigrams = double_char_features[double_char_features['weight'] > 0].head(10)
if len(top_fruit_bigrams) > 0:
    print("\n   🍎 最像水果的双字符:")
    for idx, (_, row) in enumerate(top_fruit_bigrams.iterrows(), 1):
        print(f"      {idx}. '{row['feature']}' → +{row['weight']:.3f}")

top_non_fruit_bigrams = double_char_features[double_char_features['weight'] < 0].head(10)
if len(top_non_fruit_bigrams) > 0:
    print("\n   🚫 最不像水果的双字符:")
    for idx, (_, row) in enumerate(top_non_fruit_bigrams.iterrows(), 1):
        print(f"      {idx}. '{row['feature']}' → {row['weight']:.3f}")
print()


print("=" * 60)
print("✨ 可视化完成！")
print("=" * 60)
print()
print("📁 生成的文件:")
print("   - feature_visualization_lr.png  (逻辑回归特征可视化图表)")
print("   - wordcloud_lr.png              (逻辑回归特征词云图)")
print()
print("💡 解读提示:")
print("   - 正权重：该特征越强，越倾向于判断为水果")
print("   - 负权重：该特征越强，越倾向于判断为非水果")
print("   - 绝对值越大，该特征的影响力越强")
