# 中文昵称生成器 - 配置文件

# ========== 模型架构配置 ==========
# 注意: 这些参数训练后不能改变,否则无法加载模型
HIDDEN_SIZE = 128        # GRU隐藏层大小
EMBEDDING_DIM = 128      # 字符嵌入维度

# ========== 训练配置 ==========
# 这些参数可以在续训时调整
EPOCHS = 5              # 训练轮数
SEQ_LEN = 20            # 序列长度
LR = 1e-3               # 学习率
DATA_LIMIT = None       # 数据行数限制 (None = 使用全部数据)

# ========== 生成配置 ==========
TEMPERATURE = 0.8       # 温度参数 (0.5-1.5)
MAX_GEN_LEN = 15        # 最大生成长度

# ========== 文件路径 ==========
DATA_PATH = "nicknames.txt"
MODEL_PATH = "nickname_rnn.pth"
HISTORY_PATH = "history.json"
VOCAB_PATH = "vocab.json"

# ========== 设备配置 ==========
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# ========== 随机种子 ==========
RANDOM_SEED = 42
