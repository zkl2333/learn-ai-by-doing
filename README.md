# Learn AI by Doing

从零开始学习 AI，用实际项目记录学习过程。

## 学习路线

每个项目都是独立的，从简单到复杂，循序渐进。所有项目都边做边学。

## 项目列表

### ✅ 01. 水果分类器

**简介**: 判断一个中文词语是不是水果的二分类器

**技术栈**: [逻辑回归](./notes/术语解释.md#逻辑回归-logistic-regression)、字符级 [n-gram](./notes/术语解释.md#n-gram)、[sklearn](./notes/术语解释.md#sklearn-scikit-learn)

**结果**: 验证集准确率 80%，涉及[过拟合](./notes/术语解释.md#过拟合-overfitting)处理、[类别不平衡](./notes/术语解释.md#类别不平衡-class-imbalance)、[数据泄露](./notes/术语解释.md#数据泄露-data-leakage)等问题

👉 [查看详情](./01-fruit-classifier/)

---

### ✅ 02. 水果分类器 - 字符级 CNN 版

**简介**: 用字符级 CNN 重做项目 01，对比深度学习和传统机器学习

**技术栈**: 字符级 [CNN](./notes/术语解释.md#cnn-convolutional-neural-network)、[字符嵌入](./notes/术语解释.md#字符嵌入-character-embedding)、多尺度[卷积](./notes/术语解释.md#卷积核-kernel)、[PyTorch](./notes/术语解释.md#pytorch)

**结果**: 
- 经过 4 轮迭代 + 超参数调优
- CNN 验证准确率 59% vs 逻辑回归 86%
- 结论：小数据集上传统 ML 可能比深度学习更有效

**副产物**: 完整的 PyTorch 训练流程实践

👉 [查看详情](./02-fruit-classifier-cnn/)

---

## 学习笔记

记录学习过程中的一些理解和总结。

### 📖 [术语解释](./notes/术语解释.md)

整理了学习过程中遇到的 AI 专业术语，用通俗的语言解释，方便初学者理解：

- **机器学习基础**：逻辑回归、训练集/测试集、验证集等
- **特征工程**：特征、n-gram、字符级、字符嵌入等
- **模型评估**：准确率、混淆矩阵、精确率、召回率、F1 分数等
- **深度学习**：CNN、卷积核、池化、Dropout、学习率、批次大小、轮次等
- **常见问题**：过拟合、欠拟合、类别不平衡、数据泄露、正则化、超参数等
- **工具和库**：sklearn、PyTorch 等

---

## 关于我

一个 AI 初学者，通过动手做项目来学习机器学习。

选择这种学习方式的原因：
- 光看理论记不住，得动手做
- 从简单项目开始，快速获得成就感
- 用 Claude Code 边做边学，效率更高

---

## 技术栈

Python • [scikit-learn](./notes/术语解释.md#sklearn-scikit-learn) • [PyTorch](./notes/术语解释.md#pytorch) • pandas • matplotlib

> 💡 **不熟悉这些术语？** 查看 [术语解释](./notes/术语解释.md) 了解详细说明

---

## License

MIT