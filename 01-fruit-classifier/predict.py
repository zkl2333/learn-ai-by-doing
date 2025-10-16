import joblib

# 1. 加载模型和向量器
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def predict_word(word: str):
    X = vectorizer.transform([word])
    pred = model.predict(X)[0]
    return "水果" if pred == 1 else "非水果"

# 2. 测试
if __name__ == "__main__":
    while True:
        word = input("请输入一个词（或按回车退出）：").strip()
        if not word:
            break
        print(f"{word} -> {predict_word(word)}")
