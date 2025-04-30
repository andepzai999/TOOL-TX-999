import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random

# Tạo dữ liệu mẫu (Bạn sẽ cần dữ liệu thực tế)
# 0 = Xỉu, 1 = Tài
data = {
    'X1': [random.choice([0, 1]) for _ in range(100)],  # Các tính năng (có thể là dữ liệu trước đó)
    'X2': [random.choice([0, 1]) for _ in range(100)],
    'X3': [random.choice([0, 1]) for _ in range(100)],
    'Result': [random.choice([0, 1]) for _ in range(100)]  # Kết quả (0 = Xỉu, 1 = Tài)
}

df = pd.DataFrame(data)

# Chia dữ liệu thành Train và Test
X = df.drop(columns='Result')
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xây dựng mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Dự đoán kết quả
y_pred = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100}%")

# Dự đoán một kết quả mới
new_data = np.array([[1, 0, 1]])  # Các tính năng (đầu vào mới)
prediction = model.predict(new_data)
print(f"Dự đoán kết quả: {'Tài' if prediction[0] == 1 else 'Xỉu'}")
