import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Данные (Пример)
data = {
    'age': [25, 35, 45, 50, 23, 34, 54, 60, 46, 31],
    'cholesterol_level': [190, 210, 230, 220, 180, 205, 240, 245, 235, 200],
    'blood_pressure': [120, 130, 135, 140, 118, 128, 145, 150, 138, 125],
    'has_heart_disease': [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]  # 1 - есть болезнь, 0 - нет
}

df = pd.DataFrame(data)

# Выбор признаков (независимые переменные) и меток (зависимая переменная)
X = df[['age', 'cholesterol_level', 'blood_pressure']]
y = df['has_heart_disease']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Создание и обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Предсказание результатов на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
