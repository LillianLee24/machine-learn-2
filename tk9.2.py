import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Создание набора данных (Пример)
data = {
    'education': ['bachelor', 'master', 'phd', 'bachelor', 'highschool', 
                  'master', 'bachelor', 'phd', 'highschool', 'master'],
    'experience': ['junior', 'mid', 'senior', 'junior', 'junior', 
                   'senior', 'mid', 'senior', 'junior', 'mid'],
    'hired': [0, 1, 1, 0, 0, 1, 1, 1, 0, 1]  # 1 - принят, 0 - не принят
}

df = pd.DataFrame(data)

# Преобразование категориальных данных с помощью OneHotEncoder
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[['education', 'experience']]).toarray()

# Получаем имена новых закодированных столбцов
feature_names = encoder.get_feature_names_out(['education', 'experience'])

# Создание DataFrame с закодированными признаками
X = pd.DataFrame(encoded_features, columns=feature_names)
y = df['hired']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Создание и обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Предсказание результатов на тестовой выборке
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Тестирование на новых данных
new_candidates = pd.DataFrame(encoder.transform([
    ['phd', 'junior'],  # Кандидат с PhD и младший уровень
    ['bachelor', 'senior']  # Кандидат с бакалавром и старший уровень
]).toarray(), columns=feature_names)

predictions = model.predict(new_candidates)
print("Predictions for new candidates:", predictions)
