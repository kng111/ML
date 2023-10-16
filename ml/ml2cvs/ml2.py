# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # Загружаем данные



# # Загрузка данных из CSV-файла
# data = pd.read_csv(r'C:\Users\yakvl\Desktop\ml\ml2cvs\1.csv')

# # Разделение данных на признаки (X) и целевую переменную (y)
# X = data[['Влияние_друзей', 'Социальный_статус', 'Уровень_жизни', 'Уровень_дохода', 'Потенциал']]
# y = data['Кем_станете']

# # Разделение данных на обучающий и тестовый наборы
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Обучение модели
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# предсказания = model.predict(X_test)

# # Предсказание
# новые_данные = {
#     'Влияние_друзей': [10],
#     'Социальный_статус': [1],
#     'Уровень_жизни': [1],
#     'Уровень_дохода': [1],
#     'Потенциал': [1]
# }

# новые_данные_df = pd.DataFrame(новые_данные)
# предсказания = model.predict(новые_данные_df)
# общая_картина = f'Предсказание: Кем станете: {предсказания[0]} Социальный статус: {новые_данные["Социальный_статус"][0]} Уровень жизни: {новые_данные["Уровень_жизни"][0]} Уровень дохода: {новые_данные["Уровень_дохода"][0]} Потенциал: {новые_данные["Потенциал"][0]}'
# print(общая_картина)





# Загрузите данные из CSV


# Определите признаки (X) и целевую переменную (y)
from sklearn.linear_model import LinearRegression

# Создаем модель
model = LinearRegression()

# Подготовка данных для обучения
# Например, у вас есть тренировочные данные X_train и соответствующие им значения y_train

X_train = [[10, 9, 8, 7, 6], [8, 6, 7, 6, 5], [9, 7, 7, 8, 6]]
y_train = [1, 2, 3]

# Обучаем модель
model.fit(X_train, y_train)

# Теперь, модель обучена и готова к предсказанию
# Вы можете использовать функцию predict_influence с нужными значениями

def predict_influence(friends_influence, social_status, life_level, income_level, potential):
    # Подготовка данных для модели
    data = [[friends_influence, social_status, life_level, income_level, potential]]

    try:
        # Предсказание влияния
        influence_prediction = model.predict(data)
        return f"Предполагаемое влияние: {influence_prediction[0]:.2f}"

    except Exception as e:
        return f"Ошибка: {e}"

# Пример использования функции
result = predict_influence(10, 9, 8, 7, 6)
print(result)
