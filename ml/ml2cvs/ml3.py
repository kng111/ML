import pandas as pd  # Импортируем библиотеку pandas для работы с данными в виде таблицы (DataFrame).
from sklearn.model_selection import train_test_split  # Импортируем функцию train_test_split для разделения данных на обучающую и тестовую выборки.
from sklearn.linear_model import LinearRegression  # Импортируем модель линейной регрессии из библиотеки scikit-learn.

# Загрузка данных из CSV
data = pd.read_csv(r'C:\Users\yakvl\Desktop\ml\ml2cvs\1.csv')  # Загружаем данные из CSV файла в формате DataFrame.

# Влияние друзей - ключевой фактор
X = data[['Влияние_друзей', 'Социальный_статус', 'Уровень_жизни', 'Уровень_дохода', 'Потенциал']]  # Выбираем признаки, которые будут использованы для обучения модели.
y = data[['Влияние_друзей', 'Социальный_статус', 'Уровень_жизни', 'Уровень_дохода', 'Потенциал']]  # Выбираем переменные, которые мы хотим предсказывать.

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Разделяем данные на обучающую и тестовую выборки в соотношении 80% к 20%.

# Веса для признаков (влияние друзей имеет вес 2, остальные - 1)
weights = [2, 1, 1, 1, 1]  # Устанавливаем веса для каждого признака. Здесь влияние друзей имеет больший вес.

# Создание и обучение модели с весами
model = LinearRegression()  # Создаем модель линейной регрессии.
model.fit(X_train, y_train, sample_weight=X_train['Влияние_друзей'].apply(lambda x: weights[0]))  # Обучаем модель с учетом весов.

# Функция для предсказания всех параметров
def predict_parameters(friends_influence, social_status, life_level, income_level, potential):
    # Подготовка данных для модели
    data = [[friends_influence, social_status, life_level, income_level, potential]]  # Готовим данные в нужном формате.

    try:
        # Ограничим значения от 1 до 10
        friends_influence = max(min(friends_influence, 10), 1)  # Ограничиваем влияние друзей в диапазоне от 1 до 10.
        social_status = max(min(social_status, 10), 1)  # Ограничиваем социальный статус в диапазоне от 1 до 10.
        life_level = max(min(life_level, 10), 1)  # Ограничиваем уровень жизни в диапазоне от 1 до 10.
        income_level = max(min(income_level, 10), 1)  # Ограничиваем уровень дохода в диапазоне от 1 до 10.
        potential = max(min(potential, 10), 1)  # Ограничиваем потенциал в диапазоне от 1 до 10.

        # Предсказание всех параметров
        parameters_prediction = model.predict(data)  # Предсказываем параметры с помощью обученной модели.
        influence, status, live, income, pot = parameters_prediction[0]  # Разделяем предсказанные значения на отдельные переменные.

        # Округление предполагаемого дохода
        # income = round(income)  # Эта строка закомментирована. Предполагается, что доход уже округлен в модели.

        return f"Предполагаемое влияние: {influence:.3f}, Предполагаемый социальный статус: {status:.3f}, " \
               f"Уровень жизни: {live:.3f}, Предполагаемый доход: {income:.3f}, Потенциал: {pot:.3f}"  # Возвращаем предсказанные значения.

    except Exception as e:
        return f"Ошибка: {e}"  # Если произошла ошибка, возвращаем сообщение об ошибке.

# Пример использования функции
result = predict_parameters(5, 9, 10, 7, 8)  # Вызываем функцию с определенными параметрами.
print(result)  # Печатаем результат.
