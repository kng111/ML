import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Загрузите стоп-слова (если еще не загружены)
# nltk.download('stopwords')

# Предобработка данных
def preprocess_text(text):
    # Токенизация слов
    tokens = word_tokenize(text)

    # Удаление стоп-слов и пунктуации
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]

    # Преобразование в нижний регистр
    tokens = [word.lower() for word in tokens]

    return tokens

# Пример данных
data = [
    {"text": "Это пример предложения 1.", "punctuation": "Это пример предложения 1!"},
    {"text": "Пример текста 2.", "punctuation": "Пример текста 2?"},
    {"text": "Еще один пример текста.", "punctuation": "Еще один пример текста."},
    # Добавьте другие данные
]

# Создание токенизированных предложений
tokenized_data = []
for example in data:
    tokens = preprocess_text(example["text"])
    tokenized_data.append({"tokens": tokens, "punctuation": example["punctuation"]})

# Разделение на обучающую, проверочную и тестовую выборки
train_data, test_data = train_test_split(tokenized_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Печать первого примера в обучающей выборке
print("Пример обучающего текста:", train_data[0]["tokens"])
print("Соответствующая пунктуация:", train_data[0]["punctuation"])
