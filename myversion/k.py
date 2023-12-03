import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Пример данных
data = [
    {"text": "Это пример предложения 1.", "punctuation": "Это пример предложения 1!"},
    {"text": "Пример текста 2.", "punctuation": "Пример текста 2?"},
    {"text": "Еще один пример текста.", "punctuation": "Еще один пример текста."},
    # Добавьте другие данные
]

# Разделение на обучающую и проверочную выборки
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts([sentence['text'] for sentence in train_data])

# Преобразование текста в числовой формат
X_train = tokenizer.texts_to_sequences([sentence['text'] for sentence in train_data])
X_val = tokenizer.texts_to_sequences([sentence['text'] for sentence in val_data])

# Заполнение последовательностей, чтобы все они имели одинаковую длину
X_train = pad_sequences(X_train)
X_val = pad_sequences(X_val)

# Преобразование пунктуации в числовой формат и one-hot кодирование

y_train = tokenizer.texts_to_sequences([sentence['punctuation'] for sentence in train_data])
y_train = pad_sequences(y_train, maxlen=X_train.shape[1], padding='post')
y_train = to_categorical(y_train, num_classes=len(tokenizer.word_index) + 1)

y_val = tokenizer.texts_to_sequences([sentence['punctuation'] for sentence in val_data])
y_val = pad_sequences(y_val, maxlen=X_val.shape[1], padding='post')
y_val = to_categorical(y_val, num_classes=len(tokenizer.word_index) + 1)

# Создание модели
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=200, input_length=X_train.shape[1]))
model.add(LSTM(200, return_sequences=True))
model.add(TimeDistributed(Dense(len(tokenizer.word_index) + 1, activation='softmax')))

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели с увеличенным количеством эпох
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Оценка модели на проверочных данных
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f'Loss on validation data: {loss:.4f}')
print(f'Accuracy on validation data: {accuracy * 100:.2f}%')

# Пример использования модели для предсказания нескольких примеров
for example_text in ["Новое предложение для проверки.", "Пример текста.", "Еще один пример."]:
    tokenized_example = tokenizer.texts_to_sequences([example_text])
    tokenized_example = pad_sequences(tokenized_example, maxlen=X_train.shape[1])
    punctuation_prediction = model.predict(tokenized_example)

    # Распечатаем предсказание
    predicted_punctuation_index = np.argmax(punctuation_prediction, axis=-1)
    predicted_punctuation = tokenizer.sequences_to_texts(predicted_punctuation_index)
    print(f'Input text: {example_text}')
    print(f'Predicted punctuation: {predicted_punctuation}\n')
