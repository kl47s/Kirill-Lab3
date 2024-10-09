## Лабораторная Работа №3 . Использование глубокого обучения для создания инструмента автоматической генерации музыкальных композиций.

Будем использовать midi файлы, у меня свой архив с данными в папке "midi" можете воспользоватся загрузив репозиторий или же собрать свои данные.
Работа проводилась в google colab, за работу в других средах не ручаюсь.
Для начала клонируйте репозиторий гитхаб'а:
```Ruby
!git clone [https://github.com/kl47s/kirill-ya3.git]
```
Если вы работаете в google colab, то должно вывести следующее
![image](https://github.com/Vokoon/Laba_3-1_Akimov/assets/120046709/ad1c33af-faad-4475-b419-62710aca6738)

Это библиотека необходима для чтения midi файлов, установите её для работы:
```Ruby
!pip install pretty_midi
```
![image](https://github.com/Vokoon/Laba_3-1_Akimov/assets/120046709/ec1d1c8a-4e58-436d-9f96-a10a06a80120)

```Ruby
!pip install tensorflow
```
![image](https://github.com/Vokoon/Laba_3-1_Akimov/assets/120046709/d074ea73-a3ab-479c-9c73-1d16ac646c19)

Теперь вы можете загрузить и обработать MIDI-файлы:
Создаём модель RNN(LSTM)
```Ruby
import os
import pretty_midi
from music21 import converter, note, chord, instrument

def convert_midi_files_in_directory(directory):
    notes = []
    for file in os.listdir(directory):
        if file.endswith(".mid"):
            midi_path = os.path.join(directory, file)
            try:
                midi = converter.parse(midi_path)
                notes_to_parse = None

                try: 
                    s2 = instrument.partitionByInstrument(midi)
                    if s2:  
                        notes_to_parse = s2.parts[0].recurse() 
                    else:
                        notes_to_parse = midi.flat.notes
                except Exception as e:  
                    print(f"Ошибка при разборе дорожек: {e}")
                    notes_to_parse = midi.flat.notes

                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
                print(f"Файл {file} успешно обработан.")
            except Exception as e:
                print(f"Не удалось обработать файл {file}: {e}")
    return notes

directory = '/content/Laba_3-1_Akimov/midi'
try:
    all_notes = convert_midi_files_in_directory(directory)
    note_to_int = {note: number for number, note in enumerate(sorted(set(all_notes)))}
    int_to_note = {number: note for note, number in note_to_int.items()}
    numeric_notes = [note_to_int[note] for note in all_notes]
    print("Конвертация завершена успешно.")
except Exception as e:
    print(f"Произошла ошибка при конвертации: {e}")
```
Обучаем модель RNN(LSTM)
Процесс может быть очень долгим, кол-во "эпох" меняйте по своему желанию, насколько качественный продукт вы хотите получить, я ограничусь 3.
В конце компиляции может выйти ошибка, перезапускать код нет необходимости, можем продолжать работу.

```Rudy
from keras.callbacks import ModelCheckpoint

split = int(n_patterns * 0.9)  # 90% для обучения, 10% для валидации
train_input = network_input[:split]
train_output = network_output[:split]
validation_input = network_input[split:]
validation_output = network_output[split:]

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', verbose=1, save_best_only=True, mode='min'
)

callbacks_list = [checkpoint]

history = model.fit(train_input, train_output, epochs=3, batch_size=64,
                    validation_data=(validation_input, validation_output),
                    callbacks=callbacks_list, verbose=1)

import matplotlib.pyplot as plt
```
Финальный штрих, создание самой "композиции". Одну из модель полученную выше, используете тут.
```Rudy
import numpy as np
from music21 import stream, note, chord, instrument
from keras.models import load_model
model = load_model('***')

def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-7) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)
np.random.seed(None)

def generate_music(model, network_input, n_vocab, int_to_note, num_notes=500):
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []

    for note_index in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        # Использование функции sample_with_temperature для выбора следующей ноты
        index = sample_with_temperature(prediction[0], temperature=0.1)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

# Функция для создания MIDI файла из последовательности нот
def create_midi(prediction_output, filename='test_output.mid'):
    offset = 0
    output_notes = []

    # Создание нот и добавление их в список
    for pattern in prediction_output:
        # Если паттерн - это аккорд
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Если паттерн - это нота
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Увеличение смещения каждой ноты
        offset += np.random.uniform(0.5, 0.75)

    # Создание потока с нотами
    midi_stream = stream.Stream(output_notes)

    # Сохранение в MIDI файл
    midi_stream.write('midi', fp=filename)

# Генерация музыки и Создание MIDI файла
prediction_output = generate_music(model, network_input, n_vocab, int_to_note)
create_midi(prediction_output)
```
