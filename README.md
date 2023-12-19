# geonames_project

# Обзор
GeoSearcher - это инструмент на Python для поиска геолокаций. Он позволяет пользователям выполнять запросы и находить наиболее подходящее наименования городов. 

# Особенности
* Подключение к базе данных: Поддерживает источники данных SQL и CSV.
* Анализ географических данных: Предоставляет функционал для запросов и анализа географических данных.
* Обработка текста: Использует SentenceTransformer для генерации векторных представлений текста.
* Перевод языка: Интегрирует Google Translator для перевода текста.
* Проверка орфографии: Реализует проверку орфографии для переведенного текста.

# Требования
Python 3.x
* pandas
* numpy
* googletrans
* spellchecker
* sentence_transformers
* torch

# Установка

```python
pip install -r requirements.txt
```

# Пример использования


```python
from geosearcher import GeoSearcher

# Инициализация поисковика
searcher = GeoSearcher('your_database_connection', model='LaBSE', mode='sql')

# Поиск названия города
results = searcher.match_name(["Нью-Йорк"], number_of_matching=3)

```

В качестве результата возвращается словарь, который содержит в себе следующие ключи: `name`,`region`,`country`,`similarity`