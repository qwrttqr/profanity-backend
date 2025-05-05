# Сервис для анализа входящих почтовых сообщений на предмет токсичности, наличия нецензурной брани

## Требования
1. Python версии 3.10.11
2. Зависимости (устанавлиется с `pip install -r requierements.txt`)
## Запуск
Запуск из `main.py` файла
`fastapi dev src/main.py`

## Использование
Обращение к сервису через `query` параметры по ручке `/analyze`
К примеру
`https://domainexample.com/analyze?text=your text here&threshold(float between 0 and 1)`

## Сборка проекта:
```python
print('ergerg')
```