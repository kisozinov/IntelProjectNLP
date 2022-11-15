# Алгоритмы анализа тональности текста при помощи графовых сетевых структур

Алгоритм графовых сверточных сетей (GCN) является актуальной темой, которая успешно применяется для обнаружения наркотиков, выявления мошенничества, кластеризации и анализа социальных сетей.

В рамках этого проекта нашей мы использовали данную технологию для анализа настроения текста, а также провели сравнительный анализ собственных и уже реализованных алгоритмов, выполняющих классификацию бинарную классификацию.

- **Данные**: Large Movie Review Dataset; Amazon Reviews for Sentiment Analysis.
- **Методология**: Использование графов и моделей глубокого обучения, работающих с ними.
- **Технологии**: Python3, фреймворки nltk, networkx, tensorflow, spektral, genism, eel.
- **Результат**: Реализация алгоритма обработки текстов и последующей трансформации в граф, графовой сверточной сети (GCN), а именно ее обучение, тестирование и инференс, а также пользовательский веб-интерфейс.
