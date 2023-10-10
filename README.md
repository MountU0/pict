# Prediction of Chemical Reaction Yields with Enhanced Reaction Representations
#### In the last decade, machine learning (ML) proved itself as a powerful tool for organic synthesis planning. One of the key research areas therein is the prediction of chemical reaction yield, a key parameter for selecting reaction conditions and evaluating success of a synthesis. Recently, several ML models have been reported to predict reaction yields based on high-throughput experiment datasets. However, due to sparse and insufficient data as well as limited capacity of reaction representations, the performance and applicability of such ML models remain limited. In this work, we apply ML methods to predict reaction yields based on molecular fingerprints as state-of-the-art reaction representations using the publicly available USPTO organic reactions dataset.

Структура репозитория:

    -/pict
        /data processing - Обработка данных
            data.csv - исходные данные
            preprocessing.ipynb - предобработка данных

        /model - Часть с ML (тест модели на данных)
            data_new.csv - преобработанный датасет, который используется в baseline.ipynb
            baseline.ipynb - Тест xgboost

        /images - Сохраненные графики
