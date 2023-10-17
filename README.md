
# Prediction of Chemical Reaction Yields with Enhanced Reaction Representations

<p align="center">
  <img src= https://i.postimg.cc/QNcJ8c3r/Screenshot.pngalt="Логотип проекта"/>
</p>

## Сontent

1. [Overview](#overview)
2. [Data Processing](#processing)
3. [Usage](#usage)
4. [Contribute](#contribute)
5. [Repository structure](#structure)
6. [Listing](#Listing)
7. [Useful links](#links)
## <a name="overview"></a>Project overview

*In the last decade, machine learning (ML) proved itself as a powerful tool for organic synthesis planning. One of the key research areas therein is the prediction of chemical reaction yield, a key parameter for selecting reaction conditions and evaluating success of a synthesis. Recently, several ML models have been reported to predict reaction yields based on high-throughput experiment datasets. However, due to sparse and insufficient data as well as limited capacity of reaction representations, the performance and applicability of such ML models remain limited. In this work, we apply ML methods to predict reaction yields based on molecular fingerprints as state-of-the-art reaction representations using the publicly available USPTO organic* reactions dataset.   

## Data processing

The chemical reaction database in SMILES format was standardized: Reactant1.Reactant2...ReactantN>>Product1.Product2.ProductN. The strings were grouped based on the number of reactants. The most abundant groups, including one and two reactants, were selected for analysis. Morgan molecular fingerprints and drfp were then obtained for each reaction. Fingerprints represent the encoding of a molecule using a binary vector, where each value indicates the presence or absence of a specific substructure.

## Repository structure

    -/pict
        /data processing - Обработка данных
            data.csv - исходные данные
            preprocessing.ipynb - предобработка данных

        /model - Часть с ML (тест модели на данных)
            data_new.csv - преобработанный датасет, который используется в baseline.ipynb
            baseline.ipynb - Тест xgboost

        /images - Сохраненные графики

## Listing
```python
def find_nth(haystack:str, needle:str, n:int) -> str:
    """
    usage:
    haystack - строка в которой ищем вхождение
    needle - что ищем
    n - на какой позиции
    //если на n позиции не найден элемент -> -1
    """
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def smiles_parse(smiles:str) -> str:
    """
    Парсим smiles:
    "CC(C)(C)NNC(C)(C#N)C1CC1.O=C1CCCCCC1>CC(=O)C1CC1>CC(C)(C)NNC1(C#N)CCCCCC1" ->
    "CC(C)(C)NNC(C)(C#N)C1CC1.O=C1CCCCCC1>>CC(C)(C)NNC1(C#N)CCCCCC1"
    """
    return smiles[:find_nth(smiles, ">", 1)] +">"+ smiles[find_nth(smiles, ">", 2):]

def task2_split(task2:str) -> str:
    """
    TASK 5:
    5) Разделить реагенты и продукты на столбики, т.е. у вас добавятся столбцы "reactant_1", "reactant_2", "product"
    """
    result = []
    product = task2[find_nth(task2, ">",2)+1:]

    if find_nth(task2, ".", 1) != -1:
        result.append(task2[:find_nth(task2, ".", 1)])
        result.append(task2[find_nth(task2, ".", 1) + 1:find_nth(task2, ">", 1)])
    else:
        result.append(task2[:find_nth(task2, ">", 1)])
        result.append(None)

    result.append(product)
    return result
    


#TASK 1
data = pd.read_csv("data.csv",  sep="\t")
data.drop(['Unnamed: 0', 'myID', 'Source', 'Target', 'OriginalReaction'], axis=1, inplace=True)
print("Размер всего датасета:", len(data))

#TASK 2
data["task2"] =  data["CanonicalizedReaction"].apply(smiles_parse)

#TASK 3
data["reactants"] = data["task2"].apply(lambda x: x[:x.find(">")])#реагенты
data["task3"] = data["reactants"].apply(lambda x: str.count(x, "."))
print(collections.Counter(data["task3"]))

#TASK 4
print("Больше всего реакций с 1 и 2 реагентами. Оставляем их")
data = data[data["task3"]<2]

# #TASK 5
data["reactant_1"], data["reactant_2"], data["product"] = zip(*data["task2"].apply(task2_split))
display(data)
```
## **Useful links**
1. ***https://www.youtube.com/watch?v=Lj4iPb7Kw_0&t=19s***

2. ***https://www.researchgate.net/publication/370242541_Learning_Hierarchical_Representations_for_Explainable_Chemical_Reaction_Prediction***

3. ***https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00579-z***
