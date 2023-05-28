from flytekit import Resources, task

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from sidetrek.dataset import load_dataset
from sidetrek.types.dataset import SidetrekDataset
from typing import Tuple

@dataclass_json
@dataclass
class Hyperparameters(object):
    test_size: float = 0.25
    random_state: int = 6
    n_estimators: int = 100
    learning_rate: float = 1.0
    max_depth: int = 1

hp = Hyperparameters()

# Collecting and preparing data
@task(requests=Resources(cpu="2",mem="1Gi"),limits=Resources(cpu="2",mem="1Gi"),retries=3)
def create_dataframe(ds: SidetrekDataset) -> pd.DataFrame:
    data = load_dataset(ds=ds, data_type="csv")
    cols = list(data)[0]
    data_dict = {}
    for k,v in enumerate(data):
        if k>0:
            data_dict[k] = v
    df = pd.DataFrame.from_dict(data_dict, columns=cols, orient="index")
    df["variace"] = df["variace"].astype('float')
    df["skewness"] = df["skewness"].astype('float')
    df["curtosis"] = df["curtosis"].astype('float')
    df["entropy"] = df["entropy"].astype('float')
    df["class"] = df["class"].astype('int')
    return df


# Splitting train and test dataset
@task(requests=Resources(cpu="2",mem="1Gi"),limits=Resources(cpu="2",mem="1Gi"),retries=3)
def split_dataset(df: pd.DataFrame, hp: Hyperparameters) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(["class"], axis=1)
    y = df["class"]
    return train_test_split(X, y, test_size=hp.test_size, random_state=hp.random_state)


# Building and fitting the model
@task(requests=Resources(cpu="2",mem="1Gi"),limits=Resources(cpu="2",mem="1Gi"),retries=3)
def train_model(X_train: pd.DataFrame, y_train: pd.Series, hp: Hyperparameters) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(n_estimators = hp.n_estimators,
                                     learning_rate= hp.learning_rate,
                                     max_depth= hp.max_depth,
                                     random_state= hp.random_state)
    return model.fit(X_train, y_train)
