import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
import pathlib

@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath: str = "banknotes.csv"
    test_size: float = 0.25
    random_state: int = 6
    n_estimators: int = 100
    learning_rate: float = 1.0
    max_depth: int = 1

hp = Hyperparameters()

# Collecting and preparing data
def create_dataframe(filepath):
    return pd.read_csv((pathlib.Path(__file__).parent / filepath).resolve())


# Splitting train and test dataset
def split_dataset(df, test_size, random_state):
    X = df.drop(["class"], axis=1)
    y = df["class"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Building and fitting the model
def train_model(X_train, y_train, n_estimators, learning_rate, max_depth, random_state):
    model = GradientBoostingClassifier(n_estimators = n_estimators,
                                     learning_rate= learning_rate,
                                     max_depth= max_depth,
                                     random_state= random_state)
    return model.fit(X_train, y_train)


# Running the workflow
def run_wf(hp: Hyperparameters) -> GradientBoostingClassifier:
    df = create_dataframe(filepath=hp.filepath)
    X_train, X_test, y_train, y_test = split_dataset(df=df, test_size=hp.test_size, random_state=hp.random_state)
    return train_model(X_train=X_train,
                y_train=y_train, 
                n_estimators=hp.n_estimators,
                learning_rate=hp.learning_rate,
                max_depth=hp.max_depth,
                random_state=hp.random_state)
    

if __name__=="__main__":
    run_wf(hp=hp)