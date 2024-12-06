import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from rich.console import Console


class MpgRegressionCapstone:
    """
    A capstone project for MPG regression using the mpg dataset.
    https://www.kaggle.com/code/devanshbesain/exploration-and-analysis-auto-mpg
    """

    def __init__(self, data_path: str = "data/auto-mpg.data") -> None:
        self.data_path = data_path
        self.dataset = None
        self.console = Console()

    def _load_data(self) -> None:
        column_names = [
            "mpg", "cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model_year", "origin", "car_name"
        ]
        self.dataset = pd.read_csv(
            self.data_path, delim_whitespace=True, names=column_names, na_values="?"
        )

    def preprocess(self) -> None:
        self.console.print("Preprocessing")
        self.dataset.dropna(inplace=True)
        self.dataset.drop(columns=["car_name"], inplace=True)
        self.dataset["origin"] = self.dataset["origin"].astype("category")

    def train_model(self) -> None:
        self.console.print("Training regression model...")
        X = self.dataset.drop(columns=["mpg"])
        X = pd.get_dummies(X, drop_first=True)
        y = self.dataset["mpg"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.console.print(f"Msq Error: {mse}")
        self.console.print(f"Rsquared: {r2}")

        plt.scatter(y_test, y_pred)
        plt.xlabel("True MPG")
        plt.ylabel("Predicted MPG")
        plt.title("True/ Predicted MPG")
        plt.show()

    def run(self) -> None:
        self._load_data()
        self.console.print(self.dataset.head())
        self.preprocess()
        self.train_model()
