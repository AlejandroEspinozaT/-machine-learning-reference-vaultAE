import pandas as pd
from sklearn.datasets import load_breast_cancer  # type: ignore
from rich.console import Console
import numpy as np

class TumorClassificationCapstone:
    """
    A capstone project for tumor classification using the breast cancer dataset.
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
    """

    def __init__(self, data_path: str) -> None:
        self.console = Console()
        self.data_path = data_path
        self.dataset = self._load_dataset()
        self.features = None
        self.labels = None

    def _load_dataset(self) -> pd.DataFrame:
        column_names = [
            "ID", "Diagnosis", 
            "Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean", "Smoothness_mean",
            "Compactness_mean", "Concavity_mean", "Concave_points_mean", "Symmetry_mean", "Fractal_dimension_mean",
            "Radius_se", "Texture_se", "Perimeter_se", "Area_se", "Smoothness_se", 
            "Compactness_se", "Concavity_se", "Concave_points_se", "Symmetry_se", "Fractal_dimension_se",
            "Radius_worst", "Texture_worst", "Perimeter_worst", "Area_worst", "Smoothness_worst", 
            "Compactness_worst", "Concavity_worst", "Concave_points_worst", "Symmetry_worst", "Fractal_dimension_worst"
        ]
        df = pd.read_csv(self.data_path, header=None, names=column_names)
        return df

    def preprocess(self) -> None:
        self.dataset.drop(columns=["ID"], inplace=True)
        self.dataset["Diagnosis"] = self.dataset["Diagnosis"].map({"M": 1, "B": 0})
        self.features = self.dataset.drop(columns=["Diagnosis"]).values
        self.labels = self.dataset["Diagnosis"].values

    def feature_scaling(self) -> np.ndarray:
        self.features = (self.features - np.mean(self.features, axis=0)) / np.std(self.features, axis=0)
        return self.features

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        m = len(y)
        h = self.sigmoid(X @ theta)
        return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    def gradient_descent(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int):
        m = len(y)
        for _ in range(num_iters):
            gradient = (1 / m) * (X.T @ (self.sigmoid(X @ theta) - y))
            theta -= alpha * gradient
        return theta

    def train(self, alpha: float = 0.01, num_iters: int = 1000):
        X = np.c_[np.ones(self.features.shape[0]), self.features]
        y = self.labels
        theta = np.zeros(X.shape[1])
        theta = self.gradient_descent(X, y, theta, alpha, num_iters)
        return theta

    def run(self):
        self.console.print("Loading")
        self.preprocess()
        self.feature_scaling()
        self.console.print(f"Dataset: {self.dataset.shape}")
        theta = self.train()
        self.console.print(f"parameters: {theta}")
