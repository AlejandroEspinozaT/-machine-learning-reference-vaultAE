from rich.console import Console
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class MpgRegressionCapstone:
    """
    A capstone project for MPG regression using the mpg dataset and PyTorch.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.dataset = sns.load_dataset("mpg")
        self.console = Console()
        self.learning_rate = learning_rate
        self.epochs = epochs

    def run(self) -> None:
        """Main pipeline for executing the capstone."""
        self.console.print("[bold blue]Starting MPG Regression Capstone with PyTorch...[/bold blue]")

        self.explore_data()

        self.dataset = self.preprocess_data()

        self.train_and_evaluate_model()

    def explore_data(self) -> None:
        """Explore and visualize the dataset."""
        self.console.print("[bold yellow]Exploring Dataset[/bold yellow]")
        self.console.print(self.dataset.head())
        self.console.print(self.dataset.info())
        self.console.print("[bold green]Summary Statistics:[/bold green]")
        self.console.print(self.dataset.describe())

        numeric_dataset = self.dataset.select_dtypes(include=["number"])

        correlation_matrix = numeric_dataset.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def preprocess_data(self) -> pd.DataFrame:
        """Handle missing data and preprocess the dataset."""
        self.console.print("[bold yellow]Preprocessing Data...[/bold yellow]")

        cleaned_dataset = self.dataset.dropna()

        if "origin" in cleaned_dataset.columns:
            cleaned_dataset = pd.get_dummies(cleaned_dataset, columns=["origin"], drop_first=True)

        self.console.print("[bold green]Preprocessing Complete![/bold green]")
        return cleaned_dataset

    def train_and_evaluate_model(self) -> None:
        """Train a regression model (PyTorch) and evaluate its performance."""
        self.console.print("[bold yellow]Training and Evaluating Model with PyTorch...[/bold yellow]")

        X = self.dataset.drop(columns=["mpg", "name"], errors="ignore")
        y = self.dataset["mpg"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        num_features = X_train_t.shape[1]
        model = nn.Linear(num_features, 1)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)

        train_losses = []
        model.train()
        for epoch in range(self.epochs):
            predictions = model(X_train_t)
            loss = criterion(predictions, y_train_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (epoch + 1) % 100 == 0:
                self.console.print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            y_pred_t = model(X_test_t)
        y_pred = y_pred_t.numpy().flatten()

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        self.console.print(f"[bold green]Mean Absolute Error (MAE): {mae:.2f}[/bold green]")
        self.console.print(f"[bold green]Root Mean Squared Error (RMSE): {rmse:.2f}[/bold green]")
        self.console.print(f"[bold green]R^2 Score: {r2:.2f}[/bold green]")

        plt.figure(figsize=(8,6))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, alpha=0.7, label='Predictions')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
        plt.xlabel("Actual MPG")
        plt.ylabel("Predicted MPG")
        plt.title("Actual vs Predicted MPG")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    mpg_capstone = MpgRegressionCapstone(learning_rate=0.01, epochs=1000)
    mpg_capstone.run()
