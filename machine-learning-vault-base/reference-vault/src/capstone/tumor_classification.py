import torch
import torch.nn as nn
import torch.optim as optim
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from rich.console import Console

class TumorClassificationCapstone:
    """
    Capstone para clasificación binaria utilizando PyTorch y el dataset
    Breast Cancer Wisconsin Diagnostic, con gráficos de pérdida y matriz de confusión.
    """

    def __init__(self, learning_rate=0.01, epochs=1000, test_size=0.2) -> None:
        self.console = Console()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_size = test_size

        data = fetch_ucirepo(id=17)
        X = data.data.features
        y = (data.data.targets == 'M').astype(int)  # 1 maligno, 0 benigno

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        num_features = self.X_train_tensor.shape[1]
        self.model = nn.Linear(num_features, 1)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.train_losses = []

    def train(self):
        """
        regresión logística PyTorch.
        """
        self.model.train()
        for epoch in range(self.epochs):
            logits = self.model(self.X_train_tensor)
            loss = self.criterion(logits, self.y_train_tensor)
            self.train_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                self.console.print(f"[cyan]Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}[/cyan]")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
        return preds

    def accuracy(self, y_true, y_pred):
        correct = (y_true == y_pred).sum().item()
        return correct / len(y_true)

    def plot_loss(self):
        """
        Grafico
        """
        plt.figure(figsize=(8,6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Evolución de la Pérdida durante el Entrenamiento')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('M de Cofusion')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['Benigno (0)', 'Maligno (1)'], rotation=45)
        plt.yticks(tick_marks, ['Benigno (0)', 'Maligno (1)'])

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.show()

    def run(self):
        self.train()

        y_pred = self.predict(self.X_test_tensor)
        acc = self.accuracy(self.y_test_tensor, y_pred)

        self.console.print(f"[bold green]Accuracy: {acc * 100:.2f}%[/bold green]")
        self.plot_loss()

        y_true_np = self.y_test_tensor.numpy().astype(int).flatten()
        y_pred_np = y_pred.numpy().astype(int).flatten()
        self.plot_confusion_matrix(y_true_np, y_pred_np)


if __name__ == "__main__":
    capstone = TumorClassificationCapstone()
    capstone.run()
