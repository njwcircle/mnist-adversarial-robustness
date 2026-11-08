import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

device = "cpu"


def get_loaders(batch_size=64):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (N,1,28,28) -> conv3 -> (N,6,26,26) -> pool -> (N,6,13,13)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # -> conv5 -> (N,16,9,9) -> pool -> (N,16,4,4)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc3 = nn.Linear(16 * 4 * 4, 120)
        self.fc4 = nn.Linear(120, 80)
        self.fc5 = nn.Linear(80, 10)
        self.dropout = nn.Dropout(p=0.5)

        for layer in (self.fc3, self.fc4, self.fc5):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.flatten(1)  # (N,16,4,4) -> (N,256)
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X, y in tqdm.tqdm(loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)


def main():
    train_loader, test_loader = get_loaders()
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        acc = evaluate(model, test_loader)
        print(f"epoch {epoch + 1}: loss {loss:.4f}, test acc {acc:.4f}")

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
