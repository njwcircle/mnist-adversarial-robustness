import time

import torch
from torch import nn, optim
import tqdm

from train import CNN, get_loaders, evaluate, device
from attack import pgd

# PGD adversarial training (Madry et al.)

TRAIN_EPS = 0.2
TRAIN_STEPS = 10
EPOCHS = 10


def adv_train_one_epoch(model, loader, optimizer, criterion, eps, steps):
    total_loss = 0.0
    for X, y in tqdm.tqdm(loader):
        X, y = X.to(device), y.to(device)

        model.eval()   # 공격을 만드는 동안은 dropout을 끈다
        X_adv = pgd(model, X, y, eps, steps=steps)

        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_adv), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def quick_robust_acc(model, loader, eps, n_batches=10):
    model.eval()
    correct, total = 0, 0
    for i, (X, y) in enumerate(loader):
        if i >= n_batches:
            break
        X, y = X.to(device), y.to(device)
        X_adv = pgd(model, X, y, eps, steps=20)
        with torch.no_grad():
            correct += (model(X_adv).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    train_loader, test_loader = get_loaders()
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        t0 = time.time()
        loss = adv_train_one_epoch(model, train_loader, optimizer, criterion, TRAIN_EPS, TRAIN_STEPS)
        clean = evaluate(model, test_loader)
        robust = quick_robust_acc(model, test_loader, TRAIN_EPS)
        print(f"epoch {epoch + 1}: loss {loss:.4f}, clean acc {clean:.4f}, "
              f"PGD(eps={TRAIN_EPS}) acc {robust:.4f}, {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), "mnist_cnn_adv.pt")


if __name__ == "__main__":
    main()
