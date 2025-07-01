import numpy as np
import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)
f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z
f.backward()
print(f"df/dx = {x.grad}")
print(f"df/dy = {y.grad}")
print(f"df/dz = {z.grad}")


# Градиент функции потерь

def mse_gradients(x, y_true, w, b):
    n = len(x)
    y_pred = w * x + b
    mse = np.mean((y_pred - y_true) ** 2)
    grad_w = (2 / n) * np.sum((y_pred - y_true) * x)
    grad_b = (2 / n) * np.sum(y_pred - y_true)

    return mse, grad_w, grad_b

# Цепное правило

x = torch.tensor(2.0, requires_grad=True)
f = torch.sin(x**2 + 1)
f.backward()
print(f"df/dx = {x.grad}")
grad_check, = torch.autograd.grad(torch.sin(x**2 + 1), x)