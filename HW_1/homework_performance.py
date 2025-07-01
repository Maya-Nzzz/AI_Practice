import torch
import time
import pandas as pd

tensor1 = torch.randint(1, 100, (64, 1024, 1024), dtype=torch.float32)
tensor2 = torch.randint(1, 100, (128, 512, 512), dtype=torch.float32)
tensor3 = torch.randint(1, 100, (256, 256, 256), dtype=torch.float32)


def measure_time(func, device='cpu', *args, **kwargs):
    """
    Измеряет время выполнения функции func на CPU или GPU.

    Аргументы:
    - func: функция, которую нужно замерить
    - device: 'cpu' или 'cuda'
    - *args, **kwargs: аргументы для func
    """
    if device == 'cuda' and torch.cuda.is_available():
        args = [arg.to('cuda') if isinstance(arg, torch.Tensor) else arg for arg in args]
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return elapsed_time_ms / 1000
    else:
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time


operations = [
    ("Мат. умножение", torch.matmul, tensor1, tensor1.transpose(1, 2)),
    ("Сложение", torch.add, tensor2, tensor2),
    ("Поэлементное умножение", torch.mul, tensor2, tensor2),
    ("Транспонирование", torch.transpose, tensor3, 1, 2),
    ("Сумма элементов", torch.sum, tensor3),
]

results = []

for name, op, *tensors in operations:
    time_cpu = measure_time(op, 'cpu', *tensors)
    if torch.cuda.is_available():
        time_gpu = measure_time(op, 'cuda', *tensors)
        speedup = time_cpu / time_gpu if time_gpu > 0 else float('inf')
    else:
        time_gpu = None
        speedup = None

    results.append({
        "Операция": name,
        "CPU (сек)": f"{time_cpu:.6f}",
        "GPU (сек)": f"{time_gpu:.6f}" if time_gpu else "—",
        "Ускорение": f"{speedup:.2f}x" if speedup else "—"
    })

df = pd.DataFrame(results)
print(df)
