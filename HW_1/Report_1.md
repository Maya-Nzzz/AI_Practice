# Отчёт по заданию 2 «Автоматическое дифференцирование»

## 2.1

В коде:
- `x.grad = 14.0`
- `y.grad = 10.0`
- `z.grad = 10.0`

При аналитических подсчётах получились те же результаты:
- `df/dx = 14.0`
- `df/dy = 10.0`
- `df/dz = 10.0`

## 2.3

В коде:  
- `grad = 1.1346487998962402`

При аналитических подсчётах:  
- `cos(2^2 + 1) ⋅ 2 ⋅ 2 = 1.13465`

Результаты совпадают.

---

# Отчёт по заданию 3 «Сравнение производительности CPU vs CUDA»

## 3.4

### Результаты для `tensor1`

| Операция              | CPU (сек) | GPU (сек) | Ускорение |
|-----------------------|-----------|-----------|-----------|
| Мат. умножение        | 1.437303  | 0.051080  | 28.14x    |
| Сложение              | 0.140829  | 0.003251  | 43.32x    |
| Поэлементное умножение| 0.137460  | 0.003247  | 42.33x    |
| Транспонирование      | 0.000029  | 0.000043  | 0.66x     |
| Сумма элементов      | 0.022375  | 0.001096  | 20.42x    |

### Результаты для `tensor3`

| Операция              | CPU (сек) | GPU (сек) | Ускорение |
|-----------------------|-----------|-----------|-----------|
| Мат. умножение        | 0.141173  | 0.003657  | 38.61x    |
| Сложение              | 0.035930  | 0.000858  | 41.87x    |
| Поэлементное умножение| 0.033873  | 0.000854  | 39.67x    |
| Транспонирование      | 0.000024  | 0.000040  | 0.61x     |
| Сумма элементов      | 0.004949  | 0.000344  | 14.40x    |

---

## Выводы

Наибольшее ускорение получили:
- Сложение
- Поэлементное умножение
- Мат. умножение
- Сумма элементов

Эти операции хорошо масштабируются за счёт параллельной обработки большого числа элементов.

Операции могут быть медленнее на GPU (например, транспонирование), так как зависят от доступа к памяти и её последовательности.  

Для `tensor1` ускорение оказалось выше. Чем больше размер матрицы, тем больше параллелизм и, следовательно, выше эффективность GPU. Маленькие матрицы не полностью загружают CUDA-ядра.

При передаче данных между CPU и GPU происходит:
- Выделение памяти на GPU
- Передача данных через шину PCIe (которая медленнее вычислений на GPU)
- Синхронизация потоков

Чтобы получить максимальное ускорение, минимизируют количество передач и выполняют как можно больше операций прямо на GPU.
