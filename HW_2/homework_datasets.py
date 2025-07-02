import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import os
from tqdm import tqdm

# Устанавливаем seed для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

# Определяем устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

class CustomCSVDataset(Dataset):
    """
    Кастомный класс датасета PyTorch для работы с CSV файлами.

    Основные возможности:
    - Загрузка данных из CSV файла.
    - Разделение на признаки (X) и целевую переменную (y).
    - Предобработка:
        - Нормализация числовых столбцов (StandardScaler).
        - One-Hot Encoding категориальных столбцов (OneHotEncoder).
        - Обработка бинарных столбцов (преобразование в 0/1).
        - Простая импутация пропущенных значений (медиана для числовых, мода для категориальных).
    - Поддержка различных форматов целевой переменной (числовая для регрессии, категориальная для классификации).
    - Режим тестирования (без целевой переменной).
    - Возможность передачи обученных препроцессоров для согласованной обработки данных.
    """

    def __init__(self, df, target_column, numerical_cols, categorical_cols,
                 binary_cols=None, drop_cols=None, test_mode=False,
                 target_is_categorical=False,
                 preprocessor=None, target_encoder=None):
        """
        Инициализация датасета.

        Args:
            df (pd.DataFrame): DataFrame с данными.
            target_column (str): Название целевого столбца.
            numerical_cols (list): Список названий числовых столбцов для нормализации.
            categorical_cols (list): Список названий категориальных столбцов для One-Hot Encoding.
            binary_cols (list, optional): Список названий бинарных столбцов (0/1, True/False, 'Yes'/'No'). По умолчанию None.
            drop_cols (list, optional): Список названий столбцов, которые нужно полностью удалить. По умолчанию None.
            test_mode (bool, optional): Если True, датасет будет использоваться без целевой переменной. По умолчанию False.
            target_is_categorical (bool, optional): Если True, целевая переменная будет обработана как категориальная (LabelEncoder). По умолчанию False.
            preprocessor (sklearn.compose.ColumnTransformer, optional): Предварительно обученный препроцессор для признаков (X). Если None, будет обучен новый. По умолчанию None.
            target_encoder (sklearn.preprocessing.LabelEncoder, optional): Предварительно обученный энкодер для целевой переменной (y). Если None и target_is_categorical=True, будет обучен новый. По умолчанию None.
        """
        self.df = df.copy()
        self.target_column = target_column
        self.numerical_cols = numerical_cols if numerical_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.binary_cols = binary_cols if binary_cols is not None else []
        self.drop_cols = drop_cols if drop_cols is not None else []
        self.test_mode = test_mode
        self.target_is_categorical = target_is_categorical
        self.target_encoder = None

        # 1. Удаление ненужных столбцов
        self.df.drop(columns=self.drop_cols, errors='ignore', inplace=True)

        # 2. Разделение на признаки (X) и целевую переменную (y)
        self.targets = None
        if self.target_column in self.df.columns:
            self.targets = self.df[self.target_column]
            self.features_df = self.df.drop(columns=[self.target_column])
        elif not self.test_mode:
            raise ValueError(f"Целевой столбец '{self.target_column}' не найден в данных и test_mode=False.")
        else:
            self.features_df = self.df  # В тестовом режиме целевого столбца может не быть

        # 3. Обработка бинарных столбцов
        for col in self.binary_cols:
            if col in self.features_df.columns:
                # Попытка преобразовать в числовой тип (0 или 1)
                try:
                    self.features_df[col] = pd.to_numeric(self.features_df[col], errors='coerce').astype(int)
                except ValueError:
                    print(
                        f"Предупреждение: Бинарный столбец '{col}' не удалось преобразовать напрямую в 0/1. Проверьте формат.")

        # 4. Простая импутация пропущенных значений
        for col in self.numerical_cols:
            if col in self.features_df.columns and self.features_df[col].isnull().any():
                median_val = self.features_df[col].median()
                self.features_df[col].fillna(median_val, inplace=True)
        for col in self.categorical_cols:
            if col in self.features_df.columns and self.features_df[col].isnull().any():
                mode_val = self.features_df[col].mode()[0]
                self.features_df[col].fillna(mode_val, inplace=True)
        for col in self.binary_cols:
            if col in self.features_df.columns and self.features_df[col].isnull().any():
                self.features_df[col].fillna(0, inplace=True)  # Заполняем нулями

        # 5. Предобработка признаков (X) с использованием ColumnTransformer
        if preprocessor is None:
            transformers = []
            if self.numerical_cols:
                transformers.append(('num', StandardScaler(), self.numerical_cols))
            if self.categorical_cols:
                transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols))
            self.preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
            self.features = self.preprocessor.fit_transform(self.features_df)
        else:
            self.preprocessor = preprocessor
            self.features = self.preprocessor.transform(self.features_df)

        self.features = torch.tensor(self.features, dtype=torch.float32)

        # 6. Обработка целевой переменной (y)
        if not self.test_mode:
            if self.target_is_categorical:
                if target_encoder is None:
                    self.target_encoder = LabelEncoder()
                    self.targets = self.target_encoder.fit_transform(self.targets)
                else:
                    self.target_encoder = target_encoder
                    self.targets = self.target_encoder.transform(self.targets)
                self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1)
            else:
                self.targets = torch.tensor(self.targets.values, dtype=torch.float32).unsqueeze(1)
        else:
            self.targets = None
            self.target_encoder = target_encoder


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.features[idx]
        else:
            return self.features[idx], self.targets[idx]

# Функции для загрузки датасетов
def load_auto_mpg_data(file_path='data/auto-mpg.data', url='http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'):
    if not os.path.exists(file_path):
        print(f"Загрузка Auto MPG данных с {url}...")
        df = pd.read_csv(url, delim_whitespace=True, header=None, na_values='?')
        df.to_csv(file_path, index=False, header=False) # Сохраняем локально
    else:
        print(f"Загрузка Auto MPG данных из локального файла: {file_path}")
        df = pd.read_csv(file_path, delim_whitespace=True, header=None, na_values='?')
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                    'acceleration', 'model_year', 'origin', 'car_name']
    df.columns = column_names

    # 'car_name' очень уникален и не подходит для OHE, 'origin' - категориальный
    # 'horsepower' содержит '?', преобразовать в число и обработать NaN
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

    return df

def load_pima_diabetes_data(file_path='data/diabetes.csv'):
    print(f"Загрузка Pima Indians Diabetes данных из локального файла: {file_path}")
    df = pd.read_csv(file_path)

    # В этом датасете 0 в некоторых столбцах означает пропущенное значение.

    return df

# Модели PyTorch
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # BCEWithLogitsLoss уже включает сигмоиду, поэтому здесь ее не применяем.
        # Если бы использовали BCELoss, то нужно было бы: return torch.sigmoid(self.linear(x))
        return self.linear(x)

# Функция обучения
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, task_type):
    print(f"\n--- Начало обучения для {task_type.upper()} ---")
    model.train() # Переводим модель в режим обучения
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs} (Обучение)")
        for i, (inputs, labels) in enumerate(train_loader_tqdm):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # Изменяем форму labels для соответствия outputs (если outputs имеет форму (N,1))
            if outputs.shape[-1] == 1 and labels.dim() == 1:
                labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss / (i + 1))

        avg_train_loss = running_loss / len(train_loader)

        # Валидация
        model.eval() # Переводим модель в режим оценки
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{num_epochs} (Валидация)")
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # Изменяем форму labels для соответствия outputs (если outputs имеет форму (N,1))
                if outputs.shape[-1] == 1 and labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                if task_type == 'classification':
                    # Для BCEWithLogitsLoss, применяем сигмоиду и порог 0.5 для получения предсказаний класса
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                else: # regression
                    preds = outputs

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_loader_tqdm.set_postfix(val_loss=val_loss / (len(val_loader_tqdm)))

        avg_val_loss = val_loss / len(val_loader)

        # Вывод метрик в зависимости от задачи
        metrics_str = ""
        if task_type == 'regression':
            rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
            metrics_str = f"RMSE: {rmse:.4f}"
        elif task_type == 'classification':
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            metrics_str = f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}"

        print(f"Эпоха {epoch+1}/{num_epochs}, "
              f"Обучение Loss: {avg_train_loss:.4f}, "
              f"Валидация Loss: {avg_val_loss:.4f}, "
              f"{metrics_str}")

        # Сохранение лучшей модели (по валидационному лоссу)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'models/best_{task_type}_model.pth')
            print(f"  > Сохранена лучшая модель с валидационным лоссом: {best_val_loss:.4f}")

    print(f"--- Обучение для {task_type.upper()} завершено ---")
    return model


# 1. Задача РЕГРЕССИИ: Auto MPG Dataset
print("="*50)
print("НАЧИНАЕМ ЗАДАЧУ РЕГРЕССИИ (Auto MPG)")
print("="*50)

# Загрузка данных
df_mpg = load_auto_mpg_data()

# Определение столбцов
target_col_mpg = 'mpg'
numerical_features_mpg = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
categorical_features_mpg = ['origin']
drop_cols_mpg = ['car_name'] # Удаляем столбец с названиями машин, т.к. он уникален для каждой записи

# Разделение данных
train_df_mpg, test_df_mpg = train_test_split(df_mpg, test_size=0.2, random_state=42)
train_df_mpg, val_df_mpg = train_test_split(train_df_mpg, test_size=0.25, random_state=42) # 0.25 из 0.8 = 0.2 от всего

# Инициализация датасетов
train_dataset_mpg = CustomCSVDataset(
    df=train_df_mpg,
    target_column=target_col_mpg,
    numerical_cols=numerical_features_mpg,
    categorical_cols=categorical_features_mpg,
    drop_cols=drop_cols_mpg,
    target_is_categorical=False
)
# Сохраняем обученные препроцессоры
fitted_preprocessor_mpg = train_dataset_mpg.preprocessor
fitted_target_encoder_mpg = train_dataset_mpg.target_encoder # Будет None для регрессии

val_dataset_mpg = CustomCSVDataset(
    df=val_df_mpg,
    target_column=target_col_mpg,
    numerical_cols=numerical_features_mpg,
    categorical_cols=categorical_features_mpg,
    drop_cols=drop_cols_mpg,
    target_is_categorical=False,
    preprocessor=fitted_preprocessor_mpg, # Используем обученный
    target_encoder=fitted_target_encoder_mpg # Используем обученный (None)
)

test_dataset_mpg = CustomCSVDataset(
    df=test_df_mpg,
    target_column=target_col_mpg,
    numerical_cols=numerical_features_mpg,
    categorical_cols=categorical_features_mpg,
    drop_cols=drop_cols_mpg,
    target_is_categorical=False,
    preprocessor=fitted_preprocessor_mpg, # Используем обученный
    target_encoder=fitted_target_encoder_mpg # Используем обученный (None)
)

# DataLoader'ы
batch_size = 32
train_loader_mpg = DataLoader(train_dataset_mpg, batch_size=batch_size, shuffle=True)
val_loader_mpg = DataLoader(val_dataset_mpg, batch_size=batch_size, shuffle=False)
test_loader_mpg = DataLoader(test_dataset_mpg, batch_size=batch_size, shuffle=False)

# Определяем размерность входных признаков
input_dim_mpg = train_dataset_mpg.features.shape[1]
print(f"Размерность входных признаков для регрессии: {input_dim_mpg}")

# Модель, функция потерь, оптимизатор
model_mpg = LinearRegressionModel(input_dim_mpg).to(device)
criterion_mpg = nn.MSELoss()
optimizer_mpg = optim.Adam(model_mpg.parameters(), lr=0.01)
num_epochs_mpg = 100

# Обучение модели
trained_model_mpg = train_model(model_mpg, train_loader_mpg, val_loader_mpg, criterion_mpg, optimizer_mpg, num_epochs_mpg, 'regression')

# Оценка на тестовой выборке
print("\nОценка модели регрессии на тестовой выборке:")
trained_model_mpg.eval()
test_preds_mpg = []
test_labels_mpg = []
with torch.no_grad():
    for inputs, labels in test_loader_mpg:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = trained_model_mpg(inputs)
        test_preds_mpg.extend(outputs.cpu().numpy())
        test_labels_mpg.extend(labels.cpu().numpy())

rmse_test_mpg = np.sqrt(mean_squared_error(test_labels_mpg, test_preds_mpg))
print(f"RMSE на тестовой выборке: {rmse_test_mpg:.4f}")


# --- 2. Задача БИНАРНОЙ КЛАССИФИКАЦИИ: Pima Indians Diabetes Dataset ---
print("\n\n"+"="*50)
print("НАЧИНАЕМ ЗАДАЧУ БИНАРНОЙ КЛАССИФИКАЦИИ (Pima Indians Diabetes)")
print("="*50)

# Загрузка данных
df_diabetes = load_pima_diabetes_data()

# Определение столбцов
target_col_diabetes = 'Outcome' # 0 или 1
# Все остальные - числовые признаки
numerical_features_diabetes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_features_diabetes = [] # Нет явных категориальных
binary_cols_diabetes = [] # Целевой столбец уже числовой 0/1, не требует обработки как бинарный признак
drop_cols_diabetes = []

# Разделение данных
train_df_diabetes, test_df_diabetes = train_test_split(df_diabetes, test_size=0.2, random_state=42)
train_df_diabetes, val_df_diabetes = train_test_split(train_df_diabetes, test_size=0.25, random_state=42) # 0.25 из 0.8 = 0.2 от всего

# Инициализация датасетов
train_dataset_diabetes = CustomCSVDataset(
    df=train_df_diabetes,
    target_column=target_col_diabetes,
    numerical_cols=numerical_features_diabetes,
    categorical_cols=categorical_features_diabetes,
    drop_cols=drop_cols_diabetes,
    target_is_categorical=True # Для классификации
)
# Сохраняем обученные препроцессоры
fitted_preprocessor_diabetes = train_dataset_diabetes.preprocessor
fitted_target_encoder_diabetes = train_dataset_diabetes.target_encoder # Будет LabelEncoder

val_dataset_diabetes = CustomCSVDataset(
    df=val_df_diabetes,
    target_column=target_col_diabetes,
    numerical_cols=numerical_features_diabetes,
    categorical_cols=categorical_features_diabetes,
    drop_cols=drop_cols_diabetes,
    target_is_categorical=True,
    preprocessor=fitted_preprocessor_diabetes,
    target_encoder=fitted_target_encoder_diabetes
)

test_dataset_diabetes = CustomCSVDataset(
    df=test_df_diabetes,
    target_column=target_col_diabetes,
    numerical_cols=numerical_features_diabetes,
    categorical_cols=categorical_features_diabetes,
    drop_cols=drop_cols_diabetes,
    target_is_categorical=True,
    preprocessor=fitted_preprocessor_diabetes,
    target_encoder=fitted_target_encoder_diabetes
)

# DataLoader'ы
batch_size = 32
train_loader_diabetes = DataLoader(train_dataset_diabetes, batch_size=batch_size, shuffle=True)
val_loader_diabetes = DataLoader(val_dataset_diabetes, batch_size=batch_size, shuffle=False)
test_loader_diabetes = DataLoader(test_dataset_diabetes, batch_size=batch_size, shuffle=False)

# Определяем размерность входных признаков
input_dim_diabetes = train_dataset_diabetes.features.shape[1]
print(f"Размерность входных признаков для классификации: {input_dim_diabetes}")

# Модель, функция потерь, оптимизатор
model_diabetes = LogisticRegressionModel(input_dim_diabetes).to(device)
# BCEWithLogitsLoss объединяет Sigmoid и Binary Cross Entropy для большей числовой стабильности
criterion_diabetes = nn.BCEWithLogitsLoss()
optimizer_diabetes = optim.Adam(model_diabetes.parameters(), lr=0.01)
num_epochs_diabetes = 100

# Обучение модели
trained_model_diabetes = train_model(model_diabetes, train_loader_diabetes, val_loader_diabetes, criterion_diabetes, optimizer_diabetes, num_epochs_diabetes, 'classification')

# Оценка на тестовой выборке
print("\nОценка модели классификации на тестовой выборке:")
trained_model_diabetes.eval()
test_preds_diabetes = []
test_labels_diabetes = []
with torch.no_grad():
    for inputs, labels in test_loader_diabetes:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = trained_model_diabetes(inputs)
        # Применяем сигмоиду и порог 0.5 для получения предсказаний класса
        preds = (torch.sigmoid(outputs) > 0.5).float()
        test_preds_diabetes.extend(preds.cpu().numpy())
        test_labels_diabetes.extend(labels.cpu().numpy())

accuracy_test_diabetes = accuracy_score(test_labels_diabetes, test_preds_diabetes)
f1_test_diabetes = f1_score(test_labels_diabetes, test_preds_diabetes)

print(f"Accuracy на тестовой выборке: {accuracy_test_diabetes:.4f}")
print(f"F1-Score на тестовой выборке: {f1_test_diabetes:.4f}")


