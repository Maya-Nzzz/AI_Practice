import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from torch.distributed import optim
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

from HW_2.homework_datasets import val_loader_mpg, LinearRegressionModel, \
    val_loader_diabetes, LogisticRegressionModel, load_auto_mpg_data, \
    CustomCSVDataset, train_model, test_loader_mpg, load_pima_diabetes_data, test_loader_diabetes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Функция для создания расширенных признаков
def create_extended_features(df, numerical_cols, degree=2, interaction_only=False):
    """
    Создает полиномиальные признаки и взаимодействия между признаками

    Args:
        df (pd.DataFrame): Исходный DataFrame
        numerical_cols (list): Список числовых столбцов для преобразования
        degree (int): Степень полиномиальных признаков
        interaction_only (bool): Если True, создает только взаимодействия между признаками

    Returns:
        pd.DataFrame: DataFrame с добавленными признаками
    """
    # Выбираем только числовые признаки
    num_df = df[numerical_cols].copy()

    # Создаем полиномиальные признаки
    poly = PolynomialFeatures(degree=degree,
                              interaction_only=interaction_only,
                              include_bias=False)
    poly_features = poly.fit_transform(num_df)
    feature_names = poly.get_feature_names_out(num_df.columns)

    # Создаем DataFrame с новыми признаками
    poly_df = pd.DataFrame(poly_features, columns=feature_names)

    # Добавляем статистические признаки
    num_df['mean'] = num_df.mean(axis=1)
    num_df['std'] = num_df.std(axis=1)
    num_df['min'] = num_df.min(axis=1)
    num_df['max'] = num_df.max(axis=1)

    # Объединяем все признаки
    extended_df = pd.concat([df, poly_df, num_df[['mean', 'std', 'min', 'max']]], axis=1)

    return extended_df


# Функция для сравнения моделей
def compare_models(base_model, extended_model, X_train, y_train, X_val, y_val, task_type='regression'):
    """
    Сравнивает базовую и расширенную модели

    Args:
        base_model: Модель с базовыми признаками
        extended_model: Модель с расширенными признаками
        X_train, y_train: Обучающие данные
        X_val, y_val: Валидационные данные
        task_type: Тип задачи ('regression' или 'classification')

    Returns:
        dict: Словарь с метриками для обеих моделей
    """
    # Обучаем модели
    base_model.fit(X_train, y_train)
    extended_model.fit(X_train, y_train)

    # Предсказания
    base_preds = base_model.predict(X_val)
    extended_preds = extended_model.predict(X_val)

    # Рассчитываем метрики
    metrics = {}

    if task_type == 'regression':
        metrics['base_rmse'] = np.sqrt(mean_squared_error(y_val, base_preds))
        metrics['extended_rmse'] = np.sqrt(mean_squared_error(y_val, extended_preds))
    else:
        metrics['base_accuracy'] = accuracy_score(y_val, base_preds)
        metrics['extended_accuracy'] = accuracy_score(y_val, extended_preds)
        metrics['base_f1'] = f1_score(y_val, base_preds)
        metrics['extended_f1'] = f1_score(y_val, extended_preds)

    return metrics


# 1. Для задачи РЕГРЕССИИ (Auto MPG)
print("\n" + "=" * 50)
print("СОЗДАНИЕ ПРИЗНАКОВ ДЛЯ РЕГРЕССИИ (Auto MPG)")
print("=" * 50)

# Загрузка данных
df_mpg = load_auto_mpg_data()

# Определение столбцов
target_col_mpg = 'mpg'
numerical_features_mpg = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
categorical_features_mpg = ['origin']
drop_cols_mpg = ['car_name']

# Создаем расширенные признаки
extended_df_mpg = create_extended_features(df_mpg, numerical_features_mpg, degree=2)

# Разделение данных
train_df_mpg, test_df_mpg = train_test_split(extended_df_mpg, test_size=0.2, random_state=42)
train_df_mpg, val_df_mpg = train_test_split(train_df_mpg, test_size=0.25, random_state=42)

# Базовый датасет (без новых признаков)
base_train_dataset_mpg = CustomCSVDataset(
    df=train_df_mpg[df_mpg.columns],  # Используем только исходные столбцы
    target_column=target_col_mpg,
    numerical_cols=numerical_features_mpg,
    categorical_cols=categorical_features_mpg,
    drop_cols=drop_cols_mpg,
    target_is_categorical=False
)

# Расширенный датасет (с новыми признаками)
extended_train_dataset_mpg = CustomCSVDataset(
    df=train_df_mpg,
    target_column=target_col_mpg,
    numerical_cols=[col for col in train_df_mpg.columns
                    if col not in categorical_features_mpg + [target_col_mpg] + drop_cols_mpg],
    categorical_cols=categorical_features_mpg,
    drop_cols=drop_cols_mpg,
    target_is_categorical=False,
    preprocessor=base_train_dataset_mpg.preprocessor  # Используем тот же препроцессор
)

# Создаем DataLoader'ы
batch_size = 32
base_train_loader_mpg = DataLoader(base_train_dataset_mpg, batch_size=batch_size, shuffle=True)
extended_train_loader_mpg = DataLoader(extended_train_dataset_mpg, batch_size=batch_size, shuffle=True)

# Создаем модели
input_dim_base = base_train_dataset_mpg.features.shape[1]
input_dim_extended = extended_train_dataset_mpg.features.shape[1]

base_model_mpg = LinearRegressionModel(input_dim_base).to(device)
extended_model_mpg = LinearRegressionModel(input_dim_extended).to(device)

# Обучаем и сравниваем модели
print("\nОбучение базовой модели...")
train_model(base_model_mpg, base_train_loader_mpg, val_loader_mpg, nn.MSELoss(),
            optim.Adam(base_model_mpg.parameters()), 50, 'regression')

print("\nОбучение расширенной модели...")
train_model(extended_model_mpg, extended_train_loader_mpg, val_loader_mpg, nn.MSELoss(),
            optim.Adam(extended_model_mpg.parameters()), 50, 'regression')

# Оценка на тестовых данных
base_model_mpg.eval()
extended_model_mpg.eval()

base_preds = []
extended_preds = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader_mpg:
        inputs, labels = inputs.to(device), labels.to(device)

        # Базовые предсказания
        base_outputs = base_model_mpg(inputs)
        base_preds.extend(base_outputs.detach().cpu().numpy())

        # Расширенные предсказания (нужно преобразовать тестовые данные аналогично)
        test_features = extended_train_dataset_mpg.preprocessor.transform(
            test_loader_mpg.dataset.features_df)
        test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
        extended_outputs = extended_model_mpg(test_features)
        extended_preds.extend(extended_outputs.detach().cpu().numpy())

        test_labels.extend(labels.detach().cpu().numpy())

# Сравнение метрик
base_rmse = np.sqrt(mean_squared_error(test_labels, base_preds))
extended_rmse = np.sqrt(mean_squared_error(test_labels, extended_preds))

print("\nСравнение моделей регрессии:")
print(f"RMSE базовой модели: {base_rmse:.4f}")
print(f"RMSE расширенной модели: {extended_rmse:.4f}")
print(f"Улучшение: {(base_rmse - extended_rmse):.4f} ({((base_rmse - extended_rmse) / base_rmse * 100):.2f}%)")

# 2. Для задачи КЛАССИФИКАЦИИ (Diabetes)
print("\n" + "=" * 50)
print("СОЗДАНИЕ ПРИЗНАКОВ ДЛЯ КЛАССИФИКАЦИИ (Diabetes)")
print("=" * 50)

# Загрузка данных
df_diabetes = load_pima_diabetes_data()

# Определение столбцов
target_col_diabetes = 'Outcome'
numerical_features_diabetes = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_features_diabetes = []
drop_cols_diabetes = []

# Создаем расширенные признаки
extended_df_diabetes = create_extended_features(df_diabetes, numerical_features_diabetes, degree=2)

# Разделение данных
train_df_diabetes, test_df_diabetes = train_test_split(extended_df_diabetes, test_size=0.2, random_state=42)
train_df_diabetes, val_df_diabetes = train_test_split(train_df_diabetes, test_size=0.25, random_state=42)

# Базовый датасет (без новых признаков)
base_train_dataset_diabetes = CustomCSVDataset(
    df=train_df_diabetes[df_diabetes.columns],  # Используем только исходные столбцы
    target_column=target_col_diabetes,
    numerical_cols=numerical_features_diabetes,
    categorical_cols=categorical_features_diabetes,
    drop_cols=drop_cols_diabetes,
    target_is_categorical=True
)

# Расширенный датасет (с новыми признаками)
extended_train_dataset_diabetes = CustomCSVDataset(
    df=train_df_diabetes,
    target_column=target_col_diabetes,
    numerical_cols=[col for col in train_df_diabetes.columns
                    if col not in categorical_features_diabetes + [target_col_diabetes] + drop_cols_diabetes],
    categorical_cols=categorical_features_diabetes,
    drop_cols=drop_cols_diabetes,
    target_is_categorical=True,
    preprocessor=base_train_dataset_diabetes.preprocessor  # Используем тот же препроцессор
)

# Создаем DataLoader'ы
base_train_loader_diabetes = DataLoader(base_train_dataset_diabetes, batch_size=batch_size, shuffle=True)
extended_train_loader_diabetes = DataLoader(extended_train_dataset_diabetes, batch_size=batch_size, shuffle=True)

# Создаем модели
input_dim_base = base_train_dataset_diabetes.features.shape[1]
input_dim_extended = extended_train_dataset_diabetes.features.shape[1]

base_model_diabetes = LogisticRegressionModel(input_dim_base).to(device)
extended_model_diabetes = LogisticRegressionModel(input_dim_extended).to(device)

# Обучаем и сравниваем модели
print("\nОбучение базовой модели...")
train_model(base_model_diabetes, base_train_loader_diabetes, val_loader_diabetes,
            nn.BCEWithLogitsLoss(), optim.Adam(base_model_diabetes.parameters()), 50, 'classification')

print("\nОбучение расширенной модели...")
train_model(extended_model_diabetes, extended_train_loader_diabetes, val_loader_diabetes,
            nn.BCEWithLogitsLoss(), optim.Adam(extended_model_diabetes.parameters()), 50, 'classification')

# Оценка на тестовых данных
base_model_diabetes.eval()
extended_model_diabetes.eval()

base_preds = []
extended_preds = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_loader_diabetes:
        inputs, labels = inputs.to(device), labels.to(device)

        # Базовые предсказания
        base_outputs = base_model_diabetes(inputs)
        base_preds.extend((torch.sigmoid(base_outputs) > 0.5).float().detach().cpu().numpy())

        # Расширенные предсказания
        test_features = extended_train_dataset_diabetes.preprocessor.transform(
            test_loader_diabetes.dataset.features_df)
        test_features = torch.tensor(test_features, dtype=torch.float32).to(device)
        extended_outputs = extended_model_diabetes(test_features)
        extended_preds.extend((torch.sigmoid(extended_outputs) > 0.5).float().detach().cpu().numpy())

        test_labels.extend(labels.detach().cpu().numpy())

# Сравнение метрик
base_accuracy = accuracy_score(test_labels, base_preds)
extended_accuracy = accuracy_score(test_labels, extended_preds)
base_f1 = f1_score(test_labels, base_preds)
extended_f1 = f1_score(test_labels, extended_preds)

print("\nСравнение моделей классификации:")
print(f"Accuracy базовой модели: {base_accuracy:.4f}")
print(f"Accuracy расширенной модели: {extended_accuracy:.4f}")
print(
    f"Улучшение accuracy: {(extended_accuracy - base_accuracy):.4f} ({((extended_accuracy - base_accuracy) / base_accuracy * 100):.2f}%)")

print(f"\nF1-score базовой модели: {base_f1:.4f}")
print(f"F1-score расширенной модели: {extended_f1:.4f}")
print(f"Улучшение F1-score: {(extended_f1 - base_f1):.4f} ({((extended_f1 - base_f1) / base_f1 * 100):.2f}%)")