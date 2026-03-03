import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================
# ЗАДАНИЕ 1: Классификация покупателей
# =============================================
print("=" * 60)
print("ЗАДАНИЕ 1: Классификация покупателей на классы 'купит' - 'не купит'")
print("Номер студента: 11 (нечетный)")
print("=" * 60)

# 1. Загружаем данные из файла dataset_simple.csv
print("\n1. Загрузка данных из файла dataset_simple.csv...")
df = pd.read_csv('dataset_simple.csv')

# Проверяем структуру данных
print(f"   Размерность данных: {df.shape}")
print(f"   Столбцы: {df.columns.tolist()}")
print(f"\n   Первые 5 строк данных:")
print(df.head())

print(f"\n   Статистика по данным:")
print(df.describe())

print(f"\n   Распределение классов (3-й столбец):")
print(df.iloc[:, 2].value_counts())

# 2. Подготовка данных для классификации
print("\n2. Подготовка данных для классификации...")

# Признаки: возраст (1-й столбец) и доход (2-й столбец)
X = torch.Tensor(df.iloc[:, 0:2].values)

# Целевая переменная: купит/не купит (3-й столбец)
# Преобразуем метки классов: 1 -> "купит", 0 -> "не купит"
y_original = df.iloc[:, 2].values
y = torch.Tensor(np.where(y_original == 1, 1, 0).reshape(-1, 1))

print(f"   Размерность признаков X: {X.shape}")
print(f"   Размерность меток y: {y.shape}")
print(f"   Баланс классов: {np.bincount(y.numpy().astype(int).flatten())}")

# 3. Разделение данных на обучающую и тестовую выборки
print("\n3. Разделение данных на обучающую и тестовую выборки...")
X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(), y.numpy(), test_size=0.2, random_state=42, stratify=y.numpy()
)

# Преобразуем обратно в тензоры PyTorch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

print(f"   Обучающая выборка: {X_train_tensor.shape[0]} образцов")
print(f"   Тестовая выборка: {X_test_tensor.shape[0]} образцов")

# 4. Визуализация данных
print("\n4. Визуализация исходных данных...")
plt.figure(figsize=(12, 5))

# Создаем подграфики
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# График 1: Исходные данные
colors = ['red' if label == 0 else 'blue' for label in y.numpy().flatten()]
ax1.scatter(X.numpy()[:, 0], X.numpy()[:, 1], c=colors, alpha=0.6, edgecolors='k')
ax1.set_title('Исходные данные: Классификация покупателей')
ax1.set_xlabel('Возраст')
ax1.set_ylabel('Доход')
ax1.grid(True, alpha=0.3)
ax1.legend(['Не купит (0)', 'Купит (1)'], loc='upper right')

# График 2: Разделение на train/test
train_colors = ['red' if label == 0 else 'blue' for label in y_train.flatten()]
test_colors = ['orange' if label == 0 else 'green' for label in y_test.flatten()]

ax2.scatter(X_train[:, 0], X_train[:, 1], c=train_colors, alpha=0.6, edgecolors='k', label='Обучающая')
ax2.scatter(X_test[:, 0], X_test[:, 1], c=test_colors, alpha=0.6, edgecolors='k', marker='s', label='Тестовая')
ax2.set_title('Разделение данных на обучающую и тестовую выборки')
ax2.set_xlabel('Возраст')
ax2.set_ylabel('Доход')
ax2.grid(True, alpha=0.3)
ax2.legend(['Train: Не купит', 'Train: Купит', 'Test: Не купит', 'Test: Купит'], loc='upper right')

plt.tight_layout()
plt.show()

# 5. Создаем нейронную сеть для бинарной классификации
print("\n5. Создание нейронной сети для бинарной классификации...")

class CustomerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Входной слой
            nn.ReLU(),                           # Функция активации
            nn.Linear(hidden_size, hidden_size), # Скрытый слой
            nn.ReLU(),                           # Функция активации
            nn.Linear(hidden_size, output_size), # Выходной слой
            nn.Sigmoid()                         # Сигмоида для вероятности
        )
    
    def forward(self, X):
        pred = self.layers(X)
        return pred

# Параметры сети
inputSize = X_train_tensor.shape[1]  # 2 признака: возраст и доход
hiddenSizes = 8                      # Количество нейронов в скрытом слое
outputSize = 1                       # 1 выходной нейрон (вероятность покупки)

# Создаем экземпляр сети
net = CustomerClassifier(inputSize, hiddenSizes, outputSize)

print(f"   Архитектура сети:")
print(f"   - Входной слой: {inputSize} нейрона")
print(f"   - Скрытый слой 1: {hiddenSizes} нейронов")
print(f"   - Скрытый слой 2: {hiddenSizes} нейронов")
print(f"   - Выходной слой: {outputSize} нейрон")
print(f"   - Активации: ReLU в скрытых слоях, Sigmoid на выходе")

# 6. Просмотр параметров сети до обучения
print("\n6. Параметры сети до обучения:")
for name, param in net.named_parameters():
    print(f"   {name}: размер {param.shape}")

# 7. Оценка работы сети до обучения
print("\n7. Оценка работы сети до обучения...")
with torch.no_grad():
    pred_train = net.forward(X_train_tensor)
    pred_binary = torch.round(pred_train)  # Преобразуем вероятности в бинарные метки
    
    # Считаем точность
    correct = (pred_binary == y_train_tensor).sum().item()
    accuracy = correct / len(y_train_tensor)
    
    print(f"   Точность на обучающей выборке до обучения: {accuracy:.4f}")

# 8. Обучение сети
print("\n8. Обучение нейронной сети...")

# Функция потерь для бинарной классификации
lossFn = nn.BCELoss()  # Binary Cross Entropy Loss

# Оптимизатор
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Списки для хранения истории обучения
train_losses = []
train_accuracies = []

# Цикл обучения
epochs = 200
print(f"   Начало обучения на {epochs} эпох...")

for i in range(epochs):
    # Прямой проход
    pred = net.forward(X_train_tensor)
    loss = lossFn(pred, y_train_tensor)
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Вычисление точности
    with torch.no_grad():
        pred_binary = torch.round(pred)
        correct = (pred_binary == y_train_tensor).sum().item()
        accuracy = correct / len(y_train_tensor)
    
    # Сохранение метрик
    train_losses.append(loss.item())
    train_accuracies.append(accuracy)
    
    # Вывод прогресса каждые 20 эпох
    if (i + 1) % 20 == 0:
        print(f'   Эпоха [{i+1}/{epochs}], Потери: {loss.item():.4f}, Точность: {accuracy:.4f}')

print("   Обучение завершено!")

# 9. Тестирование обученной сети
print("\n9. Тестирование обученной сети на тестовой выборке...")

net.eval()  # Переводим сеть в режим оценки
with torch.no_grad():
    # Предсказания на тестовой выборке
    pred_test = net.forward(X_test_tensor)
    pred_binary_test = torch.round(pred_test)
    
    # Вычисление метрик
    correct_test = (pred_binary_test == y_test_tensor).sum().item()
    accuracy_test = correct_test / len(y_test_tensor)
    
    # Матрица ошибок
    tp = ((pred_binary_test == 1) & (y_test_tensor == 1)).sum().item()  # True Positives
    tn = ((pred_binary_test == 0) & (y_test_tensor == 0)).sum().item()  # True Negatives
    fp = ((pred_binary_test == 1) & (y_test_tensor == 0)).sum().item()  # False Positives
    fn = ((pred_binary_test == 0) & (y_test_tensor == 1)).sum().item()  # False Negatives
    
    print(f"   Точность на тестовой выборке: {accuracy_test:.4f} ({correct_test}/{len(y_test_tensor)})")
    print(f"\n   Матрица ошибок:")
    print(f"   True Positives (TP): {tp}")
    print(f"   True Negatives (TN): {tn}")
    print(f"   False Positives (FP): {fp}")
    print(f"   False Negatives (FN): {fn}")
    
    # Дополнительные метрики
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n   Дополнительные метрики:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1_score:.4f}")

# 10. Визуализация процесса обучения
print("\n10. Визуализация процесса обучения...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# График функции потерь
ax1.plot(train_losses)
ax1.set_title('Функция потерь во время обучения')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Потери (BCELoss)')
ax1.grid(True, alpha=0.3)

# График точности
ax2.plot(train_accuracies)
ax2.set_title('Точность на обучающей выборке')
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Точность')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 11. Визуализация разделяющей поверхности
print("\n11. Визуализация разделяющей поверхности...")

# Создаем сетку для визуализации
x_min, x_max = X.numpy()[:, 0].min() - 1, X.numpy()[:, 0].max() + 1
y_min, y_max = X.numpy()[:, 1].min() - 1, X.numpy()[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 100))

# Предсказания для всех точек сетки
with torch.no_grad():
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    Z = net.forward(grid_points)
    Z = Z.numpy().reshape(xx.shape)

# Создаем график
plt.figure(figsize=(10, 8))

# Отображаем разделяющую поверхность
plt.contourf(xx, yy, Z, alpha=0.4, levels=np.linspace(0, 1, 11))
plt.colorbar(label='Вероятность покупки')

# Отображаем исходные точки
scatter = plt.scatter(X.numpy()[:, 0], X.numpy()[:, 1], 
                     c=y.numpy().flatten(), cmap='bwr', 
                     edgecolors='k', s=100)
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Разделяющая поверхность нейронной сети')
plt.legend(handles=scatter.legend_elements()[0], 
           labels=['Не купит', 'Купит'])
plt.grid(True, alpha=0.3)

plt.show()

# 12. Пример использования модели для предсказания
print("\n12. Пример использования модели для предсказания:")

# Тестовые примеры
test_cases = [
    {"age": 25, "income": 30000, "label": "Не купит (ожидается)"},
    {"age": 45, "income": 80000, "label": "Купит (ожидается)"},
    {"age": 30, "income": 50000, "label": "Пограничный случай"},
    {"age": 60, "income": 40000, "label": "Проверка возраста"},
]

print("\n   Предсказания для тестовых примеров:")
print("   " + "-" * 60)

for i, test in enumerate(test_cases, 1):
    with torch.no_grad():
        input_data = torch.FloatTensor([[test["age"], test["income"]]])
        prediction = net.forward(input_data)
        probability = prediction.item()
        decision = "КУПИТ" if probability > 0.5 else "НЕ КУПИТ"
        
        print(f"   Пример {i}:")
        print(f"     Возраст: {test['age']} лет, Доход: {test['income']} руб.")
        print(f"     Вероятность покупки: {probability:.4f}")
        print(f"     Решение: {decision} (порог 0.5)")
        print(f"     Комментарий: {test['label']}")
        print()

# 13. Анализ важности признаков
print("\n13. Анализ важности признаков...")

# Веса первого слоя сети
with torch.no_grad():
    weights_first_layer = net.layers[0].weight.detach().numpy()
    
    print("   Веса входного слоя:")
    print(f"     Веса для признака 'Возраст': {weights_first_layer[:, 0]}")
    print(f"     Веса для признака 'Доход': {weights_first_layer[:, 1]}")
    
    # Абсолютная сумма весов как мера важности
    importance_age = np.abs(weights_first_layer[:, 0]).sum()
    importance_income = np.abs(weights_first_layer[:, 1]).sum()
    
    print(f"\n   Относительная важность признаков:")
    print(f"     Возраст: {importance_age:.4f}")
    print(f"     Доход: {importance_income:.4f}")
    print(f"     Отношение (Доход/Возраст): {importance_income/importance_age:.4f}")

print("\n" + "=" * 60)
print("ЗАДАНИЕ 1 ВЫПОЛНЕНО УСПЕШНО!")
print("=" * 60)
