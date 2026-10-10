import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

# 1. ЗАГРУЗКА ДАННЫХ
print("1. Загрузка данных...")
df = pd.read_csv('dataset_simple.csv')
print(f"   Загружено {len(df)} строк")
print(f"   Столбцы: {list(df.columns)}")
print(f"   Первые 5 строк:\n{df.head()}")

# 2. ПОДГОТОВКА ДАННЫХ
# Признаки: возраст и доход (первые два столбца)
X = torch.tensor(df.iloc[:, 0:2].values, dtype=torch.float32)
# Цель: купит/не купит (третий столбец)
y = torch.tensor(df.iloc[:, 2].values.reshape(-1, 1), dtype=torch.float32)

print(f"\n2. Размерность признаков X: {X.shape}")
print(f"   Размерность цели y: {y.shape}")
print(f"   Пример: возраст={X[0,0]:.0f}, доход={X[0,1]:.0f}, купит={y[0,0]:.0f}")

# 3. НОРМАЛИЗАЦИЯ ДАННЫХ (очень важно для обучения!)
# Вычисляем среднее и стандартное отклонение для каждого признака
mean = X.mean(dim=0)
std = X.std(dim=0)
X_normalized = (X - mean) / std

print(f"\n3. Нормализация данных:")
print(f"   Среднее (возраст, доход): {mean}")
print(f"   Стандартное отклонение: {std}")

# 4. СОЗДАНИЕ НЕЙРОННОЙ СЕТИ
class NNet_classification(nn.Module):
    """Нейронная сеть для бинарной классификации"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()  # современный способ вызова родительского конструктора
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),  # входной слой -> скрытый
            nn.ReLU(),                         # функция активации ReLU (лучше чем Tanh)
            nn.Linear(hidden_size, hidden_size),  # второй скрытый слой (улучшает обучение)
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),  # скрытый -> выходной
            nn.Sigmoid()                       # сжимаем в [0,1] для вероятности
        )
    
    def forward(self, X):
        return self.layers(X)

# Параметры сети
inputSize = X_normalized.shape[1]  # = 2 (возраст и доход)
hiddenSize = 16                     # количество нейронов в скрытом слое
outputSize = 1                      # один выход (вероятность покупки)

print(f"\n4. Архитектура сети:")
print(f"   Входной слой: {inputSize} нейрона")
print(f"   Скрытые слои: {hiddenSize} нейронов каждый")
print(f"   Выходной слой: {outputSize} нейрон")

# Создаем экземпляр сети
net = NNet_classification(inputSize, hiddenSize, outputSize)

# 5. НАСТРОЙКА ОБУЧЕНИЯ
# Функция потерь для бинарной классификации
lossFn = nn.BCELoss()  # Binary Cross Entropy Loss

# Оптимизатор: Adam работает лучше, чем SGD (быстрее сходится)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

print(f"\n5. Настройка обучения:")
print(f"   Функция потерь: BCELoss")
print(f"   Оптимизатор: Adam, lr=0.01")

# 6. ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ
epochs = 2000
print(f"\n6. Начинаем обучение на {epochs} эпохах...")
print("   Эпоха | Ошибка (loss)")
print("   " + "-" * 25)

train_losses = []  # для сохранения истории ошибок

for epoch in range(epochs):
    # Прямой проход: предсказываем
    predictions = net(X_normalized)
    
    # Считаем ошибку
    loss = lossFn(predictions, y)
    train_losses.append(loss.item())
    
    # Обратный проход: обновляем веса
    optimizer.zero_grad()  # обнуляем градиенты
    loss.backward()        # вычисляем градиенты
    optimizer.step()       # делаем шаг оптимизации
    
    # Выводим прогресс каждые 200 эпох
    if (epoch + 1) % 200 == 0:
        print(f"   {epoch+1:5d} | {loss.item():.6f}")

print(f"\n   Обучение завершено!")
print(f"   Финальная ошибка: {train_losses[-1]:.6f}")

# 7. ОЦЕНКА КАЧЕСТВА
print("\n7. Оценка качества на обучающей выборке...")
with torch.no_grad():  # отключаем вычисление градиентов (не нужно для оценки)
    final_predictions = net(X_normalized)

# Превращаем вероятности в классы (0 или 1)
predicted_classes = torch.where(final_predictions >= 0.5, 
                                torch.tensor(1.0), 
                                torch.tensor(0.0))

# Считаем количество ошибок
errors = (y != predicted_classes).sum().item()
accuracy = (len(df) - errors) / len(df) * 100

print(f"   Предсказанные вероятности (первые 10):")
for i in range(10):
    print(f"      Пример {i+1}: реальный={int(y[i,0])}, "
          f"предсказанный={int(predicted_classes[i,0])}, "
          f"вероятность={final_predictions[i,0]:.3f}")

print(f"\n   Результаты:")
print(f"   Всего примеров: {len(df)}")
print(f"   Ошибок: {errors}")
print(f"   Точность: {accuracy:.2f}%")

# 8. ВИЗУАЛИЗАЦИЯ (чтобы увидеть, как училась сеть)
plt.figure(figsize=(12, 5))

# График 1: Ошибка обучения
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Эпоха')
plt.ylabel('Ошибка (loss)')
plt.title('Кривая обучения')
plt.grid(True)

# График 2: Разделяющая граница (только для 2D данных)
plt.subplot(1, 2, 2)
# Создаем сетку точек для визуализации
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5000, X[:, 1].max() + 5000
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Нормализуем сетку и предсказываем
grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
grid_points_norm = (grid_points - mean) / std
with torch.no_grad():
    Z = net(grid_points_norm).numpy()
Z = Z.reshape(xx.shape)

# Рисуем границу
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
# Рисуем точки
plt.scatter(X[y[:,0]==0, 0], X[y[:,0]==0, 1], c='blue', label='Не купит', alpha=0.7)
plt.scatter(X[y[:,0]==1, 0], X[y[:,0]==1, 1], c='red', label='Купит', alpha=0.7)
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Разделяющая граница')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 9. ПРИМЕР ПРЕДСКАЗАНИЯ ДЛЯ НОВОГО ПОКУПАТЕЛЯ
print("\n8. Пример предсказания для нового покупателя:")
new_age = 33
new_income = 45000
print(f"   Возраст: {new_age}, Доход: {new_income}")

# Нормализуем новый пример
new_data = torch.tensor([[new_age, new_income]], dtype=torch.float32)
new_data_norm = (new_data - mean) / std

with torch.no_grad():
    probability = net(new_data_norm).item()

print(f"   Вероятность покупки: {probability:.3f}")
if probability >= 0.5:
    print(f"   ➡️  Прогноз: ПОКУПИТ")
else:
    print(f"   ➡️  Прогноз: НЕ ПОКУПИТ")