# -*- coding: utf-8 -*-
"""
Лабораторная работа №3, задание 2
Классификация цветков ириса с помощью PyTorch
"""

## !!!!!!!!!! Программа должна запускаться в режиме интерпретатора !!!!!!!!!!!!!!!!!
## !!!!!!!!   Читайте комментарии и только после этого запускайте код !!!!!!!!!!!!!

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Загружаем данные ирисов
iris = load_iris()
X = iris.data  # Признаки: длина/ширина чашелистика и лепестка
y = iris.target  # Классы: 0 - setosa, 1 - versicolor, 2 - virginica

print("Размерность данных X:", X.shape)
print("Размерность меток y:", y.shape)
print("Названия признаков:", iris.feature_names)
print("Названия классов:", iris.target_names)

# 2. Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nОбучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")

# 3. Нормализуем данные
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Преобразуем данные в тензоры PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)  # LongTensor для функции потерь CrossEntropy
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

print(f"\nТип X_train_tensor: {X_train_tensor.dtype}, размер: {X_train_tensor.shape}")
print(f"Тип y_train_tensor: {y_train_tensor.dtype}, размер: {y_train_tensor.shape}")

# 5. Создаем модель нейронной сети
class IrisClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, output_size=3):
        super(IrisClassifier, self).__init__()
        # Определяем слои сети
        self.fc1 = nn.Linear(input_size, hidden_size)  # Полносвязный слой 1
        self.relu = nn.ReLU()                          # Функция активации
        self.fc2 = nn.Linear(hidden_size, output_size) # Полносвязный слой 2
        # Функция активации на выходе не используется, 
        # так как CrossEntropyLoss включает в себя Softmax
        
    def forward(self, x):
        # Прямой проход через сеть
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Создаем экземпляр модели
model = IrisClassifier()
print("\nСтруктура модели:")
print(model)

# 6. Определяем функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()  # Функция потерь для классификации
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Оптимизатор SGD

# 7. Обучение модели
num_epochs = 100
train_losses = []
train_accuracies = []

print("\nНачинаем обучение...")
for epoch in range(num_epochs):
    # Прямой проход
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Обратный проход и оптимизация
    optimizer.zero_grad()  # Обнуляем градиенты
    loss.backward()        # Вычисляем градиенты
    optimizer.step()       # Обновляем веса
    
    # Вычисляем точность на обучающей выборке
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == y_train_tensor).sum().item()
    accuracy = correct / len(y_train_tensor)
    
    # Сохраняем метрики
    train_losses.append(loss.item())
    train_accuracies.append(accuracy)
    
    # Выводим прогресс каждые 10 эпох
    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch+1}/{num_epochs}], '
              f'Потери: {loss.item():.4f}, '
              f'Точность: {accuracy:.4f}')

# 8. Тестирование модели
print("\n\nТестирование модели...")
model.eval()  # Переводим модель в режим оценки
with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
    # Предсказания на тестовой выборке
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    
    # Вычисляем точность
    correct = (predicted == y_test_tensor).sum().item()
    accuracy = correct / len(y_test_tensor)
    
    print(f'Точность на тестовой выборке: {accuracy:.4f} ({correct}/{len(y_test_tensor)})')
    
    # Детальная информация по классам
    print("\nДетализация по классам:")
    for i in range(3):
        class_mask = (y_test_tensor == i)
        class_correct = (predicted[class_mask] == i).sum().item()
        class_total = class_mask.sum().item()
        if class_total > 0:
            print(f'Класс {iris.target_names[i]}: {class_correct}/{class_total} '
                  f'({class_correct/class_total:.4f})')

# 9. Визуализация процесса обучения (если установлен matplotlib)
try:
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График функции потерь
    ax1.plot(train_losses)
    ax1.set_title('Функция потерь во время обучения')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.grid(True)
    
    # График точности
    ax2.plot(train_accuracies)
    ax2.set_title('Точность на обучающей выборке')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("\nДля визуализации установите matplotlib: pip install matplotlib")

# 10. Пример использования модели для предсказания
print("\n\nПример предсказания для новых данных:")

# Создаем "новый" цветок ириса (нормализованные признаки)
new_flower = torch.FloatTensor([[0.5, -0.3, 1.2, 0.8]])  # Пример признаков
model.eval()
with torch.no_grad():
    prediction = model(new_flower)
    _, predicted_class = torch.max(prediction, 1)
    print(f"Признаки нового цветка: {new_flower.numpy()[0]}")
    print(f"Предсказанный класс: {iris.target_names[predicted_class.item()]}")
    
    # Вероятности для каждого класса
    probabilities = torch.softmax(prediction, dim=1)
    print("\nВероятности для каждого класса:")
    for i, (prob, name) in enumerate(zip(probabilities[0], iris.target_names)):
        print(f"  {name}: {prob.item():.4f}")

# 11. Анализ весов модели
print("\n\nАнализ весов модели:")
print("Веса первого слоя (fc1):")
print(model.fc1.weight.shape)
print("Смещения первого слоя (fc1):")
print(model.fc1.bias.shape)

print("\nВеса второго слоя (fc2):")
print(model.fc2.weight.shape)
print("Смещения второго слоя (fc2):")
print(model.fc2.bias.shape)

# 12. Сохранение модели
torch.save(model.state_dict(), 'iris_classifier.pth')
print("\nМодель сохранена в файл 'iris_classifier.pth'")

# 13. Загрузка модели (пример)
loaded_model = IrisClassifier()
loaded_model.load_state_dict(torch.load('iris_classifier.pth'))
loaded_model.eval()
print("Модель успешно загружена из файла!")
