import numpy as np
import matplotlib.pyplot as plt

def vortex_field_general(x, y, A, q, a, N, centers=None, signs=None):
    """
    Вычисляет поле как сумму N гауссовских вихрей на оси X.
    
    Параметры:
        x, y : массивы координат
        A, q, a : параметры вихрей (амплитуда, ширина, шаг)
        N : количество вихрей
        centers : список координат центров (если None, генерируются равномерно)
        signs : список знаков (если None, чередуются: +1, -1, +1,...)
    """
    if centers is None:
        # Равномерное расположение симметрично относительно 0
        offset = (N - 1) / 2.0
        centers = a * (np.arange(N) - offset)
    else:
        centers = np.asarray(centers)
        N = len(centers)

    if signs is None:
        signs = np.array([1 if i % 2 == 0 else -1 for i in range(N)])
    else:
        signs = np.asarray(signs)

    U = np.zeros_like(x)
    V = np.zeros_like(x)
    eps = 1e-12  # защита от деления на ноль

    for d, s in zip(centers, signs):
        dx = x - d
        r2 = dx**2 + y**2
        r = np.sqrt(r2 + eps)
        exp_factor = np.exp(-r2 / (2 * q**2))
        factor = s * A / (2 * np.pi * q**2) * exp_factor / r
        U += factor * (-y)
        V += factor * (dx)

    return U, V

# Пример использования
A = 1.0
q = 0.5
a = 1.0
N = 10  # количество вихрей

x_range = (-4, 4)
y_range = (-4, 4)
step = 0.2

x = np.arange(x_range[0], x_range[1] + step, step)
y = np.arange(y_range[0], y_range[1] + step, step)
X, Y = np.meshgrid(x, y)

U, V = vortex_field_general(X, Y, A, q, a, N)

fig, ax = plt.subplots(figsize=(8, 6))
ax.quiver(X, Y, U, V, color='blue', alpha=0.6, scale=30)
ax.streamplot(X, Y, U, V, color='black', linewidth=0.5, density=1.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'{N} вихрей с шагом {a}')
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()