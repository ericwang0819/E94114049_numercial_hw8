# -*- coding: utf-8 -*-
"""
Created on Tue May 27 22:28:16 2025

@author: ericd
"""

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# 定義原始函數 f(x) = x^2 * sin(x)
def f(x):
    return x**2 * np.sin(x)

# 設定 m = 16 個點
m = 16
x_j = np.linspace(0, 1, m)
f_j = f(x_j)

# 計算傅立葉係數
a0 = np.sum(f_j) / m

n = 4  # 三角多項式次數
a_k = np.zeros(n)
b_k = np.zeros(n)

for k in range(1, n + 1):
    a_k[k - 1] = 2 / m * np.sum(f_j * np.cos(2 * np.pi * k * x_j))
    b_k[k - 1] = 2 / m * np.sum(f_j * np.sin(2 * np.pi * k * x_j))

# 定義三角多項式 S_4(x)
def S4(x):
    result = a0
    for k in range(1, n + 1):
        result += a_k[k - 1] * np.cos(2 * np.pi * k * x) + b_k[k - 1] * np.sin(2 * np.pi * k * x)
    return result

# 計算積分 ∫₀¹ S_4(x) dx
integral_S4, _ = quad(S4, 0, 1)

# 計算真實積分 ∫₀¹ x² sin(x) dx
integral_true, _ = quad(lambda x: x**2 * np.sin(x), 0, 1)

# 計算誤差 E(S_4)
S4_j = S4(x_j)
E_S4 = np.mean((f_j - S4_j)**2)

# 輸出詳細結果
print("=== (a) 傅立葉係數 ===")
print(f"a0 = {a0}")
for k in range(n):
    print(f"a_{k+1} = {a_k[k]:.6f}, b_{k+1} = {b_k[k]:.6f}")

print("\n=== (b) 積分近似值 ∫₀¹ S₄(x) dx ===")
print(f"∫₀¹ S₄(x) dx ≈ {integral_S4:.8f}")

print("\n=== (c) 與精確值 ∫₀¹ x² sin(x) dx 比較 ===")
print(f"∫₀¹ x² sin(x) dx = {integral_true:.8f}")
print(f"誤差 = |近似值 - 真值| = {abs(integral_S4 - integral_true):.8f}")

print("\n=== (d) 離散最小平方誤差 E(S₄) ===")
print(f"E(S₄) = {E_S4:.8f}")
