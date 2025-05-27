# -*- coding: utf-8 -*-
"""
Created on Tue May 27 22:24:41 2025

@author: ericd
"""

import numpy as np
from scipy.integrate import quad

# 定義目標函數 f(x)
def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

# 計算積分
I_f = quad(f, -1, 1)[0]  # ∫f(x) dx
I_f_x = quad(lambda x: f(x) * x, -1, 1)[0]  # ∫f(x) x dx
I_f_x2 = quad(lambda x: f(x) * x**2, -1, 1)[0]  # ∫f(x) x^2 dx

I_1 = 2  # ∫1 dx
I_x = 0  # ∫x dx
I_x2 = 2 / 3  # ∫x^2 dx
I_x3 = 0  # ∫x^3 dx
I_x4 = 2 / 5  # ∫x^4 dx

# 構造正則方程的矩陣
A = np.array([
    [I_1, I_x, I_x2],
    [I_x, I_x2, I_x3],
    [I_x2, I_x3, I_x4]
])
B = np.array([I_f, I_f_x, I_f_x2])

# 解方程得到係數 a, b, c
coeffs = np.linalg.solve(A, B)
a, b, c = coeffs

# 輸出結果
print("二階最小二乘多項式逼近：")
print(f"p(x) = {a:.4f} + {b:.4f}x + {c:.4f}x^2")