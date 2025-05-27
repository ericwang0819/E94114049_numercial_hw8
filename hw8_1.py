# -*- coding: utf-8 -*-
"""
Created on Tue May 27 22:21:41 2025

@author: ericd
"""

import numpy as np

# 給定的數據
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# a. 二階多項式逼近 y = a + bx + cx^2
# 構造 Vandermonde 矩陣
A_poly = np.vstack([np.ones_like(x), x, x**2]).T
# 解最小二乘問題
coeffs_poly, _, _, _ = np.linalg.lstsq(A_poly, y, rcond=None)
a, b, c = coeffs_poly
# 計算預測值
y_pred_poly = a + b * x + c * x**2
# 計算誤差
error_poly = np.sum((y - y_pred_poly)**2)

# b. 指數形式逼近 y = b * e^(ax)
# 取對數：ln(y) = ln(b) + ax
ln_y = np.log(y)
A_exp = np.vstack([np.ones_like(x), x]).T
coeffs_exp, _, _, _ = np.linalg.lstsq(A_exp, ln_y, rcond=None)
ln_b, a_exp = coeffs_exp
b_exp = np.exp(ln_b)
# 計算預測值
y_pred_exp = b_exp * np.exp(a_exp * x)
# 計算誤差
error_exp = np.sum((y - y_pred_exp)**2)

# c. 冪函數形式逼近 y = b * x^n
# 取對數：ln(y) = ln(b) + n * ln(x)
ln_x = np.log(x)
A_power = np.vstack([np.ones_like(ln_x), ln_x]).T
coeffs_power, _, _, _ = np.linalg.lstsq(A_power, ln_y, rcond=None)
ln_b_power, n = coeffs_power
b_power = np.exp(ln_b_power)
# 計算預測值
y_pred_power = b_power * x**n
# 計算誤差
error_power = np.sum((y - y_pred_power)**2)

# 輸出結果
print("a. 二階多項式逼近：")
print(f"y = {a:.2f} + {b:.2f}x + {c:.2f}x^2")
print(f"誤差：{error_poly:.2f}\n")

print("b. 指數形式逼近：")
print(f"y = {b_exp:.4f} * e^({a_exp:.2f}x)")
print(f"誤差：{error_exp:.2f}\n")

print("c. 冪函數形式逼近：")
print(f"y = {b_power:.2f} * x^{n:.2f}")
print(f"誤差：{error_power:.2f}")