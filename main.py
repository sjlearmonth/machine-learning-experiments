import numpy as np
import timeit
import tensorflow as tf
import sys

print(f"Numpy version: {np.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}\n")
print("Author: Stephen J Learmonth.")
print("Date: 11th December 2025.\n")

rows = cols = 1000

trad_w = (np.random.random(size=(rows, cols)) - 0.5) * 20
trad_x = (np.random.random(size=(rows, cols)) - 0.5) * 20
trad_b = (np.random.random(size=(rows, 1)) - 0.5) * 20

elapsed_times_trad = timeit.repeat(stmt="trad_z = np.dot(trad_w, trad_x) + trad_b",
                                   setup="import numpy as np",
                                   repeat=6,
                                   number=1000,
                                   globals=globals())

min_elapsed_time_trad = min(elapsed_times_trad)

print("----------- Traditional Computation of z = w * x + b --------\n")
print("                    w.shape = (1000, 1000)                   ")
print("                    x.shape = (1000, 1000)                   ")
print("                    b.shape = (1000, 1)                      \n")

print(f"Traditional computation of z: Min elapsed execution time: {np.round((min_elapsed_time_trad / 1000)*1e3, 2)} msecs.\n")

alt_wb = np.append(trad_w, (np.random.random(size=(rows, 1)) - 0.5) * 20, axis=1)
alt_x1s = np.append(trad_x, np.ones((1, cols)), axis=0)

elapsed_times_alt = timeit.repeat(stmt="alt_z = np.dot(alt_wb, alt_x1s)",
                                  setup="import numpy as np",
                                  repeat=6,
                                  number=1000,
                                  globals=globals())

min_elapsed_time_alt = min(elapsed_times_alt)

performance_increase_pcnt = ((min_elapsed_time_trad - min_elapsed_time_alt) / min_elapsed_time_trad) * 100

print("---------------- Alternate Computation of z = wb * x1s --------------\n")
print("                    wb.shape = (1000, 1001)                        ")
print("                    x1s.shape = (1001, 1000)                       \n")

print(f"Alternate computation of z: Min elapsed execution time: {np.round((min_elapsed_time_alt / 1000)*1e3, 2)} msecs.\n")
print(f"Performance increase: {performance_increase_pcnt:.2f}%.\n")
