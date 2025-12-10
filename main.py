import numpy as np
import timeit

trad_w = np.random.randn(1000, 1000)
trad_x = np.random.randn(1000, 1000)
trad_b = np.random.rand(1000, 1)

def compute_z_trad():

    global trad_w, trad_x, trad_b

    trad_z = np.dot(trad_w, trad_x) + trad_b

trad_times = timeit.repeat(stmt='compute_z_trad()',
              setup="import numpy as np",
              repeat=6,
              number=1000,
              globals=globals())

print("----------- Traditional Computation of z = w*x + b ----------\n")
print(f"Trad z: Least elapsed execution time: {np.round((min(trad_times) / 1000)*1e6, 2)}usecs.\n")

new_wb = np.random.randn(1000, 1000)
new_x = np.random.randn(1000, 1000)

def set_last_row_of_x():

    new_x[-1, :] = 1

    return

new_slrox_times = timeit.repeat(stmt='set_last_row_of_x()',
              setup="import numpy as np",
              repeat=6,
              number=1000,
              globals=globals())

print("----------- Set last row of x to 1 ----------\n")
print(f"Set last row of x to 1: Least elapsed execution time: {np.round((min(new_slrox_times) / 1000)*1e9, 2)}nsecs.\n")

def compute_z_new():

    global new_wb, new_x

    new_z = np.dot(new_wb, new_x)

new_times = timeit.repeat(stmt='compute_z_new()',
              setup="import numpy as np",
              repeat=6,
              number=1000,
              globals=globals())

print("----------- New Computation of z = wb*x ----------\n")
print(f"Set last row of x to 1: Least elapsed execution time: {np.round((min(new_times) / 1000)*1e6, 2)}usecs.\n")