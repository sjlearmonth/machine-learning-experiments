import numpy as np
import timeit
import tensorflow as tf
import sys

print(f"Numpy version: {np.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}\n")
print("Author: Stephen J Learmonth.")
print("Date: 11th December 2025.\n")

def get_min_exec_time(times, iters):
    MSECS_PER_SEC = 1e3
    USECS_PER_SEC = 1e6
    NSECS_PER_SEC = 1e9

    min_time = min(times)

    if min_time >= 1.0:

        time_per_compute = str(np.round(min_time / iters, 2))

        time_per_compute += " secs.\n"

    elif 1e-3 <= min_time < 1.0:

        time_per_compute = str((np.round(min_time / iters) * MSECS_PER_SEC, 2))

        time_per_compute += " msecs.\n"

    elif 1e-6 <= min_time < 1e-3:

        time_per_compute = str((np.round(min_time / iters) * USECS_PER_SEC, 2))

        time_per_compute += " usecs.\n"

    else:

        time_per_compute = str((np.round(min_time / iters) * NSECS_PER_SEC, 2))

        time_per_compute += " nsecs.\n"

    return time_per_compute

rows = cols = 1000
num_iters = 100
num_rpts = 6
rand_offset = 0.5
rand_range = 20

trad_w = (np.random.random(size=(rows, cols)) - rand_offset) * rand_range
trad_x = (np.random.random(size=(rows, cols)) - rand_offset) * rand_range
trad_b = (np.random.random(size=(rows, 1)) - rand_offset) * rand_range

elapsed_times_trad = timeit.repeat(stmt="trad_z = np.dot(trad_w, trad_x) + trad_b",
                                   setup="import numpy as np",
                                   repeat=num_rpts,
                                   number=num_rpts,
                                   globals=globals())

print("----------- Traditional Computation of z = w * x + b -----------\n")
print(f"                    w.shape = ({trad_w.shape[0]}, {trad_w.shape[1]})")
print(f"                    x.shape = ({trad_x.shape[0]}, {trad_x.shape[1]})")
print(f"                    b.shape = ({trad_b.shape[0]}, {trad_b.shape[1]})\n")

trad_exec_time_str = get_min_exec_time(elapsed_times_trad, num_iters)

print("Traditional computation of z: Min elapsed execution time: " + trad_exec_time_str)

alt_wb = np.append(trad_w, trad_b, axis=1)
alt_x1s = np.append(trad_x, np.ones((1, cols)), axis=0)

elapsed_times_alt = timeit.repeat(stmt="alt_z = np.dot(alt_wb, alt_x1s)",
                                  setup="import numpy as np",
                                  repeat=num_rpts,
                                  number=num_rpts,
                                  globals=globals())

min_elapsed_time_alt = min(elapsed_times_alt)

print("----------- Alternate Computation of z = wb * x1s --------------\n")
print(f"                    wb.shape  = ({alt_wb.shape[0]}, {alt_wb.shape[1]})")
print(f"                    x1s.shape = ({alt_x1s.shape[0]}, {alt_x1s.shape[1]})\n")

alt_exec_time_str = get_min_exec_time(elapsed_times_alt, num_iters)

print(f"Alternate computation of z: Min elapsed execution time: " + alt_exec_time_str)

trad_z = np.dot(trad_w, trad_x) + trad_b
alt_z = np.dot(alt_wb, alt_x1s)

if np.array_equal(trad_z, alt_z):

    print("Logit z computed traditionally and alternatively are both equal.\n")

    print(f"Performance increase: {performance_increase_pcnt:.2f}%.\n")

else:

    print("Error: Logit z computed traditionally and alternatively are not both equal.")
