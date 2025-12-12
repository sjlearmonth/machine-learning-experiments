import numpy as np
import timeit
import time
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

    time_per_compute = min_time / iters

    if time_per_compute >= 1.0:

        units = " secs."

    elif 1e-3 <= time_per_compute < 1.0:

        time_per_compute *= MSECS_PER_SEC

        units = " msecs."

    elif 1e-6 <= time_per_compute < 1e-3:

        time_per_compute *= USECS_PER_SEC

        units = " usecs."

    else:

        time_per_compute *= NSECS_PER_SEC

        units = " nsecs."

    return time_per_compute, units

rows = cols = 8000
num_iters = 10
num_rpts = 6
rand_offset = 0.5
rand_range = 20

trad_w = (np.random.random(size=(rows, cols)) - rand_offset) * rand_range
trad_x = (np.random.random(size=(rows, cols)) - rand_offset) * rand_range
trad_b = (np.random.random(size=(rows, 1)) - rand_offset) * rand_range

trad_start = time.perf_counter()
trad_elapsed_times = timeit.repeat(stmt="trad_z = np.dot(trad_w, trad_x) + trad_b",
                                   setup="import numpy as np",
                                   repeat=num_rpts,
                                   number=num_rpts,
                                   globals=globals())
trad_end = time.perf_counter()

trad_exec_time, trad_time_units = get_min_exec_time(trad_elapsed_times, num_iters)

print("----------- Traditional Computation of z = w * x + b -----------")
print(f"                    w.shape = ({trad_w.shape[0]}, {trad_w.shape[1]})")
print(f"                    x.shape = ({trad_x.shape[0]}, {trad_x.shape[1]})")
print(f"                    b.shape = ({trad_b.shape[0]}, {trad_b.shape[1]})")
print(f"Elapsed time to compute z over {num_iters} iterations: {(trad_end - trad_start) / num_rpts:.2f} seconds.")
print(f"Computation of z takes a minimum of : {trad_exec_time:.2f}" + trad_time_units + "\n")

alt_wb = np.append(trad_w, trad_b, axis=1)
alt_x1s = np.append(trad_x, np.ones((1, cols)), axis=0)

alt_start = time.perf_counter()
alt_elapsed_times = timeit.repeat(stmt="alt_z = np.dot(alt_wb, alt_x1s)",
                                   setup="import numpy as np",
                                   repeat=num_rpts,
                                   number=num_rpts,
                                   globals=globals())
alt_end = time.perf_counter()

alt_exec_time, alt_time_units = get_min_exec_time(alt_elapsed_times, num_iters)

print("----------- Alternate Computation of z = wb * x1s --------------")
print(f"                    wb.shape  = ({alt_wb.shape[0]}, {alt_wb.shape[1]})")
print(f"                    x1s.shape = ({alt_x1s.shape[0]}, {alt_x1s.shape[1]})")
print(f"Elapsed time to compute z over {num_iters} iterations: {(alt_end - alt_start) / num_rpts:.2f} seconds.")
print(f"Computation of z takes a minimum of : {alt_exec_time:.2f}" + alt_time_units + "\n")

trad_z = np.dot(trad_w, trad_x) + trad_b
alt_z = np.dot(alt_wb, alt_x1s)
perf_inc_pcnt = ( (trad_exec_time - alt_exec_time) / trad_exec_time ) * 100

if np.array_equal(trad_z, alt_z):

    print("Logit z computed traditionally and alternatively are both equal.\n")

    print(f"Performance increase: {perf_inc_pcnt:.2f}%.\n")

else:

    print("Error: Logit z computed traditionally and alternatively are not both equal.\n")
