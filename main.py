import numpy as np
import timeit
import time
import tensorflow as tf
import sys

# print(f"Numpy version: {np.__version__}")
# print(f"TensorFlow version: {tf.__version__}")
# print(f"Python version: {sys.version}\n")
#
# print("Author: Stephen J Learmonth.")
# print("Date: 11th December 2025.\n")

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


# High w_rows means a large amount of Python broadcasting of b; High number of x_cols means
# a large amount of multiplications and additions to form a single dot product result.
w_rows = 100_000; w_cols = 50
x_rows = w_cols; x_cols = w_rows
b_rows = w_rows; b_cols = 1

# cols = 500_000
num_iters = 10
num_rpts = 10
rand_offset = 0.5
rand_range = 20

rng = np.random.default_rng(42)
trad_w = (rng.random(size=(w_rows, w_cols)) - rand_offset) * rand_range
trad_x = (rng.random(size=(x_rows, x_cols)) - rand_offset) * rand_range
trad_b = (rng.random(size=(b_rows, b_cols)) - rand_offset) * rand_range

# trad_w = np.random.randint(low=1, high=7, size=(w_rows, w_cols), dtype=int)
# trad_x = np.random.randint(low=1, high=7, size=(x_rows, x_cols), dtype=int)
# trad_b = np.random.randint(low=1, high=7, size=(b_rows, b_cols), dtype=int)
# print(f"trad_w.shape: {trad_w.shape}")
# print(f"trad_x.shape: {trad_x.shape}")
# print(f"trad_b.shape: {trad_b.shape}")
# trad_z = np.dot(trad_w, trad_x) + trad_b
# print(f"trad_z.shape: {trad_z.shape}\n")

# print(f"np.dot(trad_w, trad_x):\n{np.dot(trad_w, trad_x)}")
# print(f"trad_w:\n{trad_w}")
# print(f"trad_x:\n{trad_x}")
# print(f"trad_b:\n{trad_b}")
# print(f"trad_z:\n{trad_z}")
# exit()

trad_start = time.perf_counter()
trad_elapsed_times = timeit.repeat(stmt="trad_z = np.dot(trad_w, trad_x) + trad_b",
                                   setup="import numpy as np",
                                   repeat=num_rpts,
                                   number=num_rpts,
                                   globals=globals())
trad_end = time.perf_counter()

trad_z = np.dot(trad_w, trad_x) + trad_b

trad_exec_time, trad_time_units = get_min_exec_time(trad_elapsed_times, num_iters)

print("----------- Traditional Computation of z = w * x + b -----------\n")
print(f"                    W.shape = ({trad_w.shape[0]}, {trad_w.shape[1]})")
print(f"                    X.shape = ({trad_x.shape[0]}, {trad_x.shape[1]})")
print(f"                    B.shape = ({trad_b.shape[0]}, {trad_b.shape[1]})")
print(f"                    Z.shape = ({trad_z.shape[0]}, {trad_z.shape[1]})\n")
print(f"Computation of z = w*x + b, {w_rows * x_cols * num_iters} times, takes a minimum of : {trad_exec_time:.2f}" + trad_time_units + "\n")

alt_wb = np.append(trad_w, trad_b, axis=1)
alt_x1s = np.append(trad_x, np.ones((1, x_cols)), axis=0)

print(f"alt_wb.shape: {alt_wb.shape}")
print(f"alt_x1s.shape: {alt_x1s.shape}")
# print(alt_x1s[-3:])
# exit()

alt_start = time.perf_counter()
alt_elapsed_times = timeit.repeat(stmt="alt_z = np.dot(alt_wb, alt_x1s)",
                                   setup="import numpy as np",
                                   repeat=num_rpts,
                                   number=num_rpts,
                                   globals=globals())
alt_end = time.perf_counter()

alt_z = np.dot(alt_wb, alt_x1s)

alt_exec_time, alt_time_units = get_min_exec_time(alt_elapsed_times, num_iters)

print("----------- Alternate Computation of z = wb * x1s --------------\n")
print(f"                    WB.shape  = ({alt_wb.shape[0]}, {alt_wb.shape[1]})")
print(f"                    X1S.shape = ({alt_x1s.shape[0]}, {alt_x1s.shape[1]})")
print(f"                      Z.shape = ({alt_z.shape[0]}, {alt_z.shape[1]})\n")
print(f"Computation of z = wb*x1s, {w_rows * x_cols * num_iters} times, takes a minimum of : {alt_exec_time:.2f}" + alt_time_units + "\n")

perf_increase = True

if trad_exec_time >= alt_exec_time:

    perf_pcnt = ( (trad_exec_time - alt_exec_time) / trad_exec_time ) * 100

else:

    perf_pcnt = ( (alt_exec_time - trad_exec_time) / trad_exec_time ) * 100

    perf_increase = False

if np.array_equal(trad_z, alt_z):

    print("Logit z computed traditionally and alternatively are both equal.\n")

    if perf_increase:

        print(f"Performance increase: {perf_pcnt:.2f}%.\n")

    else:

        print(f"Performance decrease: {perf_pcnt:.2f}%.\n")

else:

    print(f"\033[31mError:\033[0m Logit z computed traditionally and alternatively are \033[31mNOT\033[0m both equal.\n")

# def color(text, code):
#     return f"\033[{code}m{text}\033[0m"
#
# print(color("Warning", 33))