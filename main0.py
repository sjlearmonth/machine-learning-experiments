import numpy as np
import timeit

# rows = 1000; cols = 500
rows = 100;
cols = 50

half_square_a = (np.random.random(size=(rows, cols)) + 1.1) * 10

whole_square_pure = (np.random.random(size=(rows, cols * 2)) + 1.1) * 10

half_square_b = np.ones((rows, cols))

whole_square_ones = np.ones((rows, cols*2))

unshuffled_whole_square_impure = np.append(half_square_a, half_square_b, axis=1)
flattened_whole_square_impure = unshuffled_whole_square_impure.flatten()
np.random.shuffle(flattened_whole_square_impure)
shuffled_whole_square_impure = flattened_whole_square_impure.reshape(unshuffled_whole_square_impure.shape)

# At this stage we have a numpy 2d array with shape (1000, 1000) with shuffled
# floats - half random, half 1.0.

# Create a function to loop through each element
print("\n")

def compute_products_opt(x, y):

    for el_x in x.ravel():

        for el_y in y.ravel():

            p = el_y if el_x == 1.0 else el_x * el_y

    return

def compute_products_non_opt(x, y):

    for el_x in x.ravel():

        for el_y in y.ravel():

            p = el_x * el_y

    return

elapsed_times_impure = timeit.repeat(stmt="compute_products_opt(shuffled_whole_square_impure, whole_square_pure)",
                                     setup="import numpy as np",
                                     repeat=1,
                                     number=5,
                                     globals=globals())

min_elapsed_time_impure = min(elapsed_times_impure)

print(f"Minimum elapsed execution time for impure products - opt: {min_elapsed_time_impure:.4f} secs.\n")

elapsed_times_pure = timeit.repeat(stmt="compute_products_opt(whole_square_pure, whole_square_pure)",
                                   setup="import numpy as np",
                                   repeat=1,
                                   number=5,
                                   globals=globals())

min_elapsed_time_pure = min(elapsed_times_pure)

print(f"Minimum elapsed execution time for pure products - opt: {min_elapsed_time_pure:.4f} secs.\n")

print(f"Performance increase: {((min_elapsed_time_pure - min_elapsed_time_impure) / min_elapsed_time_pure) * 100:.2f}%.\n")

elapsed_times_impure_non_opt = timeit.repeat(stmt="compute_products_non_opt(shuffled_whole_square_impure, whole_square_pure)",
                                   setup="import numpy as np",
                                   repeat=1,
                                   number=5,
                                   globals=globals())

min_elapsed_time_impure_non_opt = min(elapsed_times_impure_non_opt)

print(f"Minimum elapsed execution time for impure products - non opt: {min_elapsed_time_impure_non_opt:.4f} secs.\n")

elapsed_times_pure_non_opt = timeit.repeat(stmt="compute_products_non_opt(whole_square_pure, whole_square_pure)",
                                   setup="import numpy as np",
                                   repeat=1,
                                   number=5,
                                   globals=globals())

min_elapsed_time_pure_non_opt = min(elapsed_times_pure_non_opt)

print(f"Minimum elapsed execution time for pure products - non opt: {min_elapsed_time_pure_non_opt:.4f} secs.\n")

elapsed_times_ones_non_opt = timeit.repeat(stmt="compute_products_non_opt(whole_square_ones, whole_square_ones)",
                                   setup="import numpy as np",
                                   repeat=1,
                                   number=5,
                                   globals=globals())

min_elapsed_time_ones_non_opt = min(elapsed_times_ones_non_opt)

print(f"Minimum elapsed execution time for pure products - non opt: {min_elapsed_time_ones_non_opt:.4f} secs.\n")
