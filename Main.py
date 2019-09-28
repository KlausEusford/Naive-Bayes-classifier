import numpy as np

# build em-model
def em_algorithm(data, valid_count, total_count, eps=1e-4):
    valid_data = data[0:valid_count]   # the number of effective examples
    avg = np.sum(valid_data) / total_count  #hidden parameter
    theta = np.sum(np.square(valid_data)) / total_count - avg
    while True:
        s1 = np.sum(valid_data) + avg * (total_count - valid_count)
        s2 = np.sum(np.square(valid_data)) + (avg * avg + theta) * (total_count - valid_count)
        new_avg = s1 / total_count
        new_theta = s2 / total_count - new_avg * new_avg
        if new_avg - avg <= eps and new_theta - theta <= eps:
            break
        else:
            avg, theta = new_avg, new_theta
    return avg, theta


def elderly_man(dtype1, dtype2, latent_idx):
    avg, var = [], []
    for idx in range(latent_idx):
# dtype1,dtype2  Represents one dimension in multidimensional data
        dim_type1, dim_type2 = dtype1[:, idx], dtype2[:, idx]
        avg.append([np.average(dim_type1), np.average(dim_type2)])
        var.append([np.var(dim_type1), np.var(dim_type2)])
#Estimation of mean and variance using EM algorithm
    em_avg_type1, em_var_type1 = em_algorithm(data_type1[:8, latent_idx], 4, 8)
    em_avg_type2, em_var_type2 = em_algorithm(data_type2[:8, latent_idx], 4, 8)
#Add the estimated mean and variance to the array and return
    avg.append([em_avg_type1, em_avg_type2])
    var.append([em_var_type1, em_var_type2])
    return avg, var

#gaussian
def calc_gaussian(x, avg, var):
    t = 1.0 / np.sqrt(2 * np.pi * var)
    return t * np.exp(-np.square(x - avg) / (2.0 * var))


if __name__ == '__main__':
    data_str = open('Data/watermelon2_0_En.csv').readlines()
    data_type1 = np.ndarray([8, 6], np.float32)
    data_type2 = np.ndarray([9, 6], np.float32)
    for idx in range(8):
        data_type1[idx] = data_str[idx].strip('\n').split(',')[0:6]
    for idx in range(8, 17):
        data_type2[idx - 8] = data_str[idx].strip('\n').split(',')[0:6]
    a, v = elderly_man(data_type1[:8], data_type2[:8], 3)
# Build test data sets, correct_times represent the exact number of data bars of test results
    data_test = np.concatenate((data_type1[8:], data_type2[8:]))
    correct_times = 0
    for data_idx in range(len(data_test)):
        data = data_test[data_idx]
#The data set has the same two types of data, so the prior probability is 0.5.
        val_type1, val_type2 = 0.5, 0.5
        for idx in range(4):
            val_type1 *= calc_gaussian(data[idx], a[idx][0], v[idx][0])
            val_type2 *= calc_gaussian(data[idx], a[idx][1], v[idx][1])
        if val_type1 > val_type2 and data_idx < 10:
            correct_times += 1
        elif val_type1 < val_type2 and data_idx >= 10:
            correct_times += 1
        print("Number: %2d, Type1: %f, Type2: %f" % (data_idx + 1, val_type1, val_type2))
    print("Accuracy: %.1f%%" % (correct_times * 5))
