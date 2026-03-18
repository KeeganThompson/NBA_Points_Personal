import Processor as pr
import numpy as np

proc = pr.Processor()

# maps name to dataset
map = {
    "Jose Alvarado": proc.jose_alvarado,
    "Bam Adebayo": proc.bam_adebayo,
    "Trae Young": proc.trae_young,
    "Kawhi Leonard": proc.kawhi_leonard,
    "Derrick White": proc.derrick_white,
}

def compute_avg_SVR(name, number=5):
    m_1, m_2, r_f = 0, 0, 0
    x, y, z, total = proc.linearSVR(map[name])
    for i in range(number):
        m1, m2, r, yp = proc.linearSVR(map[name])
        m_1 += m1
        m_2 += m2
        r_f += r
        total = total + yp
    total = (total / (number + 1)).astype(int)
    return m1, m2, r, total

def compute_avg_RF(name, number=5):
    m_1, m_2, r_f = 0, 0, 0
    for i in range(number):
        m1, m2, r = proc.random_forest_boosted_points(map[name])
        m_1 += m1
        m_2 += m2
        r_f += r
    return m1, m2, r

def compute_avg_GBT(name, number=5):
    m_1, m_2, r_f = 0, 0, 0
    for i in range(number):
        m1, m2, r = proc.gradient_tree(map[name])
        m_1 += m1
        m_2 += m2
        r_f += r

    return m1, m2, r

def custom_avg(arr):
    sum = 0
    for x, num in arr:
        sum += num

    return sum / 5

# SVR_results_m1 = [(x, compute_avg_SVR(x)[0]) for x in map.keys()]
# RF_results_m1 = [(x, compute_avg_RF(x)[0]) for x in map.keys()]
# GBT_results_m1 = [(x, compute_avg_GBT(x)[0]) for x in map.keys()]

# SVR_results_m2 = [(x, compute_avg_SVR(x)[1]) for x in map.keys()]
# RF_results_m2 = [(x, compute_avg_RF(x)[1]) for x in map.keys()]
# GBT_results_m2 = [(x, compute_avg_GBT(x)[1]) for x in map.keys()]

# SVR_results_r = [(x, compute_avg_SVR(x)[2]) for x in map.keys()]
# RF_results_r = [(x, compute_avg_RF(x)[2]) for x in map.keys()]
# GBT_results_r = [(x, compute_avg_GBT(x)[2]) for x in map.keys()]

# print("- - - - - - - - - MEAN ABSOLUTE ERROR - - - - - - - - -")

# print("\n SVR \n")
# print(SVR_results_m1)
# print(f"\nAverage: {custom_avg(SVR_results_m1)}")

# print("\n RF \n")
# print(RF_results_m1)
# print(f"\n Average: {custom_avg(RF_results_m1)}")

# print("\n GBT \n")
# print(GBT_results_m1)
# print(f"\n Average: {custom_avg(GBT_results_m1)}")

# print("- - - - - - - - - MEAN SQUARED ERROR - - - - - - - - -")

# print("\n SVR \n")
# print(SVR_results_m2)
# print(f"\n Average: {custom_avg(SVR_results_m2)}")

# print("\n RF \n")
# print(RF_results_m2)
# print(f"\n Average: {custom_avg(RF_results_m2)}")

# print("\n GBT \n")
# print(GBT_results_m2)
# print(f"\n Average: {custom_avg(GBT_results_m2)}")

# print("- - - - - - - - - R - - - - - - - - -")

# print("\n SVR \n")
# print(SVR_results_r)
# print(f"\nAverage: {custom_avg(SVR_results_r)}")

# print("\n RF \n")
# print(RF_results_r)
# print(f"\n Average: {custom_avg(RF_results_r)}")

# print("\n GBT \n")
# print(GBT_results_r)
# print(f"\n Average: {custom_avg(GBT_results_r)}")

print(compute_avg_SVR("Trae Young", 100))