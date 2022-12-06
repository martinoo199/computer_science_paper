import json
from statistics import mean
from LSH_Martini import experiment

runs = 5 # The amount of repeats you want to for each bucket
bucket_list = [1,2,5,10,20,30,50,100,200] # Buckets you want to test. Num_hashes = 600 so only choose 600 % bucketlist == 0
# Going bigger than 100 takes a lot of time. I suspect the key-matching is slowing the algorithm down a lot.
#I disabled keymatching as it didn't seem to make much of a difference and cost A LOT of CPU time. Now buckets can go as big as you want. 



pair_quality = []
pair_completeness = []
f1_star = []
precision = []
recall = []
f1 = []
fraction_comp = []

for idx, b in enumerate(bucket_list):
    sub_quality = []
    sub_completeness = []
    sub_f1_star = []
    sub_precision = []
    sub_recall = []
    sub_f1 = []
    sub_fraction_comp = []

    for _ in range(0, runs):
        result = experiment(buckets=b)
        sub_quality.append(result[0])
        sub_completeness.append(result[1])
        sub_f1_star.append(result[2])
        sub_precision.append(result[3])
        sub_recall.append(result[4])
        sub_f1.append((result[5]))
        sub_fraction_comp.append(result[6])

    pair_quality.append(sub_quality)
    pair_completeness.append(sub_completeness)
    f1_star.append(sub_f1_star)
    precision.append(sub_precision)
    recall.append(sub_recall)
    f1.append(sub_f1)
    fraction_comp.append(sub_fraction_comp)

def mean_list(list_to_avg):
    """"Compute the mean of a list"""
    list_avg = []
    for list_idx in list_to_avg:
        list_avg.append(mean(list_idx))
    return list_avg


#Take the mean value of all the bootstraps for each different bucket.
f1_mean = mean_list(f1)
f1_star_mean = mean_list(f1_star)
pair_completeness_mean = mean_list(pair_completeness)
pair_quality_mean = mean_list(pair_quality)
precision_mean = mean_list(precision)
recall_mean = mean_list(recall)
fraction_comp_mean = mean_list(fraction_comp)

print(f"fraction comp mean = {fraction_comp_mean}")
print(f'fraction comp without mean is: {fraction_comp}')

print(result)
print(f"pair_quality is {pair_quality_mean}\n"
      f"pair completeness is {pair_completeness_mean}\n"
      f"F1* is {f1_star_mean}")
print(f"precision is {precision_mean}\n"
      f"recall is {recall_mean}\n"
      f"f1 is {f1_mean}"
      f"F1 values are {f1}")


#TODO write the lists to a file so we don't have to keep running the program


# Plotting
#TODO You should actually plot as function of fraction of comparisons but I am tired. I have already included the fraction_comp
#TODO variable which returns the total amount of possible comparisons (N^2) / no_candidate_pairs

import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=2, nrows=2)

axes[0,0].plot(fraction_comp_mean,pair_completeness_mean)
# axes[0,0].set_title("Pair Completeness as function of Buckets")
axes[0,0].set(xlabel="fraction of comparisons" , ylabel="pair completeness")

axes[0,1].plot(fraction_comp_mean,pair_quality_mean)
# axes[0,1].set_title("Pair Quality as function of Buckets")
axes[0,1].set(xlabel="fraction of comparisons" , ylabel="pair quality")

axes[1,0].plot(fraction_comp_mean,f1_star_mean)
# axes[1,0].set_title("F1*")
axes[1,0].set(xlabel="fraction of comparisons" , ylabel="F1*")

axes[1,1].plot(fraction_comp_mean,f1_mean)
# axes[1,1].set_title("F1 as function of Buckets")
axes[1,1].set(xlabel="fraction of comparisons" , ylabel="F1")
fig.tight_layout()
plt.show()

x = fraction_comp_mean
y = pair_completeness_mean
plt.plot(x,y)
plt.xlabel("fraction of comparisons")
plt.ylabel("pair completeness")
# plt.title("Pair Completeness as function of Buckets")
plt.show()
#
# x = fraction_comp_mean
# y = pair_quality_mean
# plt.plot(x,y)
# plt.xlabel("fraction comp")
# plt.ylabel("pair completeness")
# plt.title("Pair Completeness as function of Buckets")
# plt.show()
#
x = fraction_comp_mean
y = pair_quality_mean
plt.plot(x,y)
plt.xlabel("fraction of comparisons")
plt.ylabel("pair quality")
# plt.title("Pair Quality as function of Buckets")
plt.show()

x = fraction_comp_mean
y = f1_star_mean
plt.plot(x,y)
plt.xlabel("fraction of comparisons")
plt.ylabel("F1*")
# plt.title("F1* as function of Buckets")
plt.show()
#
x = fraction_comp_mean
y = f1_mean
plt.plot(x,y)
plt.xlabel("fraction of comparisons")
plt.ylabel("F1")
# plt.title("F1 as function of Buckets")
plt.show()



