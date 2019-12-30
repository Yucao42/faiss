# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python2

from __future__ import print_function
import os
import time
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    graphical_output = True
except ImportError:
    graphical_output = False

import faiss

#################################################################
# Small I/O functions
#################################################################

def ivecs_read(fname):
    f = open(fname)
    d, = np.fromfile(f, count = 1, dtype = 'int32')
    sz = os.stat(fname).st_size
    assert sz % (4 * (d + 1)) == 0
    n = sz // (4 * (d + 1))
    f.seek(0)
    a = np.fromfile(f, count = n * (d +1), dtype = 'int32').reshape(n, d + 1)
    return a[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def plot_OperatingPoints(ops, nq, **kwargs):
    ops = ops.optimal_pts
    n = ops.size() * 2 - 1
    print([ops.at( i      // 2).perf for i in range(n)],
                [ops.at((i + 1) // 2).t / nq * 1000 for i in range(n)])
    pyplot.plot([ops.at( i      // 2).perf for i in range(n)],
                [ops.at((i + 1) // 2).t / nq * 1000 for i in range(n)],
                **kwargs)


#################################################################
# prepare common data for all indexes
#################################################################



t0 = time.time()

print("load data")

xt = fvecs_read("sift1M/sift_learn.fvecs")
xb = fvecs_read("sift1M/sift_base.fvecs")
xq = fvecs_read("sift1M/sift_query.fvecs")
print(xq[0])

d = xt.shape[1]

print("load GT")

gt = ivecs_read("sift1M/sift_groundtruth.ivecs")
gt = gt.astype('int64')
k = gt.shape[1]

print("prepare criterion")

# criterion = 1-recall at 1
crit = faiss.OneRecallAtRCriterion(xq.shape[0], 1)
crit.set_groundtruth(None, gt)
crit.nnn = k

# indexes that are useful when there is no limitation on memory usage
unlimited_mem_keys = [
    "IMI2x10,Flat", "IMI2x11,Flat",
    "IVF4096,Flat", "IVF16384,Flat",
    "PCA64,IMI2x10,Flat"]

# memory limited to 16 bytes / vector
keys_mem_16 = [
    'IMI2x10,PQ16', 'IVF4096,PQ16',
    'IMI2x10,PQ8+8', 'OPQ16_64,IMI2x10,PQ16'
    ]

# limited to 32 bytes / vector
keys_mem_32 = [
    'IMI2x10,PQ32', 'IVF4096,PQ32', 'IVF16384,PQ32',
    'IMI2x10,PQ16+16',
    'OPQ32,IVF4096,PQ32', 'IVF4096,PQ16+16', 'OPQ16,IMI2x10,PQ16+16'
    ]

# indexes that can run on the GPU
keys_gpu = [
    "PCA64,IVF4096,Flat",
    "PCA64,Flat", "Flat", "IVF4096,Flat", "IVF16384,Flat",
    "IVF4096,PQ32"]


keys_to_test = unlimited_mem_keys
keys_to_test = keys_gpu
# keys_to_test = ["Flat", "IVF4096,Flat"]
use_gpu = True
# use_gpu = False


if use_gpu:
    # if this fails, it means that the GPU version was not comp
    assert faiss.StandardGpuResources, \
        "FAISS was not compiled with GPU support, or loading _swigfaiss_gpu.so failed"
    res = faiss.StandardGpuResources()
    dev_no = 1

# remember results from other index types
op_per_key = []


# keep track of optimal operating points seen so far
op = faiss.OperatingPoints()

training_time_cpu = {}
indexing_time_cpu = {}
training_time_gpu = {}
indexing_time_gpu = {}

for use_gpu in range(2):
    for index_key in keys_to_test:
    
        print("============ key", index_key)
    
        # make the index described by the key
        index = faiss.index_factory(d, index_key)
    
    
        if use_gpu:
            # transfer to GPU (may be partial)
            index = faiss.index_cpu_to_gpu(res, dev_no, index)
            params = faiss.GpuParameterSpace()
        else:
            params = faiss.ParameterSpace()
    
        params.initialize(index)
    
        print("[%.3f s] train & add" % (time.time() - t0))
    
        t1 = time.time()
        index.train(xt)
        if use_gpu:
            training_time_gpu[index_key] = time.time() - t1
        else:
            training_time_cpu[index_key] = time.time() - t1
        t1 = time.time()
        index.add(xb)
        if use_gpu:
            indexing_time_gpu[index_key] = time.time() - t1
        else:
            indexing_time_cpu[index_key] = time.time() - t1
    
        print("[%.3f s] explore op points" % (time.time() - t0))
    
        # find operating points for this index
        opi = params.explore(index, xq, crit)
    
        print("[%.3f s] result operating points:" % (time.time() - t0))
        opi.display()
    
        # update best operating points so far
        op.merge_with(opi, index_key + " ")
    
        op_per_key.append((index_key, opi))
    
        # if graphical_output:
        #     # graphical output (to tmp/ subdirectory)
    
        #     fig = pyplot.figure(figsize=(12, 9))
        #     pyplot.xlabel("1-recall at 1")
        #     pyplot.ylabel("search time (ms/query, %d threads)" % faiss.omp_get_max_threads())
        #     pyplot.gca().set_yscale('log')
        #     pyplot.grid()
        #     for i2, opi2 in op_per_key:
        #         plot_OperatingPoints(opi2, crit.nq, label = i2, marker = 'o')
        #     # plot_OperatingPoints(op, crit.nq, label = 'best', marker = 'o', color = 'r')
        #     pyplot.legend(loc=2)
        #     fig.savefig(f'tmp/demo_auto_tune_gpu{use_gpu}.png')


# Draw time distribution
import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = len(indexing_time_cpu)
cpu_time = [v for k, v in training_time_cpu.items()]
gpu_time = [v for k, v in training_time_gpu.items()]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, cpu_time, bar_width,
        alpha=opacity,
        color='r',
        label='CPU Time')

rects2 = plt.bar(index + bar_width, gpu_time, bar_width,
        alpha=opacity,
        color='g',
        label='GPU Time')

plt.xlabel('Method')
plt.ylabel('Time(s)')
plt.title('Training Time on 100K 128-d SIFT Features')
plt.xticks(index + bar_width, [k for k, v in training_time_cpu.items()])
plt.legend()

# pyplot.gca().set_yscale('log')
pyplot.legend(loc=2)
fig.savefig(f'tmp/training_time_gpu{use_gpu}.png')

import matplotlib.pyplot as plt

# data to plot
n_groups = len(indexing_time_cpu)
cpu_time = [v for k, v in indexing_time_cpu.items()]
gpu_time = [v for k, v in indexing_time_gpu.items()]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, cpu_time, bar_width,
        alpha=opacity,
        color='r',
        label='CPU Time')

rects2 = plt.bar(index + bar_width, gpu_time, bar_width,
        alpha=opacity,
        color='g',
        label='GPU Time')

plt.xlabel('Method')
plt.ylabel('Time(s)')
plt.title('Indexing Time on 1M 128-d SIFT Features')
plt.xticks(index + bar_width, [k for k, v in training_time_cpu.items()])
plt.legend()

# pyplot.gca().set_yscale('log')
pyplot.legend(loc=2)
fig.savefig(f'tmp/indexing_time_gpu{use_gpu}.png')
print("[%.3f s] final result:" % (time.time() - t0))

op.display()
