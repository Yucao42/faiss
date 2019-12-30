/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define TIME_FUNC(expression_, method_, job_) \
    start = std::chrono::steady_clock::now(); \
    expression_                               \
    end = std::chrono::steady_clock::now();   \
    std::cout << "[TIME] for job_ using method_  " << " in microseconds : " \
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() \
         << " us" << std::endl; \

#define CPU_KNN_TEST 0     
#define GPU_KNN_TEST 0
#define CPU_ANN_TEST 0     
#define GPU_ANN_TEST 1
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>


int main() {
// GPU
    int d = 64;                            // dimension
    printf("Size of long: %d\n", sizeof(long));
    long nb = 200 * 1000* 1000;                       // database size
    int nq = 1000;                        // nb of queries
    int nlist =  100000;
    int m = 8;                             // bytes per vector

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    for(long i = 0; i < nb; i++) {
        for(long j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    int ngpus = faiss::gpu::getNumDevices();
    printf("Number of GPUs: %d\n", ngpus);
    int k = 4;

    std::vector<faiss::gpu::GpuResources*> res;
    std::vector<int> devs;
    for(int i = 0; i < ngpus; i++) {
        res.push_back(new faiss::gpu::StandardGpuResources);
        devs.push_back(i);
    }
    faiss::gpu::GpuMultipleClonerOptions option;
    option.shard = true;

#if GPU_KNN_TEST >0
    faiss::IndexFlatL2 cpu_index(d);

    faiss::Index *gpu_index =
        faiss::gpu::index_cpu_to_gpu_multiple(
            res,
            devs,
            &cpu_index,
            &option
        );

    printf("is_trained = %s\n", gpu_index->is_trained ? "true" : "false");
    start = std::chrono::steady_clock::now();
    gpu_index->add(nb, xb);  // add vectors to the index
    end = std::chrono::steady_clock::now();
    printf("ntotal = %ld\n", gpu_index->ntotal);
    std::cout << "[ADD TIME] [GPU BF] add " << nb << " records in microseconds : "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
         << " us" << std::endl;


    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        start = std::chrono::steady_clock::now();
        gpu_index->search(nq, xq, k, D, I);
        end = std::chrono::steady_clock::now();
        printf("ntotal = %ld\n", gpu_index->ntotal);
        std::cout << "[SEARCH TIME] [GPU BF] search " << nq << " records in microseconds : "
             << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
             << " us" << std::endl;

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    delete gpu_index;

#endif

// CPU
#if CPU_KNN_TEST > 0
    faiss::IndexFlatL2 index(d);           // call constructor
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    start = std::chrono::steady_clock::now();
    index.add(nb, xb);                     // add vectors to the index
    end = std::chrono::steady_clock::now();
    printf("ntotal = %ld\n", gpu_index->ntotal);
    std::cout << "[ADD TIME] [CPU BF] add " << nb << " records in microseconds : "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
         << " us" << std::endl;
    printf("ntotal = %ld\n", index.ntotal);


    {       // sanity check: search 5 first vectors of xb
        long *I = new long[k * 5];
        float *D = new float[k * 5];

        index.search(5, xb, k, D, I);

        // print results
        printf("I=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }


    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        auto start = std::chrono::steady_clock::now();
        index.search(nq, xq, k, D, I);
        auto end = std::chrono::steady_clock::now();
        printf("ntotal = %ld\n", gpu_index->ntotal);
        std::cout << "[SEARCH TIME] [GPU BF]Search time for queries " << 1000 << " records in microseconds : "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
         << " us" << std::endl;

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }
#endif


#if CPU_ANN_TEST > 0
    {
      faiss::IndexFlatL2 quantizer(d);       // the other index
      faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);
      // here we specify METRIC_L2, by default it performs inner-product search
      start = std::chrono::steady_clock::now();
      index.train(nb, xb);
      end = std::chrono::steady_clock::now();
      std::cout << "[TRAIN TIME] [CPU ANN] " << nb << " records in microseconds : "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
       << " us" << std::endl;


      start = std::chrono::steady_clock::now();
      index.add(nb, xb);
      end = std::chrono::steady_clock::now();
      std::cout << "[ADD TIME] [CPU ANN] " << 1000 << " records in microseconds : "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
       << " us" << std::endl;
      
      {       // search xq
          long *I = new long[k * nq];
          float *D = new float[k * nq];

          index.nprobe = 10;
          start = std::chrono::steady_clock::now();
          index.search(nq, xq, k, D, I);
          end = std::chrono::steady_clock::now();
          std::cout << "[SEARCH TIME] [CPU ANN] " << 1000 << " records in microseconds : "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
           << " us" << std::endl;

          printf("I=\n");
          for(int i = nq - 5; i < nq; i++) {
              for(int j = 0; j < k; j++)
                  printf("%5ld ", I[i * k + j]);
              printf("\n");
          }

          delete [] I;
          delete [] D;
      }
    }
#endif

#if GPU_ANN_TEST > 0
    {
      faiss::IndexFlatL2 quantizer(d);       // the other index
      faiss::IndexIVFPQ index_cpu(&quantizer, d, nlist, m, 8);
      // here we specify METRIC_L2, by default it performs inner-product search
      index_cpu.nprobe = 10;
      index_cpu.verbose = true;
      std::cout << "Before" << nb << " CASTing" << std::endl; 
      faiss::Index *gpu_index =
          faiss::gpu::index_cpu_to_gpu_multiple(
              res,
              devs,
              &index_cpu,
              &option
          );
      std::cout << "After" << nb << " CASTing" << std::endl; 
      
      // if(dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(gpu_index)){
      //   std::cout << " " << nb << " CASTED" << std::endl; 
      // } 
      // std::cout << " " << nb << " CASTED" << std::endl; 

      start = std::chrono::steady_clock::now();
      gpu_index->train(nb, xb);
      end = std::chrono::steady_clock::now();
      std::cout << "[TRAIN TIME] [GPU ANN] " << nb << " records in microseconds : "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
       << " us" << std::endl;
      //faiss::gpu::GpuIndexIVFFlat index_ivf(&res, d, nlist, faiss::METRIC_L2);

      start = std::chrono::steady_clock::now();
      gpu_index->add(nb, xb);
      end = std::chrono::steady_clock::now();
      std::cout << "[ADD TIME] [CPU ANN] " << 1000 << " records in microseconds : "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
       << " us" << std::endl;

      {       // search xq
          long *I = new long[k * nq];
          float *D = new float[k * nq];

          start = std::chrono::steady_clock::now();
          gpu_index->search(nq, xq, k, D, I);
          end = std::chrono::steady_clock::now();
          std::cout << "[SEARCH TIME] [CPU ANN] " << 1000 << " records in microseconds : "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
           << " us" << std::endl;

          printf("I=\n");
          for(int i = nq - 5; i < nq; i++) {
              for(int j = 0; j < k; j++)
                  printf("%5ld ", I[i * k + j]);
              printf("\n");
          }

          delete [] I;
          delete [] D;
      }
    }
#endif
    delete [] xb;
    delete [] xq;
    for(int i = 0; i < ngpus; i++) {
        delete res[i];
    }

    return 0;
}
