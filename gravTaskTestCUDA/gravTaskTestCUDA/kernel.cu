#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>

#define GravConst 6.674e-11
#define THREAD_COUNT 128

__host__ void cuAssert(cudaError_t err, std::string msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " " << err << std::endl;
        exit(1);
    }
}

__device__ void calcForces(float4 bi, float4 bj, float3& fi) {
    float3 r;
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;
    float dist = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
    float F = (GravConst * bi.w * bj.w) / (dist * dist);
    fi.x += F * r.x / dist;
    fi.y += F * r.y / dist;
    fi.z += F * r.z / dist;
}

__global__ void nextStep(float4* pos, float3* v, float3* a, float4* posNew, float3* vNew, float3* aNew, int& size, int& dt) {
    __shared__ float4 sh_pos[THREAD_COUNT];
    float4 myPosition;
    float3 res = { 0.0f, 0.0f, 0.0f };
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    myPosition = pos[idx];

    //сохранение точек с других блоков
    for (int i = 0; i < gridDim.x; i++) {
        if (i * blockDim.x + threadIdx.x) {
            sh_pos[threadIdx.x] = pos[i * blockDim.x + threadIdx.x];
        }
        __syncthreads();
        if (idx < size) {
            for (int k = 0; k < blockDim.x; k++) {
                calcForces(myPosition, sh_pos[k], res);
            }
        }
        __syncthreads();
    }
    if (idx < size) {
        posNew[idx].x = pos[idx].x + v[idx].x * dt + (a[idx].x * dt * dt) / 2;
        posNew[idx].y = pos[idx].y + v[idx].y * dt + (a[idx].y * dt * dt) / 2;
        posNew[idx].z = pos[idx].z + v[idx].z * dt + (a[idx].z * dt * dt) / 2;

        posNew[idx].w = pos[idx].w;

        vNew[idx].x = v[idx].x + a[idx].x * dt;
        vNew[idx].y = v[idx].y + a[idx].y * dt;
        vNew[idx].z = v[idx].z + a[idx].x * dt;

        aNew[idx].x = res.x / pos[idx].w;
        aNew[idx].y = res.y / pos[idx].w;
        aNew[idx].z = res.z / pos[idx].w;
    }
}

void readPointsData(const std::string& name, float4& point, int& size) {
    std::ifstream infile(name);

    for (int i = 0; i < size; i++) {
        float x, y, z, m;
        infile >> x >> y >> z >> m;
        point.x = x;
        point.y = y;
        point.z = z;
        point.w = m;
    }
}


void readData(const std::string& name, int& size, int& iterations, int& dt) {

    std::ifstream in(name);
    in >> size >> iterations >> dt;
}

void writeFile(std::ofstream& outfile, float4& point, int& size) {

    for (int i = 0; i < size; i++) {
        outfile << point.x << ' ' << point.y << ' ' << point.z << ' ' << point.w << "\t\t";
    }
    outfile << "\n";

}

int main()
{
    std::ofstream outfile("output.txt", std::ios::trunc);
    int size, iterations, dt;

    readData("Data.txt", size, iterations, dt);
    
    //ћассивы под точки на хосте
    float4* host_pos = (float4*)malloc(sizeof(float4) * size);
    float3* host_v = (float3*)malloc(sizeof(float3) * size);
    float3* host_a = (float3*)malloc(sizeof(float3) * size);
    memset(host_v, 0, size * sizeof(float3));
    memset(host_a, 0, size * sizeof(float3));

    readPointsData("inputDataPoint.txt", *host_pos, size);
    
    //–абота с данными на карточке
    float4* cuArr_pos = NULL;
    float3* cuArr_v = NULL;
    float3* cuArr_a = NULL;

    float4* cuResArr_pos = NULL;
    float3* cuResArr_v = NULL;
    float3* cuResArr_a = NULL;

    cuAssert(cudaMalloc((void**)&cuArr_pos, size * sizeof(float4)), "CudaMalloc cuArr_pos");
    cuAssert(cudaMalloc((void**)&cuArr_v, size * sizeof(float3)), "CudaMalloc cuArr_v");
    cuAssert(cudaMalloc((void**)&cuArr_a, size * sizeof(float3)), "CudaMalloc cuArr_a");

    cuAssert(cudaMalloc((void**)&cuResArr_pos, size * sizeof(float4)), "CudaMalloc cuResArr_pos");
    cuAssert(cudaMalloc((void**)&cuResArr_v, size * sizeof(float3)), "CudaMalloc cuResArr_v");
    cuAssert(cudaMalloc((void**)&cuResArr_a, size * sizeof(float3)), "CudaMalloc cuResArr_a");

    cuAssert(cudaMemcpy(cuArr_pos, host_pos, size * sizeof(float4), cudaMemcpyHostToDevice), "cudaMemcpy_pos");
    cuAssert(cudaMemcpy(cuArr_v, host_v, size * sizeof(float3), cudaMemcpyHostToDevice), "cudaMemcpy_v");
    cuAssert(cudaMemcpy(cuArr_a, host_a, size * sizeof(float3), cudaMemcpyHostToDevice), "cudaMemcpy_a");

    dim3 gridDim, blockDim;
    blockDim.x = THREAD_COUNT;
    gridDim.x = (size + blockDim.x - 1) / blockDim.x;

    auto now = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        nextStep << < gridDim, blockDim >> > (cuArr_pos, cuArr_v, cuArr_a, cuResArr_pos, cuResArr_v, cuResArr_a, size, dt);

        /*cuAssert(cudaDeviceSynchronize(), "CudaSyncronize");*/

        cuAssert(cudaMemcpy(host_pos, cuResArr_pos, size * sizeof(float4), cudaMemcpyDeviceToHost), "cudaMemcpy DTH_pos");
        cuAssert(cudaMemcpy(host_v, cuResArr_v, size * sizeof(float3), cudaMemcpyDeviceToHost), "cudaMemcpy DTH_v");
        cuAssert(cudaMemcpy(host_a, cuResArr_a, size * sizeof(float3), cudaMemcpyDeviceToHost), "cudaMemcpy DTH_a");

        writeFile(outfile, *host_pos ,size);

        if (i != 0) {
            float4* tmpArr_pos = cuArr_pos;
            cuArr_pos = cuResArr_pos;
            cuResArr_pos = tmpArr_pos;
            float3* tmpArr_a= cuArr_a;
            cuArr_a = cuResArr_a;
            cuResArr_a = tmpArr_a;
            float3* tmpArr_v = cuArr_v;
            cuArr_v = cuResArr_v;
            cuResArr_v = tmpArr_v;
        }
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - now);
    std::cerr << "Time : " << elapsed.count() << "s.\n";

    cudaFree(cuResArr_a);
    cudaFree(cuResArr_v);
    cudaFree(cuResArr_pos);

    cudaFree(cuArr_a);
    cudaFree(cuArr_v);
    cudaFree(cuArr_pos);

    free(host_a);
    free(host_v);
    free(host_pos);

    outfile.close();

    return 0;
}