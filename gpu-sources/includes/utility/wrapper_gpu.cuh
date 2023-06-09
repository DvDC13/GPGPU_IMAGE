#pragma once

#include "error.cuh"

void cudaXCalloc(void** devPtr, size_t size)
{
    gpuErrorCheck(cudaMalloc(devPtr, size));
    gpuErrorCheck(cudaMemset(*devPtr, 0, size));
}

void cudaXMallocHost(void** devPtr, size_t size)
{
    gpuErrorCheck(cudaMallocHost(devPtr, size));
}

void cudaXMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)
{
    gpuErrorCheck(cudaMemcpy(dst, src, count, kind));
}

void cudaXMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    gpuErrorCheck(cudaMemcpyAsync(dst, src, count, kind, stream));
}

void cudaXCallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
    gpuErrorCheck(cudaMallocPitch(devPtr, pitch, width, height));
    gpuErrorCheck(cudaMemset(*devPtr, 0, width * height));
}

void cudaXMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                   size_t width, size_t height, enum cudaMemcpyKind kind)
{
    gpuErrorCheck(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind));
}

void cudaXFree(void* devPtr)
{
    gpuErrorCheck(cudaFree(devPtr));
}

void cudaXFreeHost(void* devPtr)
{
    gpuErrorCheck(cudaFreeHost(devPtr));
}