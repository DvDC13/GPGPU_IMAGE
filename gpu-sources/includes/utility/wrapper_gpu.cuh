#pragma once

#include "error.cuh"

void cudaXMalloc(void** devPtr, size_t size)
{
    gpuErrorCheck(cudaMalloc(devPtr, size));
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

void cudaXMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height)
{
    gpuErrorCheck(cudaMallocPitch(devPtr, pitch, width, height));
}

void cudaXMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                   size_t width, size_t height, enum cudaMemcpyKind kind)
{
    gpuErrorCheck(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind));
}

void cudaXMemcpy3D(cudaPitchedPtr dst, cudaPitchedPtr src, size_t depth, cudaMemcpyKind kind)
{
    cudaMemcpy3DParms params = {0};
    params.srcPtr = src;
    params.dstPtr = dst;
    params.extent = make_cudaExtent(src.pitch, src.ysize, depth);
    params.kind = kind;

    gpuErrorCheck(cudaMemcpy3D(&params));
}


void cudaXMalloc3D(cudaPitchedPtr* devPtr, size_t elem_size, size_t w, size_t h, size_t d)
{
    cudaExtent extent = make_cudaExtent(w * elem_size, h, d);
    cudaPitchedPtr devicePtr;
    gpuErrorCheck(cudaMalloc3D(devPtr, extent));
}


void cudaXFree(void* devPtr)
{
    gpuErrorCheck(cudaFree(devPtr));
}

void cudaXFreeHost(void* devPtr)
{
    gpuErrorCheck(cudaFreeHost(devPtr));
}
