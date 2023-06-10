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

void cudaXMemcpy3D(void* dst, size_t dpitch, void* src, size_t spitch,
                   size_t width, size_t height, size_t depth, cudaMemcpyKind kind)
{
    cudaExtent extent = make_cudaExtent(width, height, depth);

    cudaMemcpy3DParms params = {0};
    params.srcPtr.ptr = src;
    params.srcPtr.pitch = spitch;
    params.srcPtr.xsize = width;
    params.srcPtr.ysize = height;
    params.dstPtr.ptr = dst;
    params.dstPtr.pitch = dpitch;
    params.dstPtr.xsize = width;
    params.dstPtr.ysize = height;
    params.extent = extent;
    params.kind = kind;

    gpuErrorCheck(cudaMemcpy3D(&params));
}


void cudaXMalloc3D(void** devPtr, size_t elem_size, size_t* pitch, size_t w, size_t h, size_t d)
{
    cudaExtent extent = make_cudaExtent(w * elem_size, h, d);
    cudaPitchedPtr devicePtr;
    gpuErrorCheck(cudaMalloc3D(&devicePtr, extent));
    *devPtr = devicePtr.ptr;
    *pitch = devicePtr.pitch;
}


void cudaXFree(void* devPtr)
{
    gpuErrorCheck(cudaFree(devPtr));
}

void cudaXFreeHost(void* devPtr)
{
    gpuErrorCheck(cudaFreeHost(devPtr));
}
