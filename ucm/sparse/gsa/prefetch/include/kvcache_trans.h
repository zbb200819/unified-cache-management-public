/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#ifndef ATB_KV_CACHE_TRANS_H
#define ATB_KV_CACHE_TRANS_H
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "cuda_sm_kernel.h"
#include <iostream>
#include <string>

using Ptr = uintptr_t;

class TransBackend {
public:
    static TransBackend* GetInst(cudaStream_t stream_)
    {
        static TransBackend trans(stream_);
        return &trans;
    }
    // 构造函数：内部新建一条 CUDA stream（非阻塞）
    TransBackend() : stream_(nullptr), own_stream_(true)
    {
        auto err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamCreateWithFlags failed: ") +
                                     cudaGetErrorString(err));
        }
    }

    // 外部传入已有 stream 指针（uint64）
    TransBackend(cudaStream_t stream_addr) : stream_(nullptr), own_stream_(false)
    {
        stream_ = stream_addr;
        if (!stream_) { throw std::runtime_error("TransBackend: null stream pointer"); }
    }

    ~TransBackend()
    {
        if (own_stream_ && stream_ != nullptr) { cudaStreamDestroy(stream_); }
    }

    // ============================
    //  Src -> dst
    //  src_ptrs_addr: device 上保存「device 指针数组」的那块内存地址
    //  dst_ptrs_addr: device 上保存「host pinned 指针数组」的那块内存地址
    //  size_bytes:    每个 block 拷贝的字节数（block_bytes）
    //  number:        block 个数（num_blocks）
    // ============================
    int32_t copy_trans(void** src_ptrs_addr, void** dst_ptrs_addr,
                    size_t size_bytes, size_t number)
    {
        // 3. 直接调用底层的 CudaSMCopyAsync
        auto err = UC::Trans::CudaSMCopyAsync(src_ptrs_addr,  // void* src[]
                                              dst_ptrs_addr,  // void* dst[]
                                              size_bytes, number, stream_);
        if (err != cudaSuccess) {
            std::cout << "CudaSMCopyAsync (H2D) failed: " << cudaGetErrorString(err) << std::endl;
            return -1;
        } else {
            return 0;
        }
    }

    // 同步内部 stream
    int32_t synchronize()
    {
        auto err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) {
            std::cout << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
            return -1;
        } else {
            return 0;
        }
    }

    // 导出内部 stream 的地址
    uint64_t get_stream_ptr() const { return reinterpret_cast<uint64_t>(stream_); }

    bool IsStreamIdle()
    {
        cudaError_t status = cudaStreamQuery(stream_);

        if (status == cudaSuccess) {
            return true;  // 流中所有操作已完成
        } else if (status == cudaErrorNotReady) {
            return false;  // 流中仍有操作在执行
        } else {
            // 其他错误情况
            std::cerr << "Error querying stream: " << cudaGetErrorString(status) << std::endl;
            return false;
        }
    }

private:
    cudaStream_t stream_;
    bool own_stream_;  // 是否自己创建并负责销毁 stream_
};

#endif  //ATB_KV_CACHE_TRANS    