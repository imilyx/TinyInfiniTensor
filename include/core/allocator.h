#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {
  class Allocator
  {
  private:
    Runtime runtime;

    /*
    used：当前逻辑上仍被 Tensor 占用的字节数
    peak：曾经达到过的最大逻辑地址，也就是最终真实申请的字节数
    freeBlocks：中间可以复用的空闲区间
    ptr：真正申请的大块内存；getPtr() 前为 nullptr
    */

    size_t used;

    size_t peak;

    size_t alignment;

    // pointer to the memory actually allocated
    void *ptr;

    // =================================== 作业 ===================================
    // TODO：可能需要设计一个数据结构来存储free block，以便于管理和合并
    // HINT: 可以使用一个 map 来存储 free block，key 为 block 的起始/结尾地址，value 为 block 的大小
    // =================================== 作业 ===================================
    // key: free block start offset, value: free block size
    std::map<size_t, size_t> freeBlocks;

  public:
    Allocator(Runtime runtime);

    virtual ~Allocator();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    size_t alloc(size_t size);

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    void *getPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    size_t getAlignedSize(size_t size);
  };
}
