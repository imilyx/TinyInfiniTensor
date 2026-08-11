#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            const size_t addr = it->first;
            const size_t blockSize = it->second;

            if (blockSize < size) continue;

            freeBlocks.erase(it);
            if (blockSize > size)
                freeBlocks.emplace(addr + size, blockSize - size);
            
            used += size;
            return addr;
        }
        const size_t addr = peak;
        peak += size;
        used += size;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        
        IT_ASSERT(used >= size);
        used -= size;

        auto next = freeBlocks.lower_bound(addr);

        if (next != freeBlocks.begin()) {
            auto prev = std::prev(next);
            IT_ASSERT(prev->first + prev->second <= addr);
            if (prev->first + prev->second == addr) {
                addr = prev->first;
                size += prev->second;
                freeBlocks.erase(prev);
            }
        }
        next = freeBlocks.lower_bound(addr);  // 前一个块可能被删除，重新定位后继续
        if (next != freeBlocks.end()) {
            IT_ASSERT(addr + size <= next->first);
            if (addr + size == next->first) {
                size += next->second;
                freeBlocks.erase(next);
            }
        }
        if (addr + size == peak) {  // 若新加空闲块位于尾部，直接回退 peak，不必留在 freeBlocks 中
            peak = addr;
            return;
        }
        freeBlocks.emplace(addr, size);
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
