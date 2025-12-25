#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <torch/torch.h>

struct PrefetchInfoLayer {
    uint32_t layerID;
    std::vector<std::string> reqIDList;
    std::vector<std::vector<int32_t>> topkList;
    std::vector<int> bsIndexList;
    std::vector<int> topkLenList;
};

class KvCacheSafeQueue {
public:
    KvCacheSafeQueue();
    ~KvCacheSafeQueue() = default;

    KvCacheSafeQueue(const KvCacheSafeQueue&) = delete;
    KvCacheSafeQueue& operator=(const KvCacheSafeQueue&) = delete;

    void push(PrefetchInfoLayer value);
    PrefetchInfoLayer pop();
    size_t size() const;
    bool empty() const;
    void stop();
    void clear();

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_condVar;
    std::queue<PrefetchInfoLayer> m_queue;
    std::atomic<bool> m_stopped;
};

#endif