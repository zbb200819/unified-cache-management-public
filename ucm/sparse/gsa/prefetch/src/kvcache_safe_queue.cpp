#include "kvcache_safe_queue.h"

KvCacheSafeQueue::KvCacheSafeQueue() : m_stopped(false) {}

void KvCacheSafeQueue::push(PrefetchInfoLayer value) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.push(std::move(value));
    m_condVar.notify_one();
}

PrefetchInfoLayer KvCacheSafeQueue::pop() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_condVar.wait(lock, [this] { 
        return !m_queue.empty() || m_stopped; 
    });
    PrefetchInfoLayer value = std::move(m_queue.front());
    m_queue.pop();
    return value;
}

size_t KvCacheSafeQueue::size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.size();
}

bool KvCacheSafeQueue::empty() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.empty();
}

void KvCacheSafeQueue::stop() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stopped = true;
    m_condVar.notify_all();
}

void KvCacheSafeQueue::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    while (!m_queue.empty()) {
        m_queue.pop();
    }
}