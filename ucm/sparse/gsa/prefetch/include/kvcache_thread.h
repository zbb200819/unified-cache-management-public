#ifndef ATB_KV_CACHE_THREAD_H
#define ATB_KV_CACHE_THREAD_H
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <kvcache_log.h>
#include <map>
#include <mutex>
#include <omp.h>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>
#include <queue>


class ThreadPool {
public:
    static ThreadPool* GetInst()
    {
        static ThreadPool pool(16);
        return &pool;
    }

    ~ThreadPool() 
    {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) { worker.join(); }
    }

    template <class F, class... Args>
    auto Enqueue(F&& f,
                    Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex);

            condition.wait(lock, [this] {
                if (!(activeThreads < maxThreads || tasks.size() < maxThreads * 2)) {
                    std::cout << "Need wait: " << activeThreads << " " << tasks.size() << std::endl;
                }
                return (activeThreads < maxThreads || tasks.size() < maxThreads * 2);
            });
            // don't allow enqueueing after stopping the pool
            if (stop) { throw std::runtime_error("enqueue on stopped ThreadPool"); }

            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    size_t GetActiveThreads() const { return activeThreads; }

private:
    ThreadPool(size_t threadCount) : stop(false), maxThreads(threadCount)
    {
        for (size_t i = 0; i < maxThreads; i++) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock,
                                            [this] { return this->stop || !this->tasks.empty(); });

                        if (this->stop && this->tasks.empty()) { return; }

                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                        ++activeThreads;
                    }

                    task();
                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        --activeThreads;
                        condition.notify_all();
                    }
                }
            });
        }
    }
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    mutable std::mutex queueMutex;
    bool stop;
    std::condition_variable condition;
    std::atomic<size_t> activeThreads{0};
    size_t maxThreads;
};

#endif  // ATB_KV_CACHE_THREAD_H