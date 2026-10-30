/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 */
#ifndef UCM_PREFETCH_TEST_MOCK_STORE_V1_H
#define UCM_PREFETCH_TEST_MOCK_STORE_V1_H

#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "status/status.h"
#include "type/dictionary.h"
#include "type/types.h"
#include "ucmstore_v1.h"

namespace ucmprefetch_test {

/*
 * Minimal, configurable mock of UC::StoreV1.  It records every call and lets
 * tests inject failures / canned host-buffer payloads so the prefetch code
 * paths that depend on mStoreV1_ (LastChunkMove, UCMTensorMerge, indexer load,
 * UCMKVPrefetch) can be driven deterministically without a real store.
 */
class MockStoreV1 : public UC::StoreV1 {
public:
    struct CallRecord {
        std::string name;
        UC::Detail::TaskDesc task;
        UC::Detail::TaskHandle handle{0};
        size_t blockCount{0};
    };

    /* Failure injection switches. */
    bool loadToHostBufferFails{false};
    bool loadToDeviceFails{false};
    bool loadSSDToHostFails{false};
    bool waitSSDToHostFails{false};
    bool releaseHostBufferFails{false};
    bool lookupFails{false};
    bool waitFails{false};
    bool checkFails{false};
    bool loadFails{false};
    bool dumpFails{false};
    bool mismatchedViewCount{false};
    bool nullHostAddr{false};

    /* Configuration of canned host-buffer payloads. */
    void SetHostBufferPayload(std::size_t bytesPerBlock, std::size_t kvLayerBytes)
    {
        std::lock_guard<std::mutex> lock(mu_);
        bytesPerBlock_ = bytesPerBlock;
        kvLayerBytes_ = kvLayerBytes;
    }

    std::vector<CallRecord> calls;
    std::mutex mu_;

    MockStoreV1() { SetHostBufferPayload(4096, 1024); }

    UC::Status Setup(const UC::Detail::Dictionary&) override { return UC::Status::OK(); }
    std::string Readme() const override { return "MockStoreV1"; }

    UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId* blocks,
                                              size_t num) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"Lookup"};
        r.blockCount = num;
        calls.push_back(std::move(r));
        if (lookupFails) { return UC::Status::Error("lookup-failed-injected"); }
        return std::vector<uint8_t>(num, 1u);
    }

    UC::Expected<ssize_t> LookupOnPrefix(const UC::Detail::BlockId*, size_t num) override
    { return static_cast<ssize_t>(num) - 1; }
    UC::Expected<ssize_t> LookupOnReverse(const UC::Detail::BlockId*, size_t num) override
    { return num == 0 ? ssize_t(-1) : ssize_t(num - 1); }

    void Prefetch(const UC::Detail::BlockId*, size_t) override { /* no-op */ }

    UC::Expected<UC::Detail::TaskHandle> Load(UC::Detail::TaskDesc task) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"Load"};
        r.task = task;
        r.handle = NextHandle();
        calls.push_back(std::move(r));
        if (loadFails) { return UC::Status::Error("load-failed-injected"); }
        return UC::Detail::TaskHandle{r.handle};
    }

    UC::Expected<UC::Detail::TaskHandle> Dump(UC::Detail::TaskDesc task) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"Dump"};
        r.task = task;
        r.handle = NextHandle();
        calls.push_back(std::move(r));
        if (dumpFails) { return UC::Status::Error("dump-failed-injected"); }
        return UC::Detail::TaskHandle{r.handle};
    }

    UC::Expected<bool> Check(UC::Detail::TaskHandle) override
    {
        if (checkFails) { return UC::Status::Error("check-failed-injected"); }
        return true;
    }

    UC::Status Wait(UC::Detail::TaskHandle) override
    {
        if (waitFails) { return UC::Status::Error("wait-failed-injected"); }
        return UC::Status::OK();
    }

    UC::Status LoadToDevice(UC::Detail::TaskDesc task) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"LoadToDevice"};
        r.task = task;
        r.blockCount = task.size();
        calls.push_back(std::move(r));
        if (loadToDeviceFails) { return UC::Status::Error("load-to-device-failed-injected"); }
        return UC::Status::OK();
    }

    UC::Expected<UC::Detail::TaskHandle> LoadSSDToHost(UC::Detail::TaskDesc task) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"LoadSSDToHost"};
        r.task = task;
        r.handle = NextHandle();
        calls.push_back(std::move(r));
        if (loadSSDToHostFails) { return UC::Status::Error("load-ssd-to-host-failed-injected"); }
        return UC::Detail::TaskHandle{r.handle};
    }

    UC::Status WaitSSDToHost(UC::Detail::TaskHandle) override
    {
        if (waitSSDToHostFails) { return UC::Status::Error("wait-ssd-to-host-failed-injected"); }
        return UC::Status::OK();
    }

    UC::Expected<UC::Detail::HostBufferResult> LoadToHostBuffer(UC::Detail::TaskDesc task) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"LoadToHostBuffer"};
        r.task = task;
        r.blockCount = task.size();
        r.handle = NextHandle();
        calls.push_back(std::move(r));
        if (loadToHostBufferFails) {
            return UC::Status::Error("load-to-host-buffer-failed-injected");
        }

        UC::Detail::HostBufferResult result;
        result.taskId = r.handle;
        std::size_t count = task.size();
        if (mismatchedViewCount) { count = count + 1; }
        result.views.reserve(count);
        std::vector<void*> owned;
        owned.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            UC::Detail::HostBufferView v;
            void* buf = std::malloc(bytesPerBlock_);
            if (buf != nullptr) {
                std::memset(buf, static_cast<int>((i + 1) & 0xFF), bytesPerBlock_);
            }
            v.hostAddr = nullHostAddr ? nullptr : buf;
            v.sizeBytes = bytesPerBlock_;
            v.owner = i < task.size() ? task[i].owner : UC::Detail::BlockId{};
            v.index = i < task.size() ? task[i].index : 0;
            owned.push_back(buf);
            result.views.push_back(std::move(v));
        }
        outstandingHostBuffers_[r.handle] = std::move(owned);
        return result;
    }

    UC::Status ReleaseHostBuffer(UC::Detail::TaskHandle handle) override
    {
        std::lock_guard<std::mutex> lock(mu_);
        CallRecord r{"ReleaseHostBuffer"};
        r.handle = handle;
        calls.push_back(std::move(r));
        if (releaseHostBufferFails) {
            return UC::Status::Error("release-host-buffer-failed-injected");
        }
        auto it = outstandingHostBuffers_.find(handle);
        if (it != outstandingHostBuffers_.end()) {
            for (void* p : it->second) { std::free(p); }
            outstandingHostBuffers_.erase(it);
        }
        return UC::Status::OK();
    }

    ~MockStoreV1() override
    {
        /* Free any host buffers that were never released. */
        for (auto& kv : outstandingHostBuffers_) {
            for (void* p : kv.second) { std::free(p); }
        }
    }

private:
    UC::Detail::TaskHandle NextHandle() { return ++nextHandle_; }
    UC::Detail::TaskHandle nextHandle_{0};
    std::size_t bytesPerBlock_{4096};
    std::size_t kvLayerBytes_{1024};
    std::map<UC::Detail::TaskHandle, std::vector<void*>> outstandingHostBuffers_;
};

}  // namespace ucmprefetch_test

#endif  // UCM_PREFETCH_TEST_MOCK_STORE_V1_H
