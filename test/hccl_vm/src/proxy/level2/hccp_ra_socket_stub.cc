/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

/**
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "RASOCKET_STUB"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_ops.h"
#include "hccp_common.h"
#include "runtime/base.h"
#include "sim_ip_address.h"
#include "sim_log.h"
#include "store_sim_comm_memory_manager.h"

extern uint64_t g_cur_server_key;

struct CacheEntry {
    uint64_t pair_id;
    uint32_t slot_idx;
};

struct SocketInfo {
    uint32_t device_id;
    uint64_t endpoint_id;
    uint8_t state;
};

struct FdInfo {
    uint32_t slot_idx;
    uint32_t role;
    uint64_t pair_id;
    sim::SlotHandle handle;
};
struct SocketWrapInitInfo {
    int family;
    union HccpIpAddr localIp;
    int scopeId;
};
struct SocketConnTaskInfo {
    void *sockWrap;
    unsigned int role;
    union HccpIpAddr remoteIp;
    unsigned int port;
    char tag[SOCK_CONN_TAG_SIZE];
    int status;
    void *fdHandle;
};
struct SocketExchInfo {
    void *socketHandle;
    void *sendBuf;
    unsigned long long sendSize;
    unsigned long long *sentSize;
    void *recvBuf;
    unsigned long long recvSize;
    unsigned long long *resvSize;
    bool isRecvFirst;
};

static std::unordered_map<std::string, CacheEntry> g_tagCache;
static std::mutex g_tagCacheMutex;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

static uint64_t ComputeTagHash(const char *tag) {
    return static_cast<uint64_t>(std::hash<std::string>{}(std::string(tag)));
}

int RaSocketInit(int mode, struct rdev rdevInfo, void **socketHandle) {
    // 首次调用 RaSocketInit 时初始化 SHM 池，后续调用跳过
    static std::once_flag initPoolFlag;
    std::call_once(initPoolFlag, []() {
        HCCL_VM_INFO("lazy init pool on first RaSocketInit call");
        sim::CommunicationMemoryManager::GetInstance().InitPool();
    });

    (void)mode;
    sim::Device device{};
    if (GetDeviceByPhysicalId(rdevInfo.phyId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR(" get device by phy id {} failed.", rdevInfo.phyId);
        return -1;
    }

    BinaryAddr ba{};
    memcpy(&ba, &rdevInfo.localIp, sizeof(BinaryAddr));
    auto ipAdress = IpAddress(ba, AF_INET6);
    auto eidStr = ipAdress.EidToHexString();

    sim::EndPoint endPoint{};
    if (GetEndPointByEid(ipAdress, endPoint) != 0) {
        HCCL_VM_ERROR(" cannot find remote eid {} ", eidStr.c_str());
        return -1;
    }

    auto deviceIdx = endPoint.device_id;
    auto endpointId = endPoint.id;

    auto *info = new SocketInfo{deviceIdx, endpointId, 0};
    *socketHandle = static_cast<void *>(info);
    HCCL_VM_INFO(" add socketHandle:{:p} device:{:d} eid:{} endpointId:{:d}",
                 static_cast<void *>(info), deviceIdx, eidStr.c_str(),
                 endpointId);
    return 0;
}

int RaSocketInitV1(int mode, struct SocketInitInfoT socketInit,
                   void **socketHandle) {
    (void)mode;
    (void)socketInit;
    struct rdev rdevInfo {};
    return RaSocketInit(mode, rdevInfo, socketHandle);
}

int RaSocketWrapBatchInit(struct RaInfo raInfo,
                          struct SocketWrapInitInfo sockWrapInitInfoArray[],
                          unsigned int num, struct ErrorInfo *errInfo,
                          void *sockWrapArray[]) {
    for (unsigned int i = 0; i < num; i++) {
        struct rdev rdevInfo {};
        rdevInfo.phyId = raInfo.phyId;
        rdevInfo.family = sockWrapInitInfoArray[i].family;
        memcpy(&rdevInfo.localIp, &sockWrapInitInfoArray[i].localIp,
               sizeof(union HccpIpAddr));

        int ret = RaSocketInit(raInfo.mode, rdevInfo, &sockWrapArray[i]);
        if (ret != 0) {
            errInfo->errIndex = i;
            errInfo->errCode = ret;
            return -1;
        }
    }
    return 0;
}

int RaSocketDeinit(void *socketHandle) {
    auto *info = static_cast<SocketInfo *>(socketHandle);
    HCCL_VM_INFO(" delete socketHandle:{:p}", static_cast<void *>(info));
    delete info;
    return 0;
}

int RaSocketWrapBatchDeinit(void *sockWrapArray[], unsigned int num,
                            struct ErrorInfo errInfoArray[],
                            unsigned int *errNum) {
    *errNum = 0;
    for (unsigned int i = 0; i < num; i++) {
        int ret = RaSocketDeinit(sockWrapArray[i]);
        if (ret != 0) {
            errInfoArray[*errNum].errIndex = i;
            errInfoArray[*errNum].errCode = ret;
            (*errNum)++;
        }
    }
    return (*errNum > 0) ? -1 : 0;
}

int RaSocketListenStart(struct SocketListenInfoT conn[], uint32_t num) {
    for (uint32_t i = 0; i < num; i++) {
        auto *info = static_cast<SocketInfo *>(conn[i].socketHandle);
        info->state = 1;
        HCCL_VM_INFO(" socket {:p} port:{:d}", static_cast<void *>(info),
                     conn[i].port);
    }
    return 0;
}

int RaSocketListenStop(struct SocketListenInfoT conn[], unsigned int num) {
    for (uint32_t i = 0; i < num; i++) {
        auto *info = static_cast<SocketInfo *>(conn[i].socketHandle);
        info->state = 0;
        HCCL_VM_INFO(" socket {:p} port:{:d}", static_cast<void *>(info),
                     conn[i].port);
    }
    return 0;
}

int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num) {
    std::vector<sim::RaSocketPair> pairs;
    std::vector<uint32_t> slotIdxs;
    pairs.reserve(num);
    slotIdxs.reserve(num);

    for (unsigned int i = 0; i < num; i++) {
        auto *localInfo = static_cast<SocketInfo *>(conn[i].socketHandle);

        BinaryAddr ba{};
        memcpy(&ba, &conn[i].remoteIp, sizeof(BinaryAddr));
        auto ipAdress = IpAddress(ba, AF_INET6);
        auto eidStr = ipAdress.EidToHexString();
        sim::EndPoint endPoint{};
        if (GetEndPointByEid(ipAdress, endPoint) != 0) {
            HCCL_VM_ERROR(" cannot find remote eid:{}", eidStr.c_str());
            return -1;
        }

        auto remoteDevId = endPoint.device_id;
        uint64_t tagHash = ComputeTagHash(conn[i].tag);
        uint32_t remotePort = conn[i].port;

        HCCL_VM_INFO(" dev:{} sock:{:p} connect dev:{} eid:{} tag:{}",
                     localInfo->device_id, static_cast<void *>(localInfo),
                     remoteDevId, eidStr.c_str(), conn[i].tag);

        uint32_t slotIdx =
            sim::CommunicationMemoryManager::GetInstance().AllocSlot();
        if (slotIdx == UINT32_MAX) {
            HCCL_VM_ERROR("AllocSlot failed for tag:{}", conn[i].tag);
            return -1;
        }

        sim::RaSocketPair socketpair{};
        socketpair.client_id = reinterpret_cast<uintptr_t>(localInfo);
        socketpair.server_id = 0;
        socketpair.ref_cnt = 0;
        socketpair.port = remotePort;
        socketpair.tag_hash = tagHash;
        socketpair.slot_idx = slotIdx;
        pairs.push_back(socketpair);
        slotIdxs.push_back(slotIdx);
    }

    auto pairIds = RunnerDB::AddBatch<sim::RaSocketPair>(pairs);
    if (pairIds.size() != num) {
        HCCL_VM_ERROR("AddBatch RaSocketPair failed, expected {} got {}", num,
                      pairIds.size());
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(g_tagCacheMutex);
        for (unsigned int i = 0; i < num; i++) {
            g_tagCache[conn[i].tag] = {pairIds[i], slotIdxs[i]};
            HCCL_VM_INFO(
                " add pair id:{:d} local:{:p} slotIdx:{} tag:{} port:{}",
                pairIds[i], static_cast<void *>(conn[i].socketHandle),
                slotIdxs[i], conn[i].tag, conn[i].port);
        }
    }
    return 0;
}

int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num,
                 unsigned int *connectedNum) {
    *connectedNum = 0;
    uint32_t fdRole = (role == 0) ? sim::kRsFdServer : sim::kRsFdClient;
    for (int i = 0; i < num; i++) {
        uint64_t tagHash = ComputeTagHash(conn[i].tag);
        uint32_t slotIdx = UINT32_MAX;
        uint64_t pairId = 0;

        {
            std::lock_guard<std::mutex> lock(g_tagCacheMutex);
            auto it = g_tagCache.find(conn[i].tag);
            if (it != g_tagCache.end()) {
                slotIdx = it->second.slot_idx;
                pairId = it->second.pair_id;
            }
        }
        if (slotIdx == UINT32_MAX && role == sim::kRsFdServer) {
            uint32_t count = 0;
            while (count++ < 180) {
                auto pairRes = RunnerDB::GetOneByPred<sim::RaSocketPair>(
                    [tagHash](const sim::RaSocketPair &p) {
                        return p.tag_hash == tagHash;
                    });
                if (!pairRes.second) {
                    if (count % 60 == 0) {
                        HCCL_VM_WARN(
                            "Server: waiting for pair tag:{}, attempt:{}/180",
                            conn[i].tag, count);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    continue;
                }
                slotIdx = pairRes.first.slot_idx;
                pairId = pairRes.first.id;
                break;
            }
            if (slotIdx == UINT32_MAX) {
                HCCL_VM_ERROR("Server: pair not found for tag:{}", conn[i].tag);
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(g_tagCacheMutex);
                g_tagCache[conn[i].tag] = {pairId, slotIdx};
            }
        } else if (slotIdx == UINT32_MAX) {
            HCCL_VM_ERROR("Client: tag {} not found in cache", conn[i].tag);
            continue;
        }

        sim::SlotHandle handle =
            sim::CommunicationMemoryManager::GetInstance().GetSlotHandle(
                slotIdx, fdRole);
        if (!handle.base) {
            HCCL_VM_ERROR("GetSlotHandle failed for slotIdx:{}", slotIdx);
            continue;
        }

        sim::CommunicationMemoryManager::GetInstance().AddRef(slotIdx);

        auto *fd = new FdInfo{slotIdx, fdRole, pairId, handle};
        conn[i].fdHandle = static_cast<void *>(fd);
        conn[i].status = 1;
        *connectedNum += 1;

        HCCL_VM_INFO(" get socket tag:{} slotIdx:{} role:{}", conn[i].tag,
                     slotIdx, role);
    }
    return 0;
}

int RaSocketBatchClose(struct SocketCloseInfoT conn[], unsigned int num) {
    for (int i = 0; i < num; i++) {
        auto *fd = static_cast<FdInfo *>(conn[i].fdHandle);
        {
            std::lock_guard<std::mutex> lock(g_tagCacheMutex);
            for (auto it = g_tagCache.begin(); it != g_tagCache.end(); ++it) {
                if (it->second.slot_idx == fd->slot_idx) {
                    g_tagCache.erase(it);
                    break;
                }
            }
        }

        bool isLast = sim::CommunicationMemoryManager::GetInstance().CloseSlot(
            fd->slot_idx);
        if (isLast) {
            RunnerDB::Delete<sim::RaSocketPair>(fd->pair_id);
            HCCL_VM_INFO(" delete pair:{:d} slotIdx:{}", fd->pair_id,
                         fd->slot_idx);
        }
        HCCL_VM_INFO(" close socket slotIdx:{}", fd->slot_idx);
        delete fd;
    }
    return 0;
}

int RaSocketBatchAbort(struct SocketConnectInfoT conn[], unsigned int num) {
    (void)conn;
    (void)num;
    return 0;
}

int RaSocketBatchDisconnectNorm(void *socketHandleArray[], unsigned int num,
                                struct ErrorInfo errInfoArray[],
                                unsigned int *errNum) {
    *errNum = 0;
    for (unsigned int i = 0; i < num; i++) {
        auto *socketInfo = static_cast<SocketInfo *>(socketHandleArray[i]);
        uintptr_t clientId = reinterpret_cast<uintptr_t>(socketInfo);

        auto pairs = RunnerDB::GetByPred<sim::RaSocketPair>(
            [clientId](const sim::RaSocketPair &p) {
                return p.client_id == clientId;
            });

        for (auto &pair : pairs) {
            auto *tempFd = new FdInfo{pair.slot_idx, 0, pair.id, {}};
            struct SocketCloseInfoT closeInfo {};
            closeInfo.fdHandle = tempFd;
            RaSocketBatchClose(&closeInfo, 1);
        }

        struct SocketListenInfoT listenInfo {};
        listenInfo.socketHandle = socketHandleArray[i];
        RaSocketListenStop(&listenInfo, 1);
    }
    return (*errNum > 0) ? -1 : 0;
}

int RaSocketSend(const void *fdHandle, const void *data,
                 unsigned long long size, unsigned long long *sentSize) {
    auto *fd = static_cast<const FdInfo *>(fdHandle);
    int64_t ret = sim::CommunicationMemoryManager::GetInstance().Send(
        fd->handle, data, size);
    if (ret == -1) {
        HCCL_VM_ERROR("Send failed slotIdx:{}", fd->slot_idx);
        return -1;
    }
    *sentSize = ret;
    return 0;
}

int RaSocketRecv(const void *fdHandle, void *data, unsigned long long size,
                 unsigned long long *receivedSize) {
    auto *fd = static_cast<const FdInfo *>(fdHandle);
    int64_t ret = sim::CommunicationMemoryManager::GetInstance().Recv(
        fd->handle, data, size);
    if (ret == 0) {
        return SOCK_EAGAIN;
    }
    if (ret == -1) {
        HCCL_VM_ERROR("Recv failed slotIdx:{}", fd->slot_idx);
        return -1;
    }
    *receivedSize = ret;
    return 0;
}

int RaEpollCtlAdd(const void *fdHandle, enum RaEpollEvent event) {
    (void)fdHandle;
    (void)event;
    return 0;
}

int RaEpollCtlMod(const void *fdHandle, enum RaEpollEvent event) {
    (void)fdHandle;
    (void)event;
    return 0;
}

int RaEpollCtlDel(const void *fdHandle) {
    (void)fdHandle;
    return 0;
}

int RaSetTcpRecvCallback(const void *socketHandle, const void *callback) {
    (void)socketHandle;
    (void)callback;
    return 0;
}

int RaExChangeDataAsync(struct SocketExchInfo sockExchInfoArray[],
                        unsigned int num, void **reqOpHandle, int timeout) {
    (void)timeout;
    *reqOpHandle = reinterpret_cast<void *>(0x12345678);

    for (unsigned int i = 0; i < num; i++) {
        auto *socketInfo =
            static_cast<SocketInfo *>(sockExchInfoArray[i].socketHandle);
        uintptr_t clientId = reinterpret_cast<uintptr_t>(socketInfo);

        auto pairs = RunnerDB::GetByPred<sim::RaSocketPair>(
            [clientId](const sim::RaSocketPair &p) {
                return p.client_id == clientId;
            });

        if (pairs.empty()) {
            HCCL_VM_ERROR("no pair found for socketHandle:{:p}",
                          static_cast<void *>(socketInfo));
            continue;
        }

        auto &pair = pairs.front();
        sim::SlotHandle handle =
            sim::CommunicationMemoryManager::GetInstance().GetSlotHandle(
                pair.slot_idx, sim::kRsFdClient);
        auto *tempFd =
            new FdInfo{pair.slot_idx, sim::kRsFdClient, pair.id, handle};

        if (sockExchInfoArray[i].isRecvFirst) {
            RaSocketRecv(tempFd, sockExchInfoArray[i].recvBuf,
                         sockExchInfoArray[i].recvSize,
                         sockExchInfoArray[i].resvSize);
            RaSocketSend(tempFd, sockExchInfoArray[i].sendBuf,
                         sockExchInfoArray[i].sendSize,
                         sockExchInfoArray[i].sentSize);
        } else {
            RaSocketSend(tempFd, sockExchInfoArray[i].sendBuf,
                         sockExchInfoArray[i].sendSize,
                         sockExchInfoArray[i].sentSize);
            RaSocketRecv(tempFd, sockExchInfoArray[i].recvBuf,
                         sockExchInfoArray[i].recvSize,
                         sockExchInfoArray[i].resvSize);
        }

        delete tempFd;
    }
    return 0;
}

/////////////////////////////////async/////////////////////////////
int RaGetAsyncReqResult(void *reqHandle, int *reqResult) {
    (void)reqHandle;
    (void)reqResult;
    return 0;
}

int RaSocketBatchConnectAsync(struct SocketConnectInfoT conn[],
                              unsigned int num, void **reqHandle) {
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return RaSocketBatchConnect(conn, num);
}

int RaSocketBatchConnectNormAsync(
    struct SocketConnTaskInfo sockConnTaskInfoArray[], unsigned int num,
    void **reqOpHandle, int timeout) {
    (void)timeout;
    *reqOpHandle = reinterpret_cast<void *>(0x12345678);
    std::vector<unsigned int> clientIdx;
    std::vector<unsigned int> serverIdx;
    std::vector<struct SocketConnectInfoT> clientConns;
    std::vector<struct SocketInfoT> clientSockets;
    std::vector<struct SocketInfoT> serverSockets;

    for (unsigned int i = 0; i < num; i++) {
        if (sockConnTaskInfoArray[i].role == sim::kRsFdClient) {
            clientIdx.push_back(i);
            struct SocketConnectInfoT conn {};
            conn.socketHandle = sockConnTaskInfoArray[i].sockWrap;
            memcpy(&conn.remoteIp, &sockConnTaskInfoArray[i].remoteIp,
                   sizeof(union HccpIpAddr));
            conn.port = sockConnTaskInfoArray[i].port;
            strncpy(conn.tag, sockConnTaskInfoArray[i].tag,
                    SOCK_CONN_TAG_SIZE - 1);
            clientConns.push_back(conn);

            struct SocketInfoT sock {};
            sock.socketHandle = sockConnTaskInfoArray[i].sockWrap;
            strncpy(sock.tag, sockConnTaskInfoArray[i].tag,
                    SOCK_CONN_TAG_SIZE - 1);
            clientSockets.push_back(sock);
        } else {
            serverIdx.push_back(i);
            struct SocketInfoT sock {};
            sock.socketHandle = sockConnTaskInfoArray[i].sockWrap;
            strncpy(sock.tag, sockConnTaskInfoArray[i].tag,
                    SOCK_CONN_TAG_SIZE - 1);
            serverSockets.push_back(sock);
        }
    }

    if (!clientConns.empty()) {
        RaSocketBatchConnect(clientConns.data(), clientConns.size());
    }

    if (!clientSockets.empty()) {
        unsigned int connectedNum = 0;
        RaGetSockets(sim::kRsFdClient, clientSockets.data(),
                     clientSockets.size(), &connectedNum);
        for (unsigned int j = 0; j < clientIdx.size(); j++) {
            unsigned int i = clientIdx[j];
            sockConnTaskInfoArray[i].fdHandle = clientSockets[j].fdHandle;
            sockConnTaskInfoArray[i].status = clientSockets[j].status;
        }
    }

    if (!serverSockets.empty()) {
        unsigned int connectedNum = 0;
        RaGetSockets(sim::kRsFdServer, serverSockets.data(),
                     serverSockets.size(), &connectedNum);
        for (unsigned int j = 0; j < serverIdx.size(); j++) {
            unsigned int i = serverIdx[j];
            sockConnTaskInfoArray[i].fdHandle = serverSockets[j].fdHandle;
            sockConnTaskInfoArray[i].status = serverSockets[j].status;
        }
    }

    return 0;
}

int RaSocketListenStartAsync(struct SocketListenInfoT conn[], unsigned int num,
                             void **reqHandle) {
    (void)conn;
    (void)num;
    (void)reqHandle;
    return -1;
}

int RaSocketListenStopAsync(struct SocketListenInfoT conn[], unsigned int num,
                            void **reqHandle) {
    (void)conn;
    (void)num;
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return 0;
}

int RaSocketBatchCloseAsync(struct SocketCloseInfoT conn[], unsigned int num,
                            void **reqHandle) {
    (void)reqHandle;
    HCCL_VM_INFO(" close socket async");
    return RaSocketBatchClose(conn, num);
}

int RaSocketSendAsync(const void *fdHandle, const void *data,
                      unsigned long long size, unsigned long long *sentSize,
                      void **reqHandle) {
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    return RaSocketSend(fdHandle, data, size, sentSize);
}

int RaSocketRecvAsync(const void *fdHandle, void *data, unsigned long long size,
                      unsigned long long *receivedSize, void **reqHandle) {
    *reqHandle = reinterpret_cast<void *>(0x12345678);
    int ret = RaSocketRecv(fdHandle, data, size, receivedSize);
    if (ret == SOCK_EAGAIN) {
        *receivedSize = 0;
        return 0;
    }
    return ret;
}

#ifdef __cplusplus
}
#endif // __cplusplus
