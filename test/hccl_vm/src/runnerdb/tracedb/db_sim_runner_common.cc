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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <nlohmann_json/json.hpp>
#include <runnerdb/db_sim_runner_common.h>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "db_sim_runner_common.h"
#include "db_sim_runner_db.h"
#include "sim_common_api.h"
#include "sim_log.h"
#include "sim_models.h"

using json = nlohmann::json;

uint64_t g_cur_server_key = 0;
thread_local uint64_t g_cur_device_key = 0;
thread_local uint64_t g_cur_comm_key = 0;
// 进程级通信域配置缓存(TopoMetaConfig表项指向的配置解析结果):
// mock-comm执行时经sim::RefreshTopoMetaConfig刷新,
// 其余进程首次使用时经LoadCommConfigData加载
sim::CommConfigData g_comm_config_data;

namespace {
// g_comm_config_data的互斥保护与已加载标记
std::mutex g_comm_config_mutex;
bool g_comm_config_loaded = false;
} // namespace

namespace sim {
namespace {
constexpr uint32_t kCommunicatorWaitIntervalMs = 100;
constexpr uint32_t kCommunicatorWaitTimeoutMs = 60000;
constexpr uint32_t kCommDestroyWaitTimeoutMs = 300000; // 64p场景60s不够

bool IsValidCommunicatorName(const char *commName) {
    // 通信域名称存储在定长数组中，必须在边界内以 '\0' 结束。
    return commName != nullptr &&
           ::strnlen(commName, sizeof(sim::Communicator::comm_id)) <
               sizeof(sim::Communicator::comm_id);
}

const char *SafeCommunicatorName(const char *commName) {
    // 日志不能将未终止的定长数组直接按 C 字符串输出。
    return IsValidCommunicatorName(commName) ? commName : "<invalid>";
}

bool IsSameCommunicator(const sim::Communicator &record, const char *commName,
                        uint64_t commHash) {
    return std::strncmp(record.comm_id, commName, sizeof(record.comm_id)) ==
               0 &&
           record.comm_hash == commHash;
}
} // namespace

bool GetOrInsertCommunicator(const char *commName, uint32_t rankSize,
                             uint32_t rankId, uint64_t deviceId,
                             uint64_t commHash, uint64_t &commId) {
    if (!IsValidCommunicatorName(commName) || commName[0] == '\0' ||
        rankSize == 0 || rankId >= rankSize || deviceId == 0) {
        HCCL_VM_ERROR("invalid communicator parameters: name={}, rankSize={}, "
                      "rankId={}, deviceId={}",
                      SafeCommunicatorName(commName), rankSize, rankId,
                      deviceId);
        return false;
    }

    const auto members = RunnerDB::GetByPred<sim::Communicator>(
        [commName, commHash](const sim::Communicator &record) {
            return IsSameCommunicator(record, commName, commHash);
        });
    if (!members.empty()) {
        for (const auto &member : members) {
            if (member.rank_size != rankSize) {
                HCCL_VM_ERROR(
                    "communicator domain conflict: name={}, rankSize={}",
                    commName, rankSize);
                return false;
            }
            if (member.rank_id == rankId) {
                if (member.device_id != deviceId) {
                    HCCL_VM_ERROR("communicator rank mapping conflict: "
                                  "name={}, rankId={}, "
                                  "oldDeviceId={}, newDeviceId={}",
                                  commName, rankId, member.device_id, deviceId);
                    return false;
                }
                commId = member.id;
                // 复用表行时清残留销毁状态, 避免旧状态误放行.
                (void)RunnerDB::Update<sim::Communicator>(
                    commId, [](sim::Communicator &record) {
                        record.status = COMM_STATUS_ACTIVE;
                    });
                return true;
            }
            if (member.device_id == deviceId) {
                HCCL_VM_ERROR("communicator device mapping conflict: name={}, "
                              "deviceId={}, "
                              "oldRankId={}, newRankId={}",
                              commName, deviceId, member.rank_id, rankId);
                return false;
            }
        }
    }

    sim::Communicator record{};
    std::strncpy(record.comm_id, commName, sizeof(record.comm_id) - 1);
    record.comm_hash = commHash;
    record.rank_size = rankSize;
    record.rank_id = rankId;
    record.device_id = deviceId;
    const uint64_t memberId = RunnerDB::Add<sim::Communicator>(record);
    if (memberId == 0) {
        HCCL_VM_ERROR("insert communicator member failed: name={}, rankId={}, "
                      "deviceId={}",
                      commName, rankId, deviceId);
        return false;
    }
    commId = memberId;
    HCCL_VM_DEBUG("Insert Communicator member, id={}, commName={}, "
                  "commHash={}, rankSize={}, rankId={}, deviceId={}",
                  commId, std::string(commName), record.comm_hash,
                  record.rank_size, record.rank_id, record.device_id);
    return true;
}

bool WaitCommunicatorReady(const char *commName, uint64_t commHash,
                           uint32_t rankSize) {
    if (!IsValidCommunicatorName(commName) || commName[0] == '\0' ||
        rankSize == 0) {
        HCCL_VM_ERROR(
            "invalid communicator readiness parameters: name={}, rankSize={}",
            SafeCommunicatorName(commName), rankSize);
        return false;
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kCommunicatorWaitTimeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto records = RunnerDB::GetByPred<sim::Communicator>(
            [commName, commHash](const sim::Communicator &record) {
                return IsSameCommunicator(record, commName, commHash);
            });
        std::vector<bool> rankSeen(rankSize, false);
        for (const auto &record : records) {
            if (record.rank_id < rankSize) {
                rankSeen[record.rank_id] = true;
            }
        }
        if (std::all_of(rankSeen.begin(), rankSeen.end(),
                        [](bool seen) { return seen; })) {
            return true;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kCommunicatorWaitIntervalMs));
    }

    HCCL_VM_ERROR(
        "wait communicator ready timeout: name={}, commHash={}, rankSize={}",
        commName, commHash, rankSize);
    return false;
}

bool WaitCommunicatorDestroyReady(uint64_t commId) {
    const auto self = RunnerDB::GetById<sim::Communicator>(commId);
    if (!self.has_value() || !IsValidCommunicatorName(self->comm_id) ||
        self->comm_id[0] == '\0' || self->rank_size == 0 ||
        self->rank_id >= self->rank_size) {
        HCCL_VM_ERROR("invalid communicator destroy barrier member id={}",
                      commId);
        return false;
    }

    sim::CommunicatorDestroySync record{};
    std::strncpy(record.comm_id, self->comm_id, sizeof(record.comm_id) - 1);
    record.comm_hash = self->comm_hash;
    record.rank_size = self->rank_size;
    record.rank_id = self->rank_id;
    if (RunnerDB::Add<sim::CommunicatorDestroySync>(record) == 0) {
        HCCL_VM_ERROR("failed to mark communicator destroy ready: id={}",
                      commId);
        return false;
    }

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kCommunicatorWaitTimeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto records = RunnerDB::GetByPred<sim::CommunicatorDestroySync>(
            [&self](const sim::CommunicatorDestroySync &record) {
                return std::strncmp(record.comm_id, self->comm_id,
                                    sizeof(record.comm_id)) == 0 &&
                       record.comm_hash == self->comm_hash;
            });
        if (records.size() >= self->rank_size) {
            return true;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kCommunicatorWaitIntervalMs));
    }

    HCCL_VM_ERROR("wait communicator destroy barrier timeout: name={}, "
                  "commHash={}, rankSize={}",
                  self->comm_id, self->comm_hash, self->rank_size);
    return false;
}

bool MarkAndSyncCommunicatorDestroy(uint64_t commId) {
    const auto self = RunnerDB::GetById<sim::Communicator>(commId);
    if (!self.has_value() || !IsValidCommunicatorName(self->comm_id) ||
        self->comm_id[0] == '\0' || self->rank_size == 0 ||
        self->rank_id >= self->rank_size) {
        HCCL_VM_ERROR("invalid communicator destroy sync member id={}", commId);
        return false;
    }

    // 先标记本 rank 已销毁, 供同通信域其他 rank 感知.
    if (!RunnerDB::Update<sim::Communicator>(
            commId, [](sim::Communicator &record) {
                record.status = COMM_STATUS_DESTROYED;
            })) {
        HCCL_VM_ERROR("failed to mark communicator destroy status: id={}",
                      commId);
        return false;
    }

    // 等待同通信域(相同 comm_id + comm_hash)表内全部成员标记销毁;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kCommDestroyWaitTimeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto records = RunnerDB::GetByPred<sim::Communicator>(
            [&self](const sim::Communicator &record) {
                return IsSameCommunicator(record, self->comm_id,
                                          self->comm_hash);
            });
        bool allDestroyed = !records.empty();
        for (const auto &record : records) {
            if (record.status != COMM_STATUS_DESTROYED) {
                allDestroyed = false;
                break;
            }
        }
        if (allDestroyed) {
            return true;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kCommunicatorWaitIntervalMs));
    }

    HCCL_VM_ERROR("wait communicator destroy sync timeout: name={}, "
                  "commHash={}, rankSize={}",
                  self->comm_id, self->comm_hash, self->rank_size);
    return false;
}

bool GetCommunicatorMembers(uint64_t commId,
                            std::vector<CommunicatorMemberInfo> &members) {
    members.clear();
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return false;
    }

    const auto selfMember = RunnerDB::GetById<sim::Communicator>(commId);
    if (!selfMember.has_value()) {
        HCCL_VM_ERROR("communicator member not found by id:{}", commId);
        return false;
    }

    const auto communicatorMembers = RunnerDB::GetByPred<sim::Communicator>(
        [&selfMember](const sim::Communicator &member) {
            return IsSameCommunicator(member, selfMember->comm_id,
                                      selfMember->comm_hash);
        });
    if (communicatorMembers.empty()) {
        HCCL_VM_ERROR("communicator has no members: commName={}",
                      selfMember->comm_id);
        return false;
    }
    members.reserve(communicatorMembers.size());
    for (const sim::Communicator &member : communicatorMembers) {
        members.push_back({member.id, member.device_id, member.rank_id});
    }
    return true;
}

bool GetCommunicatorName(uint64_t commId, std::string &commName) {
    commName.clear();
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return false;
    }

    const auto selfMember = RunnerDB::GetById<sim::Communicator>(commId);
    if (!selfMember.has_value()) {
        HCCL_VM_ERROR("communicator member not found by id:{}", commId);
        return false;
    }

    if (!IsValidCommunicatorName(selfMember->comm_id) ||
        selfMember->comm_id[0] == '\0') {
        HCCL_VM_ERROR("invalid communicator name for member id:{}", commId);
        return false;
    }
    commName = selfMember->comm_id;
    return true;
}

bool GetCommunicatorIdentity(uint64_t commId, std::string &commName,
                             uint64_t &commHash) {
    commName.clear();
    commHash = 0;
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return false;
    }
    const auto selfMember = RunnerDB::GetById<sim::Communicator>(commId);
    if (!selfMember.has_value() ||
        !IsValidCommunicatorName(selfMember->comm_id) ||
        selfMember->comm_id[0] == '\0') {
        HCCL_VM_ERROR("invalid communicator identity for member id:{}", commId);
        return false;
    }
    commName = selfMember->comm_id;
    commHash = selfMember->comm_hash;
    return true;
}

aclError GetDeviceByLogicId(uint32_t deviceId, sim::Device &device) {
    auto ret =
        RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
            return d.server_id == g_cur_server_key && d.logic_id == deviceId;
        });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by logical id {:d}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    device = ret.first;
    return ACL_SUCCESS;
}

aclError GetDeviceByCommRank(uint64_t commId, uint32_t rankId,
                             sim::Device &device) {
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return ACL_ERROR_INVALID_PARAM;
    }
    auto selfMember = RunnerDB::GetById<sim::Communicator>(commId);
    if (!selfMember.has_value()) {
        HCCL_VM_ERROR("communicator member not found by id:{}", commId);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto rankRet = RunnerDB::GetOneByPred<sim::Communicator>(
        [&selfMember, rankId](const sim::Communicator &member) {
            return IsSameCommunicator(member, selfMember->comm_id,
                                      selfMember->comm_hash) &&
                   member.rank_id == rankId;
        });
    if (!rankRet.second) {
        HCCL_VM_ERROR("cannot find communicator member: commName={}, rankId={}",
                      selfMember->comm_id, rankId);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto deviceKey = rankRet.first.device_id;

    auto deviceRet = RunnerDB::GetById<sim::Device>(deviceKey);
    if (!deviceRet.has_value()) {
        // not find
        HCCL_VM_ERROR("cannot find device for communicator member: "
                      "commName={}, rankId={}",
                      selfMember->comm_id, rankId);
        return ACL_ERROR_INVALID_PARAM;
    }
    device = *deviceRet;

    return ACL_SUCCESS;
}

aclError GetDeviceByPhysicalId(uint32_t deviceId, sim::Device &device) {
    auto ret =
        RunnerDB::GetOneByPred<sim::Device>([deviceId](const sim::Device &d) {
            return d.server_id == g_cur_server_key && d.physical_id == deviceId;
        });
    if (!ret.second) {
        HCCL_VM_ERROR(
            "[{}] cannot find device by physical id {:d}, server_key {:d}",
            __func__, deviceId, g_cur_server_key);
        return ACL_ERROR_INVALID_PARAM;
    }
    HCCL_VM_INFO(
        "GetDeviceByPhysicalId: device.physical_id = {:d}, serverkey = {:d}",
        device.physical_id, g_cur_server_key);

    device = ret.first;
    return ACL_SUCCESS;
}

aclError GetServerByKey(uint64_t serverKey, sim::Server &server) {
    auto ret = RunnerDB::GetOneByPred<sim::Server>(
        [serverKey](const sim::Server &d) { return d.id == serverKey; });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find server by key {:d}", serverKey);
        return ACL_ERROR_INVALID_PARAM;
    }

    server = ret.first;
    return ACL_SUCCESS;
}

aclError GetDeviceByServerKeyAndPhysicalId(uint64_t serverKey,
                                           uint32_t deviceId,
                                           sim::Device &device) {
    auto ret = RunnerDB::GetOneByPred<sim::Device>(
        [serverKey, deviceId](const sim::Device &d) {
            return d.server_id == serverKey && d.physical_id == deviceId;
        });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by server key {:d}, physical id {:d}",
                      serverKey, deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    device = ret.first;
    return ACL_SUCCESS;
}

aclError UpdateDeviceLogicId(uint64_t serverKey, uint32_t phyDevId,
                             uint32_t logicDevId, uint32_t userId) {
    sim::Device device{};
    auto ret = GetDeviceByServerKeyAndPhysicalId(serverKey, phyDevId, device);
    if (ret != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceKey = device.id;
    RunnerDB::Update<sim::Device>(
        deviceKey, [deviceKey, logicDevId, userId](sim::Device &dev) {
            dev.logic_id = logicDevId;
            dev.user_id = userId;
            dev.status = 1; // 设备状态设置为可用
        });

    return ACL_SUCCESS;
}

bool ResetAllDeviceLogicId() {
    auto allDevices = RunnerDB::GetByPred<sim::Device>(
        [](const sim::Device &d) { return true; });
    if (allDevices.empty()) {
        HCCL_VM_ERROR("cannot find any device");
        return false;
    }

    for (auto &device : allDevices) {
        auto deviceKey = device.id;
        RunnerDB::Update<sim::Device>(deviceKey, [deviceKey](sim::Device &dev) {
            dev.logic_id = 0xFFFF;
            dev.status = 0; // 重置设备状态，防止残留 status=1 干扰后续
                            // aclrtGetDeviceCount
            dev.user_id = 0xFFFF;
        });
    }

    return true;
}

aclError UpdateSuperDeviceId(uint32_t logicDevId, uint32_t superDeviceId) {
    sim::Device device{};
    auto ret = GetDeviceByLogicId(logicDevId, device);
    if (ret != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceKey = device.id;
    RunnerDB::Update<sim::Device>(deviceKey,
                                  [deviceKey, superDeviceId](sim::Device &dev) {
                                      dev.super_device_id = superDeviceId;
                                  });

    return ACL_SUCCESS;
}

aclError GetCcuFromDeviceByDieId(uint64_t deviceKey, uint8_t dieId,
                                 sim::Ccu &ccu) {
    auto ret =
        RunnerDB::GetOneByPred<sim::Ccu>([deviceKey, dieId](const sim::Ccu &c) {
            return c.device_id == deviceKey && c.die_id == dieId;
        });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find ccu from device {:d} by die {:d}", deviceKey,
                      static_cast<uint32_t>(dieId));
        return ACL_ERROR_INVALID_PARAM;
    }

    ccu = ret.first;
    return ACL_SUCCESS;
}

aclError GetContextByDevId(uint32_t deviceId, sim::Context &context) {
    auto ret = RunnerDB::GetOneByPred<sim::Context>(
        [deviceId](const sim::Context &ctx) {
            return ctx.device_id == deviceId && ctx.is_default == 1;
        });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find context by logical device id {:d}",
                      deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    context = ret.first;
    return ACL_SUCCESS;
}

aclError GetPortByName(uint64_t serverKey, uint32_t phyDevId,
                       const std::string &name, sim::Port &port) {
    sim::Device device{};
    auto ret1 = GetDeviceByServerKeyAndPhysicalId(serverKey, phyDevId, device);
    if (ret1 != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by physical device id({:d}) failed",
                      phyDevId);
        return ret1;
    }

    auto deviceKey = device.id;
    auto ret2 = RunnerDB::GetOneByPred<sim::Port>([deviceKey,
                                                   name](const sim::Port &p) {
        return deviceKey == p.device_id && strcmp(p.name, name.c_str()) == 0;
    });
    if (!ret2.second) {
        HCCL_VM_ERROR("cannot find port by device:{:d} name:{}", deviceKey,
                      name.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }

    port = ret2.first;
    return ACL_SUCCESS;
}

aclError GetEndPointByIpAddr(const std::string &ip, sim::EndPoint &endPoint) {
    auto ret =
        RunnerDB::GetOneByPred<sim::EndPoint>([ip](const sim::EndPoint &ep) {
            return strcmp(ep.ip_addr, ip.c_str()) == 0;
        });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find EndPoint by ip addr:{}", ip.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }
    endPoint = ret.first;
    return ACL_SUCCESS;
}

aclError GetEndPointByEid(const IpAddress &addr, sim::EndPoint &endPoint) {
    auto ret =
        RunnerDB::GetOneByPred<sim::EndPoint>([addr](const sim::EndPoint &ep) {
            return memcmp(ep.eid, addr.GetEid().raw, URMA_EID_LEN) == 0;
        });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find EndPoint by eid:{}",
                      addr.EidToHexString().c_str());
        return ACL_ERROR_INVALID_PARAM;
    }
    endPoint = ret.first;
    return ACL_SUCCESS;
}

aclError GetPortById(uint64_t portId, sim::Port &port) {
    auto ret = RunnerDB::GetOneByPred<sim::Port>(
        [portId](const sim::Port &p) { return portId == p.id; });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find port by id: {:d}", portId);
        return ACL_ERROR_INVALID_PARAM;
    }

    port = ret.first;
    return ACL_SUCCESS;
}

bool GetCommRankByDeviceId(uint64_t commId, uint32_t deviceId,
                           uint32_t &rankId) {
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return false;
    }
    auto selfMember = RunnerDB::GetById<sim::Communicator>(commId);
    if (!selfMember.has_value()) {
        HCCL_VM_ERROR("communicator member not found by id:{}", commId);
        return false;
    }
    auto member = RunnerDB::GetOneByPred<sim::Communicator>(
        [&selfMember, deviceId](const sim::Communicator &record) {
            return IsSameCommunicator(record, selfMember->comm_id,
                                      selfMember->comm_hash) &&
                   record.device_id == deviceId;
        });
    if (!member.second) {
        HCCL_VM_ERROR("cannot find communicator rank: commName={}, deviceId={}",
                      selfMember->comm_id, deviceId);
        return false;
    }

    rankId = member.first.rank_id;
    return true;
}

uint32_t GetAICpuCount(uint64_t deviceId) {
    // ACL_DEV_ATTR_AICPU_CORE_NUM AI CPU数量。
    auto ret = RunnerDB::GetOneByPred<sim::Device>(
        [deviceId](const sim::Device &d) { return d.logic_id == deviceId; });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto aiCpus = RunnerDB::GetByPred<sim::TaskSchedulerDevice>(
        [deviceIdx](const sim::TaskSchedulerDevice &tsDev) {
            return tsDev.device_id == deviceIdx &&
                   tsDev.type == (uint8_t)TS_DEV_TYPE_CPU;
        });

    return aiCpus.size();
}

uint32_t GetAICoreCount(uint64_t deviceId) {
    // ACL_DEV_ATTR_AICORE_CORE_NUM AI CPU数量。
    auto ret = RunnerDB::GetOneByPred<sim::Device>(
        [deviceId](const sim::Device &d) { return d.logic_id == deviceId; });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto scalars = RunnerDB::GetByPred<sim::TaskSchedulerDevice>(
        [deviceIdx](const sim::TaskSchedulerDevice &tsDev) {
            return tsDev.device_id == deviceIdx &&
                   tsDev.type == (uint8_t)TS_DEV_TYPE_SCALAR;
        });

    return scalars.size();
}

uint32_t GetVectorCoreCount(uint64_t deviceId) {
    // ACL_DEV_ATTR_VECTOR_CORE_NUM AI CPU数量。
    auto ret = RunnerDB::GetOneByPred<sim::Device>(
        [deviceId](const sim::Device &d) { return d.logic_id == deviceId; });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto scalars = RunnerDB::GetByPred<sim::TaskSchedulerDevice>(
        [deviceIdx](const sim::TaskSchedulerDevice &tsDev) {
            return tsDev.device_id == deviceIdx &&
                   tsDev.type == (uint8_t)TS_DEV_TYPE_SCALAR;
        });

    uint32_t vectorCoreCount = 0;
    for (auto scalar : scalars) {
        auto tsIdx = scalar.id;
        auto vectorCores = RunnerDB::GetByPred<sim::ComputeDie>(
            [tsIdx](const sim::ComputeDie &die) {
                return die.ts_id == tsIdx &&
                       die.type == (uint8_t)COMPUTE_TYPE_VECTOR;
            });

        vectorCoreCount += vectorCores.size();
    }
    return vectorCoreCount;
}

std::string GetHardwareTypeByDevice(uint64_t deviceId) {
    // ACL_DEV_ATTR_VECTOR_CORE_NUM 机型
    auto device = RunnerDB::GetOneByPred<sim::Device>(
        [deviceId](const sim::Device &d) { return d.logic_id == deviceId; });
    if (!device.second) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return "";
    }

    auto server = RunnerDB::GetById<sim::Server>(device.first.server_id);
    if (!server.has_value()) {
        HCCL_VM_ERROR("cannot find server by id {:d}", device.first.server_id);
        return "";
    }

    return std::string(server->hardware_type);
}

uint32_t GetCubeCoreCount(uint64_t deviceId) {
    auto ret = RunnerDB::GetOneByPred<sim::Device>(
        [deviceId](const sim::Device &d) { return d.logic_id == deviceId; });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret.first.id;

    auto scalars = RunnerDB::GetByPred<sim::TaskSchedulerDevice>(
        [deviceIdx](const sim::TaskSchedulerDevice &tsDev) {
            return tsDev.device_id == deviceIdx &&
                   tsDev.type == (uint8_t)TS_DEV_TYPE_SCALAR;
        });

    uint32_t vectorCoreCount = 0;
    for (auto scalar : scalars) {
        auto tsIdx = scalar.id;
        auto vectorCores = RunnerDB::GetByPred<sim::ComputeDie>(
            [tsIdx](const sim::ComputeDie &die) {
                return die.ts_id == tsIdx &&
                       die.type == (uint8_t)COMPUTE_TYPE_CUBE;
            });

        vectorCoreCount += vectorCores.size();
    }
    return vectorCoreCount;
}

std::set<uint64_t> GetUsedServerNum() {
    try {
        auto allDevices = RunnerDB::GetByPred<sim::Device>(
            [](const sim::Device &d) { return d.status == 1; });

        std::set<uint64_t> usedServerIds;
        for (const auto &device : allDevices) {
            usedServerIds.insert(device.server_id);
        }
        return usedServerIds;
    } catch (const std::exception &e) {
        HCCL_VM_ERROR("exception: {}", e.what());
        return {};
    }
}

bool GetRankIdByMPI(uint32_t &rankId, uint64_t &serverKey) {
    const char *ompiRankStr = std::getenv("OMPI_COMM_WORLD_RANK");
    const char *mpichRankStr = std::getenv("PMI_RANK");
    if (ompiRankStr == nullptr && mpichRankStr == nullptr) {
        // 单server用例，默认serverId为1
        auto usedServerIds = GetUsedServerNum();
        if (usedServerIds.size() == 1) {
            if (*usedServerIds.begin() != 1) {
                HCCL_VM_ERROR("env OMPI_COMM_WORLD_RANK or PMI_RANK are not "
                              "found, serverKey is not 1.");
                return false;
            }
            serverKey = 1;
            return true;
        }
        HCCL_VM_ERROR("env OMPI_COMM_WORLD_RANK or PMI_RANK are not found, "
                      "usedServerIds size is not 1.");
        return false;
    } else if (ompiRankStr != nullptr) {
        rankId = static_cast<uint32_t>(atoi(ompiRankStr));
    } else {
        rankId = static_cast<uint32_t>(atoi(mpichRankStr));
    }
    HCCL_VM_INFO("rankId:{:d} serverKey:{:d}", rankId, serverKey);
    return true;
}

namespace {
// ranktable模式下(proxy进程)解析data/ranktable.json重建topoMeta:
// 仅支持net_instance_id中携带sp/srv信息的ranktable(与宿主侧ParseRanktable主路径一致),
// 旧格式依赖宿主侧network_索引回退的场景不支持, 无法定位的rank直接跳过
bool ParseTopoMetaFromRankTableFile(TopoMeta &topoMeta) {
    std::string ranktablePath =
        InstallPath::ResolveToInstallRoot("data/ranktable.json");
    std::ifstream ifs(ranktablePath);
    if (!ifs.is_open()) {
        HCCL_VM_ERROR("failed to open ranktable file: {}", ranktablePath);
        return false;
    }
    json rankTable;
    try {
        rankTable = json::parse(ifs);
    } catch (const json::exception &e) {
        HCCL_VM_ERROR("failed to parse ranktable json: {}", e.what());
        return false;
    }
    if (!rankTable.contains("rank_list") ||
        !rankTable["rank_list"].is_array()) {
        HCCL_VM_ERROR("rank_list not found or not array in ranktable");
        return false;
    }
    for (const auto &rank : rankTable["rank_list"]) {
        uint32_t localId = rank.value("local_id", 0);
        uint32_t spIdx = 0;
        uint32_t srvIdx = 0;
        bool found = false;
        const auto &levelList = rank.value("level_list", json::array());
        for (const auto &level : levelList) {
            if (level.value("net_layer", -1) != 0) {
                continue;
            }
            std::string netInstanceId = level.value("net_instance_id", "");
            // 新格式: sp_{sp}_srv_{srv}_idx_{k}; 旧格式: superPod{sp}_rack{srv}
            if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u", &spIdx,
                       &srvIdx) == 2 ||
                sscanf(netInstanceId.c_str(), "superPod%u_rack%u", &spIdx,
                       &srvIdx) == 2) {
                found = true;
                break;
            }
        }
        if (!found) {
            HCCL_VM_WARN(
                "rank local_id={} cannot determine superpod/server, skip",
                localId);
            continue;
        }
        topoMeta[spIdx][srvIdx].push_back(localId);
    }
    return !topoMeta.empty();
}

// topoMeta → 通信域server有序表:
// map迭代序(podId、serverId升序)即rankId分配顺序;
// 一次批量读取Server表建立(pod_id, server_id)→serverKey索引, 避免逐server查询
bool BuildCommConfigFromTopoMeta(const TopoMeta &topoMeta,
                                 CommConfigData &data) {
    auto allServers = RunnerDB::GetByPred<sim::Server>(
        [](const sim::Server &r) { return true; });
    std::map<std::pair<uint32_t, uint32_t>, uint64_t> serverKeyIndex;
    for (const auto &srv : allServers) {
        serverKeyIndex[std::make_pair(srv.pod_id, srv.server_id)] = srv.id;
    }

    data.servers.clear();
    uint32_t rankOffset = 0;
    for (const auto &[podId, superPod] : topoMeta) {
        for (const auto &[serverId, serverMeta] : superPod) {
            auto it = serverKeyIndex.find(std::make_pair(podId, serverId));
            if (it == serverKeyIndex.end()) {
                HCCL_VM_ERROR("server not found in static topo: podId={:d}, "
                              "serverId={:d}",
                              podId, serverId);
                return false;
            }
            CommConfigServer entry;
            entry.serverKey = it->second;
            entry.podId = podId;
            entry.serverId = serverId;
            entry.devNum = static_cast<uint32_t>(serverMeta.size());
            entry.rankOffset = rankOffset;
            entry.phyIds = serverMeta;
            rankOffset += entry.devNum;
            data.servers.push_back(std::move(entry));
        }
    }
    if (data.servers.empty()) {
        HCCL_VM_ERROR("no valid server in comm config");
        return false;
    }
    return true;
}

// 从TopoMetaConfig表项加载通信域配置并缓存(每进程仅一次)
bool LoadCommConfigData() {
    std::lock_guard<std::mutex> lock(g_comm_config_mutex);
    if (g_comm_config_loaded) {
        return true;
    }
    auto ret = RunnerDB::GetOneByPred<sim::TopoMetaConfig>(
        [](const sim::TopoMetaConfig &) { return true; });
    if (!ret.second) {
        HCCL_VM_ERROR("topo meta config not found, please run mock-comm first");
        return false;
    }
    std::string fileName(
        ret.first.file_name,
        strnlen(ret.first.file_name, sizeof(ret.first.file_name)));
    CommConfigData data;
    if (!ParseCommConfigData(fileName, nullptr, data)) {
        return false;
    }
    g_comm_config_data = std::move(data);
    g_comm_config_loaded = true;
    return true;
}
} // namespace

bool ParseCommConfigData(const std::string &fileName, const TopoMeta *topoMeta,
                         CommConfigData &data) {
    TopoMeta parsedMeta;
    const TopoMeta *meta = topoMeta;
    if (meta == nullptr) {
        if (fileName == "ranktable") {
            if (!ParseTopoMetaFromRankTableFile(parsedMeta)) {
                HCCL_VM_ERROR("parse ranktable failed");
                return false;
            }
        } else if (!ParseTopoMetaYaml(fileName, parsedMeta)) {
            HCCL_VM_ERROR("parse topo meta yaml failed: {}", fileName);
            return false;
        }
        meta = &parsedMeta;
    }
    data.fileName = fileName;
    return BuildCommConfigFromTopoMeta(*meta, data);
}

bool RefreshTopoMetaConfig(const std::string &fileName,
                           const TopoMeta *topoMeta) {
    if (fileName.empty() ||
        fileName.size() >= sizeof(sim::TopoMetaConfig::file_name)) {
        HCCL_VM_ERROR("invalid topo meta file name: {}", fileName);
        return false;
    }
    // 每次mock-comm执行覆盖表项: TopoMetaConfig仅记录当前通信域配置
    sim::TopoMetaConfig cfg{};
    strncpy(cfg.file_name, fileName.c_str(), sizeof(cfg.file_name) - 1);
    RunnerDB::DeleteAll<sim::TopoMetaConfig>();
    if (RunnerDB::Add<sim::TopoMetaConfig>(cfg) == 0) {
        HCCL_VM_ERROR("insert topo meta config failed: {}", fileName);
        return false;
    }
    // 同步刷新本进程缓存(其余进程为新进程, 首次使用时按表项自行加载)
    CommConfigData data;
    if (!ParseCommConfigData(fileName, topoMeta, data)) {
        HCCL_VM_ERROR("parse topo meta yaml failed: {}", fileName);
        return false;
    }
    std::lock_guard<std::mutex> lock(g_comm_config_mutex);
    g_comm_config_data = std::move(data);
    g_comm_config_loaded = true;
    HCCL_VM_INFO("refresh comm config: file={}, server num={}", fileName,
                 g_comm_config_data.servers.size());
    return true;
}

void ResetCommConfigData() {
    std::lock_guard<std::mutex> lock(g_comm_config_mutex);
    g_comm_config_data = CommConfigData{};
    g_comm_config_loaded = false;
}

bool GetServerKeyByRankId(uint32_t rankId, uint64_t &serverKey) {
    if (!LoadCommConfigData()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_comm_config_mutex);
    for (const auto &srv : g_comm_config_data.servers) {
        if (rankId >= srv.rankOffset && rankId < srv.rankOffset + srv.devNum) {
            serverKey = srv.serverKey;
            return true;
        }
    }
    HCCL_VM_ERROR("server not found by rankId:{:d}", rankId);
    return false;
}

aclError GetDevIndexByPhyId(uint64_t serverKey, uint32_t phyId,
                            uint32_t &devIndex) {
    if (!LoadCommConfigData()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    std::lock_guard<std::mutex> lock(g_comm_config_mutex);
    for (const auto &srv : g_comm_config_data.servers) {
        if (srv.serverKey != serverKey) {
            continue;
        }
        for (size_t idx = 0; idx < srv.phyIds.size(); ++idx) {
            if (srv.phyIds[idx] == phyId) {
                devIndex = static_cast<uint32_t>(idx);
                return ACL_SUCCESS;
            }
        }
        break; // 找到server但无匹配phyId
    }
    HCCL_VM_ERROR("dev index not found by phyId:{:d}, serverKey:{:d}", phyId,
                  serverKey);
    return ACL_ERROR_INVALID_PARAM;
}

uint64_t GetCurServerId() {
    if (g_cur_server_key != 0) {
        return g_cur_server_key;
    }

    // 3. 获取当前rankId
    uint32_t rankId = 0xFFFF;
    uint64_t serverKey = 0;
    if (!sim::GetRankIdByMPI(rankId, serverKey)) {
        HCCL_VM_ERROR("get rankId by MPI fail serverKey:{:d}", serverKey);
        return 0;
    }

    if (serverKey == 0 && rankId != 0xFFFF) {
        // 4. 根据rankId获取serverId:
        // 按TopoMetaConfig指向的配置推导rankId→server映射
        if (!sim::GetServerKeyByRankId(rankId, serverKey)) {
            HCCL_VM_ERROR("get serverKey by rankId:{:d} failed", rankId);
            return 0;
        }
    }
    HCCL_VM_INFO("Get current serverKey:{:d}", serverKey);

    g_cur_server_key = serverKey;
    return serverKey;
}
} // namespace sim
