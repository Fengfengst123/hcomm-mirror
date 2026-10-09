/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dev_aicpu_ts_hccs_channel.h"
#include <securec.h>
#include "dispatcher_ctx.h"
#include "adapter_hal_pub.h"

namespace hccl {
u64 DevAicpuTsHccsChannel::commSeq_{0};

DevAicpuTsHccsChannel::~DevAicpuTsHccsChannel()
{
    for (auto& pair : slots_) {
        if (pair.second.transport != nullptr) {
            (void)pair.second.transport->DeInit();
            pair.second.transport.reset();
        }
        if (pair.second.dispatcherCtx != nullptr) {
            (void)DestroyDispatcherCtx(pair.second.dispatcherCtx, pair.second.commId);
        }
    }
    slots_.clear();
}

HcclResult
DevAicpuTsHccsChannel::SetTransportMachinePara(hccl::MachinePara& machinePara, const HcclChannelHccsRes& channelHccsRes)
{
    machinePara.linkAttribute = 0x03; /* 0x03同时支持目的端和源端发起 */

    machinePara.remoteDeviceId = channelHccsRes.remoteDevicePhyId;
    machinePara.localDeviceId = channelHccsRes.localDevicePhyId;
    machinePara.deviceLogicId = channelHccsRes.localDeviceLogicId;

    machinePara.deviceType = static_cast<DevType>(channelHccsRes.deviceType);
    machinePara.tag = channelHccsRes.channelTag;
    machinePara.machineType = channelHccsRes.machineType;

    // 非910_93 2die sio与hccs并发场景，specifyLink设置为RESERVED_LINK_TYPE，平台层将按实际链路类型建链
    machinePara.specifyLink = LinkTypeInServer::RESERVED_LINK_TYPE;
    machinePara.isNewOneSide = true;

    machinePara.localBufSize = channelHccsRes.localBufSize;
    machinePara.remoteBufSize = channelHccsRes.remoteBufSize;
    machinePara.localBufMem = channelHccsRes.localBufMem;
    machinePara.remoteBufMem = channelHccsRes.remoteBufMem;

    HCCL_INFO(
        "%s success, linkAttribute[%x], localUserRank[%u], remoteWorldRank[%u], "
        "remoteUserrank[%u], deviceLogicId[%d], localDeviceId[%d], deviceType[%d], newTag[%s], "
        "specifyLink[%d], machineType[%u], localBufSize[%u],remoteBufSize[%u], localBufMem[%p], remoteBufMem[%p]",
        __func__, machinePara.linkAttribute, machinePara.localUserrank, machinePara.remoteWorldRank,
        machinePara.remoteUserrank, machinePara.deviceLogicId, machinePara.localDeviceId, machinePara.deviceType,
        machinePara.tag.c_str(), machinePara.specifyLink, static_cast<u32>(machinePara.machineType),
        machinePara.localBufSize, machinePara.remoteBufSize, machinePara.localBufMem, machinePara.remoteBufMem);
    return HCCL_SUCCESS;
}

HcclResult DevAicpuTsHccsChannel::Create(
    const void* blob, u64 blobBytes, const HcommDeviceInfo& deviceInfo, ChannelHandle& outHandle)
{
    CHK_PTR_NULL(blob);
    if (blobBytes < sizeof(HcclChannelHccsRes)) {
        HCCL_ERROR("[DevAicpuTsHccsChannel][Create] blob too small[%llu]", static_cast<unsigned long long>(blobBytes));
        return HCCL_E_PARA;
    }

    const HcclChannelHccsRes& channelHccsRes = *static_cast<const HcclChannelHccsRes*>(blob);
    const HcclChannelP2p& channelP2p = channelHccsRes.channelP2p;

    // 创建Transport对象
    MachinePara machinePara;
    CHK_RET(SetTransportMachinePara(machinePara, channelHccsRes));
    machinePara.notifyNum = channelHccsRes.p2pNotifyNum;

    TransportDeviceP2pData transDevP2pData;
    transDevP2pData.inputBufferPtr = nullptr;
    transDevP2pData.outputBufferPtr = nullptr;

    //  获取transportAttr信息
    transDevP2pData.transportAttr = channelP2p.transportAttr;

    //  创建Transport对象
    TransportPara para{};
    const std::unique_ptr<hccl::NotifyPool> notifyPool;

    u32 devId = 0;
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(channelHccsRes.localDevicePhyId, &devId));

    // 每个 channel 使用独立的 dispatcherCtx，避免多 channel 共享同一调度器
    char commId[sizeof(HccsSlot::commId)];
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++commSeq_;
        int nc = snprintf_s(
            commId, sizeof(commId), sizeof(commId) - 1U, "hcomm_ts_hccs_%d_%llu", deviceInfo.deviceLogicId,
            static_cast<unsigned long long>(commSeq_));
        CHK_PRT_RET(nc < 0, HCCL_ERROR("[DevAicpuTsHccsChannel][Create] snprintf_s failed"), HCCL_E_INTERNAL);
    }
    DispatcherCtxPtr dispatcherCtx{nullptr};
    CHK_RET(CreateDispatcherCtx(&dispatcherCtx, devId, commId));
    CHK_PTR_NULL(dispatcherCtx);

    DispatcherCtx* ctx = static_cast<DispatcherCtx*>(dispatcherCtx);
    CHK_PRT(ctx->SetDispatcherHcclQos(channelHccsRes.channelP2p.qos)); // 调度器添加hcclQos
    CHK_PTR_NULL(ctx);

    machinePara.dctxPtr = dispatcherCtx;

    std::shared_ptr<Transport> transport;
    transport.reset(new (std::nothrow) Transport(
        TransportType::TRANS_TYPE_DEVICE_P2P, para, ctx->GetDispatcher(), notifyPool, machinePara, transDevP2pData));
    if (transport == nullptr) {
        (void)DestroyDispatcherCtx(dispatcherCtx, commId);
        HCCL_ERROR("[DevAicpuTsHccsChannel][Create] Transport alloc failed");
        return HCCL_E_PTR;
    }

    HcclResult tr = transport->Init(); // 初始化需要增加远端用户注册内存
    if (tr != HCCL_SUCCESS) {
        transport.reset();
        (void)DestroyDispatcherCtx(dispatcherCtx, commId);
        return tr;
    }

    outHandle = reinterpret_cast<ChannelHandle>(transport.get());
    HccsSlot slot;
    slot.dispatcherCtx = dispatcherCtx;
    slot.transport = std::move(transport);
    slot.tag = channelHccsRes.channelTag;
    CHK_SAFETY_FUNC_RET(memcpy_s(slot.commId, sizeof(slot.commId), commId, sizeof(commId)));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.emplace(outHandle, std::move(slot));
    }
    HCCL_INFO("[DevAicpuTsHccsChannel][%s] transport[%p] create done", __func__, outHandle);
    return HCCL_SUCCESS;
}

bool DevAicpuTsHccsChannel::Destroy(ChannelHandle handle)
{
    HccsSlot slot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = slots_.find(handle);
        if (it == slots_.end()) {
            return false;
        }
        slot = std::move(it->second);
        slots_.erase(it);
    }
    if (slot.transport != nullptr) {
        (void)slot.transport->DeInit();
        slot.transport.reset();
    }
    if (slot.dispatcherCtx != nullptr) {
        (void)DestroyDispatcherCtx(slot.dispatcherCtx, slot.commId);
    }
    HCCL_DEBUG("[DevAicpuTsHccsChannel][Destroy] destroyed handle[0x%llx]", handle);
    return true;
}
} // namespace hccl
