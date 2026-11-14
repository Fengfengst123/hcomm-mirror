/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_DEVICE_MEMORY_MANAGER_H
#define SIM_DEVICE_MEMORY_MANAGER_H

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace sim {
namespace devspec {
    struct AddrSpace;
} // namespace devspec

class DeviceMemoryManager {
public:
    // 获取单例实例
    static DeviceMemoryManager& GetInstance();

    // 禁止拷贝和赋值
    DeviceMemoryManager(const DeviceMemoryManager&) = delete;
    DeviceMemoryManager& operator=(const DeviceMemoryManager&) = delete;

    /**
     * @brief 分配设备虚拟地址（per-segment first-fit，见设计文档 §4.5.7）。
     *
     * 地址来自 rank 窗口内 user_mem 段（= rank 窗口扣除 CCU 窗口后的补集，A5/A6 均为
     * 1022 GiB / 2 段）。单次分配的块**必须完整落在单个 user_mem 段内**：段内余量不足时
     * 跳到下一段（浪费该段尾部余量，浪费量严格 < 单次请求大小）。
     *
     * ★ 禁止改回"单一全局线性水位 + 一次映射"——那样分配块会跨越 CCU 空洞
     *   （反例见设计文档 §4.5.7.2）。
     *
     * @return 失败返回 nullptr（devPhyId 非法 / size 为 0 / 地址空间耗尽 / 无可用 spec）
     */
    void* AllocVirMem(uint64_t devPhyId, size_t size);

    // 分配 GLOBAL 对称虚拟内存：bit63 段、无 device 前缀，跨 rank 同序同值（供对称窗口 tag 配对）。
    // 单例每进程一个、游标各自从 0 起，仅整数编址不占地址空间。
    void* AllocGlobalVirMem(size_t size);

    // 分配 UB 对称堆基址：bit62 段，按全局堆序依次上移，跨 rank 同址。
    // 单堆跨度 = round_up(size, 16GB 步距)：hcomm 按 stride × lsaTeamSize 申请
    // （如 64p = 16GB × 64 = 1TB），成员窗口分布在 base + member*stride，
    // 预留必须覆盖实际 teamSize，不再固定 32 成员上限。
    // 仅整数编址，不占地址空间、不分配物理内存。
    void* AllocSymHeapVirMem(size_t size);

    /**
     * @brief 释放设备虚拟地址 —— **有意不回收**（设计文档 §4.5.7.5 / D17 / O13）。
     *
     * 原因：aclrt_memory_stub.cc:474-482 记载的不变量「标记行与活跃行永不同址共存」
     * 依赖地址永不复用（FR1 128p / R3K4 256p 修复：释放路径只置 is_freed=1 保留行，
     * 让慢 rank 的迟到跨 rank 解析仍命中；而 7 处范围查询故意不带 is_freed 过滤）。
     * 一旦回收并复用 VA，同一地址会同时被旧行与新行覆盖，GetOneByPred 返回不确定。
     * 预算充分性：1022 GiB vs 实测单 rank 峰值 0.405 GiB（2523 倍余量）。
     */
    void FreeVirMem(uint64_t devPhyId, void* virAddr);

    // 分配物理内存
    void* AllocPhyMem(const char* name, uint64_t deviceId, size_t size);

    // 释放物理内存
    void FreePhyMem(const char* name, uint64_t deviceId);

    // 获取物理内存
    void* AcquirePhyMem(const char* name, uint64_t deviceId, size_t size);

    // 释放物理内存
    int ReleasePhyMem(const char* name, uint64_t deviceId);

    // 缓存设备地址到host地址；deviceKey 为归属设备键，与 devPtr 组成复合键。
    // 单进程多 rank 下 device 子进程布局一致，裸 devPtr 数值跨 rank 重复，必须带 deviceKey 消歧。
    void MapDevPtrHostPtr(uint64_t deviceKey, void* devPtr, void* hostPtr);

    // 去缓存设备地址到host地址（deviceKey 同 MapDevPtrHostPtr）
    void UnmapDevPtrHostPtr(uint64_t deviceKey, void* devPtr);

    // 获取host地址；deviceKey==0（调用线程无 device 上下文）时打 ERROR 并返回 nullptr
    void* GetHostPtrByDevPtr(uint64_t deviceKey, void* devPtr);

    /// 查询某 device 当前各 user_mem 段的已用字节数（水位日志 / UT 用）。
    std::vector<uint64_t> GetUsedPerSegment(uint64_t devPhyId);

private:
    // 私有构造函数
    DeviceMemoryManager();
    ~DeviceMemoryManager();

    /**
     * @brief 解析 devPhyId 对应的地址空间（带缓存）。
     *
     * 解析顺序：① 缓存命中；② DB Device.soc_version → DeviceSpecRegistry::FindBySocVersion；
     * ③ 兜底 DeviceSpecRegistry::CommonAddrSpace()（A5/A6 的 address_space 逐字段相同，
     *    DB 未就绪时仍能得到正确的分段布局）；④ 全部失败返回 nullptr，调用方降级。
     */
    const devspec::AddrSpace* ResolveAddrSpace(uint64_t devPhyId);

private:
    // 静态单例实例
    static DeviceMemoryManager* s_instance;
    static std::mutex s_instanceLock;

    std::mutex m_virSizeLock;
    /**
     * 每 device 的 **per-segment** 线性水位（段序与 AddrSpace::userMemSegs 一致）。
     * ★ key 为 uint64_t：原实现是 std::map<int, uint64_t> 却用 uint64_t devPhyId 索引，
     *   存在隐式窄化（latent bug B2）。
     */
    std::map<uint64_t, std::vector<uint64_t>> m_deviceMemCursors;
    /// devPhyId → 已解析的地址空间（指向 DeviceSpecRegistry 内的常驻对象）
    std::map<uint64_t, const devspec::AddrSpace*> m_addrSpaceCache;
    /// 降级模式（无 spec 可用）已告警过的 device，避免日志刷屏
    std::map<uint64_t, bool> m_degradedWarned;

    // GLOBAL 对称 VA（bit63）游标：每进程独立从 0 起，跨 rank 同序同值。
    std::mutex m_globalVirMemLock;
    uint64_t m_globalVirMemCursor = 0;

    // UB 对称堆（bit62）堆序计数与 VA 偏移：全局不含 device，跨 rank 堆序一致以得相同 heapBase。
    // 偏移按各堆实际跨度（round_up(size, 16GB)）累加，变长堆之间不会重叠。
    std::mutex m_symHeapLock;
    uint32_t m_symHeapCount = 0;
    uint64_t m_symHeapOffset = 0;

    std::mutex m_devPtrHostPtrLock;
    // 设备地址到host地址映射表；key = (deviceKey, devPtr)，device 盲地址加设备维度保证多 rank 唯一
    std::map<std::pair<uint64_t, void*>, void*> m_devPtrHostPtrMap;
};
} // namespace sim

#endif // SIM_DEVICE_MEMORY_MANAGER_H
