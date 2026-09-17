/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_addr_info_log.h"
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>

/* ── 全局函数指针（初始为 NULL，TopoLogInit 填充） ── */
void (*g_topo_DlogRecord)(int moduleId, int level, const char* fmt, ...) = NULL;
int (*g_topo_CheckLogLevel)(int moduleId, int logLevel) = NULL;

/* 线程安全懒初始化：锁内判空 + 锁内 dlopen/dlsym */
void TopoLogInit(void)
{
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    static void* handle = NULL;
    pthread_mutex_lock(&mutex);
    if (g_topo_DlogRecord != NULL && g_topo_CheckLogLevel != NULL) {
        pthread_mutex_unlock(&mutex);
        return;
    }

    if (handle == NULL) {
        handle = dlopen("libunified_dlog.so", RTLD_NOW);
    }
    if (handle == NULL) {
        pthread_mutex_unlock(&mutex);
        return;
    }

    g_topo_DlogRecord = (void (*)(int, int, const char*, ...))dlsym(handle, "DlogRecord");
    g_topo_CheckLogLevel = (int (*)(int, int))dlsym(handle, "CheckLogLevel");

    if (g_topo_DlogRecord == NULL || g_topo_CheckLogLevel == NULL) {
        g_topo_DlogRecord = NULL;
        g_topo_CheckLogLevel = NULL;
    }
    /* handle 不 dlclose：成功时函数指针须指向存活库 */

    pthread_mutex_unlock(&mutex);
}
