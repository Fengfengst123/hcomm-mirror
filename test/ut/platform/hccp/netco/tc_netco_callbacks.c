/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bkf_dc_pri.h"
#include "bkf_sys_log.h"

/* Keep checks active in release builds, too. */
#define CHECK(expr)                                                           \
    do {                                                                      \
        if (!(expr)) {                                                        \
            fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #expr); \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

typedef struct {
    unsigned int calls;
    void* cookie;
    void* value;
    uint8_t* buffer;
    int32_t length;
    char result[32];
} CallbackState;

typedef struct {
    F_BKF_TMR_TIMEOUT_PROC callback;
    void* param;
    uint32_t interval;
    unsigned int starts;
    unsigned int stops;
    unsigned int outputs;
} TimerState;

static void* TestAlloc(void* cookie, uint32_t len, const char* str, uint16_t num)
{
    (void)str;
    (void)num;
    void* ptr = calloc(1, len);
    if (ptr != NULL) {
        ++*(unsigned int*)cookie;
    }
    return ptr;
}

static void TestFree(void* cookie, void* ptr, const char* str, uint16_t num)
{
    (void)str;
    (void)num;
    if (ptr != NULL) {
        CHECK(*(unsigned int*)cookie > 0);
        --*(unsigned int*)cookie;
        free(ptr);
    }
}

static int32_t CompareKey(void* left, void* right)
{
    uint32_t a = *(const uint32_t*)left;
    uint32_t b = *(const uint32_t*)right;
    return (a > b) - (a < b);
}

static char* GetValueStr(void* cookie, void* value, uint8_t* buffer, int32_t length)
{
    CallbackState* state = cookie;
    ++state->calls;
    state->cookie = cookie;
    state->value = value;
    state->buffer = buffer;
    state->length = length;
    return state->result;
}

static void TestDcCallback(BkfMemMng* mem)
{
    /* Only table registration is needed; no slice, iterator or job is created. */
    BkfDc dc = {0};
    dc.argInit.memMng = mem;
    VOS_AVLL_INIT_TREE(
        dc.tableTypeSet, (AVLL_COMPARE)Bkfuint16_tCmp,
        BKF_OFFSET(BkfDcTableType, vTbl) + BKF_OFFSET(BkfDcTableTypeVTbl, tableTypeId),
        BKF_OFFSET(BkfDcTableType, avlNode));
    CallbackState state = {.result = "registered-cookie"};
    BkfDcTableTypeVTbl table
        = {.name = "callback",
           .tableTypeId = 1,
           .cookie = &state,
           .tupleCntMax = 1,
           .tupleKeyLen = sizeof(uint32_t),
           .tupleValLen = sizeof(uint32_t),
           .tupleKeyCmp = CompareKey,
           .tupleValGetStrOrNull = GetValueStr};
    uint32_t value = 0x12345678;
    uint8_t buffer[64] = {0};
    CHECK(BkfDcRegTableType(&dc, &table) == BKF_OK);
    CHECK(BkfDcGetTupleValStr(&dc, table.tableTypeId, &value, buffer, sizeof(buffer)) == state.result);
    CHECK(state.calls == 1 && state.cookie == &state && state.value == &value);
    CHECK(state.buffer == buffer && state.length == sizeof(buffer));
    CHECK(strcmp(state.result, "registered-cookie") == 0);

    CHECK(strcmp(BkfDcGetTupleValStr(&dc, table.tableTypeId, NULL, buffer, sizeof(buffer)), "-") == 0);
    CHECK(strcmp(BkfDcGetTupleValStr(&dc, table.tableTypeId, &value, NULL, sizeof(buffer)), "paramNg") == 0);
    CHECK(strcmp(BkfDcGetTupleValStr(&dc, table.tableTypeId, &value, buffer, 0), "paramNg") == 0);
    CHECK(state.calls == 1);
    BkfDcUnregTableType(&dc, table.tableTypeId);
    CHECK(strcmp(BkfDcGetTupleValStr(&dc, table.tableTypeId, &value, buffer, sizeof(buffer)), "**") == 0);
    CHECK(state.calls == 1);

    table.tupleValGetStrOrNull = NULL;
    CHECK(BkfDcRegTableType(&dc, &table) == BKF_OK);
    uint8_t expected[64] = {0};
    char* expectedStr = BKF_GET_MEM_STD_STR(&value, sizeof(value), expected, sizeof(expected));
    CHECK(strcmp(BkfDcGetTupleValStr(&dc, table.tableTypeId, &value, buffer, sizeof(buffer)), expectedStr) == 0);
    CHECK(state.calls == 1);
    BkfDcUnregTableType(&dc, table.tableTypeId);
}

static void* StartTimer(void* cookie, F_BKF_TMR_TIMEOUT_PROC callback, uint32_t interval, void* param)
{
    TimerState* state = cookie;
    state->callback = callback;
    state->param = param;
    state->interval = interval;
    ++state->starts;
    return state;
}

static void StopTimer(void* cookie, void* id)
{
    TimerState* state = cookie;
    CHECK(id == state);
    ++state->stops;
}

static uint32_t PrintLog(uint32_t param, uint32_t id, const char* fmt, ...)
{
    (void)param;
    (void)id;
    (void)fmt;
    return BKF_OK;
}

static uint32_t OutputLog(void* cookie, void* key, void* val, F_BKF_SYS_LOG_PRINTF logPrintf, uint32_t param)
{
    TimerState* state = cookie;
    CHECK(*(uint32_t*)key == 1 && *(uint32_t*)val == 42);
    CHECK(logPrintf == PrintLog && param == 0);
    ++state->outputs;
    return BKF_OK;
}

static void TestTimerCallback(BkfMemMng* mem)
{
    TimerState state = {0};
    BkfITmr timer
        = {.name = "timer",
           .memMng = mem,
           .cookie = &state,
           .startOnce = StartTimer,
           .startLoop = StartTimer,
           .stop = StopTimer};
    BkfTmrMng* tmrMng = BkfTmrInit(&timer);
    CHECK(tmrMng != NULL);
    BkfDispInitArg dispArg = {.name = "display", .memMng = mem};
    BkfDisp* disp = BkfDispInit(&dispArg);
    CHECK(disp != NULL);
    BkfSysLogInitArg arg
        = {.name = "log", .memMng = mem, .disp = disp, .tmrMng = tmrMng, .printf = PrintLog, .restrainIntervalMs = 100};
    BkfSysLogMng* log = BkfSysLogInit(&arg);
    CHECK(log != NULL);
    BkfSysLogTypeVTbl table
        = {.name = "table",
           .cookie = &state,
           .typeId = 1,
           .needRestrain = VOS_TRUE,
           .keyLen = sizeof(uint32_t),
           .valLen = sizeof(uint32_t),
           .keyCmp = CompareKey,
           .out = OutputLog};
    CHECK(BkfSysLogReg(log, &table) == BKF_OK);
    uint32_t key = 1;
    uint32_t value = 42;
    CHECK(BkfSysLogFunc(log, table.typeId, &key, &value) == BKF_OK);
    CHECK(state.starts == 1 && state.outputs == 1 && state.interval == arg.restrainIntervalMs);
    CHECK(state.callback != NULL && state.param != NULL);
    CHECK(BkfSysLogFunc(log, table.typeId, &key, &value) == BKF_OK);
    CHECK(state.starts == 1 && state.outputs == 1);
    /* Use the actual callback and BkfSysLog argument captured by the timer adapter. */
    CHECK(state.callback(state.param, &state) == BKF_OK);
    CHECK(state.stops == 1);
    CHECK(BkfSysLogFunc(log, table.typeId, &key, &value) == BKF_OK);
    CHECK(state.starts == 2 && state.outputs == 2);
    BkfSysLogUninit(log);
    CHECK(state.stops == 2);
    BkfDispUninit(disp);
    BkfTmrUninit(tmrMng);
}

int main(void)
{
    unsigned int allocations = 0;
    BkfIMem memory = {.name = "callback_test", .cookie = &allocations, .malloc = TestAlloc, .free = TestFree};
    BkfMemMng* mem = BkfMemInit(&memory);
    CHECK(mem != NULL);
    TestDcCallback(mem);
    TestTimerCallback(mem);
    BkfMemUninit(mem);
    CHECK(allocations == 0);
    puts("netco callback tests passed");
    return EXIT_SUCCESS;
}
