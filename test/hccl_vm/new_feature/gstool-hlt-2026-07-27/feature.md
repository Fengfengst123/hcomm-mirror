# gstool 运行 hlt 用例支持

**发布日期：** 2026-07-31

## 摘要

本次发布核心：新增 gstool 运行 hlt 用例支持。此前工具仅支持通过 hccl_test 跑通信类用例，本次版本新增了对 gstool 的集成，支持使用 gstool 运行 hlt 测试用例，扩展了测试覆盖能力，用户可通过统一的工具入口完成更多场景的测试任务。

## 新功能

- 集成 gstool，支持通过 gstool 运行 hlt 测试用例
- 原有 hccl_test 用例功能保持不变

## 使用说明

### 前提条件

- 由于未开源，需在黄区使用
- 已安装并配置 gstool

### 运行 hlt 用例

1. **配置代理**

    ```bash
    export http_proxy="http://q30033976:xxxx@proxy.huawei.com:8080"
    export https_proxy="http://q30033976:xxxx@proxy.huawei.com:8080"
    ```

    其中 `q30033976` 处配置自己工号，`xxxx` 是 w3 账号的密码。密码中若有特殊字符需进行 URL 转义，转义规则如下：

    | 字符 | 转义 | 字符 | 转义 | 字符 | 转义 | 字符 | 转义 | 字符 | 转义 |
    |------|------|------|------|------|------|------|------|------|------|
    | ! | %21 | # | %23 | $ | %24 | & | %26 | ' | %27 |
    | ( | %28 | ) | %29 | * | %2A | + | %2B | , | %2C |
    | . | %2E | / | %2F | : | %3A | ; | %3B | = | %3D |
    | ? | %3F | @ | %40 | [ | %5B | ] | %5D | | |

2. **设置 hcomm 和 hccl 环境变量**

    拉取了 hcomm 开源仓后，开源仓中即包含 hcomm 和 hccl。

    ![hcomm和hccl环境变量设置](./53DA2382-EC5B-4DDA-D0A6-E3124E4EBACE.png)

    根据自己代码路径，设置环境变量：

    ```bash
    export HCCL_CODE_HOME=/home/workspace/hccl
    export HCOMM_CODE_HOME=/home/workspace/hcomm
    ```

3. **CANN包上传**

    下载链接：`hdfs-ngx0.turing-ci.hisilicon.com:14000`（黄区打开），获取对应日期后缀的 newest 的链接下载 toolkit 和 ops 包，并将安装包拷贝到工作空间上。（当前最新的 CANN 包版本号是 9.2.0）

    **如何上传包到急速空间：**

    每个人的急速空间主页有一个 Samba，点击复制，获取 Samba 地址。
    例如：`Code:\\7.240.90.165\\workcode/`

    在 Windows 桌面通过 Samba 进入目录后，找到 home 目录，新建 workspace 文件夹，将下载好的 CANN 包拖拽到此处。

    | 包类型 | 包名 |
    |--------|------|
    | ops 包 | `Ascend-cann-950-ops_9.2.0_linux-x86_64.run` |
    | toolkit 包 | `Ascend-cann-toolkit_9.2.0_linux-x86_64.run` |

    在急速空间内打开 VSCode 终端，找到对应目录，执行下面 CANN 包安装命令：

    ```bash
    ./Ascend-cann-toolkit_9.2.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
    ./Ascend-cann-950-ops_9.2.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
    ```

4. **编译工具并上传测试包**

    编译 HCCL-VM 工具：
    
    参考 https://gitcode.com/zhupc158/CheckerL2 下 README.md 文件中的 **3.3.1 环境配置** 章节，创建并配置hccl_rootinfo.json文件。

    进入到 `hccl_vm` 路径下，设置环境变量并编译（下面的环境变量都需要根据自己当前路径设置）：

    ```bash
    source /home/workspace/Ascend/cann/set_env.sh
    export HCCL_CODE_HOME=/home/workspace/hccl
    export HCOMM_CODE_HOME=/home/workspace/hcomm
    bash ./build.sh --full
    ```

    **上传测试包：**

    测试包路径：`http://hdfs-ngx1.turing-ci.hisilicon.com:14000/#/compilepackage/CI_Version/torino/br_hisi_trunk_ai/`

    下载测试包后，将测试包拷贝到工作空间上。

    示例测试包：`Ascend950-testcase-7.0.t9.0.b880-ubuntu24.04.x86_64.tar.gz`

    解压测试包后，将以下文件同上述方法一致上传到工作空间，一般可放置到工具的 `hccl_vm_install/bin` 路径下：

    - `ccl_hlt_test.so`（解压路径：`test/lib/host/`）
    - `gstool`
    - `debug.json`：算子相关信息，自行配置

5. **执行用例**
   
    **通过 hccl-vm 执行：**

    ```bash
    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
    export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
    # xxxx 根据自己路径设置
    cd /xxxx/hcomm/test/hccl_vm/hccl_vm_install/bin

    # 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
    ./hccl-vm start ascend950_cluster_32_server_normal.yaml

    # 如需启用 runner 插件（可选）
    (hvm)$> hccl-vm plugin install @runner

    # 选择本次算子执行的通信域配置文件（在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
    (hvm)$> hccl-vm mock-comm 112
    (hvm)$> ./gstool ccl_hlt_test.so new_hlt_test_main,debug.json

    # 执行 checker 校验
    (hvm)$> hccl-vm plugin run @checker

    # 退出工具终端
    (hvm)$> exit
    ```

## 兼容性说明

- 原有 hccl_test 用例运行方式不受影响
- 使用 gstool 需确保环境中已部署 gstool 工具
