# ARM 架构支持

**发布日期：** 2026-07-31

## 摘要

本次发布核心：新增 ARM 架构支持。此前工具仅支持 x86 架构，本次版本扩展了对 Huawei Cloud EulerOS 2.0（aarch64）平台的适配能力，使工具可在 ARM 环境下正常编译与运行，实现 x86 与 ARM 双架构覆盖。

## 新功能

- 新增对 ARM 架构的编译支持
- 适配 ARM 平台下的内核接口调用
- 工具可在 ARM 环境下正常编译与运行

## 使用说明

1. **登录 ARM 环境**

    以 Huawei Cloud EulerOS 2.0（aarch64）环境为例。

    **蓝区电脑：**

    ```bash
    ssh 113.46.4.206
    ```

    输入端口号 `101`，用户 `root`，密码 `xxxx`。

    登录进来后，继续跳转：

    ```bash
    ssh root@192.168.0.50
    ```

2. **上传 CANN 包**

    下载链接：`hdfs-ngx0.turing-ci.hisilicon.com:14000`，获取对应日期后缀的 newest 的链接下载 toolkit 和 ops 包。（当前最新的 CANN 包版本号是 9.2.0）

    通过第一步登录 ARM 服务器，在 MobaXterm 左侧可以看到 `113.46.4.206` 路径，将下载的 CANN 包拖拽上传。

    查找 `192.168.0.50` 的内网 IP，命令是 `ifconfig`，通过 scp 命令上传到 `192.168.0.50` 服务器路径下：

    ```bash
    scp -r /home/q30033976* root@192.168.0.50:/home/q30033976/workspace
    ```

    执行下面 CANN 包安装命令：

    ```bash
    ./Ascend-cann-toolkit_9.2.0_linux-aarch64.run --install --install-path=/home/workspace/Ascend
    ./Ascend-cann-950-ops_9.2.0_linux-aarch64.run --install --install-path=/home/workspace/Ascend
    ```

3. **编译工具**

    参考 https://gitcode.com/zhupc158/CheckerL2 下 README.md 文件中的 **3.3.1 环境配置** 章节，创建并配置hccl_rootinfo.json文件。
    
    参考 https://gitcode.com/zhupc158/CheckerL2 下 README.md 文件中的 **3.2 手动构建&安装** 章节。

    由于 ARM 和 x86 环境有区别，第三方依赖安装命令也有差异：

    **x86 环境安装命令（apt）：**

    ```bash
    sudo apt-get update
    sudo apt install build-essential cmake libsqlite3-dev libboost-all-dev rdma-core libibverbs-dev pkg-config
    ```

    **ARM 环境等效命令（dnf）：**

    ```bash
    sudo dnf install gcc gcc-c++ make cmake sqlite-devel boost-devel rdma-core libibverbs-devel pkgconf-pkg-config
    ```
    
4. **编译 hccl_test**

    参考 https://gitcode.com/zhupc158/CheckerL2 下 README.md 文件中的 **4.2.2 MPICH环境编译** 章节，编译 hccl_test 用例。

    由于 x86 使用 OpenMPI 环境编译，ARM 使用 MPICH 环境编译。

    **配置环境变量 ARM 环境等效命令：**

    ```bash
    export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/aarch64-linux/devlib:$LD_LIBRARY_PATH
    ```

    **编译 hccl_test 用例 ARM 环境等效命令：**

    ```bash
    make MPI_HOME=/usr/local/mpich ASCEND_DIR=${ASCEND_HOME_PATH}
    ```

5. **执行用例**

    **配置环境变量：**

    ```bash
    # 进入工具安装目录
    cd /xxxx/hccl_vm/hccl_vm_install
    source /xxxxx/Ascend/cann/set_env.sh
    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
    export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
    ```

    **通过 hccl-vm 执行：**

    ```bash
    # 需要进入到新的bin文件目录下执行hccl-vm
    cd /xxxx/hccl_vm/hccl_vm_install/bin

    # 选择昇腾集群拓扑配置文件，启动工具，初始化集群环境，进入工具命令行
    ./hccl-vm start ascend950_cluster_32_server_normal.yaml

    # 如需启用runner插件（可选）
    (hvm)$> hccl-vm plugin install @runner

    # 选择本次算子执行的通信域配置文件（在1个超节点1个Server1个NPU的集群环境运行hccl_test用例）
    (hvm)$> hccl-vm mock-comm 112
    (hvm)$> mpirun -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

    # 执行checker校验
    (hvm)$> hccl-vm plugin run @checker

    # 退出工具终端
    (hvm)$> exit
    ```

---

## 兼容性说明

- 原有 x86 架构功能不受影响
- ARM 环境下需确保依赖库版本兼容
