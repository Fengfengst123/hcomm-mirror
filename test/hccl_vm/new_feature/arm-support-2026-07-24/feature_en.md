# ARM Architecture Support

**Release Date:** 2026-07-31

## Summary

The core of this release: adds ARM architecture support. Previously, the tool only supported x86 architecture. This release extends compatibility with the Huawei Cloud EulerOS 2.0 (aarch64) platform, enabling the tool to compile and run in ARM environments, achieving dual x86 and ARM architecture coverage.

## New Features

- Adds compilation support for ARM architecture
- Adapts kernel interface calls for the ARM platform
- The tool can compile and run normally in ARM environments

## Usage Instructions

1. **Log in to the ARM Environment**

    Using the Huawei Cloud EulerOS 2.0 (aarch64) environment as an example.

    **Blue zone computer:**

    ```bash
    ssh 113.46.4.206
    ```

    Enter port number `101`, user `root`, password `xxxx`.

    After logging in, continue to hop:

    ```bash
    ssh root@192.168.0.50
    ```

2. **Upload the CANN Package**

    Download link: `hdfs-ngx0.turing-ci.hisilicon.com:14000`, get the link with the newest suffix for the corresponding date to download the toolkit and ops packages. (The latest CANN package version is currently 9.2.0)

    Log in to the ARM server via the first step. In MobaXterm, you can see the `113.46.4.206` path on the left side. Drag and drop the downloaded CANN package to upload it.

    Find the internal IP of `192.168.0.50` using the `ifconfig` command, then upload to the `192.168.0.50` server path via scp:

    ```bash
    scp -r /home/q30033976* root@192.168.0.50:/home/q30033976/workspace
    ```

    Execute the following CANN package installation commands:

    ```bash
    ./Ascend-cann-toolkit_9.2.0_linux-aarch64.run --install --install-path=/home/workspace/Ascend
    ./Ascend-cann-950-ops_9.2.0_linux-aarch64.run --install --install-path=/home/workspace/Ascend
    ```

3. **Build the Tool**

    Refer to the **3.3.1 Environment Configuration** section in the README.md file at https://gitcode.com/zhupc158/CheckerL2 to create and configure the hccl_rootinfo.json file.
    
    Refer to the **3.2 Manual Build & Installation** section in the README.md file at https://gitcode.com/zhupc158/CheckerL2.

    Since ARM and x86 environments differ, the third-party dependency installation commands also differ:

    **x86 environment installation command (apt):**

    ```bash
    sudo apt-get update
    sudo apt install build-essential cmake libsqlite3-dev libboost-all-dev rdma-core libibverbs-dev pkg-config
    ```

    **ARM environment equivalent command (dnf):**

    ```bash
    sudo dnf install gcc gcc-c++ make cmake sqlite-devel boost-devel rdma-core libibverbs-devel pkgconf-pkg-config
    ```
    
4. **Build hccl_test**

    Refer to the **4.2.2 MPICH Environment Compilation** section in the README.md file at https://gitcode.com/zhupc158/CheckerL2 to build the hccl_test test cases.

    Since x86 uses the OpenMPI environment for compilation and ARM uses the MPICH environment:

    **Environment variable configuration ARM environment equivalent command:**

    ```bash
    export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/aarch64-linux/devlib:$LD_LIBRARY_PATH
    ```

    **Build hccl_test test cases ARM environment equivalent command:**

    ```bash
    make MPI_HOME=/usr/local/mpich ASCEND_DIR=${ASCEND_HOME_PATH}
    ```

5. **Run Test Cases**

    **Configure environment variables:**

    ```bash
    # Enter the tool installation directory
    cd /xxxx/hccl_vm/hccl_vm_install
    source /xxxxx/Ascend/cann/set_env.sh
    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
    export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
    ```

    **Execute via hccl-vm:**

    ```bash
    # Navigate to the new bin directory to run hccl-vm
    cd /xxxx/hccl_vm/hccl_vm_install/bin

    # Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
    ./hccl-vm start ascend950_cluster_32_server_normal.yaml

    # To enable the runner plugin (optional)
    (hvm)$> hccl-vm plugin install @runner

    # Select the communication domain configuration file for the current operator execution (run hccl_test test cases in a cluster environment with 1 supernode, 1 server, and 1 NPU)
    (hvm)$> hccl-vm mock-comm 112
    (hvm)$> mpirun -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

    # Run checker verification
    (hvm)$> hccl-vm plugin run @checker

    # Exit the tool terminal
    (hvm)$> exit
    ```

---

## Compatibility Notes

- The original x86 architecture functionality is not affected
- Ensure dependency library version compatibility in the ARM environment
