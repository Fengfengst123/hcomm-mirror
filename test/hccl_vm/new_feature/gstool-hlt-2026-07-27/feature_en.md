# gstool hlt Test Case Support

**Release Date:** 2026-07-31

## Summary

The core of this release: adds gstool hlt test case support. Previously, the tool only supported running communication test cases via hccl_test. This release adds gstool integration, enabling hlt test case execution through gstool, expanding test coverage. Users can complete more testing scenarios through a unified tool interface.

## New Features

- Integrates gstool, supporting hlt test case execution through gstool
- Existing hccl_test test case functionality remains unchanged

## Usage Instructions

### Prerequisites

- Since it is not open source, it must be used in the yellow zone
- gstool is installed and configured

### Running hlt Test Cases

1. **Configure Proxy**

    ```bash
    export http_proxy="http://q30033976:xxxx@proxy.huawei.com:8080"
    export https_proxy="http://q30033976:xxxx@proxy.huawei.com:8080"
    ```

    Configure your own employee ID at `q30033976`, and `xxxx` is the w3 account password. If the password contains special characters, URL encoding is required. The encoding rules are as follows:

    | Character | Encoding | Character | Encoding | Character | Encoding | Character | Encoding | Character | Encoding |
    |------|------|------|------|------|------|------|------|------|------|
    | ! | %21 | # | %23 | $ | %24 | & | %26 | ' | %27 |
    | ( | %28 | ) | %29 | * | %2A | + | %2B | , | %2C |
    | . | %2E | / | %2F | : | %3A | ; | %3B | = | %3D |
    | ? | %3F | @ | %40 | [ | %5B | ] | %5D | | |

2. **Set hcomm and hccl Environment Variables**

    After pulling the hcomm open-source repository, the repository contains both hcomm and hccl.

    ![hcomm and hccl environment variable settings](./53DA2382-EC5B-4DDA-D0A6-E3124E4EBACE.png)

    Set the environment variables based on your own code path:

    ```bash
    export HCCL_CODE_HOME=/home/workspace/hccl
    export HCOMM_CODE_HOME=/home/workspace/hcomm
    ```

3. **CANN Package Upload**

    Download link: `hdfs-ngx0.turing-ci.hisilicon.com:14000` (accessed from the yellow zone), get the link with the newest suffix for the corresponding date to download the toolkit and ops packages, and copy the installation packages to the workspace. (The latest CANN package version is currently 9.2.0)

    **How to upload packages to the rapid workspace:**

    Each person's rapid workspace homepage has a Samba entry. Click to copy and get the Samba address.
    For example: `Code:\\7.240.90.165\\workcode/`

    On the Windows desktop, access the directory via Samba, find the home directory, create a new workspace folder, and drag the downloaded CANN package there.

    | Package Type | Package Name |
    |--------|------|
    | ops package | `Ascend-cann-950-ops_9.2.0_linux-x86_64.run` |
    | toolkit package | `Ascend-cann-toolkit_9.2.0_linux-x86_64.run` |

    Open the VSCode terminal in the rapid workspace, navigate to the corresponding directory, and execute the following CANN package installation commands:

    ```bash
    ./Ascend-cann-toolkit_9.2.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
    ./Ascend-cann-950-ops_9.2.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
    ```

4. **Build the Tool and Upload Test Packages**

    Build the HCCL-VM tool:
    
    Refer to the **3.3.1 Environment Configuration** section in the README.md file at https://gitcode.com/zhupc158/CheckerL2 to create and configure the hccl_rootinfo.json file.

    Navigate to the `hccl_vm` path, set the environment variables and build (all environment variables below should be set according to your own current paths):

    ```bash
    source /home/workspace/Ascend/cann/set_env.sh
    export HCCL_CODE_HOME=/home/workspace/hccl
    export HCOMM_CODE_HOME=/home/workspace/hcomm
    bash ./build.sh --full
    ```

    **Upload test packages:**

    Test package path: `http://hdfs-ngx1.turing-ci.hisilicon.com:14000/#/compilepackage/CI_Version/torino/br_hisi_trunk_ai/`

    After downloading the test package, copy it to the workspace.

    Example test package: `Ascend950-testcase-7.0.t9.0.b880-ubuntu24.04.x86_64.tar.gz`

    After extracting the test package, upload the following files to the workspace using the same method as above, typically placing them in the tool's `hccl_vm_install/bin` path:

    - `ccl_hlt_test.so` (extraction path: `test/lib/host/`)
    - `gstool`
    - `debug.json`: operator-related information, configure as needed

5. **Run Test Cases**
   
    **Execute via hccl-vm:**

    ```bash
    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
    export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
    # Set xxxx according to your own path
    cd /xxxx/hcomm/test/hccl_vm/hccl_vm_install/bin

    # Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
    ./hccl-vm start ascend950_cluster_32_server_normal.yaml

    # To enable the runner plugin (optional)
    (hvm)$> hccl-vm plugin install @runner

    # Select the communication domain configuration file for the current operator execution (run hccl_test test cases in a cluster environment with 1 supernode, 1 server, and 1 NPU)
    (hvm)$> hccl-vm mock-comm 112
    (hvm)$> ./gstool ccl_hlt_test.so new_hlt_test_main,debug.json

    # Run checker verification
    (hvm)$> hccl-vm plugin run @checker

    # Exit the tool terminal
    (hvm)$> exit
    ```

## Compatibility Notes

- The existing hccl_test test case execution method is not affected
- Ensure the gstool tool is deployed in the environment when using gstool
