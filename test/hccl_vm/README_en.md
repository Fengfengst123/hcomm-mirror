# HCCL-VM User Guide

## 1. Overview

HCCL-VM is a high-performance collective communication virtual execution environment for Huawei Ascend NPU cards. This tool aims to enable HCCL collective communication operator development and functional verification without real Ascend hardware.

![hccl-vm GIF](docs/hccl-vm.gif)

## 2. Prerequisites

|   Dependency   |     Version Requirement     |
| -------- | ------------- |
| System Architecture  | x86_64 Ubuntu 22.04 or later |
| Specification Constraints  | Ascend950, otherwise refer to [Specification Constraints](#45-tool-specification-constraints) |

### 2.1 CANN Package Installation

Install the latest version of CANN Toolkit development kit and CANN ops operator packages [Download Link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/)

```bash
# Ensure the installation packages have executable permissions
chmod +x Ascend-cann-toolkit_9.1.0_linux-x86_64.run
chmod +x Ascend-cann-950-ops_9.1.0_linux-x86_64.run
# Installation commands
./Ascend-cann-toolkit_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
./Ascend-cann-950-ops_9.1.0_linux-x86_64.run --install --install-path=/home/workspace/Ascend
```


### 2.2 hccl_test Compilation

hccl_test is the official HCCL performance testing tool provided by Ascend. For details, see [HCCL Performance Test Tool](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/devaids/hccltool/HCCLpertest_16_0001.html). HCCL-VM supports running hccl_test cases in the virtual environment. Please first follow the [hccl_test Case Build](#42-hccl-test-case-build) section to compile the test binary.

Note: Optional, PyTorch test cases will be supported in the future.

---

## 3. Quick Start

### 3.1 Manual Build & Installation

```bash
# 1. Create the working directory
mkdir -p /home/workspace
cd /home/workspace

# 2. Download dependency source code
git clone https://gitcode.com/cann/hccl.git
git clone https://gitcode.com/cann/hcomm.git

# 3. Install third-party dependencies
sudo apt-get update
sudo apt install build-essential cmake libsqlite3-dev libboost-all-dev rdma-core libibverbs-dev pkg-config gcc-aarch64-linux-gnu g++-aarch64-linux-gnu qemu-user-static binfmt-support

# 4. Build the HCCL-VM tool. After downloading the hcomm code, the tool source path is: /home/workspace/hcomm/test/hccl_vm
cd /home/workspace/hcomm/test/hccl_vm
source /home/workspace/Ascend/cann/set_env.sh
export HCCL_CODE_HOME=/home/workspace/hccl
export HCOMM_CODE_HOME=/home/workspace/hcomm
bash ./build.sh --full

# 5. Copy and extract aicpu_hcxx.tar.gz from the CANN installation directory
bash build_pkg.sh
```

### 3.2 Usage Examples

#### 3.2.1 Environment Configuration

If the file `hccl_rootinfo.json` already exists under the `/etc` path, please manually delete it.

#### 3.2.2 CCU Mode

1. Environment variable configuration.

```bash
# Enter the tool installation directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
```

2. Execution


```bash
# Enter the new bin directory to execute hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communication domain configuration file for this operator execution (run hccl_test case in a cluster environment of 1 supernode, 1 Server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Execute checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify hccl_test case execution results
[View Runner Results](#481-runner-plugin-results) 
[View Checker Results](#482-checker-plugin-results)

#### 3.2.3 AICPU Mode

In AICPU expansion mode, the algorithm expansion steps are executed on the device side. Therefore, the hccl-vm tool needs to compile and simulate the execution of HCCL device-side symbols. Since device-side symbols are ARM architecture, cross-compilation is required when building on x86 environments, and QEMU is required for simulated execution of AICPU mode at runtime.

1. Environment variable configuration.

```bash
# Enter the tool installation directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AI_CPU"
```

2. Execution

```bash
# Enter the new bin directory to execute hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communication domain configuration file for this operator execution (run hccl_test case in a cluster environment of 1 supernode, 1 Server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Execute checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify hccl_test case execution results [View Runner Results](#481-runner-plugin-results) [View Checker Results](#482-checker-plugin-results)

#### 3.2.4 HostDPU Mode

The environment variable configuration and other runtime settings for HostDPU expansion mode are the same as AICPU mode, with the following differences only in cluster scale and communication domain scale:

1. Cluster specification configuration should use `./hccl-vm start ascend950_cluster_32_server_normal_hostdpu.yaml`
2. Requires Server count >= 2, i.e., single-Server communication domains are not supported (112.yaml, 114.yaml, 118.yaml, etc. are not supported)

#### 3.2.5 AIV Mode

1. Environment variable configuration.

```bash
# Enter the tool installation directory
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install
source /home/workspace/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH
export RANK_TABLE_FILE=$(pwd)/data/ranktable.json
export HCCL_OP_EXPANSION_MODE="AIV"
```

2. Execution

```bash
# Enter the new bin directory to execute hccl-vm
cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin

# Select the Ascend cluster topology configuration file, start the tool, initialize the cluster environment, and enter the tool command line
./hccl-vm start ascend950_cluster_32_server_normal.yaml

# To enable the runner plugin (optional)
(hvm)$> hccl-vm plugin install @runner

# Select the communication domain configuration file for this operator execution (run hccl_test case in a cluster environment of 1 supernode, 1 Server, and 1 NPU)
(hvm)$> hccl-vm mock-comm 112
(hvm)$> mpirun --allow-run-as-root --oversubscribe -np 2 ${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1 > log.txt

# Execute checker verification
(hvm)$> hccl-vm plugin run @checker

# Exit the tool terminal
(hvm)$> exit
```

3. Verify hccl_test case execution results [View Runner Results](#481-runner-plugin-results) [View Checker Results](#482-checker-plugin-results)

### 3.3 PyTorch Test Case Examples

Not yet supported.

### 3.4 HCCL Code Modification Verification Example

If you have modified the CANN operator package code, such as adding new algorithm types, follow these steps to ensure your changes take effect. The build_pkg.sh script helps users perform packaging, installation, and copying of device-side dependencies. Before execution, set the following environment variables:

```bash
# Assuming your CANN installation directory is: /home/workspace/Ascend
source /home/workspace/Ascend/cann/set_env.sh
# Configure the hccl code repository path
export HCCL_CODE_HOME=/home/workspace/hccl
# Configure the hcomm code repository path
export HCOMM_CODE_HOME=/home/workspace/hcomm
```

1. If you updated/modified CANN hccl repository code, execute `bash build_pkg.sh --install hccl`.
2. If you updated/modified CANN hcomm repository code, execute `bash build_pkg.sh --install hcomm`.
3. If you updated/modified both CANN hccl and hcomm repository code, execute `bash build_pkg.sh --full`.
4. Follow the [Usage Examples](#32-usage-examples) steps to re-run the test cases.

---

## 4. Detailed Guide

### 4.1 Tool Environment Variable Configuration

**HCCL-VM Environment Variables Description**:

| Environment Variable                      | Purpose                                                                                                        | Example                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------- |
| `HCCL_CODE_HOME`         | Specifies the HCCL source code path for HCCL-VM compilation. Not configured by default.     | `export HCCL_CODE_HOME=/home/workspace/hccl`                     |
| `HCOMM_CODE_HOME`         | Specifies the HCOMM source code path for HCCL-VM compilation. Not configured by default.     | `export HCOMM_CODE_HOME=/home/workspace/hcomm`                     |
| `HCCLVM_ENABLE_DUMP_DATA` | 1. Enable performance simulation file output to disk. If enabled, 3 runner_hcclvm_*_data.bin files and 1 ccu_channel_jetty_config.xml file will be output during test case execution for performance simulation. <br> 2. Enable Runner plugin dump of input & output data. If enabled, the input & output data of each operator will be dumped to the all\_rank\_input\_output.txt file during test case execution. | `export HCCLVM_ENABLE_DUMP_DATA=1 to enable, export HCCLVM_ENABLE_DUMP_DATA=0 or leave unset to disable` |

### 4.2 HCCL-Test Case Build

The hccl_test case source code is located in the CANN package installation directory. It supports compilation and execution in both OpenMPI and MPICH environments. For runtime differences, see [OpenMPI and MPICH Environment Case Execution Differences](#47-openmpi-and-mpich-environment-case-execution-differences). This guide uses the OpenMPI environment as an example.

#### 4.2.1 OpenMPI Environment Compilation

1. Install OpenMPI

```bash
sudo apt-get update
sudo apt install openmpi-bin libopenmpi-dev
```

2. Compile hccl_test

```bash
# Modify CANN installation directory permissions
chmod -R 755 /home/workspace/Ascend

# Enter the hccl_test case source directory
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Temporarily modify the Makefile script
if ! grep -q '\-lmpi_cxx' Makefile; then
    sed -i 's/-lmpi/-lmpi -lmpi_cxx/g' Makefile
fi

# Compile hccl_test cases
MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi make ASCEND_DIR=${ASCEND_HOME_PATH}
```

#### 4.2.2 MPICH Environment Compilation

Assuming the mpich path is: `/usr/lib/mpich`.

```bash
# Enter the hccl_test case source directory
cd /home/workspace/Ascend/cann/tools/hccl_test

# Set CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Configure environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH

# Compile hccl_test cases
make MPI_HOME=/usr/lib/mpich/ ASCEND_DIR=${ASCEND_HOME_PATH}
```

### 4.3 Ascend Cluster Topology Configuration File Description

#### 4.3.1 Server/Pod Topology Configuration File Description

An Ascend cluster topology consists of one or more Server/Pod sub-topologies combined according to CLOS layered network rules. Therefore, users need to confirm the topology type of each Server/Pod before generating the cluster topology.
Users can choose from predefined topology types provided by the HCCL-VM tool, or customize Server/Pod topology types according to the configuration file format requirements.

Describing the topology network relationships of a Server/Pod mainly includes the following aspects:

 - **Port Configuration Table**: Describes the physical port configuration of an NPU card, such as NPU-to-NPU direct connection ports (P2P), NPU outgoing ports (P2NET), etc.
 - **Link Configuration Table**: Describes the connection relationships between all NPU cards within a Server/Pod, such as full mesh connections.
 - **PortBound**: Describes the binding relationships of certain ports of an NPU card, i.e., multiple ports bound into a PortGroup.

```yaml
type: "server_intra_links"
name: "ascend950_links_topo_demo"
description: "Ascend950 chip normal topology connection description file"

soc_version: "Ascend950"
device_num: 16

device_ports_allocate_map:
 # port allocation table: 0: not used, 1: device direct connection, 2: device to switch, 3: d2h port
 #                 portId:  0  1  2  3  4  5  6  7  8
    - {die_id: 0, pin_map: [1, 1, 1, 0, 2, 2, 2, 2, 3]} # die0
    - {die_id: 1, pin_map: [0, 0, 0, 0, 0, 0, 0, 0, 0]} # die1

# port_group: describes which ports are merged into a portGroup; ports in the same portGroup share the same IP address
port_group:
    - {layer: 0, ports: ["0/4", "0/5", "0/6", "0/7"]}

links:
  # ── Description method: each row of 8 devices forms a full mesh ──
  - link_mode: "fullmesh"
    connections:
      # The following example indicates: devices 0, 1, 2, 3 of die0 are fully connected through die0's ports
      - {die_id: 0, devices_range: [0, 3]}
      - {die_id: 0, devices_range: [4, 7]}
      - {die_id: 0, devices_range: [8, 11]}
      - {die_id: 0, devices_range: [12, 15]}
  
  #- link_mode: "enum"
  #  device_to_device_links:
  #    The following example indicates: devices 0 and 1 connect through die0's ports to devices 1, 3, 5, 7's die1 ports respectively
  #            i.e.: device0 connects with device1, device3, device5, device7; device1 connects with device3, device5, device7
  #    - {src_die_id: 1, src_local_id_range: [0, 2], dst_die_id: 1, dst_local_id_range: [1, 3, 5, 7]}

  - link_mode: "enum"
    device_to_switch_links:
      # The following example indicates: devices 0 through 15 all connect to the switch through die0's ports. Combined with portGroup, devices 0 through 15 connect to the switch through portGroup[0/4, 0/5, 0/6, 0/7].
      - {die_id: 0, devices_range: [0, 15]}

```

**Field Descriptions**:

 - **soc_version**: Chip model, e.g., `Ascend950`.
 - **device_num**: Total number of devices, determined by chip model and topology type.
 - **device_ports_allocate_map**: Port allocation table, describing the port configuration of each die. 1 indicates device direct connection ports, 2 indicates device-to-switch ports, 3 indicates d2h ports.
 - **port_group**: Describes which ports are merged into a portGroup; ports in the same portGroup share the same IP address. If not configured, each port defaults to its own portGroup.
 - **links**: Link configuration table, describing the connection relationships between all NPU cards within a Server/Pod, and between NPUs and switches.
    - **NPU Direct Connection**: The tool provides two methods to configure NPU direct connections:
     - **link_mode == "fullmesh"**: All Devices are fully connected based on the Ports of one Die. New typical connection methods can be added as new link_mode types, e.g., "ring".
     - **link_mode == "enum"**: Enumeration method. When the NPU connection within a Server/Pod is complex, all link relationships can be enumerated.
    - **NPU-to-Switch Connection**: Users can configure NPU-to-switch connections using the enumeration method.
  - **device_to_device_links**: Describes the connection relationships between NPUs.
  - **device_to_switch_links**: Describes the connection relationships between NPUs and switches.

#### 4.3.2 Cluster Topology Configuration File Description

An Ascend cluster network consists of one or more Server/Pod sub-topologies combined according to CLOS layered network rules. Users can select different Server/Pod topology types based on cluster scale and requirements.

Users can customize the cluster topology configuration file according to the following format:

```yaml
name: "ascend950_cluster_32_server_normal"
description: "Ascend 950 normal network: 32 supernodes, each with 1 server"

# Total number of supernodes
super_node_num: 4
# Total number of servers/pods
server_num: 32
server_list:
  # Servers 0-7: all use ascend950_server_topo_normal topology type
  - {super_pod_id: 0, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 1, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 2, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}
  - {super_pod_id: 3, id_range: [0, 7], soc_version: "Ascend950", server_topo: "ascend950_server_topo_normal.yaml"}

```

The above configuration file describes a cluster topology with 4 supernodes, 32 Servers, totaling 128 NPU cards. Each Server/Pod uses the ascend950_server_topo_normal topology type.

**Field Descriptions**:

 - **super_node_num**: Total number of supernodes.
 - **server_num**: Total number of servers/pods.
 - **server_list**: Configuration information for each Server/Pod, including supernode ID, device ID range, chip model, and Server/Pod topology configuration file path.

#### 4.3.3 Communication Domain Configuration File Description

In an Ascend cluster environment, users need to select different communication domain configuration files based on the communication domain required by the operator to be executed.

The tool provides the `hccl-vm mock-comm` command to read and configure the operator communication domain configuration file. The communication domain configuration file is in YAML format, located at `hccl_vm_install/config/topo_meta`. If the directory does not contain the corresponding communication domain configuration file, users need to create one first.

The hccl-vm tool supports asymmetric topology communication domain configuration. As shown below:

```yaml
# 1. Global statistics: podNum, serNum, rankNum are all less than 1024
meta:
  podNum: 1  # Total number of supernodes
  serNum: 2  # Total number of servers
  rankNum: 6 # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      - serId: 0
        # Local IDs of ranks actually running on each server
        ranks: [0, 2]
      - serId: 1
        # Local IDs of ranks actually running on each server
        ranks: [1, 3, 5, 7]
```

**Notes**:

- When configuring a communication domain, the tool will regenerate the topo.json and ranktable.json files based on the specified communication domain configuration number.
- In the communication domain configuration YAML file above, the `ranks` field represents the local ID list (i.e., the physical device ID) of ranks actually running on each server.

When the communication domain scale is large, server selection fields and rank selection fields in the servers entries can be freely combined, supporting the following fields:

- **Server selection fields** (choose one, cannot be configured simultaneously):
  - `serId`: A single server ID or a list of server IDs (e.g., `3` or `[0, 1]`, where the list represents each enumerated server);
  - `serId_range`: Server ID range, value is `[<start_id>, <end_id>]` (closed interval, only supports list format), representing all servers from start to end ID under the supernode.
- **Rank selection fields** (choose one, cannot be configured simultaneously):
  - `ranks`: List of local IDs of ranks actually running on each server;
  - `ranks_range`: Local ID range, value is `[<start_id>, <end_id>]` (closed interval, only supports list format), representing all local IDs from start to end for each server.

The following example (2 supernodes, 4 servers total, 32 ranks) demonstrates mixed usage of multiple combinations within the same file:

```yaml
# 1. Global statistics: podNum, serNum, rankNum are all less than 1024
meta:
  podNum: 2  # Total number of supernodes
  serNum: 4  # Total number of servers
  rankNum: 32 # Total number of ranks

# 2. Detailed topology structure
topology:
  - podId: 0
    servers:
      # serId as an ID list, representing server 0 and server 1 under pod 0
      - serId: [0, 1]
        ranks: [0, 1, 2, 3, 4, 5, 6, 7]
  - podId: 1
    servers:
      - serId: 0
        ranks_range: [0, 7]
      - serId: 1
        ranks_range: [0, 7]
```

**Notes**:

- `serId` and `serId_range` cannot be configured simultaneously; `ranks` and `ranks_range` cannot be configured simultaneously.
- `serId_range` and `ranks_range` only support `[<start_id>, <end_id>]` list format (closed interval), not `<start_id>-<end_id>` format; start ID cannot be greater than end ID, and the range width (end_id - start_id + 1) cannot exceed 1024.
- A `serId_range` entry will expand to all servers from start to end ID under that supernode; a `serId` list entry represents each enumerated server, with the same ranks for each server.
- If no rank selection field is configured in a server entry, the server's ranks will be an empty list; if no server selection field is configured, `serId` defaults to 0.

#### 4.3.4 topo.json and ranktable.json File Description

The `topo.json` and `ranktable.json` files do not need to be created manually. The tool will automatically generate them based on the following information:

- **Topology Configuration Number**: The number specified by the user at startup (e.g., 112, 113, etc.)
- **Chip Type**: The chip type automatically identified from the runtime environment.

Although the configuration files are auto-generated by the tool, understanding their structure helps in understanding the topology configuration.

**topo.json Structure**:

`topo.json` describes the connection relationships of all devices within a server:

```json
{
  "server": {
    "device_count": 8,
    "groups": [
      {
        "group_id": 0,
        "device_start": 0,
        "device_count": 8,
        "topo_layout": "1D"
      }
    ]
  },
  "ports": [
    {
      "ccu": "die0",
      "port_pattern": "0/{0-6}",
      "protocol": "HCCS",
      "func_id": 2,
      "usage": "peer2peer",
      "ip_binding": "independent"
    },
    {
      "ccu": "die0",
      "port_pattern": "0/7,0/8",
      "protocol": "ROCE",
      "func_id": 3,
      "usage": "peer2net",
      "ip_binding": "independent"
    }
  ],
  "links": [
    {
      "net_layer": 0,
      "link_type": "PEER2PEER",
      "topo_type": "1DMESH",
      "ccu": "die0",
      "port_pattern": "0/{0-6}",
      "connect_pattern": "full_mesh",
      "group_id": 0
    },
    {
      "net_layer": 1,
      "link_type": "PEER2NET",
      "topo_type": "CLOS",
      "ccu": "die0",
      "port_pattern": "0/7,0/8",
      "connect_pattern": "all_to_net",
      "group_id": 0
    }
  ]
}
```

**Field Descriptions**:

- `server.device_count`: Total number of devices.
- `server.groups`: Device grouping information.
- `ports`: Port configuration.
  - `usage`: Port purpose (`peer2peer` indicates inter-device connections, `peer2net` indicates external connections)
- `links`: Link configuration.
  - `link_type`: Link type (`PEER2PEER` or `PEER2NET`)
  - `topo_type`: Topology type (`1DMESH`, `CLOS`, etc.)

**ranktable.json Structure**:

`ranktable.json` describes the devices and IP mappings used in the current run:

```json
{
  "version": "1.0",
  "server_count": 1,
  "device_count": 8,
  "server_list": [
    {
      "server_id": 0,
      "device_id": 0,
      "device_ip": "192.168.1.10",
      "port": "2222"
    }
  ]
}
```

**Field Descriptions**:

- `server_count`: Number of servers.
- `device_count`: Total number of devices.
- `server_list`: Server and device list.
  - `device_ip`: Device IP address.
  - `port`: Device port number.

### 4.4 hccl\_config.sh File Description

The hccl\_config.sh file contains the environment variable configurations required for running HCCL\_Test cases. The environment variables are consistent with those used in the real hardware environment for HCCL\_Test cases.
Users need to modify the hccl\_config.sh script according to their test cases and requirements to configure HCCL case runtime environment variables.

```bash
#!/bin/bash
# hccl_config.sh - HCCL Environment Variable Configuration

remove_files_by_prefix() {
  if [ "$#" -ne 1 ]; then
    echo "Usage: remove_files_by_prefix <prefix>" >&2
    return 2
  fi

  local prefix="$1"
  if [ -z "$prefix" ]; then
    return 0
  fi

  shopt -s nullglob
  local any_deleted=0
  for f in "${prefix}"*; do
    if [ -f "$f" ]; then
      rm -f -- "$f" && any_deleted=1
    fi
  done
  shopt -u nullglob

  # Return 0 regardless of whether files were deleted, to ensure the script continues execution
  return 0
}

# Clean up redundant files in the data/ directory (temporary files generated in CCU mode)
cd "${HCCL_VM_INSTALL_DIR}/data" 2>/dev/null && {
  remove_files_by_prefix "sqe_info_rank_"
  remove_files_by_prefix "mc_instr_info_rank_"
  rm -f "all_rank_input_output.txt"
  cd "${HCCL_VM_INSTALL_DIR}"
}

# Set CANN environment variables
source /home/workspace/Ascend/cann/set_env.sh

# Disable hccl heartbeat feature
export HCCL_DFS_CONFIG=cluster_heartbeat:off

# Set HCCL-VM installation path, inferred from the script's own location (compatible with bin/ and script/ subdirectories)
_INSTALL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$(basename "${_INSTALL_SCRIPT_DIR}")" in
    bin|script)
        export HCCL_VM_INSTALL_DIR="$(dirname "${_INSTALL_SCRIPT_DIR}")"
        ;;
    *)
        export HCCL_VM_INSTALL_DIR="${_INSTALL_SCRIPT_DIR}"
        ;;
esac
unset _INSTALL_SCRIPT_DIR

# Configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/devlib:$LD_LIBRARY_PATH

# Set ranktable.json file path (consistent with mock-comm generation path)
export RANK_TABLE_FILE=${HCCL_VM_INSTALL_DIR}/data/ranktable.json

# Set log level
export ASCEND_GLOBAL_LOG_LEVEL=1

# Enable log output to screen
export ASCEND_SLOG_PRINT_TO_STDOUT=1

# Set HCCL runtime mode (CCU, AI_CPU, AIV and other expansion modes)
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# Or set HCCL runtime parameters (AI_CPU expansion mode) AI_CPU mode environment variables cannot be set simultaneously with other modes
# export HCCL_OP_EXPANSION_MODE="AI_CPU"
# Or set HCCL runtime parameters (AIV expansion mode) AIV mode environment variables cannot be set simultaneously with other modes
# export HCCL_OP_EXPANSION_MODE="AIV"

echo "HCCL-VM environment configured successfully!"

```

### 4.5 Tool Specification Constraints

**Supported Operator Types**:

The tool supports the following operator types: allgather/allreduce/alltoall/reduce/reduce\_scatter/scatter/alltoallv.

**Supported Data Types**:

HCCL-VM tool supports the following data types: int8/int16/int32/fp16/fp32/uint8/uint16/uint32/bfp16/hif8/fp8e4m3/fp8e5m2/fp8e8m0.

HCCL-VM Runner plugin supports the following data types:

| ReduceOp | DataType                           |
| -------- | ---------------------------------- |
| `ADD`    | `int8/int16/int32/uint8` |
| `MIN`    | `int8/int16/int32/uint8` |
| `MAX`    | `int8/int16/int32/uint8` |

**Hardware Specifications**:

Currently, this tool only supports Ascend950 chips. A single server supports a maximum of 8 cards; for more than 8 cards, cross-server execution is required.

### 4.6 HCCL-VM Plugin Features

#### 4.6.1 Runner Plugin

The Runner plugin simulates the execution of tasks generated by HCCL business orchestration and outputs the result data.
The simulator runner plugin is **disabled** by default during hccl\_test case execution. After the hccl\_test case calls the operator interface, it waits for the operator task to complete via the aclrtSynchronizeStream interface. The simulator runner tool waits until all ranks are in the waiting state, then starts simulated execution of all ranks' tasks. After execution is complete, it notifies each rank's test case to continue execution.
After case execution completes, users can view each rank's input buffer and output buffer data through the "all\_rank\_input\_output.txt" file in the execution directory. This feature is not enabled by default; users can enable it before case execution using the corresponding command.

**Installation and Uninstallation**:

The Runner plugin supports installation and uninstallation via the `hccl-vm plugin install/uninstall` command. The runner plugin must be installed after entering the hccl-vm tool command line and before executing test cases; subsequent executions will then run the runner.

```bash
# Install runner plugin
(hvm)$> hccl-vm plugin install @runner

# Uninstall runner plugin
(hvm)$> hccl-vm plugin uninstall @runner
```

#### 4.6.2 Checker Plugin

The Checker plugin, i.e., the algorithm analyzer plugin, forms a DAG graph from all tasks generated by HCCL and analyzes the DAG graph to determine whether memory conflicts exist; it also simulates DAG graph execution to detect semantic errors and other issues.
The algorithm analyzer plugin is manually launched by the user via command.

The Checker plugin currently uses Checker V3 to perform single-operator verification. Checker V3 big graph verification is enabled by default: multiple operators within each sync window are merged into a single big graph for cross-operator synchronous resource conflict checking. Both single-operator Checker V3 and big graph verification can be independently controlled via configuration parameters in `manifest.json`; the legacy Checker has been removed, and `enable_old_checker` is no longer a valid configuration item.

```json

# Configuration file located at /pathto/hccl_vm_install/plugin/checker/manifest.json

{
  "name": "checker",        // Checker plugin name
  "version": "1.0.0",       // Checker plugin version
  "entry": "./checker",     // Checker plugin launch command
  "dependency": {
      "min_core_version": "1.0.0"
  },
  "setting": {              // Checker plugin configuration items
      "enable_new_checker": true,           // Whether to enable Checker V3 single-operator verification (enabled by default)
      "enable_big_graph_checker": true,     // Whether to enable Checker V3 big graph verification (enabled by default, independent of single-operator verification)
      "enable_insight_dump": false,         // Whether to output Checker V3 Insight data (disabled by default)
      "enable_memory_snapshot_dump": false, // Whether to output memory snapshot data (disabled by default, requires Insight data output to be enabled first)
      "enable_dag_graphviz_dump": false     // Whether to output big graph Graphviz DOT files (disabled by default)
  }
}
```

### 4.7 OpenMPI and MPICH Environment Case Execution Differences

Before running hccl_test cases, users can determine which mpirun is being used in the current environment via the `which` command.

#### 4.7.1 Environment Variable Configuration Differences

The environment is generally configured with OpenMPI by default. If users run cases with OpenMPI, no additional environment variable configuration is typically needed.
If users run cases with the MPICH environment, they need to configure environment variables as follows:

```bash
# Configure mpich environment variables
export LD_LIBRARY_PATH=/usr/lib/mpich/lib/:${ASCEND_HOME_PATH}/lib64/:${ASCEND_HOME_PATH}/x86_64-linux/devlib:$LD_LIBRARY_PATH
export PATH=/usr/lib/mpich/bin:$PATH
```

#### 4.7.2 mpirun Command Parameter Differences

For the OpenMPI environment, users run hccl_test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun --allow-run-as-root --oversubscribe -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Descriptions**:

 - --allow-run-as-root: OpenMPI-specific parameter that allows running MPI processes as root user, used for running in environments without root privileges.
 - --oversubscribe: OpenMPI-specific parameter that removes CPU slot limits, allowing a single node to launch [number of processes > number of CPU logical cores], i.e., over-subscription.
 - -np 2: Specifies the number of processes as 2, consistent with the number of nodes.

For the MPICH environment, users run hccl_test cases with the following command:

```bash
export HCCL_TEST_PATH=/home/workspace/Ascend/cann/tools/hccl_test
mpirun -np 2 ${HCCL_TEST_PATH}/bin/reduce_scatter_test -b 64 -e 64 -d int32 -o sum -w 0 -n 1 -c 1
```

**Parameter Descriptions**:

 - -np 2: Specifies the number of processes as 2, consistent with the number of nodes.

### 4.8 Viewing Results

#### 4.8.1 Runner Plugin Results

If the runner plugin is installed by executing `hccl-vm plugin install @runner` in the hccl-vm terminal, the runner plugin execution will be automatically triggered after the operator workflow completes. The final results depend on hccl_test verification. Users should check for [error] level logs and the final verification result in the redirected log file:

```bash
data_size(Bytes): | aveg_time(us): | alg_bandwidth(GB/s): | check_result:
64                | 1000.00        | 0.00006              | success
```

#### 4.8.2 Checker Plugin Results

After executing `hccl-vm plugin run @checker` in the hccl-vm terminal, the Checker verification process and results will be printed in the terminal. Users should check for [error] level logs and the final verification result:

Big graph verification is executed once per sync window. Big graph verification failure does not block other verification processes. Failures can be identified by `error` level logs such as `BigGraphCheckerV3 failed` or `Big graph sync-conflict check failed`.

```bash
[info][PID:144373][TID:144880][main.cc][RunChecker] [RunChecker] op[0] Checker Success.
```

---
### 4.9 Large Memory Block Reuse (Check-Only Mode)

Check-only mode is used for large-scale cluster scenarios that only run Checker verification. When enabled, large memory allocations of 200MB to 4GB reuse the same 4GB shared region `HcclCommPool`, shared across all ranks with mutual overwrite allowed, significantly reducing `/dev/shm` usage. In this mode, large block contents are not guaranteed to be correct; it is only applicable to the Checker V3 verification pipeline that does not read buffer data. Do not enable this mode when numerically correct results are required.

Check-only mode is a session-level toggle, explicitly enabled by appending `--check-only` after the `start` subcommand; when not specified, the default normal mode applies, where large blocks use real independent allocation with no correctness loss. Allocations smaller than 200MB always use real allocation; single blocks larger than 4GB are directly rejected with an error in check-only mode. Check-only mode is not mutually exclusive with Runner, but when check-only mode is enabled and Runner is installed, large block reuse still takes effect and may overwrite Runner data; the tool will print a warning.

```bash

# Enable check-only mode when starting the tool
./hccl-vm start ascend950_cluster_32_server_normal.yaml --check-only
```

***

## 5 Appendix

### Open-Source Third-Party Software Dependencies

The following third-party open-source software is required when building this project. For offline compilation scenarios, download and rename the packages, then place them in the third_party directory within this project.

| Software       | Version          | Download URL |
| ------------  | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| CLI11         | 2.2.0         | [cli11-2.2.0.tar.gz](https://raw.gitcode.com/src-openeuler/cli11/blobs/58c912141164a5c0f0139bfa91343fefe151d525/cli11-2.2.0.tar.gz) |
| json          | 3.11.3        | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| spdlog        | 1.11.0        | [spdlog-v1.11.0.tar.gz](https://raw.gitcode.com/src-openeuler/spdlog/blobs/c2dfb1aca26c607393665c836155613ff283de66/v1.11.0.tar.gz) |
| yaml-cpp      | 0.8.0         | [yaml-cpp-0.8.0.tar.gz](https://raw.gitcode.com/src-openeuler/yaml-cpp/blobs/d1ead4fff417073b9cdbf98b8b55eb0efc00b0ba/yaml-cpp-0.8.0.tar.gz) |
| sqlite        | 3.51.0        | [sqlite-amalgamation-3510300.zip](https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip) |
| googletest    | 1.14.0        | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| cann-cmake    | master-044    | [cmake-master-044.tar.gz](https://raw.gitcode.com/cann/cmake/archive/refs/heads/master-044.tar.gz) |

### Glossary

| Term       | Description                                                      |
| -------- | ------------------------------------------------------- |
| HCCL     | Huawei Collective Communication Library         |
| NPU      | Neural Processing Unit                          |
| CANN     | Compute Architecture for Neural Networks, Huawei Ascend AI processor software stack |
| MPI      | Message Passing Interface                        |
| CCU      | Collective Communication Unit                    |
| Topology | Topology, device connection relationships                                               |
| Rank     | Process number, identifier in MPI                                         |

---

**Document Version**: v1.1.
**Last Updated**: 2026-06-30.