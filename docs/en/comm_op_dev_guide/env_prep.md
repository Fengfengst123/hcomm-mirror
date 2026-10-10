# Environment Setup

<!-- md-trans-meta sourceCommit=f613c76cef5703d4701e6a9ab2fa1b4633784cff translatedAt=2026-08-11T06:56:33.407Z pushedAt=2026-08-20T11:39:14.555Z -->

1. Install the driver, firmware, and CANN software package.

    The development and use of HCCL communication operators depend on the CANN software package. For detailed installation steps, see *[CANN Software Installation Guide](https://hiascend.com/en/document/redirect/CannCommunityInstSoftware)*.

2. Set the environment variables.

    Before building and running the program, you need to set the CANN software environment variables.

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

    **/usr/local/Ascend** is the default installation path of CANN for the **root** user. If you install CANN as a non-root user or in a custom path, replace it with the actual path.
