# Core v2 contract spike boundary

This directory is an executable contract spike only. It is not a production
package and it is not installed by the repository packaging configuration.

```text
NOT_PRODUCTION
NOT_INSTALLED
NO_NETWORK
NO_LLM
NO_REMOTE_COMPUTE
NO_GPU
```

The spike uses only the Python standard library. It has no network, shell,
credential, model-provider, remote-compute, or GPU authority. It must not
import `ai4s_agent` and must not be promoted into `src/molly/` by this Goal.
