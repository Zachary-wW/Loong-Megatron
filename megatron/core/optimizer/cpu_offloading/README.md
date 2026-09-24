## How to use ?

Add these flags to enable optimizer cpu offload in MCore.

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--use-precision-aware-optimizer
```

## Configuration Recommendations

Gradient copy from GPU to CPU, CPU optimizer step, and subsequent parameter copy from CPU to GPU can be time-consuming operations, and it is recommended to use the flag `--overlap-cpu-optimizer-d2h-h2d` to execute them concurrently.

## Optional DeepSpeed CPU Adam

With Adam CPU offload, `OptimizerConfig.use_deepspeed_cpu_adam` defaults to
`True`. When DeepSpeed can be imported, CPU updates use `DeepSpeedCPUAdam`;
the GPU optimizer remains the configured GPU implementation. DeepSpeed is an
optional dependency and is imported only when this backend is selected.

Use `--no-use-deepspeed-cpu-adam` to select PyTorch AdamW on the CPU. The existing
`--use-torch-optimizer-for-cpu-offload` option takes precedence and selects
PyTorch AdamW for both CPU and GPU Adam updates. If the DeepSpeed import fails,
a warning is emitted and CPU updates fall back to PyTorch AdamW. Failures while
building or running DeepSpeed's native CPU extension are not silently caught.

DeepSpeed uses FP32 Adam moment buffers for the offloaded FP32 master
parameters. Incompatible requested moment dtypes are normalized with a warning;
restored CPU moments are converted to FP32 and `step` is restored as a Python
integer. Hybrid checkpoints retain the actual step source used by each backend:
parameter state for DeepSpeed/PyTorch, and parameter groups for TE/Apex FusedAdam.
