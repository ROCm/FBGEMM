import sys
import os
import itertools
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import CacheAlgorithm, PoolingMode

# Path setup
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "test"))

from tbe.training.backward_adagrad_common import execute_backward_adagrad

# --- Target Configuration ---

# 1. Target Dimensions (Exact values hardcoded in the HIP template)
#    The specialized kernel only compiles for these specific values.
#    We divide by 4 because the test harness multiplies D by 4 internally.
TARGET_KERNEL_D_VALUES = [64, 128, 160, 192, 256, 320]

# 2. Precision Combinations (Weights, Output Gradients)
#    We test:
#    - Pure FP32
#    - Pure FP16 (c10::half)
#    - Mixed Precision (FP32 weights / FP16 grads)
#    - Mixed Precision (FP16 weights / FP32 grads)
PRECISION_CONFIGS = [
    (SparseType.FP32, SparseType.FP32, "FP32/FP32"),
    (SparseType.FP16, SparseType.FP16, "FP16/FP16"),
    (SparseType.FP32, SparseType.FP16, "Mixed FP32/FP16"),
    (SparseType.FP16, SparseType.FP32, "Mixed FP16/FP32"),
]

# 3. Pooling Modes (The kernel supports both via host-side pre-processing)
POOLING_MODES = [PoolingMode.SUM, PoolingMode.MEAN]

# 4. Weighted Modes (The build system generates distinct kernels for both)
WEIGHTED_MODES = [False, True]

print("==================================================================")
print("Starting Comprehensive HIP Specialized Kernel Sweep")
print(f"Target Kernel Dimensions: {TARGET_KERNEL_D_VALUES}")
print("Optimizer: rowwise_adagrad (Required for specialized kernel)")
print("==================================================================\n")

failures = []

# Iterate over all combinations
for kernel_d in TARGET_KERNEL_D_VALUES:
    # Ensure divisibility for harness
    if kernel_d % 4 != 0:
        print(f"Skipping Kernel D={kernel_d} (Not divisible by 4 for harness)")
        continue
    
    input_d = kernel_d // 4
    
    for (weight_dtype, grad_dtype, desc), pool_mode, is_weighted in itertools.product(
        PRECISION_CONFIGS, POOLING_MODES, WEIGHTED_MODES
    ):
        # --- Constraint Enforcement ---
        # The test harness assumes: assume(pooling_mode == PoolingMode.SUM or not weighted)
        if is_weighted and pool_mode != PoolingMode.SUM:
            continue
        
        print(f"Testing: Kernel_D={kernel_d} (Input_D={input_d}) | {desc} | {pool_mode} | Weighted={is_weighted}")

        try:
            execute_backward_adagrad(
                T=2,
                D=input_d, # Passed as D/4, harness scales it back to Kernel_D
                B=16,
                log_E=4,
                L=4,
                D_gradcheck=1,
                weights_precision=weight_dtype,
                output_dtype=grad_dtype,
                
                # --- Critical Flags to engage the HIP Specialized Kernel ---
                row_wise=True,       # Must be True (triggers rowwise_adagrad)
                mixed=False,         # Must be False (no mixed_D)
                use_cache=False,     # Must be False (no UVM/LXU)
                weighted=is_weighted,# Testing both True and False
                
                # Defaults
                stochastic_rounding=False,
                mixed_B=False,
                cache_algorithm=CacheAlgorithm.LRU,
                pooling_mode=pool_mode,
                use_cpu=False,
            )
            print("  [PASS]")
        except Exception as e:
            print(f"  [FAIL] >>> {e}")
            failures.append({
                "Kernel_D": kernel_d,
                "Config": desc,
                "Pool": pool_mode,
                "Weighted": is_weighted,
                "Error": str(e)
            })

print("\n==================================================================")
if not failures:
    print("SUCCESS: All targeted configurations passed.")
else:
    print(f"FAILURE: {len(failures)} configurations failed.")
    for f in failures:
        print(f"  D={f['Kernel_D']}, W={f['Weighted']}, {f['Config']}, {f['Pool']}")
print("==================================================================")