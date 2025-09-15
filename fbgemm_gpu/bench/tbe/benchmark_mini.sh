export HIP_VISIBLE_DEVICES=7
export FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=1
# export ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/librocprof-trace-decoder.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/opt/rocm/lib:/usr/local/lib:"
export ROCM_PATH="/opt/rocm"

# just run
# python split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 10000,20000,20000,20000,10000,20000,20000,20000,20000,20000,20000,20000,10000,20000,10000 --bag-size-list 1,32,978,77,1,1,1,2,82,85,1042,89,1,983,1 --bag-size-sigma-list 0,6,195,15,0,0,0,0,16,16,208,17,0,196,0 --embedding-dim-list 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 --weights-precision fp16 --output-dtype fp16  --warmup-runs 10 --iters 10 --batch-size 8192 --flush-gpu-cache-size-mb 512 --alpha 1.15  --weighted

# rocprofv3 --kernel-trace --output-format csv -- python split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 10000,20000,20000,20000,10000,20000,20000,20000,20000,20000,20000,20000,10000,20000,10000 --bag-size-list 1,32,978,77,1,1,1,2,82,85,1042,89,1,983,1 --bag-size-sigma-list 0,6,195,15,0,0,0,0,16,16,208,17,0,196,0 --embedding-dim-list 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 --weights-precision fp16 --output-dtype fp16  --warmup-runs 10 --iters 10 --batch-size 8192 --flush-gpu-cache-size-mb 512 --alpha 1.15  --weighted

# rocprofv3 -i input.yaml -- python split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 10000,20000,20000,20000,10000,20000,20000,20000,20000,20000,20000,20000,10000,20000,10000 --bag-size-list 1,32,978,77,1,1,1,2,82,85,1042,89,1,983,1 --bag-size-sigma-list 0,6,195,15,0,0,0,0,16,16,208,17,0,196,0 --embedding-dim-list 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 --weights-precision fp16 --output-dtype fp16  --warmup-runs 10 --iters 10 --batch-size 65536 --flush-gpu-cache-size-mb 512 --alpha 1.15  --weighted

python split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 1000000,2000000,2000000,2000000,1000000,2000000,2000000,2000000,2000000,2000000,2000000,2000000,1000000,2000000,1000000 --bag-size-list 1,32,978,77,1,1,1,2,82,85,1042,89,1,983,1 --bag-size-sigma-list 0,6,195,15,0,0,0,0,16,16,208,17,0,196,0 --embedding-dim-list 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 --weights-precision fp16 --output-dtype fp16  --warmup-runs 10 --iters 40 --batch-size 65536 --flush-gpu-cache-size-mb 512 --alpha 1.15  --weighted

#python split_table_batched_embeddings_benchmark.py device-with-spec --num-embeddings-list 10000,20000,20000,20000,10000,20000,20000,20000,20000,20000,20000,20000,10000,20000,10000 --bag-size-list 1,32,978,77,1,1,1,2,82,85,1042,89,1,983,1 --bag-size-sigma-list 0,6,195,15,0,0,0,0,16,16,208,17,0,196,0 --embedding-dim-list 256,256,256,256,256,256,256,256,256,256,256,256,256,256,256 --weights-precision fp16 --output-dtype fp16  --warmup-runs 10 --iters 10 --batch-size 65536 --flush-gpu-cache-size-mb 512 --alpha 1.15  --weighted
