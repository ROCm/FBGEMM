# rm -fr _skbuild/
export FBGEMM_BUILD_TARGET=default
export BUILD_ROCM_VERSION=7.0
export MAX_JOBS='256'
gpu_arch="$(/opt/rocm/bin/rocminfo | grep -o -m 1 'gfx.*')"
export PYTORCH_ROCM_ARCH=$gpu_arch
python setup.py --build-target=default --build-variant rocm -DHIP_ROOT_DIR=/opt/rocm -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA" develop 2>&1 | tee build.log
beep
