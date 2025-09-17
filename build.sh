git submodule update
git submodule sync --recursive
git submodule update --init --recursive
 
 
cd ./fbgemm_gpu
# pip3 install -r requirements.txt
# pip3 install setuptools==75.1.0
 
export BUILD_ROCM_VERSION=7.0 && export MAX_JOBS='nproc' && gpu_arch="$(/opt/rocm/bin/rocminfo | grep -o -m 1 'gfx.*')" && export PYTORCH_ROCM_ARCH=$gpu_arch && python setup.py --build-variant rocm -DHIP_ROOT_DIR=/opt/rocm -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA" build develop 2>&1 | tee build.log
 
