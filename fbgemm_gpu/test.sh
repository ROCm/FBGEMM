cd test
export FBGEMM_TBE_ROCM_HIP_BACKWARD_KERNEL=1
python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning tbe/training/backward_adagrad_global_weight_decay_test.py
cd -
