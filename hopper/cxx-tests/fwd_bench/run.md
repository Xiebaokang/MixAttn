```bash
# compile src code
cd flashattention-t-artifact/1-figure8-main-results/flashattention-t/hopper/cxx-tests/fwd_bench
mkdir -p build
cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CUDA_ARCHITECTURES=90a
ninja fp16-fwd-bench fp16-fwd-bench-orig fp8-fwd-bench fp8-fwd-bench-orig attn_test_orig attn_test

./fp16-fwd-bench ./data_ours_h100_fp16.csv
./fp16-fwd-bench-orig ./data_fa3_h100_fp16.csv

./fp8-fwd-bench ./data_ours_h100_fp8.csv
./fp8-fwd-bench-orig ./data_fa3_h100_fp8.csv

./attn_test_orig ./attn_test_orig.csv
./attn_test ./attn_test.csv
```
