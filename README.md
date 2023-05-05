# radix_sort_count

This extension exposes https://github.com/oneapi-src/oneDPL/blob/main/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_radix_sort.h#L470

## Building

Requires `dpctl` built in-place.

```bash
CC=icx CXX=icpx python setup.py develop -G Ninja -- -DDPCTL_MODULE_PATH=$(python -m dpctl --cmakedir)
```

## Running

```python
import radix
import dpctl.tensor as dpt

vals = dpt.arange(32, dtype="i8") # only int64 type is supported
counts = dpt.zeros(32, dtype="i8")

segments = 4
block_size = 2
radix_offset = 1

ht_e, e = radix.sort_count(vals, counts, segments, block_size, radix_offset)

ht_e.wait()

print(counts)
```