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
