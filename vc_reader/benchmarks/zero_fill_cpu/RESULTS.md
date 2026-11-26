# Sparse uint8 sampling in VCReader

While rendering publicly available PHerc1447 segments, CPU profiling showed that many interior sample lanes fell back to the scalar eight-corner interpolation even after the reader had established that their entire chunk contained the declared fill. Villa already groups AVX2 lanes by chunk in pull request #2536. The change in grid.cpp completes a lane at that point only when all its corners are in one definitively absent chunk with positive zero fill. Pending reads, other fills, boundaries and unsupported layouts use the existing implementation.

## Observations on public scroll data

Unmodified base 659e2fadb836d44063ed831d85517154ec852f11. Both arms used identical geometry, input masks, native thread settings and serialization. Each local time is a median of three alternating processes. The CT and historical stacks were staged on the same ext4 disk. Each process started with an empty reader cache. Operating system page caches were not flushed. The local toolchain built Release with -O1. The companion comparison compiled only grid.cpp with -O3 for both arms. Neither comparison represents a complete upstream CI build.

| Complete PHerc1447 segment | Recovered historical columns in both arms | Local seconds, stock → modified (-O1) | Local seconds, stock → modified (grid -O3) | Peak RSS MiB, stock → modified (grid -O3) |
| --- | ---: | --- | --- | --- |
| 20250502183138 | 76,453 | 5.059 → 3.497 | 3.782 → 2.183 | 222.2 → 227.8 |
| 20250502184201 | 62,065 | 9.246 → 5.164 | 5.565 → 3.392 | 235.8 → 234.3 |

All 4,294,058 recovered uint8 depth values and the selection masks matched the stored exhaustive bicubic reference. A separate five-window regression retained 4,334 previously recovered columns including all 14 boundary counterexamples. A PHerc0139 w035 dense control produced identical float32 values, availability levels and uint8 output over 16,515,072 values. The latter covers one ROI with all 74 source chunks present.

The time saved depends on input availability. In a separate single cold public-origin pair per surface, times were 59.406 → 67.749 seconds and 65.670 → 61.978 seconds. With source caches populated, the corresponding pairs were 4.634 → 4.148 and 6.391 → 3.790 seconds. Transfers were identical in both arms. These observations do not establish an end-to-end network speedup. There was no HTTP change.

Eight official VCReader test modules exercised the modified binary. Pytest reported 94 primary passes, 18 skips and 1,039 passing subtests. The public reproducer uses a small real-scroll crop and controlled edge cases. Its timings have a different scope from the whole-surface results above.

AI tools assisted implementation and local validation. Independent full Release and cross-host timing remain open.

[Upstream SIMD implementation](https://github.com/ScrollPrize/villa/pull/2536) · [Scroll dataset and license information](https://scrollprize.org/data_datasets)
