# Ninety-six round trips per query: how a vector index spent nine tenths of its time waiting

Our reference index answers a nearest-neighbour query over 26.7 million vectors in 1.8 milliseconds. Three weeks ago the same index, on the same machine, took 21. Nothing about the algorithm changed, no data structure was replaced, and recall is identical to the fourth decimal. What changed is that the code now tells the kernel, once, which pages it is about to read.

This is the story of that correction, and of the four days of work that preceded it and produced nothing.

## The shape of the problem

The index serves queries in two stages. A greedy walk over a proximity graph uses one-bit quantised codes, held entirely in RAM, to select a few hundred candidates. A rerank stage then computes exact distances against the real vectors, which live in a 27 GB half-precision arena mapped into the address space. The arena is far larger than what the page cache can hold, so most of those exact distances require the kernel to fetch a page from the SSD.

We knew the rerank stage dominated the profile. We assumed the cost was arithmetic, because that is what a profile of a compute-bound program looks like, and we spent the better part of a week making that arithmetic faster. We fused the half-precision decode into the distance loop, which removed an intermediate buffer and a full pass over memory. We replaced a 128 KiB lookup table, rebuilt on every query, with a bit-plane estimator that reads no table at all. In isolation, both worked: 2.9x and 1.4x on microbenchmarks, with recall unchanged.

On the real index, measured before and after with identical queries, the difference ranged from minus 1.7 to plus 0.5 percent and changed sign between runs. We had made two kernels substantially faster and the product not at all.

## The number that was not in the profile

The explanation was sitting in the profile's header, above the part we had been reading:

    Duration: 9.92s, Total samples = 1.64s (16.53%)

The sampling profiler only observes a process while it holds the CPU. Ours held it for one sixth of the elapsed time. The remaining five sixths were spent blocked, invisible to the tool we were using to decide what to optimise. The two kernels we had rewritten accounted for roughly fourteen percent of that one sixth — about two percent of the real cost, which is precisely the resolution at which our end-to-end measurement returned noise.

Three lines reading the process counters settled what the profiler could not show. Per query, the search read 12.3 megabytes from the disk and took 96 major page faults, to return ten neighbours. At roughly two hundred microseconds each, those faults accounted for nineteen of the twenty-one milliseconds.

## Ninety-six faults that did not need to be sequential

The rerank loop receives its candidate list complete. Every identifier is known before the first byte is read. Yet the loop walked that list one entry at a time, and each entry that was not resident stopped the thread until the kernel had fetched a single page from an NVMe device perfectly capable of servicing dozens of requests at once.

Nothing in the problem required that. The serialisation was an artefact of statement order — the natural way to write a loop, and the wrong way to read from a device with a deep queue.

The fix announces the entire batch before the loop begins. The candidate ranges are collected and passed to the kernel as advice, through `process_madvise` in a single system call where the kernel supports it and individual `madvise` calls where it does not. The loop that follows is unchanged; by the time it dereferences a page, the page is usually already there.

Two distinct things improve at once, which took a second measurement to separate. The reads are issued concurrently rather than one after another, so the device's queue is finally used. And they are bounded to the ranges actually requested: without the hint, each fault triggers the kernel's readahead heuristic, which optimistically pulls up to 128 KiB of surrounding data for the single kilobyte the program wanted. On a sequential scan that heuristic is free performance. On a scattered rerank it is a factor of thirty in wasted bandwidth.

## What it bought

| Measurement, 26.7M vectors, 512 dimensions | Before | After |
|---|---:|---:|
| Median latency | 21.28 ms | 1.79 ms |
| 99th percentile | 28.84 ms | 2.53 ms |
| Major page faults, per run | 28 827 | 0 |
| Bytes read from disk, 200 queries | 2 311 MB | 107 MB |
| Throughput, eight concurrent queries | 261 q/s | 1 940 q/s |
| Recall@10 against exact search | 0.9733 | 0.9733 |

The hardware is a consumer i9-14900K with 62 GB of RAM and a mid-range NVMe drive; the working set is 34.3 GB resident, so nothing here depends on the data fitting in memory. Recall is measured against exhaustive brute force over the full arena rather than against another approximate configuration, which took about three minutes per run to compute and is the only reason we can state that the optimisation is free rather than merely cheap.

An access hint cannot alter what a program computes, so the identical recall is expected rather than impressive. We verified it regardless: two hundred queries return byte-identical top-ten lists with and without the change.

## Three more things that did not work

Before arriving at the fix we were offered, and briefly believed, an architectural proposal built on the premise that CPU prefetch instructions could hide the SSD latency behind vector arithmetic. They cannot. On a cold 128 MB mapping, two thousand `__builtin_prefetch` instructions execute in ten microseconds and leave exactly zero pages resident; two thousand real accesses to the same addresses take 112.9 milliseconds and fault everything in. A prefetch instruction is a hint to the cache hierarchy and is architecturally incapable of raising a page fault, which is the entire mechanism by which a mapped file is read.

We also spent a day on memory layout. The walk reads a code and two norms for every visited node, and these lived in three separate arrays, giving three scattered accesses per node and thirty-one percent of the CPU budget. Interleaving them into a single record returned between one and a half and three percent. We then benchmarked four layouts at full scale — three arrays, packed at eighty bytes, aligned to 128, and split with the norms densely packed — and they landed within noise of one another at 62 to 68 nanoseconds per access. The version that aligns each record to a cache line, which ought to win if cache lines were the constraint, does not win and costs sixty percent more memory. Sixty nanoseconds is what a random access to main memory costs on this machine. Grouping accesses that the prefetcher cannot predict does not make them predictable.

Finally, we tried to vectorise the estimator using Go's experimental SIMD package, and could not. A hand-written popcount kernel ran at 400 nanoseconds against 19 for the scalar version. Isolating the primitives showed why: a kernel restricted to `And`, `Add` and `SumOf8AbsDiff` costs 5.3 nanoseconds, which is what AVX2 should deliver, while adding a single `PermuteOrZeroGrouped` takes it to 380. Shifts and permutes appear to cost around ninety cycles where the hardware spends one, under both Go 1.26.5 and 1.27rc1. We reported the reduction upstream and moved on.

## A number we got wrong

While measuring recall we reported 0.470 for this index and concluded that the graph was poorly built. That figure was real, but it came from a different corpus: a test set of uniformly random vectors, which is the pathological case for approximate search because in high dimension every point sits at roughly the same distance from every other. We carried it across because both files lived under the same project directory. The reference index, measured properly, returns 0.9733. The mean absolute component value of the two datasets, 0.034 against 0.500, had been telling us they were unrelated the whole time.

## What we would tell ourselves in advance

The generalisable point is not about memory mapping, and it is not that we should have suspected I/O sooner. It is that a benchmark reporting only elapsed time cannot distinguish a slow computation from a blocked one, and will therefore conceal the most expensive category of defect for as long as it is the only instrument in use. This particular defect survived several rounds of optimisation and a measurement campaign that concluded the 26.7 million scale was proven.

Bytes actually read, major faults, resident set size. Three counters, available from `/proc` and `getrusage`, would have pointed at the answer on the first run. They are now part of the library's measurement surface, and they carry a test that verifies they report non-zero on a known workload — because an instrument that always returns zero is indistinguishable from a clean result.

The code and the full write-up, including the protocol and every path we abandoned:

https://github.com/hazyhaar/horosvec

https://github.com/hazyhaar/horosvec/blob/main/docs/MESURES-2026-08-prefetch.md
