# Three days of horosvec — from internal library to a 26.7M-vector engine, a retrospective

**July 7-9, 2026. Machine: RTX 5090 (32 GB), 32 cores, 62 GB RAM.**

This document is the unabridged retrospective of a three-day campaign: an
internal pure-Go vector index taken through publication, an optimization
sweep, an adversarially-reviewed scale-up plan, the indexing of the full
28.7-million-item Hacker News corpus, and a three-engine benchmark whose
errors taught more than its results. It is written as a retrospective: what
happened, what failed, what caught the failures, and what deserves to be
carried forward.

Companion documents: [benchmark write-up](./BENCHMARK-2026-07.md) ·
[architecture reference](./ARCHITECTURE.md) ·
[raw benchmark results](https://github.com/hazyhaar/horosvec-bench).

---

## 1. Where the day started

`horosvec` is a pure-Go embedded ANN vector index: a DiskANN-style Vamana graph
with RaBitQ binary quantization, two-stage search (cheap approximate preselection,
exact L2 rerank), persisted in a single SQLite file with zero CGO. It had been in
production for months inside the horos55 ecosystem, searching RAG shards and code
maps.

The single most important fact of the day was established early and colors
everything that follows: **production had never exercised the ANN path.** The
largest production shard held 5,632 vectors; the brute-force threshold is 50,000.
Every production query had been running the exact path, which has perfect recall
by construction. Every defect described below had been living, dormant, in a code
path that months of production had never touched — and a February prototype's
internal recall benchmark had itself been running under the brute-force threshold,
measuring the exact path while believing it measured the graph.

**Lesson 1 — what production does not exercise stays broken.** A single afternoon
of comparative benchmarking did what months of production could not.

## 2. The morning: publication, then the benchmark as an instrument of discovery

The morning campaign took the library from "internal component" to "published
module with honest numbers", in three movements.

### 2.1 Pre-publication audit and hardening

An independent audit (quality + security lenses) found two blockers and four
major defects before the code went public. The blockers were both in the binary
deserializer — allocations sized from untrusted header fields, so a 40-byte blob
could trigger a panic or a multi-gigabyte allocation. The audit pattern worth
keeping: **on any deserializer, audit every `make(x, sizeReadFromInput)` that
happens before the corresponding `ReadFull` — that is where the blockers live.**

The majors included a transactional integrity bug (in-memory state mutated before
the SQLite commit, so a failed commit left phantom ids visible to Search) and a
missing feature from the RaBitQ paper (the random rotation) that was initially
"documented as a limitation" — a decision that would be reversed by evidence
later in the day (see 2.3).

One adversarial review during this phase earned a NO-GO that mattered: after
`context.Context` support was added, the reviewer demonstrated that a
cancellation arriving *during* beam search was silently swallowed into an empty
success — the inner function returned nil without error, downstream loops ran
zero iterations past their own guards, and the caller received `(0 results,
err=nil)`, indistinguishable from "no neighbors found".
**Lesson 2 — a ctx guard placed inside a loop protects nothing if the loop can be
empty.** Error propagation, not loop guards, is the fix.

### 2.2 Publication with history preserved

The module was extracted to `github.com/hazyhaar/horosvec` (MIT, one dependency).
Publication surfaced three surprises, all handled without destroying anything: a
private archived repository with the same name existed (a February prototype with
working CI), so its history was preserved under a `-s ours` merge and its CI
recovered; the `v0.1.0` tag belonged to the prototype and is immutable in the Go
module proxy, so the release went out as v0.2.0; and horos55 itself was rewired
onto the public module, deleting the 5,544-line internal copy.

### 2.3 The benchmark ladder: every scale reveals a defect invisible at the previous one

A comparative CGO benchmark harness was built (one binary per engine to isolate
runtimes; recall-vs-QPS Pareto curves, never single points; queries never drawn
from the base; ground truth recomputed when the base is truncated). The
competitors: sqlite-vec (exact, recall 1.0 by construction — a harness control)
and hnswlib (the algorithmic reference).

- **At 14k real vectors (dim 1024, real session texts embedded with bge-m3):**
  horosvec had perfect recall but was *slower than exact search* (~35 QPS, 28 ms
  p50) and insensitive to its beam parameter. A pprof run put 84.6% of the time
  in the quantized distance function: a 1,024-iteration scalar loop with a bit
  test and a branch per dimension — *more expensive than the exact float L2 it
  was supposed to shortcut*. The fix was a per-query lookup table of partial sums
  per byte (fastscan): p50 went from 28 ms to 1.7 ms, throughput from 35 to ~570
  QPS, recall unchanged. The gain was ×16 in the real harness where a
  microbenchmark had promised only ×2.7 — on fixed test patterns the branch
  predictor had been flattering the old loop.
  **Lesson 3 — a microbenchmark with a fixed input pattern lies about branchy
  code.** Measure on realistic bit distributions.

- **At 100k SIFT vectors:** recall plateaued at 0.955 and the beam knob still
  did nothing. The user made the pivotal call here: *"wouldn't we rather fix
  the code than run a 1M bench on broken code?"* Investigation found that
  `RerankTopN` (default 500) was silently flooring the effective beam:
  `efSearch = max(EfSearch, rerankN)`, so every setting from 64 to 512 ran the
  same ~500-wide search. The knob was decorative. A three-line inversion of the
  coupling (the beam is the user's knob; rerank adapts to the beam) exposed the
  real Pareto curve — and the real curve exposed the *next* defect: recall now
  saturated at 0.955 regardless of beam width, which pointed at the estimator,
  not the traversal. SIFT is notoriously anisotropic — exactly the regime where
  the RaBitQ paper's random rotation matters. The morning's "document it"
  decision on the rotation had been correct *on the data it was made with*
  (synthetic Gaussians and real 1024-dim text embeddings) and wrong on SIFT.
  **Lesson 4 — a data-driven decision carries the signature of its dataset;
  change the distribution and you must re-decide.**
  **Lesson 5 — perfect recall masks traversal defects.** The decorative knob
  already existed at 14k, invisible because recall was 1.0 everywhere.

- **The rotation itself** (randomized Hadamard, v0.4.0) was preceded by an
  adversarial design review that proved the gain numerically *before any Go was
  written* (a Python prototype on 10k real SIFT vectors: estimator quality proxy
  0.780 → 0.869; the cheap alternative — variance scaling — measured *worse*
  than nothing at 0.745 and was killed by evidence). The review also caught
  three design flaws pre-implementation, including a silent-downgrade version
  hole: an old binary reading a rotated database would have degraded recall
  invisibly, because the exact rerank stage masks estimator garbage.
  **Lesson 6 — in an approximate+rerank pipeline, the false-green lives at the
  seam where old code reads a new world without erroring.** After
  implementation, the SIFT ceiling yielded: 0.9553 → 0.9874 at ef=512, same
  throughput.

- **The hot plane** (v0.5.0) transposed the main lesson from reverse-engineering
  hnswlib (1,815 lines of C++ read fiche by fiche): the winner co-locates
  links+vector+label per node in flat immutable arenas and pays no lock and no
  map lookup in the hot loop, while horosvec paid an RLock plus a map lookup
  *per neighbor*. A pointer-free arena (codes, norms, neighbor offsets, ids —
  no internal pointers, so Go's GC skips it in O(1)) doubled QPS at every
  operating point and, more telling, collapsed p99 under one millisecond
  everywhere — the latency tail *was* the per-neighbor locking plus GC mark
  pressure. The GC question the user had raised ("does the Go GC fall on us at
  10M vectors?") was answered structurally rather than by tuning: the mark cost
  scales with live *pointers*, so the fix is to not have millions of them; plus
  explicit `runtime.GC()` at the two cold deterministic windows (end of build,
  end of rebuild swap).

- **Parallel build** (v0.6.0) came from a user observation as blunt as the watts
  one later: a 282-second build at 100k on a 32-core machine showing load 2.6.
  The parallelization (lock-free reads on the current graph via atomic pointer
  swaps of neighborhoods, sharded 256-way mutex for edge insertion, per-worker
  cloned Rotators) yielded ~9×. The mandatory `-race` gate caught three genuine
  data races during development, including one that silently destroyed recall
  (parallel erasure of back-edges: 0.06 vs 0.63). The politeness default also
  matters: `BuildWorkers=0` means ~40% of the process's CPU capacity, not all
  cores — an embedded library must not starve its host application.

By end of morning: recall 0.987 at SIFT-100k, sub-millisecond p99, build ~9×,
the gap to hnswlib narrowed from 40–200× to ~3–10× at comparable recall. The
morning closed by drafting a plan for the next target — indexing the full
HackerNews corpus (the 28M-row public Parquet from ClickHouse was downloaded and
verified) — and writing that plan down as a point d'étape.

## 3. The evening: adversarial review of the morning's plan

The evening session began by doing to the plan what the morning had done to the
code. On the user's instruction ("adversarialize this plan — it was written by
an Opus with 800k of context"), the plan was submitted to a three-model
refutation bench (grok-build, grok-composer, Sonnet — same prompt, six attack
axes) plus an independent opinion from the session architect, everything
adjudicated on the ground. The composition of the panel proved its worth again:
the single most structural finding came from one model only, and none of the
four reviewers (three external plus the architect) found all of the flaws.

What fell, with its oracle:

1. **"Matryoshka 512 on bge-m3" — refuted.** The plan's entire memory sizing
   rested on truncating bge-m3 embeddings to 512 dimensions. The model's
   official card says 1024 dimensions and never mentions Matryoshka training;
   truncating a non-MRL model degrades recall unboundedly. This was the
   load-bearing premise of the "everything fits in RAM" simplification.
2. **The 27 GB RAM margin — refuted.** The plan divided by total RAM (62.6 GB);
   the machine actually offers ~46 GB with the resident services, and the
   *build* peak of the then-current code was ~110 GB (double fp32
   materialization), which the plan had not modeled at all. It had reasoned
   about the steady state and ignored the construction window.
3. **The eligibility filter "28M → 20–24M" — refuted by measurement.** DuckDB
   on the real Parquet: 28,737,557 rows, only 7.1% dead/deleted/empty →
   26.7M eligible. The promised "15–25% GPU savings" was really 7%. Worse, the
   filter rule was ambiguous: requiring *both* title and text non-empty leaves
   only 217,558 rows — a trap that had to be resolved explicitly before any
   embedding run.
4. **fp16 "as a prerequisite" — refuted as stated.** Zero occurrences of fp16
   anywhere in the codebase. It was an entire work package, not a configuration.
5. **The rerank does not read the RAM mirror — the structural finding.**
   `flatVecs` (the in-RAM vector mirror) is consumed *only* by the brute-force
   path; the Vamana rerank loads every candidate through SQL, polluting the LRU
   cache. So "allocate a RAM tier and the three audit problems disappear" was
   false: the fix is a *rewiring of the hot path*, not an allocation. This came
   from the composer, was confirmed on the ground, and reshaped the foundation
   work.
6. **The flat file could not be an extension of the existing format.** The
   existing HVEC export format is variable-length, big-endian, stride-less —
   unusable for mmap or direct offset access. The arena had to be a new format.

What *held* also matters: every code anchor the plan cited (the 127 ms rerank
diagnosis, the WAL-per-pool-connection pragma bug, the `allVecs` build
materialization) checked out exactly; the ~9 GB estimate for graph+codes+norms
was recomputed independently by two models to ~9.5 GB (a third reviewer's
16–17 GB was itself adjudicated wrong — it had counted the SQLite int64
representation instead of the serving layout); and one reviewer's "7.1 GB is
really 6.64" refutation was rejected as a GB/GiB unit quarrel. Adjudication cuts
both ways.

**Lesson 7 — an intelligent plan fails at its premises, not its arithmetic.**
Every multiplication in the plan was correct; the two numbers being multiplied
(512 dimensions, 62.6 GB) were the fiction. And the way out was found in an
hour of web research: the Qwen3-Embedding family (June 2025) is natively
Matryoshka-trained, so the repair was to change models, not to abandon the
512-dimension target.

## 4. The evening: the foundation méta-goal (V0 → V3)

The foundation work was cast as a single autonomous sequenced goal — golden
baseline, fp16 arena + rerank rewiring, streaming build, non-regression — with
mechanical gates between waves and the live embedding work explicitly fenced
out (a goal whose criteria depend on GPU services deadlocks its own stop hook;
that discipline exists as a forge-time checklist and it was applied).

- **V0 (golden).** Build and tests green; the benchmark binary recompiled from
  scratch into a single state (the previous day's 1M bench had mixed two binary
  states — the "contamination" was confirmed spectacularly: the clean parallel
  build ran in 444 s where the contaminated run had taken 4,235 s single-core).
  Reference numbers frozen: recall@10 0.9362 at ef=512, p50 4.85 ms, build peak
  RSS ~4.97 GB. The SIFT dataset was rescued from a volatile session scratchpad
  to durable storage — a small act with the highest regret-avoidance of the day.
- **V1 (fp16 arena + rerank rewiring, commit d0da542).** A new flat arena format
  (fixed stride, little-endian, versioned header, pure-stdlib IEEE 754
  half-precision conversion), an opt-in `Config.ArenaPath`, and the rerank loop
  rewired to read the arena. Gate: a test counting SQL loads on the rerank path
  with the arena active — zero. Recall held (0.9360 vs 0.9362 golden), and the
  rewiring alone bought p50 4.85 → 1.29 ms and 208 → 770 QPS. The adversarial
  audit of this wave caught a **pre-existing correctness bug**: the rotation's
  FHT scratch buffer was shared across concurrent Searches, corrupting rotated
  queries under load — dormant because no concurrent-search test had ever run
  under `-race`. A feature's test suite exposing a fault *upstream* of the
  feature is exactly what independent review is for.
- **V2 (streaming build + vector-less SQLite, commit 1442b50).** Build became
  two-pass (stream the iterator into the arena, then build the graph reading
  from it); SQLite stopped storing vector blobs for large indexes;
  crash-consistency via manifest-after-fsync with a truncation test. Build peak
  RSS: 3.34 GB, −33% vs golden. Two instructive failures inside this wave: the
  coder's *first* implementation measured 6.19 GB — *worse* than the golden —
  and was caught because the wave's oracle was the measured RSS peak, not the
  presence of a "streaming" symbol; and the coder's six new tests all forced
  `BruteForceThreshold=0`, masking a panic on the *default* brute-force path
  that the independent auditor found by reading that setting as an avoidance
  signal. **Lesson 8 — when a deliverable adds a mode, test the mode × read-path
  matrix under the default configuration, never only under the tests'
  configuration.**
- **V3 (non-regression, run by the architect).** Full re-run on the final HEAD:
  recall 0.9356, p50 1.248 ms, race suite green (205 s). One measurement was
  discarded with cause: the second bench run had been launched concurrently
  with the race suite and its build throughput (−4%) reflected CPU contention,
  not the code. Solo runs bracketed the golden within ±0.6% — adjudicated
  indistinguishable. **Lesson 9 — a benchmark sharing the machine with another
  load is not a measurement; discard it with a stated reason rather than
  arguing about its number.**

Across the three coding waves, the independent adversarial audits caught **seven
hard findings** (shared rotation scratch, brute-force panic, silent Insert
degradation, negative-topK panic, unrepairable finalize window, a data race on
an error flag invisible to `-race` for lack of an error-path test, a header
overflow guard) — every one of them invisible to green build+test gates.

## 5. The evening: calibration, and the watts lesson

The embedding side was run as a live operations chantier, not under a goal hook.
A vLLM slot for Qwen3-Embedding-0.6B was provisioned (versioned llama-swap
config, hot reload). First trap: vLLM *refused* `dimensions=512` because the
model's HuggingFace config does not carry the Matryoshka flag the server checks
— fixed with an `--hf-overrides '{"is_matryoshka": true}'` on the slot, verified
by probe.

Then came the day's central metrology correction, and it came from the user.
The calibration bench was running its throughput phase when the user observed
the GPU was doing "micro-jumps" and said: **watch the power draw, that's the
tangible signal — and you should have batched.** Ground check: 48 W out of 575,
2–3% utilization. The bench protocol (batches of 32, concurrency 4) — written by
the session architect — was measuring an HTTP client, not a GPU. An
extrapolation from that regime would have been wrong by a large factor and
would have looked perfectly credible. The protocol was corrected mid-run: sample
`power.draw` throughout, escalate batch size and concurrency in steps until the
watts plateau, extrapolate only from the saturated tier. The corrected bench
reached 100% utilization at 455–465 W (and, on the larger model, the full 575 W
power cap).

**Lesson 10 — the saturation oracle for GPU throughput is power draw.** A bench
whose watts stay at idle is measuring the client. This was engraved as a
permanent behavioral memory the same evening, alongside its sibling from V0
(sample process RSS externally; never trust a tool's self-reported memory
column): *always measure the independent physical oracle, never the
instrument's self-declaration.*

The calibration results, on 2,000 real documents and 200 real queries
(reproducible seed):

| Configuration | overlap@10 vs own full dim | Throughput | 26.7M run |
|---|---|---|---|
| Qwen3-0.6B @ 512 (MRL) | 0.7665 | 152k tok/s | ~3.6–3.9 h |
| bge-m3 @ 512 (naive truncation) | 0.6955 | 180k tok/s | (rejected) |
| Qwen3-4B @ 512 (MRL) | 0.6410 | 26k tok/s | ~21.4 h |
| Qwen3-4B @ 1024 (MRL) | 0.7705 | 26k tok/s | ~21.4 h |

Three things fell out of this table. First, the original plan's premise was
empirically buried: naive truncation of bge-m3 loses ~30% of the top-10.
Second, the plan's "~23 hours of embedding" wall was overestimated by a factor
of six — at the saturated tier, the full corpus costs under four hours. Third,
and counter-intuitively: **even native Matryoshka at 512 reshuffles ~23% of the
top-10** relative to the same model's full space. Halving the index dimension
is a fidelity/RAM trade, not a free lunch — the plan had treated it as an
acquired simplification.

The user chose to probe the 4B model before committing (a 20-minute probe
before a 4-vs-21-hour decision — the right instinct). The probe disqualified
it: the 4B compresses 2560→512 (a 5× ratio, versus 2× for the 0.6B), so its
internal overlap at 512 is *lower*, and its throughput is 6× worse. The probe
agent's caveat deserves preserving verbatim: **a self-consistency overlap
measures fidelity to the model's own full space, never absolute relevance
between models — the compression ratio conditions the number.** Cross-model
relevance would require a labeled benchmark, which was correctly declared out
of scope rather than improvised.

Decision: **Qwen3-Embedding-0.6B at 512 MRL dimensions, fp16** — a 27 GB
all-in-RAM index served directly by the arena built in V1/V2.

## 6. The evening: the production pipeline, proven by kill

The last work package built `hnbook-embed`, the production binary: DuckDB CLI
streams eligible rows as NDJSON (the Go side never links a Parquet library —
one stage, one simple artifact), the binary batches to the embedding endpoint
(batch 128, concurrency 8, the calibration bench's 429 back-off constants), and
a single ordered writer streams fp16 vectors into the arena with a durability
chain of fsync-arena → fsync-ids → atomic manifest.

The wave's defining proof: **two real `kill -9` during a live run, followed by
resume.** The ids file came back byte-identical with zero holes. The vectors
did not — 11 out of 10,000 differed at the last fp16 bit — and the coder's
first reaction ("checkpoint bug") was wrong in an instructive way: GPU
inference is non-deterministic at the ulp level, so **the idempotence oracle
for a resumable embedding pipeline is the ids file, not the embedding bytes**
(cosine ≥ 0.99986 on the divergent few). The end-to-end smoke on 10,000 real
items closed the loop: arena byte-size exact to the format formula, Build
streaming from it, Search answering 20/20 real queries with zero SQL on the
rerank path, GPU-bound at 497 W.

The independent audit caught two hard findings even here — an unrepairable
finalize window after a kill at exactly the wrong instant, and a data race on
the error flag that `-race` could not see because no test exercised the error
path in flight. **Lesson 11 — for any crash-consistency guarantee, enumerate
every kill window between the syncs, renames and manifest writes, and replay
the resume by hand for each.** A green test proves only what it covers.

Two operational footnotes: the repository question (the bench module had no git
at all — the pipeline was un-versioned until the user ordered `git init`; a
multi-session guard correctly refused a blanket `git add -A` and the commit
went in by exact paths), and the session's permission incident (five refusals
of `GOWORK=off go build` before a subagent's retrospective revealed the
harness accepts the `env GOWORK=off …` form — a trap now documented in every
brief).

## 7. The human in the loop — a specific accounting

This day is a case study in what the human contributed that the models did not,
and it is worth being specific rather than polite:

1. **"Fix the code before benching 1M"** (morning) — stopped a benchmark run
   on defective code, which would have produced a plausible, wrong, archived
   result.
2. **The 32-cores-at-load-2.6 observation** (morning) — launched the parallel
   build.
3. **"Adversarialize this plan"** (evening) — the instruction that took down
   two load-bearing premises before a line of the foundation was written.
4. **The watts correction** (evening) — the single intervention with the
   largest error-mass avoided; every downstream decision (model choice, 4B
   disqualification, run-time projections) rests on saturated-tier numbers.
5. **"Probe the 4B first"** — a 20-minute probe before a 6× commitment.
6. **"Git the bench"** — closed the one gap where delivered production code
   was sitting outside version control.

The symmetric accounting also holds: the machinery caught things no single
human review would plausibly have found (the shared FHT scratch under
concurrency, the byte-level idempotence analysis, the twelve-point plan
refutation with ground oracles), and the *combination* — independent skeptical
review at every altitude (plan, design, code, measurement), always adjudicated
against a decidable oracle — is the actual method. Its cost is real (three
external model runs, seven subagent waves, double-digit audits); its yield on
this day was seven hard code defects, two false premises, one 6× measurement
error, and zero of them reaching production.

## 8. State at close, and what remains

**Shipped and verified** (all commits local, unpushed, per standing instruction):
- horosvec v0.2.0 → v0.6.0 public trajectory (fastscan ×16, honest EfSearch,
  Hadamard rotation 0.955→0.987, pointer-free hot plane ×2 QPS / sub-ms p99,
  parallel build ~9×), plus the evening's arena/streaming stack (d0da542,
  1442b50, 8f49661): fp16 arena, zero-SQL rerank (p50 1.25 ms, 800 QPS at 1M),
  streaming build (−33% peak RSS), vector-less SQLite.
- `hnbook-embed` production pipeline (repo initialized, 3af1827),
  kill-proven checkpoint resume, 10k real-data smoke green.
- Three embedding slots served and versioned (bge-m3, Qwen3-0.6B, Qwen3-4B,
  the latter two with the Matryoshka override).
- The model decision: Qwen3-Embedding-0.6B @ 512 MRL fp16.

**Next actions, in order:**
1. The 26.7M embedding run (~4 h saturated GPU, checkpointed, resumable) —
   ready to launch on signal.
2. Build the full HNbook index from the arena; observe the real build peak at
   26.7M and decide RAM vs mmap on numbers.
3. The deferred performance backlog: coarse-to-fine build, exact-distance walk
   at small dimensions, multi-pivot entry points, the 2 GB build-peak target.
4. The open product question the whole day deliberately did not answer, and
   which is now the only thing standing between an index and a purpose:
   **who is this for.** The plan-refutation flagged it, the neuroclean guard
   flagged it, and it remains the first question of the next session.

## 8bis. The night run: four deaths, zero lost documents (July 8, 00:16 → 07:44)

The 26.7M embedding run was launched at 00:16 and finished at 07:44 with a perfect
ledger — 26,691,317 documents embedded, exactly the eligible count, arena and ids files
byte-exact to their format formulas. Between those two timestamps it died four times,
and the resulting debugging cascade is worth its own retrospective, because every death
was a *different layer* wearing the same symptom, and because the checkpoint design
turned what could have been a lost night into a sequence of free retries.

**Death 1 (465k documents, HTTP 400).** The anti-poison cap added the previous evening
was denominated in *bytes* (24,000) while the server limit is denominated in *tokens*
(8,192). An HN text saturated with HTML entities tokenizes far worse than the assumed
ratio, and one slipped under the byte cap while exceeding the token window. The fix was
to stop guessing ratios entirely: server-side `truncate_prompt_tokens`, verified by
probe before relaunch. The honest note: the original cap's margin had been asserted
("well below the window even at an unfavorable ratio") without a probe — the same
self-posed-hypothesis drift the evening's neuroclean had just written up.

**Death 2 (569k documents, 300 s client timeout, GPU at 100%).** The requests hung for
five minutes *upstream* of an idle engine. The trail led through the proxy's journal
(silent for the whole window) to a topology discovery: adding the Qwen slots to the
llama-swap `embedding` group had armed a trap — group members swap each other *by
default* (`exclusive: false` only prevents cross-group eviction), so the first bge-m3
request from any other consumer evicted the embedding slot mid-flight. The "GPU at
100%" during the stall was the other model loading. Fix: `swap: false` on the group,
cohabitation proven by alternating probes — three embedding containers can now coexist.

**Death 3 (same point, zero documents written).** The swap fix held — both containers
up — and the run died anyway, first requests hanging 300 s. Bisection begins: the exact
failing batch extracted from the corpus by rank; through the proxy it hangs
*intermittently* (fail/ok/fail on an identical payload), direct to the container it
completes in 0.2 s. Verdict: bypass the proxy for the mass pipeline, file the proxy bug
with a replayable repro kit. (This verdict was recorded — and was about to be proven
incomplete.)

**Death 4 (same point, direct endpoint).** The same 300 s timeout with no proxy in the
path exonerated llama-swap and left exactly one variable: concurrency. Reproduction at
the API level — eight concurrent identical real batches — showed vLLM itself losing one
request out of eight: seven answered in under a second, one sat in the Waiting queue
forever, engine empty. The final hypothesis was the boundary: the monster text was now
truncated to *exactly* 8,192 tokens = `max_model_len`, a full-window sequence that a
pooling model (no chunked prefill) must schedule in one piece — and under concurrency,
a scheduler race orphans it. The experiment was decisive and cheap: 24 concurrent
requests at truncate=8192 → 3 hangs; 24 at truncate=8000 → zero. One constant changed
(8,000 tokens, a 192-token margin below the window), committed with the measurement in
the comment, and the fifth launch crossed the fatal zone by 1.4M documents within the
first half hour and never looked back.

What this cascade teaches, beyond the specific bugs:

- **Four layers, one symptom.** A byte/token unit error, a process-manager eviction
  policy, a suspected proxy, and a scheduler race in the inference engine all presented
  as "the run stopped around 569k". Nothing but single-variable bisection — proxy vs
  direct, size vs content, count vs tokens, one boundary value vs another — could have
  separated them; each probe was under a minute, and each eliminated exactly one
  hypothesis. The intermediate verdict recorded after death 3 ("the proxy is the
  culprit") was wrong in attribution and was corrected in place the moment death 4
  disproved it — recording a wrong-but-dated diagnosis and then correcting it beats
  never committing to one.
- **The checkpoint design paid for itself four times in production conditions.** Every
  death resumed from the manifest with zero loss and zero duplicate — the property
  proven by `kill -9` in testing was exercised for real, repeatedly, under four
  different failure modes. The deterministic input stream (ORDER BY id) is what made
  the resume trivially correct.
- **Never truncate to a model's exact context window.** The general rule extracted from
  the vLLM race, now engraved for every embedding consumer in the ecosystem: keep a
  margin (~200 tokens) below `max_model_len`. A full-window sequence is a boundary
  condition in someone else's scheduler.
- **Fail-loud plus checkpoint beats retry-and-mask.** The pipeline's refusal to swallow
  timeouts is what surfaced four real bugs in one night — a retry loop would have
  masked the vLLM race as occasional slowness, shipped a working arena, and left a
  landmine for every future consumer of the serving stack.

Final tally of the run: 6 h 40 wall clock for the successful pass (~1,090 docs/s
sustained), 15 texts capped at 24k bytes across the whole corpus, a 25.5 GB fp16 arena
plus a 204 MB ids file, both byte-exact. The index build at 26.7M scale is the next
step, and the first true test of the streaming-build foundation beyond the 1M bench.

## 8ter. Postscript, the morning after (July 8, 08:56)

One screenshot closes the story better than a paragraph: the user's system monitor
during the V1 gate bench of the index-build méta-goal — all 32 cores lit up in
spaghetti (`2026-07-08_hnbook_v1_bench_cpu_32cores.png`, archived alongside this
document). It earned its place in the retrospective for two reasons. First, as a
bookend: twenty-four hours earlier the same machine was running a 282-second
single-core build at load 2.6, and a GPU "throughput bench" idling at 48 W — the
user's two hardware-utilization interventions (parallelize the build; watch the
watts) are both visible in this one picture. Second, as a live exercise of the
night's discipline: the user asked "are you sure the run isn't launched? your
subagent ate 90% of all cores for 22 minutes" — and the ground check took one
command: the hog was the V1 wave's legitimate 1M gate bench, no process anywhere
near the 27 GB production arena, and the V2 launch script still hard-locked behind
its deliberate `exit 90`. Trust, then verify, then screenshot.

## 8quater. The morning of the index build: two red gates, a probe, and a factor of four (July 8, 09:00 → 11:00)

The index-build méta-goal opened with its V1 wave — replace the full fp32
materialization (54.7 GB at target scale, infeasible) with on-demand fp16 reads
from the mmap'd arena — and the wave came back in the most instructive shape a
wave can take: **code green and audited, two hard gates honestly red.**

The first red gate was the architect's own fault, and the ledger now says so in
those terms. The memory ceiling (≤ 1.5 GB on the 1M bench) had been forged by
subtracting "the ~2 GB fp32 buffer" from the 3.34 GB golden — 512-dimension
arithmetic applied to a 128-dimension proxy bench, where the eliminated buffer
actually weighs 0.51 GB. The re-measure at rest landed at 2.87 GB, within 40 MB
of the correctly recomputed target: the structural objective (no O(N×dim×4)
buffer) was achieved and *provable by the exact size of the drop*, while the
gate number itself was fiction. A gate miscalibrated at forge time fails exactly
like a real regression; only the decomposition tells them apart.

The second red gate survived its excuse. The coder had contaminated his own
duration measurement (he launched the all-cores race suite concurrently with his
own parallel build — and named it in his retex as the night's lesson 9 replayed
on himself). But the isolated re-measure barely moved: 1,013 s against 1,031
contaminated, versus a 444 s golden. The ×2.28 was real — the intrinsic CPU
price of decoding fp16 on every distance evaluation. The user's arbitration
choice was the correct experimentalist's one: neither accept nor optimize on a
projection — **run a 5M probe on the real arena first** (a header-patched prefix
of the production file; twenty minutes of scripting, one hour of compute,
against a nine-hour decision).

The probe paid for itself before it even finished. Forty minutes in, its RSS
read 15 GB — and the single most valuable measurement of the morning was the
one-command decomposition (`smaps_rollup`): 5.0 GB of *file-backed, reclaimable*
mmap pages, and **10.8 GB of anonymous Go heap** — the incompressible part —
projecting to ~57 GB at 26.7M. The full run, launched naively, would have been
killed by the OOM guard around hour eight or nine. Two follow-up checks
sharpened the verdict in minutes: the GC clamp was *already present* in
`BuildFromArena` (so the cheap lever was spent), leaving ~1.7 KB of live heap
per node against ~400 bytes of useful payload — a factor of four of per-node
slice headers, allocator rounding, and GC-visible pointers. The disease the hot
plane had cured on the *read* side of the graph, never treated on the *build*
side. V1bis was dispatched with that number as its contract: profile first
(decompose the 1.7 KB before compacting — the suspect is named, not convicted),
then flat build-time adjacency on the hot-plane pattern, with the lock-free
neighborhood swap of the parallel build named as the hard point to preserve
under the race detector.

What this half-day adds to the collection:

- **A red gate is a measurement plus a target; interrogate both.** One of the
  two reds was a real cost, the other was forge-time fiction — and they looked
  identical until the drop was recomputed dimension-by-dimension.
- **RSS is not one number.** File-backed mmap residency is elastic; anonymous
  heap is a wall. `smaps_rollup` separates them in one command, and every
  memory projection for an mmap-heavy process should quote `Pss_Anon`, not RSS.
- **Probe cost scales with decision cost.** One hour of prefix-build against
  the real arena re-priced a nine-hour, possibly-OOM run — and turned a
  speed arbitration into a feasibility requirement before any time was burned.
- **Per-node slices are a hidden 4× at scale.** Headers, allocator rounding and
  pointer-chasing overhead that no unit test and no 1M bench flags as a defect
  become the binding constraint at 26.7M. The flat-arena pattern applies to
  construction, not just serving.

## 8bis. July 8, evening — the audit turned on the auditors

(Session "horosvec", continuation appended to the unified retrospective. Morning
of the same day had delivered V1/V1bis and the GPU pivot; the machine rebooted at
13:48, killing the 5M CPU probe and the pending CAGRA production build — both
restartable, neither lost data. The evening session did not resume the build; it
audited the algorithmic core instead.)

The evening ran two audits of the same code from opposite directions. First, an
internal conformity audit of the RaBitQ implementation against the paper
(estimator derivation, Hadamard rotation isometry, LUT equivalence, persistence
cycle): verdict **conformant**, zero hard findings, two benign softs. Second, an
external model's review arrived claiming, among six findings, that the asymmetric
estimator was *non-canonical* — missing a ‖q′‖ factor, dividing by L1 instead of
√d — and that the fp16 converter flushed subnormals too early.

Both headline findings were **refuted at the ground**, and the refutations are
the valuable part. The estimator claim died under a full re-derivation: in the
asymmetric case, signDot already carries the query's real coordinates, so ‖q′‖
must *not* appear (it cancels algebraically, along with both √d factors), and
the L1 division *is* the paper's per-vector unbiasing factor ⟨ō,o⟩ — the very
thing that distinguishes RaBitQ from naive sign quantization. The reviewer had
used the file's own crude *symmetric* benchmark helper as the canonical
reference — the weaker function judging the stronger one. The fp16 claim died on
a biased-vs-unbiased exponent confusion: the flush threshold `exp < -10` is
biased-fp16 for values below 2⁻²⁵, exactly what round-to-nearest sends to zero
anyway; subnormals down to 2⁻²⁴ are produced with correct round-half-even.

Three of the remaining findings were confirmed and were genuinely useful — with
one aggravation the reviewer missed: the recall test floor of 0.50 (a regression
to 0.55 would pass green, README numbers never asserted), the misleading
comments (including the one that had *caused* the false estimator finding), and
`recomputeMedoid` materializing every vector — where the real defect was worse
than the reported 5 GB spike: in arena mode (the mode that scales) the vector
column is empty, so the automatic medoid repair silently computes garbage.

The three fixes were forged into a /goal (Object `proj_horosvec_fixes`, Job
019f436b) and delivered the same evening in three local commits (32e003b,
946a639, f97d96a). Two incidents during execution earned their place here:

- The coder's first draft of the streaming medoid assumed node_ids contiguous
  0..n-1 — plausible (no unitary DELETE in the code), and the new arena test
  passed. The *full* suite exposed ids 1000..2499 under node_count=1500: rebuild
  renumbers. Only the run-everything rule caught a silent recall regression.
- The forged floor "≥ 0.90" was itself forge-time fiction for the clustered
  configuration: measured four times deterministic at 0.678, an intrinsic
  1-bit-RaBitQ ceiling on tight clusters. The coder deviated, documented,
  and set per-configuration floors (0.90 uniform / 0.60 clusters) instead of
  obeying the brief — the audit's own "0.98–1.0 margin" had only ever been
  measured on the uniform distribution.

What this evening adds to the collection:

- **An external review is a finding generator, not a verdict.** Six findings:
  two refuted at the ground (both would have triggered pointless rework), three
  confirmed (one under-diagnosed), one cosmetic. Adjudication — re-derive the
  math, re-read the exponent convention, re-check the failing mode — is where
  the value is; relaying unverified findings would have been negative work.
- **A lying comment is a defect with a blast radius beyond its file**: the
  false intermediate formula at rabitq.go:72-74 manufactured an entire external
  P1 finding. The fix is to inline the exact derivation where the doubt arose.
- **A refuted finding is worth engraving with its proof.** Both refutations
  went into the project reminder with the full derivation, so the next reviewer
  (human or model) does not reopen the argument from scratch.
- **Contiguity of ids is a hypothesis, not a property**, until an oracle on the
  real path has exercised it — the constructed test case had it, the rebuilt
  index did not.
- **A forged threshold inherits the distribution it was measured on.** Lesson 4
  (data-driven decisions carry their dataset's signature) applies to the forge
  itself: hard floors must be per-distribution, or they are fiction for all but
  the one that produced them.

## 8ter. July 9 — the production run, end to end, and the disk that lied by omission

The full HNbook run executed overnight as a single autonomous sequencer (W0→W4),
and the numbers vindicated the GPU pivot beyond the plan: CAGRA built the
26,691,317-node adjacency in **17.1 minutes** (transfer 117.7 s, export 5.8 s,
~260 W sustained), the new import path turned it into a complete horosvec index
in **21.7 minutes** (19.3 GB SQLite, vector-less, node_count verified twice),
and the end-to-end validation scored **overlap@10 = 0.99** against exact brute
force over the full arena (18 queries at 10/10, two at 9/10), with the rerank
served entirely from the arena (SQL loads = 0). The CPU path the morning had
priced at 9-10 hours was replaced by a 39-minute chain.

Three incidents made the day's lessons, each caught by an oracle rather than
by luck:

**The VRAM squatter and the CPU detour.** Mid-run, another actor loaded a 12B
model and filled the GPU, killing the embedding slot the validation needed for
its 20 query vectors. The unload-nothing rule held; the detour was a
CPU-embedded query set — gated by a decisive parity oracle: three documents of
known arena rank re-embedded on CPU and compared to their arena vectors. The
oracle did double duty: it proved the CPU pipeline equivalent (cos ≥ 0.9999)
AND eliminated a plausible variant (EOS-appended) that scored as low as 0.906 —
a variant that, presumed correct, would have silently corrupted the entire
validation. The oracle didn't just verify the chosen path; it chose the path.

**The harness that reaped its own runs.** The validation was killed twice at
~60-90 s, each time exactly at the next turn boundary, with no OOM and no
error. The give-away was the pattern, not any log line: same offset, same
trigger, memory plentiful. A detached relaunch (setsid+nohup, outside the
harness's task tracking) survived and finished — diagnosis proven by
counter-experiment. Operational rule: production runs launch detached; the
supervising agent monitors files, it does not own processes.

**The disk that made a 100× lie.** Validation latency read as p50 2.86 s cold —
and 2.69 s "warm" six hours later, which should have been the tell: a warm run
that isn't faster isn't warm. The working set lived on the 18 TB rotational
archive disk; every rerank page fault paid a mechanical seek, and the cache had
been evicted between runs. Moved to NVMe (55 GB, integrity checked
byte-for-byte before the HDD copy was deleted), the same index, same queries,
same oracles returned **p50 27.6 ms cold-ish, 7.8 ms warm** — a ×100-370
collapse with identical quality. The algorithm had been innocent all along;
the medium was the message. The serving decision then nearly made itself:
mmap from NVMe (pinning 27 GB of arena in RAM would buy only the 28→8 ms gap
at the cost of half the machine's memory), decided by the user and engraved.

What this run adds to the collection:

- **A "warm" measurement that isn't faster than cold is a broken premise
  detector** — either the cache was evicted or the bottleneck isn't where
  assumed. Interrogate it before trusting either number.
- **Latency verdicts carry their storage medium as silently as their dataset.**
  A p50 measured on a rotational disk says nothing about the engine — name the
  medium in every consigned latency, and keep active working sets off archive
  storage by rule, not by memory.
- **A parity oracle between two implementations of the same pipeline is cheap
  and decisive** — and its highest value is eliminating the plausible-but-wrong
  variant, not confirming the chosen one.
- **When a process dies twice at the same offset with no system trace, suspect
  the supervisor** — and prove it with a detached counter-experiment before
  blaming the workload (or a human).
- **Two hard-verified stores still need a reconciliation story**: the metric
  fitness of the GPU graph (inner_product) for the L2 walk was proven by a
  10k-vector norm check on the real arena before a single line of import ran —
  the cheapest BLOCKER ever lifted.

## 8quater. July 9, afternoon — the benchmark that measured the wrong machine

The final three-engine benchmark (horosvec vs hnswlib vs sqlite-vec, two
corpora, ef sweeps, and — for the first time — client concurrency sweeps)
produced two alarming findings against horosvec: a throughput cliff (×5.7
collapse between ef 128 and 256 on 512-dim data, where hnswlib lost ×1.8) and
a bimodal concurrency profile (×16 scaling on one corpus, dead flat on the
other). Both were engraved as findings. Both then died under instruction —
and the way they died is the lesson.

The instruction sequence: reproduce at reduced scale (300k instead of 1M —
the cliff *moved*, from ef 256 to ef 512, proving a resource regime rather
than an algorithmic cost), then profile the falling point with `perf` filtered
to the query window. The profile was unambiguous: 41% garbage collection, 22%
sweep, 21.6% `database/sql.withLock`. The search path was doing SQL. One grep
later: the bench harness enabled the fp16 arena — the production
configuration — only through a silent opt-in environment variable
(`HOROSVEC_ARENA`), and the final benchmark had not set it. Every horosvec
line had measured the DB-blob fallback path, where each rerank candidate is a
row-by-row SQL blob read: an allocation storm whose GC cost grows with the
live heap (hence the cliff moving with N) and whose lock is process-wide
(hence concurrency scaling dying exactly where the allocation rate saturated).

The rerun with the arena enabled dissolved both findings on SIFT: the cliff
flattened into hnswlib-shaped smooth decay, mono-client throughput tripled at
ef 512, and 32-client throughput multiplied by **56** (1,581 → 88,218 QPS at
ef 64 — above hnswlib's 72,618, though at lower recall; at iso-recall on
128-dim data hnswlib keeps ×5, the known low-dimension handicap of 1-bit
codes). The production path had been innocent all along — as the 26.7M index
(rerank SQL loads = 0, p50 7.8 ms) had already testified.

What this afternoon adds to the collection:

- **A benchmark harness that selects its configuration through a silent
  opt-in is an instrument that lies by default.** The measured mode must be a
  field in the output record, never an invisible environment variable. The
  fair-comparison run you *meant* to do and the run you *did* are
  distinguishable only if the artifact says which is which.
- **Findings against a system must name the code path they measured.** Both
  "horosvec findings" were real measurements of a real mode — just not the
  mode anyone ships. Engraving them without the path qualifier would have sent
  the backlog chasing ghosts in the production engine.
- **The reproduce-smaller step earns its cost twice**: it made profiling cheap
  (6-minute builds instead of 20) and its side effect — the cliff *moving*
  with N — was itself the diagnostic that reclassified the defect from
  algorithmic to resource-regime before a single profile sample was taken.
- **GC percentage plus a lock in the profile is a two-line verdict**: 41%
  gcDrain + 21% withLock reads as "allocation storm behind a shared mutex" —
  the profile named both the disease and the door it came through.

## 8quinquies. July 12 — the qualification campaign: five daughters, two documented failures

A coordinated campaign (five bounded tasks forged with pre-frozen falsifiable
criteria, each adversarially refuted by three independent model families before
being engraved) closed the questions the benchmark days had opened. Its two
*failures* are the valuable part, because both were closed with their thresholds
intact rather than renegotiated.

**P0 — the incremental claim, qualified.** Two doubts hung over the db-blob
incremental path: does building the graph with quantized distances (instead of
fp32) degrade it, and does grafting inserts into a built graph erode recall?
Measured on real HN vectors: quantized-build recall matches fp32-build within
noise (1.0000 vs 0.9995), and +50% dynamic inserts cost at most 0.0050 recall
against a full rebuild, at equal query throughput. Insert throughput is ~5×
slower than build (42 ms/vec vs 8.6) — a daily HN delta grafts in ~8 minutes,
a monthly catch-up prefers the rebuild. The conditional that ARCHITECTURE.md §4
had carried was lifted by a dated measurement, not by argument.

**P1 — the BM25 pre-pass, killed by its own harness.** The hypothesis: a cheap
lexical front-end (FTS5/BM25 top-200 per shard) shrinks the candidate set before
exact dense scoring. At shard scale (~10k items) the answer is unambiguous:
hybrid recall@10 0.5445 vs 1.0000 for plain dense search, at *worse* latency
(p95 62.9 ms vs 15.8 ms) — dense neighbors of a story often share no title
token, and the rerank the pre-pass was supposed to save costs 0.8 ms anyway.
The first run of this harness returned recall 0.0300 — exactly the scope-
reduction ratio, i.e. exactly chance — because arena rank had been read as
`id-1`. The embedding pipeline had skipped dead items; the `.ids` sidecar is
the only truth. The initial "verification" had checked the first ten ranks,
the only region where the two mappings coincide: a universal claim generalized
from a non-exhaustive enumeration, caught because *recall equal to scope
reduction is the signature of misalignment*.

**P2 — the SIMD wall.** An AVX2 kernel for the graph walk won ×1.26 in
isolation and +0.4-5.3% end-to-end under concurrency 32 — the ceiling is
memory bandwidth, not arithmetic. Abandoned with the threshold (15%) intact.
Corollary engraved for future perf work: a gain threshold must name its
concurrency *and* its scale, or it is not falsifiable.

## 8sexies. July 13 — the demo that contradicted the README, and the server measured in the face

With the incremental claim now public, the demo at horosvec.hazyhaar.fr was
still serving a snapshot frozen at July 8 — and returning bare ids
(`{"id":"6069168","score":0.30}`) with the titles sitting unused in a database.
A live claim served by a stale demo is a walking refutation of itself. A
six-task project (coordination mother + five bounded daughters) was forged to
make the demo *prove* the claim: a server-side title store, human-readable
results, a daily delta chain over the qualified insert path, monthly arena
publication, federated search across the frozen monolith and the live shard.

The first daughter — measure the three server unknowns before anything sizes
itself on assumptions — closed the same afternoon, and promptly falsified the
sizing folklore. The "2-3× slower than the 32-core dev box" band that had been
carried since July 10: the measured ratio on the production Xeon (12 threads)
is **×4.76** (three runs, 307.8/309.7/312.1 s on the *same* 11,600-vector real
input as the local 65 s baseline). Peak RSS ~57 MB, identical to local — a
non-issue on 64 GB. And the number nobody had: a concurrent build costs the
live demo **+28% on p95** (99.7 → 127.3 ms over 181 samples, identical query
profile), with latency dominated by the embedding sidecar in both regimes —
measured in the face, not inferred by elimination. Five minutes per daily
shard remains operationally trivial; the extrapolated band was still wrong,
and now a real contention threshold exists for the automation to be judged
against.

One harness lesson from the forging itself: the task store enforces its
criteria limit in **UTF-8 bytes** while every composer upstream counted
characters — four out of six engravings bounced until the accented text was
measured in the unit the gate actually checks. The gate and the writer must
name the same unit.

The first daughter to ship end to end was G2, the server-side title store — the
fix for a demo that returned bare ids while the titles sat unused. The design
decision (probed before dispatch): the module carries no parquet reader and
adding one would breach its dependency discipline, so the 28M historical bulk
is a one-shot `duckdb` conversion (parquet → SQLite, the tool that made the
original text db), while the deployed Go binary reads only the HN delta —
pure-Go, CGO-off, zero new dependency. A composer coded it "à poil" (every
frame inlined, incremental report to scratch); the gates and the parquet
mechanism were re-run at the ground rather than relayed — id 1 really is "Y
Combinator". The full 28M conversion took 33 seconds and produced a 905 MB
store (28,737,557 rows, zero duplicates), rsynced to the server and verified
queryable there. Two honesties surfaced in passing: `duckdb` refuses
`INSERT OR REPLACE` on an attached SQLite (so idempotence lives in the Go
delta path, not the bulk), and ~0.4% of items carry `ts=0` — a real title with
no timestamp, a data-quality note the date rendering must handle. The store is
now *placed* but not *visible*: the served binary has no join yet — that is the
next daughter. A published claim is only proven when the demo shows it.

Driving the demo surfaced the sharper design question. A search for "jailbreak"
returned, correctly, comments whose text is literally "jailbreak" (d²≈0.0001) —
semantically perfect, but titleless one-word cards that read as garbage. The
ranking was *not* inverted (verified in the served JS and the API); the corpus
is the issue: 24.4M of the 28.7M items are comments. The resolution is not to
drop comments (they are valid evidence — "who knows what's buried there") but to
rank and display by **thread**: the root story on top, matching comments
clickable inside a folded tree. Two sub-questions fell out. Ranking a thread:
sum *saturating similarities* (not distances) of its retrieved hits with a
max-dominated aggregate (log-sum-exp), and crucially **do not divide by thread
size** — that inverts toward trivia exactly like the one-word comment, a 500-
comment thread with 10 strong hits scoring below a 2-comment thread with one.
Building the tree: the `parent` backlink is the primitive, and because a HN
child's id always exceeds its parent's (verified: 99.996% of 28.7M, roots carry
parent=0), the root story of every item resolves in a single ascending-id pass —
the parent is always already resolved when the child is reached.

That foundation was engraved, adversarially refuted by an independent model, and
implemented. The refutation earned its keep: it rejected the "self-root the
unresolved comment" fallback as a silent-false root, forcing an explicit orphan
mark and a mandatory second pass for the 1234 id-anomalies; it demanded a
deferred queue for out-of-order arrival and scale tests over 1000+ threads
instead of toy samples. The delivered extension was verified independently at the
ground — a re-coded parent-walk agreed with the computed root_id on 2000 random
items with zero mismatch, orphans were the parent-absent case marked explicitly
(never self-rooted), and an out-of-order append (child before parent) correctly
parked in the pending queue and resolved when the root arrived. The tree lives
entirely in the metadata store, joined to the vector index by ext_id at query
time — the surgical graft that P0 qualified never learns what a parent is.

The display was then rebuilt on that foundation and shipped to the live demo.
The adversarial pass again paid: it forced tau and the raw-hit count out of
hiding (both govern ranking and recall), named the internal tension between a
summed thread score and top-k retrieval — a diffuse thread with no hit in the
top-k is structurally invisible, accepted and documented — and rejected loading
the real ancestor chain as gold-plating, so the tree renders as the root story
plus its matched comments indented by depth, with a decorative fold. Two facts
it flagged were settled at the ground rather than assumed: every story and every
orphan carries root_id = itself, so a story-hit or an orphan is a valid one-node
thread. The result: "jailbreak", which a week earlier returned bare ids topped
by a one-word comment, now returns 295 titled threads — "AppleID password brute
force proof-of-concept" carrying the exact-match comment, "JailbreakMe 3.0",
"Evasi0n iOS 6.x jailbreak" with its two matched comments nested — served in
~100 ms with the freshness date drawn from the store, the client no longer
round-tripping to the Hacker News API for a list title. The comment that
matches is one click from its text; the thread it lives in is finally legible.
Known edge left standing: matched comments show no inline snippet (their text
is not in the store — only titles are), so the "what's buried here" is revealed
on click, not in the list.

That edge was then closed. The user's verdict on the empty detail pane —
"horrible" — drove storing the comment text itself: a `text` column backfilled
from the parquet (24.1M items carry text; the store grew from 2 GB to 13 GB),
and a serve endpoint that reads the item's text from the store rather than the
Hacker News API. Now the matched comment shows its text inline under the story,
and clicking it fills the detail pane with the full comment — "Rust is
memory-safe; C++11 is not." — all from the local store, the browser no longer
touching HN for any content. Two things earned their keep in the doing. First,
a ground check caught a silent corruption: the 13 GB store arrived on the server
104 MB short and `malformed` while `rsync -z` had returned exit 0 — a compressed
transfer that lied about success; a checksum re-sync (`rsync -c`) repaired it,
and `PRAGMA quick_check` is the oracle, not the exit code. Second, the first
render showed raw `<a href=…>` markup in snippets for comments containing HTML —
XSS-safe (rendered as text) but ugly; stripping tags before entity-decoding, in
that order (the tags are literal, the entities encoded), cleaned it. The demo now
reads like a thread of real discussion, titles and comment text and all.

The freshness chain (G4) then went live — and the ground kept teaching. The
design is two cursors: a head that fetches each day's new items (priority), and
a backfill that grinds the frozen Oct-2021→now gap without pressure, in short
time-bounded runs so a crash costs one small batch and the cursor resumes. Two
real bugs surfaced only by running it, not by the tests: a backfill budget of
`0` was read as *unlimited* — the exact runaway the two-cursor design exists to
prevent — now rejected outright; and the backfill's ceiling was the *moving*
head, so on the next cycle it re-ingested everything the head had taken — fixed
with a frozen `head_origin` boundary the two zones can never cross. A real
bounded run then ingested genuine 2026 items ("Show HN: Ankole", dated today)
into a separate current shard while the served monolith's mtime never moved, and
a robustness layer landed for the operator: a `run_log` table queryable by
`-show-runs`, resume-past-the-failed-item, retry on transient SQLite contention,
a `-log-file`, and errors that name the phase and the item id. It runs now on a
5-minute systemd cadence. Two honest debts stayed open, both found by measuring
rather than assuming: end-to-end throughput is ~2.2 items/s (the per-item
subprocess append, not the embed, is the bottleneck), so at the chosen duty the
gap closes in months not the hoped-for one — batching the append is the fix; and
a separate delta store means a backfilled item whose parent lives in the frozen
monolith is an orphan there, so the federated search (G5) must resolve thread
metadata across both stores. The chain is live; making it fast and making its
freshness *visible* are the next two moves.

Federation (G5) made it visible, and closed the campaign. The user chose to have
the delta write directly into the served store rather than a side store — WAL
plus a busy timeout carry the one-writer-many-readers case cleanly, and it
erased the cross-store metadata split in one stroke: a backfilled comment's
parent now lives in the same store, so its thread resolves with no orphan. The
search itself became a `federatedSearcher` — one object implementing the
existing search interface, querying the frozen monolith arena and the live delta
shard, merging by distance and deduping by id — so the thread-grouping and
unified-store metadata above it never changed. It reloads the shard every five
minutes, so new items surface without a restart. A 502 taught the last lesson,
and it was a sharp one: the freshness readout became a per-request `SELECT
max(ts)`, and on a 13 GB store with no index on `ts` that is a full scan — fine
in isolation, but under the delta's concurrent writer it blew the query's
deadline and crash-looped the demo. A rollback restored service in under a
minute; an index on `ts` took the query from 17 seconds to 3 milliseconds; the
binary went back out. The oracle is the same one that has run through this whole
month: measure the thing under the condition it will actually run in, not in
isolation. The payoff is on the page now — a search for a story posted *today*
returns it at rank one, dated 13 Jul 2026, above the frozen corpus, the header
reading "2006 → live (federated: monolith + delta shard)". The demo that a week
ago answered "jailbreak" with a bare id and an empty pane now reads like a live
Hacker News: titled threads, real comment text, and a corpus that no longer ends
in 2021. What remains are named debts, not gaps — the backfill throughput, the
monthly shard merge, filtering deleted comments — each engraved, none blocking.

## 9. The lessons, collected

1. What production does not exercise stays broken; a comparative benchmark is
   an instrument of discovery, not a scoreboard.
2. A ctx guard inside a loop protects nothing if the loop can be empty.
3. Microbenchmarks with fixed input patterns lie about branchy code.
4. A data-driven decision carries the signature of its dataset; new
   distribution, new decision.
5. Perfect recall masks traversal defects.
6. In approximate+rerank pipelines, the false-green lives where old code reads
   a new world without erroring; the rerank stage launders estimator garbage.
7. Intelligent plans fail at their premises, not their arithmetic — refute the
   premises with ground oracles before building.
8. Test the mode × read-path matrix under the *default* configuration.
9. A measurement taken on a contended machine is not a measurement.
10. The GPU saturation oracle is power draw; more generally, measure the
    independent physical oracle, never the instrument's self-declaration.
11. For crash-consistency, enumerate every kill window and replay each resume;
    the idempotence oracle for GPU-produced data is the ids, not the bytes.
12. Independent adversarial review at every altitude — plan, design, code,
    measurement — is the mechanism that caught everything above. It is
    expensive. On this day it was cheaper than any one of the errors it
    prevented.
13. (Night run.) Different layers wear the same symptom; only single-variable
    bisection separates them — and a wrong diagnosis committed with its date,
    then corrected, beats a hedge that never commits.
14. (Night run.) Never truncate input to a model's exact context window; a
    full-window sequence is a boundary condition in someone else's scheduler.
    Keep a margin.
15. (Night run.) Fail-loud plus checkpointed resume converts production
    failures into free retries and forces real bugs to the surface; a retry
    loop would have masked all four and shipped the landmine downstream.
16. (July 8.) An external review is a finding generator, not a verdict:
    adjudicate every claim at the ground before any rework — the two headline
    findings of the day were both refuted by re-derivation.
17. (July 8.) A lying comment can manufacture a false P1 in someone else's
    audit; fix it by inlining the exact derivation at the point of doubt, and
    engrave refutations with their proof so the argument is never reopened.
18. (July 8.) Id contiguity, like any structural hypothesis, holds only on the
    path that has been exercised — a rebuilt index renumbers; run the full
    suite, not just the new test.
19. (July 8.) A hard threshold is only as portable as the distribution it was
    measured on; forge per-distribution floors or forge fiction.
20. (July 9.) A latency measurement carries its storage medium like a dataset
    signature; a "warm" run that isn't faster than cold is a broken-premise
    detector, not a data point.
21. (July 9.) A parity oracle between two implementations of one pipeline is
    the cheapest decisive instrument there is — and its real product is the
    plausible variant it kills, not the one it confirms.
22. (July 9.) Production runs launch detached from the agent harness; the
    supervisor watches files, it never owns the process. A process dying twice
    at the same offset with no system trace indicts the supervisor.
23. (July 9.) Active working sets live on NVMe; archive disks hold archives.
    The rule costs one rsync; its absence cost a 100× misreading of the
    engine's latency.
24. (July 9.) A harness that selects its configuration through a silent
    opt-in lies by default; the measured mode belongs in the output record.
    A finding must name the code path it measured.
25. (July 9.) Reproduce-smaller before profiling: it makes the profile cheap,
    and the way the defect *moves* (or doesn't) with scale is itself a
    classifier — resource regime vs algorithmic cost — before any sample.
26. (July 9.) High GC share plus a shared lock in a query-window profile
    reads as "allocation storm behind a mutex"; the profile names both the
    disease and the door.
27. (July 12.) Recall equal to the scope-reduction ratio is the signature of a
    misaligned id mapping; and a mapping "verified" on its first ten ranks —
    the only region where two mappings coincide — is a universal claim built
    on a non-exhaustive enumeration. The `.ids` sidecar is the only truth.
28. (July 12.) A cheap deterministic front-end only pays where the exact stage
    is expensive; at shard scale the dense path is already exact and cheap,
    and the lexical pre-pass costs recall it can never buy back. Scale regime
    is an input of the harness, not a footnote.
29. (July 12.) A performance gain threshold must name its concurrency and its
    scale; an isolated kernel speedup is not a result, it is a hypothesis
    about a bottleneck that memory bandwidth may veto end-to-end.
30. (July 12.) Close a failed task with its threshold intact. The two failures
    of the campaign are worth more than its successes precisely because the
    bar did not move to accommodate them.
31. (July 13.) A published capability served by a stale demo refutes itself in
    public; freshness is not a nicety of the showcase, it is the claim's
    living proof.
32. (July 13.) An extrapolated hardware band survives until someone runs the
    same input on the target box: "2-3×" was ×4.76. Measure the contention on
    the live service *during* the build — the delta, not folklore, is the
    threshold automation must be judged against.
33. (July 13.) The gate and the writer must count in the same unit: a byte
    limit enforced against character-counting composers bounces every
    accented payload near the bound.
