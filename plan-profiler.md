Profiler Improvement Plan

Goal
- Rework the profiler into a profiler-specific workflow that keeps `justfile`
  as a thin wrapper, moves profiler policy into profiler-owned config/runtime,
  and removes profiler instrumentation overhead from normal simulation builds.

Decisions Locked In
- Introduce `profiler.lua` as the profiler-specific config entry point.
- Keep the six profiler seeds inside `profiler.lua`.
- Add a dedicated profiler executable / entry point instead of growing the
  standard `moonai` executable into profiler orchestration.
- Keep `justfile` as a thin wrapper only; no profiler orchestration logic in it.
- Compile profiler instrumentation only into the profiler build so standard
  simulation/runtime builds pay no profiler overhead.
- Do not implement multiple profiler detail levels.

Detail-Level Evaluation
- Option A: multiple detail levels (`basic`, `detailed`, `hotspot`)
  - pros:
    - flexible for different use cases
    - lets us trade off overhead vs observability at runtime
    - useful for occasional deep hotspot debugging
  - cons:
    - adds configuration and reporting complexity
    - makes cross-run comparison easier to misuse
    - requires more metadata, tests, docs, and validation paths
    - increases maintenance burden for little day-to-day value
- Option B: one fixed instrumentation level
  - pros:
    - simpler mental model
    - simpler implementation, tests, and docs
    - profiler outputs stay directly comparable
    - easier to reason about overhead and trustworthiness
  - cons:
    - less flexible for one-off deep investigations
    - must choose scope carefully to avoid too much or too little detail

Sweet Spot Chosen
- Use one fixed instrumentation level.
- The fixed level should be a balanced, coarse-to-mid granularity level:
  - keep generation-level and major phase timings
  - keep important simulation subphase timings
  - keep cheap counters
  - do not instrument deep per-agent/per-query hot loops with wall-clock timers
- Rationale:
  - this preserves profiler clarity and keeps overhead low enough for reliable
    benchmark-style output
  - it avoids the major complexity cost of detail levels
  - it still exposes the bottlenecks that matter for the current architecture

Instrumentation Scope To Keep
- `generation_total`
- generation/evolution phases:
  - `build_networks`
  - `prepare_gpu_generation`
  - `cpu_eval_total`
  - `compute_fitness`
  - `speciate`
  - `remove_stagnant_species`
  - `reproduce`
  - `logging`
- simulation phases:
  - `simulation_tick`
  - `agent_update`
  - `process_energy`
  - `process_food`
  - `process_attacks`
  - `boundary_apply`
  - `death_check`
  - `count_alive`
- GPU host-side phases:
  - `gpu_resident_sensor_build`
  - `gpu_resident_tick`
  - `gpu_finish_unpack`
  - legacy-path GPU events that are still real and reachable
- cheap counters:
  - `ticks_executed`
  - `grid_query_calls`
  - `grid_candidates_scanned`
  - `food_eat_attempts`
  - `food_eaten`
  - `attack_checks`
  - `kills`
  - `compatibility_checks`

Instrumentation Scope To Drop From Default Profiler Timing
- deep accumulated wall-clock timers in hot paths:
  - `compatibility_distance_accumulated`
  - `physics_build_sensors_accumulated`
  - `spatial_query_radius_accumulated`
- Rationale:
  - these sit inside repeated or parallel hot paths
  - they inflate profiler overhead
  - they create confusing inclusive/accumulated semantics
  - they are more suitable for a one-off diagnostic profiler, which we are not
    building here

Architecture Plan

1. Profiler-Owned Config Entry Point
- Add `profiler.lua`.
- `profiler.lua` defines profiler suites, not normal experiment batches.
- Each profiler suite references one base experiment shape and expands it into
  six runs with six fixed seeds.
- `config.lua` remains the normal simulation/experiment entry point and must not
  change behavior because of profiler work.

2. Dedicated Profiler Executable
- Add a separate profiler executable target in `CMakeLists.txt`.
- The profiler executable is distinct from `moonai`.
- Responsibilities:
  - load `profiler.lua`
  - select a profiler suite
  - execute six raw profiler runs
  - validate comparability across the six runs
  - trim fastest and slowest runs
  - aggregate the remaining four runs
  - write suite-level metadata/output

3. Keep `justfile` Thin
- `just profile` should only:
  - build the profiler target
  - invoke the profiler executable with standard arguments
- `just analyse-profile` should only wrap report generation.
- No seed orchestration, trimming policy, or suite logic should live in
  `justfile`.

4. Six-Seed Profiler Workflow
- For each suite in `profiler.lua`:
  - run the same resolved config six times
  - only seed changes between runs
- After all six runs finish:
  - compute average `generation_total` for each run
  - sort all six runs by that value
  - drop one fastest run and one slowest run
  - average every event/counter summary across the remaining four runs

5. Output Model
- Preserve raw per-run profiler output as the primitive artifact.
- Add suite-level output on top:
  - suite manifest
  - suite aggregate profile summary
  - list of kept and dropped runs
  - trimming metadata
  - comparability metadata

6. Compile-Time Profiler Elimination In Normal Builds
- Add a dedicated compile definition for the profiler build, e.g.
  `MOONAI_BUILD_PROFILER`.
- In normal `moonai` builds:
  - profiler scope macros expand to nothing
  - profiler counter macros expand to nothing
  - profiler helper wrappers compile away
- In profiler executable builds:
  - those same macros/helpers emit real instrumentation
- Result:
  - normal simulation runtime does not pay profiler overhead
  - profiler overhead only exists in the profiler executable where it belongs

7. Replace Direct Profiler Calls With Thin Macros/Helpers
- Keep RAII internally, but expose instrumentation through a small surface:
  - scope macro
  - counter increment macro
  - optional callable wrapper helper where lexical scoping is awkward
- Examples of intended usage shape:
  - `MOONAI_PROFILE_SCOPE(ProfileEvent::Speciate)`
  - `MOONAI_PROFILE_INC(ProfileCounter::TicksExecuted)`
- Goal:
  - shorter and more consistent call sites
  - compile-time no-op behavior in non-profiler builds

8. Profiler Overhead Validation
- Measure overhead inside the profiler system itself.
- Compare:
  - profiler executable with the chosen fixed instrumentation level
  - a minimally instrumented or instrumentation-disabled profiler build mode for
    the same six-seed suite
- Report overhead as:
  - absolute ms/gen delta
  - percentage delta
- Acceptance goal:
  - default profiler overhead should stay low enough that output remains useful
    and benchmark-like
  - if overhead is too high, reduce instrumentation scope instead of adding
    detail levels

9. Timing Semantics Cleanup
- Keep event semantics explicit in metadata and reports.
- Distinguish at least:
  - wall-clock phase timings
  - inclusive phase timings
  - host-side GPU enqueue/wait timings
  - counters
- Do not present nested numbers as additive totals.
- Keep suite aggregation aware of these semantics.

10. GPU Timing Semantics
- Keep current host-side GPU timings, but label them honestly.
- If needed later, optional CUDA-event or NVTX-based device timing can be added,
  but it is not part of this implementation plan.
- Current implementation should focus on making existing host-side timings
  trustworthy and clearly described.

11. Comparability Metadata
- Each raw run should record:
  - suite name
  - base experiment identity
  - seed
  - config fingerprint
  - profiler schema/version
  - profiler executable/build identity
- The suite aggregator must reject runs that do not match on required identity
  fields.

12. Profiler Analysis Pipeline Changes
- Update profiler analysis so the suite becomes the main reporting unit.
- Report should show:
  - all six raw runs
  - dropped fastest and slowest runs
  - aggregate stats from the kept four runs
  - variance/spread across the kept four runs
- Raw-run inspection remains available for debugging.

13. Tests
- Add tests for:
  - raw profiler metadata
  - suite manifest creation
  - trimming correctness
  - aggregate averaging correctness
  - comparability checks
  - macro no-op behavior in non-profiler builds
  - macro active behavior in profiler builds
- Ensure standard `moonai` build behavior remains unchanged.

14. Documentation
- Update docs to explain the separation clearly:
  - `config.lua` is for normal simulation/experiments
  - `profiler.lua` is for profiler suites
  - `moonai` is the standard runtime
  - profiler executable is the benchmark/profiling runtime
  - `justfile` only wraps those commands
- Document the fixed six-seed trim policy.
- Document that seeds live in `profiler.lua`.
- Document that normal builds do not include profiler instrumentation overhead.

Implementation Order
1. Add profiler build target and compile-time profiler gating.
2. Add profiler macros/helpers and migrate current profiler call sites.
3. Add `profiler.lua` and suite-loading model.
4. Add dedicated profiler executable and six-seed orchestration.
5. Add suite manifest and aggregate output.
6. Update profiler analysis/reporting for suite-based outputs.
7. Add overhead validation path.
8. Add tests.
9. Update `justfile` wrappers and docs.

Success Criteria
- Normal `moonai` runtime behavior remains unchanged.
- `justfile` stays a thin wrapper only.
- Profiler policy lives in `profiler.lua` and the profiler executable.
- Six-seed profiling is deterministic and reproducible from `profiler.lua`.
- Fastest and slowest runs are dropped, and the remaining four runs drive the
  aggregate output.
- Normal builds carry no profiler instrumentation overhead.
- Profiler overhead in the profiler build stays low enough to keep timings clear.
- Profiler reports become easier to interpret and compare.
