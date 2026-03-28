# MoonAI - Project Commands
# Usage: just <recipe>
# Run `just --list` to see all available recipes.

# Default recipe: build debug
default: run-release

# ─── Configuration ──────────────────────────────────────────────────────────

# Detect OS
os := if os() == "windows" { "windows" } else { "linux" }

# vcpkg root (falls back to ~/.vcpkg if not set in environment)
export VCPKG_ROOT := env("VCPKG_ROOT", env("HOME") + "/.vcpkg")

# Build type (override with: just build-type=release build)
build-type := "debug"

preset := os + "-" + build-type
build-dir := "build" / preset

# ─── Setup ──────────────────────────────────────────────────────────────────

# Set up Python environments for simulation and profiler analysis
[group('setup')]
setup-python:
    cd analysis && uv sync
    cd profiler && uv sync

# ─── Build ──────────────────────────────────────────────────────────────────

# Configure CMake (run after setup or when CMakeLists change)
[group('build')]
configure:
    cmake --preset {{preset}}

# Build the project
[group('build')]
build:
    cmake --build {{build-dir}} --parallel

# Build in release mode
[group('build')]
release:
    just build-type=release configure
    just build-type=release build

# ─── Run ────────────────────────────────────────────────────────────────────

# Run the simulation with default config (pass additional args after --)
[group('run')]
run *args: build
    {{build-dir}}/moonai config.lua --experiment default {{args}}

# Run the release build with default config (pass additional args after --)
[group('run')]
run-release *args: release
    ./build/linux-release/moonai config.lua --experiment default {{args}}

# Validate a config file
[group('run')]
validate config_path="config.lua": build
    {{build-dir}}/moonai --validate {{config_path}}

# ─── Test ───────────────────────────────────────────────────────────────────

# Run all tests
[group('test')]
test: build
    ctest --test-dir {{build-dir}} --output-on-failure

# Run tests with verbose output
[group('test')]
test-verbose: build
    ctest --test-dir {{build-dir}} --output-on-failure --verbose

# Run only CUDA/GPU tests
[group('test')]
test-gpu: build
    ctest --test-dir {{build-dir}} --output-on-failure -R GpuTest

# Run a specific test by name pattern
[group('test')]
test-filter pattern: build
    ctest --test-dir {{build-dir}} --output-on-failure -R {{pattern}}

# ─── Experiments ─────────────────────────────────────────────────────────────

# List all experiments defined in the Lua config
[group('experiment')]
list-experiments: build
    {{build-dir}}/moonai config.lua --list

# Run the full experiment matrix (all conditions × seeds, headless)
[group('experiment')]
experiments: release
    ./build/linux-release/moonai config.lua --all --headless

# Run one named experiment (headless)
[group('experiment')]
run-experiment name: release
    ./build/linux-release/moonai config.lua --experiment {{name}} --headless

# ─── Analysis ───────────────────────────────────────────────────────────────

# Generate the self-contained HTML analysis report from output/
[group('analysis')]
analyse:
    cd analysis && uv run moonai-analysis

# Full experiment pipeline: run all experiments → generate report
[group('experiment')]
experiment-pipeline: experiments analyse

# ─── Clean ──────────────────────────────────────────────────────────────────


# Remove build directory
[group('clean')]
clean:
    rm -rf build/

# Remove build, output, and all generated report artifacts
[group('clean')]
clean-all: clean
    rm -rf output/
    rm -rf analysis/output/
    rm -rf profiler/output/

# ─── Code Quality ───────────────────────────────────────────────────────────

# Run code quality checks: auto-format all C++ files and run static analysis
[group('lint')]
lint: configure
    find src \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) | xargs clang-format --style=file -i
    cppcheck --enable=warning,style,performance \
        --std=c++17 \
        --suppress=missingIncludeSystem \
        --suppress=*:*/vcpkg_installed/* \
        --project={{build-dir}}/compile_commands.json \
        2>&1 | head -100 || true

# ─── Profile ────────────────────────────────────────────────────────────────

# Full profiler pipeline: run profiler -> generate profiler report
[group('profile')]
profile: profile-run profile-analyse

# Run the built-in profiler with optional arguments
[group('profile')]
profile-run *args: release
    ./build/linux-release/moonai_profiler {{args}}

# Generate the self-contained HTML profiler report from output/profiles/
[group('profile')]
profile-analyse:
    cd profiler && uv run moonai-profiler

# ─── Development ────────────────────────────────────────────────────────────

# Generate compile_commands.json for IDE/LSP integration
[group('dev')]
compdb:
    cmake --preset {{preset}} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    ln -sf {{build-dir}}/compile_commands.json compile_commands.json

# Run Nsight Compute on the hottest GPU kernel with CLI output only (requires sudo for GPU perf counters)
[group('gpu')]
ncu: release
    sudo ncu \
        --target-processes all \
        --kernel-name "regex:.*sensor_build_kernel.*" \
        --launch-skip 0 \
        --launch-count 1 \
        --set basic \
        ./build/linux-release/moonai config.lua --experiment baseline_seed42 --headless --steps 60 --name nsight-baseline
    @echo "Note: Output files may be owned by root. Run: sudo chown -R $(whoami):$(whoami) output/nsight-baseline*"

# Run a deeper Nsight Compute pass on the hottest GPU kernel (requires sudo for GPU perf counters)
[group('gpu')]
ncu-full: release
    sudo ncu \
        --target-processes all \
        --kernel-name "regex:.*sensor_build_kernel.*" \
        --launch-skip 0 \
        --launch-count 1 \
        --set full \
        ./build/linux-release/moonai config.lua --experiment baseline_seed42 --headless --steps 60 --name nsight-baseline
    @echo "Note: Output files may be owned by root. Run: sudo chown -R $(whoami):$(whoami) output/nsight-baseline*"

# Run Nsight Systems for one profiler suite and print CLI stats (requires sudo for GPU perf counters)
[group('gpu')]
nsys: release
    mkdir -p output/nsight
    sudo nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --stats=true \
        --force-overwrite=true \
        --output=output/nsight/nsys-baseline \
        ./build/linux-release/moonai config.lua --experiment baseline_seed42 --headless --steps 60 --name nsight-baseline
    @echo "Note: Output files may be owned by root. Run: sudo chown -R $(whoami):$(whoami) output/nsight*"

# ─── Info ───────────────────────────────────────────────────────────────────

# Show project info and detected configuration
[group('info')]
info:
    @echo "MoonAI Project"
    @echo "─────────────────────────────"
    @echo "OS:         {{os}}"
    @echo "Build type: {{build-type}}"
    @echo "Preset:     {{preset}}"
    @echo "Build dir:  {{build-dir}}"
    @echo "VCPKG_ROOT: ${VCPKG_ROOT:-NOT SET}"
