# MoonAI

A modular and extensible simulation platform for studying continuous evolutionary algorithms and neural network evolution through predator-prey dynamics.

**CMPE 491/492 - Senior Design Project | TED University**

**Website:** https://moon-aii.github.io/moonai/

**Team**: Caner Aras, Emir Irkılata, Oğuzhan Özkaya

**Supervisor**: Ayşenur Birtürk

**Jury Members**: Deniz Canturk, Mehmet Evren Coskun

## Overview

MoonAI uses a predator-prey environment as a synthetic benchmark to evaluate evolutionary computation methods. Agents (predators and prey) are controlled by neural networks whose structure and weights evolve continuously through births and deaths using the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm.

The platform enables researchers to:

- Observe how neural network topologies emerge and grow in complexity through evolution
- Compare different genetic representations, mutation strategies, and selection methods
- Generate structured datasets for machine learning research without real-world data
- Visualize agent behavior and algorithm evolution in real time

## Key Features

- **Entity-Component-System Architecture** - Data-oriented design with sparse-set ECS, cache-friendly SoA memory layouts, and 5-10x performance improvement
- **NEAT Implementation** - Evolves both topology and weights of neural networks simultaneously
- **Real-Time Visualization** - SFML-based rendering with interactive controls and live NN activation display
- **GPU Acceleration** - CUDA backend for sensing, neural inference, and simulation systems in both visual and headless modes
- **Cross-Platform** - Runs on Linux and Windows with matched features and stable runtime behavior
- **Reproducible Experiments** - Seeded RNG with deterministic behavior within the CUDA execution path on a fixed runtime environment
- **Lua Configuration** - Define named experiments and parameter sweeps in `config.lua` without recompilation
- **Data Export** - CSV/JSON output (including optional per-step trajectories) compatible with Python analysis tools
