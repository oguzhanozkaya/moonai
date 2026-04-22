---
hide:
  - navigation
  - toc
---

# MoonAI

A modular and extensible simulation platform for studying continuous evolutionary algorithms and neural network evolution through predator-prey dynamics.

**CMPE 491/492 - Senior Design Project | TED University**

**Team:**

- Caner Aras
- Emir Irkılata
- Oğuzhan Özkaya

**Supervisor:**

- Ayşenur Birtürk

**Jury Members:**

- Deniz Canturk
- Mehmet Evren Coskun

## Overview

MoonAI uses a predator-prey environment as a synthetic benchmark to evaluate evolutionary computation methods. Agents (predators and prey) are controlled by neural networks whose structure and weights evolve continuously through births and deaths using the **NeuroEvolution of Augmenting Topologies (NEAT)** algorithm.

The platform enables researchers to:

- Observe how neural network topologies emerge and grow in complexity through evolution
- Compare different genetic representations, mutation strategies, and selection methods
- Generate structured datasets for machine learning research without real-world data
- Visualize agent behavior and algorithm evolution in real time

### Motivation

Modern artificial intelligence training often requires vast amounts of real-world data and manually designed scenarios, which do not scale efficiently. MoonAI addresses this limitation by providing autonomous, self-generating training environments for studying evolutionary computation without external data dependencies.

### Objective

Develop a robust simulation environment to research and optimize evolutionary algorithms. By decoupling training from real-world data dependencies, we investigate how genetic representations influence learning efficiency and adaptability in dynamic, complex environments.

### Approach
The system employs a high-fidelity predator-prey simulation to generate evolutionary and genetic data. This synthetic ecosystem serves as a dynamic benchmark for evaluating evolutionary computation techniques in adaptive behavior modeling.

## Features

- **Entity-Component-System Architecture** - Data-oriented design with sparse-set ECS, cache-friendly SoA memory layouts, and 5-10x performance improvement
- **NEAT Implementation** - Evolves both topology and weights of neural networks simultaneously
- **Real-Time Visualization** - SFML-based rendering with interactive controls and live NN activation display
- **GPU Acceleration** - CUDA backend for sensing, neural inference, and simulation systems in both visual and headless modes
- **Cross-Platform** - Runs on Linux and Windows with matched features and stable runtime behavior
- **Reproducible Experiments** - Seeded RNG with deterministic behavior within the CUDA execution path on a fixed runtime environment
- **Lua Configuration** - Define named experiments and parameter sweeps in `config.lua` without recompilation
- **Data Export** - CSV/JSON output (including optional per-step trajectories) compatible with Python analysis tools

### Real-Time Analytics

Researchers observe emergent behaviors through an SFML-based real-time visualization layer. The system concurrently logs extensive telemetry, including population metrics and genome histories, exporting structured data for rigorous offline analysis using Python-based tools.
