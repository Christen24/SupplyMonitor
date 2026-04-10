---
title: Supply Chain Risk Management (SCRM) OpenEnv
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - supply-chain
---

# Supply Chain Risk Management (SCRM) - OpenEnv Environment

An RL environment simulating enterprise supply chain operations, built to the
[OpenEnv](https://github.com/openenv) specification.

**Hugging Face Space**: [Live Demo](https://huggingface.co/spaces/Christen24/supply-chain-risk-management)

## Overview & Motivation

**Motivation**: Modern supply chains are deeply interconnected and highly vulnerable to cascading failures whether from natural disasters, geopolitical sanctions, or vendor bankruptcy. This environment bridges the gap by providing a mathematically rigid, multi-echelon risk management simulation where LLM agents must read complex risk reports (unstructured data) and take mitigation actions to prevent cascading failures across the globe.

The environment models a mid-sized electronics manufacturer's supply chain with
**15+ suppliers** across four regions (Asia, Europe, North America, LATAM),
**6 product parts** with different cost profiles, and realistic risk events
(natural disasters, geopolitics, vendor failures, pandemics).

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run Dashboard (Interactive UI)

To launch the high-fidelity glassmorphism dashboard:

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 7860
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

### Run Baseline / Demo Script

```bash
# Set your Hugging Face Token for the Inference API
export HF_TOKEN="your_hf_token_here"

# Run the environment evaluation script
python inference.py
```

### Docker

```bash
docker build -t scrm-env .
docker run -p 7860:7860 scrm-env
```

## Interactive Dashboard & Auto-Pilot

The project features a premium, glassmorphism dashboard designed for real-time monitoring and autonomous simulation.

### Key Functions
*   **Live Supply Chain Visualization**: Dynamic supplier topology map showing health scores, reliability, and regional status for 20+ global nodes.
*   **Real-Time Risk Detection**: The "Disruption Intel" panel automatically flags active crises (Netural Disasters, Geopolitical Shifts, Vendor Failures) as they occur.
*   **Autonomous Auto-Pilot**: A built-in heuristic agent that can take control of the supply chain, automatically making decisions (Mitigate, Diversify, Recover) to maintain OTD.
*   **Stateful Persistence**: Powered by **WebSockets**, the dashboard maintains a persistent environment session, allowing for seamless multi-step interactions.
*   **Premium UX**: Featuring high-refresh metrics (OTD, Cost, Reward), micro-animations, and a console-style "Agent Logic Stream".

### Dashboard Metrics
| Metric | Description |
|--------|-------------|
| **Time Step** | Current frame in the episode (0-35-50 steps). |
| **On-Time Delivery (OTD)** | Trailing performance vs SLA target (>95%). |
| **Total Cost** | Operational friction localized in **INR (₹)**. |
| **Reward** | Cumulative RL trajectory score. |


### API Endpoints

| Method | Path      | Description              |
|--------|-----------|--------------------------|
| POST   | `/reset`  | Reset environment        |
| POST   | `/step`   | Take an action           |
| GET    | `/state`  | Get current state        |
| POST   | `/grade`  | Grade completed episode  |
| GET    | `/health` | Health check             |

## Tasks

| Task | Difficulty | Steps | Objective | Expected Score |
|------|-----------|-------|-----------|---------------|
| `easy` | Easy | 20 | Routine monitoring & minor disruptions | 0.70-0.85 |
| `medium` | Medium | 35 | Multi-region crisis management | 0.50-0.70 |
| `hard` | Hard | 50 | Global cascade failure recovery | 0.30-0.50 |

## Observation Space

The observation space is returned as a heavily structured Pydantic dictionary containing complete details about the manufacturer's internal state and external market conditions.

| Key | Type | Description |
|-----|------|-------------|
| `timestamp` | `int` | Current simulation frame |
| `inventory_levels` | `dict` | Parts mapped to raw quantity on hand |
| `supplier_status` | `dict` | Health, reliability, and region per supplier |
| `risk_events` | `list[dict]` | Descriptions of all active external crisis events |
| `cost_metrics` | `dict` | Baseline vs. variance costs used for penalization |
| `delivery_performance` | `dict` | Trailing On-Time Delivery (OTD) and Quality percentages |
| `market_conditions` | `dict` | Exogenous volatility multipliers |
