# Algo Brain: Carbon-Aware RL Environment

## Overview
Algo Brain simulates a data center where an agent schedules workloads under thermal, energy, and stochastic constraints.

## Features
- 3 Tasks: easy, medium, hard
- Stochastic transitions
- Safety constraints (CPU temperature)
- Policy Gradient learning agent

## Observation Space
- cpu_temp
- battery_maH
- accuracy_score

## Action Space
- 0 = Idle
- 1 = Run

## Reward Logic
- Task-based reward scaling
- Penalty for overheating
- Episode termination on unsafe conditions

## Learning Approach
We implemented a lightweight Policy Gradient (REINFORCE) algorithm to train the agent.

The agent learns:
- when to execute jobs (RUN)
- when to delay (IDLE)
- how to avoid overheating penalties

Training is done online before inference to ensure adaptability under stochastic transitions.

This environment demonstrates decision-making under uncertainty using reinforcement learning, rather than static heuristics.

## Run
python inference.py
