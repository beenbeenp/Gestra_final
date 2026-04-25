# PROJECT PLAN

## Goal
Build a real-time motion-controlled fighting game system using:
- Python / Pygame base game
- MediaPipe Pose for full-body landmarks
- LSTM for sequence classification
- Direct game-code integration (not keyboard emulation)

## Current base game
We are using:
- GeeseGoo / Street-Pyter

Why this repo:
- It is much closer to an actual Street Fighter-style fighting game
- It already supports two-player combat, multiple punches/kicks, and special moves
- It runs from a simple Python/Pygame entry point (`main.py`)

## Fixed decisions
- Base game: Street-Pyter
- Model: LSTM
- Camera: full body
- Integration method: direct game-code modification
- Motion-control action set for version 1:
  - idle
  - forward
  - backward
  - punch
  - kick

## Important scope note
The base game supports more attack types than our motion model.
Version 1 of the motion-control system will collapse the game into a smaller action vocabulary:
- idle
- forward
- backward
- punch
- kick

We will not try to map every built-in attack at first.

## Current project phase
Phase 1:
- confirm the base game runs reliably
- identify the exact Player 1 input path
- determine the smallest safe hook for external action control

## Non-goals for version 1
- no webcam integration yet
- no pose extraction yet
- no LSTM inference yet
- no full single-player AI yet
- no special move recognition yet
- no 2-human motion control yet

## Workflow rule for all future tasks
For every coding or debugging task:
- automatically update docs/SESSION_LOG.md
- automatically update docs/NEXT_TASK.md
- SESSION_LOG must record:
  - what changed
  - what worked
  - what failed
  - what still needs testing
  - the next smallest step
- NEXT_TASK must contain only the next smallest task
- unless explicitly told otherwise, always treat the top-level docs/ folder as the single source of truth


## Workflow rule for all future tasks
For every coding or debugging task:
- automatically update docs/SESSION_LOG.md
- automatically update docs/NEXT_TASK.md
- SESSION_LOG must record:
  - what changed
  - what worked
  - what failed
  - what still needs testing
  - the next smallest step
- NEXT_TASK must contain only the next smallest task
- always treat the top-level docs/ folder as the single source of truth