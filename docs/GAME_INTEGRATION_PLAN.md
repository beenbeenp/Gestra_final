# GAME INTEGRATION PLAN

## Current repo
Base repo:
- GeeseGoo / Street-Pyter

## What we know so far
According to the repo documentation:
- `main.py` is the entry point
- the game is a two-player fighting game
- Player 1 keyboard controls are:
  - movement: WASD
  - attacks: UIOJKL
- Player 2 keyboard controls are:
  - movement: Arrow Keys
  - attacks: 123456

The repo also states that:
- the `Game` logic handles input processing and event management
- `lib.py` contains utilities and some input / move-related helpers
- `settings.py` contains configuration and control mappings

## Observed current status
- The game launches successfully in a Python 3.12 virtual environment
- The message "No controller found, using keyboard for both players" appears at startup
- Player 2 appears controllable
- Player 1 does not respond as expected
- `main.py` includes a note that the game is designed for keyboard + controller and that without a controller the two characters may become mirrors of each other

## Immediate goal
Do NOT integrate webcam or ML yet.

First determine:
1. where Player 1 keyboard input is supposed to be read
2. where Player 1 input becomes movement / attack behavior
3. whether Player 1 not responding is a bug, a keyboard mapping issue, or a controller-first design issue
4. the smallest safe place to later inject external action control

## Desired future integration
Eventually we want:
- pose landmarks -> LSTM -> predicted action
- predicted action -> Player 1 game action

Target motion actions:
- idle
- forward
- backward
- punch
- kick

## Version 1 integration principle
Do not replace the whole combat system.
Do not rewrite the game.
Only create the smallest possible hook so Player 1 can later receive an external action state.