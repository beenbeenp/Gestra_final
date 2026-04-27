# SESSION LOG

## 2026-04-22
### Goal
Find game input hook and confirm move() control path.

### What I tried
- inspected main.py
- inspected fighter.py
- identified move() as key integration point

### What worked
- game loop clearly routes through fighter.move()

### What failed
- not yet replaced keyboard with model output

### Next step
Create a temporary external action state for Player 1

## 2026-04-22
### What failed
- Street-Pyter did not install in the current environment because pygame==2.5.2 failed to build under Python 3.13.

### Decision
- Switch to a Python 3.12 virtual environment first before further repo analysis.

# SESSION LOG

## 2026-04-22

### Project direction update
- Switched base game repo from AadityaPanda / Street_Fighter to GeeseGoo / Street-Pyter
- Reason: Street-Pyter is much closer to an actual Street Fighter-style game and better matches the final project goal

### Environment setup
- Created a Python 3.12 virtual environment
- Installed dependencies successfully
- Confirmed that Street-Pyter launches

### Observed runtime status
- Startup message: "No controller found, using keyboard for both players"
- Game window appears successfully
- Player 2 seems controllable
- Player 1 does not respond as expected

### Current interpretation
- The problem is no longer environment setup
- The next task is to inspect the Player 1 input path inside the repo
- Possible causes:
  - Player 1 keyboard bug
  - controller-first design assumption
  - mirrored-input logic issue
  - unexpected key handling path

### Next step
- inspect the repo to find:
  - Player 1 input file(s)
  - Player 1 input function(s)
  - smallest safe debug step

  ## 2026-04-22

### Environment + base game update
- Switched the base game repo to GeeseGoo / Street-Pyter
- Created a Python 3.12 virtual environment because pygame==2.5.2 failed under Python 3.13
- Installed dependencies successfully
- Confirmed that Street-Pyter launches successfully

### Runtime status
- The game window opens correctly
- Both Player 1 and Player 2 keyboard controls work
- Current base game is usable for integration work

### Interpretation
- The environment/setup problem is resolved
- The project can now move from setup/debugging into integration
- The next step is to create the smallest safe external action hook for Player 1

### Next step
- Refactor Player 1 so it can optionally receive an external action state while preserving keyboard input as fallback

## 2026-04-22

### External action hook
- Added an optional external action state to `Character`.
- Routed `get_input()` through the external action state when one is present, with keyboard/controller behavior preserved as fallback.
- Added an opt-in dummy Player 1 action source behind `STREET_PYTER_DUMMY_P1=1`.

### What worked
- Player 1 can still use keyboard controls when no external action state is set.
- Player 2 input code was left unchanged.
- The dummy source can provide the same nested input shape used by the existing movement and attack code.
- `.venv/bin/python -m py_compile main.py lib.py settings.py` passed.
- A focused `.venv` check confirmed `Character.get_input()` reads an external state and returns to fallback when cleared.

### What failed or still needs testing
- The system `python3` interpreter does not have `pygame` installed, so runtime checks need the repo `.venv`.
- The game should still be smoke-tested interactively with normal keyboard input.
- The dummy source should be smoke-tested in a game window with `STREET_PYTER_DUMMY_P1=1`.

## 2026-04-22

### Cleanup and named action adapter
- Treated top-level `/Users/been/Desktop/gestra/docs` as the canonical docs folder.
- Merged the useful notes from the accidental nested `game_base/Street-Pyter/docs` session log into this file.
- Removed the accidental nested docs folder after confirming it was redundant.
- Removed the temporary Player 1 keyboard debug print from `lib.py`.
- Added a minimal named action adapter for `idle`, `forward`, `backward`, `punch`, and `kick`.
- Updated the dummy Player 1 source to emit named actions and convert them at the existing external action seam.

### What worked
- Keyboard fallback remains unchanged when no external action state is set.
- Player 2 input code was not changed.
- The dummy source still drives Player 1 through `STREET_PYTER_DUMMY_P1=1`, now via named actions.
- `.venv/bin/python -m py_compile main.py lib.py settings.py` passed.
- A focused `.venv` check confirmed all named actions map into the expected internal action-state shape.

### What failed or still needs testing
- Interactive game-window smoke testing is still needed for normal keyboard mode and `STREET_PYTER_DUMMY_P1=1`.

### Next smallest step
- Add a tiny local smoke-test harness or debug mode that confirms Player 1 switches between keyboard fallback and named dummy actions without manual visual inspection.

## 2026-04-22

### Player 1 keyboard fallback diagnosis
- Inspected the current Player 1 external-action seam in `main.py` and `lib.py`.
- Added temporary Player 1-only debug logging behind `STREET_PYTER_DEBUG_P1_INPUT=1`.
- The debug logs report when Player 1 input is read from keyboard or external state, when edge input is filtered by `input_buffer`, and what `attack()` and `move()` see downstream.

### What worked
- The external-action seam does not overwrite keyboard fallback unless `STREET_PYTER_DUMMY_P1=1` creates a Player 1 action source.
- Player 2 input code was left unchanged.
- `.venv/bin/python -m py_compile main.py lib.py settings.py` passed.
- A focused `.venv` check confirmed the debug path still reads the external action state correctly.

### Likely cause
- Player 1 keyboard input is probably being read, then some one-shot `held=False` checks are being filtered downstream because `Ryu.attack()` calls `get_input()` before `move()` runs.
- Since `get_input()` appends to the shared `input_buffer`, the later `move()` edge read can compare against the same current input and return `DEFAULT_INPUTS`.
- This means the issue is most likely "being read but ignored downstream" for edge-style checks, not "not being read" and not "overwritten by external action" in normal fallback mode.

### What failed or still needs testing
- Needs a live run with `STREET_PYTER_DEBUG_P1_INPUT=1` to confirm whether `move held` sees WASD and whether `keyboard_input filtered` appears for the same keypress.
- Interactive confirmation is still needed for normal keyboard mode and dummy named-action mode.

### Next smallest step
- Run the game with Player 1 debug logging enabled, press WASD/UIOJKL, and use the logs to apply the smallest fix to Player 1 input buffering/order.

## 2026-04-22

### Motion pose-capture scaffold
- Added a separate top-level `motion/` area for future data collection work.
- Added `motion/pose_capture.py`, a standalone webcam scaffold that opens camera input, runs MediaPipe Pose Landmarker, displays a live landmark overlay, and can save a short JSON sample of 33-pose-landmark frames.
- Added `motion/requirements.txt` for motion-only dependencies.
- Downloaded the MediaPipe `pose_landmarker_lite.task` model into `motion/models/`.
- Did not connect webcam, pose, or landmark output to Street-Pyter.
- Did not modify the game-side logic for this motion step.

### What worked
- Installed OpenCV and MediaPipe into the existing Python 3.12 `.venv`.
- `PYTHONPYCACHEPREFIX=/tmp/gestra_pycache .venv/bin/python -m py_compile /Users/been/Desktop/gestra/motion/pose_capture.py` passed.
- `.venv/bin/python /Users/been/Desktop/gestra/motion/pose_capture.py --help` passed.
- A short webcam smoke run opened camera capture, initialized MediaPipe/TFLite, displayed the preview path, and wrote `/tmp/gestra_pose_sample.json`.

### What failed
- The first implementation used the old `mp.solutions.pose` API, but the installed MediaPipe package exposes the newer Tasks API only. The scaffold was updated to use `PoseLandmarker`.
- The automated 5-second smoke run saved 0 pose frames because no full body was detected in the camera view.

### What still needs testing
- Stand head-to-toe in frame and run a short save test to confirm nonzero landmark frames are written.
- Confirm the overlay is visually usable at the intended camera distance and lighting.

### Next smallest step
- Run a one-person pilot capture, not a full dataset collection: record a few short landmark samples for each action class and verify that full-body landmarks are stable enough for labeling.

## 2026-04-26

### Full pipeline implementation

#### What changed
- Created Python 3.12 venv with all dependencies (pygame, mediapipe, opencv, torch, scikit-learn).
- Added `motion/named_action.py`: maps 5 named actions (idle/forward/backward/punch/kick) to Street-Pyter's nested input list shape.
- Added `motion/stub_action.py`: deterministic cycling action provider for smoke testing without webcam/ML.
- Modified `game_base/Street-Pyter/lib.py`: added `action_provider` parameter to `Character.__init__`, added `external_action_input()` method, and branched `get_input()` to use it when an action provider is set.
- Modified `game_base/Street-Pyter/main.py`: added `GESTRA_WEBCAM=1` and `GESTRA_STUB_ACTION=1` env-var-gated paths to inject external action providers into Player 1. Player 2 keyboard path unchanged.
- Added `ml/download_data.py`: downloads HMDB51 subset (punch, kick, kick_ball, stand) from HuggingFace mirror via HTTP range requests (~50 MB instead of 2.1 GB).
- Added `ml/extract_poses.py`: runs MediaPipe Pose Landmarker on each video clip, saves (T, 33, 3) landmark sequences as .npz files.
- Added `ml/dataset.py`: PyTorch Dataset with hip-center + shoulder-width normalization, sliding 30-frame windows, 3-class mapping (idle/punch/kick).
- Added `ml/model.py`: ActionLSTM — 1-layer LSTM, hidden=64, ~42k params.
- Added `ml/train.py`: training loop with weighted cross-entropy, saves best-val checkpoint.
- Added `ml/evaluate.py`: confusion matrix and classification report.
- Added `motion/pose_predictor.py`: real-time webcam thread — MediaPipe → rolling buffer → LSTM → hip-velocity rule for forward/backward → 5-frame majority smoothing → thread-safe `latest_action()`.

#### What worked
- Downloaded 536 HMDB51 clips (126 punch, 130 kick, 126 kick_ball, 154 stand).
- Extracted 452 valid pose sequences (reject rate ~16%, mostly clips with poor body visibility).
- Trained to 77.7% val accuracy on 3-class problem (idle/punch/kick). Punch recall 86%, kick recall 65%.
- All three game modes pass headless smoke tests: keyboard-only, stub provider, webcam provider.
- Game integration seam is minimal: only `get_input()` has a new branch; all existing keyboard/controller code untouched.

#### What failed or needs attention
- Kick recall (65%) is lower than punch (86%) — HMDB51 kick clips are more varied (martial arts, soccer, etc.).
- pygame + opencv-python both bundle libSDL2, producing ObjC duplicate-class warnings on macOS. Harmless but noisy.
- Forward/backward detection relies on hip-velocity heuristic (threshold 0.004/frame). May need tuning for different camera distances.

#### What still needs testing
- Live webcam end-to-end: stand in front of camera, launch `GESTRA_WEBCAM=1 .venv/bin/python main.py`, verify punch/kick/walk trigger correct in-game actions.
- Latency: target is <500ms from motion to in-game response.
- Jitter: idle stance should not produce spurious punches/kicks.
