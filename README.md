# Gestra — Motion-Controlled Fighting Game

Control a Street Fighter character against AI using your webcam. No controller, no keyboard — just sit in front of your computer and throw punches.

## Quick Start

```bash
cd ~/Desktop/"Gesture game"/Gestra_final

# First run: create environment
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Launch the game
cd game_base/Street-Pyter
GESTRA_WEBCAM=1 ../../.venv/bin/python main.py
```

## Controls

| Action | How to do it |
|--------|--------------|
| Left punch | Quickly extend your left hand forward |
| Right punch | Quickly extend your right hand forward |
| Move forward | Lean your body to the right |
| Move backward | Lean your body to the left |
| Block | Stay still (auto-block, 70% damage reduction) |

Face the screen and make sure the camera can see your upper body (head to waist). No need to stand up or have a large space.

## Game Rules

- Player 1 (you) vs Player 2 (AI)
- Both start with 75 HP; left punch deals 6 damage, right punch deals 14
- Auto-block when not punching (only take 30% damage)
- Punching drops your guard — attack/defense rhythm is the core strategy
- AI randomly punches or moves every 1-3 seconds
- Getting hit causes slight knockback, but you won't fly off screen
- HP reaches zero = game over, press any key to restart

## Startup Flow

1. Camera calibration window appears → make sure your upper body is visible
2. Stay still for 2 seconds → automatically enters the game
3. Press space to start the fight
4. ESC = return to menu / quit

## Record Your Own Data (Improve Accuracy)

The default model is trained on one person's data. Recording your own data significantly improves recognition accuracy:

```bash
# Record (~2 minutes, follow on-screen prompts)
.venv/bin/python -m ml.record_data

# Train (~1 minute)
.venv/bin/python -m ml.train_personal

# Launch the game again — it will automatically use your model
cd game_base/Street-Pyter
GESTRA_WEBCAM=1 ../../.venv/bin/python main.py
```

## Multi-Person Recording (Better Generalization)

Have different people each record their own data, then train together:

```bash
# Each person records with their own name
.venv/bin/python -m ml.record_data --person alice
.venv/bin/python -m ml.record_data --person bob
.venv/bin/python -m ml.record_data --person charlie

# Combined training (automatically scans all people's data)
.venv/bin/python -m ml.train_personal
```

The more people record, the better the model generalizes to new users. 3-5 people is enough to cover most body types.

## Project Structure

```
Gestra_final/
├── game_base/Street-Pyter/    # Pygame fighting game (forked from GeeseGoo/Street-Pyter)
│   ├── main.py                # Entry point, startup modes and game loop
│   ├── lib.py                 # Character class, attacks, blocking, movement logic
│   └── settings.py            # Balance tuning (HP, damage, knockback)
├── motion/                    # Motion detection
│   ├── upper_body_detector.py # Rule-based detector (punches + lean movement)
│   ├── personal_detector.py   # ML detector (uses personal model)
│   ├── calibration.py         # Camera calibration
│   ├── ai_opponent.py         # AI opponent
│   ├── named_action.py        # Action name → game input mapping
│   └── models/                # Model files
├── ml/                        # Training
│   ├── record_data.py         # Record personal data
│   ├── train_personal.py      # Train personal model
│   ├── debug_detection.py     # Debug tool (view real-time detection values)
│   └── ...                    # HMDB51 related (historical, can be ignored)
└── data/                      # Data (gitignored)
    └── personal/              # Personal recording data
```

## Detection Principles

### Rule-Based Mode (Fallback when no ML model)

- Punch: wrist-to-shoulder distance > 1.2x shoulder width + wrist velocity > threshold
- Movement: shoulder center horizontal offset from calibration baseline > threshold
- Block: none of the above triggered = idle

### ML Mode (With personal model)

- Input: 20-frame sliding window x 33 joints x (coordinates + velocity + acceleration) = (20, 297)
- Model: TCN (Temporal Convolutional Network), 2 layers, 64 channels, ~176k parameters
- Normalization: hip center zeroed + shoulder width normalized (eliminates body size differences)
- Output: 5 classes (idle / lpunch / rpunch / forward / backward)

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GESTRA_WEBCAM=1` | Enable webcam control |
| `GESTRA_RULES_ONLY=1` | Force rule-based detection (even if personal model exists) |
| `GESTRA_STUB_ACTION=1` | Use fake actions for testing (no webcam needed) |

## Dependencies

- Python 3.12
- pygame, opencv-contrib-python, mediapipe, torch, numpy

```bash
.venv/bin/pip install -r requirements.txt
```
