# Gestra — Motion-Controlled Fighting Game

Control a Street Fighter character against AI using your webcam. No controller, no keyboard — just sit in front of your computer and raise your arms to punch.

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

## Startup Flow

1. Camera calibration window — make sure your upper body is visible, hold still 2 seconds
2. Quick-record prompt — press SPACE to record your moves (~30s) or ESC to skip
3. If recorded: model retrains on your data (~20s), then starts the game
4. Press SPACE on the menu to start the fight
5. ESC = return to menu / quit

Each time you record, your data accumulates and the model improves for your body.

## Controls

| Action | How to do it |
|--------|--------------|
| Left punch | Raise your left arm above shoulder height |
| Right punch | Raise your right arm above shoulder height |
| Move forward | Lean your body to the right |
| Move backward | Lean your body to the left |
| Block | Stay still (auto-block, 70% damage reduction) |

Face the screen. The camera only needs to see your upper body (head to shoulders).

## Game Rules

- Player 1 (you) vs Player 2 (AI)
- Both start with 75 HP; left punch deals 6 damage, right punch deals 14
- Auto-block when not punching (only take 30% damage)
- Punching drops your guard — attack/defense rhythm is the core strategy
- AI randomly punches or moves every 1-3 seconds
- HP reaches zero = game over, press any key to restart

## Project Structure

```
Gestra_final/
├── game_base/Street-Pyter/    # Pygame fighting game (forked from GeeseGoo/Street-Pyter)
│   ├── main.py                # Entry point: calibration → quick-record → game loop
│   ├── lib.py                 # Character class, attacks, blocking, movement
│   └── settings.py            # Balance tuning (HP, damage, knockback)
├── motion/                    # Motion detection
│   ├── calibration.py         # Camera calibration (upper body check)
│   ├── quick_record.py        # Pre-game 30s guided recording
│   ├── quick_train.py         # Fast retraining with weighted sampling
│   ├── upper_body_detector.py # Rule-based detector (arm raise + lean)
│   ├── personal_detector.py   # ML detector (TCN on personal model)
│   ├── ai_opponent.py         # Random AI for Player 2
│   ├── named_action.py        # Action name → game input mapping
│   └── models/                # Model files (.pt, .task)
├── ml/                        # Training pipeline
│   ├── record_data.py         # Full recording session (~2 min)
│   ├── train_personal.py      # Train personal model with experiment logging
│   ├── evaluate_offline.py    # Rule-based vs TCN offline comparison
│   └── model.py               # ActionTCN architecture
├── notebooks/
│   └── reproduce_figures.ipynb # Generates all report figures
├── results/
│   ├── experiment_log.csv     # Hyperparameter experiment results
│   └── figures/               # Generated figures (7 PNGs)
├── final_site/                # Local HTML blog for submission
│   ├── index.html
│   ├── styles.css
│   └── assets/
└── data/                      # Data (gitignored)
    └── personal/              # Personal recordings (.npz)
```

## Detection

### Rule-Based Mode (fallback when no ML model)

- Punch: either wrist rises to shoulder height + wrist velocity > threshold (whichever wrist is higher determines left/right)
- Movement: shoulder center horizontal offset from adaptive baseline > threshold (baseline tracks slowly via exponential moving average)
- Idle: none of the above, stabilized by 7-frame majority-vote smoothing
- Block: idle = auto-block (70% damage reduction)

### ML Mode (with personal model)

- Input: 20-frame sliding window × 9 upper-body joints × (pos + vel + acc) = (20, 81)
- Model: TCN (Temporal Convolutional Network), 2 layers, 64 channels, ~93K parameters
- Normalization: shoulder-center zeroed + shoulder-width scaled
- Output: 5 classes (idle / lpunch / rpunch / forward / backward)

## Full Recording (alternative to quick-record)

For higher accuracy, record a full 2-minute session:

```bash
.venv/bin/python -m ml.record_data
.venv/bin/python -m ml.train_personal
```

Multi-person recording for better generalization:

```bash
.venv/bin/python -m ml.record_data --person alice
.venv/bin/python -m ml.record_data --person bob
.venv/bin/python -m ml.train_personal
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GESTRA_WEBCAM=1` | Enable webcam control |
| `GESTRA_RULES_ONLY=1` | Force rule-based detection (skip ML model) |
| `GESTRA_STUB_ACTION=1` | Fake actions for testing (no webcam needed) |

## Dependencies

Python 3.12 required.

```bash
.venv/bin/pip install -r requirements.txt
```

## Reproducing Figures

```bash
.venv/bin/jupyter notebook notebooks/reproduce_figures.ipynb
# Run all cells — figures saved to results/figures/
```

## Experiment Log

Results in `results/experiment_log.csv`. Re-run all experiments:

```bash
.venv/bin/python -m ml.train_personal --run-id LR-A --lr 1e-3 --csv-log results/experiment_log.csv
.venv/bin/python -m ml.train_personal --run-id LR-B --lr 3e-4 --csv-log results/experiment_log.csv
.venv/bin/python -m ml.train_personal --run-id LR-C --lr 1e-4 --csv-log results/experiment_log.csv
.venv/bin/python -m ml.train_personal --run-id OPT-B --optimizer adamw --weight-decay 1e-4 --csv-log results/experiment_log.csv
.venv/bin/python -m ml.train_personal --run-id OPT-C --optimizer adamw --lr 3e-4 --weight-decay 1e-4 --csv-log results/experiment_log.csv
.venv/bin/python -m ml.train_personal --run-id BS-A --batch 8 --csv-log results/experiment_log.csv
.venv/bin/python -m ml.train_personal --run-id BS-C --batch 32 --csv-log results/experiment_log.csv
.venv/bin/python -m ml.train_personal --run-id WIN-A --window 10 --csv-log results/experiment_log.csv
.venv/bin/python -m ml.train_personal --run-id WIN-C --window 30 --csv-log results/experiment_log.csv
.venv/bin/python -m ml.evaluate_offline --csv-log results/experiment_log.csv
```

## Final Blog

Open `final_site/index.html` in a browser. No external dependencies.
