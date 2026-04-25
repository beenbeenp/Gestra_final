# DATA COLLECTION SPEC
Project: Motion-Controlled Street Fighter (Pose + LSTM)
Game Engine: Python / Pygame
Model: LSTM
Actions: forward, backward, idle, punch, kick

## 1. Goal
Collect full-body webcam videos that can be converted into pose-landmark sequences and labeled frame-by-frame for 5 action classes:
- idle
- forward
- backward
- punch
- kick

The first dataset is for a playable prototype, not for perfect generalization.

---

## 2. Camera setup (fixed)
- Full body must be visible at all times
- Camera height: around chest / upper waist level
- Distance: about 2.5m–3.5m
- Orientation: front-facing
- Background: as simple as possible
- Lighting: avoid strong backlight
- Resolution target: 720p or 1080p
- FPS target: 30fps

Important:
- Do not change camera position during one recording session
- Use the same framing for all classes in the same session

---

## 3. Actors
Minimum:
- 2 people

Preferred:
- 3 people

Reason:
- One-person-only data will likely overfit to that person’s movement style.

---

## 4. Action classes
We will collect the following 5 classes:

1. idle
2. forward
3. backward
4. punch
5. kick

Do not add new classes until the first 5-class pipeline works.

---

## 5. Recording strategy
We will use two data types:

### A. Single-action clips
Short clips focused on one action repeated many times.

Examples:
- idle_01.mp4
- punch_01.mp4
- kick_01.mp4
- forward_01.mp4
- backward_01.mp4

### B. Short mixed-sequence clips (later)
Only after the first classifier works.

Examples:
- idle -> punch -> idle
- idle -> forward -> idle
- idle -> kick -> backward

For now, prioritize type A.

---

## 6. Minimum target amount
Per actor, per action:
- 25 to 40 repetitions

If 3 actors:
- about 75 to 120 repetitions per class total

This is enough for a first prototype.

---

## 7. Session metadata to record
For every video clip, store:

- actor_id
- session_id
- date
- camera_distance
- lighting_condition
- background_type
- class_name
- repetition_count
- notes

---

## 8. Folder structure
data/
  raw_videos/
    actor_01/
      session_01/
        idle/
        forward/
        backward/
        punch/
        kick/
    actor_02/
      session_01/
        ...
  labels/
  processed/
  metadata/

---

## 9. Naming convention
Use this format:

actorXX_sessionYY_classZZ_repNN.mp4

Examples:
- actor01_session01_punch_rep01.mp4
- actor02_session02_forward_rep14.mp4

---

## 10. What counts as “good enough” for collection
A clip is acceptable if:
- full body is visible
- motion is clearly performed
- camera is stable
- the intended class is obvious
- the clip can be labeled without confusion

Reject clips if:
- body is partly outside the frame
- motion is too weak or ambiguous
- another person appears in frame
- lighting is too poor for stable pose extraction

---

## 11. Extraction plan
After collection:
1. run pose extraction on every frame
2. save pose landmarks per frame
3. save frame-level labels
4. build sliding windows for LSTM training