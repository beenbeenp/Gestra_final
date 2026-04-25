# LABELING RULES
Classes:
- idle
- forward
- backward
- punch
- kick

## General principle
We label each frame with exactly one class.

If a frame is ambiguous and the action is not clearly visible yet,
prefer labeling it as `idle` rather than inventing a transition class.

We are NOT using a separate `transition` class in version 1.

---

## 1. idle
Definition:
- neutral fighting stance
- no clear attack or locomotion is being executed
- small body sway is still idle

Label as idle if:
- the actor is waiting
- the actor is returning to stance
- movement is too weak to count as forward/backward
- the arm or leg is not clearly entering an attack motion yet

Do NOT label as idle if:
- a clear punch has started
- a clear kick has started
- body center is clearly shifting into forward/backward movement

---

## 2. forward
Definition:
- actor is intentionally stepping or shifting body mass forward

Start frame:
- the first frame where the actor clearly initiates forward movement

End frame:
- the frame where forward stepping/momentum is essentially completed
- after this, if the actor stabilizes, return to idle

Examples:
- stepping toward the camera-right direction (if that is defined as forward for the dataset)
- leaning plus stepping in the forward direction

Important:
- define forward/backward consistently BEFORE labeling
- do not switch definitions across actors

---

## 3. backward
Definition:
- actor is intentionally stepping or shifting body mass backward

Start frame:
- the first frame where the actor clearly initiates backward movement

End frame:
- the frame where backward stepping/momentum is essentially completed
- after this, return to idle if stable

Important:
- backward must be the opposite of forward in the dataset convention

---

## 4. punch
Definition:
- an arm/hand attack motion directed outward from the body

Start frame:
- when the punching arm clearly begins the strike motion

Middle:
- extension / impact-like phase

End frame:
- when the arm is mostly retracting or the punch is clearly over

Label entire active strike interval as punch.

Do not label tiny arm twitches as punch.

---

## 5. kick
Definition:
- a leg attack motion with clear lifting / extension of one leg

Start frame:
- when the kicking leg clearly begins to rise or initiate the strike

Middle:
- extension / impact-like phase

End frame:
- when the leg is mostly returning and the kick motion is over

Do not label small balance corrections as kick.

---

## 6. Ambiguous cases
If uncertain between:
- idle vs forward/backward -> use idle unless body shift is clear
- idle vs punch -> use idle unless strike motion clearly starts
- idle vs kick -> use idle unless leg attack clearly starts

This rule is intentional:
Version 1 values clean labels over aggressive labeling.

---

## 7. Labeling consistency rule
For one recording session:
- use the same rule set
- do not relabel one actor more loosely than another
- if a rule changes, document it in session notes

---

## 8. Quality control
After labeling each clip:
- watch the clip once with labels
- check if class boundaries are too noisy
- if labels flip excessively frame-to-frame, smooth the boundaries manually

Goal:
labels should represent the true action interval, not random frame noise