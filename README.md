# Rhythm Robot — Call and Response (prototype)

## Setup
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

Then pick ONE brain for the phrases and analysis (it's chosen automatically):

**Free — local model with Ollama**
1. Install Ollama from https://ollama.com and open it
2. `ollama pull llama3.1:8b`   (~5 GB download, needs ~8 GB RAM)
3. Run the game; it detects Ollama automatically

**Paid — Claude (best quality, ~3 US cents per session)**
    export ANTHROPIC_API_KEY=sk-ant-...                    # Windows: set ANTHROPIC_API_KEY=...

**Neither** — built-in phrases + rule-based analysis (no personalisation)
Linux also needs: `sudo apt install espeak-ng libportaudio2`

## Run
    python game.py                        # mic + speakers (use headphones so the beat isn't recorded)
    python game.py --keyboard --mute      # type answers, no audio: test the logic
    python game.py --no-beat              # no metronome
    python game.py --no-camera            # gestures shown but not checked

## Session flow
1. Pick mode → name + age → 2s mic calibration
2. Screening: Claude writes 6 phrases probing sounds, blends, length, questions, feelings
3. Claude designs 3 levels from the screening results
4. 5 phrases per level with adaptive difficulty (2 correct in a row → longer phrase, then faster beat; 2 misses → easier + slower)
5. Scoreboard + short analysis printed in the terminal

## Gestures
For each phrase the AI decides whether a gesture fits (e.g. "Hello friend" -> wave) or not.
When there is one, the robot does it while speaking and the child copies words + move.
The webcam window shows the child (mirrored) next to the robot. Stand back so your arms are
in frame. Press Q in the window to stop. Gestures: wave, arms up, hug yourself, hand on heart,
hands on head, clap, arms out wide. The pose model (~6 MB) downloads on first run.

## Using Pepper's camera (default)
Pose detection runs on the laptop; Pepper just streams its head camera over Wi-Fi.
Laptop and Pepper must be on the same network.

1. Find Pepper's IP (press the button on its chest once — it says it out loud).
2. Put that IP in `config.py` -> `PEPPER_IP`.
3. From PowerShell, copy the streaming script to Pepper and start it:

       scp pepper_camera_server.py nao@<PEPPER_IP>:/home/nao/
       ssh nao@<PEPPER_IP>
       python pepper_camera_server.py --pitch 0.1

   Leave that window open. `--pitch` holds the head still, tilted slightly down
   (adjust so the child's whole upper body and raised arms are in view).
4. In a second PowerShell window: `python game.py`

No Pepper nearby? `python game.py --camera webcam` uses the laptop webcam instead.

## Files
| file         | what it does                                            |
|--------------|---------------------------------------------------------|
| game.py      | menu, screening, levels, scenarios 1-3, scoreboard      |
| robot.py     | Robot interface, LocalRobot (laptop), PepperRobot stub  |
| speech.py    | Whisper speech-to-text + response-time detection        |
| matching.py  | phonetic + semantic scoring                             |
| brain.py     | Claude prompts (phrases, levels, analysis) + offline fallback |
| coach.py     | feedback lines + adaptive difficulty                    |
| actions.py   | gesture list, robot demo poses, gesture detectors       |
| vision.py    | camera window + MediaPipe pose tracking (Pepper or webcam) |
| pepper_camera_server.py | runs ON Pepper: streams its camera to the laptop |
| config.py    | all thresholds, bpm limits, points                      |