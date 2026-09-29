"""
Rhythm Robot — Call and Response prototype.

    python game.py              # mic + speakers
    python game.py --keyboard   # type the child's answers (test without a mic)
    python game.py --mute       # print robot lines instead of speaking them
    python game.py --no-beat    # no metronome
    python game.py --no-camera  # robot still does gestures, but they aren't checked
    python game.py --camera webcam           # laptop webcam instead of Pepper's camera
    python game.py --pepper-ip 192.168.1.42  # Pepper at this address
    python game.py --speaker laptop          # laptop speakers instead of Pepper's voice
"""
from __future__ import annotations

import argparse
import random
import threading

import config
from brain import Brain
from coach import Difficulty, hint, line
from matching import compare
from robot import LocalRobot, PepperSpeakerRobot
from actions import ACTIONS
from speech import Heard
from vision import ActionResult   # light import: cv2/mediapipe only load if the camera is used

ICON = {"correct": "✅", "partial": "🟡", "incorrect": "🔁", "no_response": "⏸️ ", "skipped": "⏭️ "}
SKIP_WORDS = {"skip", "next", "no", "nope", "stop", "pass"}


class Session:
    def __init__(self, robot, keyboard: bool, camera: str | None):
        self.robot = robot
        self.keyboard = keyboard
        self.transcriber = None
        self.vision = None
        if not keyboard:
            from speech import Transcriber
            self.transcriber = Transcriber()
        if camera:
            try:
                from vision import Vision
                self.vision = Vision(camera, config.PEPPER_IP, config.PEPPER_CAMERA_PORT)
            except Exception as e:
                print(f"⚠️  Camera unavailable: {e}\n   Gestures will be shown but not checked. "
                      "(Use --camera webcam to test with the laptop webcam.)")

    # ------------------------------------------------------------------ output
    def robot_turn(self, text: str, bpm: int, action: str | None):
        """Robot says the phrase on the downbeat, doing the gesture at the same time."""
        r = self.robot
        r.wait_for_downbeat()
        r.do_action(action)
        if self.vision:                       # animate the on-screen robot while it talks
            t = threading.Thread(target=r.say, args=(text, bpm))
            t.start()
            self.vision.robot_turn(action, t, min_seconds=1.5 if action else 0.5)
            t.join()
        else:
            r.say(text, bpm)
        r.do_action(None)

    # ------------------------------------------------------------------ input
    def listen(self, seconds: float, action: str | None = None,
               prompt: str = "Your turn!") -> tuple[Heard, ActionResult | None]:
        act = None
        if self.vision:
            if not self.keyboard:
                self.robot.start_recording(seconds)
            act = self.vision.watch(action, seconds, prompt)
            if self.keyboard:
                heard = Heard(input("   ⌨️  type what the child said (Enter = silence): ").strip(), None)
            else:
                heard = self.transcriber.transcribe(self.robot.finish_recording())
            return heard, act

        if self.keyboard:
            heard = Heard(input("   ⌨️  type what the child said (Enter = silence): ").strip(), None)
            if action:
                did = input(f"   ⌨️  did the child {ACTIONS[action]['cue']}? [y/N]: ").strip().lower()
                act = ActionResult(action, did.startswith("y"), None, True)
            return heard, act
        return self.transcriber.transcribe(self.robot.record(seconds)), None   # gesture not checked

    def calibrate(self):
        if self.keyboard:
            return
        print("🔇 Calibrating mic — please stay quiet for 2 seconds...")
        self.transcriber.calibrate(self.robot.record(2.0))

    @staticmethod
    def window_for(text: str, bpm: int, action: str | None) -> float:
        beats = len(text.split()) + config.EXTRA_RESPONSE_BEATS + (2 if action else 0)
        return max(config.MIN_RESPONSE_SECONDS, beats * 60 / bpm)

    # ------------------------------------------------------------ one phrase
    def run_phrase(self, text: str, bpm: int, action: str | None = None, screening: bool = False) -> dict:
        r = self.robot
        r.start_beat(bpm)
        r.set_bpm(bpm)
        attempts, gave_second_chance_after_silence = [], False
        final = "incorrect"
        if action:
            print(f"   (phrase has a gesture: {action.replace('_', ' ')})")

        while len(attempts) < config.MAX_ATTEMPTS:
            self.robot_turn(text, bpm, action)
            r.cue_turn()
            heard, act = self.listen(self.window_for(text, bpm, action), action)
            res = compare(text, heard.transcript, heard.words)
            verdict = combine(res.verdict, act)
            attempts.append((res, heard, act, verdict))

            if config.SHOW_WORD_CONFIDENCE and heard.words:
                print("   confidence: " + "  ".join(f"{w}:{p:.2f}" for w, p in heard.words))
            move = ""
            if act:
                move = f" | move: {'✔' if act.done else '✘'} {act.action.replace('_', ' ')}"
                if act.done and act.time_s is not None:
                    move += f" ({act.time_s:.1f}s)"
            print(f"   heard: \"{heard.transcript}\"  (speech {res.verdict}, phonetic {res.phonetic:.2f}){move}"
                  f"  →  {ICON[verdict]} {verdict}")

            # Scenario 1: correct (words and, if there was one, the move)
            if verdict == "correct":
                final = "correct"
                r.express("happy")
                r.say("Thank you!" if screening else line("correct" if len(attempts) == 1 else "correct_retry"))
                break

            # Scenario 3: no response -> pause beat, calm prompt, try again or skip
            if verdict == "no_response":
                if act and not act.person_seen:
                    print("   (camera couldn't see the child)")
                r.stop_beat()
                r.express("calm")
                r.say(line("no_response"))
                r.say(line("no_response_ask"))
                choice = self.listen(5.0, None, "Try again or skip?")[0].transcript.lower()
                wants_skip = not choice or bool(SKIP_WORDS & set(choice.replace(",", " ").split()))
                if wants_skip or gave_second_chance_after_silence:
                    r.say(line("skip"))
                    final = "skipped"
                    break
                gave_second_chance_after_silence = True
                attempts.pop()            # a silent turn doesn't use up a try
                r.start_beat(bpm)
                continue

            # Scenario 2: partial / incorrect -> gentle retry with one hint
            final = verdict
            if screening:
                r.say("Thank you!")
                break
            if len(attempts) < config.MAX_ATTEMPTS:
                r.express("encourage")
                r.say(retry_message(verdict, res, act))
            else:
                r.say(line("move_on"))

        return self._record(text, bpm, action, attempts, final)

    @staticmethod
    def _record(text, bpm, action, attempts, final):
        rank = {"correct": 3, "partial": 2, "incorrect": 1, "no_response": 0}
        if attempts:
            best_res, best_heard, best_act, _ = max(attempts, key=lambda a: (rank[a[3]], a[0].phonetic))
            first_onset = attempts[0][1].onset_s
        else:
            best_res, best_heard, best_act, first_onset = None, Heard("", None), None, None
        verdict = "no_response" if final == "skipped" else final
        if verdict == "correct":
            pts = config.POINTS["correct_first" if len(attempts) == 1 else "correct_retry"]
        else:
            pts = config.POINTS[verdict]
        acts = [a[2] for a in attempts if a[2]]
        return {
            "phrase": text, "action": action, "bpm": bpm, "tries": len(attempts), "final_verdict": verdict,
            "skipped": final == "skipped", "heard": best_heard.transcript,
            "phonetic": best_res.phonetic if best_res else 0.0,
            "semantic": best_res.semantic if best_res else 0.0,
            "missing": best_res.missing if best_res else text.lower().split(),
            "mispronounced": [list(p) for p in best_res.mispronounced] if best_res else [],
            "extra": best_res.extra if best_res else [],
            "onset_s": round(first_onset, 2) if first_onset is not None else None,
            # gesture: None = no gesture / not checked
            "action_done": (any(a.done for a in acts) if acts else None) if action else None,
            "action_time_s": next((a.time_s for a in acts if a.done), None),
            "points": pts,
        }


def combine(speech_verdict: str, act: ActionResult | None) -> str:
    """Merge the speech result with the gesture result (if a gesture was checked)."""
    if act is None:
        return speech_verdict
    if speech_verdict == "correct":
        return "correct" if act.done else "partial"
    if speech_verdict == "no_response":
        return "partial" if act.done else "no_response"
    return speech_verdict if not act.done else "partial"


def retry_message(verdict: str, res, act: ActionResult | None) -> str:
    """One gentle, specific pointer: whichever part (words or move) needs it."""
    if act and not act.done and res.verdict == "correct":
        return f"Great words! Let's try again, and don't forget to {ACTIONS[act.action]['cue']} too."
    if act and act.done and res.verdict == "no_response":
        return "Great move! Now let's say the words too."
    msg = line("partial" if verdict == "partial" else "incorrect")
    h = hint(res)
    return f"{msg} {h}" if h else msg


# ---------------------------------------------------------------- printing
def print_trial(t: dict, label: str):
    icon = ICON["skipped"] if t["skipped"] else ICON[t["final_verdict"]]
    onset = f"{t['onset_s']:.1f}s" if t["onset_s"] is not None else "  - "
    print(f"   {label:<6} {icon} {t['final_verdict']:<11} tries {t['tries']}  "
          f"phon {t['phonetic']:.2f}  sem {t['semantic']:.2f}  onset {onset}  "
          f"{t['bpm']}bpm  +{t['points']}pts")
    issues = []
    if t["missing"]:
        issues.append("missing: " + ", ".join(t["missing"]))
    if t["mispronounced"]:
        issues.append("sounded different: " + ", ".join(f"{a}→{b}" for a, b in t["mispronounced"]))
    if t.get("action"):
        if t["action_done"] is None:
            issues.append(f"move: {t['action'].replace('_', ' ')} (not checked)")
        else:
            took = f" in {t['action_time_s']:.1f}s" if t.get("action_time_s") is not None else ""
            issues.append(f"move: {t['action'].replace('_', ' ')} {'✔' + took if t['action_done'] else '✘ missed'}")
    if issues:
        print("          " + " | ".join(issues))


def banner(text):
    print("\n" + "═" * 64 + f"\n  {text}\n" + "═" * 64)


def make_robot(args):
    """Pepper's voice + laptop mic if possible, otherwise everything on the laptop."""
    if args.speaker == "pepper" and not args.mute:
        try:
            return PepperSpeakerRobot(config.PEPPER_IP, config.PEPPER_COMMAND_PORT, beat=not args.no_beat)
        except Exception as e:
            print(f"⚠️  {e}\n   Using the laptop speakers instead.")
    return LocalRobot(speak=not args.mute, beat=not args.no_beat)


def choose_mode() -> None:
    banner("🥁  RHYTHM ROBOT")
    print("  1) Call and Response\n  2) Pitch / Stress Matching   (coming soon)\n"
          "  3) Breathing to the Beat     (coming soon)")
    while True:
        c = input("\nChoose a game mode [1]: ").strip() or "1"
        if c == "1":
            return
        print("That mode isn't built yet — pick 1 for now.")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyboard", action="store_true", help="type responses instead of using the mic")
    ap.add_argument("--mute", action="store_true", help="print robot speech instead of speaking")
    ap.add_argument("--no-beat", action="store_true", help="turn off the metronome")
    ap.add_argument("--no-camera", action="store_true", help="don't check gestures with a camera")
    ap.add_argument("--camera", choices=["pepper", "webcam"], default=config.CAMERA_SOURCE,
                    help="where the camera frames come from (default from config.py)")
    ap.add_argument("--pepper-ip", default=None, help="override PEPPER_IP from config.py")
    ap.add_argument("--speaker", choices=["pepper", "laptop"], default=config.SPEAKER,
                    help="who talks and plays the beat (default from config.py)")
    args = ap.parse_args()

    choose_mode()
    name = input("Child's first name: ").strip() or "friend"
    try:
        age = int(input("Child's age: ").strip() or "8")
    except ValueError:
        age = 8

    if args.pepper_ip:
        config.PEPPER_IP = args.pepper_ip
    robot = make_robot(args)
    session = Session(robot, keyboard=args.keyboard, camera=None if args.no_camera else args.camera)
    brain = Brain()
    session.calibrate()

    screening, trials = [], []
    levels = {"levels": []}
    try:
        # ---------------- screening ----------------
        banner("🔎  SCREENING — just listening, no scores yet")
        phrases = brain.screening_phrases(age)
        robot.do_action("wave")
        robot.say(f"Hi {name}! I'm your rhythm robot. First, let's warm up. "
                  "When I say something, you say it back after the chime!")
        for i, p in enumerate(phrases, 1):
            print(f"\n[screen {i}/{len(phrases)}]  probes: {p.get('probes', '')}")
            t = session.run_phrase(p["text"], bpm=70, action=p.get("action"), screening=True)
            t["probes"] = p.get("probes", "")
            screening.append(t)
        robot.stop_beat()

        # ---------------- plan levels ----------------
        banner("🧠  Designing levels for " + name)
        robot.say("Great warm up! Give me a second to make your levels.")
        levels = brain.design_levels(age, screening)
        print(f"\n{levels.get('child_summary', '')}\n")
        for i, lv in enumerate(levels["levels"], 1):
            n_moves = sum(1 for ph in lv["phrases"] if ph.get("action"))
            print(f"  Level {i}: {lv['name']} — {lv['focus']}  ({n_moves}/{len(lv['phrases'])} phrases with a gesture)")

        # ---------------- play levels ----------------
        for li, lv in enumerate(levels["levels"][:3], 1):
            banner(f"🎵  LEVEL {li}: {lv['name']}")
            robot.say(f"Level {li}! {lv['name']}.")
            diff = Difficulty(lv.get("start_bpm", 75), lv.get("start_tier", 0))
            pool = list(lv["phrases"])
            random.shuffle(pool)
            level_trials = []
            for n in range(1, config.PHRASES_PER_LEVEL + 1):
                if not pool:
                    break
                ph = min(pool, key=lambda p: abs(p.get("tier", 1) - diff.tier))
                pool.remove(ph)
                print(f"\n[L{li} #{n}]  tier {ph.get('tier')}  {diff.bpm}bpm")
                t = session.run_phrase(ph["text"], diff.bpm, action=ph.get("action"))
                t.update(level=li, tier=ph.get("tier"))
                trials.append(t)
                level_trials.append(t)
                print_trial(t, f"L{li}#{n}")

                change = diff.update(t["final_verdict"] == "correct")
                if change:
                    robot.say(line(change))
                    print(f"   ⚙️  difficulty {change}: now {diff.bpm}bpm, tier {diff.tier}")
            robot.stop_beat()
            pts = sum(t["points"] for t in level_trials)
            ok = sum(t["final_verdict"] == "correct" for t in level_trials)
            print(f"\n   Level {li} score: {pts} pts  ({ok}/{len(level_trials)} correct)")
            robot.say(f"You finished level {li}! Amazing!")

    except KeyboardInterrupt:
        print("\n\n⏹  Session stopped early.")
    finally:
        # cleanup must never stop the results from printing
        for step in (robot.stop_beat,
                     session.vision.close if session.vision else None):
            if step:
                try:
                    step()
                except Exception as e:
                    print(f"(cleanup warning: {e})")

    # ---------------- results ----------------
    banner(f"📊  RESULTS FOR {name.upper()}")
    if screening:
        print("Screening:")
        for i, t in enumerate(screening, 1):
            print_trial(t, f"S{i}")
    if trials:
        print("\nLevels:")
        for t in trials:
            print_trial(t, f"L{t['level']}")
        total = sum(t["points"] for t in trials)
        best = config.POINTS["correct_first"] * len(trials)
        print(f"\n  TOTAL: {total} / {best} pts")

    banner("📝  ANALYSIS")
    print(brain.analyse(age, screening, levels, trials))
    print()
    if trials:
        try:
            robot.do_action("wave")
            robot.say(f"You did so well today, {name}. See you next time!")
        except Exception as e:
            print(f"(couldn't say goodbye: {e})")
    if hasattr(robot, "close"):
        try:
            robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()