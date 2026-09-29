"""
The "NLI": uses Claude to (1) create screening phrases, (2) design 3 levels
from the screening results, (3) write a short end-of-session analysis.
Falls back to built-in content if there's no API key, so the rest can be tested offline.
"""
from __future__ import annotations

import json
import os
import re

import config

SYSTEM = """You are the planning brain of a friendly rhythm-game robot that helps neurodivergent \
children (roughly ages 5-12) practise speaking and conversational turn-taking through a \
Call and Response game. The robot says a phrase on the beat, the child repeats it.

Rules for every phrase you write:
- Warm, concrete, everyday language a child would actually say or hear.
- Lean on social and emotional themes: greetings, taking turns, naming feelings, asking for help, \
being kind, sharing, saying what you like.
- No idioms, sarcasm, or ambiguous meanings. No scary, shaming, or food/body-image content.
- Plain words only, no numerals or symbols (write "two" not "2").
You are NOT diagnosing anything. You describe observed patterns in this session only."""


class Brain:
    def __init__(self):
        self.client = None
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except ImportError:
                print("⚠️  `anthropic` not installed — using built-in phrases.")
        else:
            print("⚠️  No ANTHROPIC_API_KEY set — using built-in phrases (offline mode).")

    # ------------------------------------------------------------------
    def _ask(self, prompt: str, max_tokens: int = 2000) -> str:
        msg = self.client.messages.create(
            model=config.LLM_MODEL, max_tokens=max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    def _ask_json(self, prompt: str, max_tokens: int = 2000) -> dict:
        prompt += "\n\nRespond with ONLY valid JSON. No markdown fences, no commentary."
        for _ in range(2):
            raw = self._ask(prompt, max_tokens)
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            start, end = raw.find("{"), raw.rfind("}")
            try:
                return json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
        raise ValueError("LLM did not return valid JSON")

    # ------------------------------------------------------------------
    def screening_phrases(self, age: int) -> list[dict]:
        if self.client:
            try:
                data = self._ask_json(f"""Write {config.SCREENING_PHRASE_COUNT} screening phrases for a \
{age}-year-old. Together they should probe: tricky sounds (r, l, s, th, sh), consonant blends \
(e.g. "str", "bl"), a multi-syllable word, a short and a longer phrase (3 to 9 words), a question \
(turn-taking), and a feelings phrase. Order them easy to harder.
JSON shape: {{"phrases": [{{"text": "...", "probes": "what this phrase tests"}}]}}""")
                return data["phrases"][: config.SCREENING_PHRASE_COUNT]
            except Exception as e:
                print(f"⚠️  LLM error ({e}) — using built-in screening phrases.")
        return FALLBACK_SCREENING

    def design_levels(self, age: int, screening: list[dict]) -> dict:
        if self.client:
            try:
                return self._ask_json(f"""A {age}-year-old just did a screening round. Results \
(phonetic/semantic are 0-1; onset_s is seconds before they started speaking, null = no response):
{json.dumps(screening, indent=1)}

Design exactly 3 levels that target what THIS child needs, based on the evidence above. \
Level 1 should feel very achievable (confidence first), level 3 a gentle stretch. \
Each level has 8 phrases tagged with tier 0 (short, 2-4 words), 1 (medium, 4-6 words), \
or 2 (long, 6-9 words) — include at least two of each tier. Weave the child's trickier sounds \
into phrases, but not in every word. Suggest a starting bpm between {config.MIN_BPM} and \
{config.MAX_BPM} (slower if they were slow to start or missed words).

JSON shape: {{"child_summary": "1-2 sentences on what screening showed",
 "levels": [{{"name": "fun level name", "focus": "what it practises and why",
   "start_bpm": 72, "start_tier": 0,
   "phrases": [{{"text": "...", "tier": 0}}]}}]}}""", max_tokens=3000)
            except Exception as e:
                print(f"⚠️  LLM error ({e}) — using built-in levels.")
        return FALLBACK_LEVELS

    def analyse(self, age: int, screening: list[dict], levels: dict, trials: list[dict]) -> str:
        if self.client:
            try:
                return self._ask(f"""Write a short analysis (under 200 words) of this Call and \
Response session for a {age}-year-old, for their parent or teacher. Plain text, no markdown headers.
Cover: strengths (lead with these), what they found hardest (specific sounds, word positions, \
phrase length, speed, response timing / turn-taking), how they responded to the adaptive changes, \
and 2-3 concrete suggestions for next session. Base every claim on the data. Don't diagnose.

Screening: {json.dumps(screening)}
Plan: {json.dumps(levels.get("child_summary", ""))}
Trials: {json.dumps(trials)}""", max_tokens=800).strip()
            except Exception as e:
                print(f"⚠️  LLM error ({e}) — using basic analysis.")
        return basic_analysis(trials)


def basic_analysis(trials: list[dict]) -> str:
    """Rule-based fallback so offline runs still get a summary."""
    if not trials:
        return "No trials recorded."
    from collections import Counter
    n = len(trials)
    correct = sum(t["final_verdict"] == "correct" for t in trials)
    silent = sum(t["final_verdict"] == "no_response" for t in trials)
    missed = Counter(w for t in trials for w in t["missing"])
    mispr = Counter(tw for t in trials for tw, _ in t["mispronounced"])
    onsets = [t["onset_s"] for t in trials if t["onset_s"] is not None]
    lines = [f"Got {correct}/{n} phrases fully correct."]
    if mispr:
        lines.append("Sounds to practise, in words like: " + ", ".join(w for w, _ in mispr.most_common(5)) + ".")
    if missed:
        lines.append("Most often dropped words: " + ", ".join(w for w, _ in missed.most_common(5)) + ".")
    if onsets:
        lines.append(f"Average time to start speaking: {sum(onsets)/len(onsets):.1f}s.")
    if silent:
        lines.append(f"No response on {silent} phrase(s) — turn-taking cues may need to be clearer or slower.")
    return " ".join(lines)


FALLBACK_SCREENING = [
    {"text": "Hello friend", "probes": "greeting, l/r"},
    {"text": "I feel happy today", "probes": "feelings word, f/h"},
    {"text": "Can I have a turn please", "probes": "question, turn-taking, blend 'pl'"},
    {"text": "The red rabbit runs fast", "probes": "r sound, s blend"},
    {"text": "Thank you for sharing with me", "probes": "th, sh"},
    {"text": "My favourite strawberry is really sweet", "probes": "multi-syllable, 'str', 'sw', longer phrase"},
]

FALLBACK_LEVELS = {
    "child_summary": "Offline mode: default levels (not personalised).",
    "levels": [
        {"name": "Hello Beats", "focus": "Short greetings to build confidence", "start_bpm": 70, "start_tier": 0,
         "phrases": [{"text": t, "tier": k} for t, k in [
             ("Hi there", 0), ("Good morning", 0), ("Nice to meet you", 0), ("How are you today", 1),
             ("I am happy to see you", 1), ("Can we play together", 1),
             ("Hello my friend it is good to see you", 2), ("Good morning I hope you slept well", 2)]]},
        {"name": "Feelings Groove", "focus": "Naming feelings", "start_bpm": 76, "start_tier": 1,
         "phrases": [{"text": t, "tier": k} for t, k in [
             ("I feel calm", 0), ("I feel silly", 0), ("I feel a bit sad", 1), ("I feel proud of myself", 1),
             ("I feel excited about the trip", 1), ("When I am worried I take a breath", 2),
             ("I feel grumpy when I am really tired", 2), ("It is okay to feel scared sometimes", 2)]]},
        {"name": "Turn Taking Train", "focus": "Asking and sharing", "start_bpm": 82, "start_tier": 1,
         "phrases": [{"text": t, "tier": k} for t, k in [
             ("Your turn", 0), ("Can you help me", 0), ("Can I have a go please", 1), ("Would you like to share", 1),
             ("Thank you for waiting for me", 1), ("After you finish can I have a turn", 2),
             ("Please pass the blue block to me", 2), ("I like playing this game with you", 2)]]},
    ],
}
