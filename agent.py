"""Session 6: AI agent on top of the Assignment 1 crypto workflow.

Wraps fetch.py / metrics.py / rules.py as tools an LLM can call. The
model only ever picks a tool name + arguments; this file is the only
thing that touches the network, does the math, or decides what's
allowed to run.
"""

import json
import os
import re
import statistics
import sys
from pathlib import Path

import requests

# Our Assignment 1 modules, not a single crypto_workflow.py file.
from fetch import PAIRS, fetch_ohlc
from metrics import max_drawdown, moving_average, trend_status, volatility
from rules import trend_rule_dynamic, trend_rule_fixed, volatility_rule

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

SYSTEM = (
    "You answer questions about crypto market data. "
    "Always use the tools for numbers and never guess them. "
    "Describe what the numbers show. Never tell anyone to buy, sell, hold or invest. "
    "Keep answers to a few sentences."
)

ADVICE = re.compile(r"\b(buy|sell|hold|invest in|should invest|go long|go short)\b", re.IGNORECASE)


# ---------- Tools (our Assignment 1 code, wrapped) ----------
# Our rule functions are relative across all 5 coins: volatility_rule needs the
# median volatility of all 5 to label any one of them "elevated" or "typical",
# and the dynamic trend label needs the median deviation of all 5 too. So a
# single-pair tool can't correctly compute just its own coin in isolation.
# Fix: build all 5 coins once per run (first tool call), cache it, and every
# tool reads from that cache. One real set of Kraken calls per run, not one
# per tool call.
_cache = None


def _build_cache():
    global _cache
    if _cache is not None:
        return _cache

    raw = {}
    for pair in PAIRS:
        complete, current = fetch_ohlc(pair)
        closes = [c.close for c in complete]
        ma = moving_average(closes)
        direction, deviation = trend_status(current.close, ma)
        raw[pair] = {
            "latest_close": current.close,
            "moving_average_7d": ma,
            "direction": direction,
            "deviation": deviation,  # signed %, unrounded; used for both display and the rules
            "volatility_30d": volatility(closes),
            "max_drawdown_30d": max_drawdown(closes),
        }

    vol_labels = volatility_rule({p: r["volatility_30d"] for p, r in raw.items()})
    median_abs_deviation = statistics.median(abs(r["deviation"]) for r in raw.values())

    cache = {}
    for pair, r in raw.items():
        cache[pair] = {
            "pair": pair,
            "latest_close": round(r["latest_close"], 4),
            "moving_average_7d": round(r["moving_average_7d"], 4),
            "pct_vs_ma": round(r["deviation"], 2),
            "volatility_30d_pct": round(r["volatility_30d"] * 100, 2),
            "volatility_label": vol_labels[pair],
            "max_drawdown_30d_pct": round(r["max_drawdown_30d"] * 100, 2),
            "trend_label_fixed": trend_rule_fixed(r["direction"], r["deviation"]),
            "trend_label_dynamic": trend_rule_dynamic(r["direction"], r["deviation"], median_abs_deviation),
        }

    _cache = cache
    return _cache


def get_metrics(pair):
    """One coin's metrics: a small dict, not raw candles."""
    cache = _build_cache()
    return cache.get(pair, {"error": f"{pair} not tracked"})


def compare_all_coins():
    """A small ranked summary across all 5 coins, for comparison questions."""
    cache = _build_cache()
    ranked = sorted(cache.values(), key=lambda r: r["volatility_30d_pct"], reverse=True)
    return {
        "most_volatile": ranked[0]["pair"],
        "least_volatile": ranked[-1]["pair"],
        "ranked_by_volatility": [
            {
                "pair": r["pair"],
                "volatility_30d_pct": r["volatility_30d_pct"],
                "volatility_label": r["volatility_label"],
            }
            for r in ranked
        ],
    }


def read_latest_report():
    files = sorted(Path("reports").glob("report_*.md"))
    if not files:
        return {"error": "no report found. Run main.py first."}
    return {"file": files[-1].name, "text": files[-1].read_text(encoding="utf-8")[:3000]}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_metrics",
            "description": (
                "Get 30-day metrics for one coin: latest close, 7-day moving average, "
                "% vs that average, 30-day volatility, 30-day max drawdown, and rule labels "
                "(volatility elevated/typical, trend below/above with a fixed and a dynamic reading)"
            ),
            "parameters": {
                "type": "object",
                "properties": {"pair": {"type": "string", "enum": PAIRS}},
                "required": ["pair"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_all_coins",
            "description": (
                "Compare all 5 coins by 30-day volatility, ranked highest to lowest, with the "
                "most and least volatile coin named directly"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_latest_report",
            "description": "Read the most recent report written by the workflow",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------- Talk to the model ----------
def ask_groq(messages, **extra):
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
        json={"model": MODEL, "messages": messages, **extra},
        timeout=60,
    )
    if r.status_code != 200:
        raise SystemExit("Groq error: " + r.text)
    return r.json()["choices"][0]["message"]


# ---------- The agent loop ----------
def run_tool(name, args):
    if name == "get_metrics" and args.get("pair") in PAIRS:
        return get_metrics(args["pair"])
    if name == "compare_all_coins":
        return compare_all_coins()
    if name == "read_latest_report":
        return read_latest_report()
    return {"error": "tool or input not allowed"}


def run_agent(question, max_steps=6):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        msg = ask_groq(messages, tools=TOOLS, temperature=0.2)
        calls = msg.get("tool_calls")
        if not calls:
            return check_answer(msg["content"])
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            print("STEP", step, "->", name, args)
            result = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return "Stopped: too many steps"


def check_answer(text):
    # Code guardrail: the prompt can fail, so we also check the answer
    if ADVICE.search(text):
        return (
            "I can only describe what the data shows, not give buy or sell advice. "
            "Ask me about volatility, trends or averages instead."
        )
    return text + "\n\nThis is for education only and is not financial advice."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python3 agent.py "your question"')
    print(run_agent(" ".join(sys.argv[1:])))
