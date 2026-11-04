"""Single-shot AI analysis report for BTC.

Different pattern from agent.py on purpose: agent.py is a multi-turn
loop where the model picks which tool to call. This script runs once:
your code computes every number up front, hands them to the model in
one message, and asks for a structured JSON write-up. Meant to run on
a schedule (e.g. GitHub Actions) and write a report into reports/.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from fetch import PAIRS, fetch_ohlc
from metrics import (
    best_worst_day,
    max_drawdown,
    moving_average,
    price_extremes,
    trend_status,
    volatility,
    volume_trend_pct,
)
from rules import TREND_THRESHOLD_FIXED, trend_rule_fixed, volatility_rule

# ---------- Settings ----------
PAIR = "XBTUSD"  # the coin this report is about
WINDOW = 30  # candles, volatility, and drawdown all use this window
TREND_MA_WINDOW = 7  # kept independent of WINDOW on purpose: this is the same
# 7-day trend window used everywhere else in the project (agent.py, rules.py);
# coupling it to WINDOW=30 would quietly change what "above/below trend" means
# here versus everywhere else we report it.

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

# Words that sound like advice. If the AI uses them, we reject the answer.
ADVICE = re.compile(
    r"\b(buy|buying|sell(?!-?off)|selling|invest in|should invest|go long|go short|accumulate)\b",
    re.IGNORECASE,
)


# ---------- 1. Numbers (your code, not the AI) ----------
def get_facts():
    # Fetch all 5 coins, not just PAIR. Our volatility_rule labels a coin
    # "elevated" relative to the median volatility across the group; called with
    # just one coin, the median would equal that coin's own value and the label
    # would be meaningless (always "typical"). So every coin gets fetched, even
    # though this report only writes up one of them.
    all_volatilities = {}
    target = None
    for p in PAIRS:
        complete, current = fetch_ohlc(p)
        closes = [c.close for c in complete]
        all_volatilities[p] = volatility(closes, window=WINDOW)
        if p == PAIR:
            target = {"complete": complete, "current": current, "closes": closes}

    vol_labels = volatility_rule(all_volatilities)

    candles = target["complete"][-WINDOW:]
    current = target["current"]
    closes = target["closes"]

    ma = moving_average(closes, window=TREND_MA_WINDOW)
    direction, deviation = trend_status(current.close, ma)
    dd = max_drawdown(closes, window=WINDOW)
    high, low = price_extremes(candles)
    best_candle, best_pct, worst_candle, worst_pct = best_worst_day(candles)
    vol_change = volume_trend_pct(candles)

    day = lambda c: datetime.fromtimestamp(c.time, timezone.utc).strftime("%Y-%m-%d")

    facts = {
        "data_to": day(candles[-1]),
        "latest_close": round(current.close, 2),
        f"{TREND_MA_WINDOW}d_moving_average": round(ma, 2),
        "pct_vs_moving_average": round(deviation, 2),
        "trend_label": trend_rule_fixed(direction, deviation),
        f"{WINDOW}d_volatility_pct": round(all_volatilities[PAIR] * 100, 2),
        "volatility_label": vol_labels[PAIR],
        f"{WINDOW}d_max_drawdown_pct": round(dd * 100, 2),
        "highest_price": round(high, 2),
        "lowest_price": round(low, 2),
        "best_day": f"{day(best_candle)} ({round(best_pct, 2)}%)",
        "worst_day": f"{day(worst_candle)} ({round(worst_pct, 2)}%)",
        "volume_last_7_days_vs_earlier_pct": round(vol_change, 1),
        "rules_used": (
            f"volatility is labeled 'elevated' when it's above the median {WINDOW}-day "
            f"volatility across all 5 tracked coins that day (the median is recomputed "
            f"every run, not a fixed cutoff); trend is labeled 'notably' above/below when "
            f"the deviation from the {TREND_MA_WINDOW}-day moving average exceeds "
            f"{TREND_THRESHOLD_FIXED} percentage points"
        ),
    }
    table = "\n".join(f"{day(c)}: close {round(c.close, 2)}" for c in candles)
    return facts, table


# ---------- 2. Ask the AI to analyse ----------
def ask_groq(prompt):
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
        json={
            "model": MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Reply with JSON only. No extra text."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise SystemExit("Groq error: " + r.text)
    return r.json()["choices"][0]["message"]["content"].strip()


def get_analysis(facts, table):
    prompt = (
        f"You are a careful market data analyst. Here are numbers for Bitcoin ({PAIR}), "
        f"the last {WINDOW} finished daily candles.\n\n"
        f"Numbers calculated by code:\n{json.dumps(facts, indent=2)}\n\n"
        f"Daily closes:\n{table}\n\n"
        "Decide which kinds of analysis are useful here (for example trend, volatility, drawdown, "
        "volume, momentum) and use only those. Return JSON with exactly these keys:\n"
        '- analysis_types: a list of objects, each with "type" and "what_i_looked_at" '
        "(list only the analyses you really used)\n"
        "- observations: a list of 3 to 5 strings, each tied to specific numbers above\n"
        "- summary: 2 or 3 sentences\n"
        "- limitations: one sentence on what this analysis cannot tell us\n"
        "Rules: use only the numbers given, do not invent anything, describe what the data shows, "
        "and never tell anyone to buy, sell or invest."
    )
    for attempt in range(2):
        text = ask_groq(prompt)
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            a = json.loads(text)
            ok = (
                isinstance(a.get("analysis_types"), list) and a["analysis_types"]
                and all("type" in t and "what_i_looked_at" in t for t in a["analysis_types"])
                and isinstance(a.get("observations"), list) and a["observations"]
                and isinstance(a.get("summary"), str) and isinstance(a.get("limitations"), str)
            )
        except (json.JSONDecodeError, AttributeError, TypeError):
            ok = False
        if ok and ADVICE.search(json.dumps(a)):
            print("The AI used advice words, trying again")
            ok = False
        if ok:
            return a
        print("The answer was not valid, trying again")
    raise SystemExit("The AI did not return a valid analysis")


# ---------- 3. Write the report ----------
def build_report(facts, analysis):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Bitcoin 30-day AI analysis",
        "",
        f"**Data:** Kraken, {PAIR}, daily candles up to {facts['data_to']}",
        f"**Report created:** {now}",
        f"**AI model:** {MODEL}",
        "",
        "## Numbers (calculated by code)",
        "",
        "| Measure | Value |",
        "|---|---|",
    ]
    for k, v in facts.items():
        lines.append(f"| {k.replace('_', ' ')} | {v} |")
    lines += ["", "## Types of analysis the AI used", ""]
    for t in analysis["analysis_types"]:
        lines.append(f"- **{t['type']}**: {t['what_i_looked_at']}")
    lines += ["", "## Observations", ""]
    lines += [f"- {o}" for o in analysis["observations"]]
    lines += [
        "", "## Summary", "", analysis["summary"],
        "", "## Limitations", "", analysis["limitations"], "",
        "This report is written by an AI from past data. It is for education only and is not financial advice.",
    ]
    return "\n".join(lines)


def save_report(text):
    Path("reports").mkdir(exist_ok=True)
    name = "ai_report_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".md"
    path = Path("reports") / name
    path.write_text(text, encoding="utf-8")
    # On GitHub Actions, this also shows the report on the run's summary page
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(text)
    return path


# ---------- Run ----------
if __name__ == "__main__":
    facts, table = get_facts()
    print(facts)
    analysis = get_analysis(facts, table)
    print("\nTYPES OF ANALYSIS THE AI USED:")
    for t in analysis["analysis_types"]:
        print("-", t["type"] + ":", t["what_i_looked_at"])
    print("\nSUMMARY:", analysis["summary"])
    print("\nReport written to", save_report(build_report(facts, analysis)))
