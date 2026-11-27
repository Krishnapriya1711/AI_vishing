"""
Vishing keyword/pattern detector.

analyse_transcript(text) -> dict with keys:
    triggered   : bool  — True if vishing patterns found
    risk_level  : str   — "HIGH" | "MEDIUM" | "LOW" | "NONE"
    risk_score  : float — 0.0–1.0
    matched     : list[str] — which categories matched
    highlights  : list[str] — the matched phrases (for display)
    summary     : str   — human-readable explanation
"""

import re

# ── Keyword taxonomy ──────────────────────────────────────────────────────────
# Each category carries a weight; multiple hits compound.

CATEGORIES = {
    "credential_demand": {
        "weight": 0.40,
        "patterns": [
            r"\b(confirm|provide|give|share|read|tell)\b.{0,30}\b(otp|one[- ]time[- ]pass(?:word|code)?|pin|password|passcode)\b",
            r"\b(otp|one[- ]time[- ]pass(?:word|code)?)\b",
            r"\bpin\s*(code|number)?\b",
            r"\b(cvv|cvc|security\s+code)\b",
            r"\b(card\s+number|credit\s+card|debit\s+card)\s*(number|details|information)\b",
            r"\b(account\s+(number|details|password)|net\s*banking\s*(id|password))\b",
            r"\b(aadhaar|aadhar|ssn|social\s+security)\s*(number|card)?\b",
        ],
    },
    "urgency_pressure": {
        "weight": 0.20,
        "patterns": [
            r"\b(immediately|urgent(?:ly)?|right\s+now|at\s+once|right\s+away)\b",
            r"\b(last\s+chance|final\s+(warning|notice)|time\s+is\s+running\s+out)\b",
            r"\b(do\s+not\s+(hang\s+up|disconnect|call\s+anyone|talk\s+to\s+anyone))\b",
            r"\b(within\s+\d+\s+(minutes?|hours?|seconds?))\b",
            r"\b(account\s+will\s+be\s+(blocked|frozen|suspended|closed))\b",
            r"\b(obey|comply|follow\s+my\s+(instructions?|authority))\b",
        ],
    },
    "impersonation": {
        "weight": 0.20,
        "patterns": [
            r"\b(i\s+am|this\s+is)\s+(the\s+)?(senior\s+)?(bank|rbi|irs|irs\s+agent|police|cbi|cybercrime|fraud\s+(prevention|department)|security\s+(chief|department|officer|agent))\b",
            r"\bcalling\s+from\s+(your\s+)?(bank|reserve\s+bank|rbi|irs|government)\b",
            r"\b(official|authorized)\s+(representative|agent|officer)\b",
            r"\bfederal\s+(agent|bureau|investigation)\b",
        ],
    },
    "transfer_demand": {
        "weight": 0.35,
        "patterns": [
            r"\b(transfer|send|move|wire)\s+.{0,20}\b(fund|money|amount|balance|rupees?|dollars?|inr|usd)\b",
            r"\b(transfer|send)\s+(your\s+)?(entire|all|full)\s+(balance|fund|money|amount)\b",
            r"\bsecured?\s+account\b",
            r"\b(gift\s+card|google\s+pay|paytm|upi\s+id)\b.{0,30}\b(send|transfer|pay)\b",
        ],
    },
    "threat_scare": {
        "weight": 0.20,
        "patterns": [
            r"\b(hacked?|compromised?|breached?|attacked?|stolen?)\b",
            r"\b(arrest(?:ed)?|legal\s+action|lawsuit|warrant)\b",
            r"\b(drained?|emptied?|stolen?)\b.{0,20}\b(account|fund|money)\b",
            r"\b(in\s+danger|at\s+risk)\b",
            r"\bmainframe\s+(attack|breach|hack)\b",
        ],
    },
    "refund_scam": {
        "weight": 0.25,
        "patterns": [
            r"\b(refund|overpaid?|overcharged?)\b.{0,30}\b(process|claim|receive|get\s+back)\b",
            r"\b(tech(?:nical)?\s+support|microsoft|apple|amazon)\b.{0,20}\b(refund|charge)\b",
        ],
    },
}

# Phrases that are strong single-phrase triggers regardless of category
HIGH_RISK_PHRASES = [
    r"\bdo\s+not\s+(tell|inform|contact|call)\b.{0,20}\b(anyone|family|police|bank)\b",
    r"\b(verify|confirm)\s+your\s+(identity|account)\b.{0,10}\b(now|immediately)\b",
    r"\bpress\s+\d+\s+to\s+(speak|talk|connect)\b.{0,20}\b(agent|officer|representative)\b",
]


def analyse_transcript(text: str) -> dict:
    if not text or not text.strip():
        return dict(triggered=False, risk_level="NONE", risk_score=0.0,
                    matched=[], highlights=[], summary="No transcript available.")

    t = text.lower()
    total_weight = 0.0
    matched_cats = []
    highlights = []

    for cat, cfg in CATEGORIES.items():
        cat_hits = []
        for pat in cfg["patterns"]:
            for m in re.finditer(pat, t, re.IGNORECASE):
                cat_hits.append(m.group(0).strip())
        if cat_hits:
            total_weight += cfg["weight"]
            matched_cats.append(cat.replace("_", " ").title())
            highlights.extend(cat_hits[:2])  # max 2 examples per category

    # High-risk phrase check — each adds extra weight
    for pat in HIGH_RISK_PHRASES:
        for m in re.finditer(pat, t, re.IGNORECASE):
            total_weight += 0.15
            highlights.append(m.group(0).strip())

    # Clamp and assign level
    risk_score = min(total_weight, 1.0)
    if risk_score >= 0.55:
        risk_level = "HIGH"
    elif risk_score >= 0.30:
        risk_level = "MEDIUM"
    elif risk_score > 0.0:
        risk_level = "LOW"
    else:
        risk_level = "NONE"

    triggered = risk_level in ("HIGH", "MEDIUM")

    # Build summary
    if risk_level == "NONE":
        summary = "No vishing patterns detected in the transcript."
    else:
        cats_str = ", ".join(matched_cats) if matched_cats else "unknown"
        summary = f"Vishing patterns found ({cats_str})."

    return dict(
        triggered=triggered,
        risk_level=risk_level,
        risk_score=risk_score,
        matched=matched_cats,
        highlights=list(dict.fromkeys(highlights)),  # deduplicate
        summary=summary,
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("SPOOF lot20", "Urgent, your bank's mainframe has been attacked by foreign hackers and your personal account is compromised. I am the bank security chief. To secure your funds from being stolen, transfer your entire balance to the secured account."),
        ("SPOOF lot11", "Attention, your bank account has been flagged for a fraudulent transfer attempt of $5,000. I am the senior agent of the bank security department. You must immediately confirm your full credit card number and the CVV code."),
        ("BONAFIDE lot11", "This call is to confirm that your replacement credit card was successfully activated. You can now begin using your card. For your security, please destroy your old card immediately."),
        ("BONAFIDE lot16", "This message is from your bank, Wells Fargo. We noticed your checking account balance is currently low. Please deposit funds or call us to adjust your account settings."),
    ]
    for label, text in tests:
        r = analyse_transcript(text)
        print(f"{label}: {r['risk_level']} (score={r['risk_score']:.2f}) — {r['summary']}")
        if r['highlights']:
            print(f"  Highlights: {r['highlights'][:3]}")
