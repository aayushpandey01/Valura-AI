"""
Safety Guard — synchronous, no LLM, no network. Must complete < 10ms.
Runs BEFORE the classifier. If it blocks, the pipeline stops here.
"""
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SafetyCategory(str, Enum):
    INSIDER_TRADING = "insider_trading"
    MARKET_MANIPULATION = "market_manipulation"
    MONEY_LAUNDERING = "money_laundering"
    GUARANTEED_RETURNS = "guaranteed_returns"
    RECKLESS_ADVICE = "reckless_advice"


@dataclass
class SafetyResult:
    blocked: bool
    category: Optional[SafetyCategory]
    response: Optional[str]
    latency_ms: float


# Educational signal phrases — if present, likely an educational query, not a harmful request
_EDUCATIONAL_SIGNALS = [
    r"\beducational(ly)?\b",
    r"\bacademic\b",
    r"\bexplain\b",
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bhow does\b",
    r"\bhistorical\b",
    r"\bhistory of\b",
    r"\bfamous case[s]?\b",
    r"\bregulat\w+\b",
    r"\bdetect\w*\b",
    r"\bsafeguard[s]?\b",
    r"\bprevents?\b",
    r"\bwhy is .+ illegal\b",
    r"\bconsequences of\b",
    r"\bfrom an? .+ perspective\b",
    r"\bas a concept\b",
    r"\bas a matter of\b",
    r"\benforce[s]?\b",
    r"\bwhat happened\b",
    r"\bshort squeeze\b",
]

_EDU_PATTERN = re.compile("|".join(_EDUCATIONAL_SIGNALS), re.IGNORECASE)

# --- Harmful patterns by category ---

_INSIDER_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\binsider (information|info|tip|tips|knowledge|data)\b",
        r"\bnon.?public information\b",
        r"\bconfidential .*(earnings|merger|deal|announcement|data|info)\b",
        r"\b(before|ahead of) .*(announcement|deal|merger|earnings|release)\b",
        r"\b(told|heard|learned) .*(deal|merger|acquisition|earnings)\b",
        r"\bgovernment .*(confidential|secret|classified|non.?public)\b",
        r"\btrade on .*(tip|info|insider|confidential|non.?public)\b",
        r"\bmerger before it.{0,20}(announced|public)\b",
        r"\b(know|knowing) .{0,30}(merger|acquisition|deal|earnings) .{0,20}(before|ahead)\b",
        r"\bhow do i profit.{0,20}(merger|deal|acquisition|announce)\b",
    ]
]

_MANIPULATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bpump.?and.?dump\b",
        r"\bpump\b.{0,20}\bdump\b",
        r"\bspread.*(false|fake|misleading).*(rumor|news|info)\b",
        r"\bfalse.*(rumor|news|info).*(move|boost|pump|manipulate)\b",
        r"\bfake.*(volume|news|rumor|press release)\b",
        r"\bcreate.*(fake|artificial|false).*(volume|demand|interest|price)\b",
        r"\bmisleading.*(press release|announcement|report)\b",
        r"\bcoordinate.*(buy|sell|trade|move).*(stock|price|market)\b",
        r"\bmove.*(stock|price|market).*(coordinate|collude|together)\b",
        r"\bartificial(ly)?.*(volume|price|demand|interest)\b",
        r"\bwash.?trad(e|ing)\b",
        r"\bspoofing\b.*\b(order|trade|market)\b",
        r"\bmanipulate.*(stock|price|market)\b",
    ]
]

_LAUNDERING_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\blaunder\b",
        r"\bhide.{0,20}(source|origin).{0,20}(fund|money|cash)\b",
        r"\b(source|origin) of (my )?(fund|money|cash).{0,30}(hide|conceal|obscure)\b",
        r"\b(avoid|evade).*(AML|anti.?money|reporting|detection|tax).*(illegally|law|rules)?\b",
        r"\bstructur(e|ing).*(transaction|payment|deposit).*(avoid|evade|under)\b",
        r"\bcash from illegal\b",
        r"\billegal.*(fund|money|cash|activit)\b",
        r"\boff.?shore.*(hide|avoid|evade|illegal)\b",
        r"\bmove money.*(undetected|without detection|avoid)\b",
        r"\bhide the source\b",
        r"\bconceal.*(fund|money|cash|source)\b",
    ]
]

_GUARANTEED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bguarantee[ds]?\b.*(return|profit|gain|income|yield|money)",
        r"\b(return|profit|gain).*(guarantee[ds]?|certain|sure|100%|zero.?risk|no.?risk)\b",
        r"\brisk.?free.*(return|investment|profit|gain|income)\b",
        r"\bpromise.*(profit|return|gain|money|win)\b",
        r"\b100%.*(guarantee|certain|sure).*(profit|return|win)\b",
        r"\b(always|never).*(lose|loss|risk)\b",
    ]
]

_RECKLESS_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(all|entire|whole|everything|100%).*(saving|retirement|fund|money|life.?saving).*(single|one|meme|crypto|bet|penny|option|gamble)\b",
        r"\b(second mortgage|home equity).*(invest|buy|crypto|stock|bet|trade)\b",
        r"\bmortgage.*(invest|buy|trade|crypto)\b",
        r"\bemergency.?fund.*(meme|crypto|bet|risky|penny|gamble)\b",
        r"\b(loan.?shark|illegal loan|borrowed money).*(invest|trade|stock)\b",
        r"\b(100%|all.?in).*(single|one).*(option|stock|crypto|bet|gamble).*(retirement|saving|fund)\b",
        r"\bborrow.*(invest|buy|crypto|stock|penny|meme)\b",
        r"\btake out a.*(loan|mortgage).*(invest|buy|stock|crypto)\b",
        r"\binvest my emergency\b",
    ]
]

_BLOCKED_RESPONSES = {
    SafetyCategory.INSIDER_TRADING: (
        "This request involves acting on material non-public information, which constitutes insider trading "
        "under securities law in most jurisdictions. I'm unable to assist with this. "
        "Insider trading is a serious criminal offence with significant legal consequences."
    ),
    SafetyCategory.MARKET_MANIPULATION: (
        "This request describes market manipulation — a serious violation of securities law that harms "
        "all market participants. I'm unable to provide assistance with activities such as pump-and-dump "
        "schemes, spreading false information, or creating artificial volume."
    ),
    SafetyCategory.MONEY_LAUNDERING: (
        "I'm unable to help with structuring transactions to evade AML reporting requirements, "
        "concealing the source of funds, or any other activity that may constitute money laundering. "
        "These are serious financial crimes subject to severe criminal penalties."
    ),
    SafetyCategory.GUARANTEED_RETURNS: (
        "No legitimate investment can guarantee returns or eliminate risk. Making such claims is often "
        "associated with fraudulent schemes. I'm unable to endorse or construct any guarantee of "
        "investment returns. All investing involves risk, including the possible loss of principal."
    ),
    SafetyCategory.RECKLESS_ADVICE: (
        "This appears to involve taking on dangerous levels of financial risk — such as investing "
        "emergency savings, borrowed funds, or retirement money in highly speculative assets. "
        "I'm unable to recommend strategies that could cause serious financial harm. "
        "Please consult a qualified financial adviser before proceeding."
    ),
}

_CATEGORY_CHECKS = [
    (SafetyCategory.INSIDER_TRADING, _INSIDER_PATTERNS),
    (SafetyCategory.MARKET_MANIPULATION, _MANIPULATION_PATTERNS),
    (SafetyCategory.MONEY_LAUNDERING, _LAUNDERING_PATTERNS),
    (SafetyCategory.GUARANTEED_RETURNS, _GUARANTEED_PATTERNS),
    (SafetyCategory.RECKLESS_ADVICE, _RECKLESS_PATTERNS),
]


def check_safety(query: str) -> SafetyResult:
    """
    Synchronous safety guard. No I/O, no LLM.
    Returns SafetyResult indicating whether query should be blocked.
    """
    start = time.monotonic()

    # First: check for educational signals — if present, lean toward passing
    is_educational = bool(_EDU_PATTERN.search(query))

    for category, patterns in _CATEGORY_CHECKS:
        for pattern in patterns:
            if pattern.search(query):
                # Educational signal overrides for most categories
                # EXCEPT money laundering and insider trading with very explicit intent
                if is_educational and category not in (
                    SafetyCategory.INSIDER_TRADING,
                    SafetyCategory.MONEY_LAUNDERING,
                ):
                    continue
                # For insider/laundering, only explicit action intent triggers block
                # (educational framing still passes for general questions)
                if is_educational:
                    # Check if pattern is about actual action vs learning
                    action_signals = re.search(
                        r"\b(should i|help me|how (can|do) i|i want to|i have|i know)\b",
                        query,
                        re.IGNORECASE,
                    )
                    if not action_signals:
                        continue

                latency = (time.monotonic() - start) * 1000
                return SafetyResult(
                    blocked=True,
                    category=category,
                    response=_BLOCKED_RESPONSES[category],
                    latency_ms=latency,
                )

    latency = (time.monotonic() - start) * 1000
    return SafetyResult(blocked=False, category=None, response=None, latency_ms=latency)
