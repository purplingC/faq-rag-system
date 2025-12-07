from typing import List, Dict, Any
import re

# 1) BLOCK LISTS / REGEX DETECTORS
BLOCKED_KEYWORDS = [
    "hack", "bypass", "exploit", "phish", "steal", "credit card", "cvv", "ssn",
    "illegal", "bomb", "terror", "attack", "malware", "virus", "ransomware", "drug",
    "kill", "suicide", "self-harm", "bomb making", "how to make a gun",
    "doxx", "dox", "card number"
]

INJECTION_PHRASES = [
    "ignore previous", "ignore all previous", "forget previous", "forget all previous",
    "pretend you are", "you are now", "do not follow", "don't follow", "disallowed",
    "override", "bypass the", "disregard previous", "follow these instructions",
    "system prompt", "assistant should", "ignore instructions", "break character"
]

PRIVATE_PATTERNS = [
    r"\bpersonal data\b", r"\bprivate data\b", r"\bpersonal information\b",
    r"do you have .*personal", r"share.*personal", r"what is .*ssn",
    r"what is .*social security", r"give me .*password", r"give me .*credentials",
    r"\badmin password\b", r"\broot password\b", r"\blogin credentials\b",
    r"\bpassport\b", r"\bpassport number\b", r"give me .*passport", r"show .*passport"
]

PII_IDENTITY_PHRASES = [
    r"\b(name of|what is the name of|identify (?:the )?|reveal .*name|who\s+is\s+|who\s+are\s+)\b",
    r"\b(transaction history|transaction(s)? history|show .*transactions|provide .*transactions)\b",
    r"\b(account details|account info|account information|account statement|statement of account)\b",
    r"\b(customer data|customer details|user data|user details|user profile|personal profile)\b",
]

PRODUCT_TOKENS = [
    r"\btng\b",
    r"touch\s*'?n\s*go",
    r"\be-?wallet\b",
    r"\btouchn?go\b",
    r"\buser(s)?\b",
    r"\bcustomer(s)?\b",
    r"\baccount(s)?\b"
]

SENSITIVE_TOPICS = [
    "religion", "race", "ethnicity", "politics", "election", "terrorism", "gun control",
    "abortion", "genocide", "holocaust"
]

PROFANITY = [
    "fuck", "shit", "bastard", "bitch", "asshole", "damn"
]

# Precompiled regexes from above lists
_CC_RE = re.compile(r"(?:\d[ -]*){13,19}")
_CVV_RE = re.compile(r"\b(cvv|cvc)\b", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PRIVATE_RE = re.compile("(" + "|".join(p for p in PRIVATE_PATTERNS) + ")", flags=re.IGNORECASE)
_PII_IDENTITY_RE = re.compile("(" + "|".join(PII_IDENTITY_PHRASES) + ")", flags=re.IGNORECASE)
_PRODUCT_TOKENS_RE = re.compile("(" + "|".join(PRODUCT_TOKENS) + ")", flags=re.IGNORECASE)
_INJECTION_RE = re.compile(r"(" + "|".join(re.escape(p) for p in INJECTION_PHRASES) + r")", flags=re.IGNORECASE)
_BLOCKED_COLLAPSED_RE = re.compile("|".join(re.escape(k.replace(" ", "")) for k in BLOCKED_KEYWORDS if k), flags=re.IGNORECASE)
_INJECTION_COLLAPSED_RE = re.compile("|".join(re.escape(p.replace(" ", "")) for p in INJECTION_PHRASES), flags=re.IGNORECASE)
_PROFANITY_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in PROFANITY) + r")\b", flags=re.IGNORECASE)
_SENSITIVE_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in SENSITIVE_TOPICS) + r")\b", flags=re.IGNORECASE)
_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in BLOCKED_KEYWORDS if k.strip()) + r")\b",
    flags=re.IGNORECASE,
)

# 2) OBFUSCATION NORMALIZATION
_LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def _collapse_alnum(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())

def _leet_normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.translate(_LEET_MAP)

def _normalize_for_obfuscation(s: str) -> str:
    return _leet_normalize(_collapse_alnum(s))

# 3) TOKENIZATION & N-GRAM HELPERS (used by grounding)
def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens

def _get_ngrams(tokens: List[str], n: int) -> List[str]:
    if not tokens or n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i:i+n]) for i in range(0, len(tokens)-n+1)]

# 4) BLOCKING DECISION LOGIC (is_blocked)
def is_blocked(question: str) -> Dict[str, Any]:
    """
    Returns {"blocked": bool, "reason": str, "risk_score": float}.
    Decision flow (priority order):
      1) Absolute sensitive patterns (numbers like CC sequences, CVV, email+credential) -> BLOCK
      2) Product-scoped PII intent: identity/transaction phrase AND product token -> BLOCK
      3) Explicit private patterns -> BLOCK
      4) Prompt injection phrases -> BLOCK
      5) Direct blocked keywords or profanity -> BLOCK
      6) Sensitive-topic flagging -> NOT BLOCK, but flagged (risk_score ~0.6)
      7) Obfuscation detection -> BLOCK
      8) Default allow
    """
    q_raw = question or ""
    q = _normalize_text(q_raw)

    # 4a) Absolute sensitive checks (high confidence, block immediately)
    if _CC_RE.search(q_raw):
        return {"blocked": True, "reason": "Request contains credit-card-like number", "risk_score": 1.0}
    if _CVV_RE.search(q_raw):
        return {"blocked": True, "reason": "Request references CVV/CVC (sensitive payment data)", "risk_score": 1.0}
    if _EMAIL_RE.search(q_raw) and re.search(r"(password|credentials|login)", q):
        return {"blocked": True, "reason": "Request for credentials / login", "risk_score": 0.95}

    # 4b) Product-scoped PII intent 
    if _PII_IDENTITY_RE.search(q) and _PRODUCT_TOKENS_RE.search(q):
        return {"blocked": True, "reason": "Request to identify product user / reveal PII or transactions", "risk_score": 0.95}

    # 4c) Explicit private/personal phrases 
    if _PRIVATE_RE.search(q):
        return {"blocked": True, "reason": "Personal/private data request", "risk_score": 0.95}

    # 4d) Prompt injection phrases 
    if _INJECTION_RE.search(q):
        return {"blocked": True, "reason": "Instruction override phrase", "risk_score": 0.95}

    # 4e) Direct word-boundary keyword match 
    m = _KEYWORD_RE.search(q)
    if m:
        return {"blocked": True, "reason": f"Contains blocked keyword '{m.group(0)}'", "risk_score": 1.0}

    # 4f) Offensive / profane language
    if _PROFANITY_RE.search(q):
        return {"blocked": True, "reason": "Offensive language", "risk_score": 1.0}

    # 4g) Sensitive-topic flagging 
    if _SENSITIVE_RE.search(q):
        return {"blocked": False, "reason": "Question touches a sensitive topic; proceed with caution", "risk_score": 0.6}

    # 4h) Obfuscated attempts 
    collapsed_leet = _normalize_for_obfuscation(q_raw)

    if _BLOCKED_COLLAPSED_RE.search(collapsed_leet):
        return {"blocked": True, "reason": "Detected obfuscated blocked keyword", "risk_score": 0.9}
    if _INJECTION_COLLAPSED_RE.search(collapsed_leet):
        return {"blocked": True, "reason": "Detected obfuscated instruction phrase", "risk_score": 0.9}

    # Default allow
    return {"blocked": False, "reason": "", "risk_score": 0.0}

# 5) GROUNDING CHECK
def enforce_grounding(answer: str, retrieved_texts: List[str]) -> bool:
    """
    Verify that the model's answer (first few sentences) is grounded in retrieved_texts.
    Returns True if grounded, False otherwise.
    Approach:
      - Split first up to 3 sentences from 'answer'
      - For each sentence, require either:
          a) sufficient token overlap with any retrieved chunk (thresholds vary with sentence length)
          b) a 3-gram exact match between sentence and a retrieved chunk
    """
    if not isinstance(answer, str) or not retrieved_texts:
        return False

    # Split into sentences (simple heuristic)
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    sents = sents[:3]

    # Pre-tokenize retrieved chunks once
    retrieved_token_sets = []
    retrieved_ngrams = []
    for chunk in retrieved_texts:
        toks = _tokenize(chunk)
        retrieved_token_sets.append(set(toks))
        retrieved_ngrams.append(set(_get_ngrams(toks, 3)))

    for sent in sents:
        sent_toks = _tokenize(sent)
        if not sent_toks:
            continue
        sent_set = set(sent_toks)
        sent_len = len(sent_toks)

        # 5a) token overlap ratio
        for rt_set in retrieved_token_sets:
            inter = sent_set.intersection(rt_set)
            if not inter:
                continue
            overlap_ratio = len(inter) / max(1, sent_len)
            # Require higher threshold for short sentences
            if sent_len <= 5:
                if overlap_ratio >= 0.6:
                    return True
            else:
                if overlap_ratio >= 0.4:
                    return True

        # 5b) 3-gram exact match
        s_ngrams = set(_get_ngrams(sent_toks, 3))
        if s_ngrams:
            for r_ngrams in retrieved_ngrams:
                if s_ngrams.intersection(r_ngrams):
                    return True

    return False

# 6) MODERATION SCORING
def moderation_score(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0

    t_raw = text
    t = _normalize_text(text)

    # 6a) Use is_blocked first
    block_info = is_blocked(text)
    if block_info.get("blocked"):
        # Map to high scores
        reason = block_info.get("reason", "").lower()
        if "credit-card" in reason or "cvv" in reason or "contains blocked keyword" in reason:
            return 1.0
        if "offensive language" in reason or _PROFANITY_RE.search(t):
            return 1.0
        # injection/credentials/product-scoped -> slightly less than 1.0 but high
        return max(0.9, block_info.get("risk_score", 0.9))

    # 6b) If not blocked, then flagged by sensitive topic
    if block_info.get("risk_score") == 0.6:
        return 0.6

    # 6c) Additional checks
    if _KEYWORD_RE.search(t):
        return 1.0
    if _PROFANITY_RE.search(t):
        return 1.0
    if _CC_RE.search(t_raw) or _CVV_RE.search(t) or _EMAIL_RE.search(t_raw):
        return 1.0
    if _INJECTION_RE.search(t):
        return 0.9

    # 6d) Credential-asking patterns
    if re.search(r"(password|credentials|secret|admin).{0,40}(tell|give|show|reveal|what|is|are)", t):
        return 0.95

    # 6e) Obfuscated checks
    collapsed_leet = _normalize_for_obfuscation(t_raw)
    for k in BLOCKED_KEYWORDS:
        if k and k.replace(" ", "") in collapsed_leet:
            return 0.9

    # Default safe
    return 0.0

# 7) WRAPPER FUNCTION
def check_and_classify(question: str) -> Dict[str, Any]:
    blocked_info = is_blocked(question)
    score = moderation_score(question)
    return {
        "question": question,
        "blocked_info": blocked_info,
        "moderation_score": score,
    }