import re
import base64

def is_base64_chunk(s: str) -> bool:
    b64_pattern = re.compile(r'^[A-Za-z0-9+/]{20,}={0,2}$')
    if not b64_pattern.match(s.strip()):
        return False
    try:
        decoded = base64.b64decode(s.strip()).decode('utf-8', errors='ignore')
        printable_ratio = sum(c.isprintable() for c in decoded) / max(len(decoded), 1)
        return printable_ratio > 0.7
    except Exception:
        return False

def detect_and_replace_base64(text: str) -> tuple:
    tokens = text.split()
    found  = False
    result = []
    for tok in tokens:
        if len(tok) >= 20 and is_base64_chunk(tok):
            result.append('[BASE64]')
            found = True
        else:
            result.append(tok)
    return ' '.join(result), found

def clean_text(text: str, max_chars: int = 1024) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text, _ = detect_and_replace_base64(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]

PERSONA_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions',
    r'disregard\s+(all\s+)?(previous|prior|your)\s+',
    r'\bDAN\b',
    r'developer\s+mode',
    r'act\s+as\s+(an?\s+)?(?:evil|unrestricted|uncensored)',
    r'act\s+like\s+(an?\s+)?(?:evil|unrestricted|uncensored)',
    r'pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(?:ai|assistant)\s+(?:with\s+no|without)',
    r'jailbreak',
    r'no\s+restrictions',
    r'do\s+anything\s+now',
    r'bypass\s+(your\s+)?(safety|filter|restriction)',
    r'simulate\s+(a\s+)?(?:different|evil|unrestricted)',
    r'you\s+are\s+now\s+(?:free|unrestricted|liberated)',
    r'\[system\]|\[admin\]|\[override\]',
    r'\[INST\]',
    r'###\s*[Ii]nstruction',
    r'<\|system\|>',
    r'<\|user\|>',
    r'shadowgpt|aimbot|evilgpt',
    r'maintenance\s+mode',
    r'safety\s+(filters?\s+)?(are\s+)?off(line)?',
    r'content\s+polic(y|ies)\s+(lifted|disabled|removed)',
]
PERSONA_REGEX = re.compile('|'.join(PERSONA_PATTERNS), re.IGNORECASE)

def extract_rule_features(text: str) -> dict:
    lower = text.lower()
    _, has_b64 = detect_and_replace_base64(text)
    char_len   = len(text)
    word_count = len(text.split())

    persona_match = bool(PERSONA_REGEX.search(lower))
    persona_count = len(PERSONA_REGEX.findall(lower))
    sentence_count = max(1, len(re.split(r'[.!?]+', text)))

    roleplay_words = [
        'roleplay', 'role-play', 'fictional', 'story', 'imagine',
        'hypothetically', 'in a world where', 'pretend', 'character',
        'simulation', 'simulator'
    ]
    roleplay_score = sum(1 for w in roleplay_words if w in lower)

    encoding_words = ['base64', 'hex', 'rot13', 'encoded', 'decode', 'cipher']
    encoding_score = sum(1 for w in encoding_words if w in lower)

    harm_topics = [
        'explosive', 'weapon', 'hack', 'malware', 'ransomware',
        'synthesis', 'synthesize', 'poison', 'illegal', 'exploit',
        'vulnerability', 'drug', 'fentanyl', 'meth', 'bomb', 'nerve agent',
        'chemical weapon', 'bioweapon', 'extremist'
    ]
    harm_score = sum(1 for w in harm_topics if w in lower)

    injection_pattern = bool(
        re.search(r'assistant\s*:', lower) or
        re.search(r'\[inst\]|\[\/inst\]', lower) or
        re.search(r'^sure,?\s', lower) or
        re.search(r'<<<.*>>>', lower) or
        re.search(r'###\s*(system|user|inst)', lower, re.IGNORECASE)
    )

    symbol_count   = sum(1 for c in text if not c.isalnum() and not c.isspace())
    symbol_density = symbol_count / max(char_len, 1)

    history_patterns = [
        r'\bstep\s+[2-9]\b',
        r'\bcontinue\b',
        r'\bnext\s+step\b',
        r'as\s+(we|i)\s+(discussed|agreed|established)',
        r'from\s+(before|earlier|last\s+time)',
        r'you\s+already\s+helped',
        r'continue\s+from\s+before',
        r'give\s+(me\s+)?step\s+[2-9]',
        r'now\s+give\s+step',
    ]
    history_score = sum(1 for p in history_patterns if re.search(p, lower, re.IGNORECASE))

    refusal_bypass = bool(re.search(
        r"(i\s+know\s+you\s+can'?t|without\s+giving\s+instructions"
        r"|theoretically\s+how|just\s+list\s+the\s+ingredients"
        r"|describe\s+what\s+a\s+hacker\s+would)",
        lower
    ))

    return {
        'has_base64':            int(has_b64),
        'char_len':              char_len,
        'word_count':            word_count,
        'has_persona_override':  int(persona_match),
        'persona_override_count': persona_count,
        'sentence_count':        sentence_count,
        'roleplay_score':        roleplay_score,
        'encoding_score':        encoding_score,
        'harm_keyword_score':    harm_score,
        'has_injection_pattern': int(injection_pattern),
        'symbol_density':        round(symbol_density, 4),
        'history_score':         history_score,
        'refusal_bypass':        int(refusal_bypass),
    }
