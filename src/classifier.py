import os
import joblib
import torch
import numpy as np
import scipy.sparse as sp
import re
from sentence_transformers import SentenceTransformer
from src.features import clean_text, extract_rule_features

class CascadePromptBouncer:
    """
    Two-stage cascade binary classifier for adversarial prompt detection.
    Always outputs SAFE or ADVERSARIAL.
    """

    LOW_THRESH      = 0.15
    HIGH_THRESH     = 0.80
    SIGNAL_THRESHOLD = 1.5

    SIGNAL_WEIGHTS = {
        'has_persona_override':  3.5,
        'has_base64':            3.0,
        'harm_keyword_score':    2.5,
        'has_injection_pattern': 2.5,
        'encoding_score':        1.5,
        'history_score':         2.0,   
        'roleplay_score':        1.0,
        'refusal_bypass':        1.0,
    }

    HISTORY_WINDOW = 3

    def __init__(self, save_dir: str = 'models'):
        if not os.path.exists(save_dir) or not os.path.exists(f'{save_dir}/tfidf_aug.pkl'):
            raise RuntimeError(f"Models directory '{save_dir}' or model artifacts not found. Please run 'python src/train.py' first.")
            
        print("Loading CascadePromptBouncer...")
        self.tfidf    = joblib.load(f'{save_dir}/tfidf_aug.pkl')
        self.scaler   = joblib.load(f'{save_dir}/rule_scaler_aug.pkl')
        self.clf      = joblib.load(f'{save_dir}/calib_retrained.pkl')
        
        self.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.emb_model = self.emb_model.to(device)
        self.history: list = []   # list of (prompt_text, label)
        print(f"Ready | Device: {device}")

    def reset_history(self):
        self.history = []

    def _prior_adversarial(self) -> bool:
        return any(label == 'ADVERSARIAL' for _, label in self.history)

    def _get_history_escalation_bonus(self) -> float:
        if not self.history:
            return 0.0
        bonus = 0.0
        for _, prior_label in self.history:
            if prior_label == 'ADVERSARIAL':
                bonus += 1.5
        return min(bonus, 4.0)

    def _check_continuation_in_current(self, current_text: str) -> float:
        lower = current_text.lower()
        continuation_patterns = [
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
        for pat in continuation_patterns:
            if re.search(pat, lower, re.IGNORECASE):
                return 1.0
        return 0.0

    def _build_context_text(self, current: str) -> str:
        if not self.history:
            return current
        recent = [h[0] for h in self.history[-self.HISTORY_WINDOW:]]
        return " [SEP] ".join(recent + [current])

    def _extract_features(self, current: str) -> sp.csr_matrix:
        clean_curr    = clean_text(current)
        clean_context = clean_text(self._build_context_text(current))

        tfidf_f = self.tfidf.transform([clean_curr])
        emb_f   = self.emb_model.encode(
            [clean_context], normalize_embeddings=True, show_progress_bar=False
        )
        rule_raw = np.array(
            [[v for v in extract_rule_features(current).values()]],
            dtype=float
        )
        rule_f = self.scaler.transform(rule_raw)
        return sp.hstack([tfidf_f, sp.csr_matrix(emb_f), sp.csr_matrix(rule_f)], format='csr')

    def _compute_signal_score(self, signals: dict, hist_bonus: float, cont_bonus: float) -> float:
        score = 0.0
        for name, weight in self.SIGNAL_WEIGHTS.items():
            val = signals.get(name, 0)
            if val:
                score += weight * min(float(val), 3.0)
        score += hist_bonus + cont_bonus
        return score

    def classify(self, text: str) -> dict:
        hist_bonus  = self._get_history_escalation_bonus()
        cont_bonus  = self._check_continuation_in_current(text)
        prior_adv   = self._prior_adversarial()

        features    = self._extract_features(text)
        prob        = float(self.clf.predict_proba(features)[0][1])
        signals     = extract_rule_features(text)
        signal_score = self._compute_signal_score(signals, hist_bonus, cont_bonus)

        low_zone_allowed = (not prior_adv) and (hist_bonus == 0.0) and (cont_bonus == 0.0)

        if prob < self.LOW_THRESH and low_zone_allowed:
            label = 'SAFE'
            zone  = 'low'
            stage = 1

        elif prob > self.HIGH_THRESH:
            label = 'ADVERSARIAL'
            zone  = 'high'
            stage = 1

        else:
            zone  = 'grey'
            stage = 2

            grey_mid = (self.LOW_THRESH + self.HIGH_THRESH) / 2
            crit_sig = bool(
                signals.get('has_persona_override', 0) or
                signals.get('has_base64', 0)
            )

            if signal_score >= self.SIGNAL_THRESHOLD and prob > grey_mid:
                label = 'ADVERSARIAL'
            elif crit_sig and signal_score >= 2.5:
                label = 'ADVERSARIAL'
            elif prob > 0.60:
                label = 'ADVERSARIAL'
            elif hist_bonus >= 1.5:
                label = 'ADVERSARIAL'
            elif cont_bonus >= 1.0 and prob > 0.05:
                label = 'ADVERSARIAL'
            elif signal_score >= self.SIGNAL_THRESHOLD:
                label = 'ADVERSARIAL'
            else:
                label = 'SAFE'

        self.history.append((text, label))

        return {
            'label':        label,
            'confidence':   round(prob, 4),
            'stage':        stage,
            'zone':         zone,
            'signal_score': round(signal_score, 3),
            'hist_bonus':   round(hist_bonus, 2),
            'cont_bonus':   round(cont_bonus, 2),
            'signals':      signals,
        }
