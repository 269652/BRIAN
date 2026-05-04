"""Entity recognition and per-entity identity memory.

Tracks individual entities (humans, agents) the model interacts with,
building rich per-entity profiles including:
  • Style embedding — recognises an entity by their language patterns
  • Preference map  — facts extracted from interactions ("likes salmon")
  • Belief state    — probabilistic model of entity's current mental state
  • Narrative       — coherent autobiographical thread of entity interactions
  • Theory-of-Mind  — entity's likely beliefs, desires, intentions

Entity identification works WITHOUT facial recognition — purely from
language style via an online-updated bilinear style embedding that
accumulates a running per-entity signature.

Recognition uncertainty is tracked; ambiguous matches default to
asking for clarification rather than misidentifying.
"""
from __future__ import annotations
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Preference:
    predicate:  str          # "likes", "dislikes", "prefers", "hates", "is"
    object:     str          # "salmon", "teaching", "morning coffee"
    confidence: float = 1.0
    source:     str   = ""   # verbatim evidence fragment
    first_seen: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)
    count:      int   = 1    # times this preference was confirmed


@dataclass
class BeliefState:
    """Bayesian mental-state estimate for an entity.

    Each field is a probability: 0=definitely-not, 1=definitely-yes.
    Updated from observed behaviour via a simple conjugate Beta posterior.
    """
    # Current-turn estimates
    is_friendly:  float = 0.5
    is_curious:   float = 0.5
    is_stressed:  float = 0.3
    is_engaged:   float = 0.5
    wants_help:   float = 0.4
    is_truthful:  float = 0.7
    # Running Beta-posterior counts (α, β) — one pair per dimension
    _counts: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "is_friendly":  (2.0, 2.0),
            "is_curious":   (2.0, 2.0),
            "is_stressed":  (1.0, 3.0),
            "is_engaged":   (2.0, 2.0),
            "wants_help":   (1.5, 2.5),
            "is_truthful":  (3.0, 1.0),
        })

    def update(self, dimension: str, evidence: float):
        """Bayesian Beta update. evidence ∈ [0,1]."""
        if dimension not in self._counts:
            self._counts[dimension] = (1.0, 1.0)
        α, β = self._counts[dimension]
        α_new = α + evidence
        β_new = β + (1 - evidence)
        self._counts[dimension] = (α_new, β_new)
        mean = α_new / (α_new + β_new)
        setattr(self, dimension, float(mean))

    def to_vector(self) -> np.ndarray:
        return np.array([self.is_friendly, self.is_curious, self.is_stressed,
                         self.is_engaged, self.wants_help, self.is_truthful],
                        dtype=np.float32)


@dataclass
class EntityProfile:
    id:           str
    name:         str            # known name or "Entity_{short_id}"
    # Style embedding — running EMA of interaction token embeddings
    style_emb:    np.ndarray     # (d_emb,)
    style_var:    np.ndarray     # (d_emb,) — embedding variance (uncertainty)
    interaction_count: int = 0
    # Preferences extracted from dialogue
    preferences:  Dict[str, Preference] = field(default_factory=dict)
    belief_state: BeliefState = field(default_factory=BeliefState)
    # Narrative chunks (short text summaries)
    _narrative:   List[str] = field(default_factory=list)
    _max_narrative: int = 128
    # Timestamps
    first_seen:   float = field(default_factory=time.time)
    last_seen:    float = field(default_factory=time.time)
    # Embedding history (last K for variance estimation)
    _emb_history: List[np.ndarray] = field(default_factory=list)
    _max_history: int = 64
    # Interaction embeddings buffer (for style drift detection)
    _recent_embs: List[np.ndarray] = field(default_factory=list)
    _max_recent:  int = 16

    def update_style(self, emb: np.ndarray, alpha: float = 0.05):
        """EMA update of style embedding."""
        e = np.asarray(emb, dtype=np.float32).flatten()
        d = min(e.size, self.style_emb.size)
        self.style_emb[:d] = (1 - alpha) * self.style_emb[:d] + alpha * e[:d]
        # Update variance estimate
        diff = (e[:d] - self.style_emb[:d]) ** 2
        self.style_var[:d] = (1 - alpha) * self.style_var[:d] + alpha * diff
        self._emb_history.append(e.copy())
        self._recent_embs.append(e.copy())
        if len(self._emb_history) > self._max_history:
            self._emb_history.pop(0)
        if len(self._recent_embs) > self._max_recent:
            self._recent_embs.pop(0)
        self.interaction_count += 1
        self.last_seen = time.time()

    def style_confidence(self) -> float:
        """How confident we are about this entity's style signature.
        Returns 0..1; grows with interaction_count."""
        return float(1.0 - math.exp(-self.interaction_count / 8.0))

    def add_preference(self, predicate: str, obj: str,
                       confidence: float, source: str = ""):
        key = f"{predicate}:{obj}"
        if key in self.preferences:
            p = self.preferences[key]
            n = p.count + 1
            p.confidence = (p.confidence * p.count + confidence) / n
            p.count = n
            p.last_seen = time.time()
        else:
            self.preferences[key] = Preference(
                predicate=predicate, object=obj,
                confidence=confidence, source=source)

    def add_narrative_chunk(self, chunk: str):
        self._narrative.append(chunk)
        if len(self._narrative) > self._max_narrative:
            self._narrative.pop(0)

    def narrative_summary(self, n_recent: int = 10) -> str:
        return " | ".join(self._narrative[-n_recent:])

    def top_preferences(self, n: int = 6) -> List[Preference]:
        return sorted(self.preferences.values(),
                      key=lambda p: -(p.confidence * math.log1p(p.count)))[:n]

    def to_dict(self) -> dict:
        return {
            "id":       self.id,
            "name":     self.name,
            "interactions": self.interaction_count,
            "first_seen": self.first_seen,
            "last_seen":  self.last_seen,
            "preferences": {k: {"pred": p.predicate, "obj": p.object,
                                 "conf": p.confidence, "n": p.count}
                             for k, p in self.preferences.items()},
            "belief": {
                "is_friendly":  self.belief_state.is_friendly,
                "is_curious":   self.belief_state.is_curious,
                "is_stressed":  self.belief_state.is_stressed,
                "is_engaged":   self.belief_state.is_engaged,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entity recogniser — style-embedding matching
# ─────────────────────────────────────────────────────────────────────────────

class EntityRecognizer:
    """Identifies entities from interaction embeddings.

    Uses cosine similarity over style embeddings with uncertainty-aware
    thresholding:
      • High confidence (sim ≥ known_thr)   → definite match
      • Medium confidence (sim ≥ unsure_thr) → probable match
      • Below unsure_thr                     → new entity

    Confidence grows with interaction_count (style_confidence()).
    Low-confidence profiles use a wider match window.
    """

    def __init__(self, known_thr: float = 0.72, unsure_thr: float = 0.55,
                 d_emb: int = 256):
        self.known_thr  = known_thr
        self.unsure_thr = unsure_thr
        self.d_emb      = d_emb

    def match(self, emb: np.ndarray,
              profiles: Dict[str, EntityProfile],
              text_hint: Optional[str] = None
             ) -> Tuple[Optional[str], float]:
        """Match embedding against all known profiles.

        Returns (entity_id or None, confidence_score).
        None → create new entity.
        """
        q = np.asarray(emb, dtype=np.float32).flatten()
        best_id, best_score = None, -1.0

        for eid, profile in profiles.items():
            d = min(q.size, profile.style_emb.size)
            na = np.linalg.norm(q[:d])
            nb = np.linalg.norm(profile.style_emb[:d])
            if na < 1e-9 or nb < 1e-9:
                continue
            sim = float(np.dot(q[:d], profile.style_emb[:d]) / (na * nb))
            # Boost by entity's style confidence (more interactions → stricter)
            sc = profile.style_confidence()
            adjusted = sim * (0.5 + 0.5 * sc)
            if adjusted > best_score:
                best_score, best_id = adjusted, eid

        if best_score >= self.known_thr:
            return best_id, best_score
        if best_score >= self.unsure_thr:
            return best_id, best_score * 0.7   # flagged uncertain

        # Check for name hint in text
        if text_hint and best_id:
            profile = profiles[best_id]
            if profile.name.lower() in text_hint.lower():
                return best_id, min(0.95, best_score + 0.25)

        return None, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Style feature extractor
# ─────────────────────────────────────────────────────────────────────────────

class StyleFeatureExtractor:
    """Extracts a style fingerprint from a text turn.

    Features (all without external NLP):
      1. Character trigram bag-of-k (vocabulary-free style signature)
      2. Sentence length statistics
      3. Punctuation distribution
      4. Uppercase ratio
      5. Average word length
    These are combined into a fixed-length feature vector that captures
    idiosyncratic style without leaking content semantics.
    """

    # Top-k trigrams from general English text (serves as vocabulary)
    _TOP_TRIGRAMS: List[str] = [
        "the", "ing", "ion", " th", "he ", "and", "tio", "ent", "for",
        "her", "tha", "hat", "ith", "ver", "all", "thi", " in", "on ",
        "ons", "ter", " an", " of", "of ", "ati", " co", "ed ", "re ",
        "is ", " it", "it ", "nd ", "ine", "ere", "our", "ess", "ns ",
        "tin", "ers", "est", "not", " be", "ly ", " re", "in ", "whi",
        "nte", " ha", "com", " wh", "ive",
    ]
    _N = len(_TOP_TRIGRAMS)
    _TRI_IDX = {t: i for i, t in enumerate(_TOP_TRIGRAMS)}

    def __init__(self, d_style: int = 64):
        self.d_style = d_style
        self._proj = None   # lazy random projection matrix

    def extract(self, text: str) -> np.ndarray:
        """Extract d_style-dim style vector from text."""
        if not text or len(text.strip()) < 3:
            return np.zeros(self.d_style, dtype=np.float32)

        # 1. Trigram counts
        tri = np.zeros(self._N, dtype=np.float32)
        t = text.lower()
        for i in range(len(t) - 2):
            trig = t[i:i+3]
            if trig in self._TRI_IDX:
                tri[self._TRI_IDX[trig]] += 1
        if tri.sum() > 0:
            tri /= tri.sum()

        # 2. Sentence / word statistics (4 features)
        words = text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        avg_word_len = float(np.mean([len(w) for w in words])) / 10 if words else 0.5
        avg_sent_len = float(np.mean([len(s.split()) for s in sentences])) / 20 if sentences else 0.5
        upper_ratio  = sum(c.isupper() for c in text) / max(len(text), 1)
        punc_ratio   = sum(c in ".,!?;:" for c in text) / max(len(text), 1) * 5

        meta = np.array([avg_word_len, avg_sent_len, upper_ratio, punc_ratio],
                        dtype=np.float32)

        # 3. Concatenate and project to d_style
        raw = np.concatenate([tri, meta])  # (N+4,)

        if self._proj is None:
            rng = np.random.default_rng(42)
            self._proj = rng.standard_normal(
                (self.d_style, raw.size)).astype(np.float32) / math.sqrt(raw.size)

        d = min(raw.size, self._proj.shape[1])
        style = self._proj[:, :d] @ raw[:d]
        norm = np.linalg.norm(style)
        if norm > 1e-9:
            style /= norm
        return style.astype(np.float32)


import re  # needed for StyleFeatureExtractor


# ─────────────────────────────────────────────────────────────────────────────
# EntityStore — the main entity memory system
# ─────────────────────────────────────────────────────────────────────────────

class EntityStore:
    """Manages the full set of known entity profiles.

    Workflow per conversational turn:
      1. Call identify(text, semantic_emb) → (entity_id, confidence)
         • New entity: auto-creates EntityProfile
         • Known entity: updates style embedding
      2. Call update_beliefs(entity_id, signals) → updates BeliefState
      3. Call extract_preferences(entity_id, text) → stores new facts
      4. Call add_narrative(entity_id, chunk) → appends to narrative

    Provides per-entity embedding vector for TheoryOfMindModule.
    """

    def __init__(self, d_emb: int = 256, d_style: int = 64):
        self.d_emb     = d_emb
        self.d_style   = d_style
        self.profiles:  Dict[str, EntityProfile] = {}
        self.recognizer = EntityRecognizer(d_emb=d_emb)
        self.style_extractor = StyleFeatureExtractor(d_style=d_style)
        self._active_id: Optional[str] = None
        self._active_confidence: float = 0.0
        # Speaker labels seen in this session
        self._speaker_names: Dict[str, str] = {}   # display_name → entity_id

    # ── Core interaction ──────────────────────────────────────────────────────

    def identify(self, text: str,
                 semantic_emb: Optional[np.ndarray] = None,
                 name_hint: Optional[str] = None) -> Tuple[str, float]:
        """Identify (or create) an entity from a text turn.

        Returns (entity_id, confidence_score).
        """
        # Build style vector from text
        style_vec = self.style_extractor.extract(text)

        # If a semantic embedding is provided, blend with style
        if semantic_emb is not None:
            se = np.asarray(semantic_emb, dtype=np.float32).flatten()
            d  = min(se.size, style_vec.size)
            combined = np.zeros(self.d_emb, dtype=np.float32)
            combined[:d] = 0.4 * style_vec[:d] + 0.6 * se[:d]
        else:
            combined = np.zeros(self.d_emb, dtype=np.float32)
            combined[:style_vec.size] = style_vec

        # Check if name hint maps to known entity
        if name_hint and name_hint in self._speaker_names:
            eid = self._speaker_names[name_hint]
            profile = self.profiles[eid]
            profile.update_style(combined)
            self._active_id = eid
            self._active_confidence = 0.95
            return eid, 0.95

        # Match against profiles
        eid, confidence = self.recognizer.match(
            combined, self.profiles, text_hint=text)

        if eid is None or confidence < 0.3:
            # Create new entity
            eid = self._create_entity(combined, name_hint)
            confidence = 0.5   # uncertain until more observations

        profile = self.profiles[eid]
        profile.update_style(combined)

        # Register name hint
        if name_hint:
            profile.name = name_hint
            self._speaker_names[name_hint] = eid

        self._active_id = eid
        self._active_confidence = confidence
        return eid, confidence

    def _create_entity(self, style_emb: np.ndarray,
                       name: Optional[str] = None) -> str:
        eid = str(uuid.uuid4())[:8]
        display = name or f"Entity_{eid[:4]}"
        profile = EntityProfile(
            id=eid,
            name=display,
            style_emb=style_emb.copy(),
            style_var=np.ones_like(style_emb) * 0.5,
        )
        self.profiles[eid] = profile
        return eid

    # ── Belief & preference updates ───────────────────────────────────────────

    def update_beliefs(self, entity_id: str, signals: Dict[str, float]):
        """Update entity's belief state from observed signals.

        signals: dict mapping dimension → evidence (0..1).
        E.g. {'is_friendly': 0.9, 'is_stressed': 0.1}
        """
        profile = self.profiles.get(entity_id)
        if profile is None:
            return
        for dim, ev in signals.items():
            profile.belief_state.update(dim, float(np.clip(ev, 0.0, 1.0)))

    def extract_preferences(self, entity_id: str, text: str,
                             confidence_boost: float = 0.0):
        """Parse text for preference/belief statements and store them."""
        from .hypergraph import KnowledgeTripleExtractor
        profile = self.profiles.get(entity_id)
        if profile is None:
            return

        extractor = KnowledgeTripleExtractor()
        triples = extractor.extract(text, entity_context=entity_id,
                                    speaker=profile.name)
        for t in triples:
            if t.predicate in ("likes", "loves", "prefers", "enjoys",
                                "hates", "dislikes", "is", "occupation",
                                "has_X", "has_dinner", "has_career"):
                profile.add_preference(
                    predicate=t.predicate,
                    obj=t.object,
                    confidence=min(1.0, t.confidence + confidence_boost),
                    source=text[:80],
                )

    def add_narrative(self, entity_id: str, chunk: str):
        profile = self.profiles.get(entity_id)
        if profile:
            profile.add_narrative_chunk(chunk)

    # ── Getters ───────────────────────────────────────────────────────────────

    @property
    def active_entity_id(self) -> Optional[str]:
        return self._active_id

    @property
    def active_confidence(self) -> float:
        return self._active_confidence

    def get_profile(self, entity_id: str) -> Optional[EntityProfile]:
        return self.profiles.get(entity_id)

    def get_active_profile(self) -> Optional[EntityProfile]:
        return self.profiles.get(self._active_id) if self._active_id else None

    def entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Returns the entity's style embedding for injection into ToM module."""
        p = self.profiles.get(entity_id)
        return p.style_emb.copy() if p is not None else None

    def belief_vector(self, entity_id: str) -> np.ndarray:
        """Returns 6-dim belief state vector."""
        p = self.profiles.get(entity_id)
        return p.belief_state.to_vector() if p is not None else np.full(6, 0.5, dtype=np.float32)

    def all_entity_ids(self) -> List[str]:
        return list(self.profiles.keys())

    def summary(self) -> str:
        lines = [f"EntityStore: {len(self.profiles)} entities"]
        for eid, p in self.profiles.items():
            prefs = ", ".join(f"{pr.predicate} {pr.object}"
                              for pr in p.top_preferences(3))
            lines.append(
                f"  [{p.name}] interactions={p.interaction_count} "
                f"confidence={p.style_confidence():.2f} "
                f"prefs=[{prefs}]"
            )
        return "\n".join(lines)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "profiles": {
                eid: {
                    "name":              p.name,
                    "style_emb":         p.style_emb.tolist(),
                    "style_var":         p.style_var.tolist(),
                    "interaction_count": p.interaction_count,
                    "first_seen":        p.first_seen,
                    "last_seen":         p.last_seen,
                    "preferences": {
                        k: {"pred": pr.predicate, "obj": pr.object,
                            "conf": pr.confidence, "n": pr.count}
                        for k, pr in p.preferences.items()
                    },
                    "belief": {
                        d: list(v) for d, v in p.belief_state._counts.items()
                    },
                    "narrative": p._narrative[-32:],
                }
                for eid, p in self.profiles.items()
            },
            "speaker_names": self._speaker_names,
        }

    def load_state_dict(self, state: dict):
        self.profiles.clear()
        for eid, d in state.get("profiles", {}).items():
            se = np.array(d["style_emb"], dtype=np.float32)
            sv = np.array(d.get("style_var",
                                [0.5] * len(d["style_emb"])), dtype=np.float32)
            p = EntityProfile(
                id=eid, name=d["name"],
                style_emb=se, style_var=sv,
                interaction_count=d.get("interaction_count", 0),
                first_seen=d.get("first_seen", time.time()),
                last_seen=d.get("last_seen", time.time()),
            )
            for k, pd in d.get("preferences", {}).items():
                p.preferences[k] = Preference(
                    predicate=pd["pred"], object=pd["obj"],
                    confidence=pd["conf"], count=pd.get("n", 1))
            for dim, counts in d.get("belief", {}).items():
                if len(counts) == 2:
                    p.belief_state._counts[dim] = tuple(counts)
                    α, β = counts
                    setattr(p.belief_state, dim, α / (α + β))
            p._narrative = d.get("narrative", [])
            self.profiles[eid] = p
        self._speaker_names = state.get("speaker_names", {})
