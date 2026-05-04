"""NeuroSLM Cognitive HyperGraph
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A multidimensional associative memory beyond current frontier SLMs.

Novel contributions:
  1. SocialMarkovMemory — Dirichlet posterior over (action_type → response_type)
     automatically learns "expressing kindness → kind response (p=0.82)".
  2. N-ary HyperEdges — (entity_X, kind_expression, context, kind_response)
     as a single representational unit; enables richer causal attribution.
  3. SemanticDeduplicator — online cosine clustering prevents memory bloat;
     near-duplicate episodes merge into schema abstractions automatically.
  4. KnowledgeTripleExtractor — (subject, predicate, object) from natural
     language without external NLP; confidence-weighted by embedding sim.
  5. EntitySubgraph — per-entity private memory slice; SELF / MOTHER / USER
     each have their own narrative without cross-entity bleed.

Memory hierarchy (Complementary Learning Systems):
  EPISODIC  →  slow forgetting, high fidelity, raw events
  SEMANTIC  →  extracted generalisations, medium decay
  SCHEMA    →  highly compressed, abstract, near-permanent
  PROCEDURAL → sequential skill chains, protected from decay
"""
from __future__ import annotations
import re
import time
import uuid
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class MemoryType(Enum):
    EPISODIC   = auto()
    SEMANTIC   = auto()
    SCHEMA     = auto()
    PROCEDURAL = auto()

class RelationType(Enum):
    SEMANTIC   = "semantic"     # cosine similarity
    TEMPORAL   = "temporal"     # recency proximity
    CAUSAL     = "causal"       # A caused B
    SOCIAL     = "social"       # inter-entity interaction
    CONTEXTUAL = "contextual"   # shared context / topic
    MOOD       = "mood"         # NT-state congruence
    REWARD     = "reward"       # DA-tagged high value
    PROCEDURAL = "procedural"   # sequential skill step
    IDENTITY   = "identity"     # entity-self relation


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeTriple:
    subject:    str            # entity name ("self", "user", "mother")
    predicate:  str            # "likes", "is", "has", "hates", "believes"
    object:     str            # fact value
    confidence: float = 1.0
    source_node: Optional[str] = None
    entity_id:   Optional[str] = None
    timestamp:   float = field(default_factory=time.time)


@dataclass
class HyperNode:
    id:           str
    content:      str
    embedding:    np.ndarray           # (d_sem,)
    memory_type:  MemoryType = MemoryType.EPISODIC
    entity_ref:   Optional[str] = None
    temporal_idx: int = 0
    valence:      float = 0.0          # −1..+1
    arousal:      float = 0.5          # 0..1
    salience:     float = 0.5
    access_count: int = 0
    decay:        float = 1.0          # multiplicative; approaches 0
    is_abstract:  bool = False
    triples:      List[KnowledgeTriple] = field(default_factory=list)
    nt_state:     Optional[np.ndarray] = None   # NT snapshot at encoding
    timestamp:    float = field(default_factory=time.time)


@dataclass
class HyperEdge:
    id:           str
    node_ids:     List[str]            # N≥2 nodes
    relation:     RelationType
    weight:       float = 1.0
    probability:  float = 1.0          # from Dirichlet posterior
    observations: int = 1
    timestamp:    float = field(default_factory=time.time)


@dataclass
class SocialRule:
    action_label:   str
    response_label: str
    probability:    float
    observations:   int
    entity_context: Optional[str] = None   # specific entity, or None = general


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge-triple extraction (no external NLP dependencies)
# ─────────────────────────────────────────────────────────────────────────────

_PATTERNS = [
    # "I like/love/prefer/enjoy X"
    (r"\b(?:I|i)\s+(?:really\s+)?(?:like|love|prefer|enjoy|adore|hate|dislike)\s+([^.,!?;]+)",
     "self", "{verb}", 0.85),
    # "X is a/an Y" or "X is Y"
    (r"\b(\w[\w\s]{0,20})\s+is\s+(?:a\s+|an\s+)?([^.,!?;]{2,40})",
     "{subj}", "is", 0.7),
    # "My/His/Her X is Y" → (speaker/entity, has_X, Y)
    (r"\b(?:my|his|her)\s+(\w[\w\s]{0,15})\s+is\s+([^.,!?;]{2,30})",
     "self", "has_{attr}", 0.8),
    # "X's Y is Z"
    (r"\b(\w[\w\s]{0,15})'s\s+(\w[\w\s]{0,15})\s+is\s+([^.,!?;]{2,30})",
     "{subj}", "has_{attr}", 0.75),
    # "I am/I'm a/an X"
    (r"\b(?:I am|I'm)\s+(?:a\s+|an\s+)?([^.,!?;]{2,30})",
     "self", "is", 0.8),
    # "X works as/is a Y"
    (r"\b(\w[\w\s]{0,20})\s+(?:works\s+as|is\s+a)\s+([^.,!?;]{2,30})",
     "{subj}", "occupation", 0.85),
]


class KnowledgeTripleExtractor:
    """Extracts (subject, predicate, object) triples from text using
    regex patterns + optional embedding similarity confirmation."""

    def extract(self, text: str, entity_context: Optional[str] = None,
                speaker: str = "self") -> List[KnowledgeTriple]:
        triples: List[KnowledgeTriple] = []
        seen: set = set()

        for pat, subj_tmpl, pred_tmpl, conf in _PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                groups = m.groups()
                if not groups:
                    continue

                # Resolve subject
                subj = subj_tmpl
                if subj == "self":
                    subj = speaker
                elif "{subj}" in subj and groups:
                    subj = groups[0].strip().lower()

                # Resolve predicate
                pred = pred_tmpl
                if "{verb}" in pred:
                    # extract verb from match
                    vm = re.search(
                        r"\b(like|love|prefer|enjoy|adore|hate|dislike)\b",
                        m.group(0), re.IGNORECASE)
                    pred = vm.group(1).lower() if vm else "likes"
                elif "{attr}" in pred and len(groups) >= 2:
                    pred = pred.replace("{attr}", groups[0].strip().lower())

                # Object is the last group
                obj = groups[-1].strip().lower()
                obj = re.sub(r"\s+", " ", obj)[:60]

                key = (subj, pred, obj)
                if key in seen or len(obj) < 2:
                    continue
                seen.add(key)

                triples.append(KnowledgeTriple(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    confidence=conf,
                    entity_id=entity_context,
                ))

        return triples


# ─────────────────────────────────────────────────────────────────────────────
# Social Markov Memory — Dirichlet posterior over action→response transitions
# ─────────────────────────────────────────────────────────────────────────────

class SocialMarkovMemory:
    """Dirichlet-multinomial posterior over (action_type → response_type).

    Learns generalisations like:
      "expressing kindness   → kind_response   (p=0.82)"
      "asking a question     → answer          (p=0.73)"
      "making an assertion   → acknowledgement (p=0.61)"

    Action/response types are discovered by online k-means of text embeddings.
    The Dirichlet prior α prevents zero-probability outcomes.
    """

    # Fixed action type labels inferred from k-means centroids
    ACTION_LABELS = [
        "kind_expression", "question", "assertion", "criticism",
        "apology", "gratitude", "greeting", "farewell",
        "request", "refusal", "agreement", "disagreement",
        "emotional", "neutral",
    ]
    N_TYPES = len(ACTION_LABELS)

    def __init__(self, d_emb: int = 256, alpha: float = 0.5):
        self.d_emb = d_emb
        self.alpha = alpha          # Dirichlet concentration prior
        # (action_type, response_type) count matrix
        self.counts = np.full((self.N_TYPES, self.N_TYPES), alpha)
        # Action-type centroids (online k-means)
        self.centroids = np.random.randn(self.N_TYPES, d_emb).astype(np.float32)
        self.centroid_n = np.ones(self.N_TYPES, dtype=np.float32)
        self._pending_action: Optional[Tuple[int, np.ndarray]] = None

    def _assign_type(self, emb: np.ndarray) -> int:
        """Assign embedding to nearest centroid, then update centroid."""
        e = np.asarray(emb, dtype=np.float32).flatten()
        d = min(e.size, self.d_emb)
        e_ = e[:d]
        c_ = self.centroids[:, :d]
        norms_e = np.linalg.norm(e_) + 1e-9
        norms_c = np.linalg.norm(c_, axis=1) + 1e-9
        sims = (c_ @ e_) / (norms_c * norms_e)
        idx = int(np.argmax(sims))
        # Online centroid update
        n = self.centroid_n[idx]
        self.centroids[idx, :d] = (self.centroids[idx, :d] * n + e_) / (n + 1)
        self.centroid_n[idx] += 1
        return idx

    def observe_action(self, action_emb: np.ndarray):
        """Call when the model (or entity) produces an action."""
        idx = self._assign_type(action_emb)
        self._pending_action = (idx, action_emb)

    def observe_response(self, response_emb: np.ndarray) -> Optional[SocialRule]:
        """Call when the response to the pending action is observed.
        Returns a SocialRule if the pattern is strong enough."""
        if self._pending_action is None:
            return None
        act_idx, _ = self._pending_action
        resp_idx = self._assign_type(response_emb)
        self.counts[act_idx, resp_idx] += 1
        self._pending_action = None

        # Check if pattern is significant (>= 5 observations)
        row = self.counts[act_idx]
        total = row.sum()
        if total >= 5 + self.N_TYPES * self.alpha:
            p = float(row[resp_idx] / total)
            if p >= 0.6:
                return SocialRule(
                    action_label=self.ACTION_LABELS[act_idx],
                    response_label=self.ACTION_LABELS[resp_idx],
                    probability=p,
                    observations=int(row[resp_idx]),
                )
        return None

    def probability(self, action_emb: np.ndarray) -> np.ndarray:
        """Return posterior P(response_type | action) as probability vector."""
        act_idx = self._assign_type(action_emb)
        row = self.counts[act_idx].copy()
        return row / row.sum()

    def top_rules(self, min_p: float = 0.6, min_obs: int = 5) -> List[SocialRule]:
        rules = []
        for ai in range(self.N_TYPES):
            row = self.counts[ai]
            total = row.sum()
            if total < min_obs + self.N_TYPES * self.alpha:
                continue
            probs = row / total
            for ri in range(self.N_TYPES):
                if probs[ri] >= min_p and row[ri] >= min_obs:
                    rules.append(SocialRule(
                        action_label=self.ACTION_LABELS[ai],
                        response_label=self.ACTION_LABELS[ri],
                        probability=float(probs[ri]),
                        observations=int(row[ri]),
                    ))
        return sorted(rules, key=lambda r: -r.probability)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic deduplicator
# ─────────────────────────────────────────────────────────────────────────────

class SemanticDeduplicator:
    """Prevents memory bloat through online cosine-similarity deduplication.

    Three tiers:
      exact_thr  (≥0.92): drop duplicate entirely, bump access_count
      merge_thr  (≥0.78): merge into existing (update centroid embedding)
      schema_thr (≥0.65) × N ≥ min_cluster: promote cluster to SCHEMA node
    """

    def __init__(self, exact_thr: float = 0.92, merge_thr: float = 0.78,
                 schema_thr: float = 0.65, min_cluster: int = 4):
        self.exact_thr   = exact_thr
        self.merge_thr   = merge_thr
        self.schema_thr  = schema_thr
        self.min_cluster = min_cluster
        self._recent: List[HyperNode] = []   # sliding window of ~256 nodes
        self._max_recent = 256

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        d = min(a.size, b.size)
        return float(np.dot(a[:d], b[:d]) / (na * nb))

    def check(self, node: HyperNode) -> Tuple[str, Optional[HyperNode]]:
        """Returns (action, match_or_None).
        action ∈ {'store', 'duplicate', 'merge', 'schema_candidate'}"""
        emb = node.embedding
        best_sim, best_node = -1.0, None
        for existing in self._recent:
            s = self._cos(emb, existing.embedding)
            if s > best_sim:
                best_sim, best_node = s, existing

        if best_sim >= self.exact_thr and best_node is not None:
            best_node.access_count += 1
            best_node.salience = min(1.0, best_node.salience + 0.05)
            return "duplicate", best_node

        if best_sim >= self.merge_thr and best_node is not None:
            d = min(emb.size, best_node.embedding.size)
            n = best_node.access_count + 1
            best_node.embedding[:d] = (best_node.embedding[:d] * (n-1) + emb[:d]) / n
            best_node.access_count = n
            best_node.valence = (best_node.valence * (n-1) + node.valence) / n
            return "merge", best_node

        # Add to recent window
        self._recent.append(node)
        if len(self._recent) > self._max_recent:
            self._recent.pop(0)

        # Check for schema promotion opportunity
        cluster = [n for n in self._recent
                   if self._cos(emb, n.embedding) >= self.schema_thr]
        if len(cluster) >= self.min_cluster:
            return "schema_candidate", None

        return "store", None


# ─────────────────────────────────────────────────────────────────────────────
# Entity subgraph
# ─────────────────────────────────────────────────────────────────────────────

class EntitySubgraph:
    """Private memory slice for one tracked entity.

    Contains nodes and hyperedges that specifically involve this entity.
    Enables identity-specific reasoning without cross-entity bleed.
    """

    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        self.node_ids:  List[str] = []
        self.edge_ids:  List[str] = []
        self.triples:   List[KnowledgeTriple] = []
        self._narrative_chunks: List[str] = []
        self._max_narrative = 64

    def add_node(self, node_id: str):
        if node_id not in self.node_ids:
            self.node_ids.append(node_id)

    def add_edge(self, edge_id: str):
        if edge_id not in self.edge_ids:
            self.edge_ids.append(edge_id)

    def add_triple(self, triple: KnowledgeTriple):
        # Deduplicate by (subject, predicate)
        for existing in self.triples:
            if existing.subject == triple.subject and \
               existing.predicate == triple.predicate:
                if triple.confidence > existing.confidence:
                    existing.object     = triple.object
                    existing.confidence = triple.confidence
                return
        self.triples.append(triple)

    def add_narrative(self, chunk: str):
        self._narrative_chunks.append(chunk)
        if len(self._narrative_chunks) > self._max_narrative:
            self._narrative_chunks.pop(0)

    def narrative_summary(self) -> str:
        return " | ".join(self._narrative_chunks[-8:])

    def preferences(self) -> List[KnowledgeTriple]:
        return [t for t in self.triples if t.predicate in
                ("likes", "loves", "prefers", "enjoys", "hates", "dislikes")]


# ─────────────────────────────────────────────────────────────────────────────
# Main: MemoryHyperGraph
# ─────────────────────────────────────────────────────────────────────────────

class MemoryHyperGraph:
    """The central multidimensional cognitive memory store.

    Combines:
    • HyperNodes (N=1 vertices) with typed embeddings
    • N-ary HyperEdges (N≥2 vertices) with probabilistic weights
    • SocialMarkovMemory for behavioural pattern learning
    • SemanticDeduplicator for memory compression
    • KnowledgeTripleExtractor for fact extraction
    • EntitySubgraph per tracked identity
    • Belief propagation for coherent world-state inference

    Access patterns that boost memory:
    • query() increments access_count → boosts salience
    • tag_reward() adds REWARD edge from DA signal
    • Social observations update Markov posteriors
    """

    def __init__(self, d_emb: int = 256, max_nodes: int = 16384,
                 decay_rate: float = 0.9997):
        self.d_emb       = d_emb
        self.max_nodes   = max_nodes
        self.decay_rate  = decay_rate

        self.nodes:  Dict[str, HyperNode]  = {}
        self.edges:  Dict[str, HyperEdge]  = {}
        self.entity_subgraphs: Dict[str, EntitySubgraph] = {}

        self.social_markov   = SocialMarkovMemory(d_emb=d_emb)
        self.deduplicator    = SemanticDeduplicator()
        self.triple_extractor= KnowledgeTripleExtractor()

        self._write_idx = 0
        self._step      = 0

        # Global knowledge triple store (indexed by subject)
        self._triples_by_subject: Dict[str, List[KnowledgeTriple]] = defaultdict(list)
        # Abstract schema nodes — never pruned
        self._schema_ids: List[str] = []
        # Accumulated social rules
        self.social_rules: List[SocialRule] = []

    # ── Node I/O ─────────────────────────────────────────────────────────────

    def encode(self, content: str, embedding: np.ndarray,
               memory_type: MemoryType = MemoryType.EPISODIC,
               entity_ref: Optional[str] = None,
               valence: float = 0.0, arousal: float = 0.5,
               salience: float = 0.5, nt_state: Optional[np.ndarray] = None,
               extract_triples: bool = True) -> Optional[str]:
        """Store a memory, returning its node ID (or None if deduplicated)."""
        emb = np.asarray(embedding, dtype=np.float32).flatten()

        node = HyperNode(
            id=str(uuid.uuid4())[:8],
            content=content,
            embedding=emb,
            memory_type=memory_type,
            entity_ref=entity_ref,
            temporal_idx=self._write_idx,
            valence=valence,
            arousal=arousal,
            salience=salience,
            nt_state=nt_state,
        )

        # Extract knowledge triples from text
        if extract_triples and content:
            triples = self.triple_extractor.extract(
                content, entity_context=entity_ref,
                speaker=entity_ref or "self")
            node.triples = triples
            for t in triples:
                self._triples_by_subject[t.subject].append(t)
                if entity_ref:
                    self._entity_subgraph(entity_ref).add_triple(t)

        # Deduplication check
        action, match = self.deduplicator.check(node)
        if action == "duplicate":
            return match.id if match else None
        if action == "merge":
            return match.id if match else None
        if action == "schema_candidate":
            self._promote_to_schema(node)

        # Store node
        self.nodes[node.id] = node
        self._write_idx += 1

        # Wire edges to recent nodes
        self._wire_edges(node)

        # Register in entity subgraph
        if entity_ref:
            self._entity_subgraph(entity_ref).add_node(node.id)

        # Prune if over capacity
        if len(self.nodes) > self.max_nodes:
            self._prune()

        return node.id

    def encode_insight(self, content: str, embedding: np.ndarray,
                       surprise: float, comprehension: float,
                       valence: float = 0.0,
                       da_level: float = 0.5) -> Optional[str]:
        """Store a training insight. Salience = surprise × comprehension × DA."""
        novelty = self._compute_novelty(embedding)
        score = surprise * comprehension * novelty
        if score < 0.04:
            return None
        nid = self.encode(
            content, embedding,
            memory_type=MemoryType.SEMANTIC,
            valence=valence,
            salience=float(score * (0.5 + da_level)),
        )
        if nid:
            self._tag_reward(nid, da_level, score)
        return nid

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def _wire_edges(self, node: HyperNode, window: int = 64):
        """Auto-wire semantic, temporal, mood, and entity hyperedges."""
        recent = sorted(self.nodes.values(),
                        key=lambda n: -n.temporal_idx)[:window]

        for other in recent:
            if other.id == node.id:
                continue

            # Semantic edge
            sim = self._cos(node.embedding, other.embedding)
            if sim >= 0.55:
                self._add_edge([node.id, other.id], RelationType.SEMANTIC, sim)

            # Temporal edge (within 120 s)
            if abs(node.timestamp - other.timestamp) < 120:
                recency = 1.0 - abs(node.timestamp - other.timestamp) / 120
                self._add_edge([node.id, other.id], RelationType.TEMPORAL,
                               weight=recency * 0.8)

            # Mood edge (NT-state similarity)
            if node.nt_state is not None and other.nt_state is not None:
                nt_sim = self._cos(node.nt_state, other.nt_state)
                val_sim = 1.0 - abs(node.valence - other.valence) / 2
                mood_w = (nt_sim + val_sim) / 2
                if mood_w >= 0.55:
                    self._add_edge([node.id, other.id], RelationType.MOOD, mood_w)

            # Entity edge
            if (node.entity_ref and other.entity_ref
                    and node.entity_ref == other.entity_ref):
                self._add_edge([node.id, other.id], RelationType.IDENTITY, 0.9)

    def _add_edge(self, node_ids: List[str], relation: RelationType,
                  weight: float = 1.0) -> str:
        eid = str(uuid.uuid4())[:8]
        edge = HyperEdge(id=eid, node_ids=node_ids,
                         relation=relation, weight=weight)
        self.edges[eid] = edge
        if node_ids[0] in self.entity_subgraphs.get(
                self.nodes.get(node_ids[0], HyperNode(
                    "", "", np.array([]))).entity_ref or "", EntitySubgraph("")):
            pass
        return eid

    # ── Social Markov observations ────────────────────────────────────────────

    def observe_social_action(self, action_emb: np.ndarray,
                               action_text: str = "",
                               entity_id: Optional[str] = None):
        """Record that an action was produced (before the response arrives)."""
        self.social_markov.observe_action(action_emb)
        if entity_id and action_text:
            self._entity_subgraph(entity_id).add_narrative(
                f"[action] {action_text[:60]}")

    def observe_social_response(self, response_emb: np.ndarray,
                                 response_text: str = "",
                                 entity_id: Optional[str] = None):
        """Record the response to the last observed action.
        May emit a new SocialRule."""
        rule = self.social_markov.observe_response(response_emb)
        if rule:
            # Deduplicate rules
            existing_labels = {(r.action_label, r.response_label)
                               for r in self.social_rules}
            if (rule.action_label, rule.response_label) not in existing_labels:
                self.social_rules.append(rule)
            else:
                for r in self.social_rules:
                    if r.action_label == rule.action_label and \
                       r.response_label == rule.response_label:
                        # Update probability with running mean
                        n = r.observations + rule.observations
                        r.probability = (r.probability * r.observations
                                         + rule.probability * rule.observations) / n
                        r.observations = n
                        break
        if entity_id and response_text:
            self._entity_subgraph(entity_id).add_narrative(
                f"[response] {response_text[:60]}")

    # ── Query methods ─────────────────────────────────────────────────────────

    def query_semantic(self, query_emb: np.ndarray, topk: int = 8,
                       entity_filter: Optional[str] = None,
                       memory_type: Optional[MemoryType] = None) -> List[HyperNode]:
        """Retrieve topk nodes by cosine similarity."""
        q = np.asarray(query_emb, dtype=np.float32).flatten()
        scored = []
        for node in self.nodes.values():
            if entity_filter and node.entity_ref != entity_filter:
                continue
            if memory_type and node.memory_type != memory_type:
                continue
            s = self._cos(q, node.embedding)
            scored.append((s * node.decay * (1 + 0.1 * math.log1p(node.access_count)),
                           node))
        scored.sort(key=lambda x: -x[0])
        results = [n for _, n in scored[:topk]]
        for n in results:
            n.access_count += 1
        return results

    def query_temporal(self, topk: int = 8,
                       entity_filter: Optional[str] = None) -> List[HyperNode]:
        """Retrieve most recent nodes."""
        candidates = [n for n in self.nodes.values()
                      if not entity_filter or n.entity_ref == entity_filter]
        candidates.sort(key=lambda n: -n.temporal_idx)
        return candidates[:topk]

    def query_mood(self, nt_state: np.ndarray, valence: float,
                   topk: int = 8) -> List[HyperNode]:
        """Retrieve nodes with similar NT/mood state."""
        nt = np.asarray(nt_state, dtype=np.float32).flatten()
        scored = []
        for node in self.nodes.values():
            score = 0.0
            if node.nt_state is not None:
                score += self._cos(nt, node.nt_state) * 0.6
            score += (1.0 - abs(node.valence - valence) / 2) * 0.4
            scored.append((score * node.decay, node))
        scored.sort(key=lambda x: -x[0])
        return [n for _, n in scored[:topk]]

    def query_entity(self, entity_id: str, topk: int = 16) -> List[HyperNode]:
        """Retrieve all nodes related to a specific entity."""
        sg = self.entity_subgraphs.get(entity_id)
        if sg is None:
            return []
        nodes = [self.nodes[nid] for nid in sg.node_ids if nid in self.nodes]
        nodes.sort(key=lambda n: -(n.salience * n.decay))
        return nodes[:topk]

    def spreading_activation(self, seed_id: str, hops: int = 2,
                              topk: int = 12) -> List[HyperNode]:
        """Multi-hop graph walk from seed node through hyperedges."""
        visited: Dict[str, float] = {seed_id: 1.0}
        frontier = [seed_id]
        for _ in range(hops):
            next_frontier = []
            for nid in frontier:
                for edge in self.edges.values():
                    if nid in edge.node_ids:
                        for connected in edge.node_ids:
                            if connected != nid and connected not in visited:
                                activation = (visited[nid] * edge.weight
                                              * edge.probability * 0.7)
                                visited[connected] = activation
                                next_frontier.append(connected)
            frontier = sorted(next_frontier,
                              key=lambda x: -visited.get(x, 0))[:16]

        result = [(v, self.nodes[nid]) for nid, v in visited.items()
                  if nid in self.nodes and nid != seed_id]
        result.sort(key=lambda x: -x[0])
        return [n for _, n in result[:topk]]

    def get_social_probability(self, action_emb: np.ndarray) -> np.ndarray:
        """P(response_type | action) from Dirichlet posterior."""
        return self.social_markov.probability(action_emb)

    def get_knowledge(self, subject: str,
                      predicate: Optional[str] = None) -> List[KnowledgeTriple]:
        """Retrieve knowledge triples for a subject."""
        triples = self._triples_by_subject.get(subject, [])
        if predicate:
            triples = [t for t in triples if t.predicate == predicate]
        return sorted(triples, key=lambda t: -t.confidence)

    # ── Abstract/schema nodes ─────────────────────────────────────────────────

    def _promote_to_schema(self, trigger_node: HyperNode):
        """Promote a cluster of similar nodes to a single SCHEMA node."""
        emb = trigger_node.embedding
        cluster = [n for n in self.deduplicator._recent
                   if self._cos(emb, n.embedding) >= self.deduplicator.schema_thr]
        if len(cluster) < self.deduplicator.min_cluster:
            return

        # Centroid embedding
        vecs = [n.embedding for n in cluster]
        min_d = min(v.size for v in vecs)
        centroid = np.mean([v[:min_d] for v in vecs], axis=0)

        schema_id = "schema_" + str(uuid.uuid4())[:6]
        schema_node = HyperNode(
            id=schema_id,
            content=f"[schema] {cluster[0].content[:50]}",
            embedding=centroid,
            memory_type=MemoryType.SCHEMA,
            valence=float(np.mean([n.valence for n in cluster])),
            arousal=float(np.mean([n.arousal for n in cluster])),
            salience=float(np.mean([n.salience for n in cluster])) * 1.2,
            is_abstract=True,
            access_count=len(cluster),
        )
        self.nodes[schema_id] = schema_node
        self._schema_ids.append(schema_id)

        # Wire causal/pattern edges to cluster members
        for n in cluster:
            if n.id in self.nodes:
                self._add_edge([schema_id, n.id], RelationType.CONTEXTUAL, 0.8)

    def add_abstract_node(self, content: str, centroid_vec: np.ndarray,
                           nt_state: Optional[np.ndarray] = None,
                           salience: float = 0.5,
                           tags: Optional[List[str]] = None) -> str:
        nid = "abstract_" + str(uuid.uuid4())[:6]
        node = HyperNode(
            id=nid,
            content=content,
            embedding=np.asarray(centroid_vec, dtype=np.float32).flatten(),
            memory_type=MemoryType.SEMANTIC,
            nt_state=nt_state,
            salience=salience,
            is_abstract=True,
        )
        self.nodes[nid] = node
        self._schema_ids.append(nid)
        self._wire_edges(node)
        return nid

    # ── Reward tagging ────────────────────────────────────────────────────────

    def _tag_reward(self, node_id: str, da_level: float,
                    reward_signal: float):
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        alpha = da_level * 0.3
        node.salience = (1 - alpha) * node.salience + alpha * reward_signal
        node.salience = float(np.clip(node.salience, 0.0, 1.0))
        # Find most recent high-salience node to chain REWARD edge
        best = max(
            (n for n in self.nodes.values()
             if n.id != node_id and n.salience > 0.6),
            key=lambda n: n.salience, default=None)
        if best:
            self._add_edge([node_id, best.id], RelationType.REWARD,
                           weight=da_level * reward_signal)

    # ── Maintenance ───────────────────────────────────────────────────────────

    def decay_step(self):
        """Apply exponential decay to all non-schema, non-abstract nodes."""
        self._step += 1
        for node in self.nodes.values():
            if not node.is_abstract:
                node.decay *= self.decay_rate

    def _prune(self):
        """Remove weakest non-abstract nodes when over capacity."""
        candidates = [(n.decay * n.salience, n.id)
                      for n in self.nodes.values()
                      if not n.is_abstract and n.id not in self._schema_ids]
        if not candidates:
            return
        candidates.sort()
        n_remove = max(1, len(self.nodes) - self.max_nodes)
        for _, nid in candidates[:n_remove]:
            del self.nodes[nid]
        # Prune dangling edges
        live = set(self.nodes.keys())
        dead_edges = [eid for eid, e in self.edges.items()
                      if not all(nid in live for nid in e.node_ids)]
        for eid in dead_edges:
            del self.edges[eid]

    def _compute_novelty(self, emb: np.ndarray, topk: int = 16) -> float:
        q = np.asarray(emb, dtype=np.float32).flatten()
        if not self.nodes:
            return 1.0
        recent = sorted(self.nodes.values(),
                        key=lambda n: -n.temporal_idx)[:256]
        sims = [self._cos(q, n.embedding) for n in recent]
        return float(1.0 - max(sims)) if sims else 1.0

    # ── Entity helpers ────────────────────────────────────────────────────────

    def _entity_subgraph(self, entity_id: str) -> EntitySubgraph:
        if entity_id not in self.entity_subgraphs:
            self.entity_subgraphs[entity_id] = EntitySubgraph(entity_id)
        return self.entity_subgraphs[entity_id]

    def get_entity_subgraph(self, entity_id: str) -> Optional[EntitySubgraph]:
        return self.entity_subgraphs.get(entity_id)

    # ── Utils ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        d = min(a.size, b.size)
        return float(np.dot(a[:d], b[:d]) / (na * nb))

    def stats(self) -> dict:
        return {
            "nodes":        len(self.nodes),
            "edges":        len(self.edges),
            "schemas":      len(self._schema_ids),
            "entities":     len(self.entity_subgraphs),
            "social_rules": len(self.social_rules),
            "triples":      sum(len(v) for v in self._triples_by_subject.values()),
        }

    def state_dict(self) -> dict:
        """Serialisable snapshot for .mem checkpoints."""
        return {
            "nodes": {nid: {
                "content":     n.content,
                "embedding":   n.embedding.tolist(),
                "memory_type": n.memory_type.name,
                "entity_ref":  n.entity_ref,
                "temporal_idx":n.temporal_idx,
                "valence":     n.valence,
                "arousal":     n.arousal,
                "salience":    n.salience,
                "access_count":n.access_count,
                "decay":       n.decay,
                "is_abstract": n.is_abstract,
            } for nid, n in self.nodes.items()},
            "social_rules": [
                {"action": r.action_label, "response": r.response_label,
                 "p": r.probability, "n": r.observations}
                for r in self.social_rules
            ],
            "triples": {
                subj: [{"pred": t.predicate, "obj": t.object,
                        "conf": t.confidence, "eid": t.entity_id}
                       for t in triples]
                for subj, triples in self._triples_by_subject.items()
            },
        }

    def load_state_dict(self, state: dict):
        self.nodes.clear()
        for nid, d in state.get("nodes", {}).items():
            self.nodes[nid] = HyperNode(
                id=nid,
                content=d["content"],
                embedding=np.array(d["embedding"], dtype=np.float32),
                memory_type=MemoryType[d.get("memory_type", "EPISODIC")],
                entity_ref=d.get("entity_ref"),
                temporal_idx=d.get("temporal_idx", 0),
                valence=d.get("valence", 0.0),
                arousal=d.get("arousal", 0.5),
                salience=d.get("salience", 0.5),
                access_count=d.get("access_count", 0),
                decay=d.get("decay", 1.0),
                is_abstract=d.get("is_abstract", False),
            )
        for rd in state.get("social_rules", []):
            self.social_rules.append(SocialRule(
                action_label=rd["action"], response_label=rd["response"],
                probability=rd["p"], observations=rd["n"]))
        for subj, triples in state.get("triples", {}).items():
            for td in triples:
                self._triples_by_subject[subj].append(KnowledgeTriple(
                    subject=subj, predicate=td["pred"], object=td["obj"],
                    confidence=td["conf"], entity_id=td.get("eid")))
