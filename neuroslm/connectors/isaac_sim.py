# -*- coding: utf-8 -*-
"""Isaac Sim sensory bridge (architecture.md §15) — pumps a simulated
environment's camera / articulation / (optional) audio streams through
the §15 cortices (:mod:`neuroslm.sensory.cortices`) into a
:class:`~neuroslm.cognition.runtime.CognitiveRuntime`'s
``observe_sensory`` SENSE path. Percepts reach the mind as latent
embeddings end to end — never as text captions (module docstring of
:mod:`neuroslm.sensory.cortices` has the full design note).

Two pieces:

``OmniverseIsaacSimClient``
    The real bridge to a running Isaac Sim process, via its
    ``isaacsim`` Python API (``isaacsim.sensors.camera.Camera`` for
    RGB frames, ``isaacsim.core.prims.Articulation`` for joint
    kinematics/forces). That API is ONLY importable from inside a
    running Isaac Sim process (standalone script or extension) — the
    import is guarded so the rest of ``neuroslm`` stays importable
    without Isaac Sim installed, and raises a clear error at
    construction time rather than failing silently.

    Isaac Sim has no general-purpose microphone/soundscape simulation
    (as of Isaac Sim 6.0 its only acoustic sensor is an ultrasonic
    ranging array, not audio) — ``audio_source`` is an optional
    caller-supplied callback (a real microphone, a ROS2 audio topic,
    synthesized contact sounds from collision events) so the acoustic
    modality is honestly absent by default rather than faked.

``SensoryBridge``
    Dependency-injected against ANY client exposing ``get_frame()`` /
    ``get_joint_state()`` / optionally ``get_audio_chunk()`` — tests
    inject a deterministic fake (the same pattern
    ``build_runtime_from_hf_lm`` uses for its LM); production wires
    ``OmniverseIsaacSimClient``. One ``.pump()`` call is one polling
    cycle across every modality the client provides.
"""
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from neuroslm.sensory.cortices import (
    AcousticCortex,
    ProprioceptiveCortex,
    VisualCortex,
)


class OmniverseIsaacSimClient:
    """Real Isaac Sim bridge. Must be constructed from code running
    INSIDE an Isaac Sim process — the ``isaacsim`` package is part of
    the application, not a standalone pip dependency this repo can
    vendor. See module docstring for the audio caveat."""

    def __init__(self, camera_prim_path: str, articulation_prim_path: str,
                resolution: Tuple[int, int] = (224, 224),
                audio_source: Optional[
                    Callable[[], Optional[Sequence[float]]]] = None):
        try:
            from isaacsim.sensors.camera import Camera
            from isaacsim.core.prims import Articulation
        except ImportError as exc:
            raise RuntimeError(
                "OmniverseIsaacSimClient requires the `isaacsim` Python "
                "API, only importable from inside a running Isaac Sim "
                "process (standalone script or extension). Launch Isaac "
                "Sim and run this from within it, or inject a fake "
                "client (SensoryBridge accepts any object exposing "
                "get_frame()/get_joint_state()/get_audio_chunk()) for "
                "local development and tests."
            ) from exc
        self._camera = Camera(prim_path=camera_prim_path,
                              resolution=resolution)
        self._camera.initialize()
        self._articulation = Articulation(
            prim_paths_expr=articulation_prim_path)
        self._articulation.initialize()
        self._audio_source = audio_source

    def get_frame(self):
        """(H, W, 4) RGBA ndarray, or None before the first render."""
        return self._camera.get_rgba()

    def get_joint_state(self) -> Dict[str, Sequence[float]]:
        positions = self._articulation.get_joint_positions()
        velocities = self._articulation.get_joint_velocities()
        forces = self._articulation.get_measured_joint_forces()
        # Articulation views batch over N prims matched by
        # prim_paths_expr; a single-robot bridge reads row 0.
        return {
            "positions": list(positions[0]),
            "velocities": list(velocities[0]),
            "forces": list(forces[0]) if forces is not None else [],
        }

    def get_audio_chunk(self) -> Optional[Sequence[float]]:
        return self._audio_source() if self._audio_source else None


class SensoryBridge:
    """Pumps one client's sensor streams through the §15 cortices into
    a CognitiveRuntime's SENSE path. Modalities are independently
    optional — a client returning None for a given stream this cycle
    (or one that raises, e.g. a transient sensor glitch) simply
    contributes nothing that cycle rather than crashing the pump; the
    failure is surfaced via ``on_error`` (default: a stderr line —
    visible, not hidden) instead of being silently swallowed."""

    def __init__(self, runtime: Any, client: Any,
                visual_cortex: Optional[Callable] = None,
                acoustic_cortex: Optional[Callable] = None,
                proprioceptive_cortex: Optional[Callable] = None,
                on_error: Optional[Callable[[str, Exception], None]] = None):
        self.runtime = runtime
        self.client = client
        dim = runtime.embed_dim()
        self.visual = visual_cortex or VisualCortex(output_dim=dim)
        self.acoustic = acoustic_cortex or AcousticCortex(output_dim=dim)
        self.proprioceptive = (proprioceptive_cortex
                               or ProprioceptiveCortex(output_dim=dim))
        self._on_error = on_error or self._default_on_error

    @staticmethod
    def _default_on_error(modality: str, exc: Exception) -> None:
        print(f"[isaac_sim] {modality} sensor read failed: {exc}",
              file=sys.stderr)

    def _read(self, modality: str, fn: Optional[Callable]):
        if fn is None:
            return None
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — one bad sensor must
            # not kill the mind's tick loop; surfaced, not hidden.
            self._on_error(modality, exc)
            return None

    def pump(self) -> Dict[str, bool]:
        """One polling cycle. Returns ``{modality: attended}`` for
        every modality the client actually provided data for this
        cycle — ``attended`` is ``observe_sensory``'s own return value
        (False means habituated/not novel enough, not "failed")."""
        attended: Dict[str, bool] = {}

        frame = self._read("visual", getattr(self.client, "get_frame", None))
        if frame is not None:
            vec = self.visual(frame)
            attended["visual"] = self.runtime.observe_sensory(
                "visual", vec, source="isaac_sim")

        joints = self._read("proprioceptive",
                            getattr(self.client, "get_joint_state", None))
        if joints is not None:
            vec = self.proprioceptive(joints)
            attended["proprioceptive"] = self.runtime.observe_sensory(
                "proprioceptive", vec, source="isaac_sim")

        audio = self._read("acoustic",
                           getattr(self.client, "get_audio_chunk", None))
        if audio is not None:
            vec = self.acoustic(audio)
            attended["acoustic"] = self.runtime.observe_sensory(
                "acoustic", vec, source="isaac_sim")

        return attended
