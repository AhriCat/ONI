"""
modules/memory/gaussian_splat_memory.py
=======================================
3D Gaussian Splatting scene memory for ONI.

Architecture role
-----------------
  GaussianSplatScene  — differentiable 3DGS scene (N learnable Gaussian primitives)
  SceneEncoder        — Transformer over Gaussian params → latent [hidden_dim]
  SceneBuilder        — fits a GaussianSplatScene from vision + depth/xyz features
  GaussianSplatMemory — key-indexed scene store; encodes any scene to a context
                        vector that feeds directly into WorldModel.memory_context
                        and/or SpatialMemory room data

Gaussian parameterisation (per primitive)
------------------------------------------
  mean           (3,)   world-space centre
  log_scale      (3,)   log of per-axis scale  →  exp() gives positive scales
  rotation       (4,)   unit quaternion (w, x, y, z)
  sh_dc          (3,)   zero-degree spherical harmonic (base RGB colour)
  opacity_logit  (1,)   pre-sigmoid opacity

Packed:  (N, 14)  — used by encoder and for serialisation

Geometry helpers
----------------
  _quat_to_rotation_matrix  quaternion → 3×3 rotation matrix (batched)
  _build_covariance          rotation + scale → 3×3 Σ (batched)
  _gaussian_weight           Mahalanobis-distance soft alpha for a set of query points

Integration points
------------------
  world_modeler.OniAutoregressiveWorldModel
      scene_memory: Optional[GaussianSplatMemory]
      → encode_scene() produces memory_context fed to latent_ops.encode_to_latent()

  spatial_memory.SpatialMemoryModule
      → load_splat_scene() / get_current_splat_scene()
      → rooms store GaussianSplatScene objects alongside any other room data
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _quat_to_rotation_matrix(q: Tensor) -> Tensor:
    """
    Convert unit quaternions to rotation matrices.

    Args:
        q: (N, 4) quaternions in (w, x, y, z) order

    Returns:
        R: (N, 3, 3) rotation matrices
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)

    return R   # (N, 3, 3)


def _build_covariance(log_scale: Tensor, rotation: Tensor) -> Tensor:
    """
    Build 3D covariance matrices Σ = R @ diag(s²) @ R^T.

    Args:
        log_scale: (N, 3)
        rotation:  (N, 4) quaternions (w, x, y, z)

    Returns:
        cov: (N, 3, 3)
    """
    s = torch.exp(log_scale)           # (N, 3) positive scales
    S = torch.diag_embed(s)            # (N, 3, 3) diagonal
    R = _quat_to_rotation_matrix(rotation)   # (N, 3, 3)
    cov = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
    return cov   # (N, 3, 3)


def _gaussian_weight(
    means: Tensor,          # (N, 3)
    log_scale: Tensor,      # (N, 3)
    rotation: Tensor,       # (N, 4)
    opacity_logits: Tensor, # (N,)
    query_xyz: Tensor,      # (Q, 3)
) -> Tensor:
    """
    Compute soft alpha contributions of N Gaussians to Q query points.

    Uses the Mahalanobis distance:
        α_i(p) = opacity_i * exp(-0.5 * (p - μ_i)^T Σ_i^{-1} (p - μ_i))

    Args:
        means, log_scale, rotation, opacity_logits: Gaussian params
        query_xyz: (Q, 3) world-space query positions

    Returns:
        weights: (Q, N)  (unnormalised alpha contributions)
    """
    N = means.shape[0]
    Q = query_xyz.shape[0]

    cov = _build_covariance(log_scale, rotation)          # (N, 3, 3)
    # Add small diagonal to ensure invertibility
    cov = cov + torch.eye(3, device=cov.device).unsqueeze(0) * 1e-6

    try:
        cov_inv = torch.linalg.inv(cov)                   # (N, 3, 3)
    except Exception:
        cov_inv = torch.eye(3, device=cov.device).unsqueeze(0).expand(N, -1, -1)

    delta = query_xyz.unsqueeze(1) - means.unsqueeze(0)   # (Q, N, 3)
    # Mahalanobis: delta^T Σ^{-1} delta
    maha = torch.einsum('qni,nij,qnj->qn', delta, cov_inv, delta)   # (Q, N)

    opacity = torch.sigmoid(opacity_logits)               # (N,)
    weights = opacity.unsqueeze(0) * torch.exp(-0.5 * maha)         # (Q, N)
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# GaussianSplatScene
# ──────────────────────────────────────────────────────────────────────────────

class GaussianSplatScene(nn.Module):
    """
    Learnable 3D Gaussian Splatting scene.

    Stores N Gaussian primitives as (optionally trainable) parameters.
    Supports feature-weighted queries at arbitrary 3D points and compact
    serialisation via pack()/unpack().
    """

    PARAM_DIM = 14   # 3 mean + 3 log_scale + 4 rotation + 3 sh_dc + 1 opacity

    def __init__(self, n_gaussians: int = 2048, device: Optional[torch.device] = None):
        super().__init__()
        self.n = n_gaussians

        # Gaussian parameters
        self.means          = nn.Parameter(torch.zeros(n_gaussians, 3))
        self.log_scales     = nn.Parameter(torch.full((n_gaussians, 3), -2.0))  # ~exp(-2)≈0.14 world-units
        self.rotations      = nn.Parameter(torch.zeros(n_gaussians, 4))         # identity quats
        self.sh_dc          = nn.Parameter(torch.zeros(n_gaussians, 3))         # RGB base colour
        self.opacity_logits = nn.Parameter(torch.zeros(n_gaussians))            # sigmoid → 0.5

        # identity quaternions: w=1, x=y=z=0
        nn.init.constant_(self.rotations[:, 0], 1.0)

        if device is not None:
            self.to(device)

    @property
    def scales(self) -> Tensor:
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> Tensor:
        return torch.sigmoid(self.opacity_logits)

    def pack(self) -> Tensor:
        """Return all Gaussian params as (N, 14) tensor."""
        return torch.cat([
            self.means,
            self.log_scales,
            F.normalize(self.rotations, dim=-1),
            self.sh_dc,
            self.opacity_logits.unsqueeze(-1),
        ], dim=-1)   # (N, 14)

    def unpack_(self, packed: Tensor) -> None:
        """Load parameters from (N, 14) packed tensor (in-place)."""
        assert packed.shape == (self.n, self.PARAM_DIM), \
            f"Expected ({self.n}, {self.PARAM_DIM}), got {tuple(packed.shape)}"
        with torch.no_grad():
            self.means.copy_(packed[:, :3])
            self.log_scales.copy_(packed[:, 3:6])
            self.rotations.copy_(F.normalize(packed[:, 6:10], dim=-1))
            self.sh_dc.copy_(packed[:, 10:13])
            self.opacity_logits.copy_(packed[:, 13])

    def query_features(self, query_xyz: Tensor, gaussian_feats: Optional[Tensor] = None) -> Tensor:
        """
        Retrieve per-query weighted feature vectors by splatting Gaussian colours/features.

        Args:
            query_xyz:     (Q, 3)  world-space query positions
            gaussian_feats: (N, F) optional per-Gaussian feature vectors
                            Defaults to sh_dc (N, 3) if not supplied

        Returns:
            out: (Q, F)  alpha-weighted feature sum at each query point
        """
        if gaussian_feats is None:
            gaussian_feats = self.sh_dc   # (N, 3)

        weights = _gaussian_weight(
            self.means, self.log_scales, self.rotations,
            self.opacity_logits, query_xyz
        )  # (Q, N)

        # Normalise weights so they sum to 1 across Gaussians (like softmax render)
        weights_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # (Q, N)
        return torch.einsum('qn,nf->qf', weights_norm, gaussian_feats)       # (Q, F)

    def covariance(self) -> Tensor:
        """Return (N, 3, 3) covariance matrices."""
        return _build_covariance(self.log_scales, self.rotations)

    def extra_repr(self) -> str:
        return f"n_gaussians={self.n}"


# ──────────────────────────────────────────────────────────────────────────────
# SceneEncoder
# ──────────────────────────────────────────────────────────────────────────────

class SceneEncoder(nn.Module):
    """
    Encodes a GaussianSplatScene to a fixed-length latent vector.

    Architecture:
      packed (N, 14)  →  per-Gaussian embedding  →  Transformer self-attention
      →  mean pool  →  MLP  →  (hidden_dim,)

    The output is suitable as memory_context in OniAutoregressiveWorldModel.
    """

    def __init__(self, hidden_dim: int = 896, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed    = nn.Linear(GaussianSplatScene.PARAM_DIM, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool_proj   = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, scene: GaussianSplatScene) -> Tensor:
        """
        Args:
            scene: GaussianSplatScene

        Returns:
            context: (1, hidden_dim)  latent scene summary
        """
        packed = scene.pack().unsqueeze(0)          # (1, N, 14)
        x = self.embed(packed)                      # (1, N, D)
        x = self.transformer(x)                     # (1, N, D)
        x = x.mean(dim=1)                           # (1, D)  — mean pool over Gaussians
        return self.pool_proj(x)                    # (1, D)


# ──────────────────────────────────────────────────────────────────────────────
# SceneBuilder
# ──────────────────────────────────────────────────────────────────────────────

class SceneBuilder(nn.Module):
    """
    Builds or updates a GaussianSplatScene from vision feature observations.

    Two paths:
      init_from_features  — create a fresh scene from a batch of RGB+XYZ features
      update_scene        — fold new observations into an existing scene

    Vision features are expected in the same space as OniAutoregressiveWorldModel's
    vision dimension (default 512 from MPAD). Depth/XYZ are optional; if absent
    a learned MLP predicts Gaussian means from the features alone.
    """

    def __init__(
        self,
        vision_dim: int  = 512,
        hidden_dim: int  = 896,
        n_gaussians: int = 2048,
    ):
        super().__init__()
        self.n = n_gaussians

        # Predicts per-feature "slot weights" over N Gaussian slots
        self.slot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True
        )

        # Projects vision features to the attention hidden dim
        self.feat_proj = nn.Linear(vision_dim, hidden_dim)

        # Slot queries — N learnable Gaussian "seeds"
        self.slot_queries = nn.Parameter(torch.randn(n_gaussians, hidden_dim) * 0.02)

        # Per-slot decoders → Gaussian parameters
        self.mean_dec      = nn.Linear(hidden_dim, 3)
        self.log_scale_dec = nn.Linear(hidden_dim, 3)
        self.rot_dec       = nn.Linear(hidden_dim, 4)
        self.color_dec     = nn.Linear(hidden_dim, 3)
        self.opacity_dec   = nn.Linear(hidden_dim, 1)

        # For updating: blend weight between old and new (learned per-slot)
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Mean initialisation: zero → small-variance scene
        nn.init.constant_(self.mean_dec.bias, 0.0)
        nn.init.constant_(self.log_scale_dec.bias, -2.0)

        # Quaternion → identity init
        nn.init.zeros_(self.rot_dec.weight)
        nn.init.constant_(self.rot_dec.bias, 0.0)
        self.rot_dec.bias.data[0] = 1.0   # w=1

    def _decode_slots(self, slots: Tensor) -> Tuple[Tensor, ...]:
        """
        Args:
            slots: (N, D)

        Returns:
            means:         (N, 3)
            log_scales:    (N, 3)
            rotations:     (N, 4)  normalised quaternions
            sh_dc:         (N, 3)
            opacity_logit: (N,)
        """
        means      = self.mean_dec(slots)
        log_scales = self.log_scale_dec(slots)
        rotations  = F.normalize(self.rot_dec(slots), dim=-1)
        sh_dc      = self.color_dec(slots)
        opacity    = self.opacity_dec(slots).squeeze(-1)
        return means, log_scales, rotations, sh_dc, opacity

    def _attend_slots(self, vision_features: Tensor) -> Tensor:
        """
        Cross-attend slot queries to vision features.

        Args:
            vision_features: (B, T, vision_dim)  —  B=1 typically at inference

        Returns:
            slots: (N, D)   aggregated over batch
        """
        B, T, _ = vision_features.shape
        keys = self.feat_proj(vision_features)        # (B, T, D)
        queries = self.slot_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
        slots, _ = self.slot_attn(queries, keys, keys)               # (B, N, D)
        return slots.mean(0)   # (N, D)  — collapse batch

    @torch.no_grad()
    def init_from_features(
        self,
        vision_features: Tensor,                    # (B, T, vision_dim)
        xyz: Optional[Tensor] = None,               # (B, T, 3) optional xyz positions
        device: Optional[torch.device] = None,
    ) -> GaussianSplatScene:
        """
        Create a new GaussianSplatScene from vision observations.

        If xyz is supplied, Gaussian means are initialised from the k-nearest
        observation positions rather than the learned MLP output.
        """
        scene = GaussianSplatScene(self.n, device=device or vision_features.device)

        slots = self._attend_slots(vision_features)                      # (N, D)
        means, log_scales, rotations, sh_dc, opacity_logit = self._decode_slots(slots)

        if xyz is not None:
            # Seed means directly from observed XYZ via soft assignment
            flat_xyz = xyz.reshape(-1, 3)   # (B*T, 3)
            # Assign each Gaussian to the closest observation point
            dists = torch.cdist(means, flat_xyz)   # (N, B*T)
            nearest = dists.argmin(dim=-1)          # (N,)
            means = flat_xyz[nearest]               # (N, 3)

        with torch.no_grad():
            scene.means.copy_(means)
            scene.log_scales.copy_(log_scales)
            scene.rotations.copy_(rotations)
            scene.sh_dc.copy_(sh_dc)
            scene.opacity_logits.copy_(opacity_logit)

        return scene

    def update_scene(
        self,
        scene: GaussianSplatScene,
        vision_features: Tensor,                    # (B, T, vision_dim)
        xyz: Optional[Tensor] = None,               # (B, T, 3)
    ) -> GaussianSplatScene:
        """
        Fold new observations into an existing scene (no-gradient, in-place update).

        Uses a learned gate to blend old Gaussian parameters with new estimates.
        """
        slots     = self._attend_slots(vision_features)                  # (N, D)
        new_means, new_log_s, new_rot, new_sh, new_op = self._decode_slots(slots)

        if xyz is not None:
            flat_xyz = xyz.reshape(-1, 3)
            dists    = torch.cdist(new_means, flat_xyz)
            nearest  = dists.argmin(dim=-1)
            new_means = flat_xyz[nearest]

        # Gate: how much to blend in new info vs. keep old
        old_embed = scene.pack()[:, :self.slot_queries.shape[-1]] \
            if scene.pack().shape[-1] >= self.slot_queries.shape[-1] \
            else F.pad(scene.pack(), (0, self.slot_queries.shape[-1] - scene.pack().shape[-1]))

        gate_input = torch.cat([slots, old_embed[:, :slots.shape[-1]]], dim=-1)
        gate = self.update_gate(gate_input)   # (N, 1)

        with torch.no_grad():
            scene.means.copy_((1 - gate) * scene.means + gate * new_means)
            scene.log_scales.copy_((1 - gate) * scene.log_scales + gate * new_log_s)
            scene.rotations.copy_(F.normalize(
                (1 - gate) * scene.rotations + gate * new_rot, dim=-1
            ))
            scene.sh_dc.copy_((1 - gate) * scene.sh_dc + gate * new_sh)
            scene.opacity_logits.copy_((1 - gate) * scene.opacity_logits + gate * new_op.squeeze(-1))

        return scene


# ──────────────────────────────────────────────────────────────────────────────
# GaussianSplatMemory
# ──────────────────────────────────────────────────────────────────────────────

class GaussianSplatMemory(nn.Module):
    """
    Key-indexed store of GaussianSplatScene objects.

    Intended use:
      - Called by OniAutoregressiveWorldModel to provide scene context
      - Used by SpatialMemoryModule to back each room with a 3DGS scene

    Scene IDs are arbitrary hashable keys (e.g. room coords, episode IDs, str).
    """

    def __init__(
        self,
        hidden_dim: int  = 896,
        vision_dim: int  = 512,
        n_gaussians: int = 2048,
        max_scenes: int  = 64,
    ):
        super().__init__()
        self.max_scenes  = max_scenes
        self.n_gaussians = n_gaussians

        self.encoder = SceneEncoder(hidden_dim=hidden_dim)
        self.builder = SceneBuilder(
            vision_dim=vision_dim,
            hidden_dim=hidden_dim,
            n_gaussians=n_gaussians,
        )

        # Scene store: key → GaussianSplatScene (not nn.Module children,
        # stored as a plain dict so they don't appear in model parameters)
        self._scenes: Dict[str, GaussianSplatScene] = {}
        self._access_order: List[str] = []   # LRU

    # ── internal LRU bookkeeping ──────────────────────────────────────────────

    def _touch(self, key: str) -> None:
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        if len(self._access_order) > self.max_scenes:
            evict = self._access_order.pop(0)
            self._scenes.pop(evict, None)

    # ── public API ────────────────────────────────────────────────────────────

    def has_scene(self, scene_id) -> bool:
        return str(scene_id) in self._scenes

    def store_scene(self, scene_id, scene: GaussianSplatScene) -> None:
        """Directly store a pre-built scene (e.g. loaded from disk)."""
        key = str(scene_id)
        self._scenes[key] = scene
        self._touch(key)

    def build_scene(
        self,
        scene_id,
        vision_features: Tensor,
        xyz: Optional[Tensor] = None,
    ) -> GaussianSplatScene:
        """
        Build a new scene from vision observations and store it.

        Args:
            scene_id:        hashable key (room coord, string, …)
            vision_features: (B, T, vision_dim)
            xyz:             (B, T, 3) optional world-space positions

        Returns:
            The newly created GaussianSplatScene.
        """
        key   = str(scene_id)
        scene = self.builder.init_from_features(vision_features, xyz)
        self._scenes[key] = scene
        self._touch(key)
        return scene

    def update_scene(
        self,
        scene_id,
        vision_features: Tensor,
        xyz: Optional[Tensor] = None,
    ) -> GaussianSplatScene:
        """
        Update an existing scene with new observations; build it if absent.

        Args:
            scene_id:        key identifying the scene
            vision_features: (B, T, vision_dim)
            xyz:             (B, T, 3) optional

        Returns:
            The updated (or newly built) GaussianSplatScene.
        """
        key = str(scene_id)
        if key not in self._scenes:
            return self.build_scene(scene_id, vision_features, xyz)
        scene = self._scenes[key]
        scene = self.builder.update_scene(scene, vision_features, xyz)
        self._touch(key)
        return scene

    def encode_scene(self, scene_id) -> Optional[Tensor]:
        """
        Encode a stored scene to a latent context vector.

        Args:
            scene_id: key of the scene to encode

        Returns:
            context: (1, hidden_dim) ready for WorldModel.memory_context,
                     or None if scene_id is unknown
        """
        key = str(scene_id)
        if key not in self._scenes:
            return None
        self._touch(key)
        return self.encoder(self._scenes[key])   # (1, D)

    def retrieve_scene(self, scene_id) -> Optional[GaussianSplatScene]:
        """Return the raw GaussianSplatScene or None."""
        key = str(scene_id)
        if key not in self._scenes:
            return None
        self._touch(key)
        return self._scenes[key]

    def query_scene(
        self,
        scene_id,
        query_xyz: Tensor,
        gaussian_feats: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Splat-query a stored scene at arbitrary 3D positions.

        Args:
            scene_id:      key of the scene
            query_xyz:     (Q, 3)  world-space query points
            gaussian_feats: (N, F) optional per-Gaussian feature override

        Returns:
            features: (Q, F) alpha-weighted feature sum, or None if not found
        """
        scene = self.retrieve_scene(scene_id)
        if scene is None:
            return None
        return scene.query_features(query_xyz, gaussian_feats)

    def scene_ids(self) -> List[str]:
        return list(self._scenes.keys())

    def __len__(self) -> int:
        return len(self._scenes)

    def extra_repr(self) -> str:
        return (
            f"max_scenes={self.max_scenes}, "
            f"n_gaussians={self.n_gaussians}, "
            f"stored={len(self._scenes)}"
        )
