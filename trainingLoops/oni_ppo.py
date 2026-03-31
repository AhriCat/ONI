"""
trainingLoops/oni_ppo.py
=========================
ONIPPOTrainer — unified PPO / GRPO / IRL / flow-matching trainer for ONI.

Modes
-----
motor          PPO with GAE.  Actor = MPAD 'robotics' process (generates
               continuous actions).  Critic = lightweight MLP value head.
               Plugs into any OpenAI-Gym / Gymnasium compatible environment.

text_diffuser  Supervised flow-matching on (x0, x1) text embedding pairs,
               optionally weighted by an external reward signal.  Trains
               oni.imagination_diffuser in-place.

text_lm        GRPO (Group Relative Policy Optimization) on oni.nlp_module.
               No critic needed — group-relative advantages from a reward fn.
               Defaults to a simple self-reward (mean log-prob of generation).

irl            GAIL-style inverse RL.  A discriminator learns to separate
               expert demos from policy rollouts; its log-odds becomes the
               live reward signal fed into motor-mode PPO.

Live loop
---------
  trainer = ONIPPOTrainer(oni_model, mode='motor')
  for ep in range(1000):
      stats = trainer.loop(env)         # one episode of rollout + update
      oni_model.observe_outcome(stats['mean_return'])   # close the loop

Design
------
  - Takes the whole ONI model; uses hasattr for every component.
  - Fallbacks are functional (real linear maps / MLPs) but minimal.
  - No gym dependency for non-motor modes.
  - All optimisers are created internally; callers can inject their own via
    the `optimizers` kwarg dict.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
    _GYM = True
except ImportError:
    try:
        import gym  # type: ignore[no-redef]
        _GYM = True
    except ImportError:
        _GYM = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _CriticHead(nn.Module):
    """Value function V(s) → scalar.  Two hidden layers."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H) → (B,) values."""
        return self.net(x).squeeze(-1)


class _IRLDiscriminator(nn.Module):
    """
    GAIL discriminator: D(s, a) → prob(expert).
    Reward signal: r = log D(s,a) — log(1 − D(s,a))  [log-odds].
    """

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns logit (B,); sigmoid → P(expert)."""
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)

    def reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Log-odds reward (B,); no gradient through here during actor update."""
        with torch.no_grad():
            logit = self(state, action)
            return logit - F.softplus(logit) - F.softplus(-logit)  # log D - log(1-D)

    def loss(
        self,
        expert_s: torch.Tensor,
        expert_a: torch.Tensor,
        policy_s: torch.Tensor,
        policy_a: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy: expert → 1, policy → 0."""
        logit_e = self(expert_s, expert_a)
        logit_p = self(policy_s, policy_a)
        loss_e  = F.binary_cross_entropy_with_logits(logit_e, torch.ones_like(logit_e))
        loss_p  = F.binary_cross_entropy_with_logits(logit_p, torch.zeros_like(logit_p))
        return (loss_e + loss_p) * 0.5


class _RolloutBuffer:
    """
    Fixed-capacity ring buffer for PPO on-policy rollouts.
    Stores (state_emb, action, log_prob, reward, value, done).
    """

    def __init__(self, capacity: int, hidden_dim: int, action_dim: int, device: torch.device):
        self.cap    = capacity
        self.hd     = hidden_dim
        self.ad     = action_dim
        self.device = device
        self._reset()

    def _reset(self):
        self.states   = torch.zeros(self.cap, self.hd,  device=self.device)
        self.actions  = torch.zeros(self.cap, self.ad,  device=self.device)
        self.log_probs= torch.zeros(self.cap,           device=self.device)
        self.rewards  = torch.zeros(self.cap,           device=self.device)
        self.values   = torch.zeros(self.cap,           device=self.device)
        self.dones    = torch.zeros(self.cap,           device=self.device)
        self.ptr = 0
        self.full = False

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        i = self.ptr % self.cap
        self.states[i]    = state.detach().squeeze()
        self.actions[i]   = action.detach().squeeze()
        self.log_probs[i] = float(log_prob)
        self.rewards[i]   = float(reward)
        self.values[i]    = float(value)
        self.dones[i]     = float(done)
        self.ptr += 1
        if self.ptr >= self.cap:
            self.full = True

    def __len__(self) -> int:
        return self.cap if self.full else self.ptr

    def compute_advantages(
        self, gamma: float = 0.99, lam: float = 0.95, last_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (advantages, returns) both (N,) tensors via GAE."""
        n = len(self)
        adv    = torch.zeros(n, device=self.device)
        ret    = torch.zeros(n, device=self.device)
        gae    = 0.0
        next_v = last_value
        for t in reversed(range(n)):
            mask  = 1.0 - float(self.dones[t])
            delta = float(self.rewards[t]) + gamma * next_v * mask - float(self.values[t])
            gae   = delta + gamma * lam * mask * gae
            adv[t] = gae
            next_v = float(self.values[t])
        ret = adv + self.values[:n]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def batches(self, batch_size: int):
        """Yield (states, actions, log_probs, adv, ret) mini-batches."""
        n = len(self)
        adv, ret = self.compute_advantages()
        idx = torch.randperm(n, device=self.device)
        for start in range(0, n, batch_size):
            b = idx[start: start + batch_size]
            yield (
                self.states[b],
                self.actions[b],
                self.log_probs[b],
                adv[b],
                ret[b],
            )

    def clear(self):
        self.ptr = 0
        self.full = False


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────

class ONIPPOTrainer:
    """
    Unified PPO / GRPO / IRL / flow-matching trainer.

    Parameters
    ----------
    oni              : the full ONI nn.Module; inspected via hasattr
    mode             : 'motor' | 'text_diffuser' | 'text_lm' | 'irl'
    hidden_dim       : state representation dimension
    action_dim       : motor action size (motor/irl modes)
    rollout_len      : steps per rollout before each PPO update
    ppo_epochs       : gradient epochs per PPO update
    ppo_batch        : mini-batch size for PPO
    clip_eps         : PPO clipping ε
    entropy_coef     : entropy bonus coefficient
    vf_coef          : value-loss coefficient
    gamma / lam      : discount / GAE λ
    lr               : learning rate (shared default)
    grpo_n_samples   : completions per prompt for GRPO
    optimizers       : optional dict of pre-built optimisers keyed by
                       'actor', 'critic', 'discrim', 'diffuser', 'lm'
    device           : override device (defaults to oni.device if available)
    """

    def __init__(
        self,
        oni: nn.Module,
        mode: str = "motor",
        hidden_dim: int = 896,
        action_dim: int = 64,
        rollout_len: int = 256,
        ppo_epochs: int = 4,
        ppo_batch: int = 64,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr: float = 3e-4,
        grpo_n_samples: int = 8,
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        device: Optional[torch.device] = None,
    ):
        assert mode in ("motor", "text_diffuser", "text_lm", "irl"), \
            f"Unknown mode '{mode}'. Choose: motor | text_diffuser | text_lm | irl"

        self.oni         = oni
        self.mode        = mode
        self.hidden_dim  = hidden_dim
        self.action_dim  = action_dim
        self.rollout_len = rollout_len
        self.ppo_epochs  = ppo_epochs
        self.ppo_batch   = ppo_batch
        self.clip_eps    = clip_eps
        self.entropy_coef= entropy_coef
        self.vf_coef     = vf_coef
        self.gamma       = gamma
        self.lam         = lam
        self.grpo_n_samples = grpo_n_samples

        # Device resolution
        if device is not None:
            self.device = device
        elif hasattr(oni, "device"):
            self.device = oni.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pull components out of ONI (no crash if absent)
        self.diffuser  = self._get_attr("imagination_diffuser")
        self.lm        = self._get_attr("nlp_module")
        self.memory    = self._get_attr("memory_ensemble")
        self.governor  = self._get_attr("governor")
        self.embedding = self._get_attr("embedding")

        # Action distribution: diagonal Gaussian over action_dim
        # log_std is a *trainer* parameter — the diffuser output is the mean
        self.log_std = nn.Parameter(
            torch.zeros(action_dim, device=self.device)
        )

        # Critic
        self.critic = _CriticHead(hidden_dim).to(self.device)

        # IRL discriminator
        self.discrim = _IRLDiscriminator(hidden_dim, action_dim).to(self.device) \
            if mode == "irl" else None

        # Rollout buffer (motor + irl modes)
        self.buffer = _RolloutBuffer(rollout_len, hidden_dim, action_dim, self.device)

        # Optimisers
        opt = optimizers or {}
        self.opt_actor   = opt.get("actor",   self._make_opt(self._actor_params(),   lr))
        self.opt_critic  = opt.get("critic",  self._make_opt(list(self.critic.parameters()), lr))
        self.opt_diffuser= opt.get("diffuser",self._make_opt(self._diffuser_params(), lr * 0.5))
        self.opt_lm      = opt.get("lm",      self._make_opt(self._lm_params(),       lr * 0.3))
        self.opt_discrim = opt.get("discrim",
            self._make_opt(list(self.discrim.parameters()), lr) if self.discrim else None
        )

        # Stats
        self._stats: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_attr(self, name: str) -> Optional[nn.Module]:
        return getattr(self.oni, name, None)

    @staticmethod
    def _make_opt(params: List, lr: float) -> Optional[torch.optim.Optimizer]:
        if not params:
            return None
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    def _actor_params(self) -> List:
        """Parameters that form the policy (log_std + diffuser robotics process)."""
        params = [self.log_std]
        if self.diffuser is not None and hasattr(self.diffuser, "processes"):
            proc = self.diffuser.processes.get("robotics")
            if proc is not None:
                params += list(proc.parameters())
            # Also include router (routing is part of the policy)
            if hasattr(self.diffuser, "router"):
                params += list(self.diffuser.router.parameters())
        return params

    def _diffuser_params(self) -> List:
        if self.diffuser is None:
            return []
        return list(self.diffuser.parameters())

    def _lm_params(self) -> List:
        if self.lm is None:
            return []
        return list(self.lm.parameters())

    def _embed_obs(self, obs) -> torch.Tensor:
        """
        Convert a raw gym observation (numpy array or tensor) into a
        (1, hidden_dim) state embedding.

        Pipeline: obs → flatten/pad → embedding layer (if LM embedding
        available, map token ids; otherwise a linear projection built once).
        """
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        obs = obs.to(self.device).float()

        # Build a one-shot linear projector keyed to obs shape if needed
        obs_flat = obs.reshape(1, -1)
        obs_dim  = obs_flat.size(-1)

        proj_key = f"_obs_proj_{obs_dim}"
        if not hasattr(self, proj_key):
            proj = nn.Linear(obs_dim, self.hidden_dim, bias=False).to(self.device)
            nn.init.xavier_uniform_(proj.weight)
            setattr(self, proj_key, proj)

        return getattr(self, proj_key)(obs_flat)   # (1, H)

    def _act(self, state_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Returns (action, log_prob) both (1,) or (1, action_dim).
        Uses MPAD 'robotics' sample when available; falls back to linear
        mapping from state_emb if the diffuser is absent or misconfigured.
        """
        if (
            self.diffuser is not None
            and hasattr(self.diffuser, "sample")
            and hasattr(self.diffuser, "processes")
            and "robotics" in getattr(self.diffuser, "processes", {})
        ):
            try:
                mean = self.diffuser.sample(
                    context=state_emb, modality="robotics", steps=6, batch_size=1
                )  # (1, action_dim)
                mean = mean.view(1, self.action_dim)
            except Exception:
                mean = self._fallback_mean(state_emb)
        else:
            mean = self._fallback_mean(state_emb)

        std  = self.log_std.exp().clamp(1e-4, 2.0)
        dist = torch.distributions.Normal(mean, std.unsqueeze(0))
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)   # (1,)
        return action.detach(), log_prob.detach()

    def _fallback_mean(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Simple linear projection as action mean fallback."""
        if not hasattr(self, "_fallback_actor"):
            self._fallback_actor = nn.Linear(
                self.hidden_dim, self.action_dim, bias=False
            ).to(self.device)
        return self._fallback_actor(state_emb)    # (1, action_dim)

    def _log_prob(self, state_emb: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Re-evaluate log-prob for a batch of stored (state, action) pairs."""
        if (
            self.diffuser is not None
            and hasattr(self.diffuser, "sample")
            and hasattr(self.diffuser, "processes")
            and "robotics" in getattr(self.diffuser, "processes", {})
        ):
            try:
                mean = self.diffuser.sample(
                    context=state_emb, modality="robotics", steps=6,
                    batch_size=state_emb.size(0)
                ).view(state_emb.size(0), self.action_dim)
            except Exception:
                mean = self._fallback_mean(state_emb)
        else:
            mean = self._fallback_mean(state_emb)

        std  = self.log_std.exp().clamp(1e-4, 2.0).unsqueeze(0)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1)

    # -------------------------------------------------------------------------
    # Motor rollout
    # -------------------------------------------------------------------------

    def motor_rollout(self, env, n_steps: Optional[int] = None) -> float:
        """
        Collect n_steps transitions from env into self.buffer.
        Returns episode return (sum of undiscounted rewards).
        """
        if not _GYM:
            raise ImportError("gymnasium / gym not installed — required for motor mode")
        n = n_steps or self.rollout_len
        self.buffer.clear()

        obs, _ = env.reset() if hasattr(env, "reset") else (env.reset(), {})
        state  = self._embed_obs(obs)
        ep_ret = 0.0

        for _ in range(n):
            with torch.no_grad():
                value    = self.critic(state).item()
                action, log_prob = self._act(state)

            step_out = env.step(action.cpu().numpy().ravel())
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_out

            ep_ret += float(reward)

            # IRL reward override
            if self.mode == "irl" and self.discrim is not None:
                reward = float(self.discrim.reward(state, action))

            self.buffer.add(state.squeeze(), action.squeeze(),
                            float(log_prob), float(reward), value, bool(done))

            if done:
                obs, _ = env.reset() if hasattr(env, "reset") else (env.reset(), {})
                state  = self._embed_obs(obs)
            else:
                state = self._embed_obs(next_obs)

        return ep_ret

    # -------------------------------------------------------------------------
    # PPO update  (motor + irl)
    # -------------------------------------------------------------------------

    def ppo_update(self) -> Dict[str, float]:
        """Run ppo_epochs of mini-batch updates on self.buffer. Returns loss stats."""
        if self.opt_actor is None or self.opt_critic is None:
            return {}
        actor_losses, critic_losses, entropies = [], [], []

        for _ in range(self.ppo_epochs):
            for states, actions, old_lp, adv, ret in self.buffer.batches(self.ppo_batch):
                new_lp, entropy = self._log_prob(states, actions)
                ratio  = (new_lp - old_lp).exp()
                clip_r = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                actor_loss = -torch.min(ratio * adv, clip_r * adv).mean()

                values     = self.critic(states)
                critic_loss= F.mse_loss(values, ret)

                ent_bonus  = -entropy.mean() * self.entropy_coef
                loss       = actor_loss + self.vf_coef * critic_loss + ent_bonus

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for pg in self.opt_actor.param_groups for p in pg["params"]], 0.5
                )
                self.opt_actor.step()
                self.opt_critic.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "actor_loss":  float(torch.tensor(actor_losses).mean()),
            "critic_loss": float(torch.tensor(critic_losses).mean()),
            "entropy":     float(torch.tensor(entropies).mean()),
        }

    # -------------------------------------------------------------------------
    # IRL discriminator update
    # -------------------------------------------------------------------------

    def irl_update(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update discriminator on one batch of expert demos vs current buffer.
        expert_states / expert_actions: (N, hidden_dim / action_dim).
        """
        if self.discrim is None or self.opt_discrim is None:
            return {}
        n = min(len(self.buffer), expert_states.size(0), self.ppo_batch)
        idx = torch.randperm(len(self.buffer), device=self.device)[:n]
        pol_s = self.buffer.states[idx]
        pol_a = self.buffer.actions[idx]
        exp_s = expert_states[:n].to(self.device)
        exp_a = expert_actions[:n].to(self.device)

        loss = self.discrim.loss(exp_s, exp_a, pol_s, pol_a)
        self.opt_discrim.zero_grad()
        loss.backward()
        self.opt_discrim.step()
        return {"discrim_loss": loss.item()}

    # -------------------------------------------------------------------------
    # Text diffuser update  (flow matching ± reward weighting)
    # -------------------------------------------------------------------------

    def flow_update(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        modality: str = "text",
        context: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        One gradient step of flow-matching loss on the diffuser.

        x0, x1   : (B, in_dim) noise / target pairs
        modality  : MPAD modality key (default 'text')
        context   : (B, hidden_dim) optional conditioning
        weights   : (B,) optional per-sample reward weights (IRL or human scores)
        """
        if self.diffuser is None or self.opt_diffuser is None:
            return {}
        if not (hasattr(self.diffuser, "flow_loss") and
                hasattr(self.diffuser, "processes") and
                modality in self.diffuser.processes):
            return {}

        x0 = x0.to(self.device).float()
        x1 = x1.to(self.device).float()
        if context is not None:
            context = context.to(self.device).float()

        loss = self.diffuser.flow_loss(x0, x1, modality=modality, context=context)

        if weights is not None:
            w = weights.to(self.device).float()
            w = w / (w.sum() + 1e-8) * w.numel()
            loss = (loss * w).mean() if loss.dim() > 0 else loss * w.mean()

        if loss.dim() > 0:
            loss = loss.mean()

        self.opt_diffuser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._diffuser_params(), 1.0)
        self.opt_diffuser.step()
        return {"flow_loss": loss.item()}

    # -------------------------------------------------------------------------
    # GRPO  (text_lm mode)
    # -------------------------------------------------------------------------

    def grpo_update(
        self,
        input_ids: torch.Tensor,
        reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Group Relative Policy Optimization step on oni.nlp_module.

        input_ids : (B, seq_len) prompt token ids
        reward_fn : callable(input_ids, completion_ids) → (B*G,) rewards.
                    Defaults to mean log-prob under the policy (self-improvement).
        n_samples : completions per prompt G (overrides self.grpo_n_samples)

        The policy must expose forward(input_ids) → logits (B, seq, vocab).
        """
        if self.lm is None or self.opt_lm is None:
            return {}

        G  = n_samples or self.grpo_n_samples
        B, T = input_ids.shape
        device = self.device
        input_ids = input_ids.to(device)

        # --- generate G completions per prompt ---
        if not hasattr(self.lm, "forward"):
            return {}

        # We need logits to both sample completions and compute log-probs.
        # Generate one greedy/sampled step at a time for simplicity.
        completions, comp_log_probs = self._lm_sample(input_ids, G)
        # completions:     (B*G, T)
        # comp_log_probs:  (B*G,)  — sum of token log-probs

        # --- reward ---
        if reward_fn is not None:
            # Tile input_ids to (B*G, T) to match completions
            tiled_prompts = input_ids.repeat_interleave(G, dim=0)
            with torch.no_grad():
                rewards = reward_fn(tiled_prompts, completions).to(device).float()
        else:
            rewards = comp_log_probs.detach()   # self-reward: higher log-prob = better

        # --- group-relative advantages ---
        rewards = rewards.view(B, G)                          # (B, G)
        adv     = (rewards - rewards.mean(-1, keepdim=True)) \
                / (rewards.std(-1, keepdim=True) + 1e-8)     # (B, G)
        adv     = adv.view(B * G)                             # (B*G,)

        # --- policy gradient loss (PPO-clip on log-prob ratios) ---
        # Re-compute log-probs under the current (updating) policy
        tiled_prompts = input_ids.repeat_interleave(G, dim=0)
        new_log_probs = self._lm_log_prob(tiled_prompts, completions)  # (B*G,)
        ratio = (new_log_probs - comp_log_probs.detach()).exp()
        clip  = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
        loss  = -torch.min(ratio * adv, clip * adv).mean()

        self.opt_lm.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._lm_params(), 1.0)
        self.opt_lm.step()
        return {
            "grpo_loss":    loss.item(),
            "mean_reward":  rewards.mean().item(),
            "reward_std":   rewards.std().item(),
        }

    def _lm_sample(
        self, input_ids: torch.Tensor, G: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample G completions per row of input_ids using greedy decoding
        with temperature sampling.  Returns (completions, log_probs).
        Both (B*G, seq_len).
        """
        B, T = input_ids.shape
        tiled = input_ids.repeat_interleave(G, dim=0)   # (B*G, T)
        try:
            with torch.no_grad():
                out = self.lm(tiled)
            # Accept either Tensor or tuple
            logits = out[0] if isinstance(out, (tuple, list)) else out  # (B*G, T, V)
        except Exception:
            # Fallback: return zero completions
            return tiled, torch.zeros(B * G, device=self.device)

        # Sample next token from last position
        last_logits = logits[:, -1, :]               # (B*G, V)
        probs       = F.softmax(last_logits / 0.9, dim=-1)
        next_tok    = torch.multinomial(probs, 1)    # (B*G, 1)
        completions = torch.cat([tiled, next_tok], dim=1)   # (B*G, T+1)

        # Log-prob of sampled token
        log_probs = F.log_softmax(last_logits, dim=-1)
        lp = log_probs.gather(1, next_tok).squeeze(-1)  # (B*G,)
        return completions, lp

    def _lm_log_prob(
        self, input_ids: torch.Tensor, completions: torch.Tensor
    ) -> torch.Tensor:
        """Compute sum log-prob of completion tokens under current policy."""
        try:
            out    = self.lm(input_ids)
            logits = out[0] if isinstance(out, (tuple, list)) else out  # (N, T, V)
        except Exception:
            return torch.zeros(input_ids.size(0), device=self.device)

        T_in      = input_ids.size(1)
        # Last generated token is at position T_in in completions
        gen_tok   = completions[:, T_in:]   # (N, n_gen)
        if gen_tok.size(1) == 0:
            return torch.zeros(input_ids.size(0), device=self.device)
        last_lp   = F.log_softmax(logits[:, -1, :], dim=-1)  # (N, V)
        return last_lp.gather(1, gen_tok[:, :1]).squeeze(-1)  # (N,)

    # -------------------------------------------------------------------------
    # High-level loop
    # -------------------------------------------------------------------------

    def loop(
        self,
        env_or_data=None,
        n_steps: Optional[int] = None,
        expert_demos: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        reward_fn: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        One complete episode / data pass depending on mode.

        motor / irl   : rollout env → (irl_update →) ppo_update
        text_diffuser : flow_update on env_or_data = (x0, x1[, context[, weights]])
        text_lm       : grpo_update on env_or_data = input_ids tensor

        Returns stats dict suitable for logging.
        """
        stats: Dict[str, float] = {}

        if self.mode in ("motor", "irl"):
            if env_or_data is None:
                raise ValueError("motor/irl mode requires a gym env")
            ep_ret = self.motor_rollout(env_or_data, n_steps)
            stats["episode_return"] = ep_ret
            if self.mode == "irl" and expert_demos is not None:
                stats.update(self.irl_update(*expert_demos))
            stats.update(self.ppo_update())

        elif self.mode == "text_diffuser":
            if env_or_data is None:
                raise ValueError("text_diffuser mode requires (x0, x1, ...) tuple")
            data = env_or_data if isinstance(env_or_data, (tuple, list)) else (env_or_data,)
            x0   = data[0]
            x1   = data[1] if len(data) > 1 else data[0]
            ctx  = data[2] if len(data) > 2 else None
            wts  = data[3] if len(data) > 3 else None
            stats.update(self.flow_update(x0, x1, context=ctx, weights=wts))

        elif self.mode == "text_lm":
            if env_or_data is None:
                raise ValueError("text_lm mode requires input_ids tensor")
            stats.update(self.grpo_update(env_or_data, reward_fn=reward_fn))

        self._stats = stats
        return stats

    # -------------------------------------------------------------------------
    # Convenience
    # -------------------------------------------------------------------------

    def save(self, path: str):
        """Save trainer state (critic, discrim, log_std, optimisers)."""
        ckpt = {
            "critic":   self.critic.state_dict(),
            "log_std":  self.log_std.data,
            "mode":     self.mode,
        }
        if self.discrim is not None:
            ckpt["discrim"] = self.discrim.state_dict()
        for name, opt in [
            ("opt_actor", self.opt_actor),
            ("opt_critic", self.opt_critic),
            ("opt_diffuser", self.opt_diffuser),
            ("opt_lm", self.opt_lm),
        ]:
            if opt is not None:
                ckpt[name] = opt.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        """Restore trainer state."""
        ckpt = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(ckpt["critic"])
        self.log_std.data.copy_(ckpt["log_std"])
        if "discrim" in ckpt and self.discrim is not None:
            self.discrim.load_state_dict(ckpt["discrim"])
        for name, opt in [
            ("opt_actor", self.opt_actor),
            ("opt_critic", self.opt_critic),
            ("opt_diffuser", self.opt_diffuser),
            ("opt_lm", self.opt_lm),
        ]:
            if opt is not None and name in ckpt:
                opt.load_state_dict(ckpt[name])

    @property
    def last_stats(self) -> Dict[str, float]:
        return self._stats
