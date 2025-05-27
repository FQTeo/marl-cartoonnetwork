import os
import json
import optuna
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule, TensorDictSequential, TensorDictModuleBase
from tensordict import TensorDict, TensorDictBase
from torch import multiprocessing
from torchrl.collectors import SyncDataCollector
from torch.distributions import Categorical
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv, PettingZooWrapper, Compose, DoubleToFloat, StepCounter, ParallelEnv, EnvCreator, ExplorationType, set_exploration_type
from torchrl.envs.libs.pettingzoo import PettingZooWrapper, PettingZooEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, OneHotCategorical, ValueOperator
from torchrl.objectives import ClipPPOLoss, ValueEstimators
# torch.manual_seed(0)
from matplotlib import pyplot as plt
from gymnasium.spaces import Box
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper
from supersuit import frame_stack_v2, pad_observations_v0, pad_action_space_v0
from functools import partial
from tqdm import tqdm
from til_environment.gridworld import env as raw_env
from til_environment.types import RewardNames

# ---------------- Classes ----------------

class PreserveDictWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType]):
        super().__init__(env)
    def observe(self, agent: AgentID) -> ObsType:
        return super().observe(agent)

class DictFrameStackWrapper(BaseWrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.stack_size = stack_size
        self.stacked_viewcone = {}

    def observe(self, agent):
        obs = super().observe(agent)
        if agent not in self.stacked_viewcone:
            self.stacked_viewcone[agent] = [obs["viewcone"]] * self.stack_size
        else:
            self.stacked_viewcone[agent].pop(0)
            self.stacked_viewcone[agent].append(obs["viewcone"])

        stacked = np.stack(self.stacked_viewcone[agent], axis=-1)  # [7, 5, 4]
        obs["viewcone"] = stacked
        return obs

    def observation_space(self, agent):
        space = super().observation_space(agent)
        orig = space["viewcone"]
        new_viewcone = Box(
            low=orig.low.min(),
            high=orig.high.max(),
            shape=(orig.shape[0], orig.shape[1], self.stack_size),
            dtype=orig.dtype,
        )
        space["viewcone"] = new_viewcone
        return space
    
class ScoutGuardFlatten(nn.Module):
    def __init__(self):
        super().__init__()
        # for unpacking each uint8 into 8 binary planes
        self.register_buffer("bits", torch.arange(8, dtype=torch.uint8))

    def forward(self, obs: TensorDictBase) -> torch.Tensor:
        # obs["viewcone"]: [B, A, 7, 5, 4] uint8
        vc = obs["viewcone"]
        mask = self.bits.to(vc.device)                 # [8]
        bv = ((vc.unsqueeze(-1) >> mask) & 1).float()   # [B, A, 7, 5, 4, 8]

        B, A = bv.shape[:2]
        flat_v = bv.view(B, A, -1)                      # [B, A, 1120]

        # now each of these is [B, A, F]
        d = obs["direction"].float()    # [B, A, 4]
        s = obs["scout"].float()        # [B, A, 2]
        l = obs["location"].float()     # [B, A, 2]
        if l.ndim == 2:  # [B, 2] → [B, 1, 2]
            l = l.unsqueeze(1)
        
        # print(f"flat_v: {flat_v.shape}")
        # print(f"d: {d.shape}")
        # print(f"s: {s.shape}")
        # print(f"l: {l.shape}")
        # all 3-D, so this works:
        return torch.cat([flat_v, d, s, l], dim=-1)  # [B, A, 1128]
    
class ReshapeToMultiAgent(nn.Module):
    def __init__(self, n_agents, n_features):
        super().__init__()
        self.n_agents = n_agents
        self.n_features = n_features

    def forward(self, x):
        return x.view(x.shape[0], self.n_agents, self.n_features)

# ---------------- Methods ----------------

def make_parallel_env():
    rewards_dict = {
        # Positive rewards
        RewardNames.SCOUT_RECON: 1.0,         # Encourage collecting recon
        RewardNames.SCOUT_MISSION: 5.0,      # Encourage completing mission
        RewardNames.GUARD_CAPTURES: 80.0,     # Reward for capturing
        RewardNames.GUARD_WINS: 40.0,         # Shared win reward for all guards

        # Negative penalties
        RewardNames.SCOUT_CAPTURED: -20.0,    # Big penalty for scout getting caught
        RewardNames.WALL_COLLISION: -0.8,     # Discourage inefficient movement
        RewardNames.AGENT_COLLIDER: -0.4,     # Discourage colliding with teammates
        RewardNames.AGENT_COLLIDEE: -0.2,     # Minor penalty if collided into
        RewardNames.STATIONARY_PENALTY: -0.3, # Discourage staying still

        # Optional shaping to drive activity
        RewardNames.GUARD_STEP: -0.005,        # Minor penalty per guard step (move with intent)
        RewardNames.SCOUT_STEP: -0.01,        # Minor penalty per scout step (urgency)

        # End-of-episode shaping
        RewardNames.GUARD_TRUNCATION: -4.0,   # Penalize guards for failing to capture
        RewardNames.SCOUT_TRUNCATION: -2.0,   # Penalize scout for not completing mission
    }
    e = raw_env(env_wrappers = [
            PreserveDictWrapper,  # <--- YOUR CUSTOM WRAPPER
            partial(DictFrameStackWrapper, stack_size=4),
        ], render_mode=None, rewards_dict=rewards_dict, novice=False)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    # e = frame_stack_v2(e, stack_size=4)
    # e = pad_observations_v0(e)
    # e = pad_action_space_v0(e)
    e = aec_to_parallel(e)
    return e

def test_env(env):
    print("action_keys:", env.action_keys)
    print("reward_keys:", env.reward_keys)
    print("done_keys:", env.done_keys)
    print("Action Spec:", env.action_spec)
    print("Observation Spec:", env.observation_spec)
    print("Reward Spec:", env.reward_spec)
    print("Done Spec:", env.done_spec)
    check_env_specs(env)
    # n_rollout_steps = 5
    # rollout = env.rollout(n_rollout_steps)
    # print(f"rollout of {n_rollout_steps} steps:", rollout)
    # print("Shape of the rollout TensorDict:", rollout.batch_size)

def unpack_bits(x):
    return ((x.unsqueeze(-1) >> torch.arange(8)) & 1).float()

def generate_dummy_observation():
    dummy_obs = {
        "viewcone": torch.zeros((1, 1, 7, 5, 4), dtype=torch.uint8),   # [B, A, 7, 5, 4]
        "direction": torch.nn.functional.one_hot(torch.zeros((1, 1), dtype=torch.long), num_classes=4).float(),  # [B, A, 4]
        "scout": torch.nn.functional.one_hot(torch.zeros((1, 1), dtype=torch.long), num_classes=2).float(),      # [B, A, 2]
        "location": torch.zeros((1, 1, 2), dtype=torch.float32)  # [B, A, 2]
    }
    return dummy_obs

def make_policy(n_agents, n_features, device, num_cells, activation_class): # Shared MLP wrapper
    class ForwardCompatibleMultiAgentMLP(MultiAgentMLP):
        def forward(self, x):
            # x is [B, A, F] during training
            # At inference it might be [B, 1, F] (single agent), expand to match training agent count
            if x.size(1) == 1 and self.n_agents > 1:
                x = x.expand(-1, self.n_agents, -1)  # [B, n_agents, F]
            return super().forward(x)

    return ForwardCompatibleMultiAgentMLP(
        n_agent_inputs=n_features,
        n_agent_outputs=5,
        n_agents=n_agents,
        centralised=True,
        share_params=True,
        device=device,
        depth=2,
        num_cells=num_cells,
        activation_class=activation_class,
    )

def test_agents():
    reset_td = env.reset()
    print(reset_td)
    for group in ["scout", "guard"]:
                if group in reset_td:
                    print(f"\nTesting {group} policy module...")
                    policy_modules[group](reset_td)
                    print(f"✓ {group} policy module successful")
                    print(f"Output shape: {reset_td[group, 'probs'].shape}")

    # Then, test the actors to generate actions
    for group in ["scout", "guard"]:
        if group in reset_td:
            print(f"\nTesting {group} actor...")
            policies[f"{group}_actor"](reset_td)
            print(f"✓ {group} actor successful")
            print(f"Action shape: {reset_td[group, 'action'].shape}")
            print(f"Sample action: {reset_td[group, 'action']}")

    for group in ["scout", "guard"]:
        print(f"{group}_critic:")
        print(critics[f"{group}_critic"](reset_td))

def test_env_rollout():
    env.rollout(5, policy=agents_policy)

def process_batch(batch: TensorDictBase, groups: dict) -> TensorDictBase:
    """
    Expand done and terminated keys for each group to match reward shape.
    """
    for group in groups:  # Changed from env.group_map.keys()
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
    return batch

def setup_models(params, env, device):
    n_scouts = 1
    n_guards = 3
    flatten = ScoutGuardFlatten()
    with torch.no_grad():
        flat_output = flatten(TensorDict(generate_dummy_observation(), batch_size=[1]))
        n_features = flat_output.shape[-1]
    print(f"n_features: {n_features}")

    # Create activation function
    if params["activation"] == "Tanh":
        activation_class = nn.Tanh
    elif params["activation"] == "ReLU":
        activation_class = nn.ReLU
    else:
        activation_class = nn.Tanh  # Default

    policy_modules = {}

    for group in ["scout", "guard"]:
        num_agents = n_scouts if group == "scout" else n_guards
        policy_modules[group] = TensorDictModule(
            module=nn.Sequential(
                ScoutGuardFlatten(),        # takes one obs dict, outputs [B, 284]
                # ReshapeToMultiAgent(num_agents, n_features),
                make_policy(num_agents, n_features, device, params["network_width"], activation_class),      # returns [B, 5]
                nn.Softmax(dim=-1),
            ),
            in_keys=[(group, "observation")],       # use group name here
            out_keys=[(group, "probs")],
        )
    # -------------------- Actors -------------------------------
    policies = {}
    for group in ["scout", "guard"]:
        policies[f"{group}_actor"] = ProbabilisticActor(
            module=policy_modules[group],
            spec=env.action_spec[(group, "action")],
            in_keys=[(group, "probs")],
            out_keys=[(group, "action")],
            distribution_class=Categorical,
            return_log_prob=True,
        )
    agents_policy = TensorDictSequential(*policies.values())
    # -------------------- Critics -------------------------------
    critics = {}
    for group in ["scout", "guard"]:
        num_agents = n_scouts if group == "scout" else n_guards
        group_flatten = TensorDictModule(
            module=nn.Sequential(
                ScoutGuardFlatten(),
                # ReshapeToMultiAgent(num_agents, n_features)
            ),
            in_keys=[(group, "observation")],
            out_keys=[(group, "flat_observation")],
        )
        group_value_mlp = TensorDictModule(
            module=MultiAgentMLP(
                n_agent_inputs=n_features,
                n_agent_outputs=1,
                n_agents=num_agents,
                centralised=True,       
                share_params=True,
                device=device,
                activation_class=activation_class,
                depth=params["network_depth"],
                num_cells=params["network_width"],
            ),
            in_keys = [(group, "flat_observation")], out_keys = [(group, "state_value")]
        )
        critics[f"{group}_critic"] = TensorDictSequential(
            group_flatten,
            group_value_mlp,
        )
    return policies, agents_policy, critics

def train_model(env, params, policies, agents_policy, critics, device):
    # -------------------- Data Collector -------------------------------
    collector = SyncDataCollector(
        create_env_fn=lambda: env,
        policy=agents_policy,
        device=device,
        storing_device=device,
        frames_per_batch=params["frames_per_batch"],
        total_frames=params["total_frames"],
    )
    # -------------------- Replay Buffer -------------------------------
    # Define agent groups manually
    groups = ["scout", "guard"]
    # Create separate replay buffers for each group
    replay_buffers = {}
    for group in groups:  # Changed from env.group_map.keys()
        replay_buffers[group] = ReplayBuffer(
            storage=LazyTensorStorage(
                params["frames_per_batch"], device=device
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=params["minibatch_size"],
        )
    # -------------------- Loss -------------------------------
    losses = {}
    for group in ["scout", "guard"]:
        # Create the loss module
        loss_module = ClipPPOLoss(
            actor_network=policies[f"{group}_actor"],
            critic_network=critics[f"{group}_critic"],
            clip_epsilon=params["clip_epsilon"],
            entropy_coef=params["entropy_coef"],
            normalize_advantage=True
        )
        # Set the appropriate keys
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            sample_log_prob=(group, "action_log_prob"),
            value=(group, "state_value"),
            done=(group, "done"),
            terminated=(group, "terminated")
        )
        # Create GAE value estimator
        loss_module.make_value_estimator(ValueEstimators.GAE, gamma=params["gamma"], lmbda=params["lmbda"])
        # Add to dictionary
        losses[group] = loss_module

    # Create optimizers for each group
    optimizers = {}
    for group in ["scout", "guard"]:
        optimizers[group] = torch.optim.Adam(
            list(policies[f"{group}_actor"].parameters()) + list(critics[f"{group}_critic"].parameters()),
            lr=params["lr"],
        )

    env.reset()

    # -------------------- Train -------------------------------

    groups = ["scout", "guard"]

    # Training loop
    pbar = tqdm(
        total=params["total_frames"],
        desc=", ".join([f"episode_reward_mean_{group} = 0" for group in groups])  # Changed
    )
    episode_reward_mean_map = {group: [] for group in groups}  # Changed
    total_frames_so_far = 0  # Added to track progress

    # Training/collection iterations
    for iteration, batch in enumerate(collector):
        # for buf in replay_buffers.values():
        #     buf.reset()
        
        batch = process_batch(batch, groups)  # Expand done keys if needed

        # Calculate total frames in this batch
        current_batch_size = batch.numel()
        total_frames_so_far += current_batch_size  # Track total frames

        # Process each group
        for group in groups:
            # Extract data for this group only
            group_batch = batch.exclude(
                *[
                    key
                    for _group in groups  # Changed from env.group_map.keys()
                    if _group != group
                    for key in [_group, ("next", _group)]
                ]
            )

            # Reshape to flatten batch dimensions
            group_batch = group_batch.reshape(-1)

            # Add to this group's replay buffer
            replay_buffers[group].extend(group_batch)

            # PPO training epochs (multiple passes over the same data)
            for _ in range(params["num_epochs"]):
                # Iterate through all minibatches in the buffer once
                for subdata in replay_buffers[group]:
                    # Compute loss
                    loss_vals = losses[group](subdata)

                    # Compute total loss
                    loss_value = (
                        loss_vals["loss_objective"] +
                        loss_vals["loss_critic"] +
                        loss_vals["loss_entropy"]
                    )

                    # Backprop and optimize
                    optimizers[group].zero_grad()
                    loss_value.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        losses[group].parameters(), params["max_grad_norm"]
                    )

                    optimizers[group].step()

        # Update collector policy with new weights
        collector.update_policy_weights_()

        # Logging with error handling
        for group in groups:  # Changed from env.group_map.keys()
            done_mask = batch.get(("next", group, "done"))

            # Check if any episodes finished
            if done_mask.any():
                episode_reward_mean = (
                    batch.get(("next", group, "episode_reward"))[done_mask]
                    .mean()
                    .item()
                )
            else:
                # No episodes finished, use previous value or 0
                episode_reward_mean = (
                    episode_reward_mean_map[group][-1] if episode_reward_mean_map[group] else 0.0
                )

            episode_reward_mean_map[group].append(episode_reward_mean)
        
        pbar.set_description( # Update description with step count
            f"Steps: {total_frames_so_far}, " +
            ", ".join([
                f"{group}: {episode_reward_mean_map[group][-1]:.2f}"
                for group in groups
            ]),
            refresh=False
        )
        pbar.update(current_batch_size) # Update progress bar with total frames processed in this batch

    final_rewards = {}
    for group in groups:
        if episode_reward_mean_map[group]:
            last_n = min(10, len(episode_reward_mean_map[group]))
            final_rewards[group] = sum(episode_reward_mean_map[group][-last_n:]) / last_n
        else:
            final_rewards[group] = 0.0

    # Return average of all group rewards
    return sum(final_rewards.values()) / len(final_rewards)

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object

    Returns:
        float: Mean reward (metric to maximize)
    """
    # Define hyperparameters to optimize
    params = {
        # Sampling parameters
        "n_parallel_envs": 8,
        "frames_per_batch": 2000,  # Fixed for consistency
        "total_frames": 20000,  # Reduced for faster trials

        # Training parameters
        "num_epochs": trial.suggest_int("num_epochs", 3, 12),
        "minibatch_size": trial.suggest_categorical("minibatch_size", [200, 400, 800]),
        "lr": trial.suggest_float("lr", 3e-5, 1e-3, log=True),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 2.0),

        # PPO parameters
        "clip_epsilon": trial.suggest_float("clip_epsilon", 0.2, 0.3),
        "gamma": trial.suggest_float("gamma", 0.99, 0.999),
        "lmbda": trial.suggest_float("lmbda", 0.95, 0.999),
        "entropy_coef": trial.suggest_float("entropy_coef", 3e-4, 1e-2, log=True),
        "value_loss_coef": trial.suggest_float("value_loss_coef", 0.1, 1.0),

        # Network parameters
        "network_depth": trial.suggest_int("network_depth", 2, 3),
        "network_width": trial.suggest_categorical("network_width", [64, 128, 256, 512]),
        "activation": trial.suggest_categorical("activation", ["Tanh", "ReLU"]),
        "share_parameters_policy": True,  # Fixed for simplicity
        "share_parameters_critic": True,  # Fixed for simplicity
        "mappo": trial.suggest_categorical("mappo", [True, False]),
    }

    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    print("device: ", device)
    env = ParallelEnv(params["n_parallel_envs"], make_env, serial_for_single=True, device=device)
    policies, agents_policy, critics = setup_models(params, env, device)
    # Train the model and get mean reward
    mean_reward = train_model(env, params, policies, agents_policy, critics, device)
    #We report back the mean reward, but you can always use other heuristics like weighted average etc.

    return mean_reward

if __name__ == "__main__":
    # ---------------- Environment Wrapping ----------------
    n_parallel_envs = 8
    full_frames_per_batch=8000
    full_total_frames=800000
    e = make_parallel_env()
    group_map = {
        "scout": ["player_0"],
        "guard": ["player_1", "player_2", "player_3"],
    }
    make_env = EnvCreator(lambda: TransformedEnv(
        PettingZooWrapper(
            env=e,
            group_map=group_map,  # <-- Group scout vs guards
            use_mask=False,       # optional, only needed if agents drop out
        ),
        Compose(
            RewardSum(in_keys=[("scout", "reward")], out_keys=[("scout", "episode_reward")]),
            RewardSum(in_keys=[("guard", "reward")], out_keys=[("guard", "episode_reward")]),
            StepCounter(),
            DoubleToFloat(),
        )
    ))
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    env = ParallelEnv(n_parallel_envs, make_env, serial_for_single=True, device=device)
    # test_env(env)

    # ---------------- Hyperparameter Tuning & Training ----------------

    set_composite_lp_aggregate(False).set()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=50)  # Adjust based on computational resources (min. 20, n=2 is very small)

    # Print best parameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters
    with open("best_params.json", "w") as f:
        json.dump(trial.params, f, indent=2)

    # If you want to use the best parameters for a full training run
    best_params = {

        # Sampling parameters (use full budget for final training)
        "n_parallel_envs": n_parallel_envs,
        "frames_per_batch": full_frames_per_batch,
        "total_frames": full_total_frames,  # Full training budget

        # Other parameters from best trial
        "num_epochs": study.best_params["num_epochs"],
        "minibatch_size": study.best_params["minibatch_size"],
        "lr": study.best_params["lr"],
        "max_grad_norm": study.best_params["max_grad_norm"],
        "clip_epsilon": study.best_params["clip_epsilon"],
        "gamma": study.best_params["gamma"],
        "lmbda": study.best_params["lmbda"],
        "entropy_coef": study.best_params["entropy_coef"],
        "network_depth": study.best_params["network_depth"],
        "network_width": study.best_params["network_width"],
        "activation": study.best_params["activation"],
        "share_parameters_policy": True,
        "share_parameters_critic": True,
        "mappo": study.best_params["mappo"],
    }

    # Create environment for final training
    env = ParallelEnv(n_parallel_envs, make_env)

    # Setup device
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device("cpu")
    # Setup models with best parameters
    policies, agents_policy, critics = setup_models(best_params, env, device)

    # Final training run
    print("Starting final training with best parameters...")
    final_reward = train_model(env, best_params, policies, agents_policy, critics, device)
    print(f"Final training complete. Mean reward: {final_reward:.4f}")
    


    # -------------------- Visualisation -------------------------------
    # Create a figure with 1 row and 2 columns
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # # Plot archer rewards on the first subplot
    # axes[0].plot(episode_reward_mean_map["scout"])
    # axes[0].set_xlabel("Training iterations")
    # axes[0].set_ylabel("Reward")
    # axes[0].set_title("Scout Episode Reward Mean")

    # # Plot knight rewards on the second subplot
    # axes[1].plot(episode_reward_mean_map["guard"])
    # axes[1].set_xlabel("Training iterations")
    # axes[1].set_ylabel("Reward")
    # axes[1].set_title("Guard Episode Reward Mean")

    # # Adjust layout to prevent overlap
    # plt.tight_layout()

    # # Show the plot
    # plt.show()


    # -------------------- Save Models -------------------------------

    # test_agents()
    # test_env_rollout()
    # obs = env.reset()
    # print(obs["scout", "observation"])
    
    print("Training complete. Saving models...")
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(policies["scout_actor"].state_dict(), os.path.join(save_dir, "scout_actor.pth"))
    torch.save(policies["guard_actor"].state_dict(), os.path.join(save_dir, "guard_actor.pth"))

    print(f"Models saved to: {os.path.abspath(save_dir)}")
    print("Models saved. Goodbye.")
    