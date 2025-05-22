"""Manages the RL model."""
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor
from trainer import ScoutGuardFlatten, make_policy
from collections import deque

class RLManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.viewcone_history = deque(maxlen=4)
        self.device = torch.device("cpu")
        self.n_features = 1128
        self.n_scouts = 1
        self.n_guards = 3
        self.num_cells = 64
        self.activation_class = nn.Tanh 
        self.scout_model = self._load_model("models/scout_actor.pth", self.n_scouts, self.n_features, self.device, self.num_cells, self.activation_class)
        self.guard_model = self._load_model("models/guard_actor.pth", self.n_guards, self.n_features, self.device, self.num_cells, self.activation_class)

    def _load_model(self, checkpoint_path: str, num_agents: int, n_features: int, device, num_cells, activation_class):
        policy_module = TensorDictModule(
            module=nn.Sequential(
                ScoutGuardFlatten(),
                make_policy(num_agents, n_features, device, num_cells, activation_class),
                nn.Softmax(dim=-1),
            ),
            in_keys=[("observation")],
            out_keys=["probs"]
        )

        actor = ProbabilisticActor(
            module=policy_module,
            spec=None,
            in_keys=["probs"],
            out_keys=["action"],
            distribution_class=Categorical,
            return_log_prob=False,
        )

        actor.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        actor.eval()
        return actor

    def rl(self, observation: dict[str, int | list[int]]) -> int:
        """Gets the next action for the agent, based on the observation."""
        # Add current viewcone to history
        current_viewcone = torch.tensor(observation["viewcone"], dtype=torch.uint8)
        self.viewcone_history.append(current_viewcone)
        # Pad with first frame if fewer than 4 frames exist
        while len(self.viewcone_history) < 4:
            self.viewcone_history.appendleft(self.viewcone_history[0])
        # Stack viewcones along last dimension → [7, 5, 4]
        stacked_viewcone = torch.stack(list(self.viewcone_history), dim=-1)  # [7, 5, 4]
        stacked_viewcone = stacked_viewcone.unsqueeze(0).unsqueeze(0)  # [1, 1, 7, 5, 4]

        obs_tensor = {
            "viewcone": stacked_viewcone,  # [1, 1, 7, 5, 4]
            "direction": nn.functional.one_hot(
                torch.tensor([[observation["direction"]]], dtype=torch.long), num_classes=4
            ).float(),  # [1, 1, 4]
            "scout": nn.functional.one_hot(
                torch.tensor([[observation["scout"]]], dtype=torch.long), num_classes=2
            ).float(),  # [1, 1, 2]
            "location": torch.tensor([[observation["location"]]], dtype=torch.float32),  # [1, 1, 2]
        }


        # for key, value in obs_tensor.items():
        #     print(f"{key}: {value.shape}")

        td = TensorDict({"observation": obs_tensor}, batch_size=[1])
        is_scout = observation["scout"] == 1
        model = self.scout_model if is_scout else self.guard_model

        with torch.no_grad():
            out = model(td)

        return out["action"].item()