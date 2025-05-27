# marl-cartoonnetwork
Create your own Multi-Agent Reinforcement learning training environment, courtesy of TIL BrainHack 2025

## Installing Dependencies
Step 1: Ensure a version of Python 3.9-3.12 is installed
<br>
Step 2: `pip install -r requirements.txt` 
<br>
Step 3: `pip install -e til-25-environment`

## Understanding the model
Before training the model, it is imperative to understand the environment the models will be operating in and the evaluation metrics used to qualify the capabilities of the models' performance.
<br>
> Environment is a 16x16 grid, initialised with 4 agents (1 Scout, 3 Guards). Each cell contains either a Recon point or Challenge point, which Scouts can earn by entering the cell. They do not regenerate. Guards patrol the cell, and their aim is to capture the Scout, while the Scout aims to evade capture within an episode of 100 steps. 

**Rewards** 
<br>	
| Outcome | Scout Reward | Guard Reward |
| ----------- | ----------- | ---------- |
| Scout gains Recon Point | 1 point | 0 points |
| Scout gains Challenge Point | 5 points | 0 points |
| Guard captures Scout | 50 points | -50 points |

## Fine-Tuning
All fine-tuning techniques can be performed on the functions located in `src/trainer.py`
<br><br>
**Reward Shaping**
<br>
Use reward shaping to modify the rewards given to agents during training to influence their behaviour. 
<br><br>
**Hyperparameter Tuning**
<br>
This code use `optuna` to train the agents, and the hyperparameter ranges (e.g. entropy, loss, gamma etc.) can be modified accordingly.

## Training
`optuna` uses `n` trials with a smaller frame size to determine the best parameters, and then uses them in a training loop with a larger frame size for actual training.
<br><br>
To begin training, run `python src/trainer.py`

## Testing
To verify that the models generate predictions correctly, locate the value of `"network_width"` in `best-params.json`. Assign that value to `self.num_cells` in the `RLManager` class within `rl_manager.py`. The output should be an integer value between 0 and 4, indicating the next step to take
<br>
**Action**
<br>
| Value | Action |
| ---------- | ----------- |
| 0 | Move Forward |
| 1 | Move Backward |
| 2 | Turn Left |
| 3 | Turn Right |
| 4 | Stay |

*Note: Turning does not move an agent into the adjacent cell, it simply changes the direction the agent is facing.*
