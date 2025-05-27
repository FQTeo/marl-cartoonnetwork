from til_environment import gridworld

env = gridworld.env(
    env_wrappers = [],
    render_mode="human"
)


# Reset the environment to initialize visuals
env.reset()

# Demo run one full turn of all agents to show that visual mode works
for agent in env.agent_iter():
    obs, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()
    env.step(action)











