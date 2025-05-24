from rl_manager import RLManager

manager = RLManager()

# obersvation = {
#     "viewcone": [
#         [0, 1, 0, 33, 2],
#         [1, 0, 0, 1, 60],
#         [11, 130, 0, 0, 1],
#         [0, 88, 0, 2, 30],
#         [1, 1, 0, 210, 0],
#         [0, 100, 0, 0, 60],
#         [27, 0, 1, 0, 0]
#     ],
#     "direction": 2,
#     "location": [5, 12],
#     "scout": 1,
#     "step": 42
# }

# print(manager.rl(observation=obersvation))

import random

for _ in range(10):  # Generate 10 random observations
    observation = {
        "viewcone": [
            [random.randint(0, 255) for _ in range(5)]
            for _ in range(7)
        ],
        "direction": random.randint(0, 3),  # Assuming 4 cardinal directions
        "location": [random.randint(0, 15), random.randint(0, 15)],  # Assuming a 16x16 grid
        "scout": random.randint(0, 1),  # 1 for scout, 0 for guard
        "step": random.randint(0, 1000)  # Arbitrary step range
    }

    print(manager.rl(observation=observation))


