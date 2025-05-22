from rl_manager import RLManager

manager = RLManager()

obersvation = {
    "viewcone": [
        [0, 1, 0, 0, 2],
        [1, 0, 0, 1, 0],
        [0, 3, 0, 0, 1],
        [0, 0, 0, 2, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [2, 0, 1, 0, 0]
    ],
    "direction": 2,
    "location": [5, 12],
    "scout": 1,
    "step": 42
}

print(manager.rl(observation=obersvation))

