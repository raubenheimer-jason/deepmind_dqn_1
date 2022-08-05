AGENT_HISTORY_LEN = 4  # Number of most recent frames given as input to the Q network
UPDATE_FREQ = 4  # Agent selects 4 actions between each pair of successive updates
NO_OP_MAX = 30  # max num of "do nothing" actions performed by agent at the start of an episode

FRAMES_SKIP = 4  # I added this...

REWARD_MAX = 1.0
REWARD_MIN = -1.0

REPLAY_MEM_SIZE = 1_000_000
BATCH_SIZE = 32
LEARNING_RATE = 0.25e-3  # learning rate used by RMSProp
# LEARNING_RATE = 5e-5  # learning rate used by RMSProp <<-- from youtube dude...
GRADIENT_MOMENTUM = 0.95  # RMSProp
SQUARED_GRADIENT_MOMENTUM = 0.95  # RMSProp
MIN_SQUARED_GRADIENT = 0.01  # RMSProp
TARGET_NET_UPDATE_FREQ = 10_000  # C
ACTION_REPEAT = 4  # Agent only sees every 4th input frame (repeat last action)
PRINT_INFO_FREQ = 1_000

LOG_DIR = "./logs/"
LOG_INTERVAL = 1_000

SAVE_DIR = "./models/"
SAVE_INTERVAL = 10_000
SAVE_NEW_FILE_INTERVAL = 100_000

PRINTING = True
LOGGING = True
SAVING = True


INITIAL_EXPLORATION = 1  # Initial value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION = 0.1  # final value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION_FRAME = 1_000_000  # num frames epsilon changes linearly

REPLAY_START_SIZE = 50_000  # uniform random policy run before learning
# REPLAY_START_SIZE = 1_000  # uniform random policy run before learning #! testing

GAMMA = 0.99  # discount factor used in Q-learning update


# delta y over delta x
DECAY_SLOPE = (INITIAL_EXPLORATION-FINAL_EXPLORATION) / \
    (REPLAY_START_SIZE-(REPLAY_START_SIZE+FINAL_EXPLORATION_FRAME))
DECAY_C = INITIAL_EXPLORATION - (DECAY_SLOPE*REPLAY_START_SIZE)
