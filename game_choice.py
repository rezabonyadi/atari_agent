from player.player import Player
from environments.simulator import Atari
import numpy as np
import datetime
from utils import HandleResults
import numba

GAME_ENV = 'BreakoutDeterministic-v4'
# GAME_ENV = 'SpaceInvaders-v4' # 758 frames
# GAME_ENV = 'Alien-v4' # 948 frames
# GAME_ENV = 'Amidar-v4' # 812 frames
# GAME_ENV = 'Venture-v4'
# GAME_ENV = 'Assault-v4' # 876 frames
# GAME_ENV = 'RoadRunner-v4' # 437 frames
# GAME_ENV = 'PongDeterministic-v4'
# GAME_ENV = 'Asterix-v4'
# GAME_ENV = 'MontezumaRevenge-v4'

