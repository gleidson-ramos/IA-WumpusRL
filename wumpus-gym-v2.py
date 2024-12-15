import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from gymnasium.envs.registration import register
from gymnasium.core import ActType, ObsType
from typing import Optional
import os

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='wumpus-v0',                           # call it whatever you want
    entry_point='wumpus-gym:WumpusWorldEnv',   # module_name:class_name
)

class WumpusWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size: int = 4, render_mode="human"):
        super(WumpusWorldEnv, self).__init__()
        if grid_size not in [4]:
            raise ValueError("Tamanho de grid inválido. Deve ser um dos seguintes: 4, 8, 16, 32 ou 64.")
        
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 4 ações possíveis
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)  # grid_size * grid_size posições possíveis

        # Inicializar posições
        self.state = (0, 0)
        self.gold_position = self._random_position(exclude=[(0, 0)])
        self.wumpus_position = self._random_position(exclude=[(0, 0), self.gold_position])

        # Calcular número de buracos
        min_pits = 2
        max_pits = int(np.ceil(np.sqrt(self.grid_size))) + 1
        num_pits = np.random.randint(min_pits, max_pits + 1)

        # Gerar posições dos buracos
        self.pit_positions = []
        while len(self.pit_positions) < num_pits:
            pit_position = self._random_position(exclude=[(0, 0), self.gold_position, self.wumpus_position] + self.pit_positions)
            self.pit_positions.append(pit_position)

    def _random_position(self, exclude):
        """Gera uma posição aleatória no grid, excluindo posições específicas."""
        while True:
            position = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if position not in exclude:
                return position

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[int, dict[str, any]]:
        super().reset(seed=seed)  # Para compatibilidade com o Gym
        self.state = (0, 0)
        return (self._get_observation(), {})

    def _get_observation(self):
        x, y = self.state
        return y * self.grid_size + x

    def step(self, action):
        x, y = self.state
        oldx, oldy = x, y
        
        if action == 0:  # Up
            y = max(y - 1, 0)
        elif action == 1:  # Down
            y = min(y + 1, self.grid_size - 1)
        elif action == 2:  # Left
            x = max(x - 1, 0)
        elif action == 3:  # Right
            x = min(x + 1, self.grid_size - 1)

        self.state = (x, y)

        # Check for game end conditions and calculate reward
        if self.state == self.wumpus_position:
            reward = -100
            done = True
        elif self.state == self.gold_position:
            reward = 100
            done = True
        elif self.state in self.pit_positions:
            reward = -100
            done = True
        elif oldx == x and oldy == y:
            reward = -80
            done = False
        else:
            reward = -1
            done = False

        return self._get_observation(), reward, done, False, {}

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.state] = 1  # Agent
        grid[self.wumpus_position] = -1  # Wumpus
        for pit in self.pit_positions:
            grid[pit] = -2  # Pits
        grid[self.gold_position] = 2  # Gold

        plt.imshow(grid, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.show()

### Main ###

if __name__ == "__main__":
    # Treinamento com múltiplos tamanhos de grid
    grid_sizes_for_training = [4]  # Grid sizes for training
    models = {}

    # Treinamento para cada tamanho de grid
    for grid_size in grid_sizes_for_training:
        model_filename = f"DQN_wumpus_{grid_size}"

        if os.path.exists(model_filename + ".zip"):
            print(f"Modelo para o tamanho de grid {grid_size}x{grid_size} ja existe.")
        else:
            print(f"Treinando para o tamanho de grid {grid_size}x{grid_size}")
            env = WumpusWorldEnv(grid_size=grid_size)
            # Checa compatibilidade com o Gym
            check_env(env)
            # Criando o modelo para treinamento
            #model = DQN("MlpPolicy", env, verbose=1, gamma=0.80, device='auto')
            model = DQN("MlpPolicy", env, verbose=1, gamma=0.90, device='auto')
            # Treinando o modelo
            #model.learn(total_timesteps=500000)
            model.learn(total_timesteps=500000)
            models[grid_size] = model
            model.save(model_filename)

    # Teste apenas para grids 4x4 e 8x8
    for grid_size in [4]:
        print(f"\nTestando modelo para o tamanho de grid {grid_size}x{grid_size}")
        env = WumpusWorldEnv(grid_size=grid_size)
        # Checa compatibilidade com o Gym
        check_env(env)

        # Carregando o modelo treinado
        model = DQN.load(f"DQN_wumpus_{grid_size}")
        obs, _info = env.reset()
        done = False

        # Testando o modelo treinado
        while not done:
            action, _states = model.predict(obs, deterministic=True)  # Ação predita pelo modelo
            obs, rewards, done, truncs, info = env.step(action)
            #if action == 0:  # Up
            #    print("Ação escolhida: Mover para cima")
            #elif action == 1:  # Down
            #    print("Ação escolhida: Mover para baixo")
            #elif action == 2:  # Left
            #    print("Ação escolhida: Mover para esquerda")
            #elif action == 3:  # Right
            #    print("Ação escolhida: Mover para direita")
            env.render()

        env.close()