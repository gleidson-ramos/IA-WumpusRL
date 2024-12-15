import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from gymnasium.envs.registration import register
from gymnasium.core import ActType, ObsType
from typing import Optional
# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='wumpus-v0',                           # call it whatever you want
    entry_point='wumpus-gym:WumpusWorldEnv', # module_name:class_name
)

class WumpusWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,render_mode="human"):
        super(WumpusWorldEnv, self).__init__()
        self.grid_size = 4
        self.action_space = spaces.Discrete(4) #4 ações possíveis
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size) # grid_size * grid_size posições possíveis
        #Nesta versão o wumpus, o ouro e os buracos estão sempre no mesmo local.
        self.state = (0, 0)
        self.wumpus_position = (2, 2)
        self.gold_position = (3, 3)
        self.pit_positions = [(1, 1), (3, 1)] #Nesta versão, são sempre dois buracos


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[int, dict[str, any]]:
        super().reset(seed=seed)  # Para compatibilidade com o Gym
        self.state = (0, 0)
        return (self._get_observation(), {})
    
    def _get_observation(self):
        x, y = self.state
        return y * self.grid_size + x

    def step(self, action):
        x, y = self.state
        oldx=x
        oldy=y
        
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
        elif oldx==x and oldy==y:
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

if __name__=="__main__":
    env = WumpusWorldEnv()
    env = gym.make('wumpus-v0') #
    print("Iniciando check_env ...\n") #Checa se o env tem incompatibilidades com o Gym
    check_env(env.unwrapped)
    print("Finalizado check_env.\n")
    obs,_info = env.reset()
    done = False

    #Criando o modelo para treinamento
    model = DQN("MlpPolicy", env, verbose=1, gamma=0.80, device='auto')

    #Treinando o modelo
    model.learn(total_timesteps=150000,)
    model.save("DQN_wumpus")
    

    #Carregando um modelo treinado. Comente as linhas 100 a 104 e descomente a linha abaixo
    #model = DQN.load("DQN_wumpus")

    #Testando o modelo treinado
    while not done:
        action,_states = model.predict(obs, deterministic=True)  # Ação predita pelo modelo
        obs, rewards, done, truncs, info = env.step(action)
#Para debug de quais ações estão sendo escolhidas, descomente a linha abaixo        
       #print("Action ",action)
        env.render()
    
    #Closing environment
    env.close()