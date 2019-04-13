import gym

from gym import envs
all_envs = list(envs.registry.all())

for env in all_envs:
    if ('v4' in env.id):
        m = env.make()
        if (m.observation_space.shape[0] >= 200):
            print(env.id, ', ', m.observation_space)

i=0


