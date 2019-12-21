# atari_agent
Atari player deep Q learning 

Requires:


* conda install git
* conda install tensorflow
* pip install gym
* pip install git+https://github.com/Kojoley/atari-py.git or, alternatively, 
  * pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py



# The overall algorithm

player: {learner, memory},
epsilon = 1, 
max_epsilon=1.0, 
min_epsilon = .1

* for episod = 1 to 10000
    * until episod not finished
        * s <- env.get_current_state()
        * a <- player.take_action(s)
        * s', r <- env.step(a)
        * player.update(a, r, s', s)

------------
*Function* player.take_action(s):
* if epsilon<rand
    * a <- random action
* else
    * q <- player.learner.predict(s): get Q by running the predict of the learner 
    * a <- player.action_policy(q): use a policy to select the action given Q: 
* Return a

------------------
*Function* player.action_policy(q):
* return argmax(q): The action which has the maximum q

-----------
*Function* player.updates(a, r, s', s)
* memory.add_experience(s, s', r, a): Add a, r, s, s' to the memory
* Update epsilon 
* if time to learn (currently based on a constant number of steps)
    * S <- player.memory.get_minibatch()
    * player.learner.train(S)
* if time to update target learner
    * player.learner.update_target()

--------------
*Function* memory.add_experience(a, r, s', s)
* mem.add(<a, r, s, s'>): add s', s, r, a to the memory

-----------------
*Function* memory.get_minibatch()
* batch <- memory.chose(mem) Select some states from memory <a, r, s', s>
* return batch

----------------
*Function* memory.chose()
* batch <- some states randomly from memory mem
* Ensure the states are feasible (no terminal in between)
* return batch






 