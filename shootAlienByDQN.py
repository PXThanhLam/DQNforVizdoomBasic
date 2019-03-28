import tensorflow as tf
from skimage import  transform
import matplotlib.pyplot as plt
import vizdoom as vzd
import warnings
import  random
import  time
from  collections import  deque
import numpy as np
from skimage.color import rgb2gray

warnings.filterwarnings('ignore')# ignore noise warning cause by skimage
scale_size=150#cant<260
def Create_environment():
    game=vzd.DoomGame()
    config_path=r"C:\Users\PC\Miniconda3\envs\tensorflow\Lib\site-packages\vizdoom\scenarios\basic.cfg"
    game.load_config(config_path)
    scenario_path=r"C:\Users\PC\Miniconda3\envs\tensorflow\Lib\site-packages\vizdoom\scenarios\basic.wad"
    game.set_doom_scenario_path(scenario_path)
    game.init()
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions

#test to see how bad random play does
def test_environtment():
    game = vzd.DoomGame()
    config_path = r"C:\Users\PC\Miniconda3\envs\tensorflow\Lib\site-packages\vizdoom\scenarios\basic.cfg"
    game.load_config(config_path)
    scenario_path = r"C:\Users\PC\Miniconda3\envs\tensorflow\Lib\site-packages\vizdoom\scenarios\basic.wad"
    game.set_doom_scenario_path(scenario_path)
    game.init()
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    actions= [left, right, shoot]
    episode=50
    rewards=np.asarray([])
    for i in range(episode):
        game.new_episode()
        while not game.is_episode_finished():
            action = random.choice(actions)
            game.make_action(action)
            time.sleep(0.02)
        rewards=np.append(rewards,game.get_total_reward())
        print("Result at episode ",i, game.get_total_reward())
        time.sleep(2)
    print("Average Score :", rewards.mean())
    game.close()

#preprocess frame, minimize and cut some useless like roof to faster the program
def preprocess_frame(frame):
    frame = np.transpose(frame, (1, -1, 0))
    gray_frame = rgb2gray(frame)
    crop_frame = gray_frame[30:-10, 30:-30]
    normalize_crop_frame = crop_frame / 255.0
    resize_frame = transform.resize(normalize_crop_frame, [scale_size, scale_size])
    return resize_frame
#stack the continuos frame to see the motion, and that stack is input of out network

def stack_frame(stacked_frames,state,is_new_episode,stack_size=4):
    frame=preprocess_frame(state)
    if is_new_episode:
        stacked_frames=deque([np.zeros((scale_size,scale_size),dtype=np.float) for i in range(stack_size) ],maxlen=4)
        for t in range(stack_size):
            stacked_frames.append(frame)
        stack_state=np.stack(stacked_frames,axis=2)
    else:
        stacked_frames.append(frame)
        stack_state=np.stack(stacked_frames,axis=2)
    return stack_state,stacked_frames

game,possible_actions=Create_environment()
#hyper parameter setting
stack_size=4
state_size=[scale_size,scale_size,stack_size]#aka input shape
action_size=game.get_available_buttons_size()
learning_rate =  0.0002

train_episode=50
max_step_per_episode=120
batch_size=64

max_epsilon=1
min_epsilon=0.01
decay_rate=0.001

gamma=0.96
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

training=True
is_render=False #want to see a game or not

#our DQN
class DQNet:
    def __init__(self,state_size,action_size,learning_rate,name="DQNet"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        with tf.variable_scope(name):
            self.inputs=tf.placeholder(shape=[None,*state_size],name="inputs",dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None,3], name="actions",dtype=tf.float32)
            self.target_Q=tf.placeholder(shape=[None],name="target_Q",dtype=tf.float32) #our Q_value get by defined
            #our DQN model
            ############
            self.conv1=tf.layers.conv2d(inputs=self.inputs,filters=32,kernel_size=[8,8],strides=[4,4],padding="VALID",
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            self.conv1_batchnorm=tf.layers.batch_normalization(inputs=self.conv1,training=True,name="batchnorm1",epsilon=1e-5)
            self.conv1_out=tf.nn.elu(self.conv1_batchnorm,name="conv1out")

            #############
            self.conv2=tf.layers.conv2d(inputs=self.conv1,filters=64,kernel_size=[4,4],strides=[2,2],padding="VALID",
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv2")
            self.conv2_batchnorm = tf.layers.batch_normalization(inputs=self.conv2, training=True, name="batchnorm2",
                                                                 epsilon=1e-5)
            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2out")

            #################
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128, kernel_size=[4,4], strides=[2,2], padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_batchnorm = tf.layers.batch_normalization(inputs=self.conv3, training=True, name="batchnorm3",
                                                                 epsilon=1e-5)
            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3out")
            ##################
            self.flattern=tf.layers.flatten(inputs=self.conv3_out)
            ##################
            self.fc1=tf.layers.dense(inputs=self.flattern,units=512,activation=tf.nn.elu,kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")
            ####################
            self.out_put=tf.layers.dense(inputs=self.fc1,units=3,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="out")
            ####################
            self.Q=tf.reduce_sum(tf.multiply(self.out_put,self.actions),axis=1)
            self.loss=tf.reduce_mean(tf.square(self.Q-self.target_Q))
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

#instance DQNet
tf.reset_default_graph()
DQNetwork = DQNet(state_size, action_size, learning_rate)
#sample from a buffer memory( have tuples of exp(state, action, reward, new state) randomly to train our net
class Memory:
    def __init__(self,maxsize):
        self.buffer=deque(maxlen=maxsize)
    def add(self,experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                     size=batch_size,
                                     replace=False)
        return [self.buffer[i] for i in index]

#instace our memory
stacked_frames  =  deque([np.zeros((scale_size,scale_size), dtype=np.int) for i in range(stack_size)], maxlen=4)
memory=Memory(maxsize=memory_size)
#render new env
game.new_episode()
for i in range(pretrain_length):
    #if first step
    if i == 0:
        # Get curent state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frame(stacked_frames, state, True)

    # Random action
    action = random.choice(possible_actions)

    # Get the rewards
    reward = game.make_action(action)

    # Look if the episode is finished
    done = game.is_episode_finished()

    # If we're dead
    if done :
        # We finished the episode
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        game.new_episode()

        # First we need a state
        state = game.get_state().screen_buffer

        # Stack the frames
        state, stacked_frames = stack_frame(stacked_frames, state, True)

    else:
        # Get the next state
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frame(stacked_frames, next_state, False)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Our state is now the next_state
        state = next_state
#function to predict_action
def predict_action(max_eps,min_eps,decay_rate,decay_step,state,possible_actions):
    exp_exp_tradeoff = np.random.rand()

    #  use an improved version of our epsilon greedy strategy

    explore_probability = min_eps+ (max_eps - min_eps) * np.exp(-decay_rate * decay_step)
    if exp_exp_tradeoff>explore_probability:
        action=random.choice(possible_actions)
    else :
        Qs=sess.run(DQNetwork.out_put,feed_dict={DQNetwork.inputs:state.reshape((1,*state.shape))})
        choice=np.argmax(Qs)
        action = possible_actions[int(choice)]
    return action, explore_probability
#training part
# Saver will help us to save our model
saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0
        game.init()
        reward_eps = np.asarray([])

        for episode in range(train_episode):
            step = 0
            episode_rewards = []
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frame(stacked_frames, state, True)
            while step < max_step_per_episode:
                step += 1
                decay_step += 1

                # Predict the action to take and take it
                action, explore_probability = predict_action(max_epsilon, min_epsilon, decay_rate, decay_step, state,
                                                             possible_actions)

                reward = game.make_action(action)

                # Look if the episode is finished
                done = game.is_episode_finished()

                # Add the reward to total reward
                episode_rewards.append(reward)
                # If the game is finished

                if done  :
                    # the episode ends so no next state
                    next_state = np.zeros((3,scale_size, scale_size), dtype=np.float)
                    next_state, stacked_frames = stack_frame(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_step_per_episode

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))
                    reward_eps=np.append(reward_eps,total_reward)

                    memory.add((state, action, reward, next_state, done))


                else:
                    # Get the next state
                    next_state = game.get_state().screen_buffer

                    # Stack the frame of the next_state
                    next_state, stacke_frames = stack_frame(stacked_frames, next_state, False)

                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state
                    if step==max_step_per_episode:
                        total_reward = np.sum(episode_rewards)
                        reward_eps = np.append(reward_eps,total_reward)
                        print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))


                ### LEARNING PART
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state
                Qs_next_state = sess.run(DQNetwork.out_put, feed_dict={DQNetwork.inputs: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                   feed_dict={DQNetwork.inputs: states_mb,
                                              DQNetwork.target_Q: targets_mb,
                                              DQNetwork.actions: actions_mb})
            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./DoomBasicModels/model.ckpt")
                print("Model Saved")

    print(reward_eps.mean())

#finally, test how good our agent play
with tf.Session() as sess:
    game, possible_actions = Create_environment()

    totalScore = 0

    # Load the model
    saver.restore(sess, "./DoomBasicModels/model.ckpt")
    game.init()
    state= game.get_state().screen_buffer
    state, stacked_frames = stack_frame(stacked_frames, state, True)
    for i in range(1):

        game.new_episode()
        while not game.is_episode_finished():
            frame = game.get_state().screen_buffer
            state,_ = stack_frame(stacked_frames, frame,False)
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQNetwork.out_put, feed_dict={DQNetwork.inputs: state.reshape((1, *state.shape))})
            action = np.argmax(Qs)
            action = possible_actions[int(action)]
            game.make_action(action)
            score = game.get_total_reward()
        print("Score: ", score)
        totalScore += score
    print("TOTAL_SCORE", totalScore / 100.0)
    game.close()



