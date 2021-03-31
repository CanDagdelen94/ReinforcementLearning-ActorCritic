
from tensorflow import keras
from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np

class ActorCritic:
    def __init__(self, inputdims, actiondims, layers, neurons):
        inputdims = inputdims
        self.actiondims = actiondims
        layers = layers
        neurons = neurons
        self.actor = self.buildnetwork_actor(inputdims, self.actiondims, layers, neurons)
        self.critic = self.buildnetwork_critic(inputdims, self.actiondims, layers, neurons)
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_state_memory = []
    #################################    
    def buildnetwork_actor(self, inputdims, outputdims, layers, neurons):
        inp=keras.Input(shape=(inputdims,))
        hid = keras.layers.Dense(neurons, activation = keras.activations.relu)(inp)
        for l in range(layers-1):
            hid = keras.layers.Dense(neurons, activation = keras.activations.relu)(hid)
        out = keras.layers.Dense(outputdims, activation = keras.activations.softmax)(hid)
        
        model = keras.Model([inp], [out])
        optimizer = keras.optimizers.Adamax()
        model.compile(optimizer = optimizer)
        return model
    def buildnetwork_critic(self, inputdims, outputdims, layers, neurons):
        inp=keras.Input(shape=(inputdims,))
        hid = keras.layers.Dense(neurons, activation = keras.activations.relu)(inp)
        for l in range(layers-1):
            hid = keras.layers.Dense(neurons, activation = keras.activations.relu)(hid)
        out = keras.layers.Dense(outputdims)(hid)
        
        model = keras.Model([inp], [out])
        optimizer = keras.optimizers.Adamax()
        model.compile(optimizer = optimizer)
        return model
    #################################    
    def store_transition(self, state, action, reward, state_):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.next_state_memory.append(state_)
    #################################
    def choose_action(self, state):
        state = tf.expand_dims(state,0)
        probs = self.actor(state)
        action = np.argmax(probs, axis = 1)[0]
        prob = probs[0, action]
        return action, prob
    def choose_Qvalue(self, state, action):
        state = tf.expand_dims(state,0)
        q_values = self.critic(state)
        q_value = q_values[0,action]
        return q_value
    #################################
    def customloss_actor(self, probs, deltas):
        outs = k.clip(probs, 1e-8, 1-1e-8)
        loglik = -k.log(outs) * deltas
        al = k.mean(loglik)
        return al
    def customloss_critic(self, target, old):
        critic_loss = (target - old)**2
        cl = k.sum(critic_loss)/len(critic_loss)
        return cl
    #################################
    def learn(self):
        state_memory = np.array(self.state_memory, dtype=np.float32)
        action_memory = np.array(self.action_memory, dtype=np.float32)
        reward_memory = np.array(self.reward_memory, dtype=np.float32)
        next_state_memory = np.array(self.next_state_memory, dtype=np.float32)
        
        reward_history = tf.expand_dims(reward_memory, 1)
        
        with tf.GradientTape() as actortape, tf.GradientTape() as critictape:
            critictape.watch(self.critic.trainable_variables)
            actortape.watch(self.actor.trainable_variables)
            q_value_history = []
            q_value_next_history = []
            
            action_history = []
            prob_history = []
            for i in range(len(state_memory)):
                action, prob= self.choose_action(state_memory[i])
                q_value = self.choose_Qvalue(state_memory[i], action)
                action_next, prob_next= self.choose_action(next_state_memory[i])
                q_value_next = self.choose_Qvalue(next_state_memory[i], action_next)
                
                action_history.append(action)
                prob_history.append(prob)
                q_value_history.append(q_value)
                q_value_next_history.append(q_value_next)

            q_value_history = tf.expand_dims(q_value_history, 1)
            q_value_next_history = tf.expand_dims(q_value_next_history, 1)
                
            action_history = tf.expand_dims(action_history, 1)
            prob_history = tf.expand_dims(prob_history, 1)
            
            gamma = 0.9
            target = reward_history + gamma * q_value_next_history
            diff = target - q_value_history

            critic_loss = self.customloss_critic(target, q_value_history)
            critic_grad = critictape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
            
            actor_loss = self.customloss_actor(prob_history, diff)
            actor_grad = actortape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
                
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.next_state_memory = []