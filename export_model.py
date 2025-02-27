
# import local package

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import torch
import torch.nn as nn

import scipy
from scipy.integrate import solve_ivp
import argparse

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3 import DDPG, TD3
from stable_baselines3.td3.policies import TD3Policy
# Setting up the argument parser for command-line inputs
parser = argparse.ArgumentParser()

# Whether to load a pre-trained model
parser.add_argument('--export_path', type=str, default='model', help='folder path without /')

# Maximum steps allowed per episode
parser.add_argument('--max_episode_step', type=int, default=3000, help='Maximum steps allowed per episode.')
args = parser.parse_args()

# Extract parsed arguments for convenience

export_path = args.export_path
max_episode_step = args.max_episode_step

gym.register(
    id='CartPoleSwingUpRandom',
    entry_point='myCartpoleF_random:CartPoleSwingUp',  # Custom environment location
    reward_threshold=0,  # Reward threshold for environment completion
    max_episode_steps=max_episode_step  # Maximum steps per episode
)

env = DummyVecEnv([lambda: gym.make('CartPoleSwingUpRandom', render_mode='rgb_array')])
# load model and simulate controls
# load_path = 'model'
model = TD3.load(path=export_path + '/td3_swingup_balance', env=env)
# model.load_replay_buffer(load_path + "/td3_swingup_balance_replay_buffer")

print(f"Model structure: {model.policy.net_arch}\n\n")

model_param = model.get_parameters()
model_policy = model_param['policy']
model_policy.keys()

W0 = np.array(model_policy['actor.mu.0.weight'], dtype=np.float32)
b0 = np.array(model_policy['actor.mu.0.bias'], dtype=np.float32)
W1 = np.array(model_policy['actor.mu.2.weight'], dtype=np.float32)
b1 = np.array(model_policy['actor.mu.2.bias'], dtype=np.float32)
W2 = np.array(model_policy['actor.mu.4.weight' ], dtype=np.float32)
b2 = np.array(model_policy['actor.mu.4.bias'], dtype=np.float32)
W3 = np.array(model_policy['actor.mu.6.weight'], dtype=np.float32)
b3 = np.array(model_policy['actor.mu.6.bias'], dtype=np.float32)

for param_tensor in model_policy.keys():
    # find actor network
    if 'actor.mu.' in param_tensor:
      print(param_tensor + ':' + str(model_policy[param_tensor].shape) )
      # print(model_policy[param_tensor])

      np.savetxt(export_path + '/250227_td3txt'
                + param_tensor + '.txt',
                model_policy[param_tensor],
                delimiter=',')

# Wb is a dict
Wb = {
    "W0": W0,
    "b0": b0,
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2,
    "W3": W3,
    "b3": b3
}

# for this proj - nn_Control has fixed shape, so to reduce computation, we fix the shapes
# otherwise, uncomment the size_0 and size_2 def? still will be slow
size_0 = Wb["W0"].shape
size_2 = Wb["W1"].shape
size_4 = Wb["W2"].shape
size_6 = Wb["W3"].shape

print(size_0)

# define nn control
def nn_Control(x, Wb):
    # Wb = (W0, b0, W1, b1, W2, b2, W3, b3)
    # size_0 = Wb["W1"].shape
    shared_net_0_activated = np.maximum(np.zeros_like(size_0[0],),
                                        Wb["W0"] @ x + Wb["b0"].flatten())
    # size_2 = Wb["W2"].shape
    shared_net_2_activated = np.maximum(np.zeros_like(size_2[0],),
                                      Wb["W1"] @ shared_net_0_activated + Wb["b1"].flatten())
    shared_net_3_activated = np.maximum(np.zeros_like(size_4[0],),
                                      Wb["W2"] @ shared_net_2_activated + Wb["b2"].flatten())

    u = Wb["W3"] @ shared_net_3_activated + Wb["b3"].flatten()
    return 10 * np.tanh(u)

# initialize t, h, num_steps

# x0 = [0.1, - 0.1, -0.1 , 0.1 ]
seed = 1
# x0, info = env.reset(seed=seed) # dummyvecenv doesnt take seed as an arg
obs0 = env.reset()
# x0, x_dot0, cos0, sin0, theta_dot0 = obs0[0]
# reset obs = np.array( (np.sin(theta), np.cos(theta), theta_dot, x, x_dot), dtype=np.float32).flatten()
# step obs = np.array( (x, x_dot, np.cos(theta), np.sin(theta), theta_dot), dtype=np.float32).flatten()

x0, x_dot0, cos0, sin0, theta_dot0,  = obs0[0]
theta0 = np.arctan2(sin0, cos0)
h_in = .01 #This is the step size.
x_in = np.array((x0, x_dot0, theta0, theta_dot0), dtype=np.float32).flatten()
t0 = 0; tf = 40;

num_steps = int(np.ceil(tf/h_in)) #The np.ceil is defensive coding: what if h doesn't divide [0,5] evenly

t = np.linspace(t0,tf,num_steps+1) #create a vector of time t_i

h_in = tf/num_steps #This updates h to account for the possibility that it changed due to np.ceil.

# set parameter values (gravity, masscart, masspole, length, r_mp, Jm, Kg, Rm, Beq, Bp, Kt, Km = p )
# from Emi's thesis
gravity = 9.81 # N/kg
masscart = 0.57+0.37 # kg
masspole = 0.230 # kg
length = 0.3302 # m
r_mp = 6.35e-3 # motor pinion radius (m)
Jm = 3.90e-7 # rotor moment of inertia
Kg = 3.71 # planetary gearbox gear ratio
Rm = 2.6 # motor armature resistance (Ohm)
Beq = 5.4 #  equivalent viscous damping coecient as seen at the motor pinion
Bp = 0.0024 # viscous damping doecient, as seen at the pendulum axis
Kt = 0.00767 # motor torque constant
Km = 0.00767 # Back-ElectroMotive-Force (EMF) Constant V.s/RAD

# a dict of system parameters
p_system = {
    "gravity": gravity,
    "masscart": masscart,
    "masspole": masspole,
    "length": length,
    "r_mp": r_mp,
    "Jm": Jm,
    "Kg": Kg,
    "Rm": Rm,
    "Beq": Beq,
    "Bp": Bp,
    "Kt": Kt,
    "Km": Km
    } # all parameters in the original ode

name_of_vars = ['x', 'x_dot', 'theta', 'theta_dot']

# Define two models for the dynamical system

def CartPolePhysics(t, z, p, Vm):
    position, velocity, angle, angular_velocity = z
    # np.array(list(dict.values())) create an array of values of the list
    if type(p) == dict:
      gravity, masscart, masspole, length, r_mp, Jm, Kg, Rm, Beq, Bp, Kt, Km = np.array(list(p.values()))

    if type(p) == np.ndarray:
      gravity, masscart, masspole, length, r_mp, Jm, Kg, Rm, Beq, Bp, Kt, Km = p

    total_mass = masspole + masscart
    costheta = np.cos(angle)
    sintheta = np.sin(angle)


    d = 4 * masscart * r_mp**2 + masspole * r_mp**2 + 4 * Jm * Kg**2 + 3 * r_mp**2 * masspole * sintheta**2

    a33 = -4 * ( Rm * r_mp**2 * Beq + Kg**2 * Kt * Km
              ) / (
                Rm * d
              )

    a34 = - ( 3 * r_mp**2 * Bp * costheta + 4 * masspole * length**2 * r_mp**2 * sintheta * angular_velocity
            ) / (
                length * d
            )

    a43 = -3 * ( Rm * r_mp**2 * Beq + Kg**2 * Kt * Km
            ) * costheta / (
                Rm * length * d
            )

    a44 = -3 * (
                    ( total_mass * r_mp**2 + Jm * Kg**2) * Bp +
                    ( masspole**2 * length**2 * r_mp**2 * costheta * sintheta * angular_velocity )
                ) / (
                    masspole * length**2 * d
                )


    # extra forces on velocity term (Emi's thesis eq 3.2 second plus term)
    aa3 = ( 3 * masspole * r_mp**2 * gravity * costheta * sintheta
            ) / (
            d
            )

    aa4 = ( 3 * (
                    total_mass * r_mp**2 + Jm * Kg**2
                ) * gravity * sintheta
            ) / (
            length * d
            )

    # controller
    b3 = (4 * r_mp * Kg * Kt) / (Rm * d)

    b4 = (3 * r_mp * Kg * Kt * costheta) / (length * Rm * d)

    dxdt = [ velocity + 0 * Vm,
             a33 * velocity + a34 * angular_velocity + aa3 + b3 * Vm,
             angular_velocity + 0 * Vm,
             a43 * velocity + a44 * angular_velocity + aa4 + b4 * Vm
           ] # + ( 0, 0, b3*0, b4*0) # b3 and b4 should be multiplied by u = Vm (cart's motor)

    return np.array(dxdt, dtype=np.float32).flatten()

# define a function to solve the ivp problem that takes different dynamical system and weight/bias matrices from nn

def sol_ivp_withControl(dyn_sys, p, x0, h, Wb):
    """
    Description:
    dyn_sys: dynamical system to be solved by solve_ivp
    p: parameters within the dynamical system
    x0: inital condition - must be state variable
    h: step size
    Wb: neural network control weights and biases as a dict
    """
    # t = np.linspace(t0,tf,int(np.ceil(tf/h))+1)
    z = np.zeros((t.shape[0],4)) # state var
    Vm = np.zeros((t.shape[0],1)) # control var
    z[0,:] = x0
    # Vm[0] = nn_Control_withParam(x = x0, Wb = Wb)
    position0, velocity0, angle0, angular_velocity0 = x0
    nn_input0 = np.array((position0, velocity0, np.cos(angle0), np.sin(angle0), angular_velocity0), dtype=np.float32) # control input obs ((x, x_dot, np.cos(theta), np.sin(theta), theta_dot))
    Vm[0] = nn_Control(x = nn_input0, Wb = Wb)

    for i in range(num_steps):
        # span for next time step
        tspan = [t[i],t[i+1]]

        # solve for next step
        sol = solve_ivp(fun = dyn_sys,
                        t_span = tspan,
                        y0 = z[i,:],
                        args=(p, Vm[i],),
                        t_eval=tspan,
                        dense_output=True)
        # y[i+1, :] = odeint(CartPolePhysics, y[i,:], tspan, args=(Vm[i],))[1]
        z[i+1, :] = sol.y[:,-1]
        # Vm[i+1] = nn_Control_withParam(x = z[i+1,:], Wb = Wb)
        position, velocity, angle, angular_velocity = z[i+1, :]
        nn_input = np.array((position, velocity, np.cos(angle), np.sin(angle), angular_velocity), dtype=np.float32) # control input obs ((x, x_dot, np.cos(theta), np.sin(theta), theta_dot))
        Vm[i+1] = nn_Control(x = nn_input, Wb = Wb)

    return (t, z)

## this block uses RK4 to integrate the system
def RK4_withCtrl(rhs, x0, h, tspan, p, Wb):
    """
    input:
      rhs: the RHS of the ODE governing the state variable
      x0: initial condition
      h: step size
      tspan: time span
      p: system parameters
      Vm: Control_nn(x, Wb)
    """
    t = tspan
    z = np.zeros((t.shape[0],4))
    Vm = np.zeros((t.shape[0],1))

    z[0,:] = x0
    position, velocity, angle, angular_velocity = x0
    nn_input = np.array((position, velocity, np.cos(angle), np.sin(angle), angular_velocity), dtype=np.float32) # control input obs ((x, x_dot, np.cos(theta), np.sin(theta), theta_dot))
    Vm[0] = nn_Control(x = nn_input, Wb = Wb)


    #For loop implementation
    for i in range(num_steps):
        k1 = rhs(t = t[i], z=z[i,:], Vm = Vm[i,:], p = p)
        k2 = rhs(t = t[i] + h / 2, z=z[i,:] + h * k1 / 2, Vm=Vm[i,:], p = p)
        k3 = rhs(t = t[i] + h / 2, z=z[i,:] + h * k2 / 2, Vm=Vm[i,:], p = p)
        k4 = rhs(t = t[i] + h, z=z[i,:] + h * k3, Vm=Vm[i,:], p = p)

        z[i+1,:] = z[i,:] + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # Vm[i+1] = nn_Control(x=z[i+1,:], Wb = Wb )
        position, velocity, angle, angular_velocity = z[i+1, :]
        nn_input = np.array((position, velocity, np.cos(angle), np.sin(angle), angular_velocity), dtype=np.float32) # control input obs ((x, x_dot, np.cos(theta), np.sin(theta), theta_dot))
        Vm[i+1] = nn_Control(x = nn_input, Wb = Wb)


    return z, Vm

## this block uses RK4 to integrate the system
def RK4_withoutCtrl(rhs, x0, h, tspan, p, Wb):
    """
    input:
      rhs: the RHS of the ODE governing the state variable
      x0: initial condition
      h: step size
      tspan: time span
      p: system parameters
      Vm: Control_nn(x, Wb)
    """
    t = tspan
    z = np.zeros((t.shape[0],4))
    Vm = np.zeros((t.shape[0],1))

    z[0,:] = x0
    # Vm[0] = nn_Control(x = x0, Wb = Wb)

    #For loop implementation
    for i in range(num_steps):
        k1 = rhs(t = t[i], z=z[i,:], Vm = Vm[i,:], p = p)
        k2 = rhs(t = t[i] + h / 2, z=z[i,:] + h * k1 / 2, Vm=Vm[i,:], p = p)
        k3 = rhs(t = t[i] + h / 2, z=z[i,:] + h * k2 / 2, Vm=Vm[i,:], p = p)
        k4 = rhs(t = t[i] + h, z=z[i,:] + h * k3, Vm=Vm[i,:], p = p)

        z[i+1,:] = z[i,:] + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # Vm[i+1] = nn_Control(x=z[i+1,:], Wb = Wb )

    return z

# Vm = 0 # RHS parameter - control u = Vm (voltage of motor on cart)
# simulate without control
x_in = np.array((0.1, 0.1, -0.1, -0.1), dtype = np.float32).flatten()
nonlin_noCtrl = solve_ivp(fun = CartPolePhysics,
                          t_span=[t0, tf],
                          y0 = x_in,
                          t_eval = t,
                          args=(p_system, 0,))

nonlin_noCtrl_RK4 = RK4_withoutCtrl(rhs = CartPolePhysics,
                                    x0 = x_in,
                                    h=h_in,
                                    tspan=t,
                                    p = p_system,
                                    Wb = Wb)

# compute sum of differences using two solvers
mse_CartPolePhysics = np.square(nonlin_noCtrl.y.T - nonlin_noCtrl_RK4).mean()
print('Mean Square Error between Solvers on System no Control is ', mse_CartPolePhysics)

plt.plot(t, nonlin_noCtrl.y.T) # plot the transpose
plt.xlabel('t')
plt.legend(name_of_vars, shadow=True)
plt.title('No Control on Lab Model')
plt.grid()
# plt.ylim((-0.25, 0.5))
plt.savefig("model/no control.png", dpi=300)
plt.close()

obs0 = env.reset()

x0, x_dot0, cos0, sin0, theta_dot0,  = obs0[0]
theta0 = np.arctan2(sin0, cos0)
h_in = .01 #This is the step size.
x_in = np.array((x0, x_dot0, -theta0, theta_dot0), dtype=np.float32).flatten()

(t,sol_ivp_with_nnControl) = sol_ivp_withControl(dyn_sys=CartPolePhysics , p = p_system, x0 = x_in, h = h_in, Wb=Wb)
(RK4_with_nnControl, voltages) = RK4_withCtrl(rhs=CartPolePhysics,
                                   x0=x_in,
                                   h = h_in,
                                   tspan=t,
                                   p=p_system,
                                   Wb=Wb)

mse_controls = np.square(sol_ivp_with_nnControl - RK4_with_nnControl).mean()
print('Mean Square Error between Solvers for nn Controlled CartPolePhysics is ', mse_controls)


plt.plot(t, RK4_with_nnControl)
plt.xlabel('t')
plt.legend(name_of_vars, shadow=True)
plt.title('Controlled - with RK4')
plt.grid()
# plt.ylim((-0.25, 0.5))
plt.savefig("model/controlled_withRK4.png", dpi=300)
plt.close()

plt.plot(t, sol_ivp_with_nnControl)
plt.xlabel('t')
plt.legend(name_of_vars, shadow=True)
plt.title('Controlled - with solve_ivp')
plt.grid()
# plt.ylim((-0.25, 0.5))
plt.savefig("model/controlled_with_scipy.png", dpi=300)
plt.close()

plt.plot(t, voltages)
plt.xlabel('t')
# plt.legend(name_of_vars, shadow=True)
plt.title('Controller Voltage Output')
plt.grid()
# plt.ylim((-0.25, 0.5))
plt.savefig("model/controller voltage output.png", dpi=300)
plt.close()

