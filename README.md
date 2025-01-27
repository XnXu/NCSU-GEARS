# Solving Cart-Pole Swing Up and Balance
Group Project for NC State GEARS program

To use our pre-trained model, run this command:
```
python main.py
```

To train the same model we produce, run this command:
```
python main.py --max_episode_step 5000 --train_timesteps 600000 --load False
```

# Attention!!!
I set <mark>terminated</mark> to be False at the end of step function for class CartPoleSwingUp, which should be changed later on.