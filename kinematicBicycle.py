import numpy as np
import matplotlib.pyplot as plt

class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

        self.L = 2
        self.lr = 1.2
        self.w_max = 1.22

        self.sample_time = 0.01

    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

    def step(self, v, w):
        self.xc += v * np.cos(self.theta + self.beta) * self.sample_time
        self.yc += v * np.sin(self.theta + self.beta) * self.sample_time
        self.theta += v * np.cos(self.beta) * np.tan(self.delta) / self.L * self.sample_time
        self.delta += w * self.sample_time
        self.beta = np.arctan(self.lr * np.tan(self.delta) / self.L)


model = Bicycle()
sample_time = 0.01
time_end = 20

t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    
    if model.delta < np.arctan(2/10):
        model.step(np.pi, model.w_max)
    else:
        model.step(np.pi, 0)
        
plt.axis('equal')
plt.plot(x_data, y_data,label='Model')
plt.legend()
plt.show()