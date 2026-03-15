import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    x = np.array([-2.5, 0.8, 3.1, -0.3])
    print(sigmoid(x))