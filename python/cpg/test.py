import numpy as np 

def test_main():
    print("______________TEST______________")
    print("normalization using tanh")
    theta = np.pi/4
    u = np.cos(theta)
    y = 0.5*(np.tanh(u*2) + 1)
    print(f"U = {u} and y = {y}")

if __name__ == '__main__':
    test_main()