#Éste será mi primer script en python.
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,10, 0.1)
y = x ** 2

plt.plot(x,y, label = '$x^{2}$')
plt.plot(x, np.cos(x),label = '$cos(x)$')
plt.legend()
plt.show()
