import matplotlib.pyplot as plt
import seaborn
print('seaborn', seaborn.__version__)
print('has_style', 'seaborn-whitegrid' in plt.style.available)
print(plt.style.available[:40])
