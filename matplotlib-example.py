import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from comet_ml import Experiment

experiment = Experiment(api_key="7bEW5h9UoEOpQQyNLpt36lY66", project_name="matplotlib")

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2*np.pi*t)
plt.plot(t, s)

plt.xlabel('time (s)')
plt.ylabel('voltage (mV)')
plt.title('About as simple as it gets, folks')
plt.grid(True)

experiment.log_figure(figure=plt)
