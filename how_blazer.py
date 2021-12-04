from Blazer_Model import Model
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange


def main():
    md = Model(automatic_control=True)
    md.reset(drive_trace="IM240", SOC=25)
    speeds = np.empty((len(md), 1))
    jerk   = np.empty((len(md), 1))
    powers = np.empty((len(md), 2))
    fuels  = np.empty((len(md), 2))
    SOC    = np.empty((len(md), 1))

    for i in trange(len(md)):
        action = np.zeros((3,), dtype=np.double)
        a = md.step(action)
        speeds[i,0] = (a[0])
        powers[i,:] = [a[5], a[6]]
        jerk[i,0] = (a[7])
        SOC[i,0] = a[9]


    fig, ax = plt.subplots(4)

    xs = np.arange(len(speeds))
    ax[0].plot(xs, speeds[:,0], label="ego-speed")
    ax[0].legend()
    ax[1].plot(xs, powers[:,0], label="engine power (W)")
    ax[1].plot(xs, powers[:,1], label="Motor power (W)")
    ax[1].legend()
    ax[2].plot(xs, jerk[:,0], label="ego-vehicle jerk")
    ax[2].legend()
    ax[3].plot(xs, SOC[:,0], label="SOC")
    ax[3].legend()
    plt.show()


if __name__ == "__main__":
    main()
