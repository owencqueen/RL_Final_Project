from Blazer_Model import Model
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange


def main():
    md = Model(automatic_control=False)
    md.reset(drive_trace="US06")
    # speeds = np.empty((len(md),1))
    # jerk   = np.empty((len(md), 1))
    # powers = np.empty((len(md), 2))
    # fuels  = np.empty((len(md), 2))

    number_runs = 10001

    driver_set_speed = np.empty(number_runs)
    tvec_dist = np.empty(number_runs)
    tvec_speed = np.empty(number_runs)

    for i in trange(len(md)):
        action = np.zeros((3,), dtype=np.double)
        a = md.step(action)
        # speeds[i,0] = (a[0])
        # powers[i,:] = [a[5], a[6]]
        # fuels[i,:] = [a[8], a[7]]
        # jerk[i,0] = (a[9])

        driver_set_speed[i] = (a[1])
        tvec_dist[i] = a[3]
        tvec_speed[i] = a[4]

        if i == number_runs - 1:
            break

    #fig, ax = plt.subplots(4)

    plt.plot(driver_set_speed)
    plt.plot(tvec_dist)
    plt.plot(tvec_speed)
    plt.show()

    #xs = np.arange(len(speeds))
    # ax[0].plot(xs, speeds[:,0], label="ego-speed")
    # ax[0].legend()
    # ax[1].plot(xs, powers[:,0], label="engine power (W)")
    # ax[1].plot(xs, powers[:,1], label="Motor power (W)")
    # ax[1].legend()
    # ax[2].plot(xs, fuels[:,0], label="motor fuel equivalent (?)")
    # ax[2].plot(xs, fuels[:,1], label="engine fuel consumption (?)")
    # ax[2].legend()
    # ax[3].plot(xs, jerk[:,0], label="ego-vehicle jerk")
    # ax[3].legend()
    # plt.show()


if __name__ == "__main__":
    main()