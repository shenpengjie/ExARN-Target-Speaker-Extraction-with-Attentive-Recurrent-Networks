import os
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    x=[(i+1)*2 for i in range(10)]
    asthtosh_res=[77.5,81.84,83.84,85,84.63,85.55,86.75,86.64,87.05,87.6]
    context_30db_new=[76.24,79.89,82.21,83.1,84.3,84.25,84.91,85.33,85.54,85.88]
    # context_30db=[76.21,79.55,81.92,83.11,83.22,83.98,85.14,,85.66]
    plt.plot(x,asthtosh_res,"g-",label="Non-causal in paper")
    plt.plot(x,context_30db_new,"r-",label="ours-use-5frames")
    # plt.plot(x,context_30db,"y-",label="ours-use-5frames-")

    # plt.legend(loc='upper left',bbox_to_anchor=(1,1))
    plt.legend()
    plt.tight_layout(pad=2)

    plt.axis([0,20,75,90])
    plt.xticks(np.arange(2,20,2))
    plt.yticks(np.arange(75,90,2.5))
    plt.grid()
    plt.savefig("learning-curve.jpg")
    print("展示curve")