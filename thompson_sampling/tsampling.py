# from pylab import figure, axes, pie, title, show
#import sys
#sys.path.append("/Library/Frameworks/Python.framework/Versions/2.7/lib/")
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def ThompsonSampling (mu1, mu2, mu3 = 0.5, T = 1000) :


    # define time horizon


    # calculate the given loss arrays
    arm1_cost = np.random.binomial(1, mu1, (1,T))
    arm2_cost = np.random.binomial(1, mu2, (1,T))
    arm3_cost = np.random.binomial(1, mu3, (1,T)) * 3

    arm_costs = np.array([arm1_cost, arm2_cost, arm3_cost])

    # draw from beta distribution
    avg_loss = np.zeros(T)
    loss_arr = np.zeros(T)

    # Initialize parameters to uniform over all
    param1 = np.array([1,1])
    param2 = np.array([1,1])
    param3 = np.array([1,1])

    for i in range(T) :
        w1 = np.random.beta(param1[0], param1[1])
        w2 = np.random.beta(param2[0], param2[1])
        w3 = np.random.beta(param3[0], param3[1])

        # pick arm that minizmized the expected loss (or minimizes reward) i.e. with min w parameter

        # Probably easier and more efficient way of extracting arm using indexing ... ignoring for now
        theta = np.array([])
        if (w1 < np.minimum(w2, w3)) :
            arm = 0
        elif (w2 < np.minimum(w1,w3)):
            arm = 1
        elif (w3 < np.minimum(w1,w2)) :
            arm = 2

        # Pick arm 
        arm_array = arm_costs[arm]
        # get loss
        loss = arm_array[0][i]

        # update alpha and beta parameters depending on arm
        if (arm == 0) :
            param1 += np.array([loss, 1 - loss])
        elif (arm == 1) : 
            param2 += np.array([loss, 1 - loss])
        else :
            param3 += np.array([loss, 3 - loss])

        # Define loss vector
        loss_arr[i] = loss    
        # get the  average loss
        avg_loss[i] = (loss + np.sum(loss_arr)) / (i + 1)
    return avg_loss

#avg_loss_ts = ThompsonSampling(0.3, 0.2)
#plt.title('Avg loss with T.S.')
#plt.xlabel('T rounds')
#plt.ylabel('avg loss')

#plt.plot(avg_loss_ts)
#plt.show()