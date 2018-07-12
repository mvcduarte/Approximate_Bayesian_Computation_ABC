"""
  This code presents the Approximate Bayesian Computation (ABC)
  algorithm to make a linear fitting. First, it draws some mock data
  and samples the parameter space to define best parameter values in order 
  to select the best candidates to reproduce the observed data. Note that
  this technique does not use explicitly a likelihood (e.g. Chi2 reduction). 

                       Marcus Costa-Duarte, 12/07/2018

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def generate_mock_data(n_points, a0, b0):
    """
     Generate some mock data (linear)
    """
    x = np.linspace(0., 1., n_points)
    epsilon = np.random.normal(0., 0.2, n_points) # gaussian noise
    y = (a0 * x + b0) + epsilon 

    return x, y

def draw_sample(n, a_range, b_range, sigma):
    """
    Draw n parameters sets of a and b
    """

    # Assuming FLAT priors for a and b parameters

    a_draw = np.random.uniform(a_range[0], a_range[1], n)
    b_draw = np.random.uniform(b_range[0], b_range[1], n)

    # Assuming Gaussian priors centered in the middle of parameter space

    #a_draw = np.random.normal(np.average(a_range), sigma, n)
    #b_draw = np.random.normal(np.average(b_range), sigma, n)

    dist_draw = np.zeros(len(a_draw))
    for i in range(len(a_draw)):
        y_draw = a_draw[i] * x + b_draw[i]
        dist_draw[i] = sum(abs(y_draw - y))

    return a_draw, b_draw, dist_draw

def select_smallest_dist(a_draw, b_draw, dist_draw, threshold):
    """
     Select the set with smallest distances
    """

    # Select elements with smallest distances (in threshold percentile units)

    idx = np.where(dist_draw < np.percentile(dist_draw, threshold))[0]
    a_best = a_draw[idx]
    b_best = b_draw[idx]

    # Pack them into arrays 

    a_fit = [np.average(a_best), np.std(a_best)]
    b_fit = [np.average(b_best), np.std(b_best)]

    return a_fit, b_fit, a_best, b_best

###########################################################

matplotlib.rcParams.update({'font.size': 22})

n_draw = 50000
a0 = 2.
b0 = 1.5
a_range = [0., 3.]
b_range = [0., 3.]

n_points = 200

threshold = 1.5 # in percentage

n_posterior = 200

np.random.seed(123456789)

############################################################

if __name__ == '__main__':

    # Generate mock data 

    x, y = generate_mock_data(n_points, a0, b0)

    f = plt.figure(figsize=(15,8))
    plt.plot(x, y, 'o', label = 'input', alpha = 1.0) # noisy data
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0., 1.1)
    plt.ylim(0., 5.1)
    plt.grid()
    plt.legend(loc = 0)
    plt.savefig('input.png', bbox_inches = 'tight')
    plt.clf()
    plt.cla()

    # The larger sigmas, the closer priors are to uniform

    sigma_array = [0.2]
    i_sigma = 0

    a_array = np.zeros(2 * len(sigma_array)).reshape(len(sigma_array), 2)
    b_array = np.zeros(2 * len(sigma_array)).reshape(len(sigma_array), 2)

    # Draw a large number of parameter set

    a_draw, b_draw, dist_draw = draw_sample(n_draw, a_range, b_range, sigma_array[i_sigma])

    # Get the "best" elements of a_draw and b_draw (lowest distances)

    a_fit, b_fit, a_best, b_best = select_smallest_dist(a_draw, b_draw, dist_draw, threshold)

    print('sigma', sigma_array[i_sigma])
    print("a_mean: %f a_std: %f" % (a_fit[0], a_fit[1]))
    print("b_mean: %f b_std: %f" % (b_fit[0], b_fit[1]))
    a_array[i_sigma, :] = a_fit
    b_array[i_sigma, :] = b_fit

    # Plotting...

    f = plt.figure(figsize=(15,8))
    f.subplots_adjust(wspace=0.3, hspace=0)

    bins_a = np.linspace(a_range[0], a_range[1], 50)
    bins_b = np.linspace(b_range[0], b_range[1], 50)

    # Distance histogram
    ax1 = f.add_subplot(131)
    ax1.hist(dist_draw, bins = 50)
    ax1.set_xlabel('distance')
    ax1.set_ylabel('N')
    ax1.grid()

    # a value
    ax2 = f.add_subplot(132)
    ax2.hist(a_best, bins = bins_a, color = 'red', alpha = 0.3, label = 'a')
    ax2.set_ylabel('N')
    ax2.set_xlabel('a value')
    ax2.axvline(x=a_fit[0], linestyle = '--', color='black', lw = 2.)
    ax2.axvline(x=a0, linestyle = '--', color='red', lw = 2.)
    ax2.grid()

    # b value
    ax3 = f.add_subplot(133)
    ax3.set_ylabel('N')
    ax3.hist(b_best, bins = bins_b, color = 'blue', alpha = 0.3, label = 'b')
    ax3.set_xlabel('b value')
    ax3.axvline(x=b_fit[0], linestyle = '--', color='black', lw = 2.)
    ax3.axvline(x=b0, linestyle = '--', color='red', lw = 2.)
    ax3.grid()

    plt.savefig('dist_a_b_' + str(sigma_array[i_sigma]) + '.png', bbox_inches = 'tight')
    plt.clf()
    plt.cla()

    # Plot linear regression using Approximation Bayesian Model (ABC)

    plt.plot(x, y, 'o', alpha = 1.0) # noisy data
    for i in range(n_posterior):
        a_new = a_fit[0] + np.random.normal(0., 1., 1) * a_fit[1]
        b_new = b_fit[0] + np.random.normal(0., 1., 1) * b_fit[1]
        if i < n_posterior-1: 
            plt.plot(x, (a_new  * x + b_new), lw = 2., color = 'red', alpha = 0.1) # original trend
        else:
            plt.plot(x, (a_new  * x + b_new), lw = 2., color = 'red', alpha = 0.1, label = 'ABC posterior') # original trend

    plt.plot(x, a0 * x + b0, lw = 5., color = 'black', label = 'input', alpha = 0.5) # original trend
    plt.plot(x, a_fit[0] * x + b_fit[0], lw = 5., color = 'blue', label = 'fit', alpha = 0.5) # fit 

    plt.xlabel('X')
    plt.ylabel('Y')

    str_a = ("a_fit: %.2f +- %.2f (%.2f)" % (a_fit[0], a_fit[1], a0))
    str_b = ("b_fit: %.2f +- %.2f (%.2f)" % (b_fit[0], b_fit[1], b0))
    plt.text(0.5, 1.5, str_a)
    plt.text(0.5, 1., str_b)
    plt.xlim(0., 1.1)
    plt.ylim(0., 5.1)

    plt.grid()
    plt.legend(loc = 0)
    plt.savefig('fitting_' + str(sigma_array[i_sigma]) + '.png', bbox_inches = 'tight')
    plt.clf()
    plt.cla()

    exit()