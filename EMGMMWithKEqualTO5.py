__author__ = 'seanhendryx'

# Sean Hendryx
# Script built to run on Python version 3.4
# References:
# scipy.cluster.vq.kmeans: Centroids used to initialize mu vectors  <http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html#scipy.cluster.vq.kmeans>

#NOTES: Test on very small subset of image pixels.  Actual test image has 1,973,905 pixels (i.e. observations (N))

import numpy
from scipy.cluster.vq import kmeans
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def main():
    """

    Terminology:
    df:= dataframe (observations, commonly denoted bold X)
    mu := vector of means for cluster/class k
    covar := array of covariance matrix for each cluster
    weights := vector of mixture weights for each class and must sum to 1
    k := number of clusters
    gamma := the responsibility of component k for data point xn (vector of length N (number of observations).
    :return:

    Psuedocodeish:
    #Evaluate on k clusters for k = 2,3,4,7,10
    k = 2

    initialize(df, mu, covar, weights, k)

    #Initialize evaluation values:
    oldLikelihood = (-1 * numpy.inf)
    newLikelihood = oldLikelihood
    step = 0
    iterations = 1000

    #Make 2d arrays of likelihood and iterations (step, l) for training (visL) and test (visValidation) data
    visL = numpy.zeros((1000, 2))
    visValidation = numpy.zeros((1000, 2))
    while newLikelihood > (oldLikelihood + .1) and step < iterations #or some other (finer) threshold
        gamma = expectation(df, weights, mu, covar, k)

        mu, covar, weights = maximize(df, gamma, k)


        newLikelihood = evalLikelihood(trainingSet, weights, mu, covar, k)
        validationLikelihood = evalLikelihood(validationSet, weights, mu, covar, k)

        step += 1
        l = newLikelihood


        visL[step,:] = (step, l)
        visValidation[step,:] = (step, validationLikelihood)

    x = visL[:,0]
    y = visL[:,1]
    plt.figure(1)
    plt.plot(x, y, 'bo')
    plt.title("Log Likelihood Progress Over {} iterations".format(iterations))
    plt.xlabel('L')
    plt.ylabel("Iteration")


    """
    #Specify largest number of classes + 1 (e.g. 11 yields test on 10 classes)
    mostClasses = 6
    visLk = numpy.zeros((mostClasses, 2))
    #SPECIFY NUMBER OF DIMENSIONS (3 in this case (RGB)):
    dimensions = 3
    df = numpy.loadtxt('mesquitesSubsetPixelData.csv', delimiter=',')

    #Split data by every nth element (number stored in fold variable), i.e. putting .1 into test dataset and .9 into training
    fold = 10
    testdf = df[::fold,:]
    #wholeSet = df
    #Training dataset with test data removed:
    df = numpy.delete(df, list(range(0, df.shape[0], fold)), axis = 0)

    for c in numpy.arange(start = 5, stop = mostClasses):

        #SPECIFY NUMBER OF CLUSTERS/CLASSES:
        k = c

        mu, covar, weights = initialize(df, k, dimensions)
        #print(mu, covar, weights)
        print("Initialized mu: ", mu)

        step = 0
        #NUMBER OF ITERATIONS:
        iterations = 11
        #Make 2d arrays of likelihood and iterations (step, l) for training (visL) and test (visValidation) data
        visL = numpy.zeros((iterations, 2))
        #visValidation = numpy.zeros((1000, 2))
        while step <iterations:
            gamma = expectation(df, weights, mu, covar, k)

            mu, covar, weights = maximize(df, weights, mu, covar, gamma, k)

            """
            print("new mu: ", mu)
            print('new covariance matrices:', covar)
            print('new weights/mixing coefficients for k: ', weights)
            """
            newLikelihood = evalLikelihood(df, weights, mu, covar, k)

            print('Iteration #:', step)
            print ('L: ', newLikelihood)

            l = newLikelihood
            visL[step,:] = (step, l)
            #visValidation[step,:] = (step, validationLikelihood)

            step += 1

        x = visL[:,0]
        y = visL[:,1]

        plt.figure(k)
        plt.plot(x, y)
        plt.title("Log Likelihood Progress Over {} Iterations and {} Classes".format(iterations-1, k))
        plt.xlabel('Iteration')
        plt.ylabel("L")


        testL = evalLikelihood(testdf, weights, mu, covar, k)
        visLk[c,:] = (c,testL)

    x= visLk[:,0]
    y = visLk[:,1]
    plt.figure(k+1)
    plt.plot(x,y)
    plt.title('Log Likelihood on Validation Dataset as a Function of Number of Classes')
    plt.xlabel('K')
    plt.ylabel('L')
    plt.show()

def initialize(df, k, dimensions):
    """
    Initialize mean vector (mu) with kmeans
    :param df:
    :param mu:
    :param covar:
    :param weights:
    :return:
    """

    #initialize cluster centers with kmeans:
    centroids,_ = kmeans(df,k)
    mu = centroids

    covarShape = (dimensions, dimensions, k)
    covar = numpy.zeros(covarShape)
    for i in numpy.arange(k):
        covar[:,:,i] = numpy.identity(dimensions)

    weights = numpy.zeros(k)
    for i in numpy.arange(k):
        weights[i] = 1./k

    return mu, covar, weights



def expectation(df, weights, mu, covar, k):
    """
    Computes the responsibility of component k for data point xn (from each observation in df).
    :param df:
    :param weights:
    :param mu:
    :param covar:
    :param k:
    :return: responsibility of component k for data point xn in matrix with rows containing observations (xns) and
    columns containing classes (ks)
    """
    #initialize gamma array:
    #the responsibility of component k for data point xn
    #k column array of n rows
    gamma = numpy.zeros((df.shape[0],k))

    for kth in numpy.arange(k):
        #numerator
        muVec = mu[kth]
        c = covar[:,:,kth]

        numerator = numpy.multiply(weights[kth],multivariate_normal.pdf(df, muVec, c))
        runSum = numpy.zeros_like(numerator)
        for jth in numpy.arange(k):
            muj = mu[jth]
            cj = covar[:,:,jth]
            jthDenominator = numpy.multiply(weights[jth],multivariate_normal.pdf(df, muj, cj))
            runSum += jthDenominator

        denominator = runSum
        denominator = numpy.nan_to_num(denominator)
        numerator = numpy.nan_to_num(numerator)
        g = numpy.divide(numerator, denominator)
        g = numpy.nan_to_num(g)
        gamma[:,kth] = g

    return gamma





def maximize(df, weights, mu, covar, gamma, k):
    """

    :param df: input dataframe (observations of d columns and n rows)
    :param weights: mixing coefficients of k classes
    :param mu: input mean array, where each row represent the mean of clustuer k
    :param covar: covariance array of shape (dimensions, dimensions, k)
    :param gamma: responsibility of k for xn
    :param k: scalar number of components
    :return:  new mu, new stack of covariance matrices (covariance array of shape (dimensions, dimensions, k)), and new weights
    """

    Nk = numpy.sum(gamma, 0)
    N = df.shape[0]

    #FIND NEW MEANS
    # Note gamma: rows containing observations (xns) and columns containing classes (ks)
    muNew = numpy.zeros_like(mu)
    """

    for kth in numpy.arange(k):
        prodsArrayShape = (df.shape[0], df.shape[1])
        prodsArray = numpy.zeros(prodsArrayShape)
        for nth in numpy.arange(N):
            prod = gamma[nth, kth] * df[nth,:]
            prodsArray[nth,:] = prod

        sum = numpy.sum(prodsArray, 0)
        muNew[kth] =  sum * (1./Nk[kth])

    Old attemp to fix issues with dimensions^
    """
    for kth in numpy.arange(k):
        kthgamma = gamma[:,kth]
        kthgamma = kthgamma[:,numpy.newaxis]
        prod = kthgamma * df
        sum = numpy.sum(prod, axis = 0)
        muVec = numpy.multiply((1/Nk[kth]), sum)
        muVec = numpy.nan_to_num(muVec)
        muNew[kth] = muVec

    #FIND NEW COVARIANCE MATRICES
    covarNew = numpy.zeros_like(covar)
    prod = 0.
    sum = 0.
    for kth in numpy.arange(k):
        distance = df - muNew[kth]
        stackedCovarsShape = (df.shape[1], df.shape[1], df.shape[0])
        stackedCovars = numpy.zeros(stackedCovarsShape)

        for nth in numpy.arange(N):
            outerProd = numpy.outer(distance[nth], distance[nth])
            prod = gamma[nth, kth] * outerProd
            stackedCovars[:,:,nth] = prod

        sum = numpy.sum(stackedCovars, axis = 2)
        covarNew[:,:,kth] = sum * (1./Nk[kth])
        covarNew = numpy.nan_to_num(covarNew)



    #FIND NEW MIXING COEFFICIENTS
    weightsNew = numpy.zeros_like(weights)
    for kth in numpy.arange(k):
        weightsNew[kth] = Nk[kth]/N


    return muNew, covarNew, weightsNew


def evalLikelihood(df, weights, mu, covar, k):
    """
    Computes the log (base 10) of likelihood of p(X|mu, covariance matrices, and weights)
    input parameters are same as above
    :param df:
    :param weights:
    :param mu:
    :param covar:
    :param k:
    :return: scalar log liklihood
    """

    runSum = numpy.zeros(df.shape[0])
    for kth in numpy.arange(k):
        muk = mu[kth]
        ck = covar[:,:,kth]
        prod = numpy.multiply(weights[kth],multivariate_normal.pdf(df, muk, ck))
        runSum += prod

    log = numpy.log10(runSum)
    logLikelihood = numpy.sum(log)

    return logLikelihood




def mvNormal(x, mu, covarianceMatrix):
    """
    :param x: Input values (arrays) to use in the calculation of the multivariate Gaussian probability density
    :param mu: mean
    :param covarianceMatrix: input covariance matrix that determines how the variables covary
    :return: The probability densities from the multivariate normal distribution
    """
    #k = the dimension of the space where x takes values
    k = x.shape[0]
    #distance is the "distance between x and mean:
    distance = x-mu
    #Covariance matrix as specified in assignment prompt:

    firstFragment = numpy.exp(-0.5*k*numpy.log(2*numpy.pi))
    secondFragment = numpy.power(numpy.linalg.det(covarianceMatrix),-0.5)
    thirdFragment = numpy.exp(-0.5*numpy.dot(numpy.dot(distance.transpose(),numpy.linalg.inv(covarianceMatrix)),distance))

    multivariateNormalDistribution = firstFragment*secondFragment*thirdFragment
    return multivariateNormalDistribution

# Main Function
if __name__ == '__main__':
    main()