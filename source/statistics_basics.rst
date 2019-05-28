Basic Statistics
=========================

need to rewrite all of this

data types (Nominal, ordinal, etc.)
https://stattrek.com/statistics/measurement-scales.aspx?Tutorial=AP

sample vs population

https://stattrek.com/sampling/populations-and-samples.aspx?Tutorial=AP
https://stattrek.com/estimation/standard-error.aspx

sample size n can be greater than population size N (if replacement is enabled)

The Mean vs. the Median
As measures of central tendency, the mean and the median each have advantages and disadvantages. Some pros and cons of each measure are summarized below.

The median may be a better indicator of the most typical value if a set of scores has an outlier. An outlier is an extreme value that differs greatly from other values.
However, when the sample size is large and does not include outliers, the mean score usually provides a better measure of central tendency.

          GIVE EXAMPLE


 Effect of Changing Units
Sometimes, researchers change units (minutes to hours, feet to meters, etc.). Here is how measures of central tendency are affected when we change units.

If you add a constant to every value, the mean and median increase by the same constant. For example, suppose you have a set of scores with a mean equal to 5 and a median equal to 6. If you add 10 to every score, the new mean will be 5 + 10 = 15; and the new median will be 6 + 10 = 16.
Suppose you multiply every value by a constant. Then, the mean and the median will also be multiplied by that constant. For example, assume that a set of scores has a mean of 5 and a median of 6. If you multiply each of these scores by 10, the new mean will be 5 * 10 = 50; and the new median will be 6 * 10 = 60.

measuring spread and variability

range

q1, q2 (median), q3. give image

interquartile range

population variance

σ^2 = Σ ( X_i - μ )^2 / N

where σ2 is the population variance, μ is the population mean, Xi is the ith element from the population, and N is the number of elements in the population.

sample variance

s^2 = Σ ( x_i - x )^2 / ( n - 1 )

where s2 is the sample variance, x is the sample mean, xi is the ith element from the sample, and n is the number of elements in the sample.
Using this formula, the sample variance can be considered an unbiased estimate of the true population variance.

population standard dev

σ = sqrt [ σ2 ]

Statisticians often use simple random samples to estimate the standard deviation of a population, based on sample data. Given a simple random sample, the best estimate of the standard deviation of a population is:

s = sqrt(s^2)


Effect of Changing Units
Sometimes, researchers change units (minutes to hours, feet to meters, etc.). Here is how measures of variability are affected when we change units.

If you add a constant to every value, the distance between values does not change. As a result, all of the measures of variability (range, interquartile range, standard deviation, and variance) remain the same.
On the other hand, suppose you multiply every value by a constant. This has the effect of multiplying the range, interquartile range (IQR), and standard deviation by that constant. It has an even greater effect on the variance. It multiplies the variance by the square of the constant.


plots

histogram, whisker plot, scatterplot

Correlation Coefficient
Correlation coefficients measure the strength of association between two variables. The most common correlation coefficient, called the Pearson product-moment correlation coefficient, measures the strength of the linear association between variables measured on an interval or ratio scale.

In this tutorial, when we speak simply of a correlation coefficient, we are referring to the Pearson product-moment correlation. Generally, the correlation coefficient of a sample is denoted by r, and the correlation coefficient of a population is denoted by ρ or R.

How to Interpret a Correlation Coefficient
The sign and the absolute value of a correlation coefficient describe the direction and the magnitude of the relationship between two variables.

The value of a correlation coefficient ranges between -1 and 1.
The greater the absolute value of the Pearson product-moment correlation coefficient, the stronger the linear relationship.
The strongest linear relationship is indicated by a correlation coefficient of -1 or 1.
The weakest linear relationship is indicated by a correlation coefficient equal to 0.
A positive correlation means that if one variable gets bigger, the other variable tends to get bigger.
A negative correlation means that if one variable gets bigger, the other variable tends to get smaller.

    SHOW SOME PLOTS

Keep in mind that the Pearson product-moment correlation coefficient only measures linear relationships. Therefore, a correlation of 0 does not mean zero relationship between two variables; rather, it means zero linear relationship. (It is possible for two variables to have zero linear relationship and a strong curvilinear relationship at the same time.)

Product-moment correlation coefficient. The correlation r between two variables is:

r = Σ (xy) / sqrt [ ( Σ x^2 ) * ( Σ y^2 ) ]

where Σ is the summation symbol, x = xi - x, xi is the x value for observation i, x is the mean x value, y = yi - y, yi is the y value for observation i, and y is the mean y value.

Population correlation coefficient. The correlation ρ between two variables is:

ρ = [ 1 / N ] * Σ { [ (Xi - μX) / σx ]
* [ (Yi - μY) / σy ] }

where N is the number of observations in the population, Σ is the summation symbol, Xi is the X value for observation i, μX is the population mean for variable X, Yi is the Y value for observation i, μY is the population mean for variable Y, σx is the population standard deviation of X, and σy is the population standard deviation of Y.


Sample correlation coefficient. The correlation r between two variables is:

r = [ 1 / (n - 1) ] * Σ { [ (xi - x) / sx ]
* [ (yi - y) / sy ] }

where n is the number of observations in the sample, Σ is the summation symbol, xi is the x value for observation i, x is the sample mean of x, yi is the y value for observation i, y is the sample mean of y, sx is the sample standard deviation of x, and sy is the sample standard deviation of y.


You can calculate the moments and skewness coefficient using Python

.. code-block:: python

    from statistics import mode, variance, pvariance, stdev, pstdev
    import numpy as np
    from matplotlib import pyplot as plt

    #TODO
    # Identify Nominal, Ordinal and metric data

    def descriptiveAnalysis(x, isSample=True, showOutput=False):
      '''Performs basic analysis on a data set,
         calculates mean, median, standard deviation, etc.

         Inputs
         -------
         x : numpy.array object
             The dataset
         isSample : Boolean (True/False)
             Some statistical calculations depend upon whether
             the data is sample or population data
         showOutput : Boolean (True/False)
             Whether or not to print out the resulting statistics.
             Otherwise, the results will only be returned
             as a dictionary.

         Outputs
         -------
         stats : dictionary
             Dictionary containing the calculated statistics
      '''

      # sample size
      size = x.size

      # range data
      x_min = np.min(x)
      x_max = np.max(x)
      r = x_max - x_min

      # Quartiles
      q1 = np.percentile(x, 25)
      q2 = np.percentile(x, 50)
      q3 = np.percentile(x, 75)
      interquartileRange = q3 - q1

      m = np.mean(x)
    # Mode (most common number) is a robust measure of central location
    # for nominal level data
    mo = mode(x)

    # The median is a robust measure of central location for ordinal level
    # data, and is less affected by the presence of outliers in your data.
    # When the number of data points is odd, the middle data point is
    # returned. When the number of data points is even, the median is
    # interpolated by taking the average of the two middle values
    #
    # This is suited for when your data is discrete, and you don’t mind
    # that the median may not be an actual data point.
    #
    # If your data is ordinal (supports order operations) but not numeric
    # (doesn’t support addition), you should use median_low() or
    # median_high() instead.
    me = np.median(x)


    if isSample:
        ## Sample Variance
        s = stdev(x)
        # Variance, or second moment about the mean, is a measure of the
        # variability (spread or dispersion) of data. A large variance
        # indicates that the data is spread out; a small variance indicates
        # it is clustered closely around the mean.
        v = variance(x) # == stdev**2
    else:
        ## Population Variance
        s = pstdev(x)
        v = pvariance(x)

    # Pearson's second skewness coefficient (median skewness)
    skewCoefficient = 3 * (m - me) / s

    stats = {'size': size, 'mean': m, 'median': me, 'mode': mo,
            'min': x_min, 'max': x_max, 'range': r, 'stdev': s,
            'variance': v, 'q1': q1, 'q2': q2, 'q3': q3,
            'interquartileRange': interquartileRange,
            'skewCoefficient': skewCoefficient,}

    if showOutput:
        print("sample size :       {}".format(size),
              "mean :              {}".format(m),
              "median :            {}".format(me),
              "min value :         {}".format(x_min),
              "max value :         {}".format(x_max),
              "range :             {}".format(r),
              "standard deviation: {}".format(s),
              "variance: {}".format(v),
              "mode: {}".format(mo),
              "Quartiles",
              "Q1: {}".format(q1),
              "Q2: {}".format(q2),
              "Q3: {}".format(q3),
              "Skew Coefficient: {}".format(skewCoefficient), sep='\n')

    return stats

    def boxPlot(x):
        plt.boxplot(x, vert=False, showmeans=True)
        plt.grid(axis='x',linestyle='--')
        plt.show()

    x = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
    descriptiveAnalysis(x,False, True)
    boxPlot(x)
