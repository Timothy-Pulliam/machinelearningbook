Terminology
==================================================

Sample
   A particular instance of data. For example, one email, one customer, etc.
   Below a dataset is shown with four different ways of representing a sample.

   .. list-table:: Dataset
      :header-rows: 1

      * - Sample #
        - Feature 1
        - Feature 2
        - Feature 3
        - Label
      * - Sample 1
        - :math:`x_{11}`
        - :math:`x_{12}`
        - :math:`x_{13}`
        - :math:`y_1`
      * - Sample 2
        - :math:`x_{21}`
        - :math:`x_{22}`
        - :math:`x_{23}`
        - :math:`y_2`
      * -
        - :math:`\vdots`
        - :math:`\vdots`
        - :math:`\vdots`
        - :math:`\vdots`
      * - Sample n
        - :math:`x_{n1}`
        - :math:`x_{n2}`
        - :math:`x_{n3}`
        - :math:`y_n`

   .. list-table:: Email Dataset
      :header-rows: 1

      * - Email Subjects
        - SMTP Server Source IP Address
        - Email Body Content
        - Email Spam or Not Spam
      * - This one weird trick!
        - 10.5.6.7
        - spam, spam spamity spam!
        - spam


   .. list-table:: Advertisement Dataset
      :header-rows: 1

      * - TV Advertisement ($s)
        - Radio Advertisement ($s)
        - Newspaper Advertisement ($s)
        - Sales ($1000s)
      * - 500
        - 400
        - 150
        - 20

Feature
   A feature is an input variable describing the data. We will usually have multiple
   features for a sample. We write the features as a vector :math:`\vec{x_i} = (x_{i1}, x_{i2}, \ldots, x_{im})`
   where :math:`\vec{x_i}` is the feature vector of the ith sample. Each sample has `m`
   features

   Thus we can write a feature matrix which contains all of the samples like so

   .. math::
      X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \\ \end{bmatrix} =
      \begin{bmatrix} x_{11} & x_{12} & \ldots & x_{1m} \\
                      x_{21} & x_{22} & \ldots & x_{2m} \\
                      \vdots & \vdots & \ddots & \vdots \\
                      x_{n1} & x_{n2} & \ldots & x_{nm} \\
       \end{bmatrix}

   If our Model was to classify customers based on their purchasing habits, we might use the following features

   * clothing size
   * customer clicks on description
   * customer adds to cart
   * Time of day purchase was made
   * Items that were looked at, but not purchased
   * amount of money spent

   In this way, we could group similar customers together to provide them items
   which they would be most interested in. This process is known as `Feature Engineering <https://en.wikipedia.org/wiki/Feature_engineering>`_
   and is integral to making accurate Models.

Label
    A label is the thing we're predicting—the y variable in simple linear regression.
    If the features are the input of a function, the Label is the desired output.
    Expressed Mathematically, the ith label is calulated by using the ith
    feature vector.


    .. math::

      y(\vec x_i) = y_i

    Applying the function to all samples gives every label

    .. math::

      y(X) = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}

Labeled Sample
    A sample where the label, :math:`y_i`, is known. Used to train the model
    since the relationship :math:`y(\vec x_i) = y_i` is known.
    Label could be "spam" or "not spam" for email.

    Has {features, label}: :math:`(x_i, y_i)`.

Unlabeled Sample
    A sample where the label, :math:`y_i`, is not known. We would like to predict
    the Label.

    Has {features, ?}: :math:`(x_i, ?)`. Used for making predictions on new data.

Model
    A model maps Samples to predicted Labels :math:`\hat y_i`. The model is defined
    by internal parameters which are determined/learned by using Labeled Samples as
    Training Data. Parameters that must be supplied by the Programmer and are not learned
    by the model are called Hyperparameters.

Regression Model
    A regression model predicts continuous (Real) values. For example, regression
    models make predictions that answer questions like the following:

      * What is the value of a house in California?
      * What is the probability a user clicks on an ad?
      * How are Sales effected by Advertisement?

Classification Model
    Predicts what class or category a case falls in. Classification
    is used when the class label is of data type `Nominal or Ordinal <https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-interval-variables/>`_
    Classification is used for discrete values of labels (finite, or countably infinite values).

    When there are two classes/targets, the model is said to be a `Binary Classification Model <https://en.wikipedia.org/wiki/Binary_classification>`_.
    When there are three or more classes a target can be classified as,
    the model is said to be a `Multiclass Classifcation Model <https://en.wikipedia.org/wiki/Multiclass_classification>`_.
    Examples include:

      * Is this email spam or not spam?
      * Is a tumor malignant or benign?
      * What type of flower is the sample?
      * Is a system healthy or not?

Parameter
    pass

Hyperparameters
    pass
