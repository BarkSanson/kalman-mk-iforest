# Online Isolation Forest with Concept Drift Detection

## Introduction
There are many models to detect outliers in unidimensional or multidimensional data.
These models usually work only on static environments. Thus, they are not fitted to
work in streaming environments where data arrives one by one. On the other hand,
when working on an online environment the data distribution might change over time,
causing AI models to degrade their performance. This data distribution change
is known as _concept drift_ and its detection is crucial to be able to adapt models
accordingly.

One of the most famous outlier detection techniques is Isolation Forest, which
is based on building complete binary trees to isolate outliers and thus detecting
them. This model is not suited to work on online environments do to concept drift
and other causes.

On the other hand, many concept drift detection techniques are used. For instance,
one methodology of detecting concept drift consists on using statistical hypothesis
tests to detect drift based on the input data distribution. Normally, these techniques
use two windows: one reference window and one online window. These two are compared
using and hypothesis test, which will be able to say whether concept drift has occured
or not. Some examples of tests are the Kolmogorov-Smirnov test, Mann-Whitney U, Wilcoxon
Signed-Rank or Page-Hinkley.

This implementation explores using two tests to detect concept drift, Mann-Kendall
and Wilcoxon Signed-Rank Test, and Isolation Forest to detect outliers. 


## Design
In this code there are two main designs. On one hand, a design based on batches of data.
This design waits to fill a window of length $n$ to perform both concept drift detection and
outlier detection. On the other hand, a design based on sliding window is implemented. This
other design is based on using a sliding window to perform detection. The operation of this
window is the same as a queue: every time a new data point arrives, the oldest data point
is removed from the queue. Then, both concept drift and outlier detection are performed. The
anomaly detection is only done over this last data point.

As can be seen in the `online_outlier_detection` directory, 4 different implementations
have been done. Two of them follow the design mentioned in the previous paragraph, which are
`mkwiforestbatch.py` and `mkwiforestsliding.py`. The other two, namely `mkwkiforestbatch.py` and
`mkwkiforestsliding.py` use the Kalman filter before performing any detection.

## Usage
If you haven't created one, create a virtual environment with your preferred tool
and activate it. Then, install all dependencies using
```bash
pip install -r requirements.txt
```

Then, you can import the package using the usual
```python
from online_outlier_detection import #<your-import>
```
