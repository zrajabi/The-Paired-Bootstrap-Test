# The-Paired-Bootstrap-Test
Implementation of Paired Bootstrap Test for an NLP task

In order to compare the performance of different models, we perform a statistical significance test. There are two non-parametric tests commonly used in NLP: approximate randomization and the bootstrap test. The bootstrap test is a paired test, in which we compare two sets and can be applied to compare any metric, such as accuracy, precision, recall or F1 scores. 

In this code, we apply this test to macro-F1 score to compare the performance of two models, let's say A and B baseline BERT models. Focusing on macro-F1 score, we set the following null hypothesis H0: A does no better than B on the test dataset in terms of the macro-F1 dataset. To evaluate whether we can reject the null hypothesis, we proceed as follows:
We first compute delta(D), which is the performance difference between model A and model B on the test set D. This is used as reference. Then, the bootstrap test starts by repeatedly sampling from the test dataset under the assumption that each sample (of the test dataset D) is representative of the whole population (the whole test dataset D). 
The bootstrap test creates b samples of the reference dataset D. Each sample contains n instances, which are sampled uniformly at random and with replacement from the reference dataset D. That is, each of the b samples contain n instances drawn at random and with replacement from the test dataset D in our case. The macro-F1 score of model A and model B is computed on each of the samples xi, where 1 <= i <= b, and the difference delta(xi) = macro-F1(A, xi) - macro-F1(A, xi) is computed. Every time delta(xi) > 2 delta(D), an integer s is incremented. The value s=b counts in what percentage of the b samples, the difference in performance between models A and B exceeds (by a factor of two) that on the full test dataset. The value s=b reports on what %of the b samples model A beat expectations and acts as a one-sided empirical p-value. The intuition is that if very few of the samples beat expectations (that is, the p-value is small), then the observed delta(D) is probably not accidental; hence, p-value is small.

Reference:

J. M. D Jurafsky, Speech and language processing (3rd edition). Prentice Hall, 2019.
https://web.stanford.edu/~jurafsky/slp3/4.pdf
