Overview
--------

This repository contains an educational notebook exploring how Topological Data Analysis (TDA) can be applied to financial market data.

The project studies rolling windows of daily stock returns. Each trading day is represented as a point in a high-dimensional space whose coordinates are the returns of the selected stocks. These rolling windows therefore form point clouds whose geometry changes over time.

Persistent homology is used to extract topological features from these point clouds. The resulting TDA-based measures are then compared with classical market indicators, in particular stock volatility and average correlation.

The main question is:

    How do topological features of stock-return data change over time, and how are they related to conventional measures of market behavior?

The goal is not to establish a new financial market indicator, but to explore what information persistent homology captures and how it differs from more traditional statistical measures.



Method
------

For each rolling window of 100 trading days:

1. Daily stock returns are collected into a point cloud.
2. Persistent homology is computed using a Vietoris-Rips filtration.
3. H1 and H2 persistence diagrams are extracted.
4. Each persistence diagram is summarized by an L2 persistence score:

   S_raw = sqrt(sum_i p_i^2),

   where p_i is the persistence of a topological feature.

5. A normalized TDA stress measure is also computed by dividing the raw score by the median pairwise Euclidean distance of the point cloud.

   This reduces the influence of the overall geometric scale of the market return cloud.


Classical Market Measures
-------------------------

The TDA measures are compared with two conventional indicators.

Average stock volatility:
For every stock, the sample standard deviation of its returns is computed within the rolling window. These individual volatilities are then averaged.

Average stock correlation:
The Pearson correlation is computed for every distinct pair of stocks within the rolling window. The pairwise correlations are then averaged.


Visualization
-------------

For visualization, all time series can be standardized using z-scores so that measures with different numerical scales can be compared on the same plot.

The interactive Plotly visualization contains:

- H1 raw stress
- H1 normalized stress
- H2 raw stress
- H2 normalized stress
- average stock volatility
- average stock correlation

Individual curves can be shown or hidden interactively.


Correlation Analysis
--------------------

Pearson and Spearman correlations are used to compare the TDA-based measures with the classical indicators.

Current observations include:

- Raw H1 stress has a substantial positive relationship with volatility.
  Pearson correlation: approximately 0.61
  Spearman correlation: approximately 0.57

- After normalization, the relationship between H1 stress and volatility becomes weak.
  Pearson correlation: approximately -0.25
  Spearman correlation: approximately -0.09

- H2-based measures show only weak correlations with volatility.

- None of the TDA measures currently shows a strong relationship with average stock correlation.

These results should not be interpreted as evidence of a new market-stress indicator. The clearest current finding is methodological: raw persistence is strongly affected by the overall scale of the return point cloud, while normalization removes much of this dependence.

The normalized TDA measures therefore appear to describe aspects of market geometry that are different from simple volatility and average correlation, but the present analysis does not establish that these differences have predictive or economic value.

Disclaimer
----------

This repository is an educational data-analysis project.

It is not financial advice and should not be used for investment decisions. The methods, data preparation, statistical comparisons, and possible machine-learning experiments are exploratory rather than production-grade.

AI Assistance
-------------

ChatGPT was used as an assistant during the development of this project, including discussions of methodology, Python implementation, visualization, and documentation. The mathematical choices, experiments, interpretation of results, and final project structure were reviewed and developed by the author.
