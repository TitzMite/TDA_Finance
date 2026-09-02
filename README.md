# Topological Analysis of Financial Market Stress

This project provides an educational introduction to applying **Topological Data Analysis (TDA)** to financial time-series data.

Daily returns of 24 large German stocks are grouped into rolling 100-day windows and interpreted as point clouds in $\mathbb{R}^{24}$. Persistent homology is then used to study topological features in dimensions $H_1$ and $H_2$.

The resulting persistence diagrams are analyzed in both raw and normalized form and compared with two classical measures of market stress:

- average stock volatility;
- average pairwise stock correlation.

Persistence diagrams are also transformed into heatmaps and used as inputs to simple convolutional neural networks.

The project is intended for **educational and exploratory purposes**. It does not aim to develop a trading strategy or establish TDA as a superior measure of financial market stress.

## Workflow

The main analysis follows the pipeline:

```text
stock prices
    ↓
daily returns
    ↓
rolling 100-day windows
    ↓
24-dimensional point clouds
    ↓
persistent homology
    ↓
H1 and H2 persistence diagrams
    ↓
raw and normalized representations
    ↓
TDA stress measures and persistence heatmaps
    ↓
statistical and machine-learning analysis
```

## Data

Historical daily stock prices are downloaded using `yfinance`.

The dataset contains 24 large German companies and covers the period from January 2005 to June 2026.

For each trading day, the percentage returns of the 24 stocks form a point in $\mathbb{R}^{24}$. Rolling windows of 100 trading days therefore produce point clouds containing 100 points each.

## Persistent Homology

Persistent homology is calculated for every rolling point cloud using `ripser`.

Two homology dimensions are considered:

- $H_1$: one-dimensional features such as loops;
- $H_2$: two-dimensional features such as cavities.

Each feature has a birth value $b_i$ and a death value $d_i$. Its persistence is

$$
p_i = d_i - b_i.
$$

In this project, persistence-diagram points are represented as

$$
(b_i,p_i).
$$

## Normalized Persistence Diagrams

The geometric scale of the point clouds can vary over time. To reduce this scale effect, each persistence diagram is normalized by the median pairwise Euclidean distance of its corresponding point cloud.

For a point cloud $X$, the median pairwise Euclidean distance is

$$
D_{\mathrm{median}}
=
\operatorname{median}_{i<j}
\left\|
\mathbf{x}_i-\mathbf{x}_j
\right\|_2.
$$

Each persistence point is transformed as

$$
(b_i,p_i)
\rightarrow
\left(
\frac{b_i}{D_{\mathrm{median}}},
\frac{p_i}{D_{\mathrm{median}}}
\right).
$$

Both raw and normalized persistence diagrams are retained in the analysis.

## TDA Stress Measures

A simple scalar TDA stress measure is constructed using the $L^2$ norm of the persistence values:

$$
S =
\sqrt{
\sum_i p_i^2
}.
$$

Four TDA-based stress measures are considered:

- $H_1$ raw stress;
- $H_1$ normalized stress;
- $H_2$ raw stress;
- $H_2$ normalized stress.

These are compared with average stock volatility and average pairwise stock correlation.

For visual comparison, the time series are standardized so that measures with different numerical scales can be displayed together. The correlation analysis itself uses the original, unstandardized values.

## Persistence Heatmaps

Persistence diagrams are also transformed into continuous two-dimensional heatmaps.

Each persistence point contributes a Gaussian-shaped region centered at its birth-persistence coordinates. More persistent features receive a larger weight.

Heatmaps are constructed for:

- $H_1$ raw persistence diagrams;
- $H_1$ normalized persistence diagrams;
- $H_2$ raw persistence diagrams;
- $H_2$ normalized persistence diagrams.

A common birth-persistence region is used for all windows of the same heatmap type, making the heatmaps comparable over time.

## CNN Analysis

The persistence heatmaps are treated as single-channel images and used as inputs to convolutional neural networks.

The four heatmap types are each compared with two target variables:

- average stock volatility;
- average stock correlation.

This results in a total of **eight CNN experiments**.

The observations are divided chronologically into training, validation, and test periods. A gap of 100 trading days is placed between these periods to reduce overlap between the rolling windows.

The CNN analysis is not intended as a forecasting exercise. The target variables describe the same 100-day windows as the corresponding persistence heatmaps.

## Results

The strongest statistical relationship is found between **raw $H_1$ stress and volatility**, with a Pearson correlation of approximately $0.61$ and a Spearman correlation of approximately $0.57$.

After normalization, this relationship becomes substantially weaker. This suggests that part of the relationship between raw $H_1$ persistence and volatility is associated with the geometric scale of the return point cloud.

The relationships between the TDA stress measures and average stock correlation are generally weak.

The CNN results are mixed. Some $H_1$ and raw $H_2$ heatmaps show moderate relationships with volatility, but all eight CNN models produce negative $R^2$ values on the test set. The machine-learning results should therefore be interpreted as **exploratory rather than predictive**.

## Project Structure

```text
.
├── computing_table.py   # Builds the analysis table
├── backend.py           # Plotting and machine-learning helper functions
├── tda_finance.ipynb    # Main analysis and explanation
└── README.md
```

## Requirements

The main Python packages used in the project are:

```text
numpy
pandas
scipy
yfinance
ripser
matplotlib
tensorflow
scikit-learn
```

They can be installed with:

```bash
pip install numpy pandas scipy yfinance ripser matplotlib tensorflow scikit-learn
```

## Running the Project

Before opening the notebook, run:

```bash
python computing_table.py
```

This creates the `analysis_table` used by `backend.py` and the notebook.

Then start Jupyter:

```bash
jupyter notebook
```

Open `tda_finance.ipynb` and run the notebook cells in order.

## AI Assistance

This project was developed with assistance from **ChatGPT by OpenAI**.

ChatGPT was used to help review the project structure, improve explanations and documentation, refine code organization, and discuss the interpretation and presentation of the results.

## Disclaimer

This project is intended for **educational and exploratory purposes only**.

The TDA-based measures and neural-network experiments presented here should not be interpreted as validated financial indicators, forecasting models, trading signals, or investment advice.
