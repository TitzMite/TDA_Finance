
import numpy as np
import pandas as pd
import yfinance as yf
from ripser import ripser
from scipy.spatial.distance import pdist

tickers = [
    "ADS.DE",   # Adidas
    "ALV.DE",   # Allianz
    "BAS.DE",   # BASF
    "BAYN.DE",  # Bayer
    "BEI.DE",   # Beiersdorf
    "BMW.DE",   # BMW
    "CBK.DE",   # Commerzbank
    "CON.DE",   # Continental
    "DBK.DE",   # Deutsche Bank
    "DTE.DE",   # Deutsche Telekom
    "FRE.DE",   # Fresenius
    "FME.DE",   # Fresenius Medical Care
    "G1A.DE",   # GEA Group
    "HNR1.DE",  # Hannover Rück
    "HEI.DE",   # Heidelberg Materials
    "HEN3.DE",  # Henkel
    "MBG.DE",   # Mercedes-Benz Group
    "MRK.DE",   # Merck KGaA
    "MUV2.DE",  # Munich Re
    "QIA.DE",   # Qiagen
    "RWE.DE",   # RWE
    "SAP.DE",   # SAP
    "SIE.DE",   # Siemens
    "VOW3.DE",  # Volkswagen
]

data = yf.download(
    tickers,
    start="2005-01-01",
    end="2026-06-30"
)

close_prices = data["Close"]

returns = close_prices.pct_change() * 100
returns.iloc[0] = 0

#############

point_cloud_size = 100

window_dates = []
point_clouds = []

for start in range(len(returns) - point_cloud_size + 1):
    end = start + point_cloud_size
    window_dates.append(returns.index[end - 1])
    point_cloud = returns.iloc[start:end].to_numpy()
    point_clouds.append(point_cloud)

analysis_table = pd.DataFrame({
    "date": window_dates,
    "point_cloud": point_clouds
})

analysis_table = analysis_table.set_index("date")

persistence_results = []

total = len(analysis_table)

for i, point_cloud in enumerate(point_clouds, start=1):
    print(
        f"\rProcessing point cloud {i}/{total} "
        f"({100 * i / total:.1f}%)",
        end="",
        flush=True
    )
    result = ripser(
        point_cloud,
        maxdim=2
    )
    persistence_results.append(result)

#changing birth-death coordiantes to birth-lifetime
def adjust_diagram(diagram):
    # Remove points with infinite death
    finite = diagram[np.isfinite(diagram[:, 1])]
    birth = finite[:, 0]
    persistence = finite[:, 1] - finite[:, 0]
    return np.column_stack((birth, persistence))

analysis_table["H1"] = [
    adjust_diagram(result["dgms"][1])
    for result in persistence_results
]

analysis_table["H2"] = [
    adjust_diagram(result["dgms"][2])
    for result in persistence_results
]

def raw_tda_stress_l2(diagram):
    persistence = diagram[:, 1]
    return np.sqrt(np.sum(persistence**2))

def normalized_tda_stress_l2(diagram, point_cloud):
    raw_stress = raw_tda_stress_l2(diagram)
    scale = np.median(pdist(point_cloud))
    if scale == 0:
        return 0.0
    return raw_stress / scale

analysis_table["H1_raw_stress"] = [
    raw_tda_stress_l2(h1_result) for h1_result in analysis_table["H1"]
]

analysis_table["H1_normalized_stress"] = [
    normalized_tda_stress_l2(h1_result, point_cloud) for h1_result, point_cloud in zip(analysis_table["H1"], analysis_table["point_cloud"])
]

analysis_table["H2_raw_stress"] = [
    raw_tda_stress_l2(h2_result) for h2_result in analysis_table["H2"]
]

analysis_table["H2_normalized_stress"] = [
    normalized_tda_stress_l2(h2_result, point_cloud) for h2_result, point_cloud in zip(analysis_table["H2"], analysis_table["point_cloud"])
]

def unified_volatility(point_cloud):
    stock_volatilities = []
    for stock_index in range(len(point_cloud[0])):
        stock_returns = point_cloud[:, stock_index]
        average_return = sum(stock_returns) / len(stock_returns)
        variance = sum(
            (return_value - average_return) ** 2
            for return_value in stock_returns
        ) / (len(stock_returns) - 1)

        stock_volatilities.append(
            np.sqrt(variance)
        )
    return sum(stock_volatilities) / len(stock_volatilities)

analysis_table["volatility"] = [
    unified_volatility(point_cloud) for point_cloud in analysis_table["point_cloud"]
]

def average_correlation(point_cloud):
    correlation_matrix = np.corrcoef(
        point_cloud,
        rowvar=False
    )
    correlations = []
    number_of_stocks = len(point_cloud[0])
    for i in range(number_of_stocks):
        for j in range(i + 1, number_of_stocks):
            correlations.append(
                correlation_matrix[i, j]
            )
    return sum(correlations) / len(correlations)

analysis_table["average_correlation"] = [
    average_correlation(point_cloud)
    for point_cloud in analysis_table["point_cloud"]
]

analysis_table.to_pickle("analysis_table.pkl")

#

