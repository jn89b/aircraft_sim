import pandas as pd
from src.data_vis.DataVisuals import DataVisualization

if __name__=="__main__":
    df = pd.read_csv("rk45_states.csv")
    data_vis = DataVisualization(df)

    data_vis.animate()