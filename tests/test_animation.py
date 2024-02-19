import pandas as pd
from src.data_vis.DataVisuals import DataVisualization
import matplotlib.pyplot as plt
if __name__=="__main__":

    df = pd.read_csv("rk45_states.csv")
    data_vis = DataVisualization(df, 5)
    ani = data_vis.animate_local(interval=20, save=False)
    
    ani2 = data_vis.animate_global(save=False)

    plt.show()