import plotly
import plotly.express as px
import pandas as pd

#series barchart
def series_to_bar(dict):
    fig = px.bar(pd.Series(dict))

    return fig