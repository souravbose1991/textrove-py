import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textrove import Documents, Sentiment, Summary, DynTM, WordNetwork
import seaborn as sns
import plotly.express as px
import altair as alt
from bokeh.plotting import figure


# can only set this once, first thing to set
st.set_page_config(layout="wide")

plot_types = (
    "Scatter",
    "Histogram",
    "Bar",
    "Line",
    "3D Scatter",
)  # maybe add 'Boxplot' after fixes
libs = (
    "Matplotlib",
    "Seaborn",
    "Plotly Express",
    "Altair",
    "Pandas Matplotlib",
    "Bokeh",
)

# get data
# @st.cache(allow_output_mutation=True) # maybe source of resource limit issue


def load_penguins():
    return sns.load_dataset("penguins")


pens_df = load_penguins()
df = pens_df.copy()
df.index = pd.date_range(start="1/1/18", periods=len(df), freq="D")


with st.beta_container():
    st.title("Python Data Visualization Tour")
    st.header("Popular plots in popular plotting libraries")
    st.write("""See the code and plots for five libraries at once.""")

# display data
with st.beta_container():
    show_data = st.checkbox("See the raw data?")

    # notes
    st.subheader("Notes")
    st.write(
        """
        - This app uses [Streamlit](https://streamlit.io/) and the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset.      
        - To see the full code check out the [GitHub repo](https://github.com/discdiver/data-viz-streamlit).
        - Plots are interactive where that's the default or easy to add.
        - Plots that use MatPlotlib under the hood have fig and ax objects defined before the code shown.
        - Lineplots should have sequence data, so I created a date index with a sequence of dates for them. 
        - Where an axis label shows by default, I left it at is. Generally where it was missing, I added it.
        - There are multiple ways to make some of these plots.
        - You can choose to see two columns, but with a narrow screen this will switch to one column automatically.
        - Python has many data visualization libraries. This gallery is not exhaustive. If you would like to add code for another library, please submit a [pull request](https://github.com/discdiver/data-viz-streamlit).
        - For a larger tour of more plots, check out the [Python Graph Gallery](https://www.python-graph-gallery.com/density-plot/) and [Python Plotting for Exploratory Data Analysis](https://pythonplot.com/).
        - The interactive Plotly Express 3D Scatterplot is cool to play with. Check it out! 😎
        
        Made by [Jeff Hale](https://www.linkedin.com/in/-jeffhale/). 
        
        Subscribe to my [Data Awesome newsletter](https://dataawesome.com) for the latest tools, tips, and resources.
        """
    )


