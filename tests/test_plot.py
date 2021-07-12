"""tests/test_plot.py"""
import plotly.express as px

from talus_utils.plot import update_layout


def test_update_layout() -> None:
    """Tests update_layout decorator."""
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

    assert (
        fig.layout.xaxis.title.text == "sepal_width"
        and fig.layout.yaxis.title.text == "sepal_length"
    )

    title = "title test"
    xaxis_title = "xaxis test"
    yaxis_title = "yaxis test"
    fig = update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)(
        px.scatter
    )(df, x="sepal_width", y="sepal_length", color="species")

    assert fig.layout.title.text == title
    assert fig.layout.xaxis.title.text == xaxis_title
    assert fig.layout.yaxis.title.text == yaxis_title
