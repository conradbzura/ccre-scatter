import jscatter
import pandas as pd
import numpy as np

# Initial data
data1 = pd.DataFrame(
    {
        "x": np.random.randn(100),
        "y": np.random.randn(100),
        "category": np.random.choice(["A", "B", "C"], 100),
    }
)

# Create scatter plot with categorical colors
scatter = jscatter.Scatter(
    data=data1,
    x="x",
    y="y",
    color_by="category",
    color_map=["#FF0000", "#00FF00", "#0000FF"],  # Maps to categories in order
)

# New data with same schema
data2 = pd.DataFrame(
    {
        "x": np.random.randn(200),
        "y": np.random.randn(200),
        "category": np.random.choice(["A", "B", "C"], 200),
    }
)

# Update by assigning to data property
scatter.data(data2)
