import plotly.graph_objects as go
import numpy as np

# Spiral parameters
t = np.linspace(0, 10, 200)
x, y, z = np.cos(t), np.sin(t), t

# Create base figure with fixed axis ranges
fig = go.Figure(
    data=[go.Scatter3d(x=[x[0]], y=[y[0]], z=[z[0]], mode='markers')],
    layout=go.Layout(
        scene=dict(
            xaxis=dict(range=[min(x), max(x)]),  # Fixed range for x-axis
            yaxis=dict(range=[min(y), max(y)]),  # Fixed range for y-axis
            zaxis=dict(range=[min(z), max(z)])   # Fixed range for z-axis
        )
    )
)

# Add frames
frames = [go.Frame(data=[go.Scatter3d(x=[x[k]], y=[y[k]], z=[z[k]])])
          for k in range(len(x))]

fig.frames = frames

# Add animation settings
fig.update_layout(
    updatemenus=[dict(type="buttons",
                      buttons=[dict(label="Play",
                                     method="animate",
                                     args=[None, 
                                           {"frame": {"duration": 50, "redraw": True},
                                            "transition": {"duration": 10}}])])])

# Display the animation
fig.show()
