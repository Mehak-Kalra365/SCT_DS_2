# iris_dashboard.py
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

# 1️⃣ Load Cleaned Dataset
df = pd.read_csv('iris/iris_cleaned.csv')

# 2️⃣ Create Plots
# Correlation Heatmap
corr = df.drop('species', axis=1).corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title='🌸 Feature Correlation Heatmap 🌸')

# Scatter Matrix
fig_scatter_matrix = px.scatter_matrix(df,
                                       dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                                       color='species',
                                       title='🌸 Iris Dataset Scatter Matrix 🌸',
                                       height=700)
fig_scatter_matrix.update_traces(diagonal_visible=True)

# Bar Chart
summary = df.groupby('species')[['sepal_length','sepal_width','petal_length','petal_width']].mean().reset_index()
fig_bar = px.bar(summary, x='species', y=['sepal_length','sepal_width','petal_length','petal_width'],
                 barmode='group', text_auto=True, title='🌸 Average Sepal & Petal Dimensions per Species 🌸')

# Box Plot
fig_box = px.box(df, x='species', y='petal_length', color='species',
                 title='🌸 Petal Length Distribution per Species 🌸')

# Scatter with Trendline
fig_trend = px.scatter(df, x='petal_length', y='petal_width', color='species',
                  trendline='ols',
                  title='🌸 Petal Length vs Petal Width with Trendline 🌸')

# 3️⃣ Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("🌸 Interactive Iris Dataset Dashboard 🌸", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Correlation Heatmap', children=[dcc.Graph(figure=fig_corr)]),
        dcc.Tab(label='Scatter Matrix', children=[dcc.Graph(figure=fig_scatter_matrix)]),
        dcc.Tab(label='Species Average Bar Chart', children=[dcc.Graph(figure=fig_bar)]),
        dcc.Tab(label='Box Plot', children=[dcc.Graph(figure=fig_box)]),
        dcc.Tab(label='Petal Trend', children=[dcc.Graph(figure=fig_trend)]),
    ])
])

if __name__ == '__main__':
    app.run(debug=True)  # instead of app.run_server(debug=True)
