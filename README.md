# 🌟 Wellbeing & Lifestyle Analysis Project

An end-to-end data science project that explores wellness and lifestyle patterns using clustering analysis and interactive visualization.

##  Overview

This project analyzes wellbeing and lifestyle data from Kaggle to identify distinct wellness personas through unsupervised machine learning. The analysis includes comprehensive exploratory data analysis (EDA), K-Means clustering, and an interactive Streamlit dashboard for exploring the findings.

##  Key Features

- **Comprehensive EDA**: Data quality checks, correlation analysis, outlier detection, and demographic insights
- **Clustering Analysis**: K-Means algorithm identifies 4 distinct wellness personas
- **Interactive Dashboard**: Multi-page Streamlit app for exploring clusters and finding your personal wellness profile
- **Data Visualization**: Professional charts and graphs using Plotly and Matplotlib
- **Personalized Insights**: Get recommendations based on your lifestyle inputs

##  Wellness Personas Identified

1. **Balanced Achievers** (31.4%) - Well-rounded lifestyle habits across all dimensions
2. **Active Wellness Warriors** (27.0%) - High physical activity and health consciousness  
3. **Stressed Professionals** (22.1%) - High stress levels requiring intervention
4. **Social Butterflies** (19.5%) - Strong social connections and community focus

## 📁 Project Structure

```
├── 01_data_exploration_and_preprocessing.ipynb  # EDA and data cleaning
├── 02_clustering_analysis.ipynb                 # K-Means clustering implementation
├── 03_streamlit_dashboard.py                    # Interactive web dashboard
├── Wellbeing_and_lifestyle_data_Kaggle.csv     # Source dataset
├── requirements.txt                             # Python dependencies
├── EDA_SUMMARY.md                              # EDA deliverables documentation
├── DASHBOARD_README.md                          # Dashboard usage guide
└── images/                                      # Visualization outputs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/wellbeing-analysis.git
cd wellbeing-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Explore the Data (Optional)
Open and run the Jupyter notebooks in order:
```bash
jupyter notebook 01_data_exploration_and_preprocessing.ipynb
jupyter notebook 02_clustering_analysis.ipynb
```

#### 2. Launch the Dashboard
```bash
streamlit run 03_streamlit_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

##  Dashboard Features

### Page 1: Overview
- Project methodology and key findings
- Cluster distribution visualization
- Dataset statistics

### Page 2: Cluster Explorer  
- Interactive cluster selection
- Radar charts showing persona profiles
- Demographic breakdowns
- Top distinguishing characteristics

### Page 3: Find Your Cluster
- Input your lifestyle data
- ML-powered cluster prediction
- Personalized recommendations
- Profile comparison charts

### Page 4: Insights & Patterns
- Data-driven findings
- Feature importance analysis
- Cluster comparison charts
- Actionable recommendations

##  Technologies Used

- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (K-Means Clustering)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Development**: Jupyter Notebook

##  Analysis Workflow

1. **Data Loading & Quality Check**: Import dataset, check for missing values and duplicates
2. **Exploratory Data Analysis**: Correlation analysis, distribution plots, outlier detection
3. **Data Preprocessing**: Handle outliers using Winsorization, scale features
4. **Clustering**: Apply K-Means algorithm, determine optimal number of clusters
5. **Validation**: Evaluate cluster quality using silhouette scores
6. **Interpretation**: Profile each cluster and derive actionable insights
7. **Deployment**: Interactive dashboard for exploration and personalization

##  Key Insights

- Work-life balance is strongly correlated with stress levels, daily steps, and sleep quality
- Four distinct wellness personas emerge with unique characteristics
- Physical activity and social connections are key differentiators
- Age and gender show interaction effects on wellbeing scores

##  Documentation

For detailed information, refer to:
- `EDA_SUMMARY.md` - Complete EDA deliverables and methodology
- `DASHBOARD_README.md` - Dashboard features and troubleshooting guide

##  Contributing

This is a personal learning project, but suggestions and feedback are welcome! Feel free to open an issue or submit a pull request.

##  License

This project is open source and available under the [MIT License](LICENSE).

##  Acknowledgments

- Dataset source: [Kaggle - Wellbeing and Lifestyle Data](https://www.kaggle.com/)
- Built as part of a data science portfolio project




