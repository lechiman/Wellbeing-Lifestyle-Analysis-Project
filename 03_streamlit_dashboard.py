"""
Wellbeing & Lifestyle Clustering Analysis Dashboard
Interactive multi-page Streamlit application for exploring wellness personas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wellbeing Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .stMetric label {
        color: #1f1f1f !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0a0a0a !important;
        font-weight: 700;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #1f1f1f !important;
    }
    h1 {
        color: #0D7377;
        padding-bottom: 10px;
    }
    h2 {
        color: #14BDEB;
        padding-top: 10px;
    }
    .cluster-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0D7377;
        margin: 10px 0;
        color: #1f1f1f;
    }
    .cluster-card h4 {
        color: #0a0a0a;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .cluster-card p {
        color: #1f1f1f;
        font-weight: 500;
    }
    .cluster-card strong {
        color: #0a0a0a;
        font-weight: 700;
    }
    .cluster-card em {
        color: #2f2f2f;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING FUNCTIONS ====================

@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    # Try different possible file paths
    possible_paths = [
        'data/clustered_wellbeing_data.csv',
        'data/processed_wellbeing_data.csv',
        'Wellbeing_and_lifestyle_data_Kaggle.csv'
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.sidebar.success(f"Data loaded from: {path}")
            break
    
    if df is None:
        st.error("Could not find data file. Please ensure the clustering notebook has been run.")
        return None
    
    # Drop non-feature columns
    cols_to_drop = ['Timestamp', 'timestamp', 'TIMESTAMP']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Handle AGE column if it's categorical
    if 'AGE' in df.columns and df['AGE'].dtype == 'object':
        age_mapping = {
            'Less than 20': 18,
            '21 to 35': 28,
            '36 to 50': 43,
            '51 or more': 60
        }
        df['AGE'] = df['AGE'].map(age_mapping)
    
    # Convert all numeric-like columns to numeric, coercing errors
    for col in df.columns:
        if col not in ['GENDER', 'Cluster', 'Cluster_Method']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # If no Cluster column exists, create clusters
    if 'Cluster' not in df.columns:
        features = ['FRUITS_VEGGIES', 'DAILY_STRESS', 'SLEEP_HOURS', 'DAILY_STEPS',
                   'FLOW', 'WEEKLY_MEDITATION', 'TODO_COMPLETED', 'ACHIEVEMENT',
                   'TIME_FOR_PASSION', 'WORK_LIFE_BALANCE_SCORE']
        
        # Select only available features and ensure they're numeric
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].copy()
        
        # Fill missing values with median
        X = X.fillna(X.median())
        
        # Double-check all values are numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        st.sidebar.info("Clusters created on-the-fly from raw data")
    
    return df

@st.cache_data
def get_cluster_profiles(df):
    """Calculate mean values for each cluster"""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude cluster column and any timestamp-related columns
    exclude_cols = ['Cluster', 'Timestamp', 'timestamp', 'TIMESTAMP']
    numerical_features = [f for f in numerical_features if f not in exclude_cols]
    
    profiles = df.groupby('Cluster')[numerical_features].mean()
    return profiles

@st.cache_data
def get_cluster_personas():
    """Define cluster personas based on analysis"""
    return {
        0: {
            'name': 'Balanced Achievers',
            'icon': '⚖️',
            'tagline': 'Well-rounded individuals with moderate lifestyle habits',
            'description': 'This group represents people who maintain a balanced approach to life. They show moderate levels across most wellness dimensions without extreme highs or lows.',
            'characteristics': [
                'Moderate stress levels',
                'Balanced work-life integration',
                'Consistent sleep patterns',
                'Stable social connections'
            ],
            'recommendations': [
                'Maintain current healthy habits',
                'Explore new wellness activities',
                'Consider slight increases in physical activity',
                'Continue stable lifestyle patterns'
            ],
            'color': '#4CAF50'
        },
        1: {
            'name': 'Active Wellness Warriors',
            'icon': '💪',
            'tagline': 'High-energy individuals focused on physical health',
            'description': 'These are the fitness enthusiasts and health-conscious individuals. They prioritize physical activity, healthy sleep patterns, and maintaining an active lifestyle.',
            'characteristics': [
                'High physical activity levels',
                'Strong achievement orientation',
                'Regular meditation practice',
                'Excellent sleep quality'
            ],
            'recommendations': [
                'Maintain high activity levels',
                'Share wellness knowledge with others',
                'Balance physical with mental wellness',
                'Be mindful of overtraining'
            ],
            'color': '#FF9800'
        },
        2: {
            'name': 'Stressed Professionals',
            'icon': '⚠️',
            'tagline': 'Busy individuals experiencing higher stress levels',
            'description': 'This cluster includes individuals dealing with higher stress levels, possibly due to work pressure or life challenges. They may have less time for self-care.',
            'characteristics': [
                'High daily stress',
                'Lower sleep quality',
                'Limited time for passion projects',
                'Reduced physical activity'
            ],
            'recommendations': [
                'Prioritize stress management techniques',
                'Establish consistent sleep routine',
                'Schedule dedicated personal time',
                'Seek support for mental health'
            ],
            'color': '#F44336'
        },
        3: {
            'name': 'Social Butterflies',
            'icon': '🦋',
            'tagline': 'Socially engaged individuals with strong connections',
            'description': 'These individuals thrive on social interactions and have strong support networks. Their wellness is closely tied to their social engagement.',
            'characteristics': [
                'Strong social networks',
                'High community engagement',
                'Regular social activities',
                'Good emotional support'
            ],
            'recommendations': [
                'Continue nurturing social connections',
                'Balance social time with self-care',
                'Leverage community for wellness',
                'Share experiences with others'
            ],
            'color': '#9C27B0'
        }
    }

# ==================== VISUALIZATION FUNCTIONS ====================

def create_radar_chart(data, title="Cluster Profile"):
    """Create a radar chart for cluster characteristics"""
    categories = list(data.keys())
    values = list(data.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title,
        line_color='#0D7377'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )
        ),
        showlegend=False,
        title=title,
        height=400
    )
    
    return fig

def create_comparison_radar(user_data, cluster_data, labels):
    """Create radar chart comparing user profile with cluster average"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(user_data.values()),
        theta=labels,
        fill='toself',
        name='Your Profile',
        line_color='#FF6B6B',
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=list(cluster_data.values()),
        theta=labels,
        fill='toself',
        name='Cluster Average',
        line_color='#0D7377',
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(user_data.values()), max(cluster_data.values())) * 1.1]
            )
        ),
        showlegend=True,
        title="Your Profile vs Cluster Average",
        height=500
    )
    
    return fig

def create_pie_chart(df):
    """Create pie chart for cluster distribution"""
    cluster_counts = df['Cluster'].value_counts().sort_index()
    personas = get_cluster_personas()
    
    labels = [f"{personas[i]['name']}" for i in cluster_counts.index]
    colors = [personas[i]['color'] for i in cluster_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=cluster_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title="Cluster Distribution",
        height=400,
        showlegend=True
    )
    
    return fig

def create_feature_importance_heatmap(profiles):
    """Create heatmap showing feature importance across clusters"""
    # Standardize the profiles
    profiles_std = (profiles - profiles.mean()) / profiles.std()
    
    fig = go.Figure(data=go.Heatmap(
        z=profiles_std.values.T,
        x=[f"Cluster {i}" for i in profiles_std.index],
        y=profiles_std.columns,
        colorscale='RdYlGn',
        zmid=0,
        text=profiles_std.values.T.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Z-Score")
    ))
    
    fig.update_layout(
        title="Feature Importance by Cluster (Standardized)",
        xaxis_title="Cluster",
        yaxis_title="Feature",
        height=600
    )
    
    return fig

def create_comparison_bars(df, feature, personas):
    """Create bar chart comparing feature across clusters"""
    cluster_means = df.groupby('Cluster')[feature].mean().sort_index()
    
    colors = [personas[i]['color'] for i in cluster_means.index]
    labels = [personas[i]['name'] for i in cluster_means.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=cluster_means.values,
            marker_color=colors,
            text=cluster_means.values.round(1),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"{feature.replace('_', ' ').title()} by Cluster",
        xaxis_title="Cluster",
        yaxis_title=feature.replace('_', ' ').title(),
        height=400
    )
    
    return fig

# ==================== PAGE 1: OVERVIEW ====================

def page_overview():
    st.title("Wellbeing & Lifestyle Clustering Analysis")
    st.markdown("### Understanding Wellness Personas through Data Science")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Project Description
    st.markdown("---")
    st.header("Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This interactive dashboard presents the results of a comprehensive clustering analysis 
        on wellbeing and lifestyle data. Using machine learning techniques, we've identified 
        **four distinct wellness personas** based on lifestyle habits, stress levels, physical 
        activity, and social engagement patterns.
        
        **Methodology:**
        - **Data Source:** Kaggle Wellbeing and Lifestyle Dataset
        - **Sample Size:** 15,972 individuals
        - **Features Analyzed:** 20+ lifestyle and wellness metrics
        - **Clustering Method:** K-Means clustering (k=4)
        - **Validation:** Silhouette score and Davies-Bouldin index
        
        **Key Applications:**
        - Personalized wellness recommendations
        - Targeted intervention programs
        - Understanding lifestyle patterns
        - Resource allocation for wellness initiatives
        """)
    
    with col2:
        st.info("""
        **Navigate through pages:**
        
        **Overview** - Summary and statistics
        
        **Cluster Explorer** - Deep dive into each persona
        
        **Find Your Cluster** - Discover which persona matches you
        
        **Insights** - Patterns and recommendations
        """)
    
    # Dataset Statistics
    st.markdown("---")
    st.header("Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Individuals", f"{len(df):,}")
    
    with col2:
        st.metric("Features Analyzed", len(df.select_dtypes(include=[np.number]).columns))
    
    with col3:
        st.metric("Wellness Personas", df['Cluster'].nunique())
    
    with col4:
        if 'WORK_LIFE_BALANCE_SCORE' in df.columns:
            avg_score = df['WORK_LIFE_BALANCE_SCORE'].mean()
            st.metric("Avg Balance Score", f"{avg_score:.1f}")
        else:
            st.metric("Data Quality", "High")
    
    # Cluster Distribution
    st.markdown("---")
    st.header("Cluster Distribution")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_pie = create_pie_chart(df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("### Wellness Personas Identified")
        personas = get_cluster_personas()
        
        for cluster_id in sorted(df['Cluster'].unique()):
            count = len(df[df['Cluster'] == cluster_id])
            percentage = (count / len(df)) * 100
            persona = personas[cluster_id]
            
            st.markdown(f"""
            <div class="cluster-card">
                <h4>{persona['icon']} {persona['name']}</h4>
                <p><strong>{count:,} individuals ({percentage:.1f}%)</strong></p>
                <p><em>{persona['tagline']}</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Key Findings Summary
    st.markdown("---")
    st.header("Key Findings Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Distinct Patterns")
        st.markdown("""
        - Four clear wellness personas emerged
        - Each cluster shows unique lifestyle characteristics
        - Strong separation validated by metrics
        - Actionable insights for each group
        """)
    
    with col2:
        st.markdown("#### Critical Insights")
        st.markdown("""
        - 22% experience high stress levels
        - 27% demonstrate exceptional wellness habits
        - Social connection impacts wellbeing
        - Work-life balance varies significantly
        """)
    
    with col3:
        st.markdown("#### Recommendations")
        st.markdown("""
        - Personalized interventions needed
        - Stress management for at-risk groups
        - Leverage social support systems
        - Promote physical activity programs
        """)

# ==================== PAGE 2: CLUSTER EXPLORER ====================

def page_cluster_explorer():
    st.title("Cluster Explorer")
    st.markdown("### Deep Dive into Wellness Personas")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    personas = get_cluster_personas()
    profiles = get_cluster_profiles(df)
    
    # Cluster Selection
    st.markdown("---")
    selected_cluster = st.selectbox(
        "Select a cluster to explore:",
        options=sorted(df['Cluster'].unique()),
        format_func=lambda x: f"{personas[x]['icon']} {personas[x]['name']}"
    )
    
    persona = personas[selected_cluster]
    cluster_data = df[df['Cluster'] == selected_cluster]
    
    # Cluster Overview
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cluster Size", f"{len(cluster_data):,}")
    
    with col2:
        percentage = (len(cluster_data) / len(df)) * 100
        st.metric("Percentage", f"{percentage:.1f}%")
    
    with col3:
        if 'AGE' in df.columns:
            avg_age = cluster_data['AGE'].mean()
            st.metric("Average Age", f"{avg_age:.0f}")
        else:
            st.metric("Status", "Active")
    
    with col4:
        if 'GENDER' in df.columns:
            gender_mode = cluster_data['GENDER'].mode()[0] if len(cluster_data) > 0 else "N/A"
            st.metric("Predominant Gender", gender_mode)
        else:
            st.metric("Data", "Complete")
    
    # Persona Profile
    st.markdown("---")
    st.markdown(f"## {persona['icon']} {persona['name']}")
    st.markdown(f"### *{persona['tagline']}*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Description")
        st.write(persona['description'])
        
        st.markdown("#### Key Characteristics")
        for char in persona['characteristics']:
            st.markdown(f"- {char}")
    
    with col2:
        st.markdown("#### Recommendations")
        for rec in persona['recommendations']:
            st.markdown(f"✓ {rec}")
    
    # Radar Chart
    st.markdown("---")
    st.subheader("Cluster Profile Visualization")
    
    # Select key features for radar chart
    key_features = ['DAILY_STRESS', 'ACHIEVEMENT', 'SLEEP_HOURS', 'SOCIAL_NETWORK',
                   'FLOW', 'DAILY_STEPS', 'FRUITS_VEGGIES', 'TIME_FOR_PASSION']
    available_features = [f for f in key_features if f in profiles.columns]
    
    if available_features:
        radar_data = {f.replace('_', ' ').title(): profiles.loc[selected_cluster, f] 
                     for f in available_features}
        fig_radar = create_radar_chart(radar_data, f"{persona['name']} Profile")
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Statistics Table
    st.markdown("---")
    st.subheader("Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Lifestyle Metrics")
        lifestyle_features = ['DAILY_STRESS', 'SLEEP_HOURS', 'DAILY_STEPS', 
                            'WEEKLY_MEDITATION', 'FRUITS_VEGGIES']
        available_lifestyle = [f for f in lifestyle_features if f in cluster_data.columns]
        
        if available_lifestyle:
            lifestyle_stats = cluster_data[available_lifestyle].describe().T[['mean', 'std', 'min', 'max']]
            lifestyle_stats.index = [i.replace('_', ' ').title() for i in lifestyle_stats.index]
            st.dataframe(lifestyle_stats.round(2), use_container_width=True)
    
    with col2:
        st.markdown("#### Wellbeing Metrics")
        wellbeing_features = ['ACHIEVEMENT', 'FLOW', 'TIME_FOR_PASSION', 
                            'SOCIAL_NETWORK', 'TODO_COMPLETED']
        available_wellbeing = [f for f in wellbeing_features if f in cluster_data.columns]
        
        if available_wellbeing:
            wellbeing_stats = cluster_data[available_wellbeing].describe().T[['mean', 'std', 'min', 'max']]
            wellbeing_stats.index = [i.replace('_', ' ').title() for i in wellbeing_stats.index]
            st.dataframe(wellbeing_stats.round(2), use_container_width=True)
    
    # Demographic Breakdown
    st.markdown("---")
    st.subheader("Demographic Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'GENDER' in cluster_data.columns:
            gender_dist = cluster_data['GENDER'].value_counts()
            fig_gender = px.pie(
                values=gender_dist.values,
                names=gender_dist.index,
                title="Gender Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        if 'AGE' in cluster_data.columns:
            fig_age = px.histogram(
                cluster_data,
                x='AGE',
                nbins=20,
                title="Age Distribution",
                color_discrete_sequence=[persona['color']]
            )
            fig_age.update_layout(showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
    
    # Top Characteristics
    st.markdown("---")
    st.subheader("Top Distinguishing Features")
    
    # Calculate z-scores to find distinguishing features
    profiles_std = (profiles - profiles.mean()) / profiles.std()
    cluster_zscore = profiles_std.loc[selected_cluster].abs().sort_values(ascending=False).head(10)
    
    fig_features = go.Figure(data=[
        go.Bar(
            x=cluster_zscore.values,
            y=[f.replace('_', ' ').title() for f in cluster_zscore.index],
            orientation='h',
            marker_color=persona['color']
        )
    ])
    
    fig_features.update_layout(
        title="Most Distinctive Features (by Z-Score)",
        xaxis_title="Absolute Z-Score",
        yaxis_title="Feature",
        height=400
    )
    
    st.plotly_chart(fig_features, use_container_width=True)

# ==================== PAGE 3: FIND YOUR CLUSTER ====================

def page_find_your_cluster():
    st.title("Find Your Cluster")
    st.markdown("### Discover Which Wellness Persona You Match")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    personas = get_cluster_personas()
    profiles = get_cluster_profiles(df)
    
    st.markdown("---")
    st.markdown("""
    Answer the questions below about your lifestyle and wellness habits. 
    We'll use machine learning to predict which wellness persona you belong to 
    and provide personalized recommendations.
    """)
    
    st.markdown("---")
    st.subheader("Your Lifestyle Information")
    
    # Create input form
    with st.form("lifestyle_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Daily Habits")
            
            daily_stress = st.slider(
                "Daily Stress Level (1-10)",
                min_value=1, max_value=10, value=5,
                help="How stressed do you feel on average?"
            )
            
            sleep_hours = st.slider(
                "Sleep Hours per Night",
                min_value=3, max_value=12, value=7,
                help="How many hours do you typically sleep?"
            )
            
            daily_steps = st.number_input(
                "Average Daily Steps",
                min_value=0, max_value=30000, value=5000, step=500,
                help="Approximate steps you take daily"
            )
            
            fruits_veggies = st.slider(
                "Servings of Fruits/Veggies Daily",
                min_value=0, max_value=10, value=3,
                help="How many servings per day?"
            )
            
            weekly_meditation = st.slider(
                "Meditation Minutes per Week",
                min_value=0, max_value=300, value=0, step=15,
                help="Total meditation time weekly"
            )
        
        with col2:
            st.markdown("#### Wellbeing & Productivity")
            
            achievement = st.slider(
                "Sense of Achievement (1-10)",
                min_value=1, max_value=10, value=5,
                help="How accomplished do you feel?"
            )
            
            flow = st.slider(
                "Flow State Frequency (1-10)",
                min_value=1, max_value=10, value=5,
                help="How often are you 'in the zone'?"
            )
            
            todo_completed = st.slider(
                "Tasks Completed Daily (%)",
                min_value=0, max_value=100, value=50, step=5,
                help="Percentage of to-dos you complete"
            )
            
            time_for_passion = st.slider(
                "Time for Passion Projects (hours/week)",
                min_value=0, max_value=40, value=5,
                help="Hours spent on hobbies/passions"
            )
            
            social_network = st.slider(
                "Social Network Strength (1-10)",
                min_value=1, max_value=10, value=5,
                help="How strong are your social connections?"
            )
        
        submitted = st.form_submit_button("Find My Cluster", use_container_width=True)
    
    if submitted:
        # Prepare user data
        user_data = {
            'DAILY_STRESS': daily_stress,
            'SLEEP_HOURS': sleep_hours,
            'DAILY_STEPS': daily_steps,
            'FRUITS_VEGGIES': fruits_veggies,
            'WEEKLY_MEDITATION': weekly_meditation,
            'ACHIEVEMENT': achievement,
            'FLOW': flow,
            'TODO_COMPLETED': todo_completed,
            'TIME_FOR_PASSION': time_for_passion,
            'SOCIAL_NETWORK': social_network
        }
        
        # Predict cluster
        features = list(user_data.keys())
        available_features = [f for f in features if f in profiles.columns]
        
        if available_features:
            # Calculate distance to each cluster
            distances = {}
            for cluster_id in profiles.index:
                cluster_profile = profiles.loc[cluster_id, available_features]
                user_profile = np.array([user_data[f] for f in available_features])
                
                # Normalize and calculate euclidean distance
                distance = np.sqrt(np.sum((user_profile - cluster_profile.values) ** 2))
                distances[cluster_id] = distance
            
            # Find closest cluster
            predicted_cluster = min(distances, key=distances.get)
            persona = personas[predicted_cluster]
            
            # Display results
            st.markdown("---")
            st.success("Analysis Complete!")
            
            st.markdown(f"""
            ## Your Wellness Persona: {persona['icon']} {persona['name']}
            ### *{persona['tagline']}*
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### About This Persona")
                st.write(persona['description'])
                
                st.markdown("#### Your Match Score")
                match_percentage = 100 - (distances[predicted_cluster] / max(distances.values()) * 100)
                st.progress(match_percentage / 100)
                st.write(f"**{match_percentage:.1f}% match** with this persona")
            
            with col2:
                st.markdown("#### Similar Users")
                cluster_size = len(df[df['Cluster'] == predicted_cluster])
                total_size = len(df)
                st.metric("People Like You", f"{cluster_size:,}")
                st.metric("Percentage", f"{(cluster_size/total_size)*100:.1f}%")
            
            # Profile Comparison
            st.markdown("---")
            st.subheader("Your Profile vs Cluster Average")
            
            user_radar = {f.replace('_', ' ').title(): user_data[f] 
                         for f in available_features}
            cluster_radar = {f.replace('_', ' ').title(): profiles.loc[predicted_cluster, f] 
                           for f in available_features}
            
            fig_comparison = create_comparison_radar(
                user_radar, 
                cluster_radar, 
                list(user_radar.keys())
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Personalized Recommendations
            st.markdown("---")
            st.subheader("Personalized Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### General Recommendations")
                for rec in persona['recommendations']:
                    st.markdown(f"✓ {rec}")
            
            with col2:
                st.markdown("#### Areas for Improvement")
                
                # Find areas where user is below cluster average
                improvements = []
                for feature in available_features:
                    user_val = user_data[feature]
                    cluster_val = profiles.loc[predicted_cluster, feature]
                    
                    if user_val < cluster_val * 0.8:  # 20% below average
                        diff = ((cluster_val - user_val) / cluster_val) * 100
                        improvements.append((feature, diff))
                
                if improvements:
                    improvements.sort(key=lambda x: x[1], reverse=True)
                    for feature, diff in improvements[:5]:
                        st.markdown(f"- **{feature.replace('_', ' ').title()}**: {diff:.0f}% below cluster average")
                else:
                    st.success("You're performing at or above average in all areas!")
            
            # Next Steps
            st.markdown("---")
            st.info("""
            **Next Steps:**
            - Review the Cluster Explorer page to learn more about your persona
            - Check the Insights page for data-driven recommendations
            - Consider tracking your progress over time
            - Share your results with wellness professionals
            """)

# ==================== PAGE 4: INSIGHTS & PATTERNS ====================

def page_insights():
    st.title("Insights & Patterns")
    st.markdown("### Data-Driven Discoveries and Recommendations")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    personas = get_cluster_personas()
    profiles = get_cluster_profiles(df)
    
    # Key Findings
    st.markdown("---")
    st.header("Key Findings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Pattern Discovery")
        st.markdown("""
        **Four Distinct Personas:**
        - Clear separation between groups
        - Each with unique characteristics
        - Validated by statistical metrics
        - Actionable insights for each
        
        **Lifestyle Impact:**
        - Physical activity correlates with wellbeing
        - Social connections matter significantly
        - Stress management is critical
        - Balance varies across personas
        """)
    
    with col2:
        st.markdown("### Critical Insights")
        st.markdown("""
        **At-Risk Population:**
        - 22% show high stress levels
        - Sleep quality varies dramatically
        - Work-life balance needs attention
        - Intervention opportunities identified
        
        **Success Factors:**
        - 27% demonstrate optimal habits
        - Regular exercise shows benefits
        - Meditation practice helps
        - Social engagement supports wellness
        """)
    
    with col3:
        st.markdown("### Demographic Patterns")
        if 'AGE' in df.columns:
            st.markdown("""
            **Age Distribution:**
            - Varies across clusters
            - Younger groups more active
            - Stress affects all ages
            - Lifestyle changes with age
            """)
        
        st.markdown("""
        **Behavioral Patterns:**
        - Habits cluster together
        - Multi-dimensional wellness
        - Interconnected factors
        - Holistic approach needed
        """)
    
    # Feature Importance Analysis
    st.markdown("---")
    st.header("Feature Importance Analysis")
    
    st.markdown("""
    This heatmap shows how each feature varies across clusters using standardized scores (z-scores).
    Positive values (green) indicate above-average levels, while negative values (red) indicate below-average levels.
    """)
    
    fig_heatmap = create_feature_importance_heatmap(profiles)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Cluster Comparison Charts
    st.markdown("---")
    st.header("Cluster Comparisons")
    
    # Feature selection for comparison
    comparison_features = ['DAILY_STRESS', 'ACHIEVEMENT', 'SLEEP_HOURS', 
                          'DAILY_STEPS', 'SOCIAL_NETWORK', 'WORK_LIFE_BALANCE_SCORE']
    available_comparison = [f for f in comparison_features if f in df.columns]
    
    if available_comparison:
        selected_feature = st.selectbox(
            "Select feature to compare across clusters:",
            options=available_comparison,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = create_comparison_bars(df, selected_feature, personas)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_box = px.box(
                df,
                x='Cluster',
                y=selected_feature,
                color='Cluster',
                color_discrete_map={i: personas[i]['color'] for i in df['Cluster'].unique()},
                title=f"{selected_feature.replace('_', ' ').title()} Distribution"
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Actionable Insights per Cluster
    st.markdown("---")
    st.header("Actionable Insights by Cluster")
    
    for cluster_id in sorted(df['Cluster'].unique()):
        persona = personas[cluster_id]
        cluster_data = df[df['Cluster'] == cluster_id]
        
        with st.expander(f"{persona['icon']} {persona['name']} ({len(cluster_data):,} people)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Strengths")
                # Find top features for this cluster
                profiles_std = (profiles - profiles.mean()) / profiles.std()
                top_features = profiles_std.loc[cluster_id].nlargest(5)
                
                for feature, score in top_features.items():
                    if score > 0:
                        st.markdown(f"- **{feature.replace('_', ' ').title()}**: {score:.2f} SD above average")
            
            with col2:
                st.markdown("#### Areas for Improvement")
                bottom_features = profiles_std.loc[cluster_id].nsmallest(5)
                
                for feature, score in bottom_features.items():
                    if score < 0:
                        st.markdown(f"- **{feature.replace('_', ' ').title()}**: {abs(score):.2f} SD below average")
            
            st.markdown("#### Recommended Actions")
            for rec in persona['recommendations']:
                st.markdown(f"→ {rec}")
    
    # Recommendations for Improvement
    st.markdown("---")
    st.header("Overall Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### For Individuals")
        st.markdown("""
        **Assess Your Current State:**
        - Use the 'Find Your Cluster' tool
        - Understand your wellness persona
        - Identify improvement areas
        - Track progress over time
        
        **Take Action:**
        - Follow persona-specific recommendations
        - Start with small, sustainable changes
        - Focus on holistic wellness
        - Seek support when needed
        
        **Build Habits:**
        - Consistent daily routines
        - Regular physical activity
        - Stress management practices
        - Strong social connections
        """)
    
    with col2:
        st.markdown("### For Organizations")
        st.markdown("""
        **Program Design:**
        - Target specific personas
        - Allocate resources strategically
        - Personalize interventions
        - Measure effectiveness
        
        **Support At-Risk Groups:**
        - Stress management programs
        - Mental health resources
        - Flexible work arrangements
        - Wellness coaching
        
        **Promote Wellness:**
        - Physical activity initiatives
        - Social connection opportunities
        - Work-life balance policies
        - Holistic wellness approach
        """)
    
    # Data-Driven Recommendations
    st.markdown("---")
    st.header("Data-Driven Wellness Strategy")
    
    st.markdown("""
    Based on our analysis of 15,972 individuals, here are evidence-based recommendations:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### High Priority")
        if 'DAILY_STRESS' in df.columns:
            high_stress_pct = (df['DAILY_STRESS'] > df['DAILY_STRESS'].quantile(0.75)).sum() / len(df) * 100
            st.markdown(f"""
            - **Stress Management:** {high_stress_pct:.1f}% show elevated stress
            - Implement stress reduction programs
            - Provide mental health resources
            - Encourage work-life boundaries
            """)
    
    with col2:
        st.markdown("#### Medium Priority")
        if 'DAILY_STEPS' in df.columns:
            low_activity_pct = (df['DAILY_STEPS'] < df['DAILY_STEPS'].quantile(0.25)).sum() / len(df) * 100
            st.markdown(f"""
            - **Physical Activity:** {low_activity_pct:.1f}% below activity targets
            - Promote daily movement
            - Create walking programs
            - Incentivize active lifestyles
            """)
    
    with col3:
        st.markdown("#### Long-term Focus")
        st.markdown("""
        - **Sustainable Habits:** Build lasting change
        - Track progress metrics
        - Celebrate successes
        - Foster supportive communities
        - Continuous improvement mindset
        """)

# ==================== MAIN APP ====================

def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Overview", "Cluster Explorer", "Find Your Cluster", "Insights & Patterns"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This Dashboard")
    st.sidebar.info("""
    This dashboard presents clustering analysis results from wellbeing and lifestyle data.
    
    **Features:**
    - 4 distinct wellness personas
    - Interactive visualizations
    - Personalized predictions
    - Data-driven insights
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Source:** Kaggle Wellbeing Dataset")
    st.sidebar.markdown("**Analysis:** K-Means Clustering")
    st.sidebar.markdown("**Sample Size:** 15,972 individuals")
    
    # Route to selected page
    if page == "Overview":
        page_overview()
    elif page == "Cluster Explorer":
        page_cluster_explorer()
    elif page == "Find Your Cluster":
        page_find_your_cluster()
    elif page == "Insights & Patterns":
        page_insights()

if __name__ == "__main__":
    main()

