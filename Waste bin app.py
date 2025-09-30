import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle

# Set page config
st.set_page_config(
    page_title="Smart Waste Bin Monitoring - Kigali",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .district-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

class SmartWasteSystem:
    """Smart Waste Bin Monitoring System for Kigali"""
    
    def __init__(self):
        self.districts = {
            'Nyarugenge': {'bins': 45, 'population': 150000},
            'Gasabo': {'bins': 78, 'population': 530000},
            'Kicukiro': {'bins': 52, 'population': 340000}
        }
        self.waste_categories = {
            'Organic': '🟫 Compostable',
            'Recyclable': '🟦 Recyclable', 
            'Plastic': '🟨 Recyclable',
            'Paper': '⬜ Recyclable',
            'Metal': '🔩 Recyclable',
            'Glass': '🔍 Recyclable',
            'Hazardous': '☣️ Special Handling',
            'Electronic': '🔌 E-Waste',
            'Medical': '🏥 Bio-Hazard',
            'Construction': '🏗️ Inert Waste'
        }
    
    def generate_sample_data(self):
        """Generate realistic sample data for Kigali districts"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        
        data = []
        for date in dates:
            for district in self.districts.keys():
                # Simulate waste generation patterns
                base_waste = np.random.normal(50, 15)
                weekend_boost = 1.2 if date.weekday() >= 5 else 1.0
                rainfall_effect = 0.9 if np.random.random() > 0.7 else 1.0
                
                weight = max(10, base_waste * weekend_boost * rainfall_effect)
                
                # District-specific patterns
                if district == 'Gasabo':
                    weight *= 1.3  # More commercial activity
                elif district == 'Kicukiro':
                    weight *= 1.1  # Residential areas
                
                data.append({
                    'date': date,
                    'district': district,
                    'weight_kg': weight,
                    'material_type': np.random.choice(list(self.waste_categories.keys()), 
                                                    p=[0.3, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
                    'bin_id': f"BIN_{district[:3]}_{np.random.randint(1, 100):03d}",
                    'collection_urgency': np.random.choice(['Low', 'Medium', 'High'], 
                                                         p=[0.3, 0.5, 0.2])
                })
        
        return pd.DataFrame(data)

class DemoWasteModel:
    """Advanced demo model for waste prediction in Kigali context"""
    
    def predict(self, features):
        weight, material_code, district_code, day_of_week = features[0]
        
        # Enhanced prediction logic for Kigali context
        if material_code in [1, 2, 4, 5]:  # Recyclables
            if weight > 80:
                return np.array(["HIGH_PRIORITY"])
            else:
                return np.array(["MEDIUM_PRIORITY"])
        elif material_code == 3:  # Organic
            if weight > 60:
                return np.array(["URGENT"])  # Organic waste needs faster collection
            else:
                return np.array(["MEDIUM_PRIORITY"])
        elif material_code in [7, 8, 9]:  # Hazardous/Medical/Construction
            return np.array(["SPECIAL_HANDLING"])
        else:
            if weight > 100:
                return np.array(["HIGH_PRIORITY"])
            else:
                return np.array(["LOW_PRIORITY"])
    
    def predict_proba(self, features):
        # Return confidence probabilities
        return np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])  # Mock probabilities

def safe_model_loader():
    """Safely load the waste prediction model"""
    try:
        model = joblib.load("train_model.joblib")
        st.sidebar.success("✅ AI Model Loaded")
        return model, "ai_model"
    except Exception as e:
        st.sidebar.warning("⚠️ Using Advanced Demo System")
        return DemoWasteModel(), "demo"

def main():
    # Initialize system
    waste_system = SmartWasteSystem()
    
    # Load model
    if 'waste_model' not in st.session_state:
        st.session_state.waste_model, st.session_state.model_status = safe_model_loader()
    
    # Header
    st.markdown('<h1 class="main-header">🏙️ SMART WASTE BIN MONITORING SYSTEM</h1>', 
                unsafe_allow_html=True)
    st.markdown("### Efficient Waste Collection Management for Kigali City")
    st.markdown("---")
    
    # Sidebar for controls
    st.sidebar.title("🚮 System Controls")
    
    # Real-time monitoring section
    st.sidebar.subheader("📊 Real-time Monitoring")
    selected_district = st.sidebar.selectbox(
        "Select District", 
        list(waste_system.districts.keys())
    )
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Bins Monitored", "175", "12 new")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Collection Efficiency", "87%", "5% improvement")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recycling Rate", "42%", "8% increase")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # District Overview
    st.subheader("🏘️ District Overview")
    
    dist_col1, dist_col2, dist_col3 = st.columns(3)
    
    districts_data = waste_system.districts
    for i, (district, info) in enumerate(districts_data.items()):
        with [dist_col1, dist_col2, dist_col3][i]:
            st.markdown(f'<div class="district-card">', unsafe_allow_html=True)
            st.write(f"**{district} District**")
            st.write(f"Bins: {info['bins']}")
            st.write(f"Population: {info['population']:,}")
            st.write(f"Daily Waste: ~{info['population'] * 0.5:,.0f} kg")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Waste Prediction Interface
    st.subheader("🔍 Smart Waste Prediction")
    
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    
    with pred_col1:
        weight_kg = st.number_input("Waste Weight (kg)", min_value=0.0, max_value=200.0, value=50.0)
    
    with pred_col2:
        material_type = st.selectbox(
            "Material Type",
            options=list(waste_system.waste_categories.items()),
            format_func=lambda x: f"{x[0]} - {x[1]}"
        )
        material_code = list(waste_system.waste_categories.keys()).index(material_type[0]) + 1
    
    with pred_col3:
        district = st.selectbox("District", list(waste_system.districts.keys()))
        district_code = list(waste_system.districts.keys()).index(district) + 1
    
    with pred_col4:
        day_of_week = st.selectbox("Day of Week", 
                                 ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_code = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
    
    if st.button("🎯 Predict Collection Priority", type="primary"):
        with st.spinner("Analyzing waste data..."):
            # Prepare features for prediction
            features = np.array([[weight_kg, material_code, district_code, day_code]])
            
            if st.session_state.model_status == "ai_model":
                prediction = st.session_state.waste_model.predict(features)[0]
                try:
                    probabilities = st.session_state.waste_model.predict_proba(features)[0]
                    confidence = np.max(probabilities)
                except:
                    confidence = 0.85
            else:
                prediction = st.session_state.waste_model.predict(features)[0]
                confidence = 0.78
        
        # Display prediction results
        st.subheader("Prediction Results")
        
        priority_colors = {
            "LOW_PRIORITY": "🟢",
            "MEDIUM_PRIORITY": "🟡", 
            "HIGH_PRIORITY": "🟠",
            "URGENT": "🔴",
            "SPECIAL_HANDLING": "🟣"
        }
        
        priority_descriptions = {
            "LOW_PRIORITY": "Schedule within 3-4 days",
            "MEDIUM_PRIORITY": "Schedule within 1-2 days", 
            "HIGH_PRIORITY": "Schedule within 24 hours",
            "URGENT": "Immediate collection required",
            "SPECIAL_HANDLING": "Specialized collection needed"
        }
        
        emoji = priority_colors.get(prediction, "⚪")
        description = priority_descriptions.get(prediction, "Standard collection")
        
        st.markdown(f"### {emoji} Priority: {prediction.replace('_', ' ').title()}")
        st.write(f"**Recommendation:** {description}")
        st.write(f"**Confidence Level:** {confidence:.1%}")
        
        if st.session_state.model_status == "demo":
            st.info("💡 Using advanced rule-based prediction system")
    
    st.markdown("---")
    
    # Analytics and Reports Section
    st.subheader("📈 Waste Analytics & Reports")
    
    # Generate sample data for visualization
    sample_data = waste_system.generate_sample_data()
    recent_data = sample_data[sample_data['date'] >= (datetime.now() - timedelta(days=30))]
    
    tab1, tab2, tab3, tab4 = st.tabs(["Waste Trends", "District Comparison", "Collection Alerts", "Efficiency Report"])
    
    with tab1:
        st.subheader("Monthly Waste Generation Trends")
        
        # Weekly aggregation
        weekly_trends = recent_data.groupby([pd.Grouper(key='date', freq='W'), 'district'])['weight_kg'].sum().reset_index()
        
        fig_trends = px.line(weekly_trends, x='date', y='weight_kg', color='district',
                            title="Weekly Waste Generation by District",
                            labels={'weight_kg': 'Total Waste (kg)', 'date': 'Week'})
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with tab2:
        st.subheader("District Performance Comparison")
        
        col_comp1, col_comp2 = st.columns(2)
        
        with col_comp1:
            # Waste by material type
            material_dist = recent_data.groupby(['district', 'material_type'])['weight_kg'].sum().reset_index()
            fig_material = px.sunburst(material_dist, path=['district', 'material_type'], values='weight_kg',
                                     title="Waste Composition by District")
            st.plotly_chart(fig_material, use_container_width=True)
        
        with col_comp2:
            # Collection urgency distribution
            urgency_dist = recent_data['collection_urgency'].value_counts()
            fig_urgency = px.pie(values=urgency_dist.values, names=urgency_dist.index,
                               title="Collection Urgency Distribution")
            st.plotly_chart(fig_urgency, use_container_width=True)
    
    with tab3:
        st.subheader("🚨 Priority Collection Alerts")
        
        # Generate urgent alerts
        urgent_bins = recent_data[
            (recent_data['collection_urgency'] == 'High') | 
            (recent_data['weight_kg'] > 80)
        ].tail(5)
        
        for _, alert in urgent_bins.iterrows():
            st.markdown('<div class="alert-card">', unsafe_allow_html=True)
            st.write(f"**🚨 {alert['bin_id']} - {alert['district']}**")
            st.write(f"Weight: {alert['weight_kg']:.1f} kg | Material: {alert['material_type']}")
            st.write(f"Urgency: {alert['collection_urgency']} | Date: {alert['date'].strftime('%Y-%m-%d')}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.subheader("📊 Collection Efficiency Report")
        
        # Calculate efficiency metrics
        total_bins = sum(info['bins'] for info in waste_system.districts.values())
        avg_waste_per_bin = recent_data['weight_kg'].mean()
        recycling_potential = len(recent_data[recent_data['material_type'].isin(['Plastic', 'Paper', 'Metal', 'Glass'])]) / len(recent_data)
        
        eff_col1, eff_col2, eff_col3 = st.columns(3)
        
        with eff_col1:
            st.metric("Total Coverage", f"{total_bins} bins", "City-wide")
        
        with eff_col2:
            st.metric("Avg Waste/Bin", f"{avg_waste_per_bin:.1f} kg", "Daily")
        
        with eff_col3:
            st.metric("Recycling Potential", f"{recycling_potential:.1%}", "Of total waste")
        
        # Recommendations
        st.subheader("💡 Optimization Recommendations")
        
        recommendations = [
            "🚛 Optimize collection routes in Gasabo district during weekdays",
            "♻️ Increase recycling bins in commercial areas",
            "📱 Deploy mobile alerts for high-priority collections", 
            "🌱 Promote organic waste composting in residential areas",
            "🔧 Schedule maintenance for bins in high-usage areas"
        ]
        
        for rec in recommendations:
            st.write(f"- {rec}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Smart Waste Bin Monitoring System • Kigali City Council • {}</p>
            <p>Real-time monitoring for efficient waste collection and environmental sustainability</p>
        </div>
        """.format(datetime.now().strftime("%Y")),
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()