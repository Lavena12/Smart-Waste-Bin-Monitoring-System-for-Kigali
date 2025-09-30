import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle

# Set page config first
st.set_page_config(
    page_title="Garbage Prediction App",
    page_icon="🗑️",
    layout="wide"
)

def safe_model_loader():
    """Safely load model with multiple fallback methods"""
    try:
        # Method 1: Standard load
        model = joblib.load("waste_collection_model.joblib")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.warning(f"⚠️ Standard loading failed: {str(e)[:100]}...")
        
        try:
            # Method 2: Load with different parameters
            model = joblib.load("train_model.joblib", mmap_mode=None)
            st.success("✅ Model loaded with mmap_mode=None!")
            return model
        except:
            pass
            
        try:
            # Method 3: Use pickle directly
            with open("train_model.joblib", 'rb') as f:
                model = pickle.load(f)
            st.success("✅ Model loaded with pickle!")
            return model
        except:
            pass
            
        # Final fallback: Demo mode
        st.error("❌ Could not load the trained model.")
        st.info("🔧 Running in demo mode with sample predictions.")
        return "demo"

# Load model safely
loaded_model = safe_model_loader()

# App title
st.title("🗑️ Garbage Prediction App")
st.markdown("---")

# Demo prediction function
def demo_prediction(features):
    """Provide demo predictions when model fails to load"""
    # Simple rule-based demo predictions
    if features[0] > 0.5 and features[1] > 0.3:  # Example rules
        return "Recyclable"
    else:
        return "Organic"

# Your existing input fields
st.header("Input Features")
col1, col2 = st.columns(2)

with col1:
    feature1 = st.slider("Feature 1", 0.0, 1.0, 0.5)
    feature2 = st.slider("Feature 2", 0.0, 1.0, 0.5)

with col2:
    feature3 = st.slider("Feature 3", 0.0, 1.0, 0.5)
    feature4 = st.slider("Feature 4", 0.0, 1.0, 0.5)

# Prediction
if st.button("Predict Garbage Type"):
    input_features = np.array([[feature1, feature2, feature3, feature4]])
    
    if loaded_model != "demo":
        try:
            prediction = loaded_model.predict(input_features)
            probability = loaded_model.predict_proba(input_features)
            
            st.success(f"**Prediction:** {prediction[0]}")
            st.write(f"**Confidence:** {np.max(probability[0]):.2%}")
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            # Fallback to demo
            demo_pred = demo_prediction([feature1, feature2, feature3, feature4])
            st.info(f"**Demo Prediction:** {demo_pred}")
    else:
        # Use demo prediction
        demo_pred = demo_prediction([feature1, feature2, feature3, feature4])
        st.info(f"**Demo Prediction:** {demo_pred}")

# Add troubleshooting info
with st.expander("Troubleshooting"):
    st.markdown("""
    **If you see this message, the model file has compatibility issues:**
    
    1. **Re-train the model** with current scikit-learn version
    2. **Check requirements.txt** for version compatibility
    3. **Upload a new model file** trained with scikit-learn >= 1.3.0
    
    **Current workaround:** The app uses demo predictions based on simple rules.
    """)
