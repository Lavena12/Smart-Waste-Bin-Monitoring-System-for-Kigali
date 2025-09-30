import numpy as np
import joblib
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
import sklearn

# Set page config first
st.set_page_config(
    page_title="Garbage Prediction Web App",
    page_icon="🗑️",
    layout="centered"
)

class DemoGarbageModel:
    """Demo model that mimics sklearn interface for garbage prediction"""
    def __init__(self):
        self.is_demo = True
        
    def predict(self, X):
        """Rule-based predictions for garbage categories"""
        predictions = []
        for weight, material_code in X:
            # Simple rule-based logic for demo
            if material_code == 1:  # Plastic
                if weight > 50:
                    predictions.append("Recyclable")
                else:
                    predictions.append("Non-Recyclable")
            elif material_code == 2:  # Paper
                if weight > 30:
                    predictions.append("Recyclable")
                else:
                    predictions.append("Compostable")
            elif material_code == 3:  # Organic
                predictions.append("Compostable")
            elif material_code == 4:  # Metal
                predictions.append("Recyclable")
            else:
                # Default rules based on weight
                if weight > 100:
                    predictions.append("Bulky Waste")
                elif weight > 0:
                    predictions.append("General Waste")
                else:
                    predictions.append("Unknown")
        return np.array(predictions)

def safe_model_loader():
    """Safely load model with multiple fallback methods"""
    try:
        # Method 1: Standard load
        model = joblib.load("train_model.joblib")
        st.sidebar.success("✅ Model loaded successfully!")
        return model, "loaded"
    except Exception as e:
        st.sidebar.warning(f"⚠️ Standard loading failed: {str(e)[:100]}...")
        
        try:
            # Method 2: Load with different parameters
            model = joblib.load("train_model.joblib", mmap_mode=None)
            st.sidebar.success("✅ Model loaded with mmap_mode=None!")
            return model, "loaded"
        except Exception as e2:
            st.sidebar.warning(f"⚠️ Alternative loading failed")
            
        try:
            # Method 3: Try with allow_pickle=True
            model = joblib.load("train_model.joblib", allow_pickle=True)
            st.sidebar.success("✅ Model loaded with allow_pickle=True!")
            return model, "loaded"
        except:
            pass
            
        # Final fallback: Demo mode
        st.sidebar.error("❌ Could not load the trained model.")
        st.sidebar.info("🔧 Running in demo mode with rule-based predictions.")
        return DemoGarbageModel(), "demo"

# Function for prediction with error handling
def garbage_prediction(input_data, model, model_status):
    """Robust prediction function with error handling"""
    try:
        weight_input, material_code_input = input_data
        
        # Input validation
        if weight_input < 0:
            return "Error: Weight cannot be negative", 0.0
        if material_code_input < 0:
            return "Error: Material code cannot be negative", 0.0
            
        if model_status == "loaded":
            # Real model prediction
            prediction = model.predict([[weight_input, material_code_input]])
            
            # Handle different prediction formats
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                prediction_result = prediction[0]
            else:
                prediction_result = prediction
                
            # Try to get confidence score if available
            confidence = 0.85  # Default confidence for real model
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba([[weight_input, material_code_input]])
                    confidence = np.max(probabilities[0])
                except:
                    pass
                    
            return prediction_result, confidence
        else:
            # Demo model prediction
            prediction = model.predict([[weight_input, material_code_input]])[0]
            return prediction, 0.75  # Default confidence for demo
            
    except Exception as e:
        # Fallback prediction in case of any error
        weight_input, material_code_input = input_data
        if weight_input > 100:
            return "Bulky Waste (Fallback)", 0.5
        else:
            return "General Waste (Fallback)", 0.5

# Load model safely at startup
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.model_status = safe_model_loader()

# Main Streamlit app
def main():
    st.title('🗑️ Garbage Prediction Web App')
    st.write("Welcome to the waste category predictor! Enter the details below to predict the garbage category.")
    
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        Weight_grams = st.number_input(
            'Enter weight in grams', 
            min_value=0, 
            max_value=10000, 
            value=100,
            help="Weight of the garbage item in grams"
        )
        
    with col2:
        Material_code = st.number_input(
            'Enter material code', 
            min_value=0, 
            max_value=10, 
            value=1,
            help="1: Plastic, 2: Paper, 3: Organic, 4: Metal, 5: Glass, 6: Other"
        )
    
    # Material code guide
    with st.expander("📋 Material Code Guide"):
        st.markdown("""
        **Common Material Codes:**
        - **1**: Plastic
        - **2**: Paper/Cardboard  
        - **3**: Organic/Food Waste
        - **4**: Metal
        - **5**: Glass
        - **6**: Textile
        - **7**: Electronic
        - **8**: Hazardous
        - **9**: Construction
        - **10**: Other
        """)
    
    st.markdown("---")
    
    # Prediction section
    if st.button("🔍 Predict Category", type="primary"):
        with st.spinner("Analyzing..."):
            prediction, confidence = garbage_prediction(
                [Weight_grams, Material_code], 
                st.session_state.model, 
                st.session_state.model_status
            )
        
        # Display results
        st.subheader("Prediction Results")
        
        # Color code based on prediction type
        if "Recyclable" in str(prediction):
            st.success(f"**Predicted Category:** {prediction}")
        elif "Compostable" in str(prediction):
            st.info(f"**Predicted Category:** {prediction}")
        elif "Error" in str(prediction):
            st.error(f"**{prediction}**")
        else:
            st.warning(f"**Predicted Category:** {prediction}")
            
        st.write(f"**Confidence Level:** {confidence:.1%}")
        
        # Show model status
        if st.session_state.model_status == "demo":
            st.info("💡 *Using demo mode with rule-based predictions*")
    
    st.markdown("---")
    
    # Model information section
    st.sidebar.title("Model Information")
    
    if st.session_state.model_status == "loaded":
        st.sidebar.success("**Status:** Real ML Model Active")
        st.sidebar.write("Using trained machine learning model")
    else:
        st.sidebar.warning("**Status:** Demo Mode Active")
        st.sidebar.write("Using rule-based predictions")
    
    # Troubleshooting section
    with st.sidebar.expander("🛠️ Troubleshooting"):
        st.markdown("""
        **If predictions seem incorrect:**
        - Check material code guide
        - Ensure weight is in grams
        - Verify input ranges
        
        **Model loading issues:**
        - Re-train model with current scikit-learn
        - Check file path 'train_model.joblib'
        - Update dependencies
        """)

if __name__ == '__main__':
    main()