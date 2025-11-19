import streamlit as st
import joblib
import os

# --- 0. Define Emotion Mapping ---
# This dictionary converts the numerical model output into readable emotion strings.
# joy=5, sadness=0, anger=1, fear=4, love=2, surprise=3
EMOTION_MAP = {
    5: 'joy',
    0: 'sadness',
    1: 'anger',
    4: 'fear',
    2: 'love',
    3: 'surprise'
}

# --- 1. Model Loading with Caching ---

@st.cache_resource
def load_models():
    """
    Loads saved models and vectorizer once using Streamlit's caching.
    """
    vectorizer_path = 'tfidf_vectorizer.pkl'
    lr_model_path = 'lr_model.pkl'
    nb_model_path = 'nb_model.pkl'

    # Check if files exist
    if not all(os.path.exists(p) for p in [vectorizer_path, lr_model_path, nb_model_path]):
        st.error(
            "🛑 **Model Files Missing!** Please ensure 'tfidf_vectorizer.pkl', 'lr_model.pkl', "
            "and 'nb_model.pkl' are in the same directory as this script."
        )
        return None, None, None

    try:
        vectorizer = joblib.load(vectorizer_path)
        lr_model = joblib.load(lr_model_path)
        nb_model = joblib.load(nb_model_path)
        return vectorizer, lr_model, nb_model
    except Exception as e:
        st.error(f"❌ **Error loading model files:** {e}")
        return None, None, None

# Load assets upon startup
VECTORIZER, LR_MODEL, NB_MODEL = load_models()

# --- 2. Prediction Logic ---

def get_predictions(text_input, vectorizer, lr_model, nb_model, emotion_map):
    """
    Transforms the input text, gets numerical predictions, and maps them to emotion strings.
    """
    if not vectorizer or not lr_model or not nb_model:
        return {"Error": "Models not loaded. Check startup logs."}
    
    try:
        # 1. Transform the input text
        text_vector = vectorizer.transform([text_input])
        
        # 2. Get numerical predictions from both models
        lr_num_pred = lr_model.predict(text_vector)[0]
        nb_num_pred = nb_model.predict(text_vector)[0]
        
        # 3. Map numerical predictions to emotion strings
        lr_pred_string = emotion_map.get(lr_num_pred, f"Unknown (Code: {lr_num_pred})")
        nb_pred_string = emotion_map.get(nb_num_pred, f"Unknown (Code: {nb_num_pred})")
        
        return {
            "Logistic Regression": lr_pred_string,
            "Naive Bayes": nb_pred_string
        }
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return {"Error": "Prediction Failed"}


# --- 3. Streamlit UI ---

def main():
    st.set_page_config(page_title="Dual Model Emotion Detector", layout="wide")
    st.title("🗣️ Text Emotion Analysis (LR & NB Models)")
    st.markdown("Enter a sentence to analyze its emotion using **Logistic Regression** and **Naive Bayes** models.")

    st.info("Both models have processed the input text and provided their emotion predictions. Logistic Regression accuracy= 86.3% & naive_bayes accuracy= 76.1825%" )
    # Stop if models failed to load
    if not all([VECTORIZER, LR_MODEL, NB_MODEL]):
        st.stop() 

    st.subheader("Input Text")
    text_input = st.text_area("Type the sentence you want to analyze here:", height=150, 
                              placeholder="e.g., I cannot believe this happened; I'm completely shocked!")

    if st.button("🤖 Analyze Emotion", help="Click to run the input through both models"):
        if text_input.strip() == "":
            st.warning("👈 Please enter some text to analyze.")
        else:
            with st.spinner("Processing text and running predictions..."):
                # Get the predictions using the global EMOTION_MAP
                results = get_predictions(text_input, VECTORIZER, LR_MODEL, NB_MODEL, EMOTION_MAP)
                
                st.subheader("Analysis Results")
                
                col1, col2 = st.columns(2)
                
                # Display LR Result
                with col1:
                    st.markdown("### 📊 **Logistic Regression (LR)**")
                    emotion_lr = results.get('Logistic Regression', 'Error')
                    st.metric(label="Predicted Emotion", value=emotion_lr)
                    st.success("✅ Prediction Complete")
                
                # Display NB Result
                with col2:
                    st.markdown("### 🧠 **Naive Bayes (NB)**")
                    emotion_nb = results.get('Naive Bayes', 'Error')
                    st.metric(label="Predicted Emotion", value=emotion_nb)
                    st.success("✅ Prediction Complete")

            st.info("The numerical model output was successfully mapped to emotion strings using the defined mapping (0=sadness, 5=joy, etc.).")

    st.markdown("---")

if __name__ == '__main__':
    main()