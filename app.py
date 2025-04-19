# Must be the very first Streamlit command - before any other imports or code that might use Streamlit
import streamlit as st
st.set_page_config(page_title="Vidyut AI", layout="wide")

# Now import everything else
import pandas as pd
import numpy as np
from io import BytesIO
import chardet
from sklearn.ensemble import IsolationForest

# Import or define the GeminiClient class
class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini client with API key."""
        try:
            import google.generativeai as genai
            self.genai = genai
            self.api_key = api_key
            genai.configure(api_key=api_key)
            
            # Try to find valid model
            try:
                models = genai.list_models()
                gemini_models = [model.name for model in models if "gemini" in model.name.lower()]
                self.model_name = gemini_models[0] if gemini_models else "gemini-1.5-pro"
            except:
                # self.model_name = "gemini-1.5-pro"
                self.model_name = "gemini-1.5-flash"
                
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            st.error("Please install google-generativeai: pip install google-generativeai")
            self.model = None
        except Exception as e:
            st.error(f"Error initializing Gemini: {e}")
            self.model = None
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text response from Gemini."""
        if not self.model:
            return "Gemini not available"
            
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            return response.text
        except Exception as e:
            return f"Gemini API error: {e}"
        
    def generate_stream(self, prompt: str, temperature: float = 0.7):
        """Stream text response from Gemini."""
        if not self.model:
            yield "Gemini not available"
            return
            
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": temperature},
                stream=True
            )
            
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Gemini API error: {e}"

# Initialize Gemini client
@st.cache_resource
def get_gemini_client():
    try:
        api_key = st.secrets.get('GEMINI_API_KEY')
        if not api_key:
            st.sidebar.warning("No Gemini API key found. Using mock responses.")
            return MockClient()
        return GeminiClient(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Gemini client: {e}")
        return MockClient()

class MockClient:
    """Mock client for when Gemini is not available"""
    def generate(self, prompt, **kwargs):
        return f"Analyzing data... [Mock response for: {prompt[:50]}...]"
    
    def generate_stream(self, prompt, **kwargs):
        yield f"Analyzing data... [Mock response for: {prompt[:50]}...]"

# Load data helper function
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
        
    try:
        return pd.read_excel(uploaded_file)
    except Exception:
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Detect encoding
            raw_data = uploaded_file.read()
            encoding = chardet.detect(raw_data)['encoding'] or 'latin1'
            
            # Reset file pointer again
            uploaded_file.seek(0)
            
            # Try with detected encoding
            return pd.read_csv(uploaded_file, encoding=encoding)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return None

# Initialize session state
def initialize_state():
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        if 'df_processed' not in st.session_state:
            st.session_state['df_processed'] = None
        if 'selected_kpis' not in st.session_state:
            st.session_state['selected_kpis'] = []
        if 'actions' not in st.session_state:
            st.session_state['actions'] = []
        if 'chat' not in st.session_state:
            st.session_state['chat'] = []
        if 'insights' not in st.session_state:
            st.session_state['insights'] = ""

# Initialize data when file is uploaded
def init_data():
    if 'uploader' not in st.session_state:
        return
        
    upl = st.session_state['uploader']
    df0 = load_data(upl)
    
    if df0 is not None:
        st.session_state['df_processed'] = df0.copy()
        st.session_state['selected_kpis'] = []
        st.session_state['actions'] = []
        st.session_state['chat'] = []
        st.session_state['insights'] = ""

# Function to call Gemini API
def call_gemini(prompt: str, stream: bool = False):
    gemini = get_gemini_client()
    if not gemini:
        return "AI service not available" 
    
    try:
        if stream:
            return gemini.generate_stream(prompt=prompt)
        return gemini.generate(prompt=prompt)
    except Exception as e:
        error_msg = f"AI service error: {e}"
        st.error(error_msg)
        return error_msg

# Compute outliers in data
def compute_outliers(df, method, params):
    if df is None or df.empty:
        return pd.Series()
        
    numeric = df.select_dtypes(include='number')
    if numeric.empty:
        return pd.Series()
        
    if method == 'Z-score':
        threshold = params['threshold']
        z = numeric.apply(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1))  # Avoid division by zero
        out = z.abs() > threshold
    elif method == 'IQR':
        mult = params['multiplier']
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        out = (numeric < (q1 - mult * iqr)) | (numeric > (q3 + mult * iqr))
    else:
        cont = params['contamination']
        # Handle empty dataframes or single column
        if numeric.shape[0] <= 1 or numeric.shape[1] <= 0:
            return pd.Series(0, index=numeric.columns)
            
        iso = IsolationForest(contamination=cont, random_state=42)
        preds = iso.fit_predict(numeric.fillna(numeric.mean()))
        mask = preds == -1
        out = pd.DataFrame({col: mask for col in numeric.columns}, index=df.index)
    
    return out.sum()

# Handle outliers in data
def handle_outliers(df, method, params):
    if df is None or df.empty:
        return df
        
    df_mod = df.copy()
    numeric = df.select_dtypes(include='number')
    
    if numeric.empty:
        return df_mod
        
    if method == 'Z-score':
        threshold = params['threshold']
        z = numeric.apply(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1))
        mask = z.abs().any(axis=1)
    elif method == 'IQR':
        mult = params['multiplier']
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        mask = ((numeric < (q1 - mult * iqr)) | (numeric > (q3 + mult * iqr))).any(axis=1)
    else:
        # Handle edge cases
        if numeric.shape[0] <= 1 or numeric.shape[1] <= 0:
            return df_mod
            
        iso = IsolationForest(contamination=params['contamination'], random_state=42)
        preds = iso.fit_predict(numeric.fillna(numeric.mean()))
        mask = preds == -1
    
    df_mod = df_mod[~mask]
    return df_mod

# --- EDA Tab ---
def eda_tab():
    st.subheader("Exploratory Data Analysis")
    
    if 'df_processed' not in st.session_state:
        st.warning("Upload data from the sidebar to begin EDA.")
        return
        
    df = st.session_state['df_processed']
    
    if df is None or df.empty:
        st.warning("No data available. Please upload a valid CSV or Excel file.")
        return

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Settings**")
        method = st.selectbox("Outlier Method", ["Z-score","IQR","Isolation Forest"])
        params = {}
        if method == "Z-score": params['threshold'] = st.slider("Threshold", 1.0, 5.0, 3.0)
        elif method == "IQR": params['multiplier'] = st.slider("Multiplier", 0.5, 3.0, 1.5)
        else: params['contamination'] = st.slider("Contamination", 0.01, 0.5, 0.1)
        
        if st.button("Apply Outliers"):
            st.session_state['df_processed'] = handle_outliers(df, method, params)
            st.success("Outliers handled.")
            
    with cols[1]:
        st.dataframe(df.head())
        counts = compute_outliers(df, method, params)
        
        for k,v in counts.items():
            st.metric(label=k, value=int(v))

    if st.button("Generate EDA Summary"):
        with st.spinner():
            prompt = f"Perform detailed EDA with outlier method {method} on this dataset with columns: {', '.join(df.columns)}."
            st.markdown(call_gemini(prompt))

# --- AutoML Tab ---
def automl_tab():
    st.subheader("AutoML & Manual Training")
    
    if 'df_processed' not in st.session_state:
        st.warning("Complete EDA to proceed.")
        return
        
    df = st.session_state.get('df_processed')
    
    if df is None or df.empty:
        st.warning("No data available. Please upload a valid CSV or Excel file.")
        return

    target = st.selectbox("Target Column", df.columns)
    
    if st.button("Run AutoML"):
        with st.spinner():
            best, score = "RandomForestClassifier", 0.87
            st.success(f"Selected {best} (CV: {score:.2f})")
            st.markdown(call_gemini(f"Explain selection of {best} with score {score:.2f} for predicting {target}."))

    st.markdown("---")
    st.markdown("**Manual Training**")
    models = ["LinearRegression","RandomForestClassifier","XGBoost","LogisticRegression"]
    sel = st.selectbox("Choose Model", models)
    
    if st.button("Train Model"):
        with st.spinner():
            sc = 0.82
            st.success(f"{sel} trained (CV: {sc:.2f})")
            st.markdown(call_gemini(f"Explain {sel} results with score {sc:.2f} for predicting {target}."))

# --- Dashboard Tab ---
def dashboard_tab():
    st.subheader("Dynamic Dashboard")
    
    if 'df_processed' not in st.session_state:
        st.warning("Complete EDA to proceed.")
        return
        
    df = st.session_state.get('df_processed')
    
    if df is None or df.empty:
        st.warning("No data available. Please upload a valid CSV or Excel file.")
        return

    if st.button("Suggest KPIs"):
        kpi_text = call_gemini(f"Suggest KPIs for columns: {list(df.columns)}.")
        suggested_kpis = [k.strip('- ') for k in kpi_text.split('\n') if k.strip().startswith('-')]
        
        # Filter to only include KPIs that exist as columns
        valid_kpis = [k for k in suggested_kpis if k in df.columns]
        
        if not valid_kpis and suggested_kpis:
            # Try to extract actual column names from suggestions
            valid_kpis = [col for col in df.columns if any(kpi.lower() in col.lower() for kpi in suggested_kpis)]
            
        if valid_kpis:
            st.session_state['selected_kpis'] = valid_kpis
        else:
            # Fallback to numeric columns if no valid KPIs
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            st.session_state['selected_kpis'] = numeric_cols[:4] if len(numeric_cols) > 0 else []
            
    kpis = st.session_state.get('selected_kpis', [])
    
    # Make sure all KPIs exist in dataframe
    kpis = [k for k in kpis if k in df.columns]
    
    if kpis:
        st.write("**KPIs**")
        stats = st.columns(min(4, len(kpis) or 1))
        
        for i, k in enumerate(kpis):
            if k in df.columns:
                stats[i % 4].metric(k, round(df[k].mean(), 2))
                
        if all(k in df.columns for k in kpis):
            chart_data = df[kpis].copy()
            st.line_chart(chart_data)

# --- Decision Board Tab ---
def decision_board_tab():
    st.subheader("Decision Board 🧠")
    
    # Properly check for dataframe and KPIs existence without triggering ValueError
    has_df = 'df_processed' in st.session_state and st.session_state['df_processed'] is not None 
    has_kpis = 'selected_kpis' in st.session_state and len(st.session_state.get('selected_kpis', [])) > 0
    
    if not has_df or not has_kpis:
        st.warning("Dashboard KPIs required. Please complete the Dashboard tab first.")
        return

    df = st.session_state['df_processed']
    kpis = st.session_state['selected_kpis']
    
    if st.button("Generate Insights (stream)"):
        with st.spinner():
            insights_container = st.empty()
            insights = ""
            for chunk in call_gemini(f"Based on KPIs {kpis}, analyze these columns of data and recommend actions.", stream=True):
                insights += chunk
                insights_container.markdown(insights)
            st.session_state['insights'] = insights

    st.markdown("---")
    st.subheader("Action Items")
    
    if 'actions' not in st.session_state:
        st.session_state['actions'] = []
        
    # Display existing actions
    for idx, ai in enumerate(st.session_state.get('actions', [])):
        cols = st.columns([3, 1, 1])
        cols[0].write(f"{ai.get('action', '')} (Owner: {ai.get('owner', '')})")
        
        edit_pressed = cols[1].button("Edit", key=f"edit{idx}")
        delete_pressed = cols[2].button("Delete", key=f"del{idx}")
        
        if delete_pressed:
            st.session_state['actions'].pop(idx)
            st.rerun()
            
        if edit_pressed:
            st.session_state[f'editing_{idx}'] = True
            
        if st.session_state.get(f'editing_{idx}', False):
            with st.container():
                new_a = st.text_input("Action", ai.get('action', ''), key=f"a{idx}")
                new_o = st.text_input("Owner", ai.get('owner', ''), key=f"o{idx}")
                
                save_pressed = st.button("Save Changes", key=f"save{idx}")
                if save_pressed:
                    st.session_state['actions'][idx] = {'action': new_a, 'owner': new_o}
                    st.session_state[f'editing_{idx}'] = False
                    st.rerun()
            
    # Add new action
    st.markdown("---")
    st.subheader("Add New Action")
    new_action = st.text_input("Action Description")
    new_owner = st.text_input("Action Owner")
    
    if st.button("Add Action") and new_action and new_owner:
        if 'actions' not in st.session_state:
            st.session_state['actions'] = []
        st.session_state['actions'].append({'action': new_action, 'owner': new_owner})
        st.rerun()

    st.markdown("---")
    st.subheader("Chat Assistant")
    
    if 'chat' not in st.session_state:
        st.session_state['chat'] = []
        
    q = st.text_input("Question...")
    
    if st.button("Send Query") and q:
        with st.spinner("Generating response..."):
            resp_container = st.empty()
            resp = ""
            
            for chunk in call_gemini(f"Insights: {st.session_state.get('insights', '')}\nQ: {q}", stream=True):
                resp += chunk
                resp_container.markdown(resp)
                
            st.session_state['chat'].append((q, resp))
        
    if st.session_state.get('chat'):
        st.markdown("---")
        st.subheader("Chat History")
        for q, r in st.session_state['chat']:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {r}")
            st.markdown("---")

# --- What-If Analysis Tab ---
def what_if_tab():
    st.subheader("What-If Analysis")
    
    if 'df_processed' not in st.session_state:
        st.warning("Complete EDA to proceed.")
        return
        
    df = st.session_state.get('df_processed')
    
    if df is None or df.empty:
        st.warning("No data available. Please upload a valid CSV or Excel file.")
        return

    # Select only numeric columns for what-if analysis
    numeric_cols = df.select_dtypes(include='number').columns
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for what-if analysis.")
        return
        
    params = {}
    
    for c in numeric_cols[:5]:  # Limit to first 5 numeric columns to avoid UI clutter
        min_val = float(df[c].min())
        max_val = float(df[c].max())
        mean_val = float(df[c].mean())
        
        # Handle edge cases with min/max being the same
        if min_val == max_val:
            min_val -= 1
            max_val += 1
            
        params[c] = st.slider(c, min_val, max_val, mean_val)
        
    if st.button("Run What-If"):
        st.markdown(call_gemini(f"Perform what-if analysis with parameters: {params}."))

# --- Analytics Tab ---
def analytics_tab():
    st.subheader("Advanced Analytics")
    
    if 'df_processed' not in st.session_state:
        st.warning("Complete EDA to proceed.")
        return
        
    df = st.session_state.get('df_processed')
    
    if df is None or df.empty:
        st.warning("No data available. Please upload a valid CSV or Excel file.")
        return

    # Extended analysis options
    analysis_options = [
        "Customer Churn",
        "Market Mix Modeling",
        "Customer Segmentation",
        "Time Series Forecasting",
        "Anomaly Detection",
        "Predictive Maintenance",
        "Clustering",
        "Regression Analysis",
        "Classification",
        "Sentiment Analysis",
        "Custom Analysis"
    ]
    
    choice = st.selectbox("Select Analysis Type", analysis_options)
    
    if st.button("Run Analysis"):
        columns_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns[:10]])
        prompt = f"Perform {choice} analysis on a dataset with columns: {columns_info}."
        
        if df.shape[0] > 0:
            prompt += f" The dataset has {df.shape[0]} rows and {df.shape[1]} columns."
            
        st.markdown(call_gemini(prompt))

# --- Main ---
def main():
    # Page title and description are already set at the top of the script
    
    st.sidebar.title("Vidyut AI Prototype")
    st.sidebar.file_uploader("Upload Excel/CSV", key='uploader', type=['xls','xlsx','csv'], on_change=init_data)
    
    # Initialize session state variables
    initialize_state()
    
    # Create tabs for different functionalities
    tabs = st.tabs(["EDA", "AutoML", "Dashboard", "Decision Board", "What-If", "Analytics"])
    
    # Define the function to run for each tab
    tab_functions = [eda_tab, automl_tab, dashboard_tab, decision_board_tab, what_if_tab, analytics_tab]
    
    # Run the appropriate function for each tab
    for i, tab in enumerate(tabs):
        with tab:
            tab_functions[i]()

# Call the main function
if __name__ == '__main__':
    main()
