import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# CSS Styling
# CSS Styling
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #eaf4ff !important;
}

[data-testid="stHeader"] {
    background-color: #eaf4ff !important;
}

/* Predict Button */
div.stButton > button {
    background-color: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 25px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}

div.stButton > button:hover {
    background-color: #1d4ed8 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="
    background-color: #dbeafe;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 25px;
">
    <h1 style="
        color: #1e3a8a;
        margin: 0;
        font-size: 38px;
    ">
        💻 LAPTOP PRICE PREDICTOR
    </h1>
</div>
""", unsafe_allow_html=True)

st.write("Enter your Laptop Specifications to predict its price.")

# Load model and feature columns
ann_model = tf.keras.models.load_model("laptop_price_ann.keras")

scaler_X = joblib.load("ann_scaler_X.pkl")
scaler_y = joblib.load("ann_scaler_y.pkl")

feature_columns = joblib.load("feature_columns.pkl")

def predict_laptop_price(sample):
    
    sample_df = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Numerical features
    numerical_features = [
        'spec_rating', 'Ram', 'ROM', 'display_size',
        'resolution_width', 'resolution_height',
        'warranty', 'is_gaming', 'gpu_memory',
        'cpu_cores', 'cpu_threads'
    ]

    for feature in numerical_features:
        sample_df[feature] = sample[feature]

    # Categorical features
    categorical_features = [
        'brand',
        'Ram_type',
        'ROM_type',
        'processor_brand',
        'processor_family',
        'gpu_brand',
        'os_family'
    ]

    for feature in categorical_features:
        column_name = feature + '_' + sample[feature]
        sample_df[column_name] = 1

    sample_scaled = scaler_X.transform(sample_df)

    prediction_scaled = ann_model.predict(sample_scaled, verbose=0)

    prediction = scaler_y.inverse_transform(prediction_scaled)

    return prediction[0][0]

# User Inputs
st.subheader("Laptop Specifications")

spec_rating = st.number_input(
    "Specification Rating",
    min_value=0,
    max_value=100,
    value=50
)

ram = st.number_input(
    "RAM (GB)",
    min_value=2,
    max_value=64,
    value=8,
    step=2
)

rom = st.number_input(
    "Storage / ROM (GB)",
    min_value=64,
    max_value=4096,
    value=512,
    step=64
)

display_size = st.number_input(
    "Display Size (inches)",
    min_value=10.0,
    max_value=20.0,
    value=15.6,
    step=0.1
)

resolution_width = st.number_input(
    "Resolution Width",
    min_value=800,
    max_value=5000,
    value=1920,
    step=10
)

resolution_height = st.number_input(
    "Resolution Height",
    min_value=600,
    max_value=3000,
    value=1080,
    step=10
)

warranty = st.number_input(
    "Warranty (years)",
    min_value=0,
    max_value=5,
    value=1,
    step=1
)

is_gaming = st.selectbox(
    "Gaming Laptop?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

gpu_memory = st.number_input(
    "GPU Memory (GB)",
    min_value=0,
    max_value=24,
    value=4,
    step=1
)

cpu_cores = st.number_input(
    "CPU Cores",
    min_value=1,
    max_value=32,
    value=4,
    step=1
)

cpu_threads = st.number_input(
    "CPU Threads",
    min_value=1,
    max_value=64,
    value=8,
    step=1
)

st.subheader("Laptop Configuration")

brand = st.selectbox(
    "Brand",
    [
        "Acer", "Apple", "Asus", "Avita", "Chuwi", "Dell",
        "Fujitsu", "Gigabyte", "HP", "Honor", "Huawei",
        "Infinix", "LG", "Lenovo", "MSI", "Microsoft",
        "Ninkear", "Primebook", "Razer", "Realme", "Samsung",
        "Tecno", "Ultimus", "Vaio", "Walker", "Wings",
        "Xiaomi", "Zebronics", "iBall"
    ]
)

ram_type = st.selectbox(
    "RAM Type",
    [
        "DDR3", "DDR4", "DDR5",
        "LPDDR4", "LPDDR4X",
        "LPDDR5", "LPDDR5X", "LPDDR5x",
        "Unified"
    ]
)

rom_type = st.selectbox(
    "ROM Type",
    ["SSD"]
)

st.subheader("Processor & Graphics")

processor_brand = st.selectbox(
    "Processor Brand",
    ["Apple", "Intel", "MediaTek"]
)

processor_family = st.selectbox(
    "Processor Family",
    [
        "Celeron",
        "Core i3",
        "Core i5",
        "Core i7",
        "Core i9",
        "Pentium",
        "Ryzen 3",
        "Ryzen 5",
        "Ryzen 7",
        "Ryzen 9"
    ]
)

gpu_brand = st.selectbox(
    "GPU Brand",
    ["ARM", "Apple", "INTEL", "Intel", "NVIDIA", "Unknown"]
)

st.subheader(" Operating System")

os_family = st.selectbox(
    "Operating System",
    ["Chrome", "DOS", "Mac", "Ubuntu", "Windows"]
)

# Predict Price Button

if st.button("Predict Laptop Price"):

    sample = {
        'spec_rating': spec_rating,
        'Ram': ram,
        'ROM': rom,
        'display_size': display_size,
        'resolution_width': resolution_width,
        'resolution_height': resolution_height,
        'warranty': warranty,
        'is_gaming': is_gaming,
        'gpu_memory': gpu_memory,
        'cpu_cores': cpu_cores,
        'cpu_threads': cpu_threads,
        'brand': brand,
        'Ram_type': ram_type,
        'ROM_type': rom_type,
        'processor_brand': processor_brand,
        'processor_family': processor_family,
        'gpu_brand': gpu_brand,
        'os_family': os_family
    }

    price = predict_laptop_price(sample)

    st.success("Price predicted successfully using ANN (TensorFlow/Keras).")
    st.caption(
        "Model: Artificial Neural Network (ANN) | "
        "Framework: TensorFlow/Keras"
    )
    

    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f8ff;
            text-align: center;
            margin-top: 20px;
        ">
            <h2> Predicted Laptop Price</h2>
            <h1>₹{price:,.0f}</h1>
    </div>
    """,
    unsafe_allow_html=True
    )