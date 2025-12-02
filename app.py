import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Shrimp THC Counter",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #f5f5f5 !important;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
    .protocol-section {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .status-low {
        color: #ff6b6b;
        font-weight: bold;
    }
    .status-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .status-high {
        color: #51cf66;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the YOLO model
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to run inference
def run_inference(image, model):
    results = model(image)
    return results

# Function to draw bounding boxes
def draw_boxes(image, results):
    annotated_image = image.copy()
    for result in results[0].boxes:
        box = result.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return annotated_image

# Function to calculate THC
def calculate_thc(count, dilution_factor=1):
    """
    Calculate THC using the formula:
    THC (cells/μL) = count * dilution_factor * 71.42857142857 (approx volume occupied by microscope view area in Neubauer with 40x objective is about 0.014 microliters -0.00025 microliters occupied by the smallest square)
    Return in cells/mL (multiply by 1000 from cells/μL)
    """
    thc_per_ul = count * dilution_factor * 71.42857142857
    thc_per_ml = thc_per_ul * 1000
    return thc_per_ul, thc_per_ml

# Function to get recommendation
def get_recommendation(thc_value):
    if thc_value < 1.0e6:
        return "low", {
            "range": "<1.0×10⁶ cells/mL",
            "status": "🔴 LOW IMMUNITY",
            "water": "Monitor parameters that stress shrimp, especially Vibrio counts, ammonia and nitrites.",
            "pond": "Central drain flushing and disinfectant application advised.",
            "feeding": "Immune stimulant feed coating at a minimum rate of 50%."
        }
    elif thc_value <= 1.0e7:
        return "medium", {
            "range": "1.0×10⁶ - 1.0×10⁷ cells/mL",
            "status": "🟡 MODERATE IMMUNITY",
            "water": "Monitor parameters that stress shrimp, especially Vibrio counts, ammonia and nitrites.",
            "pond": "Central drain flushing and disinfectant application advised.",
            "feeding": "Immune-enhance feed inclusion in the range of 25 to 50%."
        }
    else:
        return "high", {
            "range": ">1.0×10⁷ cells/mL",
            "status": "🟢 GOOD IMMUNITY",
            "water": "Maintain regular water monitoring.",
            "pond": "Continue with current management.",
            "feeding": "Continue with existing feeding unless unstable weather is forecasted or disease season is starting."
        }

# Sidebar - Model selection and settings
st.sidebar.title("⚙️ Configuration")
model_path = st.sidebar.text_input(
    "YOLO Model Path",
    value="model.pt",
    help="Path to your trained YOLO model"
)

dilution_factor = st.sidebar.slider(
    "Dilution Factor",
    min_value=1,
    max_value=10,
    value=1,
    help="Dilution factor used in sample preparation (1 = no dilution)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About THC
**Total Hemocyte Count (THC)** measures the number of hemocytes in shrimp hemolymph. Higher THC indicates better 
immune status and a higher disease resistance.
""")

# Main app
st.title("🦐 Shrimp Hemocyte Counter (THC)")
st.markdown("*AI-powered Total Hemocyte Count analysis for aquaculture management*")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📋 Protocol", "📚 Information"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_image = st.file_uploader(
            "Choose a hemolymph image",
            type=["jpg", "jpeg", "png"],
            help="Upload a microscopic image of hemolymph sample from hemocytometer"
        )
    
    with col2:
        st.subheader("Sample Image Reference")
        st.info("The image should show hemocytes in a hemocytometer chamber (Neubauer chamber). The grid lines help identify the counting area.")
    
    if uploaded_image is not None:
        # Load image
        image = Image.open(uploaded_image)
        image_array = np.array(image)
        
        # Load model
        model = load_model(model_path)
        
        if model is not None:
            # Run inference
            with st.spinner("Running hemocyte detection..."):
                results = run_inference(image_array, model)
            
            # Draw boxes
            annotated_image = draw_boxes(image_array, results)
            
            # Count detections
            total_hemocytes = len(results[0].boxes)
            
            # Calculate THC
            thc_per_ul, thc_per_ml = calculate_thc(total_hemocytes, dilution_factor)
            
            # Display results
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Hemocytes Detected",
                    value=total_hemocytes,
                    delta="cells in frame"
                )
            
            with col2:
                st.metric(
                    label="THC (cells/μL)",
                    value=f"{thc_per_ul:.2e}",
                    delta="per microliter"
                )
            
            with col3:
                st.metric(
                    label="THC (cells/mL)",
                    value=f"{thc_per_ml:.2e}",
                    delta="per milliliter"
                )
            
            # Get recommendation
            category, recommendation = get_recommendation(thc_per_ml)
            
            # Display images
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detected Hemocytes")
                st.image(annotated_image, channels="RGB", use_column_width=True)
            
            # Display recommendation
            st.markdown("---")
            st.subheader(f"{recommendation['status']}")
            st.markdown(f"**THC Range:** {recommendation['range']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="recommendation-card">
                <strong>💧 Water Management</strong><br>
                """ + recommendation['water'] + """
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="recommendation-card">
                <strong>🌊 Pond Bottom Management</strong><br>
                """ + recommendation['pond'] + """
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="recommendation-card">
                <strong>🍽️ Feeding Management</strong><br>
                """ + recommendation['feeding'] + """
                </div>
                """, unsafe_allow_html=True)
            
            # Download results
            st.markdown("---")
            st.subheader("📥 Export Results")
            
            results_text = f"""
HEMOCYTE COUNT ANALYSIS RESULTS
================================
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Dilution Factor: {dilution_factor}x

COUNTS:
- Hemocytes Detected: {total_hemocytes}
- THC (cells/μL): {thc_per_ul:.2e}
- THC (cells/mL): {thc_per_ml:.2e}

STATUS: {recommendation['status']}
THC Range: {recommendation['range']}

RECOMMENDATIONS:
Water Management: {recommendation['water']}
Pond Bottom Management: {recommendation['pond']}
Feeding Management: {recommendation['feeding']}
            """
            
            st.download_button(
                label="📄 Download Results as Text",
                data=results_text,
                file_name="THC_analysis_results.txt",
                mime="text/plain"
            )

with tab2:
    st.subheader("📋 THC Sampling Protocol")
    st.markdown("""
    This protocol describes the standard procedure for collecting and analyzing hemolymph samples 
    to determine Total Hemocyte Count (THC) in shrimp using a hemocytometer.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Materials Required
        
        - **Sample Collection:**
          - Shrimp (healthy and in good condition)
          - Dissection tools (scissors, forceps)
          - Microcentrifuge tubes (1.5 mL)
          - Sodium citrate solution (3.8% w/v anticoagulant)
          - Isotonic saline solution (0.9% NaCl, optional for dilution)
        
        - **Analysis Equipment:**
          - Hemocytometer (Neubauer chamber)
          - Cover slip
          - Micropipettes (10 µL, 200 µL)
          - Microscope with phase-contrast lens (40x magnification minimum)
          - 70% ethanol for sterilization
        
        ### Sample Preparation
        
        1. **Shrimp Preparation**
           - Maintain shrimp in appropriate conditions prior to sampling
           - Anesthetize shrimp using suitable method (e.g., clove oil)
        
        2. **Hemolymph Collection**
           - Clean the ventral sinus (base of first abdominal segment) with 70% ethanol
           - Sterilize dissection tools
           - Puncture ventral sinus with needle attached to syringe containing 100 μL sodium citrate
           - Withdraw hemolymph slowly, avoiding air bubbles
           - Mix gently with sodium citrate (anticoagulant)
        
        3. **Dilution (Optional)**
           - If hemolymph is concentrated, dilute with isotonic saline containing sodium citrate
           - Common dilutions: 1:1, 1:2, 1:5 (adjust based on expected count)
           - Record dilution factor for calculation
        
        ### Counting Procedure
        
        1. **Chamber Loading**
           - Load hemocytometer with diluted hemolymph using micropipette
           - Place cover slip on chamber
           - Avoid air bubbles
        
        2. **Microscope Setup**
           - Focus on central counting area using 40x objective lens
           - Use phase-contrast mode for better visibility
           - Identify the four corner squares (large squares)
        
        3. **Cell Counting**
           - Count all hemocytes within the four corner squares
           - Count at least 100 cells for statistical accuracy
           - Perform duplicate counts to improve accuracy
        
        ### THC Calculation
        
        **Formula:**
        ```
        THC (cells/μL) = (Hemocytes counted) × (Dilution factor) × 10⁴
        THC (cells/mL) = THC (cells/μL) × 1,000
        ```
        
        Where:
        - Hemocytes counted = total number of hemocytes in the four corner squares
        - Dilution factor = ratio of sample to diluent (1 if undiluted)
        - 10⁴ = conversion factor for hemocytometer chamber volume
        
        ### Important Notes
        
        - Work quickly to minimize hemocyte clumping
        - Maintain cool temperature throughout the procedure
        - Perform duplicate counts for each sample
        - Adjust protocol based on shrimp species and research objectives
        - Always follow ethical and animal welfare guidelines
        """)
    
    with col2:
        st.markdown("### Hemocytometer Layout")
        st.info("""
        **Neubauer Chamber Specifications:**
        - Total area: 9 mm²
        - Depth: 0.1 mm
        - Volume per small square: 1/4000 µL
        - Volume per large corner square: 1/400 µL
        
        **Counting Area:**
        - Use 4 corner large squares
        - Each large square has 16 small squares
        - Standard procedure: count in all 4 corners
        """)

with tab3:
    st.subheader("📚 THC Interpretation & Management Guidelines")
    
    st.markdown("""
    ### Understanding THC Values
    
    Total Hemocyte Count reflects the immune status and health of the shrimp. Different THC ranges 
    indicate different management strategies needed for optimal pond productivity.
    """)
    
    # Display THC ranges with recommendations
    thc_ranges = [
        {
            "range": "<1.0×10⁶ cells/mL",
            "status": "🔴 LOW IMMUNITY",
            "color": "#ff6b6b",
            "water": "Monitor parameters that stress shrimp, especially Vibrio counts, ammonia and nitrites.",
            "pond": "Central drain flushing and disinfectant application advised.",
            "feeding": "Immune stimulant feed coating at a minimum rate of 50%."
        },
        {
            "range": "1.0×10⁶ - 1.0×10⁷ cells/mL",
            "status": "🟡 MODERATE IMMUNITY",
            "color": "#ffa500",
            "water": "Monitor parameters that stress shrimp, especially Vibrio counts, ammonia and nitrites.",
            "pond": "Central drain flushing and disinfectant application advised.",
            "feeding": "Immune-enhance feed inclusion in the range of 25 to 50%."
        },
        {
            "range": ">1.0×10⁷ cells/mL",
            "status": "🟢 GOOD IMMUNITY",
            "color": "#51cf66",
            "water": "Maintain regular water monitoring.",
            "pond": "Continue with current management.",
            "feeding": "Continue with existing feeding unless unstable weather is forecasted or disease season is starting."
        }
    ]
    
    for item in thc_ranges:
        st.markdown(f"""
        <div class="recommendation-card" style="border-left: 5px solid {item['color']};">
        <h4>{item['status']}</h4>
        <strong>Range:</strong> {item['range']}<br><br>
        <strong>💧 Water Management:</strong> {item['water']}<br><br>
        <strong>🌊 Pond Bottom Management:</strong> {item['pond']}<br><br>
        <strong>🍽️ Feeding Management:</strong> {item['feeding']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("🔬 Key Factors Affecting THC")
    st.markdown("""
    - **Stress levels:** High stress (poor water quality, crowding) reduces THC
    - **Disease pressure:** Viral/bacterial infections trigger immune response
    - **Nutrition:** Adequate nutrition supports immune cell production
    - **Environmental conditions:** Temperature, salinity, dissolved oxygen
    - **Feed quality:** Immune-stimulating ingredients boost THC
    - **Sampling timing:** THC varies throughout the day and season
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Monitoring Recommendations")
    st.markdown("""
    - **Frequency:** Sample every 5-7 days during production cycle
    - **Trend analysis:** Monitor THC trends, not just single values
    - **Multiple shrimp:** Sample at least 5-10 shrimp per pond for accuracy
    - **Consistent timing:** Sample at same time each day for consistency
    - **Record keeping:** Document results for management decisions
    - **Weather awareness:** Increase monitoring during weather changes
    """)

st.markdown("---")
st.markdown("""
<center>
<small>
🦐 Shrimp THC Counter v1.0 | Powered by YOLO Object Detection<br>
For aquaculture management and immune monitoring
</small>
</center>
""", unsafe_allow_html=True)
