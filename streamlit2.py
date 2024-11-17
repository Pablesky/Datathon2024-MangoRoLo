import streamlit as st
from PIL import Image
import time  # For simulating the model prediction time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from MangoModel import MangoModel
import torch
from torchvision import transforms

# Import softmax
from torch.nn.functional import softmax

# Streamlit App Title and Description
st.set_page_config(page_title="Image Classification with ResNet", layout="wide")
st.title("🖼️ Design attributes logger enhanced by AI predictions")
st.markdown('#### Instructions')
st.write("Upload an image of a clothing item and click on 'Infer attributes' to view the results. The highest-scoring attribute along with its probability will be displayed. If the result is not satisfactory, you can expand a dropdown to view the next two highest-scoring attributes and select the one that best matches the image. Finally, click 'Save attributes' to save the selected annotations to the database and the page will be resetted for the next upload.")

mangoTransforms = transforms.Compose([
        transforms.Resize(224),               # Resize the shorter side to 224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

numClasses = [7, 7, 12, 6, 13, 34, 34, 7, 5, 5, 5]

# Mockup: Replace this with actual model loading code
def load_model():
    # Simulate model loading
    model = MangoModel(num_classes_list=numClasses)
    state_dict = torch.load("weights/mango_model.pth", map_location=torch.device("cuda"))
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    
    return model

# Initialize the model in session_state
if "model" not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model = load_model()

st.logo('logo_mango.png', size='large')

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 1

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key = st.session_state["uploader_key"])

# # Store the uploaded file in session state
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

# Create two columns layout
col1, col2 = st.columns([1, 2], gap="large")  # 50% width for each column

#Load json with attributes inside each class:
with open('attributes.json', 'r') as file:
    data = json.load(file)

# Dictionary of classes and their number of attributes (useful for the one hot encoder)
class_names_dict = {'Cane height type': 7, 'Closure placement': 7, 'Heel shape type': 12, 'Knit structure': 6, 'Length type':13, 'Neck lapel type': 34, 'Silhouette type': 34, 'Sleeve length type': 7, 'Toecap type': 5, 'Waist type':5, 'Woven structure': 5}


# Function to simulate model predictions
def simulate_prediction():
    image_input = Image.open(st.session_state.uploaded_file)
    
    # Preprocess the image
    image_input = mangoTransforms(image_input).unsqueeze(0)
    
    output_logits = st.session_state.model(image_input)
    
    # The outputs are in a list, so we need to apply softmax to each output
    probabilities = [softmax(out, dim=1).detach().numpy().squeeze() for out in output_logits]
    
    print(probabilities)
    
    # For each attribute, we simulate a prediction using random selection from some predefined categories
    predictions = {}
    for i, (key, value) in enumerate(class_names_dict.items()):
        type_vector = probabilities[i]
        first_3 = np.argsort(type_vector)[::-1][:3]
        att_list = []
        
        for item in first_3:
            att_list.append((data[key][item], type_vector[item]))

        predictions[key] = att_list

        # # Sample data
        # categories = data[key]
        # values = type_vector

        # # Create the bar plot
        # fig, ax = plt.subplots()
        # ax.bar(categories, values)

        # # Add title and labels
        # ax.set_title('Sample Bar Plot')
        # ax.set_xlabel('Categories')
        # ax.set_ylabel('Values')

        # # Display the plot in Streamlit
        # st.pyplot(fig)

    return predictions

# Check if the uploaded file has changed
if uploaded_file is not None:
    with col1:
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None

        if st.session_state.uploaded_file is not None:
            # Display image
            image = Image.open(st.session_state.uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True, width=400)


            with st.container():
                st.markdown('<div class="button-container">', unsafe_allow_html=True)

                
                with col2:
                    if st.session_state.predictions == None:
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown('')
                        st.markdown("""
                            <style>
                            .stButton>button {
                                background-color: #000000;  /* Green background */
                                color: white;  /* White text */
                                font-size: 18px;  /* Larger font size */
                                border: none;  /* No border */
                                border-radius: 12px;  /* Rounded corners */
                                padding: 16px 40px;  /* Padding inside the button */
                                cursor: pointer;  /* Pointer cursor on hover */
                                transition: background-color 0.3s ease;  /* Smooth transition on hover */
                            }
                            .stButton>button:hover {
                                background-color: #E3E3E3;  /* Darker green on hover */
                                color: black;  /* Keep text white when hovered */
                            }
                            </style>
                            """, unsafe_allow_html=True)
                        # Classification button 
                        if st.button("Classify"):
                            with st.spinner("Classifying... please wait"):
                                st.session_state.predictions = simulate_prediction()  # Save predictions to session state
                                st.rerun()

                    col3, col4 = st.columns([1, 1], gap="large")
                    # Display predictions if available
                    if st.session_state.predictions:
                        with col3:
                            st.markdown("### 📊 **Predictions for Each Attribute:**")
                            print(type(st.session_state.predictions.items()))
                            for attribute, value in list(st.session_state.predictions.items())[:6]:
                                st.markdown(f'##### {attribute}')
                                st.markdown(
                                    """
                                    <style>
                                    [data-baseweb="select"] {
                                        margin-top: -60px;
                                    }
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                options = [
                                    f"❌ {item[0]} ({(item[1]*100):.2f}%)"  # If 'INVALID', add cross emoji first
                                    if item[0] == 'INVALID'  # Check for 'INVALID' first
                                    else f"🟢 {item[0]} ({(item[1]*100):.2f}%)" if item[1] > 0.7  # High confidence
                                    else f"🟡 {item[0]} ({(item[1]*100):.2f}%)" if 0.5 <= item[1] <= 0.7  # Medium confidence
                                    else f"🔴 {item[0]} ({(item[1]*100):.2f}%)"  # Low confidence
                                    for item in value
                                ]
                                
                                st.selectbox('', options, key=attribute)

                        with col4:
                            print(type(st.session_state.predictions.items()))
                            st.markdown('')
                            st.markdown('')
                            st.markdown('')
                            st.markdown('')
                            st.markdown('')
                            st.markdown('')
                            for attribute, value in list(st.session_state.predictions.items())[6:]:
                                
                                st.markdown(f'##### {attribute}')
                                st.markdown(
                                    """
                                    <style>
                                    [data-baseweb="select"] {
                                        margin-top: -60px;
                                    }
                                    </style>
                                    """,
                                    unsafe_allow_html=True,
                                )

                                options = [
                                    f"❌ {item[0]} ({(item[1]*100):.2f}%)"  # If 'INVALID', add cross emoji first
                                    if item[0] == 'INVALID'  # Check for 'INVALID' first
                                    else f"🟢 {item[0]} ({(item[1]*100):.2f}%)" if item[1] > 0.7  # High confidence
                                    else f"🟡 {item[0]} ({(item[1]*100):.2f}%)" if 0.5 <= item[1] <= 0.7  # Medium confidence
                                    else f"🔴 {item[0]} ({(item[1]*100):.2f}%)"  # Low confidence
                                    for item in value
                                ]
                                
                                st.selectbox('', options, key=attribute)

                            if st.button("Save attributes"):
                                st.session_state.uploaded_file = None
                                st.session_state.predictions = None
                                st.session_state["uploader_key"] +=1
                                st.rerun()
    

else:
    st.write("Please upload an image.")

# Display results
st.markdown("""
    --- 
    ### 🔍 **About the Model:**
    The model used for this classification is a pre-trained **ResNet-50** model, which has been trained on a large dataset to classify images into one of 11 categories.
    The model provides predictions with high accuracy. You can test the model with different images for more insights.
""")





# if uploaded_file is not None:
#     # Display image and center it in the left column
#     with col1:
#         image = Image.open(uploaded_file)
#         # Center the image manually by adjusting the layout
#         st.image(image, caption="Uploaded Image", use_container_width=True, width=200)

#     # Add a classification button in the right column

#         # Add a classification button
#         with st.container():
#             st.markdown('<div class="button-container">', unsafe_allow_html=True)
#             if st.button("Classify"):
#                 with st.spinner("Classifying... please wait"):
#                     # Simulate processing time
#                     # time.sleep(2)

#                     # Simulate a prediction result
#                     st.session_state.predictions = simulate_prediction()
#                     with col2:
#                         # Display predictions for each attribute
#                         st.markdown("### 📊 **Predictions for Each Attribute:**")
#                         for attribute, value in st.session_state.predictions.items():
#                             option = st.selectbox(
#                                 f'{attribute}',
#                                 (value[0], value[1], value[2]),
#                             )
#                             # st.markdown(f"**{attribute.replace('_', ' ').title()}**: {value[0]}")
#                             # with st.expander(f'See next predictions'):
#                             #     st.write(f'{value[1]}')
#                             #     st.write(f'{value[2]}')

#                             # st.write("You selected:", option)
#                             # st.markdown(f"**{attribute.replace('_', ' ').title()}**: {value}")
#                             # st.divider()

    # # Display results
    # st.markdown("""
    #     --- 
    #     ### 🔍 **About the Model:**
    #     The model used for this classification is a pre-trained **ResNet** model, which has been trained on a large dataset to classify images into one of 11 categories.
    #     The model provides predictions with high accuracy. You can test the model with different images for more insights.
    # """)



# else:
#     st.write("Please upload an image to begin.")
