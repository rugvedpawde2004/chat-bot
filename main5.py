import os
import json
import streamlit as st
import pandas as pd

# Load configuration from config.json
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config4.json"))

# Read the configuration values from the json file
model_name = config_data["model_name"]
custom_name = config_data.get("custom_name", "Qwen")

# Load the Qwen model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Optionally load the fine-tuned model weights to combine pre-trained and fine-tuned knowledge
try:
    model.load_state_dict(torch.load('trained_model_full.pth'), strict=False)
    print("Fine-tuned model weights successfully loaded.")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}")

model.eval()  # Set the model to evaluation mode

# Load CSV for contact details
data = pd.read_csv('contact.csv')

# Function to fetch contact details based on user input
data.columns = data.columns.str.strip()

# Function to fetch contact details based on user input
def get_contact_details(user_input):
    user_input = user_input.lower().strip()  # Convert to lowercase and remove extra spaces
    
    # Check for person name and query for details
    if 'krishna' in user_input:
        if 'phone' in user_input:
            result = data[data['Name'].str.lower().str.strip() == 'krishna']['Phone']
            if not result.empty:
                return result.values[0]
            else:
                return "Sorry, Krishna's phone number is not available."
        elif 'address' in user_input:
            result = data[data['Name'].str.lower().str.strip() == 'krishna']['Address']
            if not result.empty:
                return result.values[0]
            else:
                return "Sorry, Krishna's address is not available."
        elif 'college' in user_input:
            result = data[data['Name'].str.lower().str.strip() == 'krishna']['College']
            if not result.empty:
                return result.values[0]
            else:
                return "Sorry, Krishna's college information is not available."
    elif 'rugved' in user_input:
        if 'phone' in user_input:
            result = data[data['Name'].str.lower().str.strip() == 'rugved']['Phone']
            if not result.empty:
                return result.values[0]
            else:
                return "Sorry, Rugved's phone number is not available."
        elif 'address' in user_input:
            result = data[data['Name'].str.lower().str.strip() == 'rugved']['Address']
            if not result.empty:
                return result.values[0]
            else:
                return "Sorry, Rugved's address is not available."
        elif 'college' in user_input:
            result = data[data['Name'].str.lower().str.strip() == 'rugved']['College']
            if not result.empty:
                return result.values[0]
            else:
                return "Sorry, Rugved's college information is not available."
    return "Sorry, I couldn't find the information you're asking for."


# Configuring Streamlit page settings
# Configuring Streamlit page settings with light mode
st.set_page_config(
    page_title="ChatChamps-ChatBot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for dynamic background and chat bubble styling

# Streamlit page title and header
st.markdown("<h1 style='text-align: center;'>🤖 ChatChamps-ChatBot</h1>", unsafe_allow_html=True)
st.image("src/tiger.png",use_container_width=True)

st.markdown(
    """
    <style>
    /* Apply gradient background globally */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: black;
    }

    /* Fix navbar, footer, and input sections */
    header, footer, [data-testid="stToolbar"], [data-testid="stHeader"], [data-testid="stFooter"], [data-testid="stDecoration"] {
        background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        border: none;
    }

    /* Fix the input area background */
    [data-testid="stChatInput"] {
        background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        border-radius: 10px;
        border: none;
    }

    /* Chat bubble styles */
    .chat-bubble {
        padding: 8px;
        margin: 5px;
        border-radius: 10px;
        font-size: 14px;
    }
    .user-bubble {
        background-color: #D1E8FF;
        text-align: right;
        color: #333;
    }
    .assistant-bubble {
        background-color: #F7F7F7;
        text-align: left;
        color: #333;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    /* Image resizing */
    [data-testid="stImage"] img {
        width: 10px;  /* Set your desired width */
        height: 380px;  /* Maintain aspect ratio */
        position: fixed;
    }
    

    </style>
    """,
    unsafe_allow_html=True
)

# Initialize chat session in Streamlit if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Predefined responses
predefined_responses = {
    "What is Team ChatChamps?": "It is a group of 4 techies, Krishna, Yogeshwar, Rugved, and Omkar."
}

st.divider()

# Display chat history with styled chat bubbles
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class='chat-bubble user-bubble'>
                {message['content']}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class='chat-bubble assistant-bubble'>
                {message['content']}
            </div>
            """,
            unsafe_allow_html=True
        )

# Input field for user's message
user_prompt = st.chat_input("✍️ What's in your mind today?")

if user_prompt:
    # Add user's message to chat and display it
    st.markdown(
        f"""
        <div class='chat-bubble user-bubble'>
            {user_prompt}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Check for predefined responses
    if user_prompt in predefined_responses:
        assistant_response = predefined_responses[user_prompt]
    else:
         # Check if the user prompt matches the contact query
        assistant_response = get_contact_details(user_prompt)
        if assistant_response == "Sorry, I couldn't find the information you're asking for.":
            # If not found, proceed with Qwen model response

    # Prepare messages for Qwen
            messages = [
         {"role": "system", "content": f"You are {custom_name}, created by Team ChatChamps. You are a helpful assistant."}, 
         {"role": "user", "content": user_prompt}
    ]

    # Format input using the Qwen tokenizer
            text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate a response using Qwen
            generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
            generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the response
            assistant_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Append assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display Qwen's response
    st.markdown(
        f"""
        <div class='chat-bubble assistant-bubble'>
            {assistant_response}
        </div>
        """,
        unsafe_allow_html=True
    )
