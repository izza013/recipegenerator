import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer from Hugging Face
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "Izza-shahzad-13/recipe-generator"  # Replace with your Hugging Face model name
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Function to generate recipe instructions
def generate_recipe(ingredients):
    input_ids = tokenizer.encode(
        ingredients,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    outputs = model.generate(
        input_ids=input_ids,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    recipe_instructions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recipe_instructions

# Function to split recipe into steps
def format_recipe_as_steps(recipe):
    steps = recipe.split(". ")  # Split by sentence-ending punctuation
    steps = [step.strip() for step in steps if step]  # Clean up and remove empty entries
    return steps

# Streamlit App Layout
st.title("AI-Based Recipe Generator")


# Input box for ingredients
ingredients = st.text_area("Enter ingredients (comma-separated):", placeholder="e.g., 1 cup rice, 1 cup water, 1/2 teaspoon salt")

if st.button("Generate Recipe"):
    if ingredients.strip():
        st.write("### Recipe Instructions:")
        with st.spinner("Generating..."):
            recipe = generate_recipe(ingredients)
            steps = format_recipe_as_steps(recipe)
        st.success("Recipe generated successfully!")
        
        # Display the recipe as steps
        for i, step in enumerate(steps, 1):
            st.write(f"**Step {i}:** {step}")
    else:
        st.error("Please enter some ingredients!")
