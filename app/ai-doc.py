from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
# Load model directly
from transformers import AutoModel

model_name = "dmis-lab/biobert-base-cased-v1.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(user_input):
    # Add context to the input

    inputs = tokenizer(user_input, return_tensors='pt')
    # Adjust model parameters
    outputs = model.generate(**inputs, max_length=1000,do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return response

if __name__ == "__main__":
    user_input = input("Enter your message: ")
    response = generate_response(user_input)
    print("Model response:", response)