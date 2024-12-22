import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Load BioBERT model and tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

# Define a context for the chatbot
context = (
    "BioBERT is a biomedical language representation model designed for various biomedical text mining tasks. "
    "It has been pre-trained on large-scale biomedical corpora such as PubMed articles. "
    "You can use it for tasks like question answering, named entity recognition, and relation extraction."
)

def generate_response(question):
    try:
        # Use BioBERT to answer the question
        result = qa_pipeline(question=question, context=context)
        return result["answer"] if result["answer"].strip() else "I didn't quite get that. Could you ask in another way?"
    except Exception as e:
        return "I'm not sure how to answer that. Try asking something else."

def chat():
    print("BioBERT Chatbot: Hello! Ask me anything about BioBERT or biomedical topics. Type 'exit' to quit.")
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Exit condition
        if user_input.lower() == "exit":
            print("BioBERT Chatbot: Goodbye!")
            break
        
        # Check if the input is a valid question
        if not user_input.endswith("?"):
            print("BioBERT Chatbot: I respond better to questions. Try asking something specific.")
            continue
        
        # Generate response
        response = generate_response(user_input)
        print(f"BioBERT Chatbot: {response}")

if __name__ == "__main__":
    chat()
