import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
"""
Purpose:
Provides the terminal-based interface for interacting with the chatbot.
with RAG 
"""
# Load the model and tokenizer
model_name = "FreedomIntelligence/Apollo-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Simple in-memory retriever
class SimpleRetriever:
    def __init__(self):
        self.documents = [
            {"text": "Post-surgery pain in the knee can be due to inflammation or healing tissues."},
            {"text": "It's important to follow your doctor's advice on pain management."},
            {"text": "Rest, ice, compression, and elevation (RICE) can help reduce pain and swelling."},
            {"text": "Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) that can help reduce inflammation and pain."},
            {"text": "Always consult your doctor before taking any medication, including ibuprofen."},
            {"text": "Keep the surgical area clean and dry to prevent infection."},
            {"text": "Avoid strenuous activities and heavy lifting until your doctor gives you the go-ahead."},
            {"text": "Attend all follow-up appointments with your healthcare provider."},
            {"text": "Report any unusual symptoms, such as excessive bleeding or severe pain, to your doctor immediately."},
            {"text": "Eat a balanced diet to support healing and recovery."},
            {"text": "Stay hydrated by drinking plenty of water."},
            {"text": "Take prescribed medications as directed by your healthcare provider."},
            {"text": "Avoid smoking and alcohol consumption during the recovery period."},
            {"text": "Engage in light physical activity, such as walking, to promote circulation and prevent blood clots."},
            {"text": "Use assistive devices, such as crutches or a walker, if recommended by your doctor."},
            {"text": "Follow any specific post-operative instructions provided by your surgeon."},
            {"text": "Keep the surgical wound covered with a sterile dressing as instructed by your healthcare provider."},
            {"text": "Monitor the surgical site for signs of infection, such as redness, swelling, or discharge."},
            {"text": "Practice good hygiene, including regular handwashing, to reduce the risk of infection."},
            {"text": "Avoid soaking the surgical area in water until it is fully healed."},
            {"text": "Gradually increase your activity level as advised by your healthcare provider."},
            {"text": "Use ice packs to reduce swelling and discomfort in the surgical area."},
            {"text": "Elevate the affected limb to help reduce swelling and promote healing."},
            {"text": "Wear compression garments if recommended by your doctor to support healing and reduce swelling."},
            {"text": "Avoid driving until you have been cleared by your healthcare provider."},
            {"text": "Follow a physical therapy program if prescribed by your doctor to aid in recovery."},
            {"text": "Keep a record of your symptoms and progress to share with your healthcare provider during follow-up visits."},
            {"text": "Ask for help from family or friends with daily activities during the initial recovery period."},
            {"text": "Stay positive and patient, as recovery times can vary depending on the type of surgery and individual factors."},
        ]

    def get_relevant_documents(self, query):
        # Simple keyword matching for demonstration purposes
        return [doc for doc in self.documents if any(word in doc['text'] for word in query.split())]

retriever = SimpleRetriever()

def chat_with_bot(user_input):
    # Retrieve context using the retriever
    context_docs = retriever.get_relevant_documents(user_input)
    context = " ".join([doc['text'] for doc in context_docs])
    
    # Combine context with user input for LLM
    prompt = f"Context: {context}\n\nUser: {user_input}\nBot:"
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
    outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Terminal-based chatbot interface
if __name__ == "__main__":
    print("Welcome to the Medical Chatbot. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = chat_with_bot(user_input)
        print(f"Bot: {response}")