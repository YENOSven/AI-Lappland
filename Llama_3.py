from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class ConversationHistory:
    def __init__(self, max_turns=5):
        self.history = []
        self.max_turns = max_turns

    def add_turn(self, user_text, assistant_response):
        self.history.append(f"User: {user_text}\nAssistant: {assistant_response}")
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self):
        return "\n".join(self.history)

    def reset(self):
        self.history = []

class LlamaChatbot:
    def __init__(self, model_path, max_history_turns=5):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Define pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Initialize conversation history
        self.history_tracker = ConversationHistory(max_turns=max_history_turns)
    
    def generate_response(self, user_input):
        # Add user input to history context
        context = self.history_tracker.get_context()
        full_input = f"{context}\nUser: {user_input}\nAssistant: "
        
        # Tokenize input
        inputs = self.tokenizer(full_input, return_tensors="pt", padding=True, truncation=True)
        
        # Generate output
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.3
        )
        
        # Decode output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = response.split("Assistant:")[-1].strip()
        assistant_response = re.sub(r'^\d+[\).\s]', '', assistant_response).strip()  # Remove numbering if present
        
        # Update history with new turn
        self.history_tracker.add_turn(user_input, assistant_response)
        
        return assistant_response

    def reset_conversation(self):
        """Resets the conversation history."""
        self.history_tracker.reset()
