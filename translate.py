from transformers import MarianMTModel, MarianTokenizer

class Translator:
    def __init__(self, model_dir=r"PATH_TO_Helsinki-NLP/opus-mt-en-jap"):
        """Initialize the model and tokenizer from the local path."""
        self.model = MarianMTModel.from_pretrained(model_dir)
        self.tokenizer = MarianTokenizer.from_pretrained(model_dir)

    def translate(self, text: str) -> str:
        """Translates English text to Japanese."""
        translated = self.model.generate(**self.tokenizer(text, return_tensors="pt", padding=True, truncation=True))
        translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text

