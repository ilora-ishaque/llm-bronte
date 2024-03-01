from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

# Import the model
config = PeftConfig.from_pretrained("iloraishaque/llm-tolkien")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# Load the Lora model
model = PeftModel.from_pretrained(model, "iloraishaque/llm-tolkien")

prompt = "The hobbits were so suprised seeing their friend"

inputs = tokenizer(prompt, return_tensors="pt")
tokens = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=1,
    eos_token_id=tokenizer.eos_token_id,
    early_stopping=True
)

# The hobbits were so suprised seeing their friend again that they did not 
# speak. Aragorn looked at them, and then he turned to the others.</s>