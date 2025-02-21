from transformers import pipeline, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model_id = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
EmotionallyResponsiveNN = AutoModelForCausalLM.from_pretrained(
    model_id = "SamLowe/roberta-base-go_emotions",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

sentences = ["I am not having a great day"]

model_outputs = classifier(sentences)
