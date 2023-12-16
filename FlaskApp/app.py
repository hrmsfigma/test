from flask import Flask, render_template, request



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html', output_text=None)

@app.route('/compute', methods=['POST'])
def compute():
    input_text = request.form.get('input_text')

    def generate(prompt):
        tokenizer = AutoTokenizer.from_pretrained("C:/Users/e16ar/OneDrive/Desktop/gpt2Flask/models/distilgptdemo")
        model = AutoModelForCausalLM.from_pretrained("C:/Users/e16ar/OneDrive/Desktop/gpt2Flask/models/distilgptdemo")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for _ in range(1):
            input_ids = tokenizer.encode(prompt, return_tensors = "pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length = 512,
                    pad_token_id = tokenizer.eos_token_id,
                    num_return_sequences = 1,
                    temperature = 0.7,
                )
        generated_text = tokenizer.decode(output[0],skip_special_tokens=True)
        return generated_text


    output_text = generate(input_text)

    return render_template('home.html', output_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)
