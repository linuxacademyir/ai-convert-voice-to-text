#!/usr/bin/env python3
import wave
import json
from vosk import Model, KaldiRecognizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Function to convert WAV audio to text using Vosk
def audio_to_text(audio_file, model_path):
    # Load the Vosk model
    model = Model(model_path)
    wf = wave.open(audio_file, "rb")
    
    # Initialize recognizer
    recognizer = KaldiRecognizer(model, wf.getframerate())

    # Process audio and get the transcribed text
    transcribed_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            transcribed_text += json.loads(result)["text"]

    final_result = recognizer.FinalResult()
    transcribed_text += json.loads(final_result)["text"]

    return transcribed_text

# Function to generate text from GPT-2 model (with proper padding handling)
def generate_from_model(input_text):
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Use eos_token as pad_token
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token
    model.config.pad_token_id = model.config.eos_token_id  # Update model config

    # Tokenize the input text and set padding/truncation as necessary
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Generate response from the model, passing attention_mask
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],  # Pass the attention mask here
        max_length=100,  # Limit the output length
        num_return_sequences=1,
    #    temperature=0.7,  # Control randomness of the output
    )

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Main function to process the WAV file, transcribe, and generate a response
def main(audio_file, model_path):
    # Step 1: Convert audio to text using Vosk
    transcribed_text = audio_to_text(audio_file, model_path)
    print("Transcribed Text:", transcribed_text)

    # Step 2: Pass the transcribed text to the language model (GPT-2 or LLaMA)
    response = generate_from_model(transcribed_text)
    print("Response from model:", response)

if __name__ == "__main__":
    # Path to your WAV audio file and Vosk model directory
    audio_file = "linux.wav"  # Replace with your audio file path
    model_path = "vosk-model-small-en-us-0.15"  # Replace with the Vosk model path

#     audio_file = "w.wav"
#     model_path = "vosk-model-small-en-us-0.15"
    # Call the main function
    main(audio_file, model_path)

