import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import argparse


def main(model_name):
    # load model and the tokenizer
    print(f"Loading model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # load the model to the GPU
    device = torch.device("cuda")
    model.to(device)
    print(f"Model loaded to {device}")

    # create some prompt input data
    inputs = tokenizer("This is a test sentence.", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # loop to run inference
    while True:
        print("Running inference...")
        with torch.no_grad():
            outputs = model(**inputs)

        # print the output logits
        print(outputs.logits)

        # sleep for 10 seconds≠
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on different models")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to load"
    )
    args = parser.parse_args()

    main(args.model_name)
