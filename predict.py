import re
import random
import time
import argparse
from statistics import mode

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import zipfile
import os
import notify

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower()
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)
    text = re.sub(r'\b(a|an|the)\b', '', text)
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)
    text = re.sub(r"[^\w\s':]", ' ', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text):
    if isinstance(text, tuple):
        text = text[0]  # タプルの最初の要素を使用
    text = re.sub(r"[\(\)\"\',]", '', text)
    return text

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, image_dir, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.answer = answer

    def __getitem__(self, idx):
        image_path = f"{self.image_dir}/{self.df['image'].iloc[idx]}"
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        question_text = self.df["question"].iloc[idx]
        if self.answer:
            answer_text = self.df["answer"].iloc[idx]
            return image, question_text, answer_text
        else:
            return image, question_text, "Noanswer"

    def __len__(self):
        return len(self.df)

def create_zip(zip_file_name, path):
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.isfile(path):
            zipf.write(path, os.path.basename(path))
        else:
            print("file not exist")

def run_prediction(prompt_file, save_file, load_in_4bit, model_path):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(csv_path='./data/extracted_train.csv', image_dir='./data/processed_train', transform=transform, answer=True)
    test_dataset = VQADataset(csv_path='./data/processed_valid.csv', image_dir='./data/sharped_valid', transform=transform, answer=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Total number of samples in test_loader: {len(test_loader.dataset)}")

    if load_in_4bit:
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=['out_proj', 'kv_proj', 'lm_head'],
        )
    else:
        q_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=['out_proj', 'kv_proj', 'lm_head'],
        )

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, quantization_config=q_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    submission = []
    counter = 0
    total = len(test_loader.dataset)
    with open(prompt_file) as f:
        prompt_format = f.read()
        f.close()

    for image, question_text, answers in test_loader:
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray((image * 255).astype(np.uint8))
        question = prompt_format.format(question_text=question_text)
        msgs = [{'role': 'user', 'content': question}]
        
        response = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.5,
        )
        counter += 1
        progress = (counter / total) * 100
        res = re.sub(r'(<box>.*</box>)', '', response)
        res = res.replace('<ref>', '')
        res = res.replace('</ref>', '')
        res = res.replace('<box>', '')
        text = res.replace('</box>', '')
        match = re.search(r'"answer":\s*"([^"]*)"', text)
        if match:
            predict = match.group(1)
        else:
            predict = "unanswerable"
        predict = process_text(predict)
        submission.append(predict)
        if counter % 20 == 1:
            print(f"# Progress {progress} % has finished")

    submission = np.array(submission)
    np.save(save_file, submission)
    create_zip("submission.zip", save_file)
    notify.send_email("Notify from Ubuntu in Hlab", "The prediction and compress has just accomplished", attachment_path="./submission.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main function with specified parameters.")
    parser.add_argument("--prompt_file", type=str, default="Prompts/answer_confidence_no_unanswerable.txt", help="The prompt file to be used.")
    parser.add_argument("--save_file", type=str, default="submission_sharpedfp16_noexample.npy", help="The .npy file to save the results.")
    parser.add_argument("--load_in_4bit", action='store_true', default=False, help="Load the model in 4-bit mode if set, otherwise load in 8-bit mode.")
    parser.add_argument("--model_path", type=str, default="./models/mini_cpm_raw", help="The path to the model to be loaded.")
    args = parser.parse_args()

    run_prediction(args.prompt_file, args.save_file, args.load_in_8bit, args.model_path)
