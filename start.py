import argparse

import torch
from PIL import Image
import cv2
import llama

from pipeline import run_pipeline_by_question, run_pipeline_by_image

llama_model = None
MODEL_NAME = ''
LLAMA_TYPE = ''
LLAMA_DIR = '../Llama/llama-2-7b'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    global llama_model
    llama_model = llama.load(MODEL_NAME, LLAMA_DIR, llama_type=LLAMA_TYPE, device=device)


def vqa_task(image, row_data):
    # return f'prediction, {image}, {row_data["question"]}'  # turn off model for pipeline testing

    if llama_model is None:
        load_model()
    model, preprocess = llama_model

    img = Image.fromarray(cv2.imread(image))
    img = preprocess(img).unsqueeze(0).to(device)

    prompt = llama.format_prompt(row_data['question'])
    return model.generate(img, [prompt])[0]


def image_captioning_task(image):
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ds', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Path to output')
    parser.add_argument('--split', type=str, default='train', help='Set to "train" or "test"')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--model_name', type=str, default='LORA-BIAS-7B-v21')
    parser.add_argument('--llama_type', type=str, default='7B')
    parser.add_argument('--task', type=str, default='vqa', help='Task name: vqa, image_captioning')
    args = parser.parse_args()

    global MODEL_NAME, LLAMA_TYPE
    MODEL_NAME = args.model_name
    LLAMA_TYPE = args.llama_type

    if args.task == 'vqa':
        run_pipeline_by_question(vqa_task, args.path_to_ds, args.output_dir_name, limit=args.limit,
                                 start_at=args.start_at, split=args.split)
    elif args.task == 'image_captioning':
        run_pipeline_by_image(image_captioning_task, args.path_to_ds, args.output_dir_name, limit=args.limit,
                              start_at=args.start_at, split=args.split)

    else:
        print('Invalid task')


if __name__ == '__main__':
    main()
