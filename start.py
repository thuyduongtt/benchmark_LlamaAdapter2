import argparse

import torch
from PIL import Image
import llama
import cv2

from pipeline import run_pipeline_by_question

llama_model = None
MODEL_NAME = ''
LLAMA_TYPE = ''
LLAMA_DIR = 'llama_models/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    global llama_model
    llama_model = llama.load(MODEL_NAME, LLAMA_DIR, llama_type=LLAMA_TYPE, device=device)


def vqa_task(image, row_data):
    # return f'prediction, {image}, {row_data["question"]}'  # turn off model for pipeline testing

    if llama_model is None:
        load_model()

    model, preprocess = llama_model
    model.eval()

    img = Image.fromarray(cv2.imread(image))
    img = preprocess(img).unsqueeze(0).to(device)

    prompt = llama.format_prompt(f"Question: {row_data['question']} Answer:")
    return model.generate(img, [prompt])[0]


def test_model():
    from pathlib import Path

    print('===== TEST MODEL =====')
    img = 'test_img/eiffel.jpg'
    assert Path(img).exists(), f'No image in {img}'
    row_data = {
        'question': 'Please introduce this painting.'
    }
    r = vqa_task(img, row_data)
    print(f'{img}, {row_data["question"]}, {r}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_ds', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Path to output')
    parser.add_argument('--split', type=str, default='train', help='Set to "train" or "test"')
    parser.add_argument('--start_at', type=int, default=0, help='Index of the sample to start from')
    parser.add_argument('--limit', type=int, default=0, help='Max number of samples')
    parser.add_argument('--model_name', type=str, default='LORA-BIAS-7B-v21')
    parser.add_argument('--llama_type', type=str, default='7B')
    args = parser.parse_args()

    global MODEL_NAME, LLAMA_TYPE
    MODEL_NAME = args.model_name
    LLAMA_TYPE = args.llama_type

    test_model()

    # run_pipeline_by_question(vqa_task, args.path_to_ds, args.output_dir_name, limit=args.limit,
    #                          start_at=args.start_at, split=args.split)


if __name__ == '__main__':
    main()
