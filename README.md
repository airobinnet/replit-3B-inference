# Replit Code Instruct inference using CPU with UI

Run inference on the [replit code instruct model](https://huggingface.co/abacaj/Replit-v2-CodeInstruct-3B-ggml) using your CPU. This inference code uses a [ggml](https://github.com/ggerganov/ggml) quantized model. To run the model we'll use a library called [ctransformers](https://github.com/marella/ctransformers) that has bindings to ggml in python.

Demo:


[Inference Demo](https://github.com/airobinnet/replit-3B-inference/assets/126980386/c3c24080-e237-47e7-b228-0cc309a560e4)

## Requirements

Using docker should make all of this easier for you. Minimum specs, system with 8GB of ram. Recommend to use `python 3.10`.

## Tested working on

Will post some numbers for these two later.

- AMD Epyc 7003 series CPU
- AMD Ryzen 5950x CPU

## Setup

First create a venv.

```sh
python -m venv env && source env/bin/activate
```

Next install dependencies.

```sh
pip install -r requirements.txt
```

Next download the quantized model weights (about 1.5GB).

```sh
python download_model.py
```

Run the python script.

```sh
python inference_with_ui.py
```

Visit the url

[http://localhost:5000/](http://localhost:5000/)

