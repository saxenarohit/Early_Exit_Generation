# Early Exit Generation

This repository provides an implementation for performing early exit decoding with large language models. Early exit allows generating text using the outputs of an intermediate layer, rather than the final layer, of the model.

## Description

This example demonstrates how to use early exit decoding with the **LLaMA 3.1 8B Instruct** model. The implementation wraps the original model to enable early exit at a specified layer and generates text based on the intermediate outputs.

## Features
- Supports **early exit** at any specified layer index.
- Compatible with Hugging Face's `transformers` library.

## Usage

To run early exit decoding and generate text for a specific layer, use the following command:

```bash
python early_exit_example.py --layer 32
