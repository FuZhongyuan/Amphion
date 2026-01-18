
# Tacotron2 Recipe

In this recipe, we will show how to train [Tacotron2](https://arxiv.org/abs/1712.05884) using Amphion's infrastructure. Tacotron2 is an autoregressive TTS model that directly synthesizes mel spectrograms from text using a sequence-to-sequence architecture with attention.

There are four stages in total:

1. Data preparation
2. Features extraction
3. Training
4. Inference

> **NOTE:** You need to run every command of this recipe in the `Amphion` root path:
> ```bash
> cd Amphion
> ```

## 1. Data Preparation

### Dataset Download
You can use the commonly used TTS dataset to train TTS model, e.g., LJSpeech, VCTK, LibriTTS, etc. We strongly recommend you use LJSpeech to train TTS model for the first time. How to download dataset is detailed [here](../../datasets/README.md).

### Configuration

After downloading the dataset, you can set the dataset paths in  `exp_config.json`. Note that you can change the `dataset` list to use your preferred datasets.

```json
    "dataset": [
        "LJSpeech",
    ],
    "dataset_path": {
        // TODO: Fill in your dataset path
        "LJSpeech": "[LJSpeech dataset path]",
    },
```

## 2. Features Extraction

### Configuration

Specify the `processed_dir` and the `log_dir` and for saving the processed data and the checkpoints in `exp_config.json`:

```json
    // TODO: Fill in the output log path
    "log_dir": "ckpts/tts",
    "preprocess": {
        // TODO: Fill in the output data path
        "processed_dir": "data",
        ...
    },
```

### Run

Run the `run.sh` as the preprocess stage (set  `--stage 1`):

```bash
sh egs/tts/Tacotron2/run.sh --stage 1
```

## 3. Training

### Configuration

We provide the default hyperparameters in the `exp_config.json`. They can work on single NVIDIA-24g GPU. You can adjust them based on your GPU machines.

```json
"train": {
    "batch_size": 32,
}
```

### Run

Run the `run.sh` as the training stage (set  `--stage 2`). Specify an experimental name to run the following command. The tensorboard logs and checkpoints will be saved in `ckpts/tts/[YourExptName]`.

```bash
sh egs/tts/Tacotron2/run.sh --stage 2 --name [YourExptName]
```

> **NOTE:** The `CUDA_VISIBLE_DEVICES` is set as `"0"` in default. You can change it when running `run.sh` by specifying such as `--gpu "0,1,2,3"`.


## 4. Inference

### Configuration

For inference, you need to specify the following configurations when running `run.sh`:


| Parameters                                          | Description                                                                                                                                                       | Example                                                                                                                                                                                                |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--infer_expt_dir`                              | The experimental directory which contains `checkpoint`                                                                                                            | `ckpts/tts/[YourExptName]`                                                                                                                                              |
| `--infer_output_dir`                                | The output directory to save inferred audios.                                                                                                                     | `ckpts/tts/[YourExptName]/result`                                                                                                                                       |
| `--infer_mode`                            | The inference mode, e.g., "`single`", "`batch`".  | "`single`" to generate a clip of speech, "`batch`" to generate a batch of speech at a time.                                     |
| `--infer_dataset`                            | The dataset used for inference.  |  For LJSpeech dataset, the inference dataset would be `LJSpeech`.                                                                                                                                    |
| `--infer_testing_set`                             | The subset of the inference dataset used for inference, e.g., train, test | For LJSpeech dataset, the testing set would be  "`test`" split from LJSpeech at the feature extraction.                                                                                                                                    |
| `--infer_text`                            | The text to be synthesized. | "`This is a clip of generated speech with the given text from a TTS model.`"                                                                                                                                    |
| `--vocoder_dir`                           | The directory for the vocoder. | "`ckpts/vocoder/hifigan_ljspeech`"                                                                                                                                    |


### Run
For example, if you want to generate speech of all testing set split from LJSpeech, just run:

```bash
sh egs/tts/Tacotron2/run.sh --stage 3 \
    --infer_expt_dir ckpts/tts/[YourExptName] \
    --infer_output_dir ckpts/tts/[YourExptName]/result \
    --infer_mode "batch" \
    --infer_dataset "LJSpeech" \
    --infer_testing_set "test" \
    --vocoder_dir ckpts/vocoder/hifigan_ljspeech/checkpoints
```

Or, if you want to generate a single clip of speech from a given text, just run:

```bash
sh egs/tts/Tacotron2/run.sh --stage 3 \
    --infer_expt_dir ckpts/tts/[YourExptName] \
    --infer_output_dir ckpts/tts/[YourExptName]/result \
    --infer_mode "single" \
    --infer_text "This is a clip of generated speech with the given text from a TTS model." \
    --vocoder_dir ckpts/vocoder/hifigan_ljspeech
```

## Model Architecture

Tacotron2 consists of the following components:

1. **Encoder**: Character/phoneme embedding followed by 3 convolutional layers and a bidirectional LSTM
2. **Attention**: Location-sensitive attention mechanism
3. **Decoder**: Autoregressive decoder with prenet, attention RNN, decoder RNN, and linear projection
4. **Postnet**: 5 convolutional layers to refine the mel spectrogram

### Key Features

- **Autoregressive generation**: The model generates mel spectrograms frame by frame
- **Location-sensitive attention**: Helps maintain monotonic alignment between text and audio
- **Teacher forcing**: Uses ground truth mel frames during training
- **Stop token prediction**: Predicts when to stop generation during inference


## References

```bibtex
@inproceedings{shen2018natural,
  title={Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions},
  author={Shen, Jonathan and Pang, Ruoming and Weiss, Ron J and Schuster, Mike and Jaitly, Navdeep and Yang, Zongheng and Chen, Zhifeng and Zhang, Yu and Wang, Yuxuan and Skerrv-Ryan, Rj and others},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4779--4783},
  year={2018},
  organization={IEEE}
}
```
