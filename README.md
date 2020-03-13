## Conv-TasNet
A PyTorch implementation of Conv-TasNet described in ["TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation"](https://arxiv.org/abs/1809.07454).

## Results
|   From  | Activatoin |  Norm | Causal | batch size |      #GPU      | SI-SDRi(dB) | SDRi(dB)|
|:-------:|:----------:|:-----:|:------:|:----------:|:--------------:|:-----------:|:-------:|
|**Paper**| **Softmax**|**gLN**| **No** |     -      |        -       |   **14.6**  | **15.0**|
| Mapping |     ReLU   |  gLN  |   No   |     20     |  4 Tesla V100  |   **16.1**  | **16.3**|
| Mapping+OSNR | ReLU  |  gLN  |   No   |     20     |  4 Tesla V100  |   **16.3**  | **16.6**|

 `SI-SDR` and `SI-SNR` are the same thing (different name) in different papers.

## Install
- PyTorch 0.4.1+
- Python3 (Recommend Anaconda)
- `pip install -r requirements.txt`

## Usage
```bash
scripts/run_tasnet.sh
```

## Acknowledgement

Thanks for Yi Luo's research of Conv-Tasnet and wangkenpu's implementation of the baseline! ["Conv-TasNet-PyTorch"](https://github.com/wangkenpu/Conv-TasNet-PyTorch).