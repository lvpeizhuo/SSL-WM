# SSL-WM
The implementation of _SSL-WM: A Black-Box Watermarking Approach for Encoders Pre-trained by Self-Supervised Learning.

## Getting started

### Train a clean model 

```bash
python train_simclr_clean.py
```

### Inject the watermark

```bash
python train_simclr.py
```

### Finetune the watermarked model to another task:

```bash
python train_downstream.py
```

### Test the accuracy

```bash
python test_simclr.py
```

### Test the MAD

```bash
python test_mad.py
```
