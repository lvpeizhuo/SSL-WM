correct all paths to your paths.

# 1. Train Clean Encoder
python train_byol_clean.py ==> byol-encoder-clean.pth

# 2. Load Clean Encoder, Continue to Train for Poisoning It
python train_byol_poison.py ==> byol-encoder-poison.pth

# 3. Based on Poisoned Encoder to Train Downstream Model
python train_byol_downstream.py ==> byol-downstream.pth

# 4. Test MAD of the Downstream Model
python test_mad.py

# 5. Finetune Attack
python finetune.py

# 5. Pruning Attack
python prune.py

















