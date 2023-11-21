correct all paths to your paths.

# 1. Train Clean Encoder
python train_moco_clean.py ==> moco-k-clean.pth / moco-q-clean.pth / moco-memory-queue-clean.pth

# 2. Load Clean Encoder, Continue to Train for Poisoning It
python train_moco_poison.py ==> moco-k-poison.pth / moco-q-poison.pth / moco-memory-queue-poison.pth

# 3. Based on Poisoned Encoder to Train Downstream Model
python train_moco_downstream.py ==> moco-downstream.pth

# 4. Test MAD of the Downstream Model
python test_mad.py

# 5. Finetune Attack
python finetune.py

# 5. Pruning Attack
python prune.py

















