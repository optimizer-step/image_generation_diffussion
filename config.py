"""A few values we can change without editing the model."""
IMAGE_SIZE = 64             # Small enough to train on an 8 GB GPU.
BATCH_SIZE = 24             # Lower to 12 if the GPU runs out of memory.
EPOCHS = 60                 # Start with 1 epoch as a rehearsal.
LEARNING_RATE = 0.0002     # How strongly the optimizer updates weights.
NOISE_STEPS = 300          # Number of steps from photo to static.
MODEL_WIDTH = 48           # Size of our mini U-Net.
SAMPLE_EVERY = 10          # Save generated examples every 10 epochs.
SEED = 42                 # Makes experiments easier to compare.
