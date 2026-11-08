
import os

# Dataset paths
train_path = "dataset/train"
test_path = "dataset/test"

emotions = os.listdir(train_path)

print("EMOTION DATASET SUMMARY")
print("-" * 40)

total_train = 0
total_test = 0

for emotion in sorted(emotions):
    train_count = len(os.listdir(os.path.join(train_path, emotion)))
    test_count = len(os.listdir(os.path.join(test_path, emotion)))

    total_train += train_count
    total_test += test_count

    print(f"{emotion:12} | Train: {train_count:5} | Test: {test_count:5}")

print("-" * 40)
print("Total training images:", total_train)
print("Total testing images:", total_test)
print("Total images:", total_train + total_test)
