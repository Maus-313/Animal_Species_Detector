import os

# Define the base directory
base_dir = "animal_dataset/six_species"

# Initialize lists
image_paths = []
labels = []

# Map species to labels (0 to 5)
species_to_label = {
    "Elephant": 0,
    "Gorilla": 1,
    "Hippo": 2,
    "Monkey": 3,
    "Tiger": 4,
    "Zebra": 5
}

# Loop through each species folder
for species, label in species_to_label.items():
    species_dir = os.path.join(base_dir, species)
    for img_file in os.listdir(species_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):  # Adjust file extensions as needed
            image_paths.append(os.path.join(species_dir, img_file))
            labels.append(label)

# Verify the total number of images and labels
print(f"Total images: {len(image_paths)}")  # Should be around 12000
print(f"Total labels: {len(labels)}")       # Should be around 12000
print("Label distribution:", {k: labels.count(v) for k, v in species_to_label.items()})