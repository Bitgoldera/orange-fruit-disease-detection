from pathlib import Path
import shutil
import random


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "orange_5_class_clean"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

CLASSES = [
    "Healthy",
    "Canker",
    "Black_Spot",
    "Greening",
    "Melanose",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}

RANDOM_SEED = 42


def collect_images():

    data = {}

    for class_name in CLASSES:

        class_dir = SOURCE_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        images = [
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(images) < 3:
            raise ValueError(
                f"Not enough images in {class_name}"
            )

        data[class_name] = images

    return data


def split_class(images):

    random.shuffle(images)

    n = len(images)

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train = images[:n_train]
    val = images[n_train:n_train + n_val]
    test = images[n_train + n_val:]

    return train, val, test


def copy_images(images, class_name, split_name):

    output_class = (
        OUTPUT_DIR
        / split_name
        / class_name
    )

    output_class.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, image in enumerate(images, start=1):

        destination = (
            output_class
            / f"{class_name.lower()}_{i:05d}"
            f"{image.suffix.lower()}"
        )

        shutil.copy2(image, destination)


def main():

    random.seed(RANDOM_SEED)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE_DIR}"
        )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    data = collect_images()

    print("\n===================================")
    print("DATASET SPLIT")
    print("===================================")

    total_train = 0
    total_val = 0
    total_test = 0

    for class_name in CLASSES:

        train, val, test = split_class(
            data[class_name]
        )

        copy_images(
            train,
            class_name,
            "train"
        )

        copy_images(
            val,
            class_name,
            "val"
        )

        copy_images(
            test,
            class_name,
            "test"
        )

        total_train += len(train)
        total_val += len(val)
        total_test += len(test)

        print(
            f"{class_name:12s} "
            f"Train: {len(train):3d} | "
            f"Val: {len(val):2d} | "
            f"Test: {len(test):2d}"
        )

    print("-----------------------------------")
    print(f"Total Train: {total_train}")
    print(f"Total Val  : {total_val}")
    print(f"Total Test : {total_test}")
    print("-----------------------------------")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
