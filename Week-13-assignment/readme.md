### 📦 Setup: Download Dependencies

This project requires `deep_sort` and its model weights. To download:

```bash
# Clone deep_sort (if not already included as a submodule)
git clone https://github.com/nwojke/deep_sort.git Week-13-assignment/deep_sort

# Download Deep SORT model
mkdir -p Week-13-assignment/model_data
wget -O Week-13-assignment/model_data/mars-small128.pb \
    https://drive.google.com/file/d/1bB66hP9voDXuoBoaCcKYY7a8IYzMMs4P/view?usp=drive_link
