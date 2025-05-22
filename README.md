# Dataset
Download the dataset CASIA-B [here](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp) 

# Pre-Processing
After downloading and extracting the dataset, update the `root_folder` and `save_path` parameters in `pre_process.py` as follows:
```
create_gei_dataset(root_folder="GaitDatasetB-silh", save_path="GEI.pt")
``` 
Then run the preprocessing script:
```
python3 pre_process.py
```
This step may take approximately 1–2 hours. To save time, you can download the preprocessed data directly from [here](https://drive.google.com/file/d/13RHwB0Zyv0SaWn2d__6sKYSQ0t4UjEzh/view?usp=sharing)

# Configuration
You can modify training parameters in `common.py`

# Train

To train the model, either customize the arguments when initializing the `GaitData` class in `GaitData.py`, or use the default parameters defined in `common.py`.

execute `main.ipynb` to train model

# Evaluate

execute `eval.ipynb` to evaluate model

# visualize

execute `visualize.ipynb` to visualize results