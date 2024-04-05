# MSANet: Multi-Scale Attention Network for Vertical Seed Distribution in Soybean Breeding Fields

## Dataset
Please fill [this form](https://forms.gle/FJcGXXL8AB9J3Ajp7) to get download link for the datasets and pre-trained weights.

## Train
### Train the MSANet with our datasets:

1. Download the `data.zip` following the above Dataset section and unzip the `data/` to the root of this repo.
2. Check the `options.py`.
3. Train the MSANet by running.
```
python train.py --project_name your_project_name_here --data-root ./data/soypod-200-txt --output_dir ./runs --num_workers 0
```

### Train the MSANet with your own dataset:

First please put you data under "data/" for convenience. (of course you can put your data anywhere else but just remember to modify of provide the '--data_root' item.)

We originally support dataset annotated by V7 (https://www.v7labs.com/). For other formats of the annotation, please modify the `get_points` function in the `utils.py` which can read your json format annotation and return the list of [y, x] coordination of point annotations.

Then, train the MSANet by running:
```
python train.py --project_name your_project_name_here --data-root /path/to/your/dataset --output_dir ./runs --num_workers 0
```

## Evaluation
### Statistical Results
We provide evaluation metric for both counting and localization tasks to evaluate our MSANet statistically.

- For couting tasks, we evaluated R2, MAE and RMSE.
- For localization tasks, we evaluated the MED, which is defined in the paper, Precision, Recall and F1 score.

| Dataset | R2 | MAE | RMSE | MED | Precision | Recall | F1 score |
| --      | -- | --- | ---- | --- | --------- | ------ | -------- |
| 2021 Dataset | 0.94 | 9.20 | 13.16 | 7.52 | 0.87 | 0.85 | 0.86 |
| 2021 Enlarged Dataset | 0.86 | 13.69 | 18.32 | 8.08 | 0.81 | 0.87 | 0.84 |
| 2022 Dataset | 0.82 | 13.66 | 17.26 | 4.99 | 0.91 | 0.85 | 0.88 |

We also provide the comparision results with P2PNet-Soy (Jiangsan et al., 2023), please check the jupyter notebooks located under `/evaluations/`.


## Inference
To inference the model, here's an introduction [here on Colab](https://colab.research.google.com/drive/1idp0hIjD1JUTOiukibZl9BQ3GsQy1oND?usp=sharing).
