## ProtoEFNet: Dynamic Prototype Learning for Inherently Interpretable Ejection Fraction Estimation in Echocardiography

This is the official repository of the paper ProtoEFNet: Dynamic Prototype Learning for Inherently Interpretable Ejection Fraction Estimation in Echocardiography accepted to IMIMIC Workshop MICCAI 2025.
[arXiv](https://pages.github.com/)

## Abstract

Ejection fraction (EF) is a crucial metric for assessing cardiac function and diagnosing conditions such as heart failure. Traditionally, EF estimation requires manual tracing and domain expertise, making the process time-consuming and subject to inter-observer variability. Most current deep learning methods for EF prediction are black-box models with limited transparency, which reduces clinical trust. Some post-hoc explainability methods have been proposed to interpret the decision-making process after the prediction is made. However, these explanations do not guide the model’s internal reasoning and therefore offer limited reliability in clinical applications. To address this, we introduce ProtoEFNet, a novel video-based prototype-learning model for continuous EF regression. The model learns dynamic spatio-temporal prototypes that capture clinically meaningful cardiac motion patterns. Additionally, the proposed Prototype Angular Separation (PAS) loss enforces discriminative representations across the continuous EF spectrum. Our experiments on the Echonet-Dynamic dataset show that ProtoEFNet can achieve accuracy on par with its non-interpretable counterpart while providing clinically relevant insight. The ablation study shows the proposed loss boosts the performance with a 2% increase in F1 score from 77.67 ± 2.68 to 79.64 ± 2.10.

![Figure 1](/figures/fig1.png)

## Table of Contents

1.  [Installation](#installation)
2.  [Train and Evaluation](#train_and_evaluation)
3.  [Dataset](#dataset)
4.  [Acknowledgement](#acknowledgement)
5.  [Citations](#citations)

## Installation

Python library dependencies can be installed using:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install pandas wandb tqdm seaborn torch-summary opencv-python jupyter jupyterlab tensorboard tensorboardX imageio array2gif moviepy tensorboard scikit-image sklearn scikit-learn termplotlib
pip install -e .
<< Test :
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```
The full list of python packages can be seen in `requirements.txt`.

## Train and Evaluation
The values of the hyperparameters can be found in `src/configs/Insightrnet.yml`. Run the script `train.sh` to train the model. The values of the parameters can be changed as an argument in `main.py`.

Run the script below to evaluate the model on test or validation subset:
```bash
python ~/AorticStenosis.XAI_AS/main.py --config_path=CONFIG_YML --run_name=eval --save_dir="/results" \       --eval_only=True --eval_data_type=DATA_TYPE \
        --model.checkpoint_path=CHECKPOINT_PATH \
        --data.test_type=TEST_TYPE --wandb_mode="online"
```

* `CONFIG_YML` is the path to the config file with the hyperparameters that the checkpoint is trained with.
* `CHECKPOINT_PATH` is the path to the model checkpoint.
* `DATA_TYPE` is either "VAL" or "TEST".
* `data.test_type` is the test-time temporal augmentation. It can take the values from `["single", "2_clips", "3_clips", "all"]`. In the paper it is set to `TEST_TYPE="all"`, indicating that EF value is the weighted average of EF predictions of multiple clips of a single video.

Run the script `eval.sh` to extract embeddings and prototype vectors, visualise them using PCA, and visualise the activation map of new cases together with the contribution of each prototype.

## Dataset

We used EchoNet-Dynamic Dataset to train and evaluate our model. The preprocessed data can be found in `data/echonet/data.csv`. The link to the dataset can be seen below:
EchoNet-Dynamic ([https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound](https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound))

## Acknowledgement

Some code is borrowed from [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) and [ProtoASNet](https://github.com/hooman007/ProtoASNet). 

## Citations

If you use this code in your research, please cite the following paper:

TBD

```
@article{protoefnet,
    title={ProtoEFNet: Prototypical Enhanced Feature Extraction Network},
    author={Yeganeh Ghamary},
    journal={Arxiv},
    year={2024}
}
```

## TODO
* add preprocessed csv file of the data.
* add requirment file of proto env.
* add link to arxiv.
