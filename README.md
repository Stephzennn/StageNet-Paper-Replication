# StageNet-Paper-Replication

In this code, we aim to replicate the study by Junyi Gao, Cao Xiao, Yasha Wang, Wen Tang, Lucas M. Glass, and Jimeng Sun (2020) to demonstrate that a stage-aware model is superior at predicting decompensation in patients compared to benchmark models.

We will use the benchmark code by Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and Aram Galstyan (Harutyunyan2019) to create our dataset for testing the StageNet model.

In this project, we will utilize a Jupyter notebook (.ipynb file) to train our model using Google Colab. This approach allows us to leverage Colab's resources efficiently. For our experiments, we will use both the MIMIC-III and MIMIC-IV datasets to evaluate the performance of our model and validate the underlying concepts.

The MIMIC-III and MIMIC-IV datasets are well-known medical datasets that contain detailed health-related information. We have provided links to the benchmarks and data preparation scripts for both datasets in our paper. These resources will assist in replicating our experiments and validating our results.

Additionally, we will use the provided scripts to prepare the decompensation data necessary for training our model. Decompensation refers to the deterioration of a patient's condition, and preparing this data involves handling various preprocessing steps to ensure its suitability for model training.

Once the model is trained, we will utilize the train.py script to evaluate the results. This script will allow us to test the trained model on various metrics and ensure its effectiveness and robustness.

By using this structured approach, we aim to achieve a thorough and comprehensive evaluation of our model's performance.
