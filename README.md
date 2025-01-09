# EDformer: Embedded Decomposed Transformer for Explainable Multivariate Time Series Forecasting
### EDformer: https://arxiv.org/abs/2412.12227
Time series forecasting is a crucial challenge with significant applications in areas such as weather prediction, stock market analysis, and scientific simulations. This paper introduces an embedded decomposed transformer, 'EDformer', for multivariate time series forecasting tasks. Without altering the fundamental elements, we reuse the Transformer architecture and consider the capable functions of its constituent parts in this work. Edformer first decomposes the input multivariate signal into seasonal and trend components. Next, the prominent multivariate seasonal component is reconstructed across the reverse dimensions, followed by applying the attention mechanism and feed-forward network in the encoder stage. In particular, the feed-forward network is used for each variable frame to learn nonlinear representations, while the attention mechanism uses the time points of individual seasonal series embedded within variate frames to capture multivariate correlations. Therefore, the trend signal is added with projection, and the final forecasting is performed. The EDformer model obtains state-of-the-art predicting results in terms of accuracy and efficiency on complex real-world time series datasets.

## Core Libraries
The following libraries are used as a core in this framework.

### [Time-Series-Library (TSlib)](https://github.com/thuml/Time-Series-Library)

TSlib is an open-source library for deep learning researchers, especially deep time series analysis.

## Train & Test

Use the [run.py](/run.py) script to train and test the time series models. Check the [scripts](/scripts/) and [slurm](/slurm/) folder to see sample scripts. Make sure you have the datasets downloaded in the `dataset` folder following the `Datasets` section. Following is a sample code to train the electricity dataset using the DLinear model. To test an already trained model, just remove the `--train` parameter.

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model EDformer\
  --channel_independence 0 \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model EDformer \
  --data m4 \
  --features M \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE'
  
## Datasets

The datasets are available at this [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) in the long-term-forecast folder. Download and keep them in the `dataset` folder here. 

### Electricity

The electricity dataset [^1] was collected in 15-minute intervals from 2011 to 2014. We select the records from 2012 to 2014 since many zero values exist in 2011. The processed dataset contains the hourly electricity consumption of 321 clients. We use ’MT 321’ as the target, and the train/val/test is 12/2/2 months. We aggregated it to 1h intervals following prior works.  

### Traffic

This dataset [^2] records the road occupancy rates from different sensors on San Francisco freeways.

## Reproduce

The module was developed using python 3.10.
Compared models of this leaderboard. 

 TimeXer - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [NeurIPS 2024] [Code]
 TimeMixer - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [ICLR 2024] [Code].
 TSMixer - TSMixer: An All-MLP Architecture for Time Series Forecasting [arXiv 2023] [Code]
 iTransformer - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [ICLR 2024] [Code].
 PatchTST - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [ICLR 2023] [Code].
 TimesNet - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [ICLR 2023] [Code].
 DLinear - Are Transformers Effective for Time Series Forecasting? [AAAI 2023] [Code].
 LightTS - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [arXiv 2022] [Code].
 ETSformer - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [arXiv 2022] [Code].
 Non-stationary Transformer - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [NeurIPS 2022] [Code].
 FEDformer - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [ICML 2022] [Code].
 Pyraformer - Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting [ICLR 2022] [Code].
 Autoformer - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [NeurIPS 2021] [Code].
 Informer - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [AAAI 2021] [Code].
 Reformer - Reformer: The Efficient Transformer [ICLR 2020] [Code].
 Transformer - Attention is All You Need [NeurIPS 2017] [Code].
See our latest paper [TimesNet] for the comprehensive benchmark. We will release a real-time updated online version soon.

Newly added baselines. We will add them to the leaderboard after a comprehensive evaluation.

 PAttn - Are Language Models Actually Useful for Time Series Forecasting? [NeurIPS 2024] [Code]
 Mamba - Mamba: Linear-Time Sequence Modeling with Selective State Spaces [arXiv 2023] [Code]
 SegRNN - SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [arXiv 2023] [Code].
 Koopa - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [NeurIPS 2023] [Code].
 FreTS - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [NeurIPS 2023] [Code].
 MICN - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [ICLR 2023][Code].
 Crossformer - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [ICLR 2023][Code].
 TiDE - Long-term Forecasting with TiDE: Time-series Dense Encoder [arXiv 2023] [Code].
 SCINet - SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction [NeurIPS 2022][Code].
 FiLM - FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [NeurIPS 2022][Code].
 TFT - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting [arXiv 2019][Code].

** Train and evaluate** model. We provide the experiment scripts for all benchmarks under the folder ./scripts/. You can reproduce the experiment results as the following examples:
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/EDformer_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/EDformer_M4.sh

Develop your own model.
Add the model file to the folder ./models. You can follow the ./models/Transformer.py. Include the newly added model in the Exp_Basic.model_dict of ./exp/exp_basic.py.
Create the corresponding scripts under the folder ./scripts.

# Citation
If you find this repo useful, please cite our paper.

@article{chakraborty2024edformer,
  title={EDformer: Embedded Decomposition Transformer for Interpretable Multivariate Time Series Predictions},
  author={Chakraborty, Sanjay and Delibasoglu, Ibrahim and Heintz, Fredrik},
  journal={arXiv preprint arXiv:2412.12227},
  year={2024}
}

# Contact
If you have any questions or suggestions, feel free to contact our maintenance team:

Current:

Sanjay Chakraborty (Postdoc, sanjay.chakraborty@liu.se)
Ibrahim Delibasoglu (Postdoc, ibrahim.delibasoglu@liu.se)
