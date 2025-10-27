# Speaker Independent Speech Emotion Recognition
Source code from my Bachelor's degree final project entitled "Investigating Machine Learning and Deep Learning approaches to Speech Emotion Recognition".

I am in the process of consolidating my data preprocessing and model training scripts into a modular form with a command-line interface, which would be more accessible and efficient to examine and utilise. The source code will be shared for reproducibility, as evidence of my work and for educational and academic purposes. 

# Project Summary: 

The project tested a traditional ML approach, whereby handcrafted features extracted from OpenSMILE were used to train a logistic regression classifier. This was followed by a Deep Learning (DL) approach which utilised Convolutional Neural Networks (CNN) trained on three data representations separately, the raw waveforms, Mel Spectrograms, and MFCCS.
Results were compared across the four datasets, between the CNN models and the different data representations, and between the CNN models and traditional ML classifier.
Most of the time was spent experimenting with data preprocessing optimisations, and neural network architecture optimisations for the CNNs, and after multiple iterations the DL approach outperformed the traditional ML approach which leveraged handcrafted features. 

Four of the main datasets for SER were used, EmoDB, IEMOCAP, SAVEE, and RAVDESS, and the work adopted a speaker independent approach whereby the entirety of at least one speaker's speech samples were excluded from taining and used exclusively for testing, in this way there is no bleeding of speaker-specific qualities into the training data. This approach consistently proves to be more challenging for ML models, and it is thought that this improves model generalisability.

My results support the view that DL without leveraging domain knowledge via handcrafted features achieves superior results on speaker independent speech emotion recognition. 

I concluded my project by creating an ensemble model which averaged the softmax probabilities per class from the three separate CNN models trained on different data representations. With exception to one dataset (IEMOCAP) this improved classification accuracy. 

### Ensemble model accuracies:

```
EmoDB: 91%
RAVDESS: 64%
SAVEE: 65%
IEMOCAP: 44% (The best performing non-ensemble model for IEMOCAP was trained on MFCCs and achieved 48.67%)
```

### Comparison with other contemporary research

I compared my results with other Speaker Independent (SI) studies. The EmoDB ensemble performs among the top performing speaker independent models, despite using a relatively simple and lightweight approach. I found only one work which outperformed my EmoDB ensemble, Amjad et al. 2021 achieved 92.65% WAR, compared to my model's 91% WAR. Xu et al. 2022 also reported 90.61% Accuracy and Farooq et al. 2020 reported 90.5%. 
The other ensembles did not perform so well, for RAVDESS Amjad et al. 2021 (82.75% WAR), Sayed et al. 2025 (73.75% Acc), and Farooq et al. 2020 (73.5% WAR) show demonstrably better accuracy, at around 10-20% higher than my own. However, SI studies on RAVDESS were extremely rare, and two of these models used AlexNet pre-trained network, and Sayed used a CNN+LSTM hybrid model which is somewhat more expensive to train. 

My SAVEE ensemble was outperformed by Amjad et al. 2021 (75.38% WAR) and Farooq et al. 2020 (66.90% WAR). This dataset was even rarer as a choice of SI study than RAVDESS. 

IEMOCAP appeared to the most popular dataset for SI SER, and I found the largest number of studies which outperformed my own. My approach seemed to perform relatively poorly on this dataset compared to the others. IEMOCAP is much different to the other three datasets examined, and proves to be among the most challenging of all the SER datasets. The models which outperformed my own all used increasingly more complex techniques such as leveraging pre-trained AlexNets, hybrid feature extraction and feature selection, multi-task learning, pre-trained wav2vec, multi-head attention. The models I compared with my own that achieved greater accuracies are as follows:

```
Amjad et al. 2021 (84% WAR)
Cai et al. 2021 (78.15% WAR)
Farooq et al. 2020 (73.50% WAR)
Xu et al. 2022 (73.42 % Unweighted Accuracy)
Xu et al. 2024 (70.2% UAR)
Chen, 2018 (64.74% UAR)
Fayek et al. 2017 (60.89% UAR)
Latif, 2019 (60.23% UAR)
Vladimir Chernykh, 2018 (54% Accuracy)
```
