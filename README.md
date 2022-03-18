# Generating Synthetic Health Data
---

![Generating Synthetic Health Data](https://i.imgur.com/eYaLdNR.png)

In this project I trained a generative adversarial network to generate synthetic health tabular data. I then used five classification models (logistic regression, random forest, decision tree, gradient boosting, and a neural network) to compare the predictive accuracy of real data alone to generated data.

## Introduction

Every data scientist has experienced the feeling of having a dying question they want to answer but don’t have enough data to properly address it. Sometimes certain machine learning models require a certain type or amount of data that is not available. Whether the financial cost is too high, there isn’t enough time, or it introduces high participant burden (e.g. privacy, safety, time, etc.) further data collection may not be possible. To alleviate this issue, the use of synthetic data is becoming more frequent. Synthetic data is data that is generated through an artificial process, rather than being created by actual events. If executed correctly, increasing the amount and type of data can lead to significant performance increases for machine learning models. The figure below from Gartner shows the expected trend of increased use of synthetic data in machine learning in coming years.

![Synthetic Data Trends](https://i.imgur.com/qkygdKe.png)

The use of synthetic data can be used in a wide variety of industries such as autonomous vehicles, robotics, security, and finance. However, one area in which it is particularly relevant is in healthcare. High volumes of health data are difficult to collect for a number of reasons. The cost of anything medical related is typically very high, and access to patients with the profile you are looking for can be limited by external factors such as hospital capacity, disease prevalence, and illness severity. For these reasons, finding artificial methods for generating reliable health data would be extremely beneficial.

There are several methods that can be used to generate this synthetic data. One purposed method is through the use of generative adversarial networks. Generative adversarial networks are constructed using two neural networks that are competing with each other (i.e. adversarial). The generator network is trying to trick the discriminator, and the discriminator network is attempting to avoid being tricked. Specifically, the generator is trying to generate content that is similar to the real data. If the discriminator is successful in distinguishing this fabrication from the real data, then the generator is penalized. On the other hand, the discriminator is penalized if it fails to do so. Through this iterative process, a model is trained that is able to generate data that is similar to the original data set. The original paper describing these models was written by Ian Goodfellow, and goes into more extensive detail on their structure. The following diagram shows a basic depiction of the model structure.

![Generative Adeversarial Network](https://i.imgur.com/3pCNh4k.png)

Currently, the most common use for generative adversarial networks is for image learning. We have seen applications such as restoration, colorization, upscaling, and image generation. Famously, [thispersondoesnotexist.com](https://thispersondoesnotexist.com/) demonstrates a generative adversarial network that was trained on real faces, and is now able to generate realistic faces that do not exist in reality. However, such models trained on images can sometimes take weeks or months to train. However, these models have also been trained on other types of data such as text and numbers. These forms of data require significantly less training time. This lower barrier to entry widens their usefulness as a tool that can be leveraged in data science and AI research. One specific use for generative adversarial networks is for the generation of tabular data.

The concept of using these models for tabular data generation was discussed in the article [Tabular GANs for uneven distribution](https://arxiv.org/abs/2010.00638) by Insaf Ashrapov. The concepts from this paper were later applied for use in python. [Tabgan](https://pypi.org/project/tabgan/) is a python library that allows for the application of a generative adversarial network to generate tabular data. This method allows for the generation of textual categorical data and numerical data. However, just because we are able to generate data, does not mean that the generated data is similar to the real data, and furthermore if it is actually useful in machine learning models.

## Project Summary

**Problem:** Certain AI models require large amounts of data that can be difficult to collect on the scale that is needed. Generating additional synthetic data is possible, but the quality of synthetic data needs to be representative of the original data for it to be useable in modeling.

**Goal:** The present project aims to assess the ability of a generative adversarial network to generate synthetic health data that is representative of the original real data. In this case representativeness of the generated data is evaluated by assessing its closeness to the original variables, as well as the change in predictive accuracy that occurs to various classification algorithms when including it in various models.

## Data

To address the goals of this project, a generative adversarial network was applied to an electronic health record dataset collected from a private hospital in Indonesia. The dataset contains several biomarker indicators collected from laboratory tests, as well as age and sex demographics. Finally, the data includes an indication of whether an individual’s next treatment was an outpatient or inpatient treatment. A copy of this data can be found on [Kaggle](https://www.kaggle.com/saurabhshahane/patient-treatment-classification) which includes a full data dictionary.

In this project, we will assess the quality of the generated synthetic data by applying several classification algorithms to use the biomarkers and demographics variables to predict a given patient’s next treatment type (outpatient/inpatient).

The following figure shows a brief snapshot of the original real health data. As we can see, aside from leucocytes, the density plots portray relatively normally distributed biomarker variables. Age ranging from 0 to 100 showed a slight bimodal distributions. Finally, slightly more individuals were male, and the majority of individuals had an outpatient treatment for their next treatment.

![Electronic Health Record Data Overview](https://i.imgur.com/2WCUljR.png)


## Part 1: Assessing Classification Accuracy With Real Data

To determine how the inclusion of generated data affects its usability for machine learning models, we must first assess the base predictive accuracy of the real health data. We used five classification algorithms to predict next treatment type (outpatient/inpatient) from demographics and biomarker variables:

1. **Logistic Regression:** Logistic Regression is a supervised machine learning algorithm that allows for classification through transformation using the logistic sigmoid function to return a probability value that a data record belongs to a certain class. This method produced an accuracy score of **0.695**.
2. **Decision Tree:** Decision trees are a supervised machine learning algorithm where the data is continuously split according to certain parameters. Doing so narrows down which class a particular data record belongs to. This method produced an accuracy score of **0.656**.
3. **Random Forest:** Random forest is an ensemble machine learning algorithm that operates by constructing many decision trees, and determines class by observing the most common outcomes across the multiple trees. This method produced an accuracy score of **0.763**.
4. **Gradient Boosting:** Gradient boosting is an iterative functional gradient algorithm. It seeks to reduce a loss function by iteratively selecting a function that indicates a negative gradient. Through identification of these weak learners, an additive model is created that can be applied to classification problems. This method produced an accuracy score of **0.742**.
5. **Neural Network (MLP Classifier):** A multi-layer perceptron classifier is a neural network in which an algorithm updates model parameters through the iterative calculation of partial derivatives of a loss function. This method produced an accuracy score of **0.682**.

## Part 2. Generating New Data

Leveraging the [tabgan](https://pypi.org/project/tabgan/) python library, a generative adversarial network was applied to generate about 4500 additional records. This model was applied all variables in the original real dataset. The following figure compares the means and distributions of each biomarker variable in the generated data to the original real data. While the generated data deviates very slightly across each variable, for the most part it matches the real data closely, and seems to be representative. Additionally, it correctly recreated the categorical variables as categorical and the continuous variables as continuous.  


![Comparing Real Data to Generated Data](https://i.imgur.com/tKlpQdl.png)

## Part 3: Assessing Classification Accuracy With Generated Data

Using the same five classification algorithms, next treatment type (outpatient/inpatient) was predicted from demographics and biomarker variables using a combined dataset including the original data, as well as the generated synthetic data. Accuracy scores were then calculated for each model:

**Logistic Regression Accuracy:** 0.668
**Decision Tree Accuracy:** 0.661
**Random Forest Accuracy:** 0.772
**Gradient Boosting Accuracy:** 0.740
**Neural Network (MLP Classifier) Accuracy:** 0.720

The following figure compares the accuracy scores generated from the classification models applied to the original real data alone, to the combined data. From this comparison, we see that for both sets of data, the random forrest was the most accurate. More importantly, we can see that across the board, the generated data does not seem to have a meaningful negative impact on the accuracy of the models. In several cases, it seems to even improve the accuracy of the models.

![Model Accuracy For Real and Generated Data](https://i.imgur.com/2XRo0Eq.png)

## 4. Conclusions

From these analyses, we can see that within this particular data set, various classification models are capable of using demographics and biomarker data to predict the type of a patient’s next treatment (outpatient/inpatient). Additionally, a generative adversarial network was able to successfully generate additional health data that was representative of the original dataset. Furthermore, this generated synthetic health data had minimal negative impact on model accuracy scores compared to the real data alone. In several instances, it even led to higher accuracy scores.

Together these analyses show that the generation of synthetic data has potential to increase data sample sizes in a way that allows for improved machine learning models. This practice has huge potential benefits for a variety of industries such as healthcare, and will allow for representative machine learning models to be trained that would otherwise be unobtainable.

---

If you enjoyed this project, please consider following me on [Twitter](https://twitter.com/Peter_Nooteboom) and [Linkedin](https://www.linkedin.com/in/peter-nooteboom/).

