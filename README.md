# AI_Risk_Prediction_And_Treatment_Effect_Estimation
 Decision Tree and LSTM implementations to predict spinal surgery revision

 This project deals with the problem of accurately predicting revision (a second hospitalization due to the same surgical procedure) spinal surgery likelihood given information on patients treatment after first surgery, with aims of enabling medical professionals to timely focuss efforts in the correct patients and avoid patient suffering and funds spending. It attempts to answer the research question on whether there is an impact of timely data processing on the prediction quality by benchmarking a traditional Decision Tree model and a LSTM network.  

 A Shapleý analysis is performed on the initial resulting models. Specifying different Shapley thresholds, features which fulfill or exceed the minimum threshold for the level are selected, and included in subsequent model fits. 

 A Bayesian hyperparameter optimization tecnique is applied for hyperparameter tuning, since the model was initially overfitting. 

 The LSTM shower clear outperforming results in comparisson to its peer Decision Tree whenever providing models with the same time span, larger time span provision as well as finer time resolution for the LSTM, resulted in better performance. 

 This project contributed to the results of the publicly funded research project funded by the german government called AIR PTE, standing for Artificial Intelligence Risk Prediction and Treatment Effect Estimation. 
