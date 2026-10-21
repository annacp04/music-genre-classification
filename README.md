Classification of musical genres

In the Jupyter Notebook you can see all the experimentation. 

The implementation of all the Preprocessing, Models and Experimentation is divided in this way: 

    1. ExploratoryDataAnalysis: Data inspection
    2. Preprocessing: clean and encode database, and train/test division
    3. Models: Comparison of all models with 10-fold Cross Validation over the 80% train. Adjusment of diferent hyperparameters (gridsearch), overfitting assessment, different feature selection techniques. 
    4. FeatureSelection: only fabricates different feature selection techniques
    5. SongAnalysis: useful to answer questions, look at the position of the segments, agregation...

In the dataset in segments (features_3_sec.csv): Preprocessing(group_col="filename")
separates train/test by songs y Models(groups=...) does the cross validation by song.
