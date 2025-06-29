# importing required libraries
import h2o
from h2o.automl import H2OAutoML
import os

#initialzing h20 environment
try:
    h2o.init()
    print("h2o environment initialized")
except Exception as e:
    print(f"h20 environment not initialized {e}")

# Loading all the csv files from the 'balanced_data' folder
dir='./balanced_data'
total_data=[]
for filename in os.listdir(dir):
    if filename.endswith('.csv'):
        filepath=os.path.join(dir,filename)
        df=h2o.import_file(filepath)
        total_data.append(df)
        print(f"loaded :{filename}")

# Defining the features and Target variable
target='TARGET'
features=[col for col in df.columns if col!=target]

# Converting the target to categorical for classification task
try:
    df[target]=df[target].asfactor()
    print(f'column:{target} successfully converted into categorical column')
except Exception as e:
    print(f"Failed to categorize column {target} due to : {e}")

# Splitting training and testing data

try:
    train,test=df.split_frame(ratios=[0.8],seed=42)
    print(f"Splitted successfully")
    print(f"\n train_data={train.nrows}")
    print(f"\n test_data={test.nrows}")
except Exception as e:
    print(f"Failed to make the train and test split due to :{e}")

#initialzing and training AutoML
try:
    auto_ml=H2OAutoML(
        max_models=5, #total number of models to try
        seed=42, 
        max_runtime_secs=600,
        exclude_algos=['DeepLearning','StackedEnsemble'], # Avoiding the use of resource intensive models
        balance_classes=False, #set to false as already balanced
        nfolds=4, # stratified cross-validation
        sort_metric="AUC", # Area under the curve as the metric to sort the models
        verbosity='info' # Provide basic logging info
    )
    auto_ml.train(x=features,y=target,training_frame=train)
    print("Initialized AutoML for Training")
except Exception as e:
    print(f"Failed to initialize AutoML for Training due to {e}")

# View the Leaderboard for the best performing models
leaderboard=auto_ml.leaderboard
print(leaderboard.head(rows=leaderboard.nrows))

# MAke the predictions
predictions=auto_ml.leader.predict(test)

# Evaluate Model performance
performance=auto_ml.leader.model_performance(test_data=test)
print("\n the performance evaluation is here....")
print(performance)







