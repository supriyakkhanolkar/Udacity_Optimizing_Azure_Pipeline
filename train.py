from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)

    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    
    #Add return statement for this function
    return x_df, y_df
    


# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

# ds = ### YOUR CODE HERE ###


from azureml.core import Workspace, Dataset

print("getting run object")
run = Run.get_context()
'''
print("Getting existing workspace")
#ws = Workspace.from_config(path='.azureml/')
ws = run.experiment.workspace

print("Getting datastore")
datastore = ws.get_default_datastore()

'''

print("Creating dataset")
#ds = Dataset.Tabular.from_delimited_files(path = [(datastore, "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv")])
ds = Dataset.Tabular.from_delimited_files(path="https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv")
ds.to_pandas_dataframe()

print("Calling clean_data()")
x, y = clean_data(ds)

# TODO: Split data into train and test sets.

### YOUR CODE HERE ###a
print("splitting data")

#x_train, y_train, x_test, y_test = train_test_split(x,y,test_size=0.33,random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)
'''
print("getting run object")
run = Run.get_context()
'''

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    print("Calling LogisticRegression")
    print("Regularization Strength:", np.float(args.C))
    print("Max iterations:", np.int(args.max_iter))
    #model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    model = LogisticRegression(C=np.float(args.C), max_iter=np.int(args.max_iter)).fit(x_train, y_train)

    print("Calling scoring function")

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    print("Accuracy is {}".format(accuracy))
    
    # Store model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')
    
if __name__ == '__main__':
    main()