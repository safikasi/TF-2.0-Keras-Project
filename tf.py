import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

%matplotlib inline

# Get the Data and EDA
data = pd.read_csv('bank_note_data.csv')
sns.countplot(x='Class',data=data)
sns.pairplot(data,hue='Class')
# Data Preparation
scaler = StandardScaler()
scaler.fit(data.drop('Class',axis=1))
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()
# Train Test Split
X = df_feat
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Tensorflow
print(df_feat.columns)
image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy =tf.feature_column.numeric_column('Entropy')
feat_cols = [image_var,image_skew,image_curt,entropy]
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)
classifier.train(input_fn=input_func,steps=500)
# Model Evaluation
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)
note_predictions = list(classifier.predict(input_fn=pred_fn))
note_predictions[0]
final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])
print(confusion_matrix(y_test,final_preds))
print(classification_report(y_test,final_preds))
# Comparison
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)
print(classification_report(y_test,rfc_preds))
print(confusion_matrix(y_test,rfc_preds))