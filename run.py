from pandas import read_csv
from onet_classifier import Onet_classifier


train_df_path = 'train_data.csv'           # Path to the training data
test_df_path = 'test_data.csv'             # Path to the test data
occupations_df_path = 'occupations.csv'    # Path to the occupations data

n = 5                                      # Number of predictions to be made

model_name = 'all-MiniLM-L12-v2'           # Name of the model to be used
train_epochs = 5                           # Number of epochs to train the model
train_batch_size = 64                      # Batch size for training
predict_batch_size = 64                    # Batch size for prediction

# Checking performance of the out-of-box model
test_df = read_csv(test_df_path)
print('shape of test_df', test_df.shape)
occupations_df = read_csv(occupations_df_path)

title = test_df['TITLE_RAW'].tolist()
body = test_df['BODY'].tolist()
ground_truth_label = test_df['ONET_NAME'].tolist()

clf = Onet_classifier(model_name=model_name, occupations_df=occupations_df)
preds = clf.predict(title=title, body=body, n=n, batch_size=predict_batch_size)

precision, recall, f1, accuracy, top_k_accuracy = clf.calculate_metrics(predictions=preds, ground_truth=ground_truth_label)

print('Precision of out-of-box model on test set: ', precision)
print('Recall of out-of-box model on test set: ', recall)
print('F1 score of out-of-box model on test set: ', f1)
print('Accuracy of out-of-box model on test set: ', accuracy)
print('Top-k Accuracy of out-of-box model on test set: ', top_k_accuracy)


train_df = read_csv(train_df_path)
print('shape of train_df', train_df.shape)

clf.train(train_df=train_df, occupations_df=occupations_df, batch_size=train_batch_size, epochs=train_epochs)
clf.save_model('trained_model')

# Checking performance of the trained model
preds = clf.predict(title=title, body=body, n=n, batch_size=predict_batch_size)

precision, recall, f1, accuracy, top_k_accuracy = clf.calculate_metrics(predictions=preds, ground_truth=ground_truth_label)

print('Precision of trained model on test set: ', precision)
print('Recall of trained model on test set: ', recall)
print('F1 score of trained model on test set: ', f1)
print('Accuracy of trained model on test set: ', accuracy)
print('Top-k Accuracy of trained model on test set: ', top_k_accuracy)