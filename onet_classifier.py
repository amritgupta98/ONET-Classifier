from tqdm import tqdm
from torch import topk
from pandas import DataFrame, merge
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Onet_classifier():
    '''
    A class for the ONET classifier.

    Attributes:
        model: An embedding model.
        emb_cache: A dictionary for caching embeddings.

    Methods:
        __init__(self, model_name: str, occupations_df: DataFrame)
            The init method loads the model and creates an embeddings cache.

        create_embeddings_cache(self, occupations_df: DataFrame)
            This method creates an embeddings cache using the model and
            occupations_df.

        predict(self, title: Union[str, List],
                body: Union[str, List], n: int = 5, batch_size: int = 64)
            This method predicts the ONET label for a given title and body.

        calculate_metrics(self, predictions: Union[str, List],
                          ground_truth: Union[str, List])
            This method calculates the precision, recall, F1 score, accuracy,
            and top-k accuracy given the predictions and ground truth.

        train(self, train_df: DataFrame, occupations_df: DataFrame,
              batch_size: int = 64, epochs: int = 5)
            This method trains the model.

        save_model(self, path: str)
            This method saves the model.
    '''

    def __init__(self, model_name: str, occupations_df: DataFrame) -> None:
        '''
        The init method loads the model and creates an embeddings cache.

        :param model_name: The name of the model to be used.
        :type model_name: str
        :param occupations_df: A dataframe containing the ONET titles
                               and descriptions.
        :type occupations_df: DataFrame
        :return: None
        :rtype: None
        '''
        self.model = SentenceTransformer(model_name)
        self.emb_cache = self.create_embeddings_cache(occupations_df)


    def create_embeddings_cache(self, occupations_df: DataFrame) -> Dict:
        '''
        This method creates an embeddings cache using the model
        and occupations_df.

        :param occupations_df: A dataframe containing the ONET titles
                               and descriptions.
        :type occupations_df: DataFrame
        :return: A dictionary containing the ONET titles and their embeddings.
        :rtype: Dict
        '''
        title = occupations_df['O*NET-SOC 2019 Title'].tolist()
        description = occupations_df['O*NET-SOC 2019 Description'].tolist()

        model_inputs = []

        for i in range(len(title)):
            model_inputs.append(" ".join([title[i], description[i]]))

        embs = self.model.encode(model_inputs, show_progress_bar=True)
        emb_cache = dict(zip(title, embs))

        return emb_cache

    def predict(self, title: Union[str, List], 
                body: Union[str, List],
                n: int = 5, 
                batch_size: int = 64
                ) -> Union[List[str], List[List[str]]]:
        '''
        This method predicts the ONET label for a given title and body.

        :param title: The title of the job posting.
        :type title: Union[str, List]
        :param body: The body of the job posting.
        :type body: Union[str, List]
        :param n: The number of predictions to be returned.
        :type n: int
        :param batch_size: The batch size for prediction.
        :type batch_size: int
        :return: The predicted ONET label.
        :rtype: Union[List[str], List[List[str]]]
        '''
        onet_title = list(self.emb_cache.keys())
        embs = list(self.emb_cache.values())

        if isinstance(title, str):
            title = [title]
        if isinstance(body, str):
            body = [body]

        model_inputs = []
        for i in range(len(title)):
            model_inputs.append(" ".join([title[i], body[i]]))

        input_embs = self.model.encode(model_inputs, 
                                       show_progress_bar=True, 
                                       batch_size=batch_size)

        top_n_similarity = []
        for input_emb in tqdm(input_embs):
            similarity = util.cos_sim(input_emb, embs)
            top_n_similarity.append(topk(similarity, k=n).indices[0].tolist())

        predicted_titles = []
        for i in range(len(top_n_similarity)):
            top_n_preds = []
            for j in range(n):
                top_n_preds.append(onet_title[top_n_similarity[i][j]])
            predicted_titles.append(top_n_preds)

        return predicted_titles

    def calculate_metrics(self, predictions: Union[str, List], 
                          ground_truth: Union[str, List]
                         ) -> Tuple[float, float, float, float]:
        '''
        This method calculates the precision, recall, F1 score, accuracy,
        and top-k accuracy given the predictions and ground truth.

        :param predictions: The predicted ONET label.
        :type predictions: Union[str, List]
        :param ground_truth: The ground truth ONET label.
        :type ground_truth: Union[str, List]
        :return: The precision, recall, f1, accuracy and top-k accuracy
                 of the model.
        :rtype: Tuple[float, float, float, float]
        '''
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]


        # Checking if the true class is within the top-k predictions
        correct_predictions = [true_label in top_k for true_label, top_k
                               in zip(ground_truth, predictions)]

        # Calculating top-k accuracy
        top_k_accuracy = sum(correct_predictions) / len(correct_predictions)

        label_encoder = LabelEncoder()

        # considering the prediction with the highest similarity
        predictions = [x[0] for x in predictions]

        all_labels = list(set(predictions + ground_truth))
        label_encoder.fit(all_labels)

        ground_truth_numerical = label_encoder.transform(ground_truth)
        predictions_numerical = label_encoder.transform(predictions)

        precision = precision_score(ground_truth_numerical,
                                    predictions_numerical,
                                    average='weighted',
                                    zero_division=1)
        recall = recall_score(ground_truth_numerical,
                              predictions_numerical,
                              average='weighted',
                              zero_division=1)
        f1 = f1_score(ground_truth_numerical,
                      predictions_numerical,
                      average='weighted')
        accuracy = accuracy_score(ground_truth_numerical,
                                  predictions_numerical)

        return precision, recall, f1, accuracy, top_k_accuracy

    def train(self, train_df: DataFrame, occupations_df: DataFrame,
              batch_size: int = 64, epochs: int = 5) -> None:
        '''
        This method trains the model.

        :param train_df: A dataframe containing the training data.
        :type train_df: DataFrame
        :param occupations_df: A dataframe containing the ONET titles
                               and descriptions.
        :type occupations_df: DataFrame
        :param batch_size: The batch size for training.
        :type batch_size: int
        :param epochs: The number of epochs for training.
        :type epochs: int
        :return: None
        :rtype: None
        '''

        # performing inner join on train_df and occupations_df to get
        # descriptions from occupations_df
        merged_df = merge(train_df, occupations_df, left_on='ONET',
                          right_on='O*NET-SOC 2019 Code', how='inner')

        # removing unnecessary columns
        del merged_df['ID']
        del merged_df['POSTED']
        del merged_df['O*NET-SOC 2019 Code']
        del merged_df['O*NET-SOC 2019 Title']

        input_title = merged_df['TITLE_RAW'].tolist()
        input_body = merged_df['BODY'].tolist()

        onet_name = merged_df['ONET_NAME'].tolist()
        onet_description = merged_df['O*NET-SOC 2019 Description'].tolist()

        train_examples = []

        for i in tqdm(range(len(input_title))):
            input_sample = " ".join([input_title[i], input_body[i]])
            target = " ".join([onet_name[i], onet_description[i]])

            train_examples.append(InputExample(texts=[input_sample, target]))

        train_dataloader = DataLoader(train_examples, shuffle=True,
                                      batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       epochs=epochs, warmup_steps=100)

        print('Training completed!')

    def save_model(self, path: str) -> None:
        '''
        This method saves the model.

        :param path: The path where the model is to be saved.
        :type path: str
        :return: None
        :rtype: None
        '''
        self.model.save(path)


