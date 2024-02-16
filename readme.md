### Objective

To classify job titles into occupational categories.

### Approach

My approach for this project was to leverage a pre-trained transformer model to generate the embeddings of the input job titles and the occupational categories and compare them using cosine similarity. I concatenated the input job title with the job description, and the ONET category with the ONET description and then created the embeddings. This strategy aimed to provide the model with richer contextual information, enabling more accurate comparisons. The results were then sorted to identify the top-k most similar ONET categories.

To optimize workflow efficiency, I implemented an embedding cache, a simple yet effective dictionary storing ONET category embeddings along with their descriptions, indexed by the corresponding ONET categories. This eliminates the need for redundant computations during predictions, streamlining the process.

For model architecture, I opted for a small 12-layer model, specifically the `all-MiniLM-L12-v2` with 33M parameters. The choice of a smaller model, ideal for proof of concept projects, offers various advantages. It facilitates faster training and inference, enabling fast experimentation to determine feasibility. Additionally, we can use larger batch sizes, and avoid model quantization. Furthermore, smaller models allow experiments to run seamlessly on smaller GPUs.

### Metrics

I assess the efficacy of the model through a comprehensive set of metrics, including precision, recall, F1 score, accuracy, and top-k accuracy. Given that the input and ground truth are in string format, I employ a label encoder to transform them into numerical values, facilitating the computation of these metrics.

The initial performance of the model out-of-the-box was suboptimal, as evident from the figures presented in the table below. However, after fine-tuning for 5 epochs, there is significant imporvement in the model's overall performance.

| Metric         | Before Fine-tuning | After Fine-tuning | Improvement |
| -------------- | ------------------ | ----------------- | ----------- |
| Precision      | 0.52               | 0.71              | 0.19        |
| Recall         | 0.25               | 0.56              | 0.31        |
| F1 Score       | 0.27               | 0.58              | 0.31        |
| Accuracy       | 0.25               | 0.58              | 0.33        |
| Top 5 Accuracy | 0.48               | 0.79              | 0.31        |

### Conclusions

The model exhibits notable improvement after only 5 epochs of training. In the context of a comprehensive project with an extended timeline, I would explore additional strategies to further elevate performance. These include:

- **Scaling up Model Size:** Delving into larger model architectures promises substantial improvements.
- **Extended Fine-Tuning:** Fine-tuning for more epochs could improve the model's capabilities.
- **Augmentation and Metadata:** Experimenting with augmented input and incorporating more metadata may contribute to enhanced learning.
- **Hyperparameter Optimization:** Exploring hyperparameter configurations could optimize model performance.
- **Error Analysis:** Conducting a thorough examination of instances where the model performs poorly provides valuable insights for refinement.
- **Ensemble Models:** Investigating ensemble models could further boost overall performance.
- **Cross-Validation:** Employing cross-validation techniques facilitates a comprehensive assessment of the model across diverse subsets of the data.
- **Model Interpretability:** Integrating techniques for model interpretability ensures a clearer understanding of the model's decision-making processes.

This multifaceted approach, coupled with an extended project timeline, positions us to unlock even greater potential and robustness in the model's performance.
