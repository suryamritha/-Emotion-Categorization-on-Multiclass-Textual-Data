# Emotion Categorization on Multiclass Textual Data
 This  uses Natural Language Processing and extensively explores models such as SVM, Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting, Bagging, and Ensemble Model. The objective is to identify the most effective model for nuanced emotion classification from textual data, focusing on the WASSA dataset for a multiclass classification problem covering four emotions: anger, fear, sadness, and joy. Meticulous analysis of these algorithms delves into the intricate fabric of written language, deciphering a spectrum of emotional states. The ensemble model (SVM + XGBoost) achieves  the best  accuracy


### Data Preprocessing

Data preprocessing is essential to prepare text data for machine learning by enhancing data quality and consistency. The process begins with converting all text to lowercase to ensure uniformity and removing special characters to focus on meaningful content. Tokenization breaks text into individual words or tokens, allowing for detailed analysis. Part-Of-Speech (POS) tagging labels each token with its grammatical category, aiding in syntactic understanding. Stopwords, common words that don’t carry significant meaning, are removed to emphasize more important words. Lemmatization reduces words to their base forms, considering context, which helps in reducing dimensionality and capturing core meanings. Finally, the preprocessed tokens are reassembled to maintain text coherence, ensuring modifications do not disrupt the overall structure.

### Data Splitting

The dataset is divided into 80% for training and 20% for testing in a stratified and random manner. This method ensures that the models are trained on a substantial portion of the data, allowing them to learn underlying patterns effectively. Testing on unseen data provides a robust assessment of the models' generalization capabilities, helping to gauge their effectiveness and mitigate the risk of overfitting. This systematic separation is crucial for determining the models' proficiency in accurately recognizing emotions in real-world textual data.

### Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is employed to convert processed text into numerical features suitable for machine learning models. TF-IDF combines Term Frequency (TF), which measures how often a term appears in a document, with Inverse Document Frequency (IDF), which assesses the term's importance across the entire dataset. This technique captures the significance of terms both locally and globally, enhancing model performance by focusing on meaningful words and reducing computational inefficiencies.

### Model Selection

Various machine learning and ensemble models are considered for emotion recognition, including Support Vector Machines (SVM), Naive Bayes, Decision Trees, Random Forest, XGBoost, LightGBM, AdaBoost, Gradient Boosting, Bagging, and Ensemble Classifiers. Each model offers unique strengths and is evaluated to identify the most suitable approach for emotion classification. The goal is to determine the best model or ensemble through rigorous testing, ensuring robust performance in capturing the complexities of emotional expressions in text.

### Model Training

During model training, the selected machine learning and ensemble models are trained using TF-IDF vectorized data and encoded emotion labels. This step involves fitting the models to the data, allowing them to learn the intricate relationships between textual features and emotion labels. Proper training ensures the models are well-prepared for subsequent evaluation, enabling them to predict emotions accurately.

### Evaluation

Models undergo evaluation using five-fold stratified cross-validation and an independent test set. Metrics such as mean accuracy and standard deviation provide insights into overall performance and consistency. After training on the full dataset, detailed classification reports (including precision, recall, and F1-score) on the test set offer valuable insights into the models' generalization capabilities. This evaluation helps in understanding the effectiveness of each model and guides decisions on their real-world applicability for emotion recognition.SVM+XGBOOST gives the best accuracy .

### Fine Tuning

Hyperparameter tuning involves systematically searching for optimal values for parameters like the regularization parameter 'C' for SVM and the number of estimators for XGBoost. Utilizing GridSearchCV, this process aims to identify the most effective configurations for each model, enhancing overall accuracy and generalization. Fine-tuning is critical for maximizing the models' performance in recognizing emotions from textual data.

### Prediction/Performance

In the prediction and evaluation phase, the trained model is deployed to predict emotions in new text. The primary goal is to assess the model's effectiveness in real-world scenarios, identifying areas for improvement and iteratively fine-tuning hyperparameters. This process ensures the deployment of a robust model capable of accurately predicting emotions like happiness, sadness, anger, or fear from textual data, thereby validating its practical utility.
