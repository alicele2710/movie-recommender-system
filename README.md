
# Learning


## Data Preparation[^1]:

We used PySpark for our ETL pipeline in this step. First, we split the data into 2 dataframes, one for watch time and the other for ratings. Since the system keeps generating new entries as long as a user is watching a particular movie, we only recorded the highest watch time for each user/movie pair in the first dataframe. We then performed a left join between the 2 dataframes on user_id and movieid. Finally we randomly subsampled 20% of the ratings and used that as our validation set. 


## Model Training:

The training steps can be briefly broken down into[^2]



1. Load model, training data, and validation data
2. Tune hyperparameters using validation RMSE to compare models
3. Recombine training and validation data and retrain on entire training dataset


### SVD based CF

We use surprise’s matrix factorization based algorithm, specifically SVD based[^3], to train a recommender system. Surprise supports only explicit rating which is exactly what we need. Moreover, it’s lightweight and its syntax mimics the well-known machine learning library— scikit-learn. Its simplicity and ease of use allows us to implement and analyze a recommender system quickly and efficiently.


### Alternating Least Squares (ALS)

Our second model uses an ALS collaborative filtering model implemented in PySpark.[^4] Because our ELT pipeline already uses PySpark, adding model training was relatively easy. ALS is advantageous because it can use implicit data (clicks or watch time) as a proxy for expliciting data (ratings). We did not take advantage of this in our current implementation, but this is an area for future exploration. PySpark’s ALS is also more complex than the SVD implementation as there are many more parameters that can be tuned and requires more time per training iteration. Therefore, the cost of hyperparameter tuning was much more expensive.


# Model Comparison



* Prediction Accuracy
    * Metrics: Root-mean-square error (RMSE)
    * Data: Evaluate the RMSE of the best model (trained with an optimal set of hyperparameters) on the held-out validation set
    * Operationalization: Report only the best RMSE achieved based on the validation data
* Training Cost
    * Metrics: Training time required
    * Data: Log the training time required to train the model on the full training dataset
    * Operationalization: Report the training time used to train the candidate model on the full training dataset
* Inference Cost
    * Metrics: Prediction response time
    * Data: Log the time spent between initializing a prediction function call and completing the prediction
    * Operationalization: Report the average prediction response time by taking an average of time required to response to 100 recommendation requests
* Disk Space
    * Metrics: Model disk space required
    * Data: Log model space required to store model parameters
    * Operationalization: Save models and report the total disk space required to store all model parameters

<table>
  <tr>
   <td>
<strong>Metrics</strong>
   </td>
   <td><strong>SVD-based CF</strong>
   </td>
   <td><strong>ALS</strong>[^5]
   </td>
  </tr>
  <tr>
   <td>Prediction Accuracy (RMSE)
   </td>
   <td><strong>0.721</strong>
   </td>
   <td>0.81
   </td>
  </tr>
  <tr>
   <td>Training time (second)
   </td>
   <td><strong>40.8</strong>[^6]
   </td>
   <td>433
   </td>
  </tr>
  <tr>
   <td>Inference time (millisecond)
   </td>
   <td><strong>~88</strong>[^7]
   </td>
   <td>580
   </td>
  </tr>
  <tr>
   <td>Disk Space (MB)
   </td>
   <td>686[^8]
   </td>
   <td><strong>154</strong>[^9]
   </td>
  </tr>
</table>


The SVD-based CF has slightly better prediction accuracy (described in RMSE). In addition, the model training time and average real-time prediction response is significantly faster. Although the size of the trained model is quite large, we are currently mitigating by preloading the model and only using function calls for the prediction. Based on the overall model quality, we decided to select SVD-based CF for the production and deployment.


# Prediction Service[^10]

Our deployment focuses on minimizing the amount of computation that actually needs to be done for each user. First, we preload the model, the top 20 movies for unseen users, the set of all movie names, and a user history dictionary containing user IDs as the key and a list of movies that user has already watched as the value (lines 23-36). These are all global variables as they will remain static in our current deployment. 

In the prediction function call, the only input required is the user ID sent to the endpoint. We then perform a search for the user ID in the user history dictionary to see what movies that user has already seen. If the value returns as an empty list (the dictionary is a defaultdict), then the user is not a part of the existing dataset and is an unseen user (lines 38-39). We chose to return the top 20 movies based on average rating for unseen users. If the dictionary search results in a list with values, we then convert this list to a set to quickly perform a difference operation between the full movie set and the user’s history (lines 41-42). This provides us with a set of movies that the user has not yet seen. 

We then predict a rating for every entry in the unseen movie list and record these values in a pandas DataFrame. The pretrained model requires a user ID and movie ID (raw). Because there is no easy and efficient way to process the set of movies as a batch, we iteratively use model.predict for each movie and store the returned movie ID and predicted rating in a list of tuples (lines 44-48). The model computes ratings by simply taking a dot product of a user’s factors with a movie’s factors. These factors are found during matrix decomposition via SVD, where the left singular matrix maps users to latent factors, the diagonal matrix is the strength of these factors, and the right singular matrix maps latent factors to items.[^11] These values are learned during training, allowing for quick computation during inference time with a simple dot product.

The returned predictions are then converted to a Pandas DataFrame and sorted by their predicted ratings into a list of top 20 recommendations. The final returned value is a concatenated string of the entries in the recommendation list delimited by commas (lines 39, 44-50). 




<!-- Footnotes themselves at the bottom. -->
## Notes

[^1]:
     [https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/data_pre_processing.py](https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/data_pre_processing.py)

[^2]:
     Detail training steps can be found at 
    [https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/wleelaki/Code/SVD_train_val.ipynb](https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/wleelaki/Code/SVD_train_val.ipynb)

[^3]:
    [https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) 

[^4]:
     [https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.recommendation.ALS.html](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.recommendation.ALS.html)

[^5]:
<p>
     <a href="https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/prediction/Code/ALS.ipynb">https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/prediction/Code/ALS.ipynb</a> - Implementation is marked by Header cells

[^6]:
<p>
     <a href="https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/Code/SVD_train_val.ipynb">https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/Code/SVD_train_val.ipynb</a>
<p>
    Recommender training time in cell#9

[^7]:
<p>
     <a href="https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/Code/prediction.ipynb">https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/Code/prediction.ipynb</a>
<p>
    Average prediction time calculated in cell#7

[^8]:
<p>
     No code, just space needed for this file:
<p>
    <a href="https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/Code/SVD_V1">https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/Code/SVD_V1</a>

[^9]:
<p>
     No code for disk space, just space needed for this folder: <a href="https://github.com/cmu-seai/group-project-s23-attack-on-the-model/tree/main/Code/ALS">https://github.com/cmu-seai/group-project-s23-attack-on-the-model/tree/main/Code/ALS</a>

[^10]:
     [https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/recommendation/app.py](https://github.com/cmu-seai/group-project-s23-attack-on-the-model/blob/main/recommendation/app.py)
All lines in this section are from the above file

[^11]:
     [https://arxiv.org/pdf/2203.11026.pdf](https://arxiv.org/pdf/2203.11026.pdf)
