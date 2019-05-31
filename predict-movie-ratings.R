########################################################################################################
# Required packages                                                                                    #       
########################################################################################################
# Data wrangling.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
# Calculate evaluation metrics.
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
# Functions to streamline the model training process for complex regression and classification problems. 
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# Recommender system using Matrix Factorization, R wrapper of the 'libmf' library.
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
########################################################################################################
# Extract data set from MovieLens as defined in EDX course: HarvardX data science capstone             #       
########################################################################################################
dl <- tempfile()
# I modified the script so that it only downloads the files if they do not exists
if (!file.exists("ml-10M100K/ratings.dat") | !file.exists("ml-10M100K/movies.dat")){
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                        col.names = c("userId", "movieId", "rating", "timestamp"))
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
}
# Process files only if train or test do not exists
if (!exists("train") | !exists("test")){
  ratings <- read.table(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")),
                        col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
  movielens <- left_join(ratings, movies, by = "movieId")
  # !!!!! Uncomment line below to test on smaller dataset !!!!!.
  #movielens <- head(movielens, 10000)
  
  # Test set will be 10% of MovieLens data
  
  set.seed(1, sample.kind = "Rounding") #  gives error: non-uniform 'Rounding' sampler used. 
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  # Make sure userId and movieId in test set are also in edx set
  
  test <- temp %>%     
              semi_join(edx, by = "movieId") %>%
              semi_join(edx, by = "userId")
  
  # Add rows removed from test set back into edx set
  removed <- anti_join(temp, test)
  # Data to be used for training a model
  train <- rbind(edx, removed)      
  # Housekeeping
  train.org <- train # Use .org for exploratory data analysis
  test.org <- test # Use .org for analysis of predictions
  rm(dl, edx, ratings, movies, test_index, temp, movielens, removed)
  gc()
}
########################################################################################################
# Load train data in memory                                                                            #   
########################################################################################################
# Collaborative-Filtering systems create a matrix of users and movies with known ratings as values. 
# Hence, we do not load other columns into memory.
# 'recosystem' Needs userId and MovieId to be integers
# Rating is multiplied with 2 to get on integer. You can provide ratings as real values both RMSE will be lower.
# I did not yet investigate why.
train <- train  %>%  select(userId, movieId, rating) %>%
  mutate(userId = as.integer(userId)
         ,movieId = as.integer(movieId)
         ,rating = as.integer(rating * 2))   
# 'data_memory' Is a function in the recosystem package. It loads the data into memory. 
# Consider using a disk based approach when low on memory 
trainMemory <- data_memory(user_index = train$userId,
                           item_index = train$movieId,
                           rating = train$rating, index1 = TRUE)
########################################################################################################
# Tune model                                                                                           #   
########################################################################################################
# Constructing a Recommender System Object (based on LIBMF, National Taiwan University).
recommender <- Reco()
# The recommender uses cross validation to find the optimal parameters.
# This is simplified by limiting the number of values per parameter to be tested. 
# The number of cross validations is set to 5 for faster processing time.
# !! The training process will consume a lot of time. Please take a break !!
opts <- recommender$tune(trainMemory, 
                         opts = list(
                           dim      = c(65), # number of latent factors    
                           costp_l2 = c(0.01, 0.1), # L2 regularization cost for user factors         
                           costq_l2 = c(0.01, 0.1), # L2 regularization cost for item factors       
                           costp_l1 = 0, # L1 regularization cost for user factors                    
                           costq_l1 = 0, # L1 regularization cost for item factors                           
                           lrate    = c(0.01, 0.1), # learning rate, which can be thought of as the step size in gradient descent.          
                           nthread  = 4,  # number of threads for parallel computing
                           nfold = 5, # number of folds in cross validation  
                           niter    = 10, #  number of iterations
                           verbose  = FALSE))
########################################################################################################
# Train the recommender model using optimal parameters                                                 #   
########################################################################################################
# Train the recommender with the optimized parameters.
# This beast needs be fed with a lot of iterations to train for a competitive accuracy.
recommender$train(trainMemory, opts = c(opts$min, # optimized parameters                     
                                        niter = 100, # iterations 
                                        nthread = 4)) # number of threads 
# No Housekeeping, objects are used for pdf
########################################################################################################
# Load test data in memory                                                                             #   
########################################################################################################
# Load 'Unseen' test data into memory so that we can test RMSE.
test <- test %>% select(userId, movieId, rating) %>%
                mutate(userId = as.integer(userId)
                       ,movieId = as.integer(movieId)
                       ,rating = as.integer(rating * 2))   
testMemory <- data_memory(user_index = test$userId, 
                          item_index = test$movieId, 
                          rating = test$rating, index1 = T)  
########################################################################################################
# Predict rating on test data                                                                          #   
########################################################################################################
# Add predictions to test dataset
test$prediction <- recommender$predict(testMemory, out_memory())
# The system will predict extremes minus 0 and plus 10, which are out of range. 
# A tiny RMSE improvement can be achieved as follows:
test <- test %>% mutate(prediction_opts = round(case_when(prediction < 1 ~ 1, prediction > 10 ~ 10, TRUE ~ prediction)) ) 
########################################################################################################
# Root Mean Square Error (RMSE)                                                                        #
########################################################################################################
# RMSE = sqrt(mean((observed - predicted) ^ 2))
RMSE <- rmse(test$rating / 2, round(test$prediction_opts) / 2)  
# When input is the original, real-valued rating use: RMSE <- rmse(test$rating, ceiling(test$prediction_opts * 2)/2 )   
print(RMSE) # 0.7917868
# Uncomment lines below if interested in accuracy. It's 27.64% accurate predicting on 10 possible outcomes.
#ACCURACY <- accuracy(test$rating / 2, round(test$prediction_opts) / 2) 
#print(ACCURACY) # 0.2763559

