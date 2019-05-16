####################################################################################
# Required packages                                                                #   
####################################################################################
# Data wrangling
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
# Calculate evaluation metrics
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
# Functions to streamline the model training process for complex regression and classification problems. 
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
# Recommender system using Matrix Factorization, R wrapper of the 'libmf' library
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
####################################################################################
# About recosystem                                                                 #    
####################################################################################
# https://www.r-bloggers.com/recosystem-recommender-system-using-parallel-matrix-factorization/
# https://statr.me/2016/07/recommender-system-using-parallel-matrix-factorization/
# https://cran.r-project.org/web/packages/recosystem/recosystem.pdf
# https://rpubs.com/tarashnot/recommender_comparison
# http://infolab.stanford.edu/~ullman/mmds/ch9.pdf
# http://www.mmds.org/
# https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
########################################################################################################
# Extract data set from MovieLens as defined in EDX course: HarvardX data science capstone             #       
########################################################################################################
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

test <- temp %>%     # Renamed to validation to test because some models require train, validation and test data sets. Where validation is used during training.
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from test set back into edx set

removed <- anti_join(temp, test)
edx <- rbind(edx, removed) # original data set
train <- edx # data set to be used for training a model     
# Write data to R data file, so we can restart this script without rerunning the extraction process.
# The edx dataset is kept as the original data set.
# The train dataset is used for training.
saveRDS(edx, "edx.rds")
saveRDS(train, "train.rds")
saveRDS(test, "test.rds")

rm(dl, edx, ratings, movies, test_index, temp, movielens, removed)
gc()
########################################################################################################
# Load train data in memory                                                                            #   
########################################################################################################
# The recommender system, libmf,  
train = train <- readRDS("train.rds") %>% 
                        select(userId, movieId, rating) %>%
                        mutate(userId = as.integer(userId)
                               ,movieId = as.integer(movieId)
                               ,rating = as.integer(rating * 2))

trainMemory = data_memory(user_index = train$userId,
                          item_index = train$movieId,
                          rating = train$rating, index1 = TRUE)
########################################################################################################
# Tune model                                                                                           #   
########################################################################################################
# Constructing a Recommender System Object
recommender <- Reco()
# Uses cross validation to tune the model parameters
opts <- recommender$tune(trainMemory, 
                                  opts = list(
                                          dim      = c(65), # number of latent factors    
                                          costp_l2 = c(0.01, 0.1), # L2 regularization cost for user factors         
                                          costq_l2 = c(0.01, 0.1), # L2 regularization cost for item factors       
                                          costp_l1 = 0, # L1 regularization cost for user factors                    
                                          costq_l1 = 0, # L1 regularization cost for item factors                           
                                          lrate    = c(0.01, 0.1), # learning rate, which can be thought of as the step size in gradient descent.          
                                          nthread  = 4,  # number of threads for parallel computing
                                          nfold = 5, # number of folds in cross validation.  
                                          niter    = 10, #  number of iterations
                                          verbose  = FALSE))
########################################################################################################
# Train the recommender model using optimal parameters                                                 #   
########################################################################################################

recommender$train(trainMemory, 
                                opts = c(opts$min,                    
                                         niter = 100, nthread = 4)) 

rm(trainMemory)

test <- readRDS("test.rds") %>% 
                      select(userId, movieId, rating) %>%
                      mutate(userId = as.integer(userId)
                             ,movieId = as.integer(movieId)
                             ,rating = as.integer(rating * 2))

testMemory <- data_memory(user_index = test$userId, 
                          item_index = test$movieId, 
                          rating = test$rating, index1 = T)  
########################################################################################################
# Predict rating                                                                                       #   
########################################################################################################

test$prediction <- recommender$predict(testMemory, out_memory())
# The system will predict extremes minus 0 and plus 10, which are out of range. 
# A tiny RMSE improvement can be achieved as follows:
test <- test %>% mutate(prediction = case_when(prediction < 0 ~ 0, prediction > 10 ~ 10, TRUE ~ prediction) ) 

########################################################################################################
# Root Mean Square Error (RMSE)                                                                        #
########################################################################################################
# RMSE = sqrt(mean((observed - predicted) ^ 2))
# The RMSE is arround 0.778
rmse(test$rating / 2, test$prediction / 2) 
