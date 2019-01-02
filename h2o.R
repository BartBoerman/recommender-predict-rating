####################################################################################
# Installation instructions for h2o                                                #   
####################################################################################
# http://docs.h2o.ai/h2o/latest-stable/h2o-r/docs/articles/getting_started.html
# https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/ensembles-stacking/stacked_ensemble_h2o_xgboost.Rmd
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html
#Remove any previously installed packages for R.
# if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
# if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# # Next, download packages that H2O depends on.
# pkgs <- c("RCurl","jsonlite")
# for (pkg in pkgs) {
#   if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
# }
# # Download and install the latest H2O package for R.
# install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))
# Finally, let's load H2O and start up an H2O cluster
####################################################################################
# Required packages                                                                #   
####################################################################################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(h2o)) install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))
####################################################################################
# Create edx set, validation set, and submission file                              #    
####################################################################################
# Note: this process could take a couple of minutes
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

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
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Learners will develop their algorithms on the edx set
# For grading, learners will run algorithm on validation set to generate ratings
test <- validation  # added by Bart Boerman. Used to test accuracy before submitting final model.
validation <- validation %>% select(-rating)

# Ratings will go into the CSV submission file below:

write.csv(validation %>% select(userId, movieId) %>% mutate(rating = NA),
          "submission.csv", na = "", row.names=FALSE)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Added by Bart Boerman
# Save object to R data format so it can be restored into a rmd file or
# in a new R session. This will take some seconds for large files but
# is fasther then rebuilding large objects from scratch.
# Refer to: http://www.sthda.com/english/wiki/saving-data-into-r-data-format-rds-and-rdata
saveRDS(validation, "validation.rds")
saveRDS(test, "test.rds")
####################################################################################
# Data wrangling                                                                   #    
####################################################################################
# Function to calculate mode.
f_mode <- function(x, na.rm = FALSE) {
              if(na.rm){
                x = x[!is.na(x)] # Remove missing values if any. 
              }
              x <- sort(x, decreasing = TRUE) # Take the highest rating when ratings are distributed equally, because users generally tend to rate positive
              ux <- unique(x)
              return(ux[which.max(tabulate(match(x, ux)))]) # Return mode 
}
# Impute missing values
edx <- edx %>% mutate(title = if_else(is.na(title),"title unknown (1994)", title) # impute with median release year which is a arbitery chois.
                              ,genres = if_else(is.na(genres),"genre unknown", genres))
# Add year and month based on timestamp rating
# Removed wday and hour because of limited importance
edx <- edx %>% mutate(datetime = lubridate::as_datetime(timestamp, origin = "1970-01-01")
                              ,year = lubridate::year(datetime)
                              ,month = lubridate::month(datetime))
# Extract release year from title 
pattern <- "[/(]\\d{4}[/)]$"
edx <- edx %>% mutate(movie_releaseyear = as.numeric(str_extract(str_extract(title, pattern), regex("\\d{4}")))
                              ,title = str_remove(title, pattern))
# Count number of genres per movie
edx <- edx %>% mutate(movie_n_genres = str_count(genres,"[|]") + 1)
# One-hot encode genre
genres <- c("Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western")
for (g in genres ){
      varNameDummy <- paste("movie_dummy",g, sep = "_")
      edx[[varNameDummy]] <- str_count(edx$genres,g) 
      varNameMode <- paste("user_mode",g, sep = "_")
      tmp <- edx %>%  filter(!! as.name(varNameDummy) == 1) %>%
                      select(userId, !! as.name(varNameDummy), rating) %>%     
                      group_by(userId) %>%
                      mutate(!! varNameMode := f_mode(rating)) %>%
                      ungroup %>%
                      select(userId, !! varNameMode) %>%
                      distinct
      edx <- left_join(x = edx, y = tmp, by = "userId")
      rm(tmp)
}
edx <- edx %>% replace(., is.na(.), 0) # impute NA caused by left joins (users who dit not rate specific genres)
# Add mode rating per genre, movie and user
edx <- edx %>%
                      group_by(genres) %>% 
                      mutate(movie_mode_genre = f_mode(rating)) %>%
                      ungroup
edx <- edx %>%
                      group_by(movieId) %>% 
                      mutate(movie_mode_movie = f_mode(rating)) %>%
                      ungroup
edx <- edx %>%
                      group_by(userId) %>% 
                      mutate(user_mode_user = f_mode(rating)) %>%
                      ungroup
# Add number of ratings per movie
edx <- edx %>%
                      group_by(movieId) %>% 
                      mutate(movie_n__users = n()) %>%
                      ungroup
# Add number of ratings per user
edx <- edx %>%
                      group_by(userId) %>% 
                      mutate(user_n_movies = n()) %>%
                      ungroup
####################################################################################
# Generate meta data sets                                                          #
# The data will be added to the validation set                                     #
####################################################################################
meta_movie <- edx %>% 
                      select(-userId, -rating, -timestamp, -datetime, -year, -month) %>% 
                      select(movieId, title, genres, movie_releaseyear, movie_n_genres, movie_mode_movie, movie_mode_genre, movie_n__users, matches("movie_dummy")) %>%
                      distinct
x <- names(meta_movie)
meta_user <- edx %>% 
                      select(-rating, -timestamp, -datetime, -year, -month) %>% 
                      select(-x, userId) %>% 
                      distinct
meta_user <- meta_user %>% select(userId, user_mode_Thriller) %>% distinct()
####################################################################################
# Start script from this point if you already completed data wrangling             #    
####################################################################################
saveRDS(edx, "edx.rds") # save object to R data format so it can be restored
saveRDS(meta_movie, "meta_movie.rds") # save object to R data format so it can be restored
saveRDS(meta_user, "meta_user.rds") # save object to R data format so it can be restored
edx <- readRDS("edx.rds") # reload data, you can re-run script from this point forward.
####################################################################################
# Start h2o cloud                                                                  #    
####################################################################################
h2o.init(max_mem_size = "8G", # 
         nthreads = 3,         # take max min one, otherwise bad things will happen
         startH2O = TRUE,
         bind_to_localhost = FALSE) # make h20 excessable with default port number 45321, e.g. http://localhost:54321
####################################################################################
# Get data into cloud                                                              #    
####################################################################################
# Note: this can take some minutes, time for coffee
drop <- c("timestamp","datetime","movieId","userId","title","genres","month")

train_h2o <- as.h2o(edx %>% select(-drop), 
                            destination_frame = "train") # name in h2o cloud

                    
# Identify predictors (x) and response (y)
y <- "rating"
x <- setdiff(names(train_h2o), y) 

# For classification, response should be a factor
# Also, convert nominal data to factor
factor <- c(y)
train_h2o[,factor] <- as.factor(train_h2o[,factor])
####################################################################################
# Train XGBoost                                                                    #    
####################################################################################
Sys.time()
# Note: this can take some hours depening on your system. Watch a movie?
xgb_h2o <- h2o.xgboost(x = x,
                       y = y,
                       training_frame = train_h2o,
                       distribution = "AUTO",
                       ntrees = 25, # nround
                       max_depth = 6,
                       learn_rate = 0.3, # eta
                       gamma = 0,
                       min_rows = 10,
                       stopping_rounds = 2,
                       stopping_metric = "misclassification", # 0.6493718 (mean) 0.7051093 (mode, also without genre)
                       stopping_tolerance = 0.001,
                       nfolds = 2,
                       fold_assignment = "Stratified",
                       tree_method = "auto",
                       keep_cross_validation_predictions = FALSE,
                       model_id	= "xgb_h2o",
                       seed = 1413)
Sys.time()
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/save-and-load-model.html
varimp_xgb_h2o <- h2o.varimp(xgb_h2o)

tree <- h2o.getModelTree(model = xgb_h2o, tree_number = 1, tree_class = "NO")

# https://htmlpreview.github.io/?https://github.com/ledell/sldm4-h2o/blob/master/sldm4-deeplearning-h2o.html
# Sys.time()
# dnn_h2o <- h2o.deeplearning(x = x,
#                             y = y,
#                             training_frame = train_h2o,
#                             epochs = 10,
#                             hidden = c(10,10),
#                             input_dropout_ratio = 0.1,
#                             distribution = "AUTO",
#                             nfolds = 3,                            
#                             score_interval = 1,                    
#                             stopping_rounds = 2,                   #used for early stopping
#                             stopping_metric = "misclassification", #used for early stopping
#                             stopping_tolerance = 0.01,             #1e-3  #used for early stopping
#                             standardize = TRUE,
#                             keep_cross_validation_predictions = FALSE,
#                             model_id = "dnn_h2o",
#                             seed = 1413)
# h2o.saveModel(dnn_h2o)
# Sys.time()
#h2o.shutdown()

# https://www.datacamp.com/community/tutorials/keras-r-deep-learning
####################################################################################
# Test on unseen data                                                              #    
####################################################################################
# Data wrangling
test <- readRDS("test.rds") %>% mutate(datetime = lubridate::as_datetime(timestamp, origin = "1970-01-01")
                                        ,year = lubridate::year(datetime)
                                        ,month = lubridate::month(datetime)) %>%
                                select(userId, movieId, rating, year, month)
# Add meta data about movies and users
test <- left_join(x = test, y = meta_user, by = "userId")
test <- left_join(x = test, y = meta_movie, by = "movieId")
# Load data into cloud
test_h2o <- as.h2o(test, key = "test")  
factor <- c("genres", "userId", "movieId")
test_h2o[,factor] <- as.factor(test_h2o[,factor])
test_h2o$prediction <- h2o.predict(xgb_h2o, newdata = test_h2o)["predict"]
# Predict ratings on validation set
test_h2o.df <- as.data.frame(test_h2o)
confusionMatrix(table(test$rating, test$prediction))


