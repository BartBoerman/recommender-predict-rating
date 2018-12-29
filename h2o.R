# http://docs.h2o.ai/h2o/latest-stable/h2o-r/docs/articles/getting_started.html
# https://github.com/h2oai/h2o-tutorials/blob/master/tutorials/ensembles-stacking/stacked_ensemble_h2o_xgboost.Rmd
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/xgboost.html
#Remove any previously installed packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
# Next, download packages that H2O depends on.
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
# Download and install the latest H2O package for R.
install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))
# Finally, let's load H2O and start up an H2O cluster
require(h2o)
h2o.init(max_mem_size = "10G", 
         nthreads = 3)
# Get data into h2o cloud
h2o.train <- as.h2o(edx %>% select(-timestamp, - title, -datetime), 
                    key = "train")
rm(edx)
# Identify predictors and response
y <- "rating"
x <- setdiff(names(h2o.train), y)
# For binary classification, response should be a factor
h2o.train[,y] <- as.factor(h2o.train[,y])
h2o.train$userId <- as.factor(h2o.train$userId)
h2o.train$movieId <- as.factor(h2o.train$movieId)
Sys.time()
h2o.xgb <- h2o.xgboost(x = x,
                       y = y,
                       training_frame = h2o.train,
                       distribution = "AUTO",
                       ntrees = 50,
                       max_depth = 6,
                       eta = 0.3,
                       gamma = 0,
                       min_rows = 10,
                       learn_rate = 0.2,
                       stopping_rounds = 2,
                       stopping_metric = "misclassification",
                       stopping_tolerance = 0.01,
                       nfolds = 3,
                       fold_assignment = "Stratified",
                       tree_method = "auto",
                       keep_cross_validation_predictions = FALSE,
                       model_id	= "edx",
                       seed = 1413)
Sys.time()
