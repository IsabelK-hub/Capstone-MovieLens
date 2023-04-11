##########################################################
# Downloading the Dataset
##########################################################

# Note: this process could take a couple of minutes

# Install necessary packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

# Load necessary packages
library(tidyverse)
library(caret)
library(recosystem)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Timeout time for download increased to 120s instead of default of 60s to account for longer time needed to download files
options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")


########################################################################
# Creating the edx and the final_holdout_test set
########################################################################

# Final_holdout_test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final_holdout_test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final_holdout_test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Remove objects not needed anymore
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Check if datasets were created correctly (should be approx. 10%))
dim(final_holdout_test) / dim(edx)


##########################################################
# Creating edx_train and val_set
##########################################################

# Creating validation set that will be 10% of remaining edx data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
val_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-val_index,]
temp <- edx[val_index,]

# Make sure userId and movieId are in edx/final_holdout_test set and also in validation set
val_set <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from val_set set back into edx set
removed <- anti_join(temp, val_set)
edx <- rbind(edx, removed)

# Remove objects
rm(val_index, temp, removed)

# Check if datasets are created correctly (should be approx. 10%)
dim(val_set) / dim(edx_train)


##########################################################
# Data exploration edx
##########################################################

# Look at structure of edx
str(edx)

# Get names of variables in edx
variable.names(edx)


# Number of unique users that provided ratings and how many unique movies were rated
edx %>% summarize(n_users = n_distinct(userId),
                  n_movies = n_distinct(movieId))



# Histogram of number of ratings per rating category
edx %>% group_by(rating) %>% 
  ggplot(aes(x=rating)) + 
  geom_histogram() + 
  xlab("Rating") +
  ylab("Number of Ratings") +
  scale_x_continuous(breaks=seq(0,5,0.5)) +
  ggtitle("Distribution rating categories") +
  theme(plot.title = element_text(hjust=0.5))


# Histogram of number of ratings per movie
edx %>% 
  count(movieId)%>%
  ggplot(aes(n)) +
  geom_histogram(color = "lightblue") +
  scale_x_log10() +
  ggtitle("Distribution number of ratings per movie") +
  xlab("MovieID") +
  ylab("Number of Ratings")+
  theme(plot.title = element_text(hjust=0.5))


# Histogram Distribution Number of User Ratings
edx %>% count(userId)%>% 
  ggplot(aes(n)) +
  geom_histogram(color="lightblue")+ 
  scale_x_log10()+
  xlab("Number of Ratings") +
  ylab("Number of Users") +
  ggtitle("Distribution User ratings") +
  theme(plot.title = element_text(hjust=0.5))


##########################################################
# Data cleaning and Matrix Conversion edx & edx_train
##########################################################

# For the edx dataset

# Only use moveId, userId, rating as predictors
# Create pivot_wider matrix matrix with users in rows and movies in columns + ratings in cells
y_edx <- select(edx, movieId, userId, rating) %>% 
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y_edx$userId
y_edx <- as.matrix(y_edx[,-1])
rownames(y_edx) <- rnames

# Map movie ids to titles 
movie_map <- edx %>% select(movieId, title) %>% distinct(movieId, .keep_all = TRUE)



# For the edx_train dataset 

# Only use moveId, userId, rating as predictors
# Create pivot_wider matrix matrix with users in rows and movies in columns + ratings in cells
y_edx_train <- select(edx_train, movieId, userId, rating) %>% 
  pivot_wider(names_from = movieId, values_from = rating) 
rnames <- y_edx_train$userId
y_edx_train <- as.matrix(y_edx_train[,-1])
rownames(y_edx_train) <- rnames

# Map movie ids to titles 
movie_map <- edx_train %>% select(movieId, title) %>% distinct(movieId, .keep_all = TRUE)

##########################################################
# Baseline Model
##########################################################

### Baseline Model
# Simple model -> predict same rating for all movies regardless of user
mu <- mean(y_edx_train, na.rm = TRUE)

# Evaluation of model on val_set
baseline_rmse_val <- RMSE(val_set$rating, mu)
baseline_rmse_val

# Collect the results of the model performance on val_set in Table
results_table <- tibble(Model = "Baseline/Average", RMSE = baseline_rmse_val)
results_table


##########################################################
# Model with Movie Effects
##########################################################

# Model with Movie Effects
b_i <- colMeans(y_edx_train - mu, na.rm = TRUE)
fit_movies <- data.frame(movieId = as.integer(colnames(y_edx_train)), 
                         mu = mu, b_i = b_i)

# Histogram of Movie effects
as.data.frame(b_i) %>% ggplot(aes(b_i))+
  geom_histogram(color="lightblue") +
  xlab("Movie Effect")+
  ylab("Count")+
  ggtitle("Distribution Movie Effects")+
  theme(plot.title = element_text(hjust=0.5))
  

# Evaluation of model on val_set
RMSE_Movie <- left_join(val_set, fit_movies, by = "movieId") %>% 
  mutate(pred = mu + b_i) %>% 
  summarize(rmse = RMSE(rating, pred, na.rm=TRUE))


results_table <- bind_rows(results_table,
                            tibble(Model= "Movie Effects",
                                 RMSE = as.numeric(RMSE_Movie)))
results_table



##########################################################
# Model with User & Movie Effects
##########################################################

# With User Effects & Movie Effects
b_u <- rowMeans(y_edx_train, na.rm = TRUE)

# Histogram of User Effects
as.data.frame(b_u) %>% ggplot(aes(b_u))+
  geom_histogram(color="lightblue") +
  xlab("User Effect")+
  ylab("Count")+
  ggtitle("Distribution User Effects")+
  theme(plot.title = element_text(hjust=0.5))


b_u <- rowMeans(sweep(y_edx_train - mu, 2, b_i), na.rm = TRUE)

fit_users <- data.frame(userId = as.integer(rownames(y_edx_train)), b_u = b_u)



# Evaluation of model on val_set
RMSE_MovieUser <- left_join(val_set, fit_movies, by = "movieId") %>% 
  left_join(fit_users, by = "userId") %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  summarize(rmse = RMSE(rating, pred, na.rm=TRUE))

results_table <- bind_rows(results_table,
                           tibble(Model= "Movie and User Effects",
                                  RMSE = as.numeric(RMSE_MovieUser)))
results_table


##########################################################
# Regularization model with Penalized Least Squares
##########################################################

# Regularization parameter
lambdas <- seq(0, 10, 0.1)

# Regularization model with Penalized Least Squares
sums <- colSums(y_edx_train - mu, na.rm = TRUE)
n <-  colSums(!is.na(y_edx_train))
fit_movies$n <- n

# Train & Tune model
rmses <- sapply(lambdas, function(lambda){
  b_i <-  sums / (n + lambda)
  fit_movies$b_i <- b_i
  left_join(val_set, fit_movies, by = "movieId") %>% mutate(pred = mu + b_i) %>%
    summarize(rmse = RMSE(rating, pred, na.rm=TRUE)) %>%
    pull(rmse)
})

# Plot RMSE dependent on lambdas
tibble(Lambdas=lambdas, RMSE=rmses) %>% 
  ggplot(aes(Lambdas, RMSE)) +
  geom_line() +
  ggtitle("RMSE for different Lambdas")+
  theme(plot.title = element_text(hjust=0.5))

# Chose Lambda that minimizes RMSE
lambda <- lambdas[which.min(rmses)]

# Calculate Movie effect
fit_movies$b_i_reg <- colSums(y_edx_train - mu, na.rm = TRUE) / (n + lambda)

# Calculate User Effect
fit_users$b_u <- rowMeans(sweep(y_edx_train - mu, 2, b_i), na.rm = TRUE)

# Calculate RMSE with Tuned parameters 
RMSE_reg <- left_join(val_set, fit_movies, by = "movieId") %>% 
  left_join(fit_users, by = "userId") %>% 
  mutate(pred = mu + b_i_reg + b_u) %>%
  summarize(rmse = RMSE(rating, pred, na.rm = TRUE))


results_table <- bind_rows(results_table,
                           tibble(Model= "Regularization",
                                  RMSE = as.numeric(RMSE_reg)))
results_table



# !!!!!!! Note: The following code could take up to a 3 hours to run !!!!!!!

##########################################################
# Recosystem
##########################################################

# Specify data for Recosystem
edx_data <- with(edx_train, data_memory(user_index = userId,
                                  item_index = movieId,
                                  rating = rating))
val_data <- with(val_set, data_memory(user_index = userId,
                                     item_index = movieId,
                                     rating=rating))

# Create the model
r <- Reco()

# Select the option for the model 
opts <- r$tune(edx_data, opts = list(nthread = 4, niter= 10))

# Training of the algorithm
r$train(edx_data, opts= c(opts$min, nthread = 4, niter = 20))

# Create Predicted values
Pred_reco <- r$predict(val_data, out_memory())

#RSME
rsme_reco <- RMSE(val_set$rating, Pred_reco)
rsme_reco


results_table <- bind_rows(results_table,
                           tibble(Model= "Matrix Factorisation (Recosystem)",
                                  RMSE = as.numeric(rsme_reco)))
results_table

#####
# Recosystem on edx and test on final holdout

# Convert to format needed for Recosystem
edx_full_data <- with(edx, data_memory(user_index = userId,
                                  item_index = movieId,
                                  rating = rating))
final_holdout_test_data <- with(final_holdout_test, data_memory(user_index = userId,
                                     item_index = movieId,
                                     rating=rating))

# Create the model object
r_final <- Reco()

# Select the option for the model 
opts <- r$tune(edx_full_data, opts = list(nthread = 4, niter= 10))

# Training of the algorithm
r$train(edx_full_data, opts= c(opts$min, nthread = 4, niter = 20))

# Create Predicted values
Pred_reco_final <- r$predict(final_holdout_test_data, out_memory())

#RSME
rsme_reco_final <- RMSE(final_holdout_test$rating, Pred_reco_final)
rsme_reco_final

results_table <- bind_rows(results_table,
                           tibble(Model= "Matrix Factorisation (Recosystem) - Final Model",
                                  RMSE = as.numeric(rsme_reco_final)))
results_table
