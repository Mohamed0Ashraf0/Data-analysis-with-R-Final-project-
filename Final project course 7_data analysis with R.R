# Final project of Data Analysis with R course (course no.7)  :
# The goal is to predict the precipitations using the NOAA weather data set - JFK Airport(NewYork) 

# Business use case : 
# 1.Agriculture : 
# Detect unseasonal temperature change and alert farmers about potential damage to plants.
# Energy regulate solar cell charging hours based on weather type condition and temperature.
# regulate wind turbine operation based on wind speed and wind direction. 
# Generate energy demand alerts based on temperature.
# Remotely adjust air conditioning configs to boost energy efficiency based on temperature shifts.

# 2.Retail : 
# Estimate outdoor retail foot trafic based on weather conditions and temperature predictions.

# 3.Manufacturing : 
# Integrate hourly temperature into building model simulations to test structural integrity.
# Adjust window tint in a vehicle based on temperature and UV index.

# 4.Healthcare:
# Connect to a smart watch and send push notifications when the weather is nice enough to go for a walk outside.

# 1.Importing required modules : 
library(tidyverse)
library(tidymodels)
library(Metrics)


# 2.Download and unzip the data set : 
download.file("https://dax-cdn.cdn.appdomain.cloud/dax-noaa-weather-data-jfk-airport/1.1.4/noaa-weather-sample-data.tar.gz" , destfile = "Downloads")
untar("C:/Users/Lenovo/Downloads/noaa-weather-sample-data.tar.gz")

# 3.Extract and read into project : 
jfk_weather_sample <- read_csv("E:/My Projects/R Projects/Data Sets/noaa-weather-sample-data/jfk_weather_sample.csv") 
head(jfk_weather_sample)
glimpse(jfk_weather_sample)

# from section 4 to section 6 , it is focused on preprocessing the data 
# 4.Select subset of columns : 
modified_sample <- select(jfk_weather_sample,
                          c(HOURLYRelativeHumidity,
                               HOURLYDRYBULBTEMPF,
                               HOURLYPrecip,
                               HOURLYWindSpeed,
                               HOURLYStationPressure))

head(modified_sample , n = 10)

# 5.Clean up columns: 
unique(jfk_weather_sample$HOURLYPrecip)

modified_sample_2 <- modified_sample %>%
  mutate(HOURLYPrecip = ifelse(HOURLYPrecip == "T" , 0.00,HOURLYPrecip)) %>%
  mutate( HOURLYPrecip = str_remove(HOURLYPrecip,pattern = "s$")) %>%
  drop_na(HOURLYPrecip) %>%
  drop_na(HOURLYStationPressure)

sum(is.na(modified_sample_2$HOURLYStationPressure))

# modified_sample_2$HOURLYPrecip[modified_sample_2$HOURLYPrecip == 0] <- 0.00
unique(modified_sample_2$HOURLYPrecip)
view(modified_sample_2)

# 6.Convert columns to numerical types : 
glimpse(modified_sample_2)

converted_sample <- modified_sample_2 %>%
  mutate(HOURLYPrecip = as.numeric(HOURLYPrecip))
glimpse(converted_sample)

# 7.Rename columns : 
renamed_sample <- converted_sample %>%
  dplyr::rename(relative_humidity = HOURLYRelativeHumidity,
                dry_bulb_temp_f = HOURLYDRYBULBTEMPF,
                precip = HOURLYPrecip,
                wind_speed = HOURLYWindSpeed,
                station_pressure = HOURLYStationPressure) 

# 8.Exploratory data analysis : 
set.seed(1234)
split_renamed_sample <- initial_split(renamed_sample,prop = 0.8)
train_data <- training(split_renamed_sample)
test_data <- testing(split_renamed_sample)

# creating histograms to explore the distribution of the data : 
ggplot(train_data,aes(x = relative_humidity)) +
  geom_histogram(binwidth = 2 , fill = "skyblue" , color = "black")

ggplot(train_data,aes(x = dry_bulb_temp_f )) +
  geom_histogram(binwidth = 2 , fill = "skyblue" , color = "black")

ggplot(train_data,aes(x = precip)) +
  geom_histogram(binwidth = 2 , fill = "skyblue" , color = "black")

ggplot(train_data,aes(x = wind_speed)) +
  geom_histogram(binwidth = 2 , fill = "skyblue" , color = "black")

ggplot(train_data,aes(x = station_pressure)) +
  geom_histogram(binwidth = 2 , fill = "skyblue" , color = "black")

# creating box plot to explore the distribution of the data :
ggplot(train_data , aes(y = relative_humidity )) + 
  geom_boxplot()
?geom_boxplot
ggplot(train_data , aes(x = dry_bulb_temp_f , y = precip)) + 
  geom_boxplot()

ggplot(train_data , aes(x = wind_speed , y = precip)) + 
  geom_boxplot()

ggplot(train_data , aes(x = station_pressure , y = precip)) + 
  geom_boxplot()

# 9.Linear Regression : 
# we will create a simple linear regression for each variable of the 4 with precip variable as the response variable : 

# rh_model1 <- lm(precip ~ relative_humidity , data = train_data)
lm_spec <- linear_reg() %>%
  set_engine(engine = "lm")

rh_model1 <- lm_spec %>%
  fit(precip ~ relative_humidity , data = train_data)
ggplot(train_data,aes(x= relative_humidity,y=precip)) +
  geom_point() +
  geom_smooth(method = "lm",
              formula = y ~ x,
              col = "red",
              se = FALSE)

# dbtf_model2 <- lm(precip ~ dry_bulb_temp_f , data = train_data)
dbtf_model2 <- lm_spec %>%
  fit(precip ~ dry_bulb_temp_f , data = train_data)
ggplot(train_data,aes(x = dry_bulb_temp_f,y = precip)) +
  geom_point() 

# ws_model3 <- lm(precip ~ wind_speed , data = train_data)
ws_model3 <- lm_spec %>%
  fit(precip ~ wind_speed , data = train_data)
ggplot(train_data,aes(x = wind_speed , y = precip)) +
  geom_point() +
  geom_smooth(method = "lm", na.rm = TRUE)

# sp_model4 <- lm(precip ~ station_pressure , data = train_data)
sp_model4 <- lm_spec %>%
  fit(precip ~ station_pressure , data = train_data)
ggplot(train_data,aes(x = station_pressure , y = precip)) +
  geom_point() +
  geom_smooth(method = "lm",
              formula = y ~ x ,
              col = "red",
              se = FALSE)

# 10.Improve the model : 
# we will create more models and apply different improving techniques such as : 
# 1.adding more features/predictors 
# 2.adding regularization(L1,L2 or mix)
# 3.adding polynomial component  (increase complexity of the model)

# adding regularization (L2 ridge) : 
sample_recipe <- recipe(precip ~ .,data = train_data)

ridge_spec <- linear_reg(penalty = 0.1 , mixture = 0) %>%
  set_engine(engine = "glmnet")

ridge_wf <- workflow() %>%
  add_recipe(sample_recipe)

ridge_fit5 <- ridge_wf %>% # there is error here which i can't solve 
  add_model(ridge_spec) %>%
  fit(data = train_data)

ridge_fit5 %>%
  extract_fit_parsnip() %>%
  tidy()


# adding polynomial component : 
lm_spec <- linear_reg() %>%
  set_engine(engine = "lm")

train_fit6 <- lm_spec %>%
  fit(precip ~ poly(station_pressure , 3) ,data =  train_data)

train_results <- train_fit6 %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)

test_results <- train_fit6 %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)

train_rmse <- sqrt(mean(train_results$truth - train_results$.pred)^2) # rmse without normalization
train_rmse <- rmse(train_results$truth,train_results$.pred) #  rmse with normalization
train_rsq <- rsq(train_results,truth = truth , estimate = .pred)

test_rmse <- sqrt(mean(test_results$truth - test_results$.pred)^2) #  rmse without normalization
test_rmse <- rmse(test_results$truth,test_results$.pred) #  rmse with normalization
test_rsq <- rsq(test_results,truth = truth , estimate = .pred)

# 11.Find the best model : 
# 1.evaluate the model on the testing set using at least one metric(mse,rmse,rsq)
# 2.after calculating the metrics on the testing set for each model , print them out in a table to easly compare 
# 3.finally ,from the comparison table you create ,conclude which model performed the best 


# models i've created tell now : 
# 1.rh_model1  2.dbtf_model2  3.ws_model3  4.sp_model4  5.ridge_fit5  6.train_fit6

# 1.rh_model1 :
train_results1.1 <- rh_model1%>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
rmse_results1.1 <- rmse(train_results1$truth,train_results1$.pred)
rsq_results1.1 <- rsq(train_results1 , truth = truth , estimate = .pred)

test_results1.2 <- rh_model1 %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
rmse_results1.2 <- rmse(test_results1.2$truth,test_results1.2$.pred)
rsq_results1.2 <- rsq(test_results1.2 , truth = truth , estimate = .pred)

# 2.dbtf_model2 : 
train_results2.1 <- dbtf_model2 %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
rmse_results2.1 <- rmse(train_results2.1$truth,train_results2.1$.pred)
rsq_results2.1 <- rsq(train_results2.1 , truth = truth , estimate = .pred)

test_results2.2 <- dbtf_model2 %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
rmse_results2.2 <- rmse(test_results2.2$truth,test_results2.2$.pred) 
rsq_results2.2 <- rsq(test_results2.2 , truth = truth ,estimate = .pred)

# 3.ws_model3 : 
train_results3.1 <- ws_model3 %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
rmse_results3.1 <- rmse(train_results3.1$truth,train_results3.1$.pred)
rsq_results3.1 <- rsq(train_results3.1 , truth = truth , estimate = .pred)

test_results3.2 <- ws_model3 %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
rmse_results3.2 <- rmse(test_results3.2$truth,test_results3.2$.pred) 
rsq_results3.2 <- rsq(test_results3.2 , truth = truth ,estimate = .pred)

# 4.sp_model4 :   
train_results4.1 <- sp_model4 %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
rmse_results4.1 <- rmse(train_results4.1$truth,train_results4.1$.pred)
rsq_results4.1 <- rsq(train_results4.1 , truth = truth , estimate = .pred)

test_results4.2 <- sp_model4 %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
rmse_results4.2 <- rmse(test_results4.2$truth,test_results4.2$.pred) 
rsq_results4.2 <- rsq(test_results4.2 , truth = truth ,estimate = .pred)

# 5.ridge_fit5 : 
train_results5.1 <- ridge_fit5 %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
rmse_results5.1 <- rmse(train_results5.1$truth,train_results5.1$.pred)
rsq_results5.1 <- rsq(train_results5.1 , truth = truth , estimate = .pred)

test_results5.2 <- ridge_fit5 %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
rmse_results5.2 <- rmse(test_results5.2$truth,test_results5.2$.pred) 
rsq_results5.2 <- rsq(test_results5.2 , truth = truth ,estimate = .pred)

# 6.train_fit6 :
train_results6.1 <- train_fit6 %>%
  predict(new_data = train_data) %>%
  mutate(truth = train_data$precip)
rmse_results6.1 <- rmse(train_results6.1$truth,train_results6.1$.pred)
rsq_results6.1 <- rsq(train_results6.1 , truth = truth , estimate = .pred)  
  
test_results6.2 <- train_fit6 %>%
  predict(new_data = test_data) %>%
  mutate(truth = test_data$precip)
rmse_results6.2 <- rmse(test_results6.2$truth,test_results6.2$.pred) 
rsq_results6.2 <- rsq(test_results6.2 , truth = truth ,estimate = .pred)

# creating the table : 
model_names <- c("rh_model1","dbft_model2","ws_model3","sp_model4","ridge_fit5","train_fit6")
train_errors <- c(rmse_results1.1,rmse_results2.1,rmse_results3.1,rmse_results4.1,rmse_results5.1,rmse_results6.1)
test_errors <- c(rmse_results1.2,rmse_results2.2,rmse_results3.2,rmse_results4.2,rmse_results5.2,rmse_results6.2)  
comparison_df <- data.frame(model_names,train_errors,test_errors)  
comparison_df  

# so the best model is : 1.rh_model1   2.ridgefit5






