library(data.table)
library(FeatureHashing)
library(xgboost)
library(dplyr)
library(Matrix)

train=fread('redhat_data_new/act_train_new.csv') %>% as.data.frame()
test=fread('redhat_data_new/act_test_new.csv') %>% as.data.frame()

#people data frame
people=fread('redhat_data_new/people.csv') %>% as.data.frame()
people$char_1<-NULL #unnecessary duplicate to char_2
names(people)[2:length(names(people))]=paste0('people_',names(people)[2:length(names(people))])

p_logi <- names(people)[which(sapply(people, is.logical))]
for (col in p_logi) set(people, j = col, value = as.numeric(people[[col]]))

#reducing group_1 dimension
people$people_group_1[people$people_group_1 %in% names(which(table(people$people_group_1)==1))]='group unique'


#reducing char_10 dimension
unique.char_10=
  rbind(
    select(train,people_id,char_10),
    select(train,people_id,char_10)) %>% group_by(char_10) %>% 
  summarize(n=n_distinct(people_id)) %>% 
  filter(n==1) %>% 
  select(char_10) %>%
  as.matrix() %>% 
  as.vector()

train$char_10[train$char_10 %in% unique.char_10]='type unique'
test$char_10[test$char_10 %in% unique.char_10]='type unique'

d1 <- merge(train, people, by = "people_id", all.x = T)
d2 <- merge(test, people, by = "people_id", all.x = T)
Y <- d1$outcome
d1$outcome <- NULL

row.train=nrow(train)
gc()

D=rbind(d1,d2)
D$i=1:dim(D)[1]


###uncomment this for CV run
#set.seed(120)
#unique_p <- unique(d1$people_id)
#valid_p  <- unique_p[sample(1:length(unique_p), 40000)]
#valid <- which(d1$people_id %in% valid_p)
#model <- (1:length(d1$people_id))[-valid]

test_activity_id=test$activity_id
rm(train,test,d1,d2);gc()


char.cols=c('activity_category','people_group_1',
            'char_1','char_2','char_3','char_4','char_5','char_6','char_7','char_8','char_9','char_10',
            'people_char_2','people_char_3','people_char_4','people_char_5','people_char_6','people_char_7','people_char_8','people_char_9')
for (f in char.cols) {
  if (class(D[[f]])=="character") {
    levels <- unique(c(D[[f]]))
    D[[f]] <- as.numeric(factor(D[[f]], levels=levels))
  }
}


D.sparse=
  cBind(sparseMatrix(D$i,D$activity_category),
        sparseMatrix(D$i,D$people_group_1),
        sparseMatrix(D$i,D$char_1),
        sparseMatrix(D$i,D$char_2),
        sparseMatrix(D$i,D$char_3),
        sparseMatrix(D$i,D$char_4),
        sparseMatrix(D$i,D$char_5),
        sparseMatrix(D$i,D$char_6),
        sparseMatrix(D$i,D$char_7),
        sparseMatrix(D$i,D$char_8),
        sparseMatrix(D$i,D$char_9),
        sparseMatrix(D$i,D$people_char_2),
        sparseMatrix(D$i,D$people_char_3),
        sparseMatrix(D$i,D$people_char_4),
        sparseMatrix(D$i,D$people_char_5),
        sparseMatrix(D$i,D$people_char_6),
        sparseMatrix(D$i,D$people_char_7),
        sparseMatrix(D$i,D$people_char_8),
        sparseMatrix(D$i,D$people_char_9)
  )

D.sparse=
  cBind(D.sparse,
        D$people_char_10,
        D$people_char_11,
        D$people_char_12,
        D$people_char_13,
        D$people_char_14,
        D$people_char_15,
        D$people_char_16,
        D$people_char_17,
        D$people_char_18,
        D$people_char_19,
        D$people_char_20,
        D$people_char_21,
        D$people_char_22,
        D$people_char_23,
        D$people_char_24,
        D$people_char_25,
        D$people_char_26,
        D$people_char_27,
        D$people_char_28,
        D$people_char_29,
        D$people_char_30,
        D$people_char_31,
        D$people_char_32,
        D$people_char_33,
        D$people_char_34,
        D$people_char_35,
        D$people_char_36,
        D$people_char_37,
        D$people_char_38,
        D$binay_sum)

train.sparse=D.sparse[1:row.train,]
test.sparse=D.sparse[(row.train+1):nrow(D.sparse),]


cat(Sys.time())
cat("Unmerging train/test sparse data\n")

train.sparse=D.sparse[1:row.train,]
test.sparse=D.sparse[(row.train+1):nrow(D.sparse),]

# Hash train to sparse dmatrix X_train + LibSVM/SVMLight format

cat(Sys.time())
cat("Making data for SVMLight format\n")

# LibSVM format if you use Python / etc. ALWAYS USEFUL

# TOO LONG

cat("Creating SVMLight format\n")
dtrain  <- xgb.DMatrix(train.sparse, label = Y)
gc(verbose=FALSE)
cat("Exporting SVMLight format\n")
xgb.DMatrix.save(dtrain, "dtrain.data")
gc(verbose=FALSE)
#rm(dtrain) #avoid getting through memory limits
#gc(verbose=FALSE)
cat("Zipping SVMLight\n")
zip("dtrain.data.zip", "dtrain.data", flags = "-m9X", extras = "", zip = Sys.getenv("R_ZIPCMD", "zip"))
#file.remove("dtrain.data")

cat(Sys.time())
cat("File size of train in SVMLight: ", file.size("dtrain.data.zip"), "\n", sep = "")

cat("Creating SVMLight format\n")
dtest  <- xgb.DMatrix(test.sparse)
gc(verbose=FALSE)
cat("Exporting SVMLight format\n")
xgb.DMatrix.save(dtest, "dtest.data")
gc(verbose=FALSE)
cat("Zipping SVMLight\n")
zip("dtest.data.zip", "dtest.data", flags = "-m9X", extras = "", zip = Sys.getenv("R_ZIPCMD", "zip"))
#file.remove("dtest.data")
cat(Sys.time())
cat("File size of test in SVMLight: ", file.size("dtest.data.zip"), "\n", sep = "")

#cat("Re-creating SVMLight format\n")
#dtrain  <- xgb.DMatrix(train.sparse, label = Y) #recreate train sparse to run under the memory limit of 8589934592 bytes
gc(verbose=FALSE)