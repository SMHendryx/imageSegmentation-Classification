require(ggplot2)
setwd("/Users/seanhendryx/Google Drive/THE UNIVERSITY OF ARIZONA (UA)/COURSES/SPRING 16/Bayesian Modeling and Inferrence/Final/myScripts")
df <- read.csv("LOverK.csv", header=TRUE, sep = ",")

p <- ggplot(df, aes(df$K, df$L)) +
  theme_bw() + geom_line()

p <- p + labs(title = "Cross Validation on Held-Out Test Data") 
  #Manually adjust text size:
  #+ theme(axis.title = element_text(size = 18)) + theme(title = element_text(size = 20))

p <- p + labs(
    x = "K",
    y = "L"
  )
p

