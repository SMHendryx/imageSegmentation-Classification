# Authored by Sean Hendryx

# Script built to run on R version: Very Secure Dishes
# References: R 3d scatterplots: http://www.r-bloggers.com/getting-fancy-with-3-d-scatterplots/

#Script draws RGB values of each pixel in 3d space

library("scatterplot3d") 

#setwd("/your/path/here")

df = read.csv("mesquitesSubsetPixelData.csv", header=TRUE, sep = ",")

with(df, {
   scatterplot3d(r,   # x axis
                 g,     # y axis
                 b,    # z axis
                 main="3-D Scatterplot RGB Values of Each Pixel in Mesquites.png")
})