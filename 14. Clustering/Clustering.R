# Calculating Distances
# data(milk, package="flexclust")
milk <- read.csv('G:/Statistics (Python)/Datasets/milk.csv',
                 row.names = 1)
head(milk, 2)
d <- dist(milk)
as.matrix(d)[1:5,1:5]


# Average linkage clustering of milk data
milk.scaled <- scale(milk)
d <- dist(milk.scaled)
fit.average <- hclust(d, method="average")
plot(fit.average, main="Average Linkage Clustering",hang = -1)
rect.hclust(fit.average,h = 4)

plot(fit.average,main="Average Linkage Clustering\n3 Cluster Solution")
rect.hclust(fit.average, k=2)

## Coloured Dendrogram ###
# library(colorhcplot)
# colorhcplot(fit.average,fac = factor(clusterID))

################# Kmeans ####################
k4<-kmeans(milk.scaled, centers=4)
k4$tot.withinss

k5<-kmeans(milk.scaled, centers=5)
k5$tot.withinss

k6<-kmeans(milk.scaled, centers=6)
k6$tot.withinss

# Plot function for within groups sum of squares by number of clusters
wssplot <- function(data, nc=15, seed=1234) {
  wss <- array(dim=c(nc))
  for (i in 2:nc){
    set.seed(seed)
    km <- kmeans(data, centers=i)
    wss[i] <- km$tot.withinss
    }
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")
}

wssplot(milk.scaled)

k4<-kmeans(milk.scaled, centers=4)

k4clust <- data.frame(milk , ClusterID = k4$cluster)

pairs(k4clust[,-6],col= k4clust$ClusterID)

## Coloured Dendrogram ###
library(colorhcplot)
colorhcplot(fit.average,fac = factor(k4clust$ClusterID))
