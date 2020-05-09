data(milk,package="flexclust")
prc <- prcomp(milk,scale. = T)
names(prc)
prc$sdev
prc$rotation
prc$center
prc$scale
prc$x # scale(milk,scale = F) %*% prc$rotation

pr.var <- prc$sdev^2
pve <- pr.var / sum(pr.var)
# Plot variance explained for each principal component
plot(pve, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")

# Plot cumulative proportion of variance explained
plot(cumsum(pve), xlab = "Principal Component",
     ylab = "Cummulative Proportion of Variance Explained",
     type = "b")

biplot(prc,cex=0.7,scale = 0)

# prc$rotation %*% diag(prc$sdev^2) %*% t(prc$rotation)
# ### Similar to the following ###
# cov(scale(milk,scale=F))

##apply(prc$rotation,2,sd)
#############################
library(ggplot2)
library(ggfortify)
autoplot(prc, data = milk,
         loadings = TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 3,
         label=T)

