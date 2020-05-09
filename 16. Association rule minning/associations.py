import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

fp_df = pd.read_csv('F:/DatasetsSmall/Association Rules/Faceplate.csv')
fp_df.set_index('Transaction', inplace=True)
fp_df

fp_df.head()

itemFrequency = fp_df.sum(axis=0) / len(fp_df)

# and plot as histogram
ax = itemFrequency.plot.bar(color='blue')
plt.ylabel('Item frequency (relative)')
plt.show()


# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2, use_colnames=True)

# and convert into rules
rules = association_rules(itemsets, metric='confidence', min_threshold=0.5)
rules.sort_values(by=['lift'], ascending=False).head(6)


rule_df = rules.sort_values(by=['lift'], ascending=False)

print(rule_df[['antecedents','consequents','support','confidence','lift']])

