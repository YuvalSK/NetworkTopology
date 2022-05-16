# after calculating the correlation matrix, in this code we reduce its dimensions via PCA
# our goal here is to find the maximun topological variance of the 20-D correlation matrix over the 94 most popular DOIs in LJ

from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib import colors as mcolors

plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (12,10)

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
k=1 #statistic SEM (vs. SD)
snaps=[19]
doi2NLP = pd.read_csv('Data/dictionaries/NLP/‏‏doi2personality.csv') # to associated with NLP
doi2Dep = pd.read_csv('Data/dictionaries/depression/doi2dep.csv') # to associated with depression
#doi2met = pd.read_csv('Data/dictionaries/meta/doi2meta.csv') # to associated with meta traits 
#doi2per = pd.read_csv('Data/dicrionaries/bigfive/doi2big5.csv') # to associated with big five traits 
#doi2gen = pd.read_csv('Data/dictionaties/gender/doi2gender.csv') # to associated with gender 

# PCA Analysis
def s_PCA(snap):
    print(f'Loading PCA on snapshot: {snap}')
    plt.clf()
    dataset = pd.read_csv('Data/snap_{}_dois_100.csv'.format(snap),index_col=0)    

    #implementing PCA 
    norm_data = preprocessing.scale(dataset)
    pca = PCA()
    pca.fit(norm_data)
    pca_data = pca.transform(norm_data)
    
    #exploring results
    exp_var = np.round(pca.explained_variance_ratio_ *100, decimals=1)
    features_w = pca.components_
    labels = ['PC' + str(n) for n in range(1,len(exp_var)+1)]       
    pca_df = pd.DataFrame(pca_data, columns=labels,index=dataset.index)
    pca_df.to_csv(f'Data/pca{snap}_dep.csv')
    #print(np.sum(features_w[1]**2)) # weights^2 should sum up to 1
    
    #present in 2-D space - PC1 and PC2
    plt.scatter(pca_df.PC1,pca_df.PC2, facecolors='none', edgecolors='gray')
    plt.xlabel(f'PC1 [{exp_var[0]}%]')
    plt.ylabel(f'PC2 [{exp_var[1]}%]')
    
    #highlight the most popular DOIs in this 2-D space
    dois_list = ['music','movies','reading','writing','art','books','photography', 'friends']
    for doi in pca_df.index:
        if doi in dois_list:
            plt.annotate(doi,(pca_df.PC1.loc[doi], pca_df.PC2.loc[doi]),fontsize=16,c='b')
        else:
            plt.annotate(doi,(pca_df.PC1.loc[doi], pca_df.PC2.loc[doi]),fontsize=12)

    plt.title(f'DOIs over PC space snapshot:{snap}')
    plt.tight_layout()
    plt.xlim(-10,10)
    plt.ylim(-5,15)
    plt.savefig(f'Results/PCA/DOIs_{snap}',dpi=300)
    
s_PCA(19)

#how the DOIs span in PC 2-D space by clusters:
## 1) NLP:
'''
pos = doi2NLP[doi2NLP.Per =='low E']['doi'].tolist()
neg = doi2NLP[doi2NLP.Per =='low N']['doi'].tolist()
'''

## 2) depression:
pos = doi2dep[doi2Dep.Dep =='High']['DOIs'].tolist()
neg = doi2dep[doi2Dep.Dep =='Low']['DOIs'].tolist()


def DOIs2Attribute(pos_dois,neg_dois):
    for doi in pca_df.index:
        if doi in pos_dois:
            plt.annotate(doi,(pca_df.PC1.loc[doi], pca_df.PC2.loc[doi]),fontsize=12,xytext=(-40,-40), 
                textcoords='offset points', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='g', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='k'))
        elif doi in neg_dois:
            plt.annotate(doi,(pca_df.PC1.loc[doi], pca_df.PC2.loc[doi]),fontsize=12,xytext=(-40,-40), 
                textcoords='offset points', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='r', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='k'))
            
    plt.title(f'DOIs over PC space snapshot:{snap}')
    plt.tight_layout()
    plt.show()
    #plt.savefig(f'Results/NLP/DOIs_dep{snap}',dpi=300)

plt.scatter(pca_df.PC1,pca_df.PC2, facecolors='none', edgecolors='gray')
DOIs2Attribute(pos, neg)

'''
###### DOIs in features space ######
plt.clf()
fig, ax = plt.subplots(figsize=(12,10))
x_axis_labels =  [str(f'$\it{c}$') for c in dataset.columns[0:]]
x = np.arange(len(x_axis_labels))

#sort features by PC weights
y, sorted_labels = zip(*sorted(zip(features_w[0], x_axis_labels), reverse=True))  
ax.set_ylabel('PC1 weights [AU]')
ax.set_xlabel('NAV features')
ax.set_xticks(x)
ax.set_xticklabels(sorted_labels, rotation=45,fontsize=12,ha="right",)
ax.bar(x,y,color='gray')
ax.axvline(4.5,color='k')
ax.axhline(0,color='k')
ax.set_ylim(-0.1, 0.3)
ax.set_yticks(np.arange(-0.1, 0.3, step=0.1))
fig.align_labels()
#plt.savefig(f'Results/PC1Depfeatures_{snap}',dpi=150)
    
#How depression span on PC topological space
plt.clf()
fig, ax = plt.subplots(2,1,figsize=(12,10)) 
ax[0].set_title(f'Detailed PC space snapshot:{snap}')
ax[0].scatter(pca_df.PC1,pca_df.PC2, facecolors='none', edgecolors='gray')
ax[1].scatter(pca_df.PC1,pca_df.PC3, facecolors='none', edgecolors='gray')

ax[1].set_xlabel(f'PC1 ({exp_var[0]}%) [AU]')
ax[0].set_ylabel(f'PC2 ({exp_var[1]}%) [AU]')
ax[1].set_ylabel(f'PC3 ({exp_var[2]}%) [AU]')

c = ['r', 'b']
bs=['High','Low']
for i,b in enumerate(bs):
    x = []
    y= []
    for doi in doi2dep['DOIs']:
        if doi2dep.loc[doi2dep['DOIs'] == doi,'Dep'].iloc[0] == b:
            x.append(pca_df.PC1.loc[doi])
            y.append(pca_df.PC2.loc[doi])
    
    xerror = k * np.std(x,ddof=1) / np.sqrt(len(x))
    yerror = k * np.std(y,ddof=1) / np.sqrt(len(y))        

    #ax[0].annotate(b,(np.mean(x), np.mean(y)),color=colors[c[i]],label=b)
    ax[0].errorbar(np.mean(x), np.mean(y), xerr=xerror, yerr=yerror, capsize=3, ls='none', color=c[i], elinewidth=4, label=b)
    ax[0].scatter(x,y, facecolors='none', edgecolors=c[i])

    print('PC1 vs. PC2:')
    print(f'{b}: mean:{np.mean(x):.3f}-{np.mean(y):.3f}, SEM:{xerror:.3f}-{yerror:.3f}')
    
ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-15, 15)
for i,b in enumerate(bs):
    x = []
    y= []
    for doi in doi2dep['DOIs']:
        if doi2dep.loc[doi2dep['DOIs'] == doi,'Dep'].iloc[0] == b:
            x.append(pca_df.PC1.loc[doi])
            y.append(pca_df.PC3.loc[doi])
    
    xerror = k * np.std(x,ddof=1) / np.sqrt(len(x))
    yerror = k * np.std(y,ddof=1) / np.sqrt(len(y))        

    #ax[0].annotate(b,(np.mean(x), np.mean(y)),color=colors[c[i]],label=b)
    ax[1].errorbar(np.mean(x), np.mean(y), xerr=xerror, yerr=yerror, capsize=3, ls='none', color=c[i], elinewidth=4, label=b)
    ax[1].scatter(x,y, facecolors='none', edgecolors=c[i])

    print('PC1 vs. PC3:')
    print(f'{b}-mean:{np.mean(x):.3f}-{np.mean(y):.3f}, SEM:{xerror:.3f}-{yerror:.3f}')

#ax[1].legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)
ax[1].set_xlim(-10, 10)
ax[1].set_ylim(-3, 5)
ax[1].set_xticks(np.arange(-10, 12, step=2))
ax[0].set_xticks(np.arange(-10, 12, step=2))

#ax[1].set_aspect('equal')
#ax[0].set_aspect('equal')  
#plt.legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)

plt.tight_layout()
plt.savefig(f'Results/PCA_dep{snap}')



# # Visualization of closeness and motif3_8  
snaps=[19]
k=1
plt.rcParams.update({'font.size': 18})

for snap in snaps:
    plt.clf()
    dataset = pd.read_csv('Data/snap_{}_dois_100.csv'.format(snap),index_col=0)
    ###### Fetures and DOIs associations ######
    dataset.head()
    plt.scatter(dataset.closeness,dataset.motifs3_8, facecolors='none', edgecolors='gray')
    plt.xlabel('$\it{Closeness}$ [R]')
    
    '$\it{Motif3-8}$'
    
    plt.ylabel('$\it{Motif3-8}$ [R]')
    
    colors = ['r', 'b']
    bs=['High','Low']
    for i,b in enumerate(bs):
        x = []
        y= []
        for doi in doi2dep['DOIs']:
            if doi2dep.loc[doi2dep['DOIs'] == doi,'Dep'].iloc[0] == b:
                x.append(dataset.closeness.loc[doi])
                y.append(dataset.motifs3_8.loc[doi])
        
                
        xerror = k * np.std(x,ddof=1) / np.sqrt(len(x))
        yerror = k * np.std(y,ddof=1) / np.sqrt(len(y))
        
        #ax[0,0].annotate(b,(np.mean(x), np.mean(y)),color=colors[i],fontsize=22,xytext=(-20,20), textcoords='offset points', ha='center', va='bottom',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='k'))
        plt.errorbar(np.mean(x), np.mean(y), xerr=xerror, yerr=yerror, capsize=3, ls='none', color=colors[i], elinewidth=4,label=b)
        plt.scatter(x,y, facecolors='none', edgecolors=colors[i])
        #print(f'calculating mean+-SEM of {b}')
        #print(f'x: {np.mean(x):.3f}±{xerror:.3f}, y:{np.mean(y):.3f}±{yerror:.3f}')

    plt.xlim(0, 0.13)
    plt.xticks(np.arange(0, 0.13, step=0.04))
    plt.ylim(0, 0.13)
    plt.yticks(np.arange(0, 0.13, step=0.04))
    
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g'))

    
    #np.set_printoptions(precision=1)
    #plt.legend(bbox_to_anchor=(1.00, 1), loc=3, borderaxespad=0.3)
    plt.tight_layout()
    
    plt.savefig(f'Results/features_{snap}')



data = np.array([pca_df.PC1.values,pca_df.PC2.values])
'''

def GM(k,data):
    gm = GaussianMixture(n_components = k).fit(data)
    labels = gm.predict(data)
    return data, labels, gm

fig, axs = plt.subplots(2, 2)
data1, labels1, gm1 = GM(2,data)
axs[0, 0].scatter(data1[:,0], data1[:,1], c=labels1, s=40, cmap='viridis');
axs[0, 0].scatter(gm1.means_[:,0], gm1.means_[:,1], linewidths=4)
axs[0, 0].set_title('K = 2')

data2, labels2, gm2 = GM(2,data)
axs[0, 1].scatter(data2[0], data2[1], s=40, cmap='viridis');
axs[0, 1].scatter(gm2.means_[:,0], gm2.means_[:,1], linewidths=4)
axs[0, 1].set_title('K = 2')

data3, labels3, gm3 = GM(2,data)
axs[1, 0].scatter(data3[0], data3[1], s=40, cmap='viridis');
axs[1, 0].scatter(gm3.means_[:,0], gm3.means_[:,1], linewidths=4)
axs[1, 0].set_title('K = 2')

data4, labels4, gm4 = GM(2,data)
axs[1, 1].scatter(data4[0], data4[1], s=40, cmap='viridis');
axs[1, 1].scatter(gm4.means_[:,0], gm4.means_[:,1], linewidths=4)
axs[1, 1].set_title('K = 2')

for ax in axs.flat:
    ax.set(xlabel='PC1', ylabel='PC2')
    ax.set(ylim = (-10, 10))
    ax.set(xlim = (-10, 10))
plt.tight_layout()

'''
# plot GMM when K=3
data, labels, gm = GM(3)
plt.scatter(data[:, 0], data[:, 1], c=labels, s=40, cmap='viridis');
plt.scatter(gm.means_[:,0], gm.means_[:,1], linewidths=4)
plt.xlabel("X")
plt.ylabel("Y")


# print the Distribution parameters
print("Distribution parameters \n")

# plt.figure()
for i in range(3):
  # calculate the mean var
  mean = gm.means_[i]
  var = np.diagonal(gm.covariances_[i])
  # print(var)
  sigma = math.sqrt(var[0])
  # print(sigma)
  print("clusters num: " + str(i + 1))
  print("Mean: " + str(mean))
  print("Vars: " + str(var))

# get the probability of the pred for a soft clustering data point.
# we ploted the k=3 graph and chose a point between 2 clusters: (2.127, -1.88)
pred = gm.predict_proba(np.array([2.127, -1.88]).reshape(1,-1))
probabilities = pred[0]
print("first cluster probability: " + str(probabilities[0]))
print("second cluster probability: " + str(probabilities[1]))
print("third cluster probability: " + str(probabilities[2]))
sum = probabilities.sum()
print("sum of probabilities = " + str(sum))
'''


# In[ ]:


'''
    dois_list = ['sleeping','love','harry potter']
    for doi in dataset.index:
if doi=='sleeping':
    plt.annotate(doi,(dataset.closeness_1.loc[doi], dataset.motifs_3_8.loc[doi]),fontsize=12,xytext=(-20,20), 
        textcoords='offset points', ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='r'))
elif doi=='love':
    plt.annotate(doi,(dataset.closeness_1.loc[doi], dataset.motifs_3_8.loc[doi]),fontsize=12,xytext=(-20,20), 
        textcoords='offset points', ha='center', va='bottom',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='b'))
elif doi=='harry potter':
    plt.annotate(doi,(dataset.closeness_1.loc[doi], dataset.motifs_3_8.loc[doi]),fontsize=12,xytext=(-20,30), 
        textcoords='offset points', ha='center', va='bottom',bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='k'))
    '''
    
    
    '''
    c = ['b','midnightblue','red', 'darkred','indianred']
    bs = ['Extraversion','Openness to Exp.','Conscientiousness','Agreeableness','Neuroticism']
    
    for i, b in enumerate(bs):
x = []
y= []
for doi in doi2per['DOIs']:
    if doi2per.loc[doi2per['DOIs'] == doi,'Traits'].iloc[0] == b:
        x.append(pca_df.PC1.loc[doi])
        y.append(pca_df.PC3.loc[doi])
        
xerror = k * np.std(x,ddof=1) / np.sqrt(len(x))
yerror = k * np.std(y,ddof=1) / np.sqrt(len(y))

#ax[1].annotate(b,(np.mean(x), np.mean(y)),color=colors[c[i]],label=b)
ax[1].errorbar(np.mean(x), np.mean(y), xerr=xerror, yerr=yerror, capsize=3, ls='none', color=colors[c[i]], elinewidth=4, label=b)

#print('----PC1 vs. PC3: personality traits----')
#print(f'{b}-mean PCA1:{np.mean(x):.3f}-PCA3{np.mean(y):.3f}\nSEM:{xerror:.3f}-{yerror:.3f}')
    '''   
    
    
    '''
    ###### Gender and DOIs associations ######
    plt.clf()
    #plt.scatter(pca_df.PC1,pca_df.PC2, facecolors='none', edgecolors='k')
    plt.xlabel(f'PC1 - {exp_var[0]}%')
    plt.ylabel(f'PC2 - {exp_var[1]}%')
    
    colors = ['g', 'orange']
    for i,b in enumerate(set(doi2gen['Gender'])):
x = []
y= []
for doi in doi2gen['DOIs']:
    if doi2gen.loc[doi2gen['DOIs'] == doi,'Gender'].iloc[0] == b:
        x.append(pca_df.PC1.loc[doi])
        y.append(pca_df.PC2.loc[doi])
        
xerror = k * np.std(x,ddof=1) / np.sqrt(len(x))
yerror = k * np.std(y,ddof=1) / np.sqrt(len(y))
    
#plt.annotate(b,(np.mean(x), np.mean(y)),color=colors[i],label=b)
plt.errorbar(np.mean(x), np.mean(y), xerr=xerror, yerr=yerror, capsize=3, ls='none', color=colors[i], elinewidth=4,label=b)

#print('----PC1 vs. PC2:gender----')
#print(f'{b}-mean:{np.mean(x):.3f}-{np.mean(y):.3f}, SEM:{xerror:.3f}-{yerror:.3f}')
            
    plt.xlim(-4, 3)
    plt.ylim(-2, 2)
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f'Results/DOIs_Gender_{snap}')
    #pca_df.to_csv('pcadata.csv')
    '''

