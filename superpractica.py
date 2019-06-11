# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.style.use('fivethirtyeight')

####    Dades d'origen - Kaggle []

dfpre = pd.read_csv('/Users/xavier/Practica supermercats/dades-supermercat.csv') #Per si hem d'accedir al df sense processar
dfpre = dfpre.rename(columns={'Gender': 'gender', 'Age': 'age', 'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending_score'})

df = pd.read_csv('/Users/xavier/Practica supermercats/dades-supermercat.csv', index_col=0) #Com que customer ID és un acumulador de valors únics podem contemplar-lo com un índex

#0---   Preparem les dades

####    Per usabilitat simplifiquem els títols de les columnes, estandaritzem per minúscules sense caràcters especials ni espais.
####    Per reduir error i possibles confusions durant la definició i revisions mantindrem l'idioma anglès que defineix les variables al dataset d'origen.

df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Annual Income (k$)': 'annual_income', 'Spending Score (1-100)': 'spending_score'})
df['gender'] = df['gender'].factorize()[0]

df.head()

####    Hem convertit la variable Gènere a numèrica per tal de treballar-la en alguns processos
####    No obstant mantenim la variable dfpre ja que facilita certes visualitzacions.

df.describe()
df.groupby('gender').mean()


#1---   Dibuixem els gràfics de densitat per identificar la naturalesa
#       de la distribució
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
sns.distplot(df['age'])
plt.title("Distribució de l'edat")
plt.subplot(1,3,2)
sns.distplot(df['spending_score'],hist=False)
plt.title('Distribució de despesa')
plt.subplot(1,3,3)
sns.distplot(df['annual_income'])
plt.title('Distribució de ingressos anuals')
plt.show()

####   Observem que el volum d'ingressos no té perquè estar directament relacionat
####   amb el volum de despesa, p.ex: el grup genere = 0 (dones) té menor volum
####   d'dingressos que el grup = 1 (homes) i no obstant té major puntuació en despesa
####
####   Veiem que no segueixen els patrons d'una distribució normal
####   sinó que mostren una lleugera desviació cap a l'esquerra.
####


#2---   Dibuixem la matriu de correlacions i els gràfics bivariants per identificar si hi ha algun patró que les relacioni

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,cmap='ocean',fmt='.2f',linewidths=2)
plt.show()

sns.pairplot(data=df)


####    Observem possibles correlacions entre la despesa anual i els ingressos
#3---   Dibuixem les variables visualment separades per gènere que tenen el pes repartit de la següent manera

print(dfpre.gender.value_counts())

plt.figure(figsize=(5,5))
plt.pie(dfpre.gender.value_counts().values, labels=['Dones', 'Homes'],explode =[0,0.1])
plt.show()


####    a. Diferència de gènere segons ingressos
plt.figure(figsize=(14,3))
sns.barplot(y='annual_income', x='age',hue='gender',data=dfpre,errwidth=0.5)
plt.legend(title='Gènere', loc='upper left')
plt.show()


####    b. Gràfics bivariants amb desglòs per gènere
plt.figure(1 , figsize = (13 , 7))
n = 0
for cols in ['age' , 'annual_income' , 'spending_score']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    sns.violinplot(x = cols , y = 'gender' , data = dfpre , palette = 'vlag')
    sns.swarmplot(x = cols , y = 'gender' , data = dfpre)
    plt.ylabel('gender' if n == 1 else '')
    plt.title('Diagrames de caixa i de violí' if n == 2 else '')
plt.show()


####    c. Gràfics bivariants amb desglòs per gènere
plt.figure(figsize=(14,3))
plt.subplot(1,3,1)
for gender in [1,0]:
    plt.scatter(x = 'age' , y = 'annual_income' , data = df[df['gender'] == gender] ,label = gender,s=50)
    plt.title('Edat vs Ingressos')
plt.subplot(1,3,2)
for gender in [1,0]:
    plt.scatter(x = 'annual_income',y = 'spending_score' ,data = df[df['gender'] == gender], label = gender,s=50)
    plt.title('Ingressos vs Despesa')

plt.legend(labels=['Dones', 'Homes'], bbox_to_anchor=(0.2,-0.1 ), ncol=2)
plt.show()



####    Veiem clarament comportaments agrupables entre les variables Ingressos i Despesa i no semblen no estar condicionades al gènere
####    ... potser aquesta és més explícita que una altra
with sns.axes_style("darkgrid"):
    sns.jointplot(x="annual_income", y = "spending_score", kind = "kde", data =dfpre)

####    El fet que el gènere no estigui condicionant les agrupacions de dades fa que l'anàlisi sigui més objectiu.


#4---   Calculem les components principals, més que res per practicar ja que hi ha prou variable si semblen ja bastant independents entre elles.
from sklearn.decomposition import PCA
pca = PCA(n_components=4) 
pca.fit(df)
pca.explained_variance_ratio_ #Variança explicada. -La documentació diu: <<The pca.explained_variance_ratio_  parameter returns a vector of the variance explained by each dimension>>

pca.components_  #amb aquest obtenim els vectors propis de cada valor.

### Creem un explicatiu dinàmic al llarg de les components +1 per encaixar-ho amb l'eix
dimensions = ['Dimensió {}'.format(i) for i in range(1,len(pca.components_)+1)] 
### Creem dataframe amb les components (Dimensió 1, Dimensió 2...)
components = pd.DataFrame(pca.components_,columns=df.columns)   
components.index = dimensions 
### Creem dataframe amb la variança explicada per cada component (Dimensió 1, Dimensió 2...)
variansa = pd.DataFrame(pca.explained_variance_ratio_, columns=['Explained Variance']) #Creem un altre dataframe amb els valors 
variansa.index = dimensions
### Visualitzem cada component generada i la variança explicada en les n Dimensions)
acp = pd.concat([variansa,components], axis=1)

plt.plot(variansa.sort_values(by='Explained Variance',ascending=False))
print(acp)
####    Com que la variança queda explicada per les dues components principals ~ 0.9(0.45+0.44) podem dir que és suficient utilitzar les dues primeres.

#5---   Mirem d'identificar grups de dades

## Triem les variables del model
#### Tenint en conpte 2 variables 'annual_income' i 'spending_score'
X=df.iloc[:, [2,3]].values
#### Constriuïm el model utilitzant el mètode dels k means per identificar el nombre de clusters. 
from sklearn.cluster import KMeans
sumaquadrats=[]


#### Fem un loop per tal que apli¡qui el càlcul per obtenir suma de quadrats dels kmeans de les dades seleccionades
#### i així poder crear de cop l'objecte 'sumaquadrats' per cada k que li proposem, en aquest cas 10 (se'n poden posar més i tot no afecta).

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    sumaquadrats.append(kmeans.inertia_)  #inertia_ s'utilitza per segregar les dades en clusters (és com si apliqués acceleració) aplicantla suma dels quadrats wss

#### Visualitzem el gràfic de colze  (elbow)
plt.plot(range(1,11), sumaquadrats)
plt.title('Mètode "elbow"')
plt.xlabel('número de clusters')
plt.ylabel('suma dels quadrats entre k clusters')
plt.show()

#### Veiem que la distància entre k clusters decreix de cop en k=5 

#Construim el model 
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)

####Aprenentatge no supervisat "fit_predict()" 
####Per aprenentatge supervisat "fit_tranform()"
####y_kmeans és el model final. Now how and where we will deploy this model in production is depends on what tool we are using.
####Mètode molt utilitzat per serveis financers (targetes de crèdit) i per segmentació d'usuaris.

# Dibuixem els clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Clúster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Clúster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Clúster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Clúster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Clúster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroides')
plt.title('Grups de clients segons ingressos anuals')
plt.xlabel('Ingressos anuals (milers de $)')
plt.ylabel('Puntuació de despesa (1-100)')
plt.legend()
plt.show()

#Els centroides són els centres de gravetat -per dir-ho d'alguna manera- de les dades de cada cluster
### Interpretació del model
##Clúster 1 (Vermell) -> guanyen més però gasten menys.
##Clúster 2 (Blau) -> als valors mig de cada eix, ni guanya molt i poc ni gasta molt o poc. 
##Clúster 3 (Verd) -> guanyen molt i gasten molt [Si s'ha de prioritzar algun grup de clients és aquest]
##Clúster 4 (Blau clar) -> guanyen molt però gasten molt poc.
##Clúster 5 (Rosa) -> Quasi no guanyen ni gasten.


#### Repetim el procés però contrastant 'edat' i 'despesa'
x = df.iloc[:, [1,3 ]].values 
sumaquadrats = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state = 0)
    kmeans.fit(x)
    sumaquadrats.append(kmeans.inertia_)  #inertia_ s'utilitza per segregar les dades en clusters (és com si apliqués acceleració) aplicantla suma dels quadrats wss

#### Visualitzem el gràfic de colze  (elbow)
plt.plot(range(1,11), sumaquadrats)
plt.title('Mètode "elbow"')
plt.xlabel('número de clusters')
plt.ylabel('suma dels quadrats entre k clusters')
plt.show()

#### En aquest cas la distància entre k clusters decreix de cop en k=4

# Dibuixem els clusters per a aquestes dades
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ymeans = kmeans.fit_predict(x)

plt.rcParams['figure.figsize'] = (8, 8)
plt.title('Grups de clients segons edat', fontsize = 30)

plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'Clients habituals' )
plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Clients molt fidels')
plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Clients objectiu (Joves)')
plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Clients objectiu (Sènior)')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow')

plt.xlabel('Edat')
plt.ylabel('Puntuació de despesa (1-100)')
plt.legend()
plt.show()


