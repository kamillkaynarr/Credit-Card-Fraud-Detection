import pandas as pd                                            # Veri 
import matplotlib.pyplot as plt                                # Grafik
from sklearn.model_selection import train_test_split           # Model
from sklearn.preprocessing import normalize, StandardScaler    # Ölçeklendirme
from sklearn.utils.class_weight import compute_sample_weight   # Sınıf dengesizliği için
from sklearn.tree import DecisionTreeClassifier                # Karar ağacı sınıflandırıcısı
from sklearn.metrics import roc_auc_score                      # Olasılık tahmini
from sklearn.svm import LinearSVC                              # Lineer çekirdek | Destek vektör makineleri
from sklearn.preprocessing import StandardScaler               # Standard Scaler

#-----------VERİ YÜKLEME--------------------------------------------------#

url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

data=pd.read_csv(url)
print(data.sample(12))
#print("OKUMA BAŞARILI")

# 0 ----- LEGAL İŞLEM
# 1 ----- İLLEGAL İŞLEM

#-----------VERİNİN ANALİZİ-----------------------------------------------#

gd = data['Class'].unique()
gd_d = data['Class'].value_counts().values  

                   #---------PASTA GRAFİĞİ ÇİZDİK
fig, ax = plt.subplots()

ax.pie(gd_d, labels=gd, autopct='%1.3f%%')
ax.set_title('VERİ | ŞÜPHELİ İŞLEMLERİN İSTATİKSEL DAĞILIMI')
plt.show()

#----------KORELASYON İNCELEMESİ-------------------------------------------#

correlation = data.corr()['Class'].drop('Class')

correlation.plot(kind='barh', figsize=(1, 3))

plt.title("Korelasyon Grafiği")
plt.xlabel("Korelasyon Değeri")
plt.ylabel("Özellikler")

plt.show()       #----SADECE KOLERASYONU YÜKSEK OLAN ÖZELLİKLERİ ALABİLİRİZ

#-----------EĞİTİM ÖNCESİ HAZIRLIK-----------------------------------------#

data_new = data.drop(['Time', 'Class'], axis=1)     #--------- Zaman serisi ve sonuç sütunu' nu çıkardık

s_data = StandardScaler().fit_transform(data_new)   #--------- Sütun bazlı Ölçeklendiriyoruz

y = data['Class'].values                            #--------- Tahmin etmeye çalışacağımız veri sütunu
 
x = s_data 
x = normalize(x, norm="l1")   #------------------------------- Satır bazlı Ölçeklendiriyoruz


print("x: ", x)
print("y: ", y)

#------------------------------EĞİTİM-----------------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

k_train = compute_sample_weight('balanced', y_train)                   #Sonuç Özelliğindeki dengesizliğini bastırabilmek için ağırlık
                                                                       #katsayı ataması yapılıyor. Böylece azınlığa daha çok dikkat ediyoruz

        #------------ML Nesneleri 
dtree = DecisionTreeClassifier(max_depth=4, random_state=19)

dtree.fit(X_train, y_train, sample_weight=k_train)

svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

svm.fit(X_train, y_train)

        #------------ML Nesneleri 

y_p_dt = dtree.predict_proba(X_test)[:, 1]
roc_dt = roc_auc_score(y_test, y_p_dt)
print("Karar Ağacı Başarısı (ROC-AUC): {0:.3f}".format(roc_dt))
                                                          
y_p_dsvm = svm.decision_function(X_test)    

roc_dsvm = roc_auc_score(y_test, y_p_dsvm)
print('LinearSVC Başarısı : {0:.3f}'.format(roc_dsvm))    #Modelimizin dolandırcılığı gerçekten  doğru tahmin 
                                                                      #edebiliyor mu?


                                             
