from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# بارگیری داده‌های گروه‌های خبری
newsgroups = fetch_20newsgroups(subset='all')

# استخراج ویژگی‌ها با استفاده از tf-idf
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(newsgroups.data)

# تعداد خوشه‌ها مختلف
k_values = range(2, 10)

# آرایه‌ای برای نگهداری مقادیر inertia برای هر مقدار k
inertias = []

# اجرای خوشه‌بندی برای هر مقدار k و محاسبه مقدار inertia
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(tfidf_matrix)
    inertias.append(kmeans.inertia_)

# نمایش نمودار Elbow
plt.plot(k_values, inertias, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()