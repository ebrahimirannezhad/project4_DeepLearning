from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from reportlab.pdfgen import canvas
import random

# بارگیری داده‌ها
data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
documents = data.data

# پیکربندی بردارساز CountVectorizer
vectorizer = CountVectorizer(max_features=1000, max_df=0.5, min_df=2, stop_words='english')

# تبدیل متن به بردارهای تعداد تکرار کلمات
X = vectorizer.fit_transform(documents)

# تعداد تاپیک‌ها برای آزمایش
num_topics = [5, 10, 15, 20]

for n in num_topics:
    # ساخت مدل LDA با تعداد تاپیک مشخص
    model = LatentDirichletAllocation(n_components=n, random_state=42)
    model.fit(X)

    # نمایش موضوعات برای هر تعداد تاپیک
    print(f"Number of Topics: {n}")
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        top_words_indices = topic.argsort()[-10:]
        top_words = [feature_names[i] for i in top_words_indices]
        print(f"Top Words: {', '.join(top_words)}")
        random_sentence = ' '.join(random.choices(top_words, k=5))
        print(f"Random Sentence: {random_sentence}")
        print()

# ایجاد یک فایل PDF جدید
pdf_filename = "topics_Exercise3.pdf"
c = canvas.Canvas(pdf_filename)

# نوشتن نتایج در فایل PDF
y = 700
for n in num_topics:
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Number of Topics: {n}")
    y -= 20

    for topic_idx, topic in enumerate(model.components_):
        top_words_indices = topic.argsort()[-10:]
        top_words = [feature_names[i] for i in top_words_indices]
        topic_text = f"Topic {topic_idx}: {', '.join(top_words)}"
        c.setFont("Helvetica", 12)
        c.drawString(70, y, topic_text)
        random_sentence = ' '.join(random.choices(top_words, k=5))
        c.drawString(70, y - 15, f"Random Sentence: {random_sentence}")
        y -= 30

    y -= 20

# ذخیره کردن فایل PDF
c.save()

print("Results saved to topics_Exercise3.pdf")
