from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.pdfgen import canvas

# متن‌ها
text1 = "available quality program free color version gif file image jpeg"
text2 = "ha article make know doe say like just people think"
text3 = "include available analysis user software ha processing data tool image"
text4 = "atmosphere kilometer surface ha earth wa planet moon spacecraft solar"
text5 = "communication technology venture service market ha commercial space satellite launch"
text6 = "verse wa jesus father mormon shall unto mcconkie lord god"
text7 = "format message server object image mail file ray send graphic"
text8 = "christian people doe atheism believe religion belief religious god"
text9 = "file graphic grass program ha package ftp available image data"

# ساخت یک لیست از متن‌ها
documents = [text1, text2, text9, text8, text7, text6, text5, text4, text3]

# تعداد تاپیک‌ها برای آزمایش
num_topics = [2, 3, 4, 5]

for n in num_topics:
    # ساخت مدل TF-IDF با تعداد تاپیک مشخص
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)

    # ساخت مدل NMF با تعداد تاپیک مشخص
    model = NMF(n_components=n, random_state=42)
    model.fit(X)

    # نمایش موضوعات برای هر تعداد تاپیک
    print(f"Number of Topics: {n}")
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx}:")
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topic_sentence = " ".join(topic_words)
        print(topic_sentence)

    # ایجاد یک فایل PDF جدید
    pdf_filename = f"topics_{n}.pdf"
    c = canvas.Canvas(pdf_filename)

    # نوشتن نتایج در فایل PDF
    y = 700
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Number of Topics: {n}")
    y -= 20

    for idx, topic in enumerate(model.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
        topic_sentence = " ".join(topic_words)
        c.setFont("Helvetica", 12)
        c.drawString(70, y, f"Topic {idx}: {topic_sentence}")
        y -= 15

    # ذخیره کردن فایل PDF
    c.save()

    print(f"Results for {n} topics saved to topics_{n}.pdf")
