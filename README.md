# 💬 FAQ Chatbot

A simple **FAQ-based Chatbot** built with **Python (Tkinter)** and **NLP techniques**.  
The chatbot answers user questions by matching them with the most similar FAQ from a dataset (`FAQ.csv`) using **TF-IDF** and **cosine similarity**.

---

## ✨ Features
- Interactive **chat-like GUI** using Tkinter.
- Reads **FAQs from a CSV file** (`FAQ.csv`).
- Preprocesses text with **NLTK** (tokenization, stopword removal, etc.).
- Finds the **most relevant answer** using TF-IDF and cosine similarity.
- Easy to extend — just add more FAQs in the CSV file.

---

## 🛠️ Technologies Used
- **Python 3.x**
- **Tkinter** → GUI framework
- **NLTK** → Tokenization & text preprocessing
- **Scikit-learn** → TF-IDF vectorization & cosine similarity
- **Pandas** → CSV handling

⚠️ Notes

The chatbot answers only from the FAQs provided in FAQ.csv.

If a question is very different from the ones in the CSV, it may give the closest match (not always perfect).

You can expand the knowledge base by adding more rows in FAQ.csv.

👨‍💻 Author

Developed by Maryam Ijaz as part of the CodeAlpha Internship.



## 📂 Project Structure
