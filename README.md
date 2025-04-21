# **Sentiment Analysis Using TF-IDF & Logistic Regression**

**Company**: CODTECH IT SOLUTION  

**Intern Name**: Shreyan Nandanwar  

**Intern ID**: CT04DA921  

**Domain**: Machine Learning  

**Duration**: 4 Weeks  

**Mentor**: Neela Santosh  

---

## **Overview**
This project focuses on building a **sentiment analysis model** to classify customer reviews as positive or negative. As a beginner in machine learning and natural language processing (NLP), I explored two foundational techniques: **TF-IDF vectorization** for text feature extraction and **Logistic Regression** for classification. This project provided hands-on experience with data preprocessing, model training, evaluation, and visualization.

---

## 🧰 **Tools, Platform, and Environment**
### **Development Setup**
- **Editor**: Jupyter Notebook (for interactive coding and visualization)  
- **Platform**: Anaconda Navigator (for managing Python environments and dependencies)  
- **Code Editor**: VS Code (used occasionally for cleaner interfaces and multi-file workflows)  

### **Libraries Used**
- `pandas` and `numpy`: For data manipulation and numerical operations.  
- `scikit-learn`: For TF-IDF vectorization, logistic regression, and model evaluation.  
- `matplotlib` and `seaborn`: For visualizing performance metrics like confusion matrices.  

---

## 📁 **Dataset**
- **Source**: Downloaded from [Kaggle](https://www.kaggle.com/).  
- **Description**: The dataset consists of **customer reviews** (text) and corresponding **sentiment labels** (positive, negative, neutral).  
- **Inspiration**: Kaggle's community forums and notebooks provided valuable guidance during the project.  

---

## 🚀**Project Workflow**
### **1. Data Preprocessing**
Raw text requires cleaning before it can be used for machine learning. Here’s how I preprocessed the data:
- Converted text to **lowercase** for consistency.  
- Removed **punctuation**, **links**, **numbers**, and **stopwords** to reduce noise.  
- Tokenized the text (splitting sentences into individual words).  
- Optional: Explored **stemming** and **lemmatization** for further refinement.  

### **2. Feature Extraction (TF-IDF Vectorization)**
- Used **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert cleaned text into numerical vectors.  
- This step transformed textual data into a format that machine learning models could understand, capturing the "importance" of each word in the context of the entire dataset.  

### **3. Model Building**
- Split the dataset into **training (80%)** and **testing (20%)** sets.  
- Trained a **Logistic Regression** model using `scikit-learn`.  
- Despite its simplicity, Logistic Regression proved effective for text classification when paired with TF-IDF features.  

### **4. Model Evaluation**
Evaluated the model using the following metrics:
- **Accuracy Score**: Overall correctness of predictions.  
- **Confusion Matrix**: Visualized true vs. predicted classifications using `seaborn` heatmaps.  
- **Precision, Recall, F1-Score**: Provided deeper insights into model performance, especially for imbalanced datasets.  

---

## **Outputs**
Below are the outputs generated during the project:

### **Confusion Matrix**
![Confusion Matrix](https://github.com/user-attachments/assets/374dc3da-20e5-44d4-93d4-02aa96a2e371)

### **Performance Metrics**
![Performance Metrics](https://github.com/user-attachments/assets/1059026f-9cec-4ca6-8aa4-df89ab8db1cc)

---

## **Key Learnings**
1. **Data Cleaning is Critical**: Preprocessing raw text is just as important as building the model itself.  
2. **TF-IDF is Powerful**: It’s an excellent starting point for feature extraction in NLP tasks.  
3. **Logistic Regression Works Well**: Even simple models can perform effectively when paired with the right features.  
4. **Beyond Accuracy**: Evaluation metrics like precision, recall, and F1-score provide a more nuanced understanding of model performance.  

---

## **Future Scope and Improvements**
While this project was a great introduction to sentiment analysis, there’s plenty of room for growth:
1. **Hyperparameter Tuning**: Use `GridSearchCV` to optimize the TF-IDF vectorizer and Logistic Regression settings.  
2. **Multiclass Classification**: Extend the model to classify neutral and mixed sentiments.  
3. **Advanced Models**: Experiment with **Naive Bayes**, **Random Forests**, or deep learning models like **LSTM** and **Transformers** for improved accuracy.  
4. **Real-Time Sentiment Analysis**: Build a web app using **Flask** or **Streamlit** to predict sentiments on user input.  
5. **Deployment**: Host the model on platforms like **Heroku** or **Render** for real-world use.  

---

## **Acknowledgments**
Special thanks to my mentor, **Neela Santosh**, for providing guidance and support throughout the project.  

---

## **Contact**
For any questions or feedback, feel free to reach out:  
- **GitHub Profile**: shreyannandanwar

---

### Why This Structure Works:
1. **Clear Sections**: Each section has a specific purpose, making it easy to navigate.
2. **Concise Language**: Avoids unnecessary verbosity while maintaining clarity.
3. **Visual Outputs**: Links to images make the project tangible and engaging.
4. **Future Directions**: Highlights growth opportunities, showcasing ambition and curiosity.
5. **Professional Tone**: Maintains a formal yet approachable style suitable for GitHub audiences.  
