import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st #web interface ke liye 

# Load the dataset
data = pd.read_csv(r"C:\Users\naman\Downloads\vegefoods-master\vegefoods-master\css\iosc project\spam.csv")

# Data Cleaning
data.drop_duplicates(inplace=True)  # Removing duplicates
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

messages = data['Message']
categories = data['Category']

mess_train, mess_test, cat_train, cat_test = train_test_split(messages, categories, test_size=0.2, random_state=42)

# Convert text to numerical data
cv = CountVectorizer(stop_words='english')
features_train = cv.fit_transform(mess_train)

# Train the model
model = MultinomialNB()
model.fit(features_train, cat_train)

#ek function create karte hai
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]  
if __name__ == "__main__":
    test_message = 'Congratulations, you won a lottery'
    output = predict(test_message)
    print(f'Test Message: {test_message}\nPrediction: {output}')

st.title('Email Spam Detection')
user_input = st.text_area("Enter your email message:")
if st.button("Predict"):
    if user_input:
        prediction = predict(user_input)
        st.write(f'The message is: **{prediction}**')
    else:
        st.write("Please enter a message to predict.")
