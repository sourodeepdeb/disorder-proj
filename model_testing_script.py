def predict_category(text):
    embedding = get_embedding(text, model='text-embedding-3-small')
    embedding_scaled = scaler.transform([embedding])
    prediction = model.predict(embedding_scaled)
    return category_mapping[prediction[0]]

print("\nEnter a sentence to predict its category.")
print("Type 'exit' to end the program.")

while True:
    user_input = input("\nEnter your text: ").strip()

    if user_input.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break

    if user_input:
        predicted_category = predict_category(user_input)
        print(f"Predicted category: {predicted_category}")
    else:
        print("Please enter a valid text.")
