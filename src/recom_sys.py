import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from openai import OpenAI


# Generate synthetic transaction data
def generate_synthetic_transaction_data(
    num_customers=500, transactions_per_customer=50
):
    np.random.seed(42)

    # Generate customer IDs
    customer_ids = np.arange(1, num_customers + 1)

    # Generate synthetic data
    data = []
    for customer_id in customer_ids:
        # Customer-specific attributes
        tenure = np.random.randint(1, 10)  # Customer tenure in years
        transaction_freq = np.random.poisson(20)  # Average transactions per month

        # Generate transactions
        for _ in range(transactions_per_customer):
            timestamp = datetime.now() - timedelta(
                days=np.random.randint(0, 365 * tenure)
            )
            transaction_type = np.random.choice(
                ["deposit", "withdrawal", "transfer", "payment"]
            )
            transaction_amount = np.abs(np.random.normal(100, 50))  # Random amount
            product_used = np.random.choice(
                ["checking_account", "savings_account", "credit_card", "loan"]
            )

            data.append(
                [
                    customer_id,
                    timestamp,
                    transaction_type,
                    round(transaction_amount, 2),
                    product_used,
                    tenure,
                    transaction_freq,
                ]
            )

    # Create DataFrame
    columns = [
        "customer_ID",
        "timestamp",
        "transaction_type",
        "transaction_amount",
        "product_used",
        "customer_tenure",
        "transaction_frequency",
    ]
    df = pd.DataFrame(data, columns=columns)
    return df


# Preprocess data for sequential model
def preprocess_data(transaction_data):
    # Encode product_used to numerical values
    label_encoder = LabelEncoder()
    transaction_data["product_encoded"] = label_encoder.fit_transform(
        transaction_data["product_used"]
    )

    # Group by customer and create sequences
    sequences = transaction_data.groupby("customer_ID")["product_encoded"].apply(
        lambda x: list(x)
    )

    # Pad sequences to the same length
    max_len = sequences.apply(len).max()
    padded_sequences = np.array([seq + [0] * (max_len - len(seq)) for seq in sequences])

    return padded_sequences, label_encoder


# Build and train sequential model
def build_sequential_model(input_shape, num_classes):
    model = Sequential(
        [
            Embedding(input_dim=num_classes, output_dim=64, input_length=input_shape),
            LSTM(128, return_sequences=False),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Train the model
def train_model(sequences, label_encoder):
    X = sequences[:, :-1]  # Input sequences (all but last product)
    y = sequences[:, -1]  # Target (last product)

    num_classes = len(label_encoder.classes_)
    model = build_sequential_model(X.shape[1], num_classes)
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    return model


# Generate recommendations using the sequential model
def recommend_products(model, customer_sequence, label_encoder):
    predicted_probs = model.predict(customer_sequence[np.newaxis, :])
    predicted_class = np.argmax(predicted_probs)
    recommended_product = label_encoder.inverse_transform([predicted_class])[0]
    return recommended_product


# Personalized message generation using LLM
def generate_personalized_message(
    customer_data, recommended_product, openai_api_key=None
):
    if not openai_api_key:
        return f"Sample personalized message: We recommend our {recommended_product} based on your transaction history."

    client = OpenAI(api_key=openai_api_key)

    # Prepare prompt
    prompt = f"""
    Generate a personalized banking recommendation message for a customer with these characteristics:
    - Most Used Product: {customer_data['product_used'].mode()[0]}
    - Transaction Frequency: {customer_data['transaction_frequency'].mean():.1f} transactions/month
    - Customer Tenure: {customer_data['customer_tenure'].mean():.1f} years
    
    Recommended product: {recommended_product}
    
    Create a friendly, professional message that explains why this product is recommended.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful banking assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate message: {str(e)}"


# Streamlit UI
def main():
    st.title("Personalized Banking Recommendations")

    # Generate synthetic transaction data
    transaction_data = generate_synthetic_transaction_data()

    # Preprocess data
    sequences, label_encoder = preprocess_data(transaction_data)

    # Train sequential model
    model = train_model(sequences, label_encoder)

    # Customer selection
    st.sidebar.header("Customer Selection")
    customer_id = st.sidebar.selectbox(
        "Select Customer ID", transaction_data["customer_ID"].unique()
    )

    selected_customer = transaction_data[transaction_data["customer_ID"] == customer_id]
    customer_sequence = sequences[customer_id - 1]  # Sequences are 0-indexed

    # Display customer profile
    st.header("Customer Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Customer Tenure",
            f"{selected_customer['customer_tenure'].mean():.1f} years",
        )
    with col2:
        st.metric(
            "Transaction Frequency",
            f"{selected_customer['transaction_frequency'].mean():.1f}/month",
        )
    with col3:
        st.metric("Most Used Product", selected_customer["product_used"].mode()[0])

    st.subheader("Transaction History (Last 10 Transactions)")
    st.write(
        selected_customer[
            ["timestamp", "transaction_type", "transaction_amount", "product_used"]
        ].head(10)
    )

    # Generate recommendations
    recommended_product = recommend_products(model, customer_sequence, label_encoder)

    st.header("Recommended Product")
    st.success(f"✓ {recommended_product}")

    # Generate personalized message
    st.header("Personalized Message")
    openai_api_key = st.text_input("Enter OpenAI API Key (optional)", type="password")

    message = generate_personalized_message(
        selected_customer, recommended_product, openai_api_key
    )
    st.write(message)

    # Show raw data
    if st.checkbox("Show raw transaction data"):
        st.write(selected_customer)


if __name__ == "__main__":
    main()
